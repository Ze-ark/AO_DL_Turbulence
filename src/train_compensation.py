"""训练 AO-DL 相位补偿模型并保存最佳检查点。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
import yaml

# 将项目根目录加入模块搜索路径，确保直接运行本脚本时也能导入 src 包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_holo import HoloH5Dataset, split_indices
from src.losses import compensation_loss
from src.models.resunet_phase import ResUNetPhase
from src.runtime import resolve_device
from src.training_progress import progress_bar, progress_message, update_progress


def train(config_path: str | Path) -> Path:
    """根据配置训练相位补偿网络，并返回最佳模型检查点路径。"""
    # 读取配置并确定训练设备与模拟数据集路径。
    config_path = _resolve_project_path(config_path)
    config = _load_config(config_path)
    device = resolve_device(config["runtime"].get("device", "cpu"))
    h5_path = _resolve_project_path(config["data"].get("simulated_h5_path", config["data"].get("h5_path")))
    dataset = HoloH5Dataset(h5_path)
    # 使用固定随机种子划分训练、验证和测试索引，保证结果可复现。
    splits = split_indices(
        len(dataset),
        train_fraction=config["data"].get("train_fraction", 0.8),
        val_fraction=config["data"].get("val_fraction", 0.0),
        seed=config["data"].get("seed", 42),
    )
    # 训练集启用随机打乱；测试集保持固定顺序以便稳定评估。
    train_loader = DataLoader(
        HoloH5Dataset(h5_path, indices=splits["train"]),
        batch_size=config["train"].get("batch_size", 4),
        shuffle=True,
    )
    test_loader = DataLoader(
        HoloH5Dataset(h5_path, indices=splits["test"]),
        batch_size=config["train"].get("batch_size", 4),
        shuffle=False,
    )

    # 初始化相位预测网络及 AdamW 优化器。
    model = ResUNetPhase(base_channels=config["model"].get("base_channels", 32)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"].get("learning_rate", 1e-3),
        weight_decay=config["train"].get("weight_decay", 1e-4),
    )

    # 创建日志与检查点目录，并记录当前最优测试损失。
    output_dir = _resolve_project_path(config["train"].get("output_dir", "outputs/train_sim_gaussian_v1"))
    checkpoint_dir = _resolve_project_path(config["train"].get("checkpoint_dir", "checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "loss_history.csv"
    best_path = checkpoint_dir / "sim_gaussian_v1_best.pt"
    best_val = float("inf")
    total_epochs = int(config["train"].get("epochs", 2))
    progress_message(
        f"训练设备：{torch.cuda.get_device_name(device)}；"
        f"轮数：{total_epochs}；批大小：{config['train'].get('batch_size', 4)}"
    )

    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_total", "test_total"])
        epochs = progress_bar(
            range(1, total_epochs + 1),
            description="模型训练总进度",
            unit="轮",
        )
        for epoch in epochs:
            # 训练阶段更新参数，测试阶段关闭梯度并仅计算损失。
            train_loss = _run_epoch(
                model,
                train_loader,
                device,
                optimizer,
                epoch=epoch,
                total_epochs=total_epochs,
                stage="训练",
            )
            test_loss = _run_epoch(
                model,
                test_loader,
                device,
                optimizer=None,
                epoch=epoch,
                total_epochs=total_epochs,
                stage="测试",
            )
            writer.writerow([epoch, train_loss, test_loss])
            f.flush()
            # 仅保存测试损失最低的模型，避免后续较差轮次覆盖最佳权重。
            if test_loss < best_val:
                best_val = test_loss
                torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, best_path)
                progress_message(
                    f"第 {epoch}/{total_epochs} 轮保存新的最佳权重：测试损失={best_val:.6f}"
                )
            update_progress(
                epochs,
                device=device,
                metrics={"训练损失": train_loss, "测试损失": test_loss, "最佳损失": best_val},
            )
            progress_message(
                f"epoch={epoch} train_total={train_loss:.6f} test_total={test_loss:.6f}"
            )

    return best_path


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    *,
    epoch: int,
    total_epochs: int,
    stage: str,
) -> float:
    """运行一个训练或评估轮次，并返回按样本数加权的平均损失。"""
    # 是否传入优化器决定当前处于训练模式还是评估模式。
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    batches = progress_bar(
        loader,
        description=f"{stage} {epoch}/{total_epochs}",
        unit="批",
        leave=False,
    )
    for batch in batches:
        # 将批次中的张量移动到目标设备，非张量元数据保持不变。
        batch = _move_batch(batch, device)
        with torch.set_grad_enabled(training):
            # 网络预测校正相位，再结合输入场与目标场计算补偿损失。
            phi_corr = model(batch["input"])
            losses = compensation_loss(
                phi_corr=phi_corr,
                input_intensity=batch["input_intensity"],
                input_phase=batch["input_phase"],
                target_intensity=batch["target_intensity"],
                target_phase=batch["target_phase"],
            )
            if training:
                # 清空旧梯度，反向传播总损失并更新模型参数。
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                optimizer.step()
        # 按批次样本数累计损失，以兼容最后一个不足整批的批次。
        batch_size = int(batch["input"].shape[0])
        total += float(losses["total"].detach().cpu().item()) * batch_size
        count += batch_size
        update_progress(
            batches,
            device=device,
            metrics={"平均损失": total / count},
        )
    return total / max(count, 1)


def _move_batch(batch: dict, device: torch.device) -> dict:
    """将批次字典中的所有张量移动到指定计算设备。"""
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _load_config(path: str | Path) -> dict:
    """从 YAML 文件加载训练配置。"""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_project_path(path: str | Path) -> Path:
    """将相对路径解析为相对于项目根目录的绝对路径。"""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    """解析命令行参数并启动训练。"""
    parser = argparse.ArgumentParser(description="Train AO-DL phase compensation.")
    parser.add_argument("--config", default="configs/sim_gaussian_v1.yaml")
    args = parser.parse_args()
    print(f"saved_best_checkpoint={train(args.config)}")


if __name__ == "__main__":
    main()
