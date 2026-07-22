"""评估模拟数据上的相位补偿效果并导出分组指标与对比图。"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys
from typing import Iterable

import matplotlib

# 使用无界面后端，保证脚本可在服务器和自动化测试环境中运行。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import yaml

# 支持直接执行该文件时从项目根目录导入 src 包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_holo import HoloH5Dataset, split_indices
from src.losses import compensate_field, complex_field_from_intensity_phase
from src.models.resunet_phase import ResUNetPhase
from src.runtime import resolve_device


def compute_batch_metrics(
    comp_real: torch.Tensor,
    comp_imag: torch.Tensor,
    target_real: torch.Tensor,
    target_imag: torch.Tensor,
) -> dict[str, float]:
    """按批次计算复相关、光强误差、相位误差、Strehl 比和质心偏移。"""
    # 展平空间维度后计算归一化复相关系数。
    comp = torch.complex(comp_real, comp_imag).flatten(1)
    target = torch.complex(target_real, target_imag).flatten(1)
    numerator = torch.abs(torch.sum(comp * torch.conj(target), dim=1))
    denominator = torch.linalg.norm(comp, dim=1) * torch.linalg.norm(target, dim=1)
    corr = torch.where(denominator > 0, numerator / denominator, torch.ones_like(denominator))

    # 由复光场重建光强，并计算逐样本均方误差。
    comp_intensity = comp_real.square() + comp_imag.square()
    target_intensity = target_real.square() + target_imag.square()
    intensity_mse = torch.mean((comp_intensity - target_intensity).square(), dim=(1, 2, 3))

    # 使用复指数将相位差包裹到 [-π, π]，避免相位跳变放大误差。
    comp_phase = torch.atan2(comp_imag, comp_real)
    target_phase = torch.atan2(target_imag, target_real)
    phase_error = torch.angle(torch.exp(1j * (comp_phase - target_phase)))
    phase_rmse = torch.sqrt(torch.mean(phase_error.square(), dim=(1, 2, 3)))

    # 峰值比近似 Strehl 比，质心偏移平方反映光斑稳定性。
    target_peak = torch.amax(target_intensity.flatten(1), dim=1)
    comp_peak = torch.amax(comp_intensity.flatten(1), dim=1)
    strehl = torch.where(target_peak > 0, comp_peak / target_peak, torch.ones_like(target_peak))
    rc2 = _centroid_shift_squared(comp_intensity, target_intensity)

    return {
        "complex_corr": _round_metric(corr.mean()),
        "intensity_mse": _round_metric(intensity_mse.mean()),
        "phase_rmse": _round_metric(phase_rmse.mean()),
        "strehl_ratio": _round_metric(strehl.mean()),
        "rc2": _round_metric(rc2.mean()),
    }


def evaluate(config_path: str | Path) -> dict[str, dict[str, float]]:
    """加载最佳检查点，在指定数据划分上评估并保存结果。"""
    config_path = _resolve_project_path(config_path)
    config = _load_config(config_path)
    device = resolve_device(config["runtime"].get("device", "cpu"))
    h5_path = _resolve_project_path(config["data"].get("simulated_h5_path", config["data"].get("h5_path")))
    dataset = HoloH5Dataset(h5_path)
    splits = split_indices(
        len(dataset),
        train_fraction=config["data"].get("train_fraction", 0.8),
        val_fraction=config["data"].get("val_fraction", 0.0),
        seed=config["data"].get("seed", 42),
    )
    # 按配置选择验证集或测试集，评估时不打乱样本顺序。
    eval_indices = splits[config["eval"].get("split", "test")]
    loader = DataLoader(
        HoloH5Dataset(h5_path, indices=eval_indices),
        batch_size=config["eval"].get("batch_size", 4),
        shuffle=False,
    )

    model = ResUNetPhase(
        in_channels=2,
        base_channels=config["model"].get("base_channels", 32),
    ).to(device)
    checkpoint_path = _resolve_project_path(config["eval"]["checkpoint_path"])
    # 将检查点映射到目标设备并切换为推理模式。
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rows = []
    output_dir = _resolve_project_path(config["eval"].get("output_dir", "outputs/eval_sim_gaussian_v1"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 禁用梯度以降低评估阶段的显存占用。
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            batch = _move_batch(batch, device)
            phi_corr = model(batch["input"])
            comp_real, comp_imag = compensate_field(batch["input_intensity"], batch["input_phase"], phi_corr)
            target_real, target_imag = complex_field_from_intensity_phase(
                batch["target_intensity"], batch["target_phase"]
            )
            metrics = compute_batch_metrics(comp_real, comp_imag, target_real, target_imag)
            metrics["turbulence_strength"] = float(batch["meta"]["turbulence_strength"].float().mean().item())
            rows.append(metrics)
            # 仅保存首批样本的可视化，避免生成大量重复图片。
            if batch_index == 0:
                _save_comparison_figure(output_dir / "comparison_first_batch.png", batch, comp_real, comp_imag)

    grouped = _group_metrics(rows)
    _write_metrics_csv(output_dir / "metrics_by_strength.csv", grouped)
    return grouped


def main() -> None:
    """解析命令行配置路径并打印各湍流强度组的评估结果。"""
    parser = argparse.ArgumentParser(description="Evaluate AO-DL phase compensation.")
    parser.add_argument("--config", default="configs/sim_gaussian_v1.yaml")
    args = parser.parse_args()
    grouped = evaluate(args.config)
    for group, metrics in grouped.items():
        print(group, metrics)


def _centroid_shift_squared(comp_intensity: torch.Tensor, target_intensity: torch.Tensor) -> torch.Tensor:
    """计算补偿光斑与目标光斑的二维质心距离平方。"""
    _, _, h, w = comp_intensity.shape
    y = torch.arange(h, device=comp_intensity.device, dtype=comp_intensity.dtype).view(1, 1, h, 1)
    x = torch.arange(w, device=comp_intensity.device, dtype=comp_intensity.dtype).view(1, 1, 1, w)
    comp_sum = torch.clamp(comp_intensity.sum(dim=(1, 2, 3), keepdim=True), min=1e-12)
    target_sum = torch.clamp(target_intensity.sum(dim=(1, 2, 3), keepdim=True), min=1e-12)
    comp_x = (comp_intensity * x).sum(dim=(1, 2, 3), keepdim=True) / comp_sum
    comp_y = (comp_intensity * y).sum(dim=(1, 2, 3), keepdim=True) / comp_sum
    target_x = (target_intensity * x).sum(dim=(1, 2, 3), keepdim=True) / target_sum
    target_y = (target_intensity * y).sum(dim=(1, 2, 3), keepdim=True) / target_sum
    return ((comp_x - target_x).square() + (comp_y - target_y).square()).flatten()


def _round_metric(value: torch.Tensor) -> float:
    """将标量张量转换为稳定的七位小数结果。"""
    number = float(value.detach().cpu().item())
    return 0.0 if abs(number) < 5e-8 else round(number, 7)


def _group_metrics(rows: Iterable[dict[str, float]]) -> dict[str, dict[str, float]]:
    """按弱、中、强湍流区间汇总各项指标均值。"""
    buckets: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        strength = row["turbulence_strength"]
        if strength < 0.4:
            key = "weak"
        elif strength < 0.75:
            key = "medium"
        else:
            key = "strong"
        buckets[key].append(row)

    grouped = {}
    for key, values in buckets.items():
        grouped[key] = {
            metric: float(sum(row[metric] for row in values) / len(values))
            for metric in ("complex_corr", "intensity_mse", "phase_rmse", "strehl_ratio", "rc2")
        }
    return grouped


def _write_metrics_csv(path: Path, grouped: dict[str, dict[str, float]]) -> None:
    """将分组评估指标写入 CSV 文件。"""
    fields = ["complex_corr", "intensity_mse", "phase_rmse", "strehl_ratio", "rc2"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group", *fields])
        for group, metrics in grouped.items():
            writer.writerow([group, *[metrics[name] for name in fields]])


def _save_comparison_figure(path: Path, batch: dict, comp_real: torch.Tensor, comp_imag: torch.Tensor) -> None:
    """保存首个样本的输入、目标、补偿光强和补偿相位对比图。"""
    input_intensity = batch["input_intensity"][0, 0].detach().cpu()
    target_intensity = batch["target_intensity"][0, 0].detach().cpu()
    comp_intensity = (comp_real.square() + comp_imag.square())[0, 0].detach().cpu()
    comp_phase = torch.atan2(comp_imag, comp_real)[0, 0].detach().cpu()

    fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
    for ax, image, title in (
        (axes[0, 0], input_intensity, "Turbulent intensity"),
        (axes[0, 1], target_intensity, "Clean target intensity"),
        (axes[1, 0], comp_intensity, "Compensated intensity"),
        (axes[1, 1], comp_phase, "Compensated phase"),
    ):
        ax.imshow(image.numpy(), cmap="viridis")
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _move_batch(batch: dict, device: torch.device) -> dict:
    """递归一层地将批次及元数据字典中的张量移动到目标设备。"""
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = {
                sub_key: sub_value.to(device) if isinstance(sub_value, torch.Tensor) else sub_value
                for sub_key, sub_value in value.items()
            }
        else:
            moved[key] = value
    return moved


def _load_config(path: str | Path) -> dict:
    """读取 YAML 评估配置。"""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_project_path(path: str | Path) -> Path:
    """将相对路径转换为基于项目根目录的绝对路径。"""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
