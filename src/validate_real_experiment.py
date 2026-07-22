"""使用已训练模型验证真实实验复光场数据。"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib

# 使用无界面绘图后端，适配服务器执行环境。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import yaml

# 支持将本文件作为独立脚本直接执行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_holo import RealComplexH5Dataset
from src.models.resunet_phase import ResUNetPhase
from src.runtime import resolve_device


def validate_real_experiment(config_path: str | Path) -> Path:
    """在真实无标签数据上推理，并返回分温度统计表路径。"""
    config_path = _resolve_project_path(config_path)
    config = _load_config(config_path)
    validation_config = config["validation"]
    device = resolve_device(config["runtime"].get("device", "cuda"))
    dataset = RealComplexH5Dataset(_resolve_project_path(validation_config["real_h5_path"]))
    loader = DataLoader(
        dataset,
        batch_size=validation_config.get("batch_size", 8),
        shuffle=False,
        num_workers=0,
    )

    # 模型结构必须与训练检查点中的结构参数保持一致。
    model = ResUNetPhase(
        in_channels=2,
        base_channels=config["model"].get("base_channels", 32),
    ).to(device)
    checkpoint_path = _resolve_project_path(config["eval"]["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    output_dir = _resolve_project_path(validation_config.get("output_dir", "outputs/real_validation_2x"))
    output_dir.mkdir(parents=True, exist_ok=True)
    # 按温度分组保存逐帧指标，最后统一计算统计量。
    summary_rows: dict[int, list[dict[str, float]]] = defaultdict(list)
    first_batch_saved = False

    # 真实验证仅执行前向推理，不构建梯度图。
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            intensities = batch["input_intensity"].to(device)
            phi_corr = model(inputs)
            temperatures = batch["meta"]["temperature"]
            batch_rows = _batch_real_metrics(intensities, phi_corr)
            for i, row in enumerate(batch_rows):
                temperature = int(temperatures[i].item() if isinstance(temperatures, torch.Tensor) else temperatures[i])
                summary_rows[temperature].append(row)
            # 示例图只保存一次，控制输出文件数量。
            if not first_batch_saved:
                _save_real_examples(output_dir / "real_validation_examples.png", batch, phi_corr)
                first_batch_saved = True

    summary_path = output_dir / "real_validation_summary.csv"
    _write_summary(summary_path, summary_rows)
    return summary_path


def _batch_real_metrics(intensity: torch.Tensor, phi_corr: torch.Tensor) -> list[dict[str, float]]:
    """计算每帧能量、峰值、质心位置及预测相位统计量。"""
    _, _, h, w = intensity.shape
    y = torch.arange(h, device=intensity.device, dtype=intensity.dtype).view(1, 1, h, 1)
    x = torch.arange(w, device=intensity.device, dtype=intensity.dtype).view(1, 1, 1, w)
    energy = torch.clamp(intensity.sum(dim=(1, 2, 3), keepdim=True), min=1e-12)
    centroid_x = ((intensity * x).sum(dim=(1, 2, 3), keepdim=True) / energy).flatten()
    centroid_y = ((intensity * y).sum(dim=(1, 2, 3), keepdim=True) / energy).flatten()
    peak = intensity.flatten(1).amax(dim=1)
    phase_abs = phi_corr.abs().flatten(1).mean(dim=1)
    phase_std = phi_corr.flatten(1).std(dim=1)

    rows = []
    for i in range(intensity.shape[0]):
        rows.append(
            {
                "energy": float(energy.flatten()[i].detach().cpu().item()),
                "peak": float(peak[i].detach().cpu().item()),
                "centroid_x": float(centroid_x[i].detach().cpu().item()),
                "centroid_y": float(centroid_y[i].detach().cpu().item()),
                "phi_corr_abs_mean": float(phase_abs[i].detach().cpu().item()),
                "phi_corr_std": float(phase_std[i].detach().cpu().item()),
            }
        )
    return rows


def _write_summary(path: Path, rows_by_temperature: dict[int, list[dict[str, float]]]) -> None:
    """按温度汇总逐帧指标并写入 CSV。"""
    fields = [
        "temperature",
        "frame_count",
        "energy_mean",
        "peak_mean",
        "centroid_x_var",
        "centroid_y_var",
        "rc2",
        "phi_corr_abs_mean",
        "phi_corr_std_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for temperature in sorted(rows_by_temperature):
            rows = rows_by_temperature[temperature]
            centroid_x = [row["centroid_x"] for row in rows]
            centroid_y = [row["centroid_y"] for row in rows]
            # 横纵质心方差之和作为光斑漂移指标 rc2。
            centroid_x_var = _variance(centroid_x)
            centroid_y_var = _variance(centroid_y)
            writer.writerow(
                {
                    "temperature": temperature,
                    "frame_count": len(rows),
                    "energy_mean": _mean(row["energy"] for row in rows),
                    "peak_mean": _mean(row["peak"] for row in rows),
                    "centroid_x_var": centroid_x_var,
                    "centroid_y_var": centroid_y_var,
                    "rc2": centroid_x_var + centroid_y_var,
                    "phi_corr_abs_mean": _mean(row["phi_corr_abs_mean"] for row in rows),
                    "phi_corr_std_mean": _mean(row["phi_corr_std"] for row in rows),
                }
            )


def _save_real_examples(path: Path, batch: dict, phi_corr: torch.Tensor) -> None:
    """保存最多四帧真实光强与预测校正相位的对照图。"""
    count = min(4, int(phi_corr.shape[0]))
    fig, axes = plt.subplots(count, 2, figsize=(7, 3 * count), constrained_layout=True)
    if count == 1:
        axes = axes[None, :]
    temperatures = batch["meta"]["temperature"]
    for i in range(count):
        temp = int(temperatures[i].item() if isinstance(temperatures, torch.Tensor) else temperatures[i])
        intensity = batch["input_intensity"][i, 0].detach().cpu().numpy()
        phase = phi_corr[i, 0].detach().cpu().numpy()
        axes[i, 0].imshow(intensity, cmap="gray")
        axes[i, 0].set_title(f"Real input intensity, dT={temp}")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(phase, cmap="twilight")
        axes[i, 1].set_title("Predicted phi_corr")
        axes[i, 1].axis("off")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _mean(values) -> float:
    """计算可迭代对象的算术平均值，空输入返回零。"""
    values = list(values)
    return float(sum(values) / max(len(values), 1))


def _variance(values: list[float]) -> float:
    """计算样本方差；样本不足两个时返回零。"""
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return float(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _load_config(path: str | Path) -> dict:
    """读取 YAML 验证配置。"""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_project_path(path: str | Path) -> Path:
    """将相对路径转换为基于项目根目录的绝对路径。"""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    """解析命令行参数并启动真实实验验证。"""
    parser = argparse.ArgumentParser(description="Validate AO-DL checkpoint on real 2x complex-field HDF5 data.")
    parser.add_argument("--config", default="configs/sim_gaussian_v1.yaml")
    args = parser.parse_args()
    print(f"real_validation_summary={validate_real_experiment(args.config)}")


if __name__ == "__main__":
    main()
