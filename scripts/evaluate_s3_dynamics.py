"""在模型冻结后打开S3封存条件，比较GRU与线性动力学基线。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime import resolve_device
from src.simulation import (
    DynamicsCondition,
    GRUModalDynamics,
    TemporalDynamicsData,
    condition_gate,
    constant_velocity_forecast,
    generate_temporal_dynamics_data,
    load_s1_config,
    paired_episode_comparison,
    ridge_autoregressive_forecast,
)
from src.simulation.temporal_dynamics import denormalize_modal, normalize_modal
from src.training_progress import (
    advance_to,
    counted_progress,
    progress_bar,
    progress_message,
    update_progress,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/s3_dynamics_v1.yaml")
    parser.add_argument(
        "--acknowledge-sealed-test",
        action="store_true",
        help="确认模型已冻结并允许首次打开预声明封存条件。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.acknowledge_sealed_test:
        raise SystemExit(
            "封存测试未打开：先让助手读取训练结果并确认验证门槛，"
            "再显式添加 --acknowledge-sealed-test。"
        )

    experiment_path = _project_path(args.config)
    experiment = _load_yaml(experiment_path)
    output_settings = experiment["outputs"]
    training_dir = _project_path(output_settings["training_directory"])
    training_summary_path = training_dir / "summary.json"
    if not training_summary_path.exists():
        raise FileNotFoundError("S3 training summary is missing")
    training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    if training_summary["gate"]["validation_gate"] != "PASS":
        raise SystemExit("封存测试未打开：S3验证门槛没有通过。")
    if training_summary["inputs"]["experiment_config_sha256"] != _sha256(experiment_path):
        raise SystemExit("封存测试未打开：训练后的实验配置已发生变化。")

    environment_path = _project_path(experiment["environment_config"])
    environment_config, requested_device = load_s1_config(environment_path)
    if training_summary["inputs"]["environment_config_sha256"] != _sha256(environment_path):
        raise SystemExit("封存测试未打开：训练后的环境配置已发生变化。")
    checkpoint_path = _project_path(output_settings["checkpoint"])
    if training_summary["outputs"]["checkpoint_sha256"] != _sha256(checkpoint_path):
        raise SystemExit("封存测试未打开：最佳模型权重与训练汇总不匹配。")

    device = resolve_device(requested_device)
    dataset_settings = experiment["dataset"]
    conditions = [
        DynamicsCondition.from_mapping(item)
        for item in dataset_settings["sealed_test_conditions"]
    ]
    total = len(conditions) * int(dataset_settings["frames_per_episode"])
    progress_message("配置、环境和检查点哈希匹配；现在首次打开封存条件。")
    with counted_progress(total=total, description="生成S3封存回合", unit="步") as bar:
        test_data = generate_temporal_dynamics_data(
            environment_config,
            device,
            conditions,
            int(dataset_settings["sequence_length"]),
            int(dataset_settings["frames_per_episode"]),
            float(dataset_settings["random_action_std_rad"]),
            progress_callback=lambda completed, _: advance_to(bar, completed),
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_settings = checkpoint["model_settings"]
    model = GRUModalDynamics(
        num_modes=int(checkpoint["num_modes"]),
        hidden_size=int(model_settings["hidden_size"]),
        num_layers=int(model_settings["num_layers"]),
        dropout=float(model_settings["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    mean = checkpoint["normalization_mean"].to(device)
    scale = checkpoint["normalization_scale"].to(device)
    prediction = _predict(
        model,
        test_data,
        device,
        mean,
        scale,
        int(experiment["training"]["batch_size"]),
    )
    linear_prediction = ridge_autoregressive_forecast(
        test_data.histories,
        checkpoint["normalization_mean"],
        checkpoint["normalization_scale"],
        checkpoint["ridge_weight"],
        checkpoint["ridge_bias"],
    )
    raw_comparison = paired_episode_comparison(
        prediction,
        linear_prediction,
        test_data.targets,
        checkpoint["normalization_scale"],
        test_data.episode_seed,
        test_data.condition_index,
    )
    gate_settings = experiment["gate"]
    gate = condition_gate(
        raw_comparison,
        min_mean_skill_score=float(gate_settings["min_mean_skill_score"]),
        min_ci95_low=float(gate_settings["min_ci95_low"]),
        require_every_condition_positive=bool(
            gate_settings["require_every_condition_positive"]
        ),
    )
    comparison = _name_conditions(raw_comparison, conditions)
    gate["condition_mean_skill_scores"] = {
        conditions[int(index)].identifier: value
        for index, value in gate["condition_mean_skill_scores"].items()
    }
    gate["full_s3_model_gate"] = gate.pop("validation_gate")
    constant_velocity_prediction = constant_velocity_forecast(
        test_data.histories,
        float(experiment["baseline"]["secondary_velocity_gain"]),
    )
    secondary_comparison = paired_episode_comparison(
        prediction,
        constant_velocity_prediction,
        test_data.targets,
        checkpoint["normalization_scale"],
        test_data.episode_seed,
        test_data.condition_index,
    )
    secondary_comparison = _name_conditions(secondary_comparison, conditions)
    output_dir = _project_path(output_settings["sealed_test_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    test_h5 = output_dir / "sealed_test_data.h5"
    _export_test_data(test_h5, test_data, conditions)
    result = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "validate",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "ANALYZED",
            "version_label": "s3_dynamics_sealed_test_v1",
        },
        "source": "AO-S3-GRU-DYNAMICS-GATE",
        "status": "completed",
        "inputs": {
            "experiment_config": str(experiment_path.relative_to(PROJECT_ROOT)),
            "experiment_config_sha256": _sha256(experiment_path),
            "environment_config_sha256": _sha256(environment_path),
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "training_summary_sha256": _sha256(training_summary_path),
            "sealed_conditions": [asdict(item) for item in conditions],
            "sealed_test_accessed": True,
        },
        "sealed_test_comparison": comparison,
        "secondary_constant_velocity_comparison": secondary_comparison,
        "gate": gate,
        "outputs": {
            "sealed_test_h5": str(test_h5.relative_to(PROJECT_ROOT)),
            "sealed_test_h5_sha256": _sha256(test_h5),
        },
        "interpretation_boundary": (
            "Passing only licenses the next pure-simulation controller comparison. "
            "It does not show that RL, holographic sensing, or real SLM control works."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "validation_report.md").write_text(
        _markdown_report(result), encoding="utf-8"
    )
    progress_message(f"S3封存模型门槛：{gate['full_s3_model_gate']}")
    progress_message(f"结果：{summary_path}")


def _predict(
    model: GRUModalDynamics,
    data: TemporalDynamicsData,
    device: torch.device,
    mean: torch.Tensor,
    scale: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    loader = DataLoader(TensorDataset(data.histories), batch_size=batch_size, shuffle=False)
    predictions: list[torch.Tensor] = []
    model.eval()
    batches = progress_bar(loader, description="评估S3封存回合", unit="批", leave=True)
    with torch.no_grad():
        for (histories,) in batches:
            normalized = normalize_modal(histories.to(device), mean, scale)
            predictions.append(denormalize_modal(model(normalized), mean, scale).cpu())
            update_progress(batches, device=device, metrics={})
    return torch.cat(predictions)


def _export_test_data(
    path: Path,
    data: TemporalDynamicsData,
    conditions: list[DynamicsCondition],
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["stage"] = "S3"
        handle.attrs["split"] = "sealed_test"
        handle.attrs["conditions_json"] = json.dumps(
            [asdict(item) for item in conditions], ensure_ascii=False
        )
        handle.create_dataset("history", data=data.histories.numpy(), compression="gzip")
        handle.create_dataset("target", data=data.targets.numpy(), compression="gzip")
        handle.create_dataset("episode_seed", data=data.episode_seed.numpy())
        handle.create_dataset("condition_index", data=data.condition_index.numpy())
        handle.create_dataset("target_step", data=data.target_step.numpy())


def _name_conditions(
    comparison: dict[str, Any], conditions: list[DynamicsCondition]
) -> dict[str, Any]:
    renamed = dict(comparison)
    renamed["by_condition"] = {
        conditions[int(index)].identifier: values
        for index, values in comparison["by_condition_index"].items()
    }
    del renamed["by_condition_index"]
    return renamed


def _markdown_report(result: dict[str, Any]) -> str:
    passport = result["material_passport"]
    comparison = result["sealed_test_comparison"]
    gate = result["gate"]
    return f"""## Material Passport

- Origin Skill: {passport['origin_skill']}
- Origin Mode: {passport['origin_mode']}
- Origin Date: {passport['origin_date']}
- Verification Status: {passport['verification_status']}
- Version Label: {passport['version_label']}

## S3 Sealed Dynamics Validation

| Metric | GRU | Same-history linear ridge baseline |
|---|---:|---:|
| Normalized RMSE | {comparison['gru_normalized_rmse']['mean']:.6f} | {comparison['linear_normalized_rmse']['mean']:.6f} |

- **Paired episode skill score**: {comparison['skill_score']['mean']:.6f}
- **95% CI**: [{comparison['skill_score']['ci95_low']:.6f}, {comparison['skill_score']['ci95_high']:.6f}]
- **Full S3 Model Gate**: {gate['full_s3_model_gate']}
- **Boundary**: {result['interpretation_boundary']}
"""


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
