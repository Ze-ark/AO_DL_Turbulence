"""训练S3多帧GRU动力学模型；只使用训练集和验证集，不打开封存测试。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
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
    fit_ridge_autoregression,
    generate_temporal_dynamics_data,
    load_s1_config,
    modal_normalization,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_path = _project_path(args.config)
    experiment = _load_yaml(experiment_path)
    environment_path = _project_path(experiment["environment_config"])
    environment_config, requested_device = load_s1_config(environment_path)
    device = resolve_device(requested_device)
    training = experiment["training"]
    dataset_settings = experiment["dataset"]
    model_settings = experiment["model"]
    baseline_settings = experiment["baseline"]
    gate_settings = experiment["gate"]
    output_settings = experiment["outputs"]
    train_conditions = _conditions(dataset_settings["train_conditions"])
    validation_conditions = _conditions(dataset_settings["validation_conditions"])
    _assert_disjoint_conditions(
        train_conditions,
        validation_conditions,
        _conditions(dataset_settings["sealed_test_conditions"]),
    )
    seed = int(training["seed"])
    _enable_determinism(seed)
    started = perf_counter()

    output_dir = _project_path(output_settings["training_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _project_path(output_settings["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path = _project_path(output_settings["train_validation_h5"])
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    progress_message(
        f"S3训练设备：{torch.cuda.get_device_name(device)}；"
        f"GRU轮数：{int(training['epochs'])}；批大小：{int(training['batch_size'])}"
    )
    progress_message(
        "安全边界：本命令只校验封存条件元数据是否重叠，不会生成或评估封存回合。"
    )

    train_data = _generate_with_progress(
        "生成S3训练回合",
        environment_config,
        device,
        train_conditions,
        dataset_settings,
    )
    validation_data = _generate_with_progress(
        "生成S3验证回合",
        environment_config,
        device,
        validation_conditions,
        dataset_settings,
    )
    _export_dataset(
        dataset_path,
        train_data,
        validation_data,
        train_conditions,
        validation_conditions,
        experiment_path,
        environment_path,
    )

    mean, scale = modal_normalization(train_data)
    progress_message("拟合与GRU使用相同8帧历史的线性岭回归基线……")
    ridge_weight, ridge_bias = fit_ridge_autoregression(
        train_data.histories,
        train_data.targets,
        mean,
        scale,
        float(baseline_settings["ridge_alpha"]),
    )
    model = GRUModalDynamics(
        num_modes=environment_config.num_modes,
        hidden_size=int(model_settings["hidden_size"]),
        num_layers=int(model_settings["num_layers"]),
        dropout=float(model_settings["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    shuffle_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(train_data.histories, train_data.targets),
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=shuffle_generator,
    )
    validation_loader = DataLoader(
        TensorDataset(validation_data.histories, validation_data.targets),
        batch_size=int(training["batch_size"]),
        shuffle=False,
    )
    mean_device = mean.to(device)
    scale_device = scale.to(device)
    total_epochs = int(training["epochs"])
    best_validation_mse = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    history_path = output_dir / "loss_history.csv"
    epochs = progress_bar(
        range(1, total_epochs + 1),
        description="S3 GRU训练总进度",
        unit="轮",
    )
    for epoch in epochs:
        train_mse = _run_epoch(
            model,
            train_loader,
            device,
            mean_device,
            scale_device,
            optimizer,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            epoch=epoch,
            total_epochs=total_epochs,
            stage="训练",
        )
        validation_mse = _run_epoch(
            model,
            validation_loader,
            device,
            mean_device,
            scale_device,
            optimizer=None,
            gradient_clip_norm=0,
            epoch=epoch,
            total_epochs=total_epochs,
            stage="验证",
        )
        row = {
            "epoch": epoch,
            "train_normalized_mse": train_mse,
            "validation_normalized_mse": validation_mse,
        }
        history.append(row)
        _write_history(history_path, history)
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_settings": dict(model_settings),
                    "normalization_mean": mean,
                    "normalization_scale": scale,
                    "ridge_weight": ridge_weight,
                    "ridge_bias": ridge_bias,
                    "best_epoch": best_epoch,
                    "best_validation_normalized_mse": best_validation_mse,
                    "experiment_config_sha256": _sha256(experiment_path),
                    "environment_config_sha256": _sha256(environment_path),
                    "sequence_length": int(dataset_settings["sequence_length"]),
                    "num_modes": environment_config.num_modes,
                },
                checkpoint_path,
            )
            progress_message(
                f"第 {epoch}/{total_epochs} 轮保存新的最佳权重："
                f"验证归一化MSE={best_validation_mse:.6f}"
            )
        update_progress(
            epochs,
            device=device,
            metrics={
                "训练MSE": train_mse,
                "验证MSE": validation_mse,
                "最佳MSE": best_validation_mse,
            },
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    prediction = _predict(
        model,
        validation_data,
        device,
        mean_device,
        scale_device,
        int(training["batch_size"]),
    )
    linear_prediction = ridge_autoregressive_forecast(
        validation_data.histories,
        mean,
        scale,
        ridge_weight,
        ridge_bias,
    )
    raw_comparison = paired_episode_comparison(
        prediction,
        linear_prediction,
        validation_data.targets,
        scale,
        validation_data.episode_seed,
        validation_data.condition_index,
    )
    gate = condition_gate(
        raw_comparison,
        min_mean_skill_score=float(gate_settings["min_mean_skill_score"]),
        min_ci95_low=float(gate_settings["min_ci95_low"]),
        require_every_condition_positive=bool(
            gate_settings["require_every_condition_positive"]
        ),
    )
    comparison = _name_conditions(raw_comparison, validation_conditions)
    gate["condition_mean_skill_scores"] = {
        validation_conditions[int(index)].identifier: value
        for index, value in gate["condition_mean_skill_scores"].items()
    }
    constant_velocity_prediction = constant_velocity_forecast(
        validation_data.histories,
        float(baseline_settings["secondary_velocity_gain"]),
    )
    secondary_comparison = paired_episode_comparison(
        prediction,
        constant_velocity_prediction,
        validation_data.targets,
        scale,
        validation_data.episode_seed,
        validation_data.condition_index,
    )
    secondary_comparison = _name_conditions(
        secondary_comparison, validation_conditions
    )
    duration = perf_counter() - started
    result = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "s3_dynamics_training_v1",
        },
        "experiment": {
            "id": "AO-S3-GRU-DYNAMICS-GATE",
            "type": "training",
            "status": "completed",
            "command": (
                ".\\.venv\\Scripts\\python.exe scripts\\train_s3_dynamics.py "
                "--config configs\\experiments\\s3_dynamics_v1.yaml"
            ),
            "working_directory": str(PROJECT_ROOT),
            "duration_seconds": duration,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "inputs": {
            "experiment_config": str(experiment_path.relative_to(PROJECT_ROOT)),
            "experiment_config_sha256": _sha256(experiment_path),
            "environment_config": str(environment_path.relative_to(PROJECT_ROOT)),
            "environment_config_sha256": _sha256(environment_path),
            "dataset_h5": str(dataset_path.relative_to(PROJECT_ROOT)),
            "dataset_sha256": _sha256(dataset_path),
            "train_samples": len(train_data.histories),
            "validation_samples": len(validation_data.histories),
            "train_conditions": [asdict(item) for item in train_conditions],
            "validation_conditions": [asdict(item) for item in validation_conditions],
            "sealed_test_accessed": False,
        },
        "training": {
            "best_epoch": best_epoch,
            "best_validation_normalized_mse": best_validation_mse,
            "history": history,
        },
        "validation_comparison": comparison,
        "secondary_constant_velocity_comparison": secondary_comparison,
        "gate": gate,
        "outputs": {
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "loss_history": str(history_path.relative_to(PROJECT_ROOT)),
        },
        "next_action": (
            "Tell the assistant that training is complete. Do not open the sealed test until "
            "the validation result and hashes have been read and the model is frozen."
        ),
        "interpretation_boundary": (
            "This gate compares one-step oracle-modal GRU prediction with a linear ridge model "
            "using the same eight-frame history and training data in pure simulation. "
            "It is not an RL policy, closed-loop superiority, holographic sensing, or real-SLM evidence."
        ),
        "anomalies": [],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "experiment_result.md").write_text(
        _markdown_result(result), encoding="utf-8"
    )
    progress_message(f"训练完成：{summary_path}")
    progress_message(
        f"验证门槛：{gate['validation_gate']}。请停止在这里并告诉助手“训练完成”。"
    )


def _generate_with_progress(
    description: str,
    environment_config: Any,
    device: torch.device,
    conditions: list[DynamicsCondition],
    settings: dict[str, Any],
) -> TemporalDynamicsData:
    total = len(conditions) * int(settings["frames_per_episode"])
    with counted_progress(total=total, description=description, unit="步") as bar:
        return generate_temporal_dynamics_data(
            environment_config,
            device,
            conditions,
            int(settings["sequence_length"]),
            int(settings["frames_per_episode"]),
            float(settings["random_action_std_rad"]),
            progress_callback=lambda completed, _: advance_to(bar, completed),
        )


def _run_epoch(
    model: GRUModalDynamics,
    loader: DataLoader,
    device: torch.device,
    mean: torch.Tensor,
    scale: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    *,
    gradient_clip_norm: float,
    epoch: int,
    total_epochs: int,
    stage: str,
) -> float:
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
    for histories, targets in batches:
        histories = histories.to(device)
        targets = targets.to(device)
        normalized_history = normalize_modal(histories, mean, scale)
        normalized_target = normalize_modal(targets, mean, scale)
        with torch.set_grad_enabled(training):
            prediction = model(normalized_history)
            loss = torch.mean((prediction - normalized_target).square())
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
        batch_size = len(histories)
        total += float(loss.detach()) * batch_size
        count += batch_size
        update_progress(
            batches,
            device=device,
            metrics={"平均MSE": total / count},
        )
    return total / max(count, 1)


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
    batches = progress_bar(loader, description="汇总验证回合", unit="批", leave=True)
    with torch.no_grad():
        for (histories,) in batches:
            normalized = normalize_modal(histories.to(device), mean, scale)
            prediction = denormalize_modal(model(normalized), mean, scale)
            predictions.append(prediction.cpu())
            update_progress(batches, device=device, metrics={})
    return torch.cat(predictions)


def _export_dataset(
    path: Path,
    train: TemporalDynamicsData,
    validation: TemporalDynamicsData,
    train_conditions: list[DynamicsCondition],
    validation_conditions: list[DynamicsCondition],
    experiment_path: Path,
    environment_path: Path,
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["stage"] = "S3"
        handle.attrs["observation_kind"] = "oracle_disturbance_modal_history"
        handle.attrs["target_kind"] = "next_disturbance_modal"
        handle.attrs["experiment_config"] = str(experiment_path.relative_to(PROJECT_ROOT))
        handle.attrs["environment_config"] = str(environment_path.relative_to(PROJECT_ROOT))
        handle.attrs["sealed_test_included"] = False
        for split, data, conditions in (
            ("train", train, train_conditions),
            ("validation", validation, validation_conditions),
        ):
            group = handle.create_group(split)
            group.attrs["conditions_json"] = json.dumps(
                [asdict(item) for item in conditions], ensure_ascii=False
            )
            group.create_dataset("history", data=data.histories.numpy(), compression="gzip")
            group.create_dataset("target", data=data.targets.numpy(), compression="gzip")
            group.create_dataset("episode_seed", data=data.episode_seed.numpy())
            group.create_dataset("condition_index", data=data.condition_index.numpy())
            group.create_dataset("target_step", data=data.target_step.numpy())


def _write_history(path: Path, history: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


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


def _markdown_result(result: dict[str, Any]) -> str:
    passport = result["material_passport"]
    experiment = result["experiment"]
    comparison = result["validation_comparison"]
    gate = result["gate"]
    return f"""## Material Passport

- Origin Skill: {passport['origin_skill']}
- Origin Mode: {passport['origin_mode']}
- Origin Date: {passport['origin_date']}
- Verification Status: {passport['verification_status']}
- Version Label: {passport['version_label']}

## Experiment Result

- **ID**: {experiment['id']}
- **Type**: {experiment['type']}
- **Status**: {experiment['status']}
- **Command**: `{experiment['command']}`
- **Duration**: {experiment['duration_seconds']:.3f} seconds
- **Sealed Test Accessed**: False

### Validation Comparison

| Metric | GRU | Same-history linear ridge baseline |
|---|---:|---:|
| Normalized RMSE | {comparison['gru_normalized_rmse']['mean']:.6f} | {comparison['linear_normalized_rmse']['mean']:.6f} |

- **Paired episode skill score**: {comparison['skill_score']['mean']:.6f}
- **95% CI**: [{comparison['skill_score']['ci95_low']:.6f}, {comparison['skill_score']['ci95_high']:.6f}]
- **Validation Gate**: {gate['validation_gate']}
- **Boundary**: {result['interpretation_boundary']}

### Next Action

停止在这里并通知助手读取结果；不要自行打开封存测试。
"""


def _conditions(values: list[dict[str, Any]]) -> list[DynamicsCondition]:
    return [DynamicsCondition.from_mapping(item) for item in values]


def _assert_disjoint_conditions(*groups: list[DynamicsCondition]) -> None:
    identifiers = [item.identifier for group in groups for item in group]
    seeds = [item.base_seed for group in groups for item in group]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("condition identifiers must be unique across all splits")
    if len(seeds) != len(set(seeds)):
        raise ValueError("base seeds must be unique across all splits")


def _enable_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


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
