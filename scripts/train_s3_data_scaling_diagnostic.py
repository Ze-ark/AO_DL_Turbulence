"""运行S3数据量/模型容量学习曲线；不读取原S3封存测试。"""

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
    fit_ridge_autoregression,
    generate_temporal_dynamics_data,
    load_s1_config,
    modal_normalization,
    paired_episode_comparison,
    ridge_autoregressive_forecast,
)
from src.simulation.learning_curve import (
    balanced_episode_subset,
    classify_capacity,
    classify_learning_curve,
    summarize_replicates,
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
    parser.add_argument(
        "--config",
        default="configs/experiments/s3_data_scaling_diagnostic_v1.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _project_path(args.config)
    config = _load_yaml(config_path)
    environment_path = _project_path(config["environment_config"])
    environment_config, requested_device = load_s1_config(environment_path)
    device = resolve_device(requested_device)
    dataset_settings = config["dataset"]
    diagnostic = config["diagnostic"]
    training = config["training"]
    outputs = config["outputs"]
    pool_mappings = list(dataset_settings["train_pool_conditions"])
    pool_conditions = [DynamicsCondition.from_mapping(item) for item in pool_mappings]
    validation_conditions = [
        DynamicsCondition.from_mapping(item)
        for item in dataset_settings["diagnostic_validation_conditions"]
    ]
    _validate_seed_safety(config, pool_conditions, validation_conditions)

    output_dir = _project_path(outputs["directory"])
    checkpoint_dir = _project_path(outputs["checkpoint_directory"])
    dataset_path = _project_path(outputs["dataset_h5"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    progress_message(
        f"诊断设备：{torch.cuda.get_device_name(device)}；固定更新次数："
        f"{int(training['optimizer_steps'])}；每个设置重复"
        f"{len(diagnostic['initialization_seeds'])}次。"
    )
    progress_message("原S3封存种子仅做冲突检查，不生成、不读取、不评估。")

    pool_data = _generate_with_progress(
        "生成384回合训练池",
        environment_config,
        device,
        pool_conditions,
        dataset_settings,
    )
    validation_data = _generate_with_progress(
        "生成新诊断验证集",
        environment_config,
        device,
        validation_conditions,
        dataset_settings,
    )
    _export_dataset(
        dataset_path,
        pool_data,
        validation_data,
        pool_mappings,
        validation_conditions,
        config_path,
        environment_path,
    )

    scales = [int(value) for value in diagnostic["episode_scales"]]
    hidden_sizes = [int(value) for value in diagnostic["capacity_hidden_sizes"]]
    primary_hidden = int(diagnostic["primary_hidden_size"])
    capacity_scale = int(diagnostic["capacity_episode_scale"])
    initialization_seeds = [int(value) for value in diagnostic["initialization_seeds"]]
    physical_ids = [str(item["physical_id"]) for item in pool_mappings]
    subsets = {
        scale: balanced_episode_subset(pool_data, physical_ids, scale)
        for scale in scales
    }
    run_specs = {
        (scale, primary_hidden, seed)
        for scale in scales
        for seed in initialization_seeds
    }
    run_specs |= {
        (capacity_scale, hidden_size, seed)
        for hidden_size in hidden_sizes
        for seed in initialization_seeds
    }
    ordered_specs = sorted(run_specs)
    histories_dir = output_dir / "loss_histories"
    histories_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    records_path = output_dir / "run_records.csv"
    overall = progress_bar(
        ordered_specs,
        description="学习曲线模型总进度",
        unit="模型",
    )
    baseline_cache: dict[int, dict[str, torch.Tensor]] = {}

    for episode_scale, hidden_size, initialization_seed in overall:
        subset = subsets[episode_scale]
        if episode_scale not in baseline_cache:
            mean, scale = modal_normalization(subset)
            ridge_weight, ridge_bias = fit_ridge_autoregression(
                subset.histories,
                subset.targets,
                mean,
                scale,
                float(config["baseline"]["ridge_alpha"]),
            )
            ridge_prediction = ridge_autoregressive_forecast(
                validation_data.histories,
                mean,
                scale,
                ridge_weight,
                ridge_bias,
            )
            baseline_cache[episode_scale] = {
                "mean": mean,
                "scale": scale,
                "ridge_weight": ridge_weight,
                "ridge_bias": ridge_bias,
                "ridge_prediction": ridge_prediction,
            }
        baseline = baseline_cache[episode_scale]
        run_id = f"episodes_{episode_scale}_hidden_{hidden_size}_seed_{initialization_seed}"
        record = _train_one_model(
            run_id=run_id,
            train_data=subset,
            validation_data=validation_data,
            device=device,
            hidden_size=hidden_size,
            initialization_seed=initialization_seed,
            mean=baseline["mean"],
            scale=baseline["scale"],
            ridge_weight=baseline["ridge_weight"],
            ridge_bias=baseline["ridge_bias"],
            ridge_prediction=baseline["ridge_prediction"],
            config=config,
            checkpoint_path=checkpoint_dir / f"{run_id}.pt",
            history_path=histories_dir / f"{run_id}.csv",
        )
        record["episode_scale"] = episode_scale
        record["hidden_size"] = hidden_size
        record["initialization_seed"] = initialization_seed
        records.append(record)
        _write_records(records_path, records)
        update_progress(
            overall,
            device=device,
            metrics={
                "GRU误差": float(record["gru_rmse"]),
                "岭回归误差": float(record["ridge_rmse"]),
            },
        )
        torch.cuda.empty_cache()

    learning_records = [
        item for item in records if int(item["hidden_size"]) == primary_hidden
    ]
    capacity_records = [
        item for item in records if int(item["episode_scale"]) == capacity_scale
    ]
    scale_summaries = summarize_replicates(learning_records, "episode_scale")
    capacity_summaries = summarize_replicates(capacity_records, "hidden_size")
    classification = config["classification"]
    data_verdict = classify_learning_curve(
        scale_summaries,
        supported_min_192_to_384_rmse_reduction=float(
            classification["supported_min_192_to_384_rmse_reduction"]
        ),
        supported_min_gap_ratio_reduction_48_to_384=float(
            classification["supported_min_gap_ratio_reduction_48_to_384"]
        ),
        not_supported_max_192_to_384_rmse_reduction=float(
            classification["not_supported_max_192_to_384_rmse_reduction"]
        ),
    )
    capacity_verdict = classify_capacity(
        capacity_summaries,
        reference_hidden_size=primary_hidden,
        min_smaller_model_improvement=float(
            classification["overparameterized_min_smaller_model_improvement"]
        ),
    )
    _write_summary_csv(output_dir / "learning_curve.csv", "episode_scale", scale_summaries)
    _write_summary_csv(output_dir / "capacity_curve.csv", "hidden_size", capacity_summaries)
    result = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "s3_data_scaling_diagnostic_v1",
        },
        "experiment": {
            "id": "AO-S3-DATA-SCALING-DIAGNOSTIC",
            "type": "training",
            "status": "completed",
            "command": (
                ".\\.venv\\Scripts\\python.exe "
                "scripts\\train_s3_data_scaling_diagnostic.py --config "
                "configs\\experiments\\s3_data_scaling_diagnostic_v1.yaml"
            ),
            "duration_seconds": perf_counter() - started,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "inputs": {
            "config": str(config_path.relative_to(PROJECT_ROOT)),
            "config_sha256": _sha256(config_path),
            "environment_config": str(environment_path.relative_to(PROJECT_ROOT)),
            "environment_config_sha256": _sha256(environment_path),
            "dataset_h5": str(dataset_path.relative_to(PROJECT_ROOT)),
            "dataset_sha256": _sha256(dataset_path),
            "pool_samples": len(pool_data.histories),
            "validation_samples": len(validation_data.histories),
            "protected_original_sealed_accessed": False,
        },
        "design": {
            "episode_scales": scales,
            "primary_hidden_size": primary_hidden,
            "capacity_episode_scale": capacity_scale,
            "capacity_hidden_sizes": hidden_sizes,
            "initialization_seeds": initialization_seeds,
            "fixed_optimizer_steps": int(training["optimizer_steps"]),
            "run_count": len(records),
        },
        "run_records": records,
        "learning_curve": scale_summaries,
        "capacity_curve": capacity_summaries,
        "data_limited_diagnosis": data_verdict,
        "capacity_diagnosis": capacity_verdict,
        "outputs": {
            "run_records_csv": str(records_path.relative_to(PROJECT_ROOT)),
            "learning_curve_csv": str(
                (output_dir / "learning_curve.csv").relative_to(PROJECT_ROOT)
            ),
            "capacity_curve_csv": str(
                (output_dir / "capacity_curve.csv").relative_to(PROJECT_ROOT)
            ),
            "checkpoint_directory": str(checkpoint_dir.relative_to(PROJECT_ROOT)),
        },
        "interpretation_boundary": (
            "This is an exploratory development diagnostic with fresh train/validation seeds. "
            "It does not open the original S3 sealed test and cannot license RL by itself."
        ),
        "anomalies": [],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "experiment_result.md").write_text(
        _markdown_result(result), encoding="utf-8"
    )
    progress_message(f"诊断训练完成：{summary_path}")
    progress_message("请停止在这里并告诉助手“诊断训练完成”。")


def _train_one_model(
    *,
    run_id: str,
    train_data: TemporalDynamicsData,
    validation_data: TemporalDynamicsData,
    device: torch.device,
    hidden_size: int,
    initialization_seed: int,
    mean: torch.Tensor,
    scale: torch.Tensor,
    ridge_weight: torch.Tensor,
    ridge_bias: torch.Tensor,
    ridge_prediction: torch.Tensor,
    config: dict[str, Any],
    checkpoint_path: Path,
    history_path: Path,
) -> dict[str, Any]:
    _enable_determinism(initialization_seed)
    model_settings = config["model"]
    training = config["training"]
    model = GRUModalDynamics(
        num_modes=train_data.histories.shape[-1],
        hidden_size=hidden_size,
        num_layers=int(model_settings["num_layers"]),
        dropout=float(model_settings["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    loader = DataLoader(
        TensorDataset(train_data.histories, train_data.targets),
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(initialization_seed),
    )
    iterator = iter(loader)
    mean_device = mean.to(device)
    scale_device = scale.to(device)
    total_steps = int(training["optimizer_steps"])
    validation_interval = int(training["validation_interval_steps"])
    best_validation_mse = float("inf")
    best_step = 0
    interval_loss = 0.0
    interval_count = 0
    history: list[dict[str, float | int]] = []
    started = perf_counter()
    steps = progress_bar(
        range(1, total_steps + 1),
        description=run_id,
        unit="步",
        leave=False,
    )
    for step in steps:
        try:
            histories, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            histories, targets = next(iterator)
        histories = histories.to(device)
        targets = targets.to(device)
        normalized_history = normalize_modal(histories, mean_device, scale_device)
        normalized_target = normalize_modal(targets, mean_device, scale_device)
        model.train()
        prediction = model(normalized_history)
        loss = torch.mean((prediction - normalized_target).square())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip_norm"])
        )
        optimizer.step()
        interval_loss += float(loss.detach())
        interval_count += 1

        if step % validation_interval == 0 or step == total_steps:
            validation_mse = _validation_mse(
                model,
                validation_data,
                device,
                mean_device,
                scale_device,
                int(training["batch_size"]),
            )
            train_mse = interval_loss / interval_count
            row = {
                "step": step,
                "effective_epochs": step
                * int(training["batch_size"])
                / len(train_data.histories),
                "train_normalized_mse": train_mse,
                "validation_normalized_mse": validation_mse,
            }
            history.append(row)
            _write_history(history_path, history)
            interval_loss = 0.0
            interval_count = 0
            if validation_mse < best_validation_mse:
                best_validation_mse = validation_mse
                best_step = step
                torch.save(
                    {
                        "model": model.state_dict(),
                        "hidden_size": hidden_size,
                        "initialization_seed": initialization_seed,
                        "best_step": best_step,
                        "best_validation_normalized_mse": best_validation_mse,
                        "normalization_mean": mean,
                        "normalization_scale": scale,
                        "ridge_weight": ridge_weight,
                        "ridge_bias": ridge_bias,
                    },
                    checkpoint_path,
                )
            update_progress(
                steps,
                device=device,
                metrics={
                    "训练MSE": train_mse,
                    "验证MSE": validation_mse,
                    "最佳MSE": best_validation_mse,
                },
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    gru_prediction = _predict(
        model,
        validation_data,
        device,
        mean_device,
        scale_device,
        int(training["batch_size"]),
    )
    comparison = paired_episode_comparison(
        gru_prediction,
        ridge_prediction,
        validation_data.targets,
        scale,
        validation_data.episode_seed,
        validation_data.condition_index,
    )
    return {
        "run_id": run_id,
        "best_step": best_step,
        "best_validation_normalized_mse": best_validation_mse,
        "gru_rmse": comparison["gru_normalized_rmse"]["mean"],
        "ridge_rmse": comparison["linear_normalized_rmse"]["mean"],
        "skill_score": comparison["skill_score"]["mean"],
        "skill_ci95_low": comparison["skill_score"]["ci95_low"],
        "skill_ci95_high": comparison["skill_score"]["ci95_high"],
        "duration_seconds": perf_counter() - started,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "loss_history": str(history_path.relative_to(PROJECT_ROOT)),
    }


def _validation_mse(
    model: GRUModalDynamics,
    data: TemporalDynamicsData,
    device: torch.device,
    mean: torch.Tensor,
    scale: torch.Tensor,
    batch_size: int,
) -> float:
    loader = DataLoader(
        TensorDataset(data.histories, data.targets),
        batch_size=batch_size,
        shuffle=False,
    )
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for histories, targets in loader:
            histories = histories.to(device)
            targets = targets.to(device)
            prediction = model(normalize_modal(histories, mean, scale))
            normalized_target = normalize_modal(targets, mean, scale)
            loss = torch.mean((prediction - normalized_target).square())
            total += float(loss) * len(histories)
            count += len(histories)
    return total / count


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
    with torch.no_grad():
        for (histories,) in loader:
            normalized = normalize_modal(histories.to(device), mean, scale)
            predictions.append(denormalize_modal(model(normalized), mean, scale).cpu())
    return torch.cat(predictions)


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


def _validate_seed_safety(
    config: dict[str, Any],
    pool: list[DynamicsCondition],
    validation: list[DynamicsCondition],
) -> None:
    protected = _load_yaml(_project_path(config["protected_original_experiment"]))
    protected_items = (
        list(protected["dataset"]["train_conditions"])
        + list(protected["dataset"]["validation_conditions"])
        + list(protected["dataset"]["sealed_test_conditions"])
    )
    protected_seeds = {int(item["base_seed"]) for item in protected_items}
    diagnostic_seeds = {item.base_seed for item in pool + validation}
    if protected_seeds & diagnostic_seeds:
        raise ValueError("diagnostic seeds overlap the original S3 experiment")
    if len(diagnostic_seeds) != len(pool) + len(validation):
        raise ValueError("diagnostic base seeds must be unique")


def _export_dataset(
    path: Path,
    pool: TemporalDynamicsData,
    validation: TemporalDynamicsData,
    pool_mappings: list[dict[str, Any]],
    validation_conditions: list[DynamicsCondition],
    config_path: Path,
    environment_path: Path,
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["stage"] = "S3"
        handle.attrs["purpose"] = "data_scaling_diagnostic"
        handle.attrs["original_sealed_test_included"] = False
        handle.attrs["config"] = str(config_path.relative_to(PROJECT_ROOT))
        handle.attrs["environment_config"] = str(environment_path.relative_to(PROJECT_ROOT))
        for split, data in (("train_pool", pool), ("validation", validation)):
            group = handle.create_group(split)
            group.create_dataset("history", data=data.histories.numpy(), compression="gzip")
            group.create_dataset("target", data=data.targets.numpy(), compression="gzip")
            group.create_dataset("episode_seed", data=data.episode_seed.numpy())
            group.create_dataset("condition_index", data=data.condition_index.numpy())
            group.create_dataset("target_step", data=data.target_step.numpy())
        handle["train_pool"].attrs["conditions_json"] = json.dumps(
            pool_mappings, ensure_ascii=False
        )
        handle["validation"].attrs["conditions_json"] = json.dumps(
            [asdict(item) for item in validation_conditions], ensure_ascii=False
        )


def _write_history(path: Path, history: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _write_summary_csv(
    path: Path,
    grouping_key: str,
    summaries: dict[int, dict[str, float]],
) -> None:
    rows = [{grouping_key: key, **value} for key, value in sorted(summaries.items())]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_result(result: dict[str, Any]) -> str:
    passport = result["material_passport"]
    return f"""## Material Passport

- Origin Skill: {passport['origin_skill']}
- Origin Mode: {passport['origin_mode']}
- Origin Date: {passport['origin_date']}
- Verification Status: {passport['verification_status']}
- Version Label: {passport['version_label']}

## S3 Data Scaling Diagnostic

- **Status**: {result['experiment']['status']}
- **Runs**: {result['design']['run_count']}
- **Fixed Optimizer Steps per Run**: {result['design']['fixed_optimizer_steps']}
- **Data-Limited Verdict**: {result['data_limited_diagnosis']['verdict']}
- **Capacity Verdict**: {result['capacity_diagnosis']['verdict']}
- **Original Sealed Test Accessed**: False
- **Boundary**: {result['interpretation_boundary']}
"""


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
