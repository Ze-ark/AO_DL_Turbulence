"""运行S3的128/256维GRU收敛与容量复查；不读取原S3封存测试。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, stdev
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
    GRUModalDynamics,
    TemporalDynamicsData,
    fit_ridge_autoregression,
    load_s1_config,
    modal_normalization,
    paired_episode_comparison,
    ridge_autoregressive_forecast,
)
from src.simulation.convergence import (
    ConvergenceTracker,
    classify_larger_model,
    classify_more_data,
    convergence_label,
)
from src.simulation.learning_curve import balanced_episode_subset
from src.simulation.temporal_dynamics import denormalize_modal, normalize_modal
from src.training_progress import progress_bar, progress_message, update_progress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/s3_convergence_capacity_v1.yaml",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只检查CUDA、数据哈希和实验矩阵，不启动训练",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _project_path(args.config)
    config = _load_yaml(config_path)
    environment_path = _project_path(config["environment_config"])
    environment_config, requested_device = load_s1_config(environment_path)
    device = resolve_device(requested_device)
    dataset_path = _project_path(config["dataset"]["h5"])
    upstream_path = _project_path(config["upstream_diagnostic_summary"])
    upstream = _validate_upstream(
        config,
        dataset_path,
        upstream_path,
        environment_config.batch_size,
    )
    pool_data, validation_data, pool_mappings, validation_mappings = _load_dataset(
        dataset_path,
        sequence_length=int(config["dataset"]["sequence_length"]),
        num_modes=int(config["dataset"]["num_modes"]),
    )

    experiment = config["experiment"]
    episode_scales = [int(value) for value in experiment["episode_scales"]]
    hidden_sizes = [int(value) for value in experiment["hidden_sizes"]]
    initialization_seeds = [int(value) for value in experiment["initialization_seeds"]]
    physical_ids = [str(item["physical_id"]) for item in pool_mappings]
    _, metric_scale = modal_normalization(pool_data)
    subsets = {
        scale: balanced_episode_subset(pool_data, physical_ids, scale)
        for scale in episode_scales
    }
    run_specs = [
        (scale, hidden_size, seed)
        for scale in episode_scales
        for hidden_size in hidden_sizes
        for seed in initialization_seeds
    ]
    if args.preflight_only:
        progress_message(f"CUDA设备：{torch.cuda.get_device_name(device)}")
        progress_message(
            f"数据检查通过：训练池{len(pool_data.histories)}个窗口，"
            f"开发验证集{len(validation_data.histories)}个窗口。"
        )
        for hidden_size in hidden_sizes:
            model = GRUModalDynamics(
                num_modes=int(config["dataset"]["num_modes"]),
                hidden_size=hidden_size,
                num_layers=int(config["model"]["num_layers"]),
                dropout=float(config["model"]["dropout"]),
            )
            parameters = sum(parameter.numel() for parameter in model.parameters())
            progress_message(f"隐藏维度{hidden_size}：{parameters}个参数。")
        progress_message(f"预检完成：正式训练共{len(run_specs)}个模型；未启动训练。")
        return

    outputs = config["outputs"]
    output_dir = _project_path(outputs["directory"])
    checkpoint_dir = _project_path(outputs["checkpoint_directory"])
    histories_dir = output_dir / "loss_histories"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    histories_dir.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    progress_message(
        f"收敛复查设备：{torch.cuda.get_device_name(device)}；共{len(run_specs)}个模型；"
        f"每个模型最少{int(training['min_optimizer_steps'])}步、最多"
        f"{int(training['max_optimizer_steps'])}步。"
    )
    progress_message("复用已哈希的开发数据；原S3封存测试不会被读取。")

    records: list[dict[str, Any]] = []
    condition_records: list[dict[str, Any]] = []
    records_path = output_dir / "run_records.csv"
    condition_records_path = output_dir / "condition_records.csv"
    baseline_cache: dict[int, dict[str, torch.Tensor]] = {}
    started = perf_counter()
    overall = progress_bar(run_specs, description="收敛复查总进度", unit="模型")
    for episode_scale, hidden_size, initialization_seed in overall:
        train_data = subsets[episode_scale]
        if episode_scale not in baseline_cache:
            baseline_cache[episode_scale] = _fit_baseline(
                train_data,
                validation_data,
                ridge_alpha=float(config["baseline"]["ridge_alpha"]),
            )
        baseline = baseline_cache[episode_scale]
        run_id = f"episodes_{episode_scale}_hidden_{hidden_size}_seed_{initialization_seed}"
        record, per_condition = _train_one_model(
            run_id=run_id,
            episode_scale=episode_scale,
            hidden_size=hidden_size,
            initialization_seed=initialization_seed,
            train_data=train_data,
            validation_data=validation_data,
            validation_mappings=validation_mappings,
            device=device,
            mean_tensor=baseline["mean"],
            scale_tensor=baseline["scale"],
            metric_scale=metric_scale,
            ridge_weight=baseline["ridge_weight"],
            ridge_bias=baseline["ridge_bias"],
            ridge_prediction=baseline["ridge_prediction"],
            config=config,
            checkpoint_path=checkpoint_dir / f"{run_id}.pt",
            history_path=histories_dir / f"{run_id}.csv",
        )
        records.append(record)
        condition_records.extend(per_condition)
        _write_csv(records_path, records)
        _write_csv(condition_records_path, condition_records)
        update_progress(
            overall,
            device=device,
            metrics={
                "GRU误差": float(record["gru_rmse"]),
                "岭回归误差": float(record["ridge_rmse"]),
            },
        )
        torch.cuda.empty_cache()

    setting_summaries = _summarize_settings(records)
    condition_summaries = _summarize_conditions(condition_records)
    classification = config["classification"]
    capacity_verdict = classify_larger_model(
        setting_summaries,
        episode_scales=episode_scales,
        reference_hidden_size=128,
        candidate_hidden_size=256,
        min_improvement=float(classification["min_larger_model_rmse_improvement"]),
    )
    data_verdict = classify_more_data(
        setting_summaries,
        smaller_episode_scale=min(episode_scales),
        larger_episode_scale=max(episode_scales),
        hidden_sizes=hidden_sizes,
        min_improvement=float(classification["min_more_data_rmse_improvement"]),
    )
    setting_rows = [
        {"episode_scale": key[0], "hidden_size": key[1], **values}
        for key, values in sorted(setting_summaries.items())
    ]
    condition_rows = [
        {
            "episode_scale": key[0],
            "hidden_size": key[1],
            "condition_id": key[2],
            **values,
        }
        for key, values in sorted(condition_summaries.items())
    ]
    settings_path = output_dir / "setting_summary.csv"
    conditions_path = output_dir / "condition_summary.csv"
    _write_csv(settings_path, setting_rows)
    _write_csv(conditions_path, condition_rows)

    anomalies = [
        f"{record['run_id']} reached max steps while still improving"
        for record in records
        if record["convergence_label"] == "STILL_IMPROVING_AT_MAX_STEPS"
    ]
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "s3_convergence_capacity_v1",
        },
        "experiment": {
            "id": "AO-S3-CONVERGENCE-CAPACITY",
            "type": "training",
            "status": "completed",
            "command": (
                ".\\.venv\\Scripts\\python.exe "
                "scripts\\train_s3_convergence_capacity.py --config "
                "configs\\experiments\\s3_convergence_capacity_v1.yaml"
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
            "upstream_diagnostic_summary": str(upstream_path.relative_to(PROJECT_ROOT)),
            "upstream_diagnostic_summary_sha256": _sha256(upstream_path),
            "upstream_experiment_status": upstream["experiment"]["status"],
            "dataset_h5": str(dataset_path.relative_to(PROJECT_ROOT)),
            "dataset_sha256": _sha256(dataset_path),
            "pool_samples": len(pool_data.histories),
            "validation_samples": len(validation_data.histories),
            "metric_scale_source": config["dataset"]["metric_scale_source"],
            "protected_original_sealed_accessed": False,
        },
        "design": {
            "episode_scales": episode_scales,
            "hidden_sizes": hidden_sizes,
            "initialization_seeds": initialization_seeds,
            "run_count": len(records),
            "min_optimizer_steps": int(training["min_optimizer_steps"]),
            "max_optimizer_steps": int(training["max_optimizer_steps"]),
            "early_stopping_patience_checks": int(
                training["early_stopping_patience_checks"]
            ),
            "early_stopping_relative_min_delta": float(
                training["early_stopping_relative_min_delta"]
            ),
        },
        "run_records": records,
        "setting_summaries": {
            f"episodes_{key[0]}_hidden_{key[1]}": values
            for key, values in sorted(setting_summaries.items())
        },
        "condition_summaries": {
            f"episodes_{key[0]}_hidden_{key[1]}_{key[2]}": values
            for key, values in sorted(condition_summaries.items())
        },
        "larger_model_diagnosis": capacity_verdict,
        "more_data_diagnosis": data_verdict,
        "convergence": {
            "early_stopped_runs": sum(bool(item["stopped_early"]) for item in records),
            "still_improving_at_max_runs": len(anomalies),
        },
        "outputs": {
            "run_records_csv": str(records_path.relative_to(PROJECT_ROOT)),
            "condition_records_csv": str(condition_records_path.relative_to(PROJECT_ROOT)),
            "setting_summary_csv": str(settings_path.relative_to(PROJECT_ROOT)),
            "condition_summary_csv": str(conditions_path.relative_to(PROJECT_ROOT)),
            "checkpoint_directory": str(checkpoint_dir.relative_to(PROJECT_ROOT)),
        },
        "interpretation_boundary": (
            "This is an exploratory convergence and capacity follow-up using development "
            "data already inspected in the prior diagnostic. It does not open the original "
            "S3 sealed test and cannot license RL by itself."
        ),
        "anomalies": anomalies,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "experiment_result.md").write_text(
        _markdown_result(summary), encoding="utf-8"
    )
    progress_message(f"收敛复查训练完成：{summary_path}")
    progress_message("请停止在这里并告诉助手“收敛复查训练完成”。")


def _train_one_model(
    *,
    run_id: str,
    episode_scale: int,
    hidden_size: int,
    initialization_seed: int,
    train_data: TemporalDynamicsData,
    validation_data: TemporalDynamicsData,
    validation_mappings: list[dict[str, Any]],
    device: torch.device,
    mean_tensor: torch.Tensor,
    scale_tensor: torch.Tensor,
    metric_scale: torch.Tensor,
    ridge_weight: torch.Tensor,
    ridge_bias: torch.Tensor,
    ridge_prediction: torch.Tensor,
    config: dict[str, Any],
    checkpoint_path: Path,
    history_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _enable_determinism(initialization_seed)
    model_settings = config["model"]
    training = config["training"]
    model = GRUModalDynamics(
        num_modes=train_data.histories.shape[-1],
        hidden_size=hidden_size,
        num_layers=int(model_settings["num_layers"]),
        dropout=float(model_settings["dropout"]),
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training["scheduler_factor"]),
        patience=int(training["scheduler_patience_checks"]),
        threshold=float(training["early_stopping_relative_min_delta"]),
        threshold_mode="rel",
        min_lr=float(training["minimum_learning_rate"]),
    )
    tracker = ConvergenceTracker(
        min_steps=int(training["min_optimizer_steps"]),
        patience_checks=int(training["early_stopping_patience_checks"]),
        relative_min_delta=float(training["early_stopping_relative_min_delta"]),
    )
    loader = DataLoader(
        TensorDataset(train_data.histories, train_data.targets),
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(initialization_seed),
    )
    iterator = iter(loader)
    mean_device = mean_tensor.to(device)
    scale_device = scale_tensor.to(device)
    max_steps = int(training["max_optimizer_steps"])
    validation_interval = int(training["validation_interval_steps"])
    best_validation_mse = float("inf")
    best_step = 0
    interval_loss = 0.0
    interval_count = 0
    history: list[dict[str, float | int | bool]] = []
    stopped_early = False
    started = perf_counter()
    steps = progress_bar(
        range(1, max_steps + 1),
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

        if step % validation_interval == 0 or step == max_steps:
            validation_mse = _validation_mse(
                model,
                validation_data,
                device,
                mean_device,
                scale_device,
                int(training["batch_size"]),
            )
            train_mse = interval_loss / interval_count
            scheduler.step(validation_mse)
            should_stop = tracker.update(step, validation_mse)
            if validation_mse < best_validation_mse:
                best_validation_mse = validation_mse
                best_step = step
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "hidden_size": hidden_size,
                        "initialization_seed": initialization_seed,
                        "episode_scale": episode_scale,
                        "best_step": best_step,
                        "best_validation_normalized_mse": best_validation_mse,
                        "normalization_mean": mean_tensor,
                        "normalization_scale": scale_tensor,
                        "metric_scale": metric_scale,
                        "ridge_weight": ridge_weight,
                        "ridge_bias": ridge_bias,
                    },
                    checkpoint_path,
                )
            row = {
                "step": step,
                "effective_epochs": step
                * int(training["batch_size"])
                / len(train_data.histories),
                "train_normalized_mse": train_mse,
                "validation_normalized_mse": validation_mse,
                "best_validation_normalized_mse": best_validation_mse,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "stale_checks": tracker.stale_checks,
                "stop_requested": should_stop,
            }
            history.append(row)
            _write_csv(history_path, history)
            interval_loss = 0.0
            interval_count = 0
            update_progress(
                steps,
                device=device,
                metrics={
                    "训练MSE": train_mse,
                    "验证MSE": validation_mse,
                    "最佳MSE": best_validation_mse,
                    "学习率": float(optimizer.param_groups[0]["lr"]),
                },
            )
            if should_stop:
                stopped_early = True
                progress_message(f"{run_id} 已满足早停条件，停止于第{step}步。")
                break
    steps.close()

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
        metric_scale,
        validation_data.episode_seed,
        validation_data.condition_index,
    )
    convergence = convergence_label(
        [float(item["validation_normalized_mse"]) for item in history],
        early_stopped=stopped_early,
        final_window_checks=int(training["final_window_checks"]),
        max_final_window_improvement=float(
            training["max_final_window_improvement"]
        ),
    )
    stop_step = int(history[-1]["step"])
    record = {
        "run_id": run_id,
        "episode_scale": episode_scale,
        "hidden_size": hidden_size,
        "initialization_seed": initialization_seed,
        "parameter_count": parameter_count,
        "best_step": best_step,
        "stop_step": stop_step,
        "stopped_early": stopped_early,
        "convergence_label": convergence["label"],
        "final_window_relative_improvement": convergence[
            "final_window_relative_improvement"
        ],
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
    condition_records = _condition_rows(
        run_id,
        episode_scale,
        hidden_size,
        initialization_seed,
        comparison["by_condition_index"],
        validation_mappings,
    )
    return record, condition_records


def _fit_baseline(
    train_data: TemporalDynamicsData,
    validation_data: TemporalDynamicsData,
    *,
    ridge_alpha: float,
) -> dict[str, torch.Tensor]:
    mean_tensor, scale_tensor = modal_normalization(train_data)
    ridge_weight, ridge_bias = fit_ridge_autoregression(
        train_data.histories,
        train_data.targets,
        mean_tensor,
        scale_tensor,
        ridge_alpha,
    )
    return {
        "mean": mean_tensor,
        "scale": scale_tensor,
        "ridge_weight": ridge_weight,
        "ridge_bias": ridge_bias,
        "ridge_prediction": ridge_autoregressive_forecast(
            validation_data.histories,
            mean_tensor,
            scale_tensor,
            ridge_weight,
            ridge_bias,
        ),
    }


def _validation_mse(
    model: GRUModalDynamics,
    data: TemporalDynamicsData,
    device: torch.device,
    mean_tensor: torch.Tensor,
    scale_tensor: torch.Tensor,
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
            prediction = model(normalize_modal(histories, mean_tensor, scale_tensor))
            normalized_target = normalize_modal(targets, mean_tensor, scale_tensor)
            total += float(torch.mean((prediction - normalized_target).square())) * len(
                histories
            )
            count += len(histories)
    return total / count


def _predict(
    model: GRUModalDynamics,
    data: TemporalDynamicsData,
    device: torch.device,
    mean_tensor: torch.Tensor,
    scale_tensor: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    loader = DataLoader(TensorDataset(data.histories), batch_size=batch_size, shuffle=False)
    predictions: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for (histories,) in loader:
            normalized = normalize_modal(histories.to(device), mean_tensor, scale_tensor)
            predictions.append(
                denormalize_modal(model(normalized), mean_tensor, scale_tensor).cpu()
            )
    return torch.cat(predictions)


def _validate_upstream(
    config: dict[str, Any],
    dataset_path: Path,
    upstream_path: Path,
    batch_size: int,
) -> dict[str, Any]:
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    if upstream["experiment"]["status"] != "completed":
        raise RuntimeError("upstream data-scaling diagnostic is not complete")
    if upstream["inputs"]["protected_original_sealed_accessed"]:
        raise RuntimeError("upstream diagnostic reports access to the original sealed test")
    upstream_dataset = _project_path(upstream["inputs"]["dataset_h5"])
    if upstream_dataset.resolve() != dataset_path.resolve():
        raise ValueError("configured dataset does not match the upstream diagnostic")
    if _sha256(dataset_path) != upstream["inputs"]["dataset_sha256"]:
        raise ValueError("diagnostic dataset hash no longer matches the upstream summary")

    protected = _load_yaml(_project_path(config["protected_original_experiment"]))
    protected_seeds = {
        int(item["base_seed"]) + offset
        for split in (
            "train_conditions",
            "validation_conditions",
            "sealed_test_conditions",
        )
        for item in protected["dataset"][split]
        for offset in range(batch_size)
    }
    with h5py.File(dataset_path, "r") as handle:
        if bool(handle.attrs.get("original_sealed_test_included", True)):
            raise RuntimeError("dataset does not prove that the original sealed test is absent")
        if set(handle.keys()) != {"train_pool", "validation"}:
            raise ValueError("diagnostic dataset must contain only train_pool and validation")
        diagnostic_seeds = set(handle["train_pool/episode_seed"][:].tolist())
        diagnostic_seeds.update(handle["validation/episode_seed"][:].tolist())
    if diagnostic_seeds & protected_seeds:
        raise ValueError("development data overlaps protected original S3 seeds")
    return upstream


def _load_dataset(
    path: Path,
    *,
    sequence_length: int,
    num_modes: int,
) -> tuple[
    TemporalDynamicsData,
    TemporalDynamicsData,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    with h5py.File(path, "r") as handle:
        pool = _read_temporal_group(handle["train_pool"])
        validation = _read_temporal_group(handle["validation"])
        pool_mappings = json.loads(handle["train_pool"].attrs["conditions_json"])
        validation_mappings = json.loads(handle["validation"].attrs["conditions_json"])
    pool.validate(sequence_length, num_modes)
    validation.validate(sequence_length, num_modes)
    train_seeds = set(pool.episode_seed.tolist())
    validation_seeds = set(validation.episode_seed.tolist())
    if train_seeds & validation_seeds:
        raise ValueError("train pool and validation episode seeds overlap")
    return pool, validation, pool_mappings, validation_mappings


def _read_temporal_group(group: h5py.Group) -> TemporalDynamicsData:
    return TemporalDynamicsData(
        histories=torch.from_numpy(group["history"][:]).to(torch.float32),
        targets=torch.from_numpy(group["target"][:]).to(torch.float32),
        episode_seed=torch.from_numpy(group["episode_seed"][:]).to(torch.int64),
        condition_index=torch.from_numpy(group["condition_index"][:]).to(torch.int64),
        target_step=torch.from_numpy(group["target_step"][:]).to(torch.int64),
    )


def _condition_rows(
    run_id: str,
    episode_scale: int,
    hidden_size: int,
    initialization_seed: int,
    by_condition: dict[str, Any],
    mappings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for index_text, metrics in sorted(by_condition.items(), key=lambda item: int(item[0])):
        mapping = mappings[int(index_text)]
        identifier = str(mapping.get("identifier", mapping.get("id", index_text)))
        rows.append(
            {
                "run_id": run_id,
                "episode_scale": episode_scale,
                "hidden_size": hidden_size,
                "initialization_seed": initialization_seed,
                "condition_id": identifier,
                "wind_speed_mps": float(mapping["wind_speed_mps"]),
                "wind_direction_deg": float(mapping["wind_direction_deg"]),
                "episodes": int(metrics["episodes"]),
                "gru_rmse": metrics["gru_normalized_rmse"]["mean"],
                "ridge_rmse": metrics["linear_normalized_rmse"]["mean"],
                "skill_score": metrics["skill_score"]["mean"],
                "skill_ci95_low": metrics["skill_score"]["ci95_low"],
                "skill_ci95_high": metrics["skill_score"]["ci95_high"],
            }
        )
    return rows


def _summarize_settings(
    records: list[dict[str, Any]],
) -> dict[tuple[int, int], dict[str, float]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (int(record["episode_scale"]), int(record["hidden_size"]))
        grouped.setdefault(key, []).append(record)
    return {
        key: {
            "runs": float(len(values)),
            "parameter_count": float(values[0]["parameter_count"]),
            "gru_rmse_mean": mean(float(item["gru_rmse"]) for item in values),
            "gru_rmse_std": _sample_std(float(item["gru_rmse"]) for item in values),
            "ridge_rmse_mean": mean(float(item["ridge_rmse"]) for item in values),
            "skill_score_mean": mean(float(item["skill_score"]) for item in values),
            "skill_score_std": _sample_std(
                float(item["skill_score"]) for item in values
            ),
            "best_step_mean": mean(float(item["best_step"]) for item in values),
            "stop_step_mean": mean(float(item["stop_step"]) for item in values),
            "early_stopped_fraction": mean(
                float(bool(item["stopped_early"])) for item in values
            ),
        }
        for key, values in grouped.items()
    }


def _summarize_conditions(
    records: list[dict[str, Any]],
) -> dict[tuple[int, int, str], dict[str, float]]:
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            int(record["episode_scale"]),
            int(record["hidden_size"]),
            str(record["condition_id"]),
        )
        grouped.setdefault(key, []).append(record)
    return {
        key: {
            "runs": float(len(values)),
            "wind_speed_mps": float(values[0]["wind_speed_mps"]),
            "wind_direction_deg": float(values[0]["wind_direction_deg"]),
            "gru_rmse_mean": mean(float(item["gru_rmse"]) for item in values),
            "gru_rmse_std": _sample_std(float(item["gru_rmse"]) for item in values),
            "ridge_rmse_mean": mean(float(item["ridge_rmse"]) for item in values),
            "skill_score_mean": mean(float(item["skill_score"]) for item in values),
            "skill_score_std": _sample_std(
                float(item["skill_score"]) for item in values
            ),
        }
        for key, values in grouped.items()
    }


def _sample_std(values: Any) -> float:
    materialized = list(values)
    return stdev(materialized) if len(materialized) > 1 else float("nan")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_result(summary: dict[str, Any]) -> str:
    passport = summary["material_passport"]
    return f"""## Material Passport

- Origin Skill: {passport['origin_skill']}
- Origin Mode: {passport['origin_mode']}
- Origin Date: {passport['origin_date']}
- Verification Status: {passport['verification_status']}
- Version Label: {passport['version_label']}

## S3 Convergence and Capacity Follow-up

- **Status**: {summary['experiment']['status']}
- **Runs**: {summary['design']['run_count']}
- **Larger-Model Verdict**: {summary['larger_model_diagnosis']['verdict']}
- **More-Data Verdict**: {summary['more_data_diagnosis']['verdict']}
- **Original Sealed Test Accessed**: False
- **Boundary**: {summary['interpretation_boundary']}
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
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
