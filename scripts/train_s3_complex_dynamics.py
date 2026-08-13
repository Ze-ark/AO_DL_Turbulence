"""训练S3-B复杂动态GRU，并与按动态类型拟合的同历史岭回归比较。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, stdev
import sys
from time import perf_counter
from typing import Any, Sequence

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
    fit_regime_conditioned_ridge,
    generate_temporal_dynamics_data,
    initialization_seed_gate,
    load_s1_config,
    paired_episode_comparison,
)
from src.simulation.convergence import ConvergenceTracker, convergence_label
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
        default="configs/experiments/s3_complex_dynamics_v1.yaml",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只检查CUDA、种子、配置和模型规模，不生成数据或启动训练。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_path = _project_path(args.config)
    experiment = _load_yaml(experiment_path)
    environment_path = _project_path(experiment["environment_config"])
    environment_config, requested_device = load_s1_config(environment_path)
    device = resolve_device(requested_device)
    dataset_settings = experiment["dataset"]
    training = experiment["training"]
    model_settings = experiment["model"]
    output_settings = experiment["outputs"]
    train_conditions = _conditions(dataset_settings["train_conditions"])
    validation_conditions = _conditions(dataset_settings["validation_conditions"])
    sealed_conditions = _conditions(dataset_settings["sealed_test_conditions"])

    upstream_path = _project_path(experiment["upstream_convergence_summary"])
    upstream = _validate_upstream(upstream_path)
    _validate_condition_design(
        environment_config,
        train_conditions,
        validation_conditions,
        sealed_conditions,
    )
    _validate_protected_seeds(
        experiment,
        train_conditions + validation_conditions + sealed_conditions,
        environment_config.batch_size,
    )
    _validate_training_settings(training)

    model = GRUModalDynamics(
        num_modes=environment_config.num_modes,
        hidden_size=int(model_settings["hidden_size"]),
        num_layers=int(model_settings["num_layers"]),
        dropout=float(model_settings["dropout"]),
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    del model
    if args.preflight_only:
        sample_count = (
            int(dataset_settings["frames_per_episode"])
            - int(dataset_settings["sequence_length"])
            - int(dataset_settings["prediction_horizon_frames"])
            + 2
        )
        progress_message(f"CUDA设备：{torch.cuda.get_device_name(device)}")
        progress_message(
            f"S3-B数据：训练{len(train_conditions) * environment_config.batch_size}回合，"
            f"验证{len(validation_conditions) * environment_config.batch_size}回合；"
            f"每回合{sample_count}个预测样本。"
        )
        progress_message(
            f"GRU隐藏维度{int(model_settings['hidden_size'])}，参数量{parameter_count}；"
            f"共{len(training['initialization_seeds'])}次独立初始化。"
        )
        progress_message(
            "预检完成：原S3封存测试和S3-B封存测试均未读取，正式训练尚未启动。"
        )
        return

    dataset_path = _project_path(output_settings["dataset_h5"])
    checkpoint_dir = _project_path(output_settings["checkpoint_directory"])
    output_dir = _project_path(output_settings["directory"])
    _require_fresh_destination(dataset_path, checkpoint_dir, output_dir)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    progress_message(
        f"S3-B正式训练设备：{torch.cuda.get_device_name(device)}；"
        f"预测提前量：{int(dataset_settings['prediction_horizon_frames'])}帧。"
    )
    progress_message(
        "科学边界：这是纯仿真oracle模态预测，不是RL、闭环控制或真实SLM结果。"
    )

    train_data = _generate_with_progress(
        "生成S3-B训练回合",
        environment_config,
        device,
        train_conditions,
        dataset_settings,
    )
    validation_data = _generate_with_progress(
        "生成S3-B验证回合",
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
        int(dataset_settings["prediction_horizon_frames"]),
    )
    dataset_hash = _sha256(dataset_path)
    progress_message("拟合按动态类型分组的8帧岭回归强基线……")
    ridge = fit_regime_conditioned_ridge(
        train_data,
        validation_data,
        train_conditions,
        validation_conditions,
        float(experiment["baseline"]["ridge_alpha"]),
    )

    run_records: list[dict[str, Any]] = []
    condition_records: list[dict[str, Any]] = []
    run_records_path = output_dir / "run_records.csv"
    condition_records_path = output_dir / "condition_records.csv"
    initialization_seeds = [int(value) for value in training["initialization_seeds"]]
    runs = progress_bar(
        initialization_seeds,
        description="S3-B模型总进度",
        unit="模型",
    )
    for initialization_seed in runs:
        run_record, one_condition_records = _train_one_model(
            initialization_seed=initialization_seed,
            parameter_count=parameter_count,
            train_data=train_data,
            validation_data=validation_data,
            validation_conditions=validation_conditions,
            ridge=ridge,
            model_settings=model_settings,
            training=training,
            device=device,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            experiment_path=experiment_path,
            environment_path=environment_path,
            dataset_hash=dataset_hash,
            upstream_path=upstream_path,
        )
        run_records.append(run_record)
        condition_records.extend(one_condition_records)
        _write_csv(run_records_path, run_records)
        _write_csv(condition_records_path, condition_records)
        update_progress(
            runs,
            device=device,
            metrics={
                "GRU_RMSE": float(run_record["gru_rmse"]),
                "岭回归RMSE": float(run_record["ridge_rmse"]),
                "技能分数": float(run_record["skill_score"]),
            },
        )

    gate = initialization_seed_gate(
        run_records,
        condition_records,
        min_mean_skill_score=float(experiment["gate"]["min_mean_skill_score"]),
        min_ci95_low=float(experiment["gate"]["min_ci95_low"]),
        require_every_run_positive=bool(
            experiment["gate"]["require_every_run_positive"]
        ),
        require_every_condition_positive=bool(
            experiment["gate"]["require_every_condition_positive"]
        ),
    )
    condition_summary = _summarize_condition_records(condition_records)
    condition_summary_path = output_dir / "condition_summary.csv"
    _write_csv(condition_summary_path, condition_summary)
    convergence_counts: dict[str, int] = {}
    for record in run_records:
        label = str(record["convergence_label"])
        convergence_counts[label] = convergence_counts.get(label, 0) + 1

    result = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "s3_complex_dynamics_v1",
        },
        "experiment": {
            "id": "AO-S3-B-COMPLEX-DYNAMICS-GATE",
            "type": "training",
            "status": "completed",
            "command": (
                ".\\.venv\\Scripts\\python.exe scripts\\train_s3_complex_dynamics.py "
                "--config configs\\experiments\\s3_complex_dynamics_v1.yaml"
            ),
            "duration_seconds": perf_counter() - started,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "inputs": {
            "experiment_config": str(experiment_path.relative_to(PROJECT_ROOT)),
            "experiment_config_sha256": _sha256(experiment_path),
            "environment_config": str(environment_path.relative_to(PROJECT_ROOT)),
            "environment_config_sha256": _sha256(environment_path),
            "upstream_convergence_summary": str(upstream_path.relative_to(PROJECT_ROOT)),
            "upstream_convergence_summary_sha256": _sha256(upstream_path),
            "upstream_larger_model_verdict": upstream["larger_model_diagnosis"]["verdict"],
            "dataset_h5": str(dataset_path.relative_to(PROJECT_ROOT)),
            "dataset_sha256": dataset_hash,
            "train_samples": len(train_data.histories),
            "validation_samples": len(validation_data.histories),
            "train_conditions": [asdict(item) for item in train_conditions],
            "validation_conditions": [asdict(item) for item in validation_conditions],
            "prediction_horizon_frames": int(
                dataset_settings["prediction_horizon_frames"]
            ),
            "protected_original_sealed_accessed": False,
            "s3b_sealed_test_accessed": False,
        },
        "design": {
            "dynamic_regimes": sorted({item.regime for item in train_conditions}),
            "model_hidden_size": int(model_settings["hidden_size"]),
            "parameter_count": parameter_count,
            "initialization_seeds": initialization_seeds,
            "run_count": len(initialization_seeds),
            "baseline": str(experiment["baseline"]["kind"]),
            "min_optimizer_steps": int(training["min_optimizer_steps"]),
            "max_optimizer_steps": int(training["max_optimizer_steps"]),
        },
        "run_records": run_records,
        "condition_summary": condition_summary,
        "gate": gate,
        "convergence": convergence_counts,
        "outputs": {
            "checkpoint_directory": str(checkpoint_dir.relative_to(PROJECT_ROOT)),
            "run_records_csv": str(run_records_path.relative_to(PROJECT_ROOT)),
            "condition_records_csv": str(
                condition_records_path.relative_to(PROJECT_ROOT)
            ),
            "condition_summary_csv": str(
                condition_summary_path.relative_to(PROJECT_ROOT)
            ),
        },
        "next_action": (
            "停止在这里并告诉助手“S3-B训练完成”。助手将只读审计哈希、收敛、"
            "三次初始化和各复杂动态条件结果；不要自行打开封存测试或进入RL。"
        ),
        "interpretation_boundary": (
            "本结果只比较纯仿真中使用相同8帧历史的256维GRU和按动态类型拟合的"
            "线性岭回归，输入是带可控噪声的oracle湍流模态，目标是两帧后的无噪模态。"
            "它不能证明RL、真实全息观测、闭环补偿或真实SLM优越。"
        ),
        "anomalies": [],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "experiment_result.md").write_text(
        _markdown_result(result), encoding="utf-8"
    )
    progress_message(f"S3-B训练完成：{summary_path}")
    progress_message(
        f"复杂动态门槛：{gate['validation_gate']}。请停止并告诉助手“S3-B训练完成”。"
    )


def _train_one_model(
    *,
    initialization_seed: int,
    parameter_count: int,
    train_data: TemporalDynamicsData,
    validation_data: TemporalDynamicsData,
    validation_conditions: Sequence[DynamicsCondition],
    ridge: dict[str, Any],
    model_settings: dict[str, Any],
    training: dict[str, Any],
    device: torch.device,
    checkpoint_dir: Path,
    output_dir: Path,
    experiment_path: Path,
    environment_path: Path,
    dataset_hash: str,
    upstream_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _enable_determinism(initialization_seed)
    run_id = f"gru256_seed{initialization_seed}"
    checkpoint_path = checkpoint_dir / f"{run_id}.pt"
    history_path = output_dir / f"{run_id}_loss.csv"
    model = GRUModalDynamics(
        num_modes=train_data.histories.shape[-1],
        hidden_size=int(model_settings["hidden_size"]),
        num_layers=int(model_settings["num_layers"]),
        dropout=float(model_settings["dropout"]),
    ).to(device)
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
    normalization_mean = ridge["normalization_mean"]
    normalization_scale = ridge["normalization_scale"]
    mean_device = normalization_mean.to(device)
    scale_device = normalization_scale.to(device)
    max_steps = int(training["max_optimizer_steps"])
    validation_interval = int(training["validation_interval_steps"])
    best_validation_mse = float("inf")
    best_step = 0
    interval_loss = 0.0
    interval_count = 0
    history: list[dict[str, Any]] = []
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
                        "model_settings": dict(model_settings),
                        "initialization_seed": initialization_seed,
                        "parameter_count": parameter_count,
                        "best_step": best_step,
                        "best_validation_normalized_mse": best_validation_mse,
                        "normalization_mean": normalization_mean,
                        "normalization_scale": normalization_scale,
                        "regime_ridge_models": ridge["models"],
                        "experiment_config_sha256": _sha256(experiment_path),
                        "environment_config_sha256": _sha256(environment_path),
                        "upstream_summary_sha256": _sha256(upstream_path),
                        "dataset_sha256": dataset_hash,
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
                progress_message(f"{run_id}满足早停条件，停止于第{step}步。")
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
        ridge["prediction"],
        validation_data.targets,
        normalization_scale,
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
    run_record = {
        "run_id": run_id,
        "initialization_seed": initialization_seed,
        "parameter_count": parameter_count,
        "best_step": best_step,
        "stop_step": int(history[-1]["step"]),
        "stopped_early": stopped_early,
        "convergence_label": convergence["label"],
        "final_window_relative_improvement": convergence[
            "final_window_relative_improvement"
        ],
        "best_validation_normalized_mse": best_validation_mse,
        "gru_rmse": comparison["gru_normalized_rmse"]["mean"],
        "ridge_rmse": comparison["linear_normalized_rmse"]["mean"],
        "skill_score": comparison["skill_score"]["mean"],
        "episode_skill_ci95_low": comparison["skill_score"]["ci95_low"],
        "episode_skill_ci95_high": comparison["skill_score"]["ci95_high"],
        "duration_seconds": perf_counter() - started,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "loss_history": str(history_path.relative_to(PROJECT_ROOT)),
    }
    condition_records = _condition_rows(
        run_id,
        initialization_seed,
        comparison["by_condition_index"],
        validation_conditions,
    )
    return run_record, condition_records


def _generate_with_progress(
    description: str,
    environment_config: Any,
    device: torch.device,
    conditions: Sequence[DynamicsCondition],
    settings: dict[str, Any],
) -> TemporalDynamicsData:
    total = len(conditions) * int(settings["frames_per_episode"])
    with counted_progress(total=total, description=description, unit="步") as bar:
        return generate_temporal_dynamics_data(
            environment_config,
            device,
            conditions,
            sequence_length=int(settings["sequence_length"]),
            frames_per_episode=int(settings["frames_per_episode"]),
            random_action_std_rad=float(settings["random_action_std_rad"]),
            prediction_horizon_frames=int(settings["prediction_horizon_frames"]),
            progress_callback=lambda completed, _: advance_to(bar, completed),
        )


def _validation_mse(
    model: GRUModalDynamics,
    data: TemporalDynamicsData,
    device: torch.device,
    normalization_mean: torch.Tensor,
    normalization_scale: torch.Tensor,
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
            prediction = model(
                normalize_modal(histories, normalization_mean, normalization_scale)
            )
            normalized_target = normalize_modal(
                targets, normalization_mean, normalization_scale
            )
            total += float(torch.mean((prediction - normalized_target).square())) * len(
                histories
            )
            count += len(histories)
    return total / count


def _predict(
    model: GRUModalDynamics,
    data: TemporalDynamicsData,
    device: torch.device,
    normalization_mean: torch.Tensor,
    normalization_scale: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    loader = DataLoader(TensorDataset(data.histories), batch_size=batch_size, shuffle=False)
    predictions: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for (histories,) in loader:
            normalized = normalize_modal(
                histories.to(device), normalization_mean, normalization_scale
            )
            predictions.append(
                denormalize_modal(
                    model(normalized), normalization_mean, normalization_scale
                ).cpu()
            )
    return torch.cat(predictions)


def _export_dataset(
    path: Path,
    train: TemporalDynamicsData,
    validation: TemporalDynamicsData,
    train_conditions: Sequence[DynamicsCondition],
    validation_conditions: Sequence[DynamicsCondition],
    experiment_path: Path,
    environment_path: Path,
    prediction_horizon_frames: int,
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["stage"] = "S3-B"
        handle.attrs["observation_source"] = (
            "oracle_turbulence_modal_with_condition_specific_gaussian_noise"
        )
        handle.attrs["target_source"] = "future_noise_free_oracle_turbulence_modal"
        handle.attrs["prediction_horizon_frames"] = prediction_horizon_frames
        handle.attrs["original_sealed_test_included"] = False
        handle.attrs["s3b_sealed_test_included"] = False
        handle.attrs["experiment_config_sha256"] = _sha256(experiment_path)
        handle.attrs["environment_config_sha256"] = _sha256(environment_path)
        _write_temporal_group(handle, "train", train, train_conditions)
        _write_temporal_group(
            handle, "validation", validation, validation_conditions
        )


def _write_temporal_group(
    handle: h5py.File,
    name: str,
    data: TemporalDynamicsData,
    conditions: Sequence[DynamicsCondition],
) -> None:
    group = handle.create_group(name)
    group.attrs["conditions_json"] = json.dumps(
        [asdict(condition) for condition in conditions], ensure_ascii=False
    )
    for dataset_name, value in (
        ("history", data.histories),
        ("target", data.targets),
        ("episode_seed", data.episode_seed),
        ("condition_index", data.condition_index),
        ("target_step", data.target_step),
    ):
        group.create_dataset(
            dataset_name,
            data=value.numpy(),
            compression="gzip",
            shuffle=True,
        )


def _condition_rows(
    run_id: str,
    initialization_seed: int,
    by_condition: dict[str, Any],
    conditions: Sequence[DynamicsCondition],
) -> list[dict[str, Any]]:
    rows = []
    for index_text, metrics in sorted(by_condition.items(), key=lambda item: int(item[0])):
        condition = conditions[int(index_text)]
        rows.append(
            {
                "run_id": run_id,
                "initialization_seed": initialization_seed,
                "condition_id": condition.identifier,
                "regime": condition.regime,
                "wind_speed_mps": condition.wind_speed_mps,
                "wind_direction_deg": condition.wind_direction_deg,
                "frozen_flow_rho": condition.frozen_flow_rho,
                "observation_noise_std_rad": condition.observation_noise_std_rad,
                "episodes": int(metrics["episodes"]),
                "gru_rmse": metrics["gru_normalized_rmse"]["mean"],
                "ridge_rmse": metrics["linear_normalized_rmse"]["mean"],
                "skill_score": metrics["skill_score"]["mean"],
                "episode_skill_ci95_low": metrics["skill_score"]["ci95_low"],
                "episode_skill_ci95_high": metrics["skill_score"]["ci95_high"],
            }
        )
    return rows


def _summarize_condition_records(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["condition_id"]), []).append(record)
    rows = []
    for condition_id, values in sorted(grouped.items()):
        skills = [float(value["skill_score"]) for value in values]
        rows.append(
            {
                "condition_id": condition_id,
                "regime": values[0]["regime"],
                "runs": len(values),
                "gru_rmse_mean": mean(float(value["gru_rmse"]) for value in values),
                "gru_rmse_std": _sample_std(
                    float(value["gru_rmse"]) for value in values
                ),
                "ridge_rmse_mean": mean(
                    float(value["ridge_rmse"]) for value in values
                ),
                "skill_score_mean": mean(skills),
                "skill_score_std": _sample_std(skills),
            }
        )
    return rows


def _validate_upstream(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing upstream convergence summary: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary["experiment"]["status"] != "completed":
        raise RuntimeError("upstream convergence/capacity experiment is not complete")
    if summary["inputs"]["protected_original_sealed_accessed"]:
        raise RuntimeError("upstream experiment reports access to the original sealed test")
    diagnosis = summary["larger_model_diagnosis"]
    if diagnosis["verdict"] != "LARGER_MODEL_SUPPORTED":
        raise RuntimeError("the upstream result does not support selecting the 256-wide GRU")
    return summary


def _validate_condition_design(
    base_config: Any,
    train: Sequence[DynamicsCondition],
    validation: Sequence[DynamicsCondition],
    sealed: Sequence[DynamicsCondition],
) -> None:
    if not train or not validation or not sealed:
        raise ValueError("train, validation, and sealed conditions must all be declared")
    all_identifiers = [item.identifier for item in train + validation + sealed]
    if len(all_identifiers) != len(set(all_identifiers)):
        raise ValueError("condition identifiers must be globally unique")
    train_regimes = {item.regime for item in train}
    if {item.regime for item in validation + sealed} - train_regimes:
        raise ValueError("every validation and sealed regime must be represented in training")
    batch_size = int(base_config.batch_size)
    seed_sets = []
    for split_name, conditions in (
        ("train", train),
        ("validation", validation),
        ("sealed", sealed),
    ):
        split_seeds: set[int] = set()
        for condition in conditions:
            candidate = replace(
                base_config,
                wind_speed_mps=condition.wind_speed_mps,
                wind_direction_deg=condition.wind_direction_deg,
                frozen_flow_rho=condition.frozen_flow_rho,
                wind_speed_modulation_fraction=condition.wind_speed_modulation_fraction,
                wind_direction_modulation_deg=condition.wind_direction_modulation_deg,
                wind_modulation_period_frames=condition.wind_modulation_period_frames,
                wind_modulation_phase_deg=condition.wind_modulation_phase_deg,
            )
            candidate.validate()
            if condition.observation_noise_std_rad < 0:
                raise ValueError("observation noise must be non-negative")
            one_condition = set(range(condition.base_seed, condition.base_seed + batch_size))
            if split_seeds & one_condition:
                raise ValueError(f"episode seeds overlap within the {split_name} split")
            split_seeds.update(one_condition)
        seed_sets.append(split_seeds)
    if seed_sets[0] & seed_sets[1] or seed_sets[0] & seed_sets[2] or seed_sets[1] & seed_sets[2]:
        raise ValueError("train, validation, and sealed episode seeds must be disjoint")


def _validate_protected_seeds(
    experiment: dict[str, Any],
    current_conditions: Sequence[DynamicsCondition],
    batch_size: int,
) -> None:
    current_seeds = {
        condition.base_seed + offset
        for condition in current_conditions
        for offset in range(batch_size)
    }
    protected_seeds: set[int] = set()
    for path_text in experiment["protected_experiments"]:
        protected = _load_yaml(_project_path(path_text))
        for key, values in protected["dataset"].items():
            if not key.endswith("conditions") or not isinstance(values, list):
                continue
            for item in values:
                if "base_seed" not in item:
                    continue
                base_seed = int(item["base_seed"])
                protected_seeds.update(range(base_seed, base_seed + batch_size))
    if current_seeds & protected_seeds:
        raise ValueError("S3-B episode seeds overlap a protected earlier experiment")


def _validate_training_settings(training: dict[str, Any]) -> None:
    seeds = [int(value) for value in training["initialization_seeds"]]
    if len(seeds) < 3 or len(seeds) != len(set(seeds)):
        raise ValueError("at least three unique initialization seeds are required")
    minimum = int(training["min_optimizer_steps"])
    maximum = int(training["max_optimizer_steps"])
    interval = int(training["validation_interval_steps"])
    if not 0 < minimum <= maximum or interval <= 0:
        raise ValueError("optimizer step limits and validation interval are invalid")
    if maximum % interval:
        raise ValueError("max_optimizer_steps must be divisible by validation_interval_steps")


def _require_fresh_destination(
    dataset_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
) -> None:
    occupied = []
    if dataset_path.exists():
        occupied.append(dataset_path)
    for directory in (checkpoint_dir, output_dir):
        if directory.exists() and any(directory.iterdir()):
            occupied.append(directory)
    if occupied:
        joined = ", ".join(str(path) for path in occupied)
        raise FileExistsError(
            "S3-B输出位置已有文件；为保护已有训练结果，本脚本不会覆盖：" + joined
        )


def _conditions(values: Sequence[dict[str, Any]]) -> list[DynamicsCondition]:
    return [DynamicsCondition.from_mapping(dict(value)) for value in values]


def _sample_std(values: Any) -> float:
    materialized = list(values)
    return stdev(materialized) if len(materialized) > 1 else float("nan")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_result(result: dict[str, Any]) -> str:
    passport = result["material_passport"]
    gate = result["gate"]
    return f"""## Material Passport

- Origin Skill: {passport['origin_skill']}
- Origin Mode: {passport['origin_mode']}
- Origin Date: {passport['origin_date']}
- Verification Status: {passport['verification_status']}
- Version Label: {passport['version_label']}

## S3-B复杂动态门槛

- **状态**：{result['experiment']['status']}
- **独立初始化次数**：{result['design']['run_count']}
- **验证门槛**：{gate['validation_gate']}
- **平均技能分数**：{gate['run_skill_score']['mean']:.6f}
- **95%区间**：[{gate['run_skill_score']['ci95_low']:.6f}, {gate['run_skill_score']['ci95_high']:.6f}]
- **原S3封存测试是否读取**：否
- **S3-B封存测试是否读取**：否
- **结论边界**：{result['interpretation_boundary']}
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
