"""在复杂动态条件下调优并比较S4-A线性闭环控制器。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Sequence

import h5py
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime import resolve_device
from src.simulation import (
    RobustnessCondition,
    S1EnvConfig,
    TemporalDynamicsData,
    closed_loop_robustness_gate,
    fit_ridge_autoregression,
    load_s1_config,
    make_controller,
    modal_normalization,
)
from src.simulation.evaluation import (
    ControllerRollout,
    export_rollout_h5,
    paired_delta,
    run_controller_rollout,
    summarize_oracle_upper_bound,
    summarize_rollouts,
)
from src.training_progress import counted_progress, gpu_memory_status, progress_message


ControllerFactory = Callable[[], Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/s4_linear_robustness_v1.yaml",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只检查CUDA、上游证据、配置和数据隔离，不写结果。",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="只跑一个条件和最多20步；结果仅用于代码与CUDA冒烟。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preflight_only and args.quick:
        raise SystemExit("--preflight-only and --quick cannot be used together")

    experiment_path = _project_path(args.config)
    experiment = _load_yaml(experiment_path)
    environment_path = _project_path(experiment["environment_config"])
    env_config, requested_device = load_s1_config(environment_path)
    device = resolve_device(requested_device)
    preflight = _preflight(experiment, experiment_path, environment_path, env_config)

    if args.preflight_only:
        document = {
            "stage": "S4-A",
            "status": "READY",
            "writes_performed": False,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            **preflight,
        }
        print(json.dumps(document, ensure_ascii=False, indent=2))
        return

    output_dir = _project_path(experiment["outputs"]["directory"])
    if args.quick:
        output_dir = output_dir / "quick_smoke"
    checkpoint_path = _project_path(experiment["outputs"]["ridge_checkpoint"])
    _refuse_existing_run(output_dir, None if args.quick else checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc)
    ridge = _fit_pooled_ridge(experiment, device)
    if not args.quick:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "stage": "S4-A",
                "kind": "pooled_ridge_autoregression",
                "sequence_length": ridge["sequence_length"],
                "prediction_horizon": ridge["prediction_horizon"],
                "ridge_alpha": ridge["ridge_alpha"],
                "normalization_mean": ridge["normalization_mean"].cpu(),
                "normalization_scale": ridge["normalization_scale"].cpu(),
                "weight": ridge["weight"].cpu(),
                "bias": ridge["bias"].cpu(),
                "experiment_config_sha256": _file_sha256(experiment_path),
                "environment_config_sha256": _file_sha256(environment_path),
                "upstream_dataset_sha256": preflight["upstream"]["dataset_sha256"],
            },
            checkpoint_path,
        )

    tuning_conditions, development_conditions = _active_conditions(experiment, args.quick)
    steps = min(
        int(experiment["evaluation"]["steps"]),
        20 if args.quick else env_config.episode_length,
    )
    tuning_records: list[dict[str, Any]] = []
    leaky_parameters = _tune_controller(
        "leaky_integrator",
        _leaky_candidates(experiment),
        env_config,
        device,
        tuning_conditions,
        steps,
        tuning_records,
        float(experiment["gate"]["max_violation_fraction"]),
    )
    linear_parameters = _tune_controller(
        "linear_predictor",
        _linear_candidates(experiment),
        env_config,
        device,
        tuning_conditions,
        steps,
        tuning_records,
        float(experiment["gate"]["max_violation_fraction"]),
    )

    public_parameters: dict[str, dict[str, Any]] = {
        "no_correction": {},
        "direct_projection": {},
        "leaky_integrator": leaky_parameters,
        "linear_predictor": linear_parameters,
        "ridge_predictor": {
            "sequence_length": ridge["sequence_length"],
            "prediction_horizon": ridge["prediction_horizon"],
            "ridge_alpha": ridge["ridge_alpha"],
            "warmup_velocity_gain": ridge["warmup_velocity_gain"],
            "training_split": "S3-B/train only",
        },
    }
    factories: dict[str, ControllerFactory] = {
        name: _controller_factory(name, parameters, env_config, ridge)
        for name, parameters in public_parameters.items()
    }

    total_steps = len(factories) * len(development_conditions) * steps
    all_rollouts: dict[str, list[ControllerRollout]] = {}
    with counted_progress(
        total=total_steps,
        description="S4-A开发比较",
        unit="步",
    ) as bar:
        for name, factory in factories.items():
            bar.set_postfix_str(f"控制器={name} 显存={gpu_memory_status(device)}")
            all_rollouts[name] = _run_condition_set(
                env_config,
                device,
                development_conditions,
                steps,
                factory,
                include_oracle_upper_bound=name == "no_correction",
                progress_step=bar.update,
            )

    results = _controller_results(all_rollouts, public_parameters)
    condition_records = _condition_records(all_rollouts, development_conditions)
    deployable = [
        item
        for item in results
        if item["controller"] != "no_correction"
        and item["violation_fraction"]["mean"]
        <= float(experiment["gate"]["max_violation_fraction"])
    ]
    if not deployable:
        raise RuntimeError("all deployable controllers exceeded the violation threshold")
    selected = max(deployable, key=lambda item: item["power_in_bucket"]["mean"])
    nonlearned_names = {"direct_projection", "leaky_integrator", "linear_predictor"}
    best_nonlearned = max(
        (item for item in deployable if item["controller"] in nonlearned_names),
        key=lambda item: item["power_in_bucket"]["mean"],
    )
    gate = (
        {"validation_gate": "NOT_EVALUATED", "reason": "quick_smoke_only"}
        if args.quick
        else closed_loop_robustness_gate(
            selected,
            condition_records,
            **experiment["gate"],
        )
    )
    ridge_increment = _ridge_increment(
        all_rollouts["ridge_predictor"],
        all_rollouts[best_nonlearned["controller"]],
        development_conditions,
        best_nonlearned["controller"],
    )
    oracle = summarize_oracle_upper_bound(all_rollouts["no_correction"])

    if bool(experiment["outputs"].get("export_development_trajectories", True)) and not args.quick:
        _export_trajectories(
            output_dir,
            all_rollouts,
            development_conditions,
            env_config,
            experiment_path,
        )

    duration = (datetime.now(timezone.utc) - started).total_seconds()
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": started.isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "s4_linear_robustness_v1",
        },
        "experiment": {
            "id": "AO-S4-A-LINEAR-CLOSED-LOOP-ROBUSTNESS",
            "type": "quick_smoke" if args.quick else "development_algorithm_comparison",
            "status": "quick_smoke_only" if args.quick else "completed_pending_audit",
            "duration_seconds": duration,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "inputs": {
            "experiment_config": _relative(experiment_path),
            "experiment_config_sha256": _file_sha256(experiment_path),
            "environment_config": _relative(environment_path),
            "environment_config_sha256": _file_sha256(environment_path),
            "runner": _relative(Path(__file__).resolve()),
            "runner_sha256": _file_sha256(Path(__file__).resolve()),
            **preflight["upstream"],
        },
        "environment": asdict(env_config),
        "git": _git_state(),
        "design": {
            "split": "quick_smoke" if args.quick else "development_only",
            "steps": steps,
            "episodes_per_condition": env_config.batch_size,
            "tuning_conditions": [asdict(item) for item in tuning_conditions],
            "development_conditions": [asdict(item) for item in development_conditions],
            "sealed_test_conditions_declared_but_not_accessed": preflight["conditions"]["sealed"],
            "s4_sealed_test_accessed": False,
            "observation": "noisy_oracle_residual_modal_plus_true_applied_command",
            "scientific_metrics_use_noise_free_environment_truth": True,
            "common_action_constraints_and_delay": True,
        },
        "ridge_model": {
            **public_parameters["ridge_predictor"],
            "checkpoint": None if args.quick else _relative(checkpoint_path),
            "checkpoint_sha256": None if args.quick else _file_sha256(checkpoint_path),
            "weight_shape": list(ridge["weight"].shape),
        },
        "tuning": tuning_records,
        "controller_results": results,
        "condition_results": condition_records,
        "oracle_modal_upper_bound": oracle,
        "selected_controller_by_development_bucket_power": selected["controller"],
        "best_nonlearned_controller": best_nonlearned["controller"],
        "ridge_increment_vs_best_nonlearned": ridge_increment,
        "s4a_development_gate": gate,
        "interpretation_boundary": (
            "这是带理想模态观测的纯仿真开发比较，不是封存测试、真实全息感知、"
            "真实SLM闭环或强化学习结果。quick结果不得形成科学结论。"
        ),
        "next_action": (
            "通知助手只读审计正式结果；通过审计后再决定是否打开S4封存条件。"
            if not args.quick
            else "quick只验证代码链路；正式比较须由用户在IDE单独运行。"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_tuning_csv(output_dir / "tuning_results.csv", tuning_records)
    _write_controller_csv(output_dir / "controller_summary.csv", results)
    _write_condition_csv(output_dir / "condition_summary.csv", condition_records)
    progress_message(
        "S4-A快速冒烟完成。" if args.quick else "S4-A开发比较完成，请告诉助手：S4-A运行完成。"
    )
    print(json.dumps({
        "summary": _relative(output_dir / "summary.json"),
        "selected_controller": selected["controller"],
        "gate": gate["validation_gate"],
        "ridge_increment_power_mean": ridge_increment["power_in_bucket"]["mean"],
    }, ensure_ascii=False, indent=2))


def _preflight(
    experiment: dict[str, Any],
    experiment_path: Path,
    environment_path: Path,
    env_config: S1EnvConfig,
) -> dict[str, Any]:
    if experiment["metadata"]["stage"] != "S4-A":
        raise ValueError("experiment metadata stage must be S4-A")
    summary_path = _project_path(experiment["upstream_s3b_summary"])
    dataset_path = _project_path(experiment["upstream_s3b_dataset"])
    if not summary_path.exists() or not dataset_path.exists():
        raise FileNotFoundError("S3-B summary and dataset are required")
    upstream = json.loads(summary_path.read_text(encoding="utf-8"))
    if upstream["experiment"]["status"] != "completed":
        raise RuntimeError("S3-B run is not complete")
    if upstream["gate"]["validation_gate"] != "FAIL":
        raise RuntimeError("S4-A route expects the audited S3-B nonlinear gate to be FAIL")
    expected_hash = upstream["inputs"]["dataset_sha256"]
    actual_hash = _file_sha256(dataset_path)
    if actual_hash != expected_hash:
        raise RuntimeError("S3-B dataset hash does not match its summary")

    ridge_config = experiment["ridge_model"]
    with h5py.File(dataset_path, "r") as handle:
        if set(handle.keys()) != {"train", "validation"}:
            raise RuntimeError("S3-B HDF5 must contain only train and validation groups")
        if str(handle.attrs["stage"]) != "S3-B":
            raise RuntimeError("upstream HDF5 stage is not S3-B")
        if bool(handle.attrs["original_sealed_test_included"]) or bool(
            handle.attrs["s3b_sealed_test_included"]
        ):
            raise RuntimeError("upstream HDF5 unexpectedly contains sealed-test data")
        history_shape = tuple(handle["train/history"].shape)
        target_shape = tuple(handle["train/target"].shape)
        if history_shape[1:] != (
            int(ridge_config["sequence_length"]),
            env_config.num_modes,
        ):
            raise RuntimeError("ridge sequence/mode dimensions do not match S3-B train data")
        if target_shape != (history_shape[0], env_config.num_modes):
            raise RuntimeError("S3-B train target shape is invalid")
        horizon = int(handle.attrs["prediction_horizon_frames"])
    if horizon != int(ridge_config["prediction_horizon_frames"]):
        raise RuntimeError("ridge horizon does not match the S3-B dataset")
    if upstream["inputs"]["environment_config_sha256"] != _file_sha256(environment_path):
        raise RuntimeError("current environment configuration drifted from S3-B")

    groups = {
        name: [RobustnessCondition.from_mapping(item) for item in experiment["conditions"][name]]
        for name in ("tuning_conditions", "development_conditions", "sealed_test_conditions")
    }
    smoke = RobustnessCondition.from_mapping(experiment["conditions"]["smoke_condition"])
    all_conditions = [*groups["tuning_conditions"], *groups["development_conditions"], *groups["sealed_test_conditions"], smoke]
    identifiers = [item.identifier for item in all_conditions]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("S4 condition identifiers must be unique")
    for condition in all_conditions:
        condition.environment_config(env_config)
        if condition.slm_delay_frames != horizon:
            raise ValueError("every S4 condition delay must equal the ridge prediction horizon")
    split_seeds = {
        "tuning": _episode_seed_set(groups["tuning_conditions"], env_config.batch_size),
        "development": _episode_seed_set(groups["development_conditions"], env_config.batch_size),
        "sealed": _episode_seed_set(groups["sealed_test_conditions"], env_config.batch_size),
        "smoke": _episode_seed_set([smoke], env_config.batch_size),
    }
    _require_disjoint(split_seeds)
    upstream_seeds = {
        int(item["base_seed"]) + offset
        for split in ("train_conditions", "validation_conditions")
        for item in upstream["inputs"][split]
        for offset in range(env_config.batch_size)
    }
    if upstream_seeds & set().union(*split_seeds.values()):
        raise RuntimeError("S4 episode seeds overlap S3-B train or validation episodes")

    return {
        "upstream": {
            "upstream_s3b_summary": _relative(summary_path),
            "upstream_s3b_summary_sha256": _file_sha256(summary_path),
            "upstream_s3b_dataset": _relative(dataset_path),
            "dataset_sha256": actual_hash,
            "upstream_s3b_gate": upstream["gate"]["validation_gate"],
            "upstream_s3b_sealed_accessed": False,
            "train_samples": history_shape[0],
        },
        "conditions": {
            "tuning": [asdict(item) for item in groups["tuning_conditions"]],
            "development": [asdict(item) for item in groups["development_conditions"]],
            "sealed": [asdict(item) for item in groups["sealed_test_conditions"]],
            "smoke": asdict(smoke),
            "episode_seed_splits_disjoint": True,
        },
        "controller_candidates": {
            "leaky_integrator": len(_leaky_candidates(experiment)),
            "linear_predictor": len(_linear_candidates(experiment)),
            "development_controllers": 5,
        },
        "config_hashes": {
            "experiment": _file_sha256(experiment_path),
            "environment": _file_sha256(environment_path),
        },
    }


def _fit_pooled_ridge(experiment: dict[str, Any], device: torch.device) -> dict[str, Any]:
    dataset_path = _project_path(experiment["upstream_s3b_dataset"])
    with h5py.File(dataset_path, "r") as handle:
        train = TemporalDynamicsData(
            histories=torch.from_numpy(handle["train/history"][:]).to(device),
            targets=torch.from_numpy(handle["train/target"][:]).to(device),
            episode_seed=torch.from_numpy(handle["train/episode_seed"][:]).to(device),
            condition_index=torch.from_numpy(handle["train/condition_index"][:]).to(device),
            target_step=torch.from_numpy(handle["train/target_step"][:]).to(device),
        )
    model_config = experiment["ridge_model"]
    train.validate(int(model_config["sequence_length"]), train.targets.shape[1])
    mean, scale = modal_normalization(train)
    weight, bias = fit_ridge_autoregression(
        train.histories,
        train.targets,
        mean,
        scale,
        float(model_config["ridge_alpha"]),
    )
    return {
        "sequence_length": int(model_config["sequence_length"]),
        "prediction_horizon": int(model_config["prediction_horizon_frames"]),
        "ridge_alpha": float(model_config["ridge_alpha"]),
        "warmup_velocity_gain": float(model_config["warmup_velocity_gain"]),
        "normalization_mean": mean,
        "normalization_scale": scale,
        "weight": weight,
        "bias": bias,
    }


def _active_conditions(
    experiment: dict[str, Any], quick: bool
) -> tuple[list[RobustnessCondition], list[RobustnessCondition]]:
    if quick:
        smoke = RobustnessCondition.from_mapping(experiment["conditions"]["smoke_condition"])
        return [smoke], [smoke]
    return (
        [RobustnessCondition.from_mapping(item) for item in experiment["conditions"]["tuning_conditions"]],
        [RobustnessCondition.from_mapping(item) for item in experiment["conditions"]["development_conditions"]],
    )


def _leaky_candidates(experiment: dict[str, Any]) -> list[dict[str, float]]:
    values = experiment["tuning"]["leaky_integrator"]
    return [
        {"gain": float(gain), "leak": float(leak)}
        for gain in values["gains"]
        for leak in values["leaks"]
    ]


def _linear_candidates(experiment: dict[str, Any]) -> list[dict[str, float | int]]:
    horizon = int(experiment["ridge_model"]["prediction_horizon_frames"])
    return [
        {"prediction_horizon": horizon, "velocity_gain": float(gain)}
        for gain in experiment["tuning"]["linear_predictor"]["velocity_gains"]
    ]


def _tune_controller(
    controller_name: str,
    candidates: Sequence[dict[str, Any]],
    base_config: S1EnvConfig,
    device: torch.device,
    conditions: Sequence[RobustnessCondition],
    steps: int,
    records: list[dict[str, Any]],
    max_violation_fraction: float,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    best_score = -float("inf")
    controller_records: list[dict[str, Any]] = []
    with counted_progress(
        total=len(candidates) * len(conditions) * steps,
        description=f"调参 {controller_name}",
        unit="步",
    ) as bar:
        for parameters in candidates:
            bar.set_postfix_str(
                f"参数={json.dumps(parameters, ensure_ascii=False)} 显存={gpu_memory_status(device)}"
            )
            factory = lambda parameters=parameters: make_controller(
                controller_name,
                base_config.num_modes,
                base_config.modal_limit_rad,
                **parameters,
            )
            rollouts = _run_condition_set(
                base_config,
                device,
                conditions,
                steps,
                factory,
                progress_step=bar.update,
            )
            result = summarize_rollouts(rollouts)
            safe = result["violation_fraction"]["mean"] <= max_violation_fraction
            record = {
                "controller": controller_name,
                "parameters": dict(parameters),
                "tuning_conditions": [item.identifier for item in conditions],
                "episodes": result["episodes"],
                "power_in_bucket": result["power_in_bucket"]["mean"],
                "strehl": result["strehl"]["mean"],
                "phase_rmse": result["phase_rmse"]["mean"],
                "violation_fraction": result["violation_fraction"]["mean"],
                "eligible": safe,
                "selected": False,
                "selection_rule": "",
            }
            records.append(record)
            controller_records.append(record)
            if safe and record["power_in_bucket"] > best_score:
                best_score = float(record["power_in_bucket"])
                best = dict(parameters)
    if best is None:
        fallback = min(
            controller_records,
            key=lambda item: (
                float(item["violation_fraction"]),
                -float(item["power_in_bucket"]),
            ),
        )
        best = dict(fallback["parameters"])
        fallback["selected"] = True
        fallback["selection_rule"] = "no_safe_candidate_choose_lowest_violation"
    else:
        selected = next(
            item
            for item in controller_records
            if item["eligible"] and item["parameters"] == best
        )
        selected["selected"] = True
        selected["selection_rule"] = "highest_bucket_power_among_safe_candidates"
    return best


def _controller_factory(
    name: str,
    public_parameters: dict[str, Any],
    config: S1EnvConfig,
    ridge: dict[str, Any],
) -> ControllerFactory:
    if name == "ridge_predictor":
        internal = {
            "sequence_length": ridge["sequence_length"],
            "prediction_horizon": ridge["prediction_horizon"],
            "normalization_mean": ridge["normalization_mean"],
            "normalization_scale": ridge["normalization_scale"],
            "weight": ridge["weight"],
            "bias": ridge["bias"],
            "warmup_velocity_gain": ridge["warmup_velocity_gain"],
        }
    else:
        internal = public_parameters
    return lambda: make_controller(
        name,
        config.num_modes,
        config.modal_limit_rad,
        **internal,
    )


def _run_condition_set(
    base_config: S1EnvConfig,
    device: torch.device,
    conditions: Sequence[RobustnessCondition],
    steps: int,
    factory: ControllerFactory,
    include_oracle_upper_bound: bool = False,
    progress_step: Callable[[int], Any] | None = None,
) -> list[ControllerRollout]:
    rollouts: list[ControllerRollout] = []
    for condition in conditions:
        progress_message(f"运行条件 {condition.identifier}")
        callback = None
        if progress_step is not None:
            callback = lambda _completed, _total: progress_step(1)
        rollouts.append(
            run_controller_rollout(
                condition.environment_config(base_config),
                device,
                factory(),
                condition.base_seed,
                steps,
                include_oracle_upper_bound=include_oracle_upper_bound,
                observation_noise_std_rad=condition.observation_noise_std_rad,
                progress_callback=callback,
            )
        )
    return rollouts


def _controller_results(
    all_rollouts: dict[str, list[ControllerRollout]],
    parameters: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    reference = all_rollouts["no_correction"]
    results: list[dict[str, Any]] = []
    for name, rollouts in all_rollouts.items():
        result = summarize_rollouts(rollouts)
        result["parameters"] = parameters[name]
        if name != "no_correction":
            result["paired_delta_vs_no_correction"] = {
                metric: paired_delta(rollouts, reference, metric)
                for metric in ("strehl", "power_in_bucket", "phase_rmse")
            }
        results.append(result)
    return results


def _condition_records(
    all_rollouts: dict[str, list[ControllerRollout]],
    conditions: Sequence[RobustnessCondition],
) -> list[dict[str, Any]]:
    reference = all_rollouts["no_correction"]
    records: list[dict[str, Any]] = []
    for name, rollouts in all_rollouts.items():
        for index, (condition, rollout) in enumerate(zip(conditions, rollouts, strict=True)):
            summary = summarize_rollouts([rollout])
            record = {
                "controller": name,
                "condition": condition.identifier,
                "regime": condition.regime,
                "episodes": summary["episodes"],
                "strehl": summary["strehl"]["mean"],
                "power_in_bucket": summary["power_in_bucket"]["mean"],
                "phase_rmse": summary["phase_rmse"]["mean"],
                "action_cost": summary["action_cost"]["mean"],
                "violation_fraction": summary["violation_fraction"]["mean"],
                "action_latency_ms": summary["action_latency_ms"]["mean"],
            }
            if name == "no_correction":
                record.update({
                    "strehl_delta_vs_no_correction": 0.0,
                    "power_delta_vs_no_correction": 0.0,
                    "phase_rmse_delta_vs_no_correction": 0.0,
                })
            else:
                record.update({
                    "strehl_delta_vs_no_correction": paired_delta([rollout], [reference[index]], "strehl")["mean"],
                    "power_delta_vs_no_correction": paired_delta([rollout], [reference[index]], "power_in_bucket")["mean"],
                    "phase_rmse_delta_vs_no_correction": paired_delta([rollout], [reference[index]], "phase_rmse")["mean"],
                })
            records.append(record)
    return records


def _ridge_increment(
    ridge_rollouts: list[ControllerRollout],
    baseline_rollouts: list[ControllerRollout],
    conditions: Sequence[RobustnessCondition],
    baseline_name: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "reference_controller": baseline_name,
        "interpretation": "positive Strehl/power and negative phase RMSE favor ridge",
    }
    for metric in ("strehl", "power_in_bucket", "phase_rmse"):
        result[metric] = paired_delta(ridge_rollouts, baseline_rollouts, metric)
    result["by_condition"] = [
        {
            "condition": condition.identifier,
            "regime": condition.regime,
            "power_mean": paired_delta([ridge], [baseline], "power_in_bucket")["mean"],
            "strehl_mean": paired_delta([ridge], [baseline], "strehl")["mean"],
            "phase_rmse_mean": paired_delta([ridge], [baseline], "phase_rmse")["mean"],
        }
        for condition, ridge, baseline in zip(
            conditions, ridge_rollouts, baseline_rollouts, strict=True
        )
    ]
    return result


def _export_trajectories(
    output_dir: Path,
    all_rollouts: dict[str, list[ControllerRollout]],
    conditions: Sequence[RobustnessCondition],
    base_config: S1EnvConfig,
    experiment_path: Path,
) -> None:
    for name, rollouts in all_rollouts.items():
        for condition, rollout in zip(conditions, rollouts, strict=True):
            export_rollout_h5(
                rollout,
                output_dir / "trajectories" / f"{name}_{condition.identifier}.h5",
                condition.environment_config(base_config),
                metadata={
                    "stage": "S4-A",
                    "condition": condition.identifier,
                    "regime": condition.regime,
                    "observation_kind": "noisy_oracle_modal_features",
                    "observation_noise_std_rad": condition.observation_noise_std_rad,
                    "experiment_config": _relative(experiment_path),
                    "sealed_test": False,
                },
            )


def _episode_seed_set(
    conditions: Sequence[RobustnessCondition], batch_size: int
) -> set[int]:
    return {
        condition.base_seed + offset
        for condition in conditions
        for offset in range(batch_size)
    }


def _require_disjoint(groups: dict[str, set[int]]) -> None:
    names = list(groups)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            if groups[left] & groups[right]:
                raise ValueError(f"episode seed splits overlap: {left} and {right}")


def _refuse_existing_run(output_dir: Path, checkpoint_path: Path | None) -> None:
    if output_dir.exists():
        existing = list(output_dir.iterdir())
        formal_dir_with_only_smoke = checkpoint_path is not None and all(
            item.name == "quick_smoke" for item in existing
        )
        if not formal_dir_with_only_smoke:
            raise FileExistsError(
                f"output directory already contains run artifacts; refusing overwrite: {output_dir}"
            )
    if checkpoint_path is not None and checkpoint_path.exists():
        raise FileExistsError(f"ridge checkpoint already exists; refusing overwrite: {checkpoint_path}")


def _write_tuning_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "controller", "parameters_json", "episodes", "power_in_bucket", "strehl",
            "phase_rmse", "violation_fraction", "eligible", "selected", "selection_rule",
        ])
        for item in records:
            writer.writerow([
                item["controller"],
                json.dumps(item["parameters"], ensure_ascii=False, sort_keys=True),
                item["episodes"], item["power_in_bucket"], item["strehl"],
                item["phase_rmse"], item["violation_fraction"], item["eligible"],
                item["selected"], item["selection_rule"],
            ])


def _write_controller_csv(path: Path, results: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "controller", "episodes", "power_in_bucket", "power_ci95_low", "strehl",
            "strehl_ci95_low", "phase_rmse", "phase_rmse_ci95_high",
            "violation_fraction", "action_cost", "action_latency_ms",
        ])
        for item in results:
            writer.writerow([
                item["controller"], item["episodes"], item["power_in_bucket"]["mean"],
                item["power_in_bucket"]["ci95_low"], item["strehl"]["mean"],
                item["strehl"]["ci95_low"], item["phase_rmse"]["mean"],
                item["phase_rmse"]["ci95_high"], item["violation_fraction"]["mean"],
                item["action_cost"]["mean"], item["action_latency_ms"]["mean"],
            ])


def _write_condition_csv(path: Path, records: list[dict[str, Any]]) -> None:
    columns = list(records[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
        ).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "dirty": None}


if __name__ == "__main__":
    main()
