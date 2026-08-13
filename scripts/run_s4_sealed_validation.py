"""一次性打开S4-B封存条件，验证冻结的泄漏积分器。"""

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
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime import resolve_device
from src.simulation.config import S1EnvConfig, load_s1_config
from src.simulation.controllers import ModalController, make_controller
from src.simulation.evaluation import (
    ControllerRollout,
    export_rollout_h5,
    paired_delta,
    run_controller_rollout,
    summarize_rollouts,
)
from src.simulation.robust_control import (
    RobustnessCondition,
    sealed_closed_loop_gate,
)
from src.training_progress import counted_progress, gpu_memory_status, progress_message


ControllerFactory = Callable[[], ModalController]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/s4_sealed_validation_v1.yaml",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只检查冻结控制器、源码哈希和封存隔离，不生成封存轨迹。",
    )
    parser.add_argument(
        "--acknowledge-sealed-test",
        action="store_true",
        help="明确确认一次性打开预声明的S4-B封存条件。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preflight_only and args.acknowledge_sealed_test:
        raise SystemExit(
            "--preflight-only cannot be combined with --acknowledge-sealed-test"
        )

    experiment_path = _project_path(args.config)
    experiment = _load_yaml(experiment_path)
    environment_path = _project_path(experiment["environment_config"])
    env_config, requested_device = load_s1_config(environment_path)
    device = resolve_device(requested_device)
    preflight = _preflight(experiment, experiment_path, environment_path, env_config)

    if args.preflight_only:
        print(
            json.dumps(
                {
                    "stage": "S4-B",
                    "status": "READY_WITH_SEALED_TEST_STILL_CLOSED",
                    "writes_performed": False,
                    "sealed_test_accessed": False,
                    "device": str(device),
                    "gpu_name": torch.cuda.get_device_name(device),
                    **preflight,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if not args.acknowledge_sealed_test:
        raise SystemExit(
            "S4-B is a one-time sealed evaluation. Re-run with "
            "--acknowledge-sealed-test only after accepting that no retuning is allowed."
        )

    output_dir = _project_path(experiment["outputs"]["directory"])
    if output_dir.exists():
        raise FileExistsError(
            f"sealed output already exists; refusing a second access: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    opening_record = {
        "stage": "S4-B",
        "sealed_test_opened_at": started.isoformat(),
        "acknowledgement_flag": "--acknowledge-sealed-test",
        "experiment_config": _relative(experiment_path),
        "experiment_config_sha256": _file_sha256(experiment_path),
        "frozen_controller": experiment["frozen_controller"],
        "sealed_condition_ids": experiment["evaluation"]["sealed_condition_ids"],
        "note": "This marker is written before the first sealed trajectory and blocks automatic reruns.",
    }
    (output_dir / "SEALED_TEST_OPENED.json").write_text(
        json.dumps(opening_record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    conditions = [
        RobustnessCondition.from_mapping(item)
        for item in preflight["sealed_conditions"]
    ]
    steps = int(experiment["evaluation"]["steps"])
    frozen = experiment["frozen_controller"]
    parameters = dict(frozen["parameters"])
    factories: dict[str, ControllerFactory] = {
        "no_correction": lambda: make_controller(
            "no_correction", env_config.num_modes, env_config.modal_limit_rad
        ),
        str(frozen["name"]): lambda: make_controller(
            str(frozen["name"]),
            env_config.num_modes,
            env_config.modal_limit_rad,
            **parameters,
        ),
    }

    all_rollouts: dict[str, list[ControllerRollout]] = {}
    total_steps = len(factories) * len(conditions) * steps
    with counted_progress(total=total_steps, description="S4-B封存验证", unit="步") as bar:
        for controller_name, factory in factories.items():
            rollouts: list[ControllerRollout] = []
            for condition in conditions:
                bar.set_postfix_str(
                    f"控制器={controller_name} 条件={condition.identifier} "
                    f"显存={gpu_memory_status(device)}"
                )
                rollout = run_controller_rollout(
                    condition.environment_config(env_config),
                    device,
                    factory(),
                    condition.base_seed,
                    steps,
                    observation_noise_std_rad=condition.observation_noise_std_rad,
                    progress_callback=lambda _completed, _total: bar.update(1),
                )
                rollouts.append(rollout)
                if bool(experiment["outputs"].get("export_trajectories", True)):
                    export_rollout_h5(
                        rollout,
                        output_dir
                        / "trajectories"
                        / f"{controller_name}_{condition.identifier}.h5",
                        condition.environment_config(env_config),
                        metadata={
                            "stage": "S4-B",
                            "condition": condition.identifier,
                            "regime": condition.regime,
                            "observation_kind": experiment["evaluation"][
                                "observation_kind"
                            ],
                            "observation_noise_std_rad": (
                                condition.observation_noise_std_rad
                            ),
                            "experiment_config": _relative(experiment_path),
                            "sealed_test": True,
                        },
                    )
            all_rollouts[controller_name] = rollouts

    reference_name = str(experiment["evaluation"]["reference_controller"])
    candidate_name = str(frozen["name"])
    reference_summary = summarize_rollouts(all_rollouts[reference_name])
    reference_summary["parameters"] = {}
    candidate_summary = summarize_rollouts(all_rollouts[candidate_name])
    candidate_summary["parameters"] = parameters
    candidate_summary["paired_delta_vs_no_correction"] = {
        metric: paired_delta(
            all_rollouts[candidate_name], all_rollouts[reference_name], metric
        )
        for metric in ("strehl", "power_in_bucket", "phase_rmse")
    }
    condition_records = _condition_records(
        all_rollouts[candidate_name],
        all_rollouts[reference_name],
        conditions,
    )
    gate = sealed_closed_loop_gate(
        candidate_summary,
        reference_summary,
        condition_records,
        **experiment["gate"],
    )
    source_manifest = _actual_source_manifest(experiment)
    duration = (datetime.now(timezone.utc) - started).total_seconds()
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": started.isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "s4_sealed_validation_v1",
        },
        "experiment": {
            "id": "AO-S4-B-ONE-TIME-SEALED-VALIDATION",
            "type": "sealed_algorithm_validation",
            "status": "completed_pending_audit",
            "duration_seconds": duration,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "command": (
                ".\\.venv\\Scripts\\python.exe "
                "scripts\\run_s4_sealed_validation.py "
                "--acknowledge-sealed-test"
            ),
        },
        "inputs": {
            "experiment_config": _relative(experiment_path),
            "experiment_config_sha256": _file_sha256(experiment_path),
            "environment_config": _relative(environment_path),
            "environment_config_sha256": _file_sha256(environment_path),
            "upstream_s4a_summary": experiment["upstream_s4a_summary"],
            "upstream_s4a_summary_sha256": _file_sha256(
                _project_path(experiment["upstream_s4a_summary"])
            ),
            "upstream_s4a_config": experiment["upstream_s4a_config"],
            "upstream_s4a_config_sha256": _file_sha256(
                _project_path(experiment["upstream_s4a_config"])
            ),
            "source_manifest": source_manifest,
            "runtime_versions": _runtime_versions(),
            "git": _git_state(),
        },
        "design": {
            "split": "one_time_sealed_test",
            "sealed_test_accessed": True,
            "retuning_allowed": False,
            "controller_selection_allowed": False,
            "primary_metric": experiment["evaluation"]["primary_metric"],
            "steps": steps,
            "episodes_per_condition": env_config.batch_size,
            "sealed_conditions": [asdict(item) for item in conditions],
            "observation": "noisy_oracle_residual_modal_plus_true_applied_command",
            "scientific_metrics_use_noise_free_environment_truth": True,
        },
        "frozen_controller": frozen,
        "controller_results": [reference_summary, candidate_summary],
        "condition_results": condition_records,
        "s4b_sealed_gate": gate,
        "interpretation_boundary": (
            "这是一次性纯仿真封存验证，只能支持冻结泄漏积分器在声明条件下的"
            "仿真结论；不是强化学习、真实全息感知、真实SLM闭环或自然大气结果。"
        ),
        "next_action": (
            "停止并通知助手“S4-B运行完成”；只读审计后再决定是否进入S4-C硬件化仿真。"
        ),
    }
    (output_dir / "source_manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_controller_csv(
        output_dir / "controller_summary.csv",
        [reference_summary, candidate_summary],
    )
    _write_condition_csv(output_dir / "condition_summary.csv", condition_records)
    progress_message("S4-B一次性封存验证完成，请停止并告诉助手：S4-B运行完成。")
    print(
        json.dumps(
            {
                "summary": _relative(output_dir / "summary.json"),
                "gate": gate["validation_gate"],
                "mean_relative_power_gain": gate["mean_relative_power_gain"],
                "sealed_test_accessed": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _preflight(
    experiment: dict[str, Any],
    experiment_path: Path,
    environment_path: Path,
    env_config: S1EnvConfig,
) -> dict[str, Any]:
    if experiment["metadata"]["stage"] != "S4-B":
        raise ValueError("experiment metadata stage must be S4-B")
    output_dir = _project_path(experiment["outputs"]["directory"])
    if output_dir.exists():
        raise FileExistsError(
            f"sealed output already exists; preflight refuses further access: {output_dir}"
        )

    s4a_summary_path = _project_path(experiment["upstream_s4a_summary"])
    s4a_config_path = _project_path(experiment["upstream_s4a_config"])
    if not s4a_summary_path.exists() or not s4a_config_path.exists():
        raise FileNotFoundError("S4-A summary and configuration are required")
    s4a_summary = json.loads(s4a_summary_path.read_text(encoding="utf-8"))
    s4a_config = _load_yaml(s4a_config_path)
    if s4a_summary["experiment"]["status"] != "completed_pending_audit":
        raise RuntimeError("S4-A formal run is not complete")
    if s4a_summary["s4a_development_gate"]["validation_gate"] != "PASS":
        raise RuntimeError("S4-A development gate did not pass")
    if s4a_summary["design"]["s4_sealed_test_accessed"]:
        raise RuntimeError("S4-A summary reports that sealed conditions were already accessed")

    _verify_upstream_hashes(s4a_summary)
    if _file_sha256(s4a_config_path) != s4a_summary["inputs"][
        "experiment_config_sha256"
    ]:
        raise RuntimeError("S4-A configuration changed after the development run")
    if _file_sha256(environment_path) != s4a_summary["inputs"][
        "environment_config_sha256"
    ]:
        raise RuntimeError("environment configuration changed after S4-A")

    frozen = experiment["frozen_controller"]
    if s4a_summary["selected_controller_by_development_bucket_power"] != frozen[
        "name"
    ]:
        raise RuntimeError("S4-B controller is not the S4-A selected controller")
    selected_result = next(
        item
        for item in s4a_summary["controller_results"]
        if item["controller"] == frozen["name"]
    )
    if selected_result["parameters"] != frozen["parameters"]:
        raise RuntimeError("S4-B controller parameters differ from the S4-A result")

    condition_values = s4a_config["conditions"]["sealed_test_conditions"]
    conditions = [RobustnessCondition.from_mapping(item) for item in condition_values]
    declared_ids = [str(item) for item in experiment["evaluation"]["sealed_condition_ids"]]
    if [item.identifier for item in conditions] != declared_ids:
        raise RuntimeError("S4-B condition order or identity differs from S4-A declaration")
    if len(conditions) != 6:
        raise RuntimeError("S4-B requires exactly six predeclared sealed conditions")
    for condition in conditions:
        condition.environment_config(env_config)
        if condition.slm_delay_frames != int(
            experiment["evaluation"]["common_slm_delay_frames"]
        ):
            raise RuntimeError("sealed condition delay differs from the frozen delay")
    _verify_seed_isolation(s4a_config, env_config.batch_size)

    actual_manifest = _actual_source_manifest(experiment)
    expected_manifest = {
        str(key): str(value) for key, value in experiment["source_manifest"].items()
    }
    if actual_manifest != expected_manifest:
        differences = {
            path: {"expected": expected_manifest.get(path), "actual": digest}
            for path, digest in actual_manifest.items()
            if expected_manifest.get(path) != digest
        }
        raise RuntimeError(f"frozen source manifest mismatch: {differences}")

    return {
        "upstream_s4a": {
            "summary": _relative(s4a_summary_path),
            "summary_sha256": _file_sha256(s4a_summary_path),
            "development_gate": "PASS",
            "selected_controller": frozen,
        },
        "sealed_conditions": condition_values,
        "sealed_condition_count": len(conditions),
        "episodes": len(conditions) * env_config.batch_size,
        "steps_per_episode": int(experiment["evaluation"]["steps"]),
        "source_manifest_verified": True,
        "source_manifest": actual_manifest,
        "episode_seed_splits_disjoint": True,
        "formal_command": (
            ".\\.venv\\Scripts\\python.exe "
            "scripts\\run_s4_sealed_validation.py --acknowledge-sealed-test"
        ),
        "experiment_config_sha256": _file_sha256(experiment_path),
    }


def _verify_upstream_hashes(summary: dict[str, Any]) -> None:
    inputs = summary["inputs"]
    checks = {
        "experiment_config": "experiment_config_sha256",
        "environment_config": "environment_config_sha256",
        "runner": "runner_sha256",
        "upstream_s3b_summary": "upstream_s3b_summary_sha256",
        "upstream_s3b_dataset": "dataset_sha256",
    }
    for path_key, hash_key in checks.items():
        path = _project_path(inputs[path_key])
        if not path.exists() or _file_sha256(path) != inputs[hash_key]:
            raise RuntimeError(f"S4-A upstream hash mismatch: {path_key}")
    checkpoint = _project_path(summary["ridge_model"]["checkpoint"])
    if _file_sha256(checkpoint) != summary["ridge_model"]["checkpoint_sha256"]:
        raise RuntimeError("S4-A ridge checkpoint hash mismatch")


def _verify_seed_isolation(s4a_config: dict[str, Any], batch_size: int) -> None:
    groups: dict[str, set[int]] = {}
    for name in (
        "tuning_conditions",
        "development_conditions",
        "sealed_test_conditions",
    ):
        groups[name] = {
            int(item["base_seed"]) + offset
            for item in s4a_config["conditions"][name]
            for offset in range(batch_size)
        }
    smoke = s4a_config["conditions"]["smoke_condition"]
    groups["smoke_condition"] = {
        int(smoke["base_seed"]) + offset for offset in range(batch_size)
    }
    names = list(groups)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            if groups[left] & groups[right]:
                raise RuntimeError(f"episode seeds overlap: {left} and {right}")


def _condition_records(
    candidate_rollouts: Sequence[ControllerRollout],
    reference_rollouts: Sequence[ControllerRollout],
    conditions: Sequence[RobustnessCondition],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for condition, candidate, reference in zip(
        conditions, candidate_rollouts, reference_rollouts, strict=True
    ):
        candidate_summary = summarize_rollouts([candidate])
        reference_summary = summarize_rollouts([reference])
        records.append(
            {
                "controller": candidate.controller_name,
                "condition": condition.identifier,
                "regime": condition.regime,
                "episodes": candidate_summary["episodes"],
                "reference_power_in_bucket": reference_summary["power_in_bucket"][
                    "mean"
                ],
                "power_in_bucket": candidate_summary["power_in_bucket"]["mean"],
                "strehl": candidate_summary["strehl"]["mean"],
                "phase_rmse": candidate_summary["phase_rmse"]["mean"],
                "action_cost": candidate_summary["action_cost"]["mean"],
                "violation_fraction": candidate_summary["violation_fraction"]["mean"],
                "action_latency_ms": candidate_summary["action_latency_ms"]["mean"],
                "power_delta_vs_no_correction": paired_delta(
                    [candidate], [reference], "power_in_bucket"
                )["mean"],
                "strehl_delta_vs_no_correction": paired_delta(
                    [candidate], [reference], "strehl"
                )["mean"],
                "phase_rmse_delta_vs_no_correction": paired_delta(
                    [candidate], [reference], "phase_rmse"
                )["mean"],
            }
        )
    return records


def _actual_source_manifest(experiment: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative_path in experiment["source_manifest"]:
        path = _project_path(relative_path)
        if not path.exists():
            raise FileNotFoundError(f"frozen source file is missing: {relative_path}")
        result[str(relative_path)] = _file_sha256(path)
    return result


def _runtime_versions() -> dict[str, Any]:
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "h5py": h5py.__version__,
    }


def _write_controller_csv(path: Path, results: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "controller",
                "episodes",
                "power_in_bucket",
                "power_ci95_low",
                "strehl",
                "phase_rmse",
                "violation_fraction",
                "action_cost",
                "action_latency_ms",
            ]
        )
        for item in results:
            writer.writerow(
                [
                    item["controller"],
                    item["episodes"],
                    item["power_in_bucket"]["mean"],
                    item["power_in_bucket"]["ci95_low"],
                    item["strehl"]["mean"],
                    item["phase_rmse"]["mean"],
                    item["violation_fraction"]["mean"],
                    item["action_cost"]["mean"],
                    item["action_latency_ms"]["mean"],
                ]
            )


def _write_condition_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
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
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                text=True,
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "dirty": None}


if __name__ == "__main__":
    main()
