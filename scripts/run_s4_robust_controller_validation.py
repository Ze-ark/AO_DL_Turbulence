"""一次性运行S4-D1鲁棒传统控制器未见种子验证。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_s4_robust_controller_development import (
    _actual_source_manifest,
    _analyze,
    _append_progress,
    _controller_definition,
    _controller_factories,
    _file_sha256,
    _git_state,
    _load_yaml,
    _physical_conditions,
    _profiles,
    _project_path,
    _relative,
    _runtime_versions,
    _trajectory_manifest,
    _write_csv,
)
from src.runtime import resolve_device
from src.simulation.config import S1EnvConfig, load_s1_config
from src.simulation.controller_selection import validate_frozen_robust_controller
from src.simulation.evaluation import (
    ControllerRollout,
    export_rollout_h5,
    paired_delta,
    run_controller_rollout,
)
from src.simulation.hardware_effects import HardwareProfile
from src.simulation.robust_control import RobustnessCondition
from src.training_progress import counted_progress, gpu_memory_status, progress_message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/s4_robust_controller_validation_v1.yaml",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只核对冻结参数、未见种子、源码哈希和输出目录，不生成轨迹。",
    )
    parser.add_argument(
        "--acknowledge-unseen-validation",
        action="store_true",
        help="明确确认只打开一次S4-D1未见验证，结果不得用于重新调参。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preflight_only and args.acknowledge_unseen_validation:
        raise SystemExit(
            "--preflight-only cannot be combined with "
            "--acknowledge-unseen-validation"
        )

    experiment_path = _project_path(args.config)
    experiment = _load_yaml(experiment_path)
    environment_path = _project_path(experiment["environment_config"])
    env_config, requested_device = load_s1_config(environment_path)
    device = resolve_device(requested_device)
    profiles = _profiles(experiment)
    preflight = _preflight(
        experiment,
        experiment_path,
        environment_path,
        env_config,
        profiles,
    )

    if args.preflight_only:
        print(
            json.dumps(
                {
                    "stage": "S4-D1",
                    "status": "READY_WITH_UNSEEN_VALIDATION_STILL_CLOSED",
                    "writes_performed": False,
                    "unseen_validation_accessed": False,
                    "device": str(device),
                    "gpu_name": torch.cuda.get_device_name(device),
                    **preflight,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if not args.acknowledge_unseen_validation:
        raise SystemExit(
            "S4-D1 is a one-time unseen-seed validation. Re-run with "
            "--acknowledge-unseen-validation only after accepting that "
            "controller selection and retuning are forbidden."
        )
    _run_validation(
        experiment,
        experiment_path,
        environment_path,
        env_config,
        device,
        profiles,
    )


def _run_validation(
    experiment: dict[str, Any],
    experiment_path: Path,
    environment_path: Path,
    env_config: S1EnvConfig,
    device: torch.device,
    profiles: Sequence[HardwareProfile],
) -> None:
    output_dir = _project_path(experiment["outputs"]["directory"])
    if output_dir.exists():
        raise FileExistsError(
            f"unseen validation output already exists; refusing reuse: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    (output_dir / "UNSEEN_VALIDATION_OPENED.json").write_text(
        json.dumps(
            {
                "stage": "S4-D1",
                "opened_at": started.isoformat(),
                "acknowledgement_flag": "--acknowledge-unseen-validation",
                "experiment_config": _relative(experiment_path),
                "experiment_config_sha256": _file_sha256(experiment_path),
                "frozen_controller": experiment["evaluation"]["frozen_controller"],
                "note": (
                    "Marker written before the first unseen trajectory; "
                    "do not delete, retry, reselect, or retune after access."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    physical_conditions = _physical_conditions(experiment)
    rollouts = _execute_rollouts(
        experiment,
        env_config,
        device,
        physical_conditions,
        profiles,
        int(experiment["evaluation"]["steps"]),
        output_dir,
    )
    controller_results, condition_results = _analyze(
        experiment,
        rollouts,
        profiles,
        physical_conditions,
    )
    _attach_incremental_comparisons(experiment, rollouts, controller_results, profiles)
    validation = validate_frozen_robust_controller(
        controller_results,
        frozen_controller_id=str(experiment["evaluation"]["frozen_controller"]),
        incumbent_id=str(experiment["evaluation"]["incumbent_controller"]),
        required_profile_ids=experiment["required_profile_ids"],
        **experiment["validation"],
    )

    source_manifest = _actual_source_manifest(experiment)
    trajectory_manifest = _trajectory_manifest(output_dir / "trajectories")
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": started.isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "s4_robust_controller_validation_v1",
        },
        "experiment": {
            "id": "AO-S4-D1-ONE-TIME-UNSEEN-ROBUST-VALIDATION",
            "type": "software_only_frozen_controller_validation",
            "status": "completed_pending_audit",
            "duration_seconds": (
                datetime.now(timezone.utc) - started
            ).total_seconds(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "command": (
                ".\\.venv\\Scripts\\python.exe "
                "scripts\\run_s4_robust_controller_validation.py "
                "--acknowledge-unseen-validation"
            ),
        },
        "inputs": {
            "experiment_config": _relative(experiment_path),
            "experiment_config_sha256": _file_sha256(experiment_path),
            "environment_config": _relative(environment_path),
            "environment_config_sha256": _file_sha256(environment_path),
            "upstream_s4d0_summary": experiment["upstream_s4d0_summary"],
            "upstream_s4d0_summary_sha256": _file_sha256(
                _project_path(experiment["upstream_s4d0_summary"])
            ),
            "upstream_s4d0_audit_record": experiment[
                "upstream_s4d0_audit_record"
            ],
            "upstream_s4d0_audit_record_sha256": _file_sha256(
                _project_path(experiment["upstream_s4d0_audit_record"])
            ),
            "source_manifest": source_manifest,
            "trajectory_manifest": trajectory_manifest,
            "runtime_versions": _runtime_versions(),
            "git": _git_state(),
        },
        "design": {
            "kind": "one_time_unseen_seed_validation_software_only",
            "unseen_validation_accessed": True,
            "retuning_allowed": False,
            "controller_selection_allowed": False,
            "primary_metric": experiment["evaluation"]["primary_metric"],
            "steps": int(experiment["evaluation"]["steps"]),
            "episodes_per_physical_condition": env_config.batch_size,
            "physical_conditions": [asdict(item) for item in physical_conditions],
            "hardware_profiles": [item.as_record() for item in profiles],
            "controllers": experiment["controllers"],
            "scientific_metrics_use_noise_free_environment_truth": True,
            "hardware_parameters_are_assumptions_not_h1_measurements": True,
        },
        "controller_profile_results": controller_results,
        "condition_results": condition_results,
        "s4d1_unseen_validation_gate": validation,
        "interpretation_boundary": (
            "S4-D1只验证冻结传统控制器在新种子和新物理参数下的纯仿真"
            "鲁棒性；不是RL、FSLM-2K73-P04实测性能或真实闭环。"
        ),
        "next_action": (
            "停止并通知助手“S4-D1运行完成”。只读审计前不得建立或启动"
            "残差SAC；FAIL也不得用本批未见结果重新调参或重跑。"
        ),
    }
    (output_dir / "source_manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "trajectory_manifest.json").write_text(
        json.dumps(trajectory_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(
        output_dir / "controller_profile_summary.csv",
        _profile_csv_rows(controller_results),
    )
    _write_csv(output_dir / "condition_summary.csv", condition_results)
    progress_message("S4-D1未见验证完成，请停止并告诉助手：S4-D1运行完成。")
    print(
        json.dumps(
            {
                "summary": _relative(output_dir / "summary.json"),
                "gate": validation["validation_gate"],
                "frozen_controller": validation["frozen_controller"],
                "unseen_validation_accessed": True,
                "formal_results_generated": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _execute_rollouts(
    experiment: dict[str, Any],
    base_config: S1EnvConfig,
    device: torch.device,
    physical_conditions: Sequence[RobustnessCondition],
    profiles: Sequence[HardwareProfile],
    steps: int,
    output_dir: Path,
) -> dict[str, dict[str, list[ControllerRollout]]]:
    factories = _controller_factories(experiment, base_config)
    results = {
        name: {profile.identifier: [] for profile in profiles}
        for name in factories
    }
    total_steps = len(factories) * len(profiles) * len(physical_conditions) * steps
    with counted_progress(
        total=total_steps, description="S4-D1未见种子验证", unit="步"
    ) as bar:
        for controller_name, factory in factories.items():
            definition = _controller_definition(experiment, controller_name)
            for profile in profiles:
                for physical in physical_conditions:
                    bar.set_postfix_str(
                        f"控制器={controller_name} 档位={profile.identifier} "
                        f"物理={physical.identifier} 显存={gpu_memory_status(device)}"
                    )
                    configured = profile.environment_config(
                        physical.environment_config(base_config)
                    )
                    rollout = run_controller_rollout(
                        configured,
                        device,
                        factory(),
                        seed=physical.base_seed,
                        steps=steps,
                        observation_noise_std_rad=profile.observation_noise_std_rad,
                        hardware_effects=profile.effects_config(),
                        progress_callback=lambda _completed, _total: bar.update(1),
                    )
                    results[controller_name][profile.identifier].append(rollout)
                    path = output_dir / "trajectories" / (
                        f"{controller_name}_{profile.identifier}_{physical.identifier}.h5"
                    )
                    export_rollout_h5(
                        rollout,
                        path,
                        configured,
                        metadata={
                            "stage": "S4-D1",
                            "physical_condition": physical.identifier,
                            "regime": physical.regime,
                            "hardware_profile": profile.identifier,
                            "hardware_profile_json": json.dumps(
                                profile.as_record(), ensure_ascii=False
                            ),
                            "controller_definition_json": json.dumps(
                                definition, ensure_ascii=False
                            ),
                            "observation_kind": experiment["evaluation"][
                                "observation_kind"
                            ],
                            "observation_noise_std_rad": (
                                profile.observation_noise_std_rad
                            ),
                            "unseen_validation": True,
                            "retuning_allowed": False,
                            "assumed_hardware_parameters_not_h1_measurements": True,
                        },
                    )
                    _append_progress(
                        output_dir / "progress.jsonl",
                        controller_name,
                        profile,
                        physical,
                        path,
                    )
    return results


def _attach_incremental_comparisons(
    experiment: dict[str, Any],
    rollouts: dict[str, dict[str, list[ControllerRollout]]],
    controller_results: list[dict[str, Any]],
    profiles: Sequence[HardwareProfile],
) -> None:
    frozen = str(experiment["evaluation"]["frozen_controller"])
    incumbent = str(experiment["evaluation"]["incumbent_controller"])
    by_key = {
        (str(item["controller"]), str(item["profile"])): item
        for item in controller_results
    }
    for profile in profiles:
        result = by_key[(frozen, profile.identifier)]
        result["paired_delta_vs_incumbent"] = {
            metric: paired_delta(
                rollouts[frozen][profile.identifier],
                rollouts[incumbent][profile.identifier],
                metric,
            )
            for metric in ("strehl", "power_in_bucket", "phase_rmse")
        }


def _preflight(
    experiment: dict[str, Any],
    experiment_path: Path,
    environment_path: Path,
    env_config: S1EnvConfig,
    profiles: Sequence[HardwareProfile],
) -> dict[str, Any]:
    if experiment["metadata"]["stage"] != "S4-D1":
        raise ValueError("experiment metadata stage must be S4-D1")
    if bool(experiment["evaluation"].get("controller_selection_allowed", True)):
        raise RuntimeError("S4-D1 must disable controller selection")
    if bool(experiment["evaluation"].get("retuning_allowed", True)):
        raise RuntimeError("S4-D1 must disable retuning")
    output_dir = _project_path(experiment["outputs"]["directory"])
    if output_dir.exists():
        raise FileExistsError(
            f"unseen validation output already exists: {output_dir}"
        )
    if _file_sha256(environment_path) != str(
        experiment["environment_config_sha256"]
    ):
        raise RuntimeError("environment config hash mismatch")
    profile_source = _project_path(experiment["hardware_profile_source"])
    if _file_sha256(profile_source) != str(
        experiment["hardware_profile_source_sha256"]
    ):
        raise RuntimeError("hardware profile source hash mismatch")

    upstream, upstream_integrity = _verify_upstream_d0(experiment)
    evaluation = experiment["evaluation"]
    frozen = str(evaluation["frozen_controller"])
    incumbent = str(evaluation["incumbent_controller"])
    reference = str(evaluation["reference_controller"])
    selection = upstream["selection"]
    if selection["validation_gate"] != "PASS":
        raise RuntimeError("S4-D0 selection gate did not pass")
    if selection["selected_controller"] != frozen:
        raise RuntimeError("S4-D1 frozen controller differs from S4-D0 selection")

    definitions = {str(item["id"]): item for item in experiment["controllers"]}
    if set(definitions) != {reference, incumbent, frozen}:
        raise RuntimeError("S4-D1 must contain only reference, incumbent, and frozen controller")
    upstream_definition = next(
        item for item in upstream["design"]["controllers"] if item["id"] == frozen
    )
    if definitions[frozen]["parameters"] != upstream_definition["parameters"]:
        raise RuntimeError("frozen controller parameters changed after S4-D0")
    if definitions[frozen]["kind"] != upstream_definition["kind"]:
        raise RuntimeError("frozen controller kind changed after S4-D0")

    required_ids = list(map(str, experiment["required_profile_ids"]))
    if required_ids != list(map(str, selection["required_profile_ids"])):
        raise RuntimeError("required profile order differs from S4-D0")
    profile_ids = [item.identifier for item in profiles]
    upstream_profile_ids = [
        str(item["identifier"]) for item in upstream["design"]["hardware_profiles"]
    ]
    if profile_ids != upstream_profile_ids:
        raise RuntimeError("hardware profile order differs from S4-D0")

    frozen_profile_results = [
        item
        for item in upstream["controller_profile_results"]
        if item["controller"] == frozen
    ]
    upstream_thresholds = {
        json.dumps(item["profile_gate"]["thresholds"], sort_keys=True)
        for item in frozen_profile_results
    }
    if len(upstream_thresholds) != 1:
        raise RuntimeError("S4-D0 profile thresholds are inconsistent")
    if json.loads(next(iter(upstream_thresholds))) != experiment["profile_gate"]:
        raise RuntimeError("S4-D1 profile gate differs from S4-D0")

    conditions = _physical_conditions(experiment)
    if env_config.batch_size != int(evaluation["episodes_per_physical_condition"]):
        raise ValueError("environment batch size does not match evaluation episodes")
    new_seeds = {
        condition.base_seed + offset
        for condition in conditions
        for offset in range(env_config.batch_size)
    }
    expected_seed_count = len(conditions) * env_config.batch_size
    if len(new_seeds) != expected_seed_count:
        raise RuntimeError("S4-D1 physical seed ranges overlap each other")
    upstream_seed_sets = {
        int(seed)
        for item in upstream["controller_profile_results"]
        for seed in item["episode_seeds"]
    }
    if new_seeds & upstream_seed_sets:
        raise RuntimeError("S4-D1 seeds overlap S4-D0 development episodes")
    s4c_path = _project_path(upstream["inputs"]["upstream_s4c_summary"])
    if _file_sha256(s4c_path) != upstream["inputs"]["upstream_s4c_summary_sha256"]:
        raise RuntimeError("S4-C0 summary changed after S4-D0")
    s4c = json.loads(s4c_path.read_text(encoding="utf-8"))
    s4c_seeds = {
        int(seed)
        for item in s4c["profile_results"]
        for seed in item["episode_seeds"]
    }
    if new_seeds & s4c_seeds:
        raise RuntimeError("S4-D1 seeds overlap S4-C0 episodes")

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
        "upstream_s4d0_gate": "PASS",
        "frozen_controller": upstream_definition,
        "controller_count": len(definitions),
        "physical_condition_count": len(conditions),
        "hardware_profile_count": len(profiles),
        "required_profile_count": len(required_ids),
        "episodes_per_controller_profile": expected_seed_count,
        "unseen_cuda_steps": (
            len(definitions)
            * len(profiles)
            * len(conditions)
            * int(evaluation["steps"])
        ),
        "upstream_integrity": upstream_integrity,
        "source_manifest_verified": True,
        "seed_isolation_verified": True,
        "controller_selection_allowed": False,
        "retuning_allowed": False,
        "output_directory_absent": True,
        "formal_command": (
            ".\\.venv\\Scripts\\python.exe "
            "scripts\\run_s4_robust_controller_validation.py "
            "--acknowledge-unseen-validation"
        ),
        "experiment_config": _relative(experiment_path),
        "experiment_config_sha256": _file_sha256(experiment_path),
        "real_slm_actions": False,
        "rl_training": False,
    }


def _verify_upstream_d0(
    experiment: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = _project_path(experiment["upstream_s4d0_summary"])
    if not summary_path.exists():
        raise FileNotFoundError("S4-D0 summary is required")
    if _file_sha256(summary_path) != str(
        experiment["upstream_s4d0_summary_sha256"]
    ):
        raise RuntimeError("S4-D0 summary hash mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["experiment"]["status"] != "completed_pending_audit":
        raise RuntimeError("S4-D0 formal run is incomplete")

    audit_path = _project_path(experiment["upstream_s4d0_audit_record"])
    if _file_sha256(audit_path) != str(
        experiment["upstream_s4d0_audit_record_sha256"]
    ):
        raise RuntimeError("S4-D0 audit record hash mismatch")
    audit_text = audit_path.read_text(encoding="utf-8")
    if "Verification Status: `ANALYZED`" not in audit_text:
        raise RuntimeError("S4-D0 audit record is not marked ANALYZED")

    source_path = _project_path(experiment["upstream_s4d0_source_manifest"])
    trajectory_path = _project_path(experiment["upstream_s4d0_trajectory_manifest"])
    if _file_sha256(source_path) != str(
        experiment["upstream_s4d0_source_manifest_sha256"]
    ):
        raise RuntimeError("S4-D0 source manifest hash mismatch")
    if _file_sha256(trajectory_path) != str(
        experiment["upstream_s4d0_trajectory_manifest_sha256"]
    ):
        raise RuntimeError("S4-D0 trajectory manifest hash mismatch")
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    trajectory_manifest = json.loads(trajectory_path.read_text(encoding="utf-8"))
    if source_manifest != summary["inputs"]["source_manifest"]:
        raise RuntimeError("S4-D0 source manifest differs from its summary")
    if trajectory_manifest != summary["inputs"]["trajectory_manifest"]:
        raise RuntimeError("S4-D0 trajectory manifest differs from its summary")
    for relative, expected_hash in trajectory_manifest.items():
        path = _project_path(relative)
        if not path.exists() or _file_sha256(path) != expected_hash:
            raise RuntimeError(f"S4-D0 trajectory hash mismatch: {relative}")
    return summary, {
        "summary_sha256": _file_sha256(summary_path),
        "source_manifest_sha256": _file_sha256(source_path),
        "trajectory_manifest_sha256": _file_sha256(trajectory_path),
        "trajectory_count": len(trajectory_manifest),
        "all_trajectory_hashes_verified": True,
        "audit_record": _relative(audit_path),
    }


def _profile_csv_rows(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in results:
        gate = item["profile_gate"]
        incremental = item.get("paired_delta_vs_incumbent", {}).get(
            "power_in_bucket", {}
        )
        rows.append(
            {
                "controller": item["controller"],
                "profile": item["profile"],
                "required_for_validation": item[
                    "profile_required_for_selection"
                ],
                "episodes": item["episodes"],
                "reference_power_in_bucket": item["reference_power_in_bucket"],
                "power_in_bucket": item["power_in_bucket"]["mean"],
                "power_delta_ci95_low": item["paired_delta_vs_no_correction"][
                    "power_in_bucket"
                ]["ci95_low"],
                "mean_relative_power_gain": gate["mean_relative_power_gain"],
                "power_delta_vs_incumbent_ci95_low": incremental.get("ci95_low"),
                "strehl": item["strehl"]["mean"],
                "phase_rmse": item["phase_rmse"]["mean"],
                "violation_fraction": item["violation_fraction"]["mean"],
                "profile_gate": gate["validation_gate"],
            }
        )
    return rows


if __name__ == "__main__":
    main()
