"""运行S4-C0保守硬件误差纯仿真压力测试。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
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
from src.simulation.hardware_effects import HardwareProfile
from src.simulation.robust_control import RobustnessCondition, sealed_closed_loop_gate
from src.training_progress import counted_progress, gpu_memory_status, progress_message


ControllerFactory = Callable[[], ModalController]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/s4_hardware_stress_v1.yaml",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只检查上游结果、种子、源码哈希、CUDA和输出目录，不运行轨迹。",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="运行两个硬件档位的短CUDA诊断冒烟，不写正式输出。",
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
        print(
            json.dumps(
                {
                    "stage": "S4-C0",
                    "status": "READY_FOR_USER_FORMAL_RUN",
                    "writes_performed": False,
                    "formal_results_generated": False,
                    "device": str(device),
                    "gpu_name": torch.cuda.get_device_name(device),
                    **preflight,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.quick:
        result = _run_quick(experiment, env_config, device)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    _run_formal(experiment, experiment_path, environment_path, env_config, device)


def _run_quick(
    experiment: dict[str, Any],
    env_config: S1EnvConfig,
    device: torch.device,
) -> dict[str, Any]:
    quick = experiment["quick"]
    physical = RobustnessCondition.from_mapping(quick["physical_condition"])
    profile_by_id = {
        profile.identifier: profile
        for profile in _profiles(experiment)
    }
    profiles = [profile_by_id[str(item)] for item in quick["profile_ids"]]
    quick_config = replace(
        env_config,
        batch_size=int(quick["batch_size"]),
        episode_length=int(quick["steps"]),
    )
    rollouts = _execute_rollouts(
        experiment,
        quick_config,
        device,
        [physical],
        profiles,
        int(quick["steps"]),
        output_dir=None,
        description="S4-C0快速冒烟",
    )
    profile_results, _ = _analyze_profiles(experiment, rollouts, profiles, [physical])
    return {
        "stage": "S4-C0",
        "status": "QUICK_SMOKE_COMPLETED",
        "diagnostic_only": True,
        "formal_gate_not_evaluated": True,
        "writes_performed": False,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "steps": int(quick["steps"]),
        "batch_size": int(quick["batch_size"]),
        "profiles": [
            {
                "profile": item["profile"],
                "power_delta": item["paired_delta_vs_no_correction"][
                    "power_in_bucket"
                ]["mean"],
                "violation_fraction": item["violation_fraction"]["mean"],
            }
            for item in profile_results
        ],
        "interpretation_boundary": "仅证明S4-C软件链可运行，不能用于算法排名或正式结论。",
    }


def _run_formal(
    experiment: dict[str, Any],
    experiment_path: Path,
    environment_path: Path,
    env_config: S1EnvConfig,
    device: torch.device,
) -> None:
    output_dir = _project_path(experiment["outputs"]["directory"])
    if output_dir.exists():
        raise FileExistsError(f"formal output already exists; refusing overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    start_record = {
        "stage": "S4-C0",
        "started_at": started.isoformat(),
        "experiment_config": _relative(experiment_path),
        "experiment_config_sha256": _file_sha256(experiment_path),
        "note": "Development stress test; retain this directory if interrupted and do not auto-retry.",
    }
    (output_dir / "RUN_STARTED.json").write_text(
        json.dumps(start_record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    physical_conditions = _physical_conditions(experiment)
    profiles = _profiles(experiment)
    rollouts = _execute_rollouts(
        experiment,
        env_config,
        device,
        physical_conditions,
        profiles,
        int(experiment["evaluation"]["steps"]),
        output_dir=output_dir,
        description="S4-C0硬件压力",
    )
    profile_results, condition_records = _analyze_profiles(
        experiment,
        rollouts,
        profiles,
        physical_conditions,
    )
    required = [item for item in profile_results if item["required_for_gate"]]
    overall_gate = "PASS" if all(
        item["profile_gate"]["validation_gate"] == "PASS" for item in required
    ) else "FAIL"

    trajectory_manifest = _trajectory_manifest(output_dir / "trajectories")
    source_manifest = _actual_source_manifest(experiment)
    duration = (datetime.now(timezone.utc) - started).total_seconds()
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": started.isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "s4_hardware_stress_v1",
        },
        "experiment": {
            "id": "AO-S4-C0-HARDWARE-STRESS",
            "type": "hardware_error_simulation_stress_test",
            "status": "completed_pending_audit",
            "duration_seconds": duration,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "command": ".\\.venv\\Scripts\\python.exe scripts\\run_s4_hardware_stress.py",
        },
        "inputs": {
            "experiment_config": _relative(experiment_path),
            "experiment_config_sha256": _file_sha256(experiment_path),
            "environment_config": _relative(environment_path),
            "environment_config_sha256": _file_sha256(environment_path),
            "upstream_s4b_summary": experiment["upstream_s4b_summary"],
            "upstream_s4b_summary_sha256": _file_sha256(
                _project_path(experiment["upstream_s4b_summary"])
            ),
            "source_manifest": source_manifest,
            "trajectory_manifest": trajectory_manifest,
            "runtime_versions": _runtime_versions(),
            "git": _git_state(),
        },
        "design": {
            "kind": "development_hardware_stress_not_sealed",
            "primary_metric": experiment["evaluation"]["primary_metric"],
            "steps": int(experiment["evaluation"]["steps"]),
            "episodes_per_physical_condition": env_config.batch_size,
            "physical_conditions": [asdict(item) for item in physical_conditions],
            "hardware_profiles": [item.as_record() for item in profiles],
            "profile_statistics_do_not_pool_repeated_physical_seeds": True,
            "scientific_metrics_use_noise_free_environment_truth": True,
            "power_meter_noise_only_changes_measured_power_dataset": True,
        },
        "frozen_controller": experiment["frozen_controller"],
        "profile_results": profile_results,
        "condition_results": condition_records,
        "s4c0_gate": {
            "validation_gate": overall_gate,
            "required_profile_count": len(required),
            "all_required_profiles_pass": overall_gate == "PASS",
            "combined_severe_is_diagnostic_only": True,
        },
        "interpretation_boundary": (
            "S4-C0使用保守假设的纯仿真硬件误差，只能寻找失效边界；"
            "不是FSLM-2K73-P04实测性能、真实SLM闭环、强化学习或自然大气结果。"
        ),
        "next_action": "停止并通知助手“S4-C0运行完成”，由助手只读审计结果。",
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
    _write_csv(output_dir / "profile_summary.csv", _profile_csv_rows(profile_results))
    _write_csv(output_dir / "condition_summary.csv", condition_records)
    progress_message("S4-C0正式压力测试完成，请停止并告诉助手：S4-C0运行完成。")
    print(
        json.dumps(
            {
                "summary": _relative(output_dir / "summary.json"),
                "gate": overall_gate,
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
    *,
    output_dir: Path | None,
    description: str,
) -> dict[str, dict[str, list[ControllerRollout]]]:
    frozen = experiment["frozen_controller"]
    parameters = dict(frozen["parameters"])
    factories: dict[str, ControllerFactory] = {
        "no_correction": lambda: make_controller(
            "no_correction", base_config.num_modes, base_config.modal_limit_rad
        ),
        str(frozen["name"]): lambda: make_controller(
            str(frozen["name"]),
            base_config.num_modes,
            base_config.modal_limit_rad,
            **parameters,
        ),
    }
    results: dict[str, dict[str, list[ControllerRollout]]] = {
        name: {profile.identifier: [] for profile in profiles}
        for name in factories
    }
    total_steps = len(factories) * len(profiles) * len(physical_conditions) * steps
    with counted_progress(total=total_steps, description=description, unit="步") as bar:
        for controller_name, factory in factories.items():
            for profile in profiles:
                for physical in physical_conditions:
                    bar.set_postfix_str(
                        f"控制器={controller_name} 档位={profile.identifier} "
                        f"物理={physical.identifier} 显存={gpu_memory_status(device)}"
                    )
                    physical_config = physical.environment_config(base_config)
                    configured = profile.environment_config(physical_config)
                    rollout = run_controller_rollout(
                        configured,
                        device,
                        factory(),
                        physical.base_seed,
                        steps,
                        observation_noise_std_rad=profile.observation_noise_std_rad,
                        hardware_effects=profile.effects_config(),
                        progress_callback=lambda _completed, _total: bar.update(1),
                    )
                    results[controller_name][profile.identifier].append(rollout)
                    if output_dir is not None:
                        trajectory_path = (
                            output_dir
                            / "trajectories"
                            / f"{controller_name}_{profile.identifier}_{physical.identifier}.h5"
                        )
                        export_rollout_h5(
                            rollout,
                            trajectory_path,
                            configured,
                            metadata={
                                "stage": "S4-C0",
                                "physical_condition": physical.identifier,
                                "regime": physical.regime,
                                "hardware_profile": profile.identifier,
                                "hardware_profile_json": json.dumps(
                                    profile.as_record(), ensure_ascii=False
                                ),
                                "observation_kind": experiment["evaluation"][
                                    "observation_kind"
                                ],
                                "observation_noise_std_rad": (
                                    profile.observation_noise_std_rad
                                ),
                                "assumed_hardware_parameters_not_h1_measurements": True,
                            },
                        )
                        _append_progress_record(
                            output_dir / "progress.jsonl",
                            controller_name,
                            profile,
                            physical,
                            trajectory_path,
                        )
    return results


def _analyze_profiles(
    experiment: dict[str, Any],
    rollouts: dict[str, dict[str, list[ControllerRollout]]],
    profiles: Sequence[HardwareProfile],
    physical_conditions: Sequence[RobustnessCondition],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference_name = str(experiment["evaluation"]["reference_controller"])
    candidate_name = str(experiment["frozen_controller"]["name"])
    profile_results: list[dict[str, Any]] = []
    condition_records: list[dict[str, Any]] = []
    for profile in profiles:
        candidate_rollouts = rollouts[candidate_name][profile.identifier]
        reference_rollouts = rollouts[reference_name][profile.identifier]
        reference_summary = summarize_rollouts(reference_rollouts)
        candidate_summary = summarize_rollouts(candidate_rollouts)
        candidate_summary["parameters"] = dict(
            experiment["frozen_controller"]["parameters"]
        )
        candidate_summary["paired_delta_vs_no_correction"] = {
            metric: paired_delta(candidate_rollouts, reference_rollouts, metric)
            for metric in ("strehl", "power_in_bucket", "phase_rmse")
        }
        one_profile_records = _physical_records(
            profile,
            candidate_rollouts,
            reference_rollouts,
            physical_conditions,
        )
        gate = sealed_closed_loop_gate(
            candidate_summary,
            reference_summary,
            one_profile_records,
            **experiment["gate"],
        )
        profile_results.append(
            {
                "profile": profile.identifier,
                "label": profile.label,
                "severity": profile.severity,
                "required_for_gate": profile.required_for_gate,
                **candidate_summary,
                "reference_power_in_bucket": reference_summary["power_in_bucket"][
                    "mean"
                ],
                "profile_gate": gate,
            }
        )
        condition_records.extend(one_profile_records)
    return profile_results, condition_records


def _physical_records(
    profile: HardwareProfile,
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
                "profile": profile.identifier,
                "profile_required_for_gate": profile.required_for_gate,
                "physical_condition": condition.identifier,
                "regime": condition.regime,
                "episodes": candidate_summary["episodes"],
                "reference_power_in_bucket": reference_summary["power_in_bucket"][
                    "mean"
                ],
                "power_in_bucket": candidate_summary["power_in_bucket"]["mean"],
                "measured_power_in_bucket": candidate_summary[
                    "measured_power_in_bucket"
                ]["mean"],
                "strehl": candidate_summary["strehl"]["mean"],
                "phase_rmse": candidate_summary["phase_rmse"]["mean"],
                "violation_fraction": candidate_summary["violation_fraction"]["mean"],
                "saturated_fraction": candidate_summary["saturated_fraction"]["mean"],
                "slew_limited_fraction": candidate_summary["slew_limited_fraction"][
                    "mean"
                ],
                "settling_limited_fraction": candidate_summary[
                    "settling_limited_fraction"
                ]["mean"],
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


def _preflight(
    experiment: dict[str, Any],
    experiment_path: Path,
    environment_path: Path,
    env_config: S1EnvConfig,
) -> dict[str, Any]:
    upstream_path = _project_path(experiment["upstream_s4b_summary"])
    if not upstream_path.exists():
        raise FileNotFoundError("S4-B summary is required")
    if _file_sha256(upstream_path) != str(experiment["upstream_s4b_summary_sha256"]):
        raise RuntimeError("S4-B summary hash mismatch")
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    if upstream["s4b_sealed_gate"]["validation_gate"] != "PASS":
        raise RuntimeError("S4-B sealed gate did not pass")
    if upstream["frozen_controller"]["name"] != experiment["frozen_controller"]["name"]:
        raise RuntimeError("S4-C controller differs from S4-B")
    if upstream["frozen_controller"]["parameters"] != experiment["frozen_controller"][
        "parameters"
    ]:
        raise RuntimeError("S4-C controller parameters differ from S4-B")

    physical_conditions = _physical_conditions(experiment)
    profiles = _profiles(experiment)
    if len({item.identifier for item in physical_conditions}) != len(physical_conditions):
        raise RuntimeError("physical condition ids must be unique")
    if len({item.identifier for item in profiles}) != len(profiles):
        raise RuntimeError("hardware profile ids must be unique")
    if not any(item.severity == "combined_moderate" for item in profiles):
        raise RuntimeError("combined_moderate profile is required")
    severe = [item for item in profiles if item.severity == "combined_severe"]
    if len(severe) != 1 or severe[0].required_for_gate:
        raise RuntimeError("exactly one diagnostic-only combined_severe profile is required")
    _verify_seed_isolation(upstream, physical_conditions, experiment, env_config.batch_size)

    actual_manifest = _actual_source_manifest(experiment)
    expected_manifest = {
        str(key): str(value) for key, value in experiment["source_manifest"].items()
    }
    if actual_manifest != expected_manifest:
        raise RuntimeError("S4-C source manifest mismatch")
    output_dir = _project_path(experiment["outputs"]["directory"])
    if output_dir.exists():
        raise FileExistsError(f"formal output already exists: {output_dir}")
    return {
        "experiment_config": _relative(experiment_path),
        "experiment_config_sha256": _file_sha256(experiment_path),
        "environment_config": _relative(environment_path),
        "environment_config_sha256": _file_sha256(environment_path),
        "upstream_s4b_gate": "PASS",
        "frozen_controller": experiment["frozen_controller"],
        "physical_condition_count": len(physical_conditions),
        "hardware_profile_count": len(profiles),
        "required_profile_count": sum(item.required_for_gate for item in profiles),
        "episodes_per_profile": len(physical_conditions) * env_config.batch_size,
        "formal_cuda_steps": (
            2
            * len(physical_conditions)
            * len(profiles)
            * int(experiment["evaluation"]["steps"])
        ),
        "source_manifest_verified": True,
        "output_directory_absent": True,
        "real_slm_actions": False,
        "rl_training": False,
    }


def _verify_seed_isolation(
    upstream: dict[str, Any],
    physical_conditions: Sequence[RobustnessCondition],
    experiment: dict[str, Any],
    batch_size: int,
) -> None:
    new_seeds = {
        condition.base_seed + offset
        for condition in physical_conditions
        for offset in range(batch_size)
    }
    prior_seeds = {
        int(seed)
        for result in upstream["controller_results"]
        for seed in result["episode_seeds"]
    }
    s4a_path = _project_path(upstream["inputs"]["upstream_s4a_summary"])
    if s4a_path.exists():
        s4a = json.loads(s4a_path.read_text(encoding="utf-8"))
        prior_seeds.update(
            int(seed)
            for result in s4a["controller_results"]
            for seed in result["episode_seeds"]
        )
    if new_seeds & prior_seeds:
        raise RuntimeError("S4-C physical seeds overlap with S4-A or S4-B")
    quick = RobustnessCondition.from_mapping(experiment["quick"]["physical_condition"])
    quick_seeds = {
        quick.base_seed + offset for offset in range(int(experiment["quick"]["batch_size"]))
    }
    if new_seeds & quick_seeds or prior_seeds & quick_seeds:
        raise RuntimeError("S4-C quick seeds overlap with formal or upstream seeds")


def _profiles(experiment: dict[str, Any]) -> list[HardwareProfile]:
    return [HardwareProfile.from_mapping(item) for item in experiment["hardware_profiles"]]


def _physical_conditions(experiment: dict[str, Any]) -> list[RobustnessCondition]:
    return [
        RobustnessCondition.from_mapping(item)
        for item in experiment["physical_conditions"]
    ]


def _trajectory_manifest(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _file_sha256(path)
        for path in sorted(directory.glob("*.h5"))
    }


def _actual_source_manifest(experiment: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative_path in experiment["source_manifest"]:
        path = _project_path(relative_path)
        if not path.exists():
            raise FileNotFoundError(f"frozen source file is missing: {relative_path}")
        result[str(relative_path)] = _file_sha256(path)
    return result


def _append_progress_record(
    path: Path,
    controller_name: str,
    profile: HardwareProfile,
    physical: RobustnessCondition,
    trajectory_path: Path,
) -> None:
    record = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "controller": controller_name,
        "hardware_profile": profile.identifier,
        "physical_condition": physical.identifier,
        "trajectory": _relative(trajectory_path),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _profile_csv_rows(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "profile": item["profile"],
            "label": item["label"],
            "severity": item["severity"],
            "required_for_gate": item["required_for_gate"],
            "episodes": item["episodes"],
            "reference_power_in_bucket": item["reference_power_in_bucket"],
            "power_in_bucket": item["power_in_bucket"]["mean"],
            "power_delta_ci95_low": item["paired_delta_vs_no_correction"][
                "power_in_bucket"
            ]["ci95_low"],
            "mean_relative_power_gain": item["profile_gate"][
                "mean_relative_power_gain"
            ],
            "strehl": item["strehl"]["mean"],
            "phase_rmse": item["phase_rmse"]["mean"],
            "violation_fraction": item["violation_fraction"]["mean"],
            "profile_gate": item["profile_gate"]["validation_gate"],
        }
        for item in results
    ]


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("CSV rows must not be empty")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _runtime_versions() -> dict[str, Any]:
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "h5py": h5py.__version__,
    }


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
