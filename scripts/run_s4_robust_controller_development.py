"""运行S4-D0纯仿真鲁棒传统控制器开发比较。"""

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
from src.simulation.controller_selection import select_robust_controller
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
        default="configs/experiments/s4_robust_controller_development_v1.yaml",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只检查上游、种子、源码哈希、CUDA和输出目录。",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="运行不落盘的短CUDA诊断，不做控制器排名。",
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
    profiles = _profiles(experiment)
    preflight = _preflight(experiment, experiment_path, environment_path, env_config, profiles)

    if args.preflight_only:
        print(json.dumps({
            "stage": "S4-D0",
            "status": "READY_FOR_USER_DEVELOPMENT_RUN",
            "writes_performed": False,
            "formal_results_generated": False,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            **preflight,
        }, ensure_ascii=False, indent=2))
        return
    if args.quick:
        print(json.dumps(
            _run_quick(experiment, env_config, device, profiles),
            ensure_ascii=False,
            indent=2,
        ))
        return
    _run_development(experiment, experiment_path, environment_path, env_config, device, profiles)


def _run_quick(
    experiment: dict[str, Any],
    env_config: S1EnvConfig,
    device: torch.device,
    all_profiles: Sequence[HardwareProfile],
) -> dict[str, Any]:
    quick = experiment["quick"]
    profile_by_id = {item.identifier: item for item in all_profiles}
    profiles = [profile_by_id[str(item)] for item in quick["profile_ids"]]
    physical = RobustnessCondition.from_mapping(quick["physical_condition"])
    configured = replace(
        env_config,
        batch_size=int(quick["batch_size"]),
        episode_length=int(quick["steps"]),
    )
    rollouts = _execute_rollouts(
        experiment,
        configured,
        device,
        [physical],
        profiles,
        int(quick["steps"]),
        output_dir=None,
        description="S4-D0快速冒烟",
    )
    rows: list[dict[str, Any]] = []
    for controller, profile_map in rollouts.items():
        if controller == str(experiment["evaluation"]["reference_controller"]):
            continue
        for profile in profiles:
            summary = summarize_rollouts(profile_map[profile.identifier])
            rows.append({
                "controller": controller,
                "profile": profile.identifier,
                "power_in_bucket": summary["power_in_bucket"]["mean"],
                "violation_fraction": summary["violation_fraction"]["mean"],
            })
    return {
        "stage": "S4-D0",
        "status": "QUICK_SMOKE_COMPLETED",
        "diagnostic_only": True,
        "controller_ranking_not_allowed": True,
        "writes_performed": False,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "steps": int(quick["steps"]),
        "batch_size": int(quick["batch_size"]),
        "records": rows,
        "interpretation_boundary": "只证明新控制器和硬件误差链可运行，不能选择最终控制器或支撑论文结论。",
    }


def _run_development(
    experiment: dict[str, Any],
    experiment_path: Path,
    environment_path: Path,
    env_config: S1EnvConfig,
    device: torch.device,
    profiles: Sequence[HardwareProfile],
) -> None:
    output_dir = _project_path(experiment["outputs"]["directory"])
    if output_dir.exists():
        raise FileExistsError(f"development output already exists; refusing overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    (output_dir / "RUN_STARTED.json").write_text(
        json.dumps({
            "stage": "S4-D0",
            "started_at": started.isoformat(),
            "experiment_config": _relative(experiment_path),
            "experiment_config_sha256": _file_sha256(experiment_path),
            "note": "Software-only development comparison; retain failures and do not overwrite.",
        }, ensure_ascii=False, indent=2),
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
        output_dir=output_dir,
        description="S4-D0鲁棒控制器开发",
    )
    controller_results, condition_results = _analyze(
        experiment,
        rollouts,
        profiles,
        physical_conditions,
    )
    evaluation = experiment["evaluation"]
    selection = select_robust_controller(
        controller_results,
        candidate_ids=evaluation["candidate_controllers"],
        required_profile_ids=experiment["required_profile_ids"],
        incumbent_id=str(evaluation["incumbent_controller"]),
        nominal_profile_id=str(experiment["selection"]["nominal_profile_id"]),
        max_nominal_power_drop_fraction=float(
            experiment["selection"]["max_nominal_power_drop_fraction"]
        ),
    )
    source_manifest = _actual_source_manifest(experiment)
    trajectory_manifest = _trajectory_manifest(output_dir / "trajectories")
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": started.isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "s4_robust_controller_development_v1",
        },
        "experiment": {
            "id": "AO-S4-D0-ROBUST-CONTROLLER-DEVELOPMENT",
            "type": "software_only_classical_controller_comparison",
            "status": "completed_pending_audit",
            "duration_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "command": ".\\.venv\\Scripts\\python.exe scripts\\run_s4_robust_controller_development.py",
        },
        "inputs": {
            "experiment_config": _relative(experiment_path),
            "experiment_config_sha256": _file_sha256(experiment_path),
            "environment_config": _relative(environment_path),
            "environment_config_sha256": _file_sha256(environment_path),
            "upstream_s4c_summary": experiment["upstream_s4c_summary"],
            "upstream_s4c_summary_sha256": _file_sha256(
                _project_path(experiment["upstream_s4c_summary"])
            ),
            "source_manifest": source_manifest,
            "trajectory_manifest": trajectory_manifest,
            "runtime_versions": _runtime_versions(),
            "git": _git_state(),
        },
        "design": {
            "kind": "development_not_sealed_software_only",
            "primary_metric": evaluation["primary_metric"],
            "steps": int(evaluation["steps"]),
            "episodes_per_physical_condition": env_config.batch_size,
            "physical_conditions": [asdict(item) for item in physical_conditions],
            "hardware_profiles": [item.as_record() for item in profiles],
            "controllers": experiment["controllers"],
            "s4c0_results_are_diagnostic_inputs_not_reused_episodes": True,
            "scientific_metrics_use_noise_free_environment_truth": True,
        },
        "controller_profile_results": controller_results,
        "condition_results": condition_results,
        "selection": selection,
        "interpretation_boundary": (
            "S4-D0只在假设硬件误差和新开发种子上选择传统控制器；"
            "不是RL、FSLM-2K73-P04实测性能、封存测试或真实闭环。"
        ),
        "next_action": "停止并通知助手“S4-D0运行完成”，由助手只读审计后决定是否建立未见仿真验证。",
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
    _write_csv(output_dir / "controller_profile_summary.csv", _profile_csv_rows(controller_results))
    _write_csv(output_dir / "condition_summary.csv", condition_results)
    progress_message("S4-D0开发比较完成，请停止并告诉助手：S4-D0运行完成。")
    print(json.dumps({
        "summary": _relative(output_dir / "summary.json"),
        "gate": selection["validation_gate"],
        "selected_controller": selection["selected_controller"],
        "formal_results_generated": True,
    }, ensure_ascii=False, indent=2))


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
    factories = _controller_factories(experiment, base_config)
    results = {
        name: {profile.identifier: [] for profile in profiles}
        for name in factories
    }
    total_steps = len(factories) * len(profiles) * len(physical_conditions) * steps
    with counted_progress(total=total_steps, description=description, unit="步") as bar:
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
                    if output_dir is not None:
                        path = output_dir / "trajectories" / (
                            f"{controller_name}_{profile.identifier}_{physical.identifier}.h5"
                        )
                        export_rollout_h5(rollout, path, configured, metadata={
                            "stage": "S4-D0",
                            "physical_condition": physical.identifier,
                            "regime": physical.regime,
                            "hardware_profile": profile.identifier,
                            "hardware_profile_json": json.dumps(profile.as_record(), ensure_ascii=False),
                            "controller_definition_json": json.dumps(definition, ensure_ascii=False),
                            "observation_kind": experiment["evaluation"]["observation_kind"],
                            "observation_noise_std_rad": profile.observation_noise_std_rad,
                            "assumed_hardware_parameters_not_h1_measurements": True,
                        })
                        _append_progress(output_dir / "progress.jsonl", controller_name, profile, physical, path)
    return results


def _analyze(
    experiment: dict[str, Any],
    rollouts: dict[str, dict[str, list[ControllerRollout]]],
    profiles: Sequence[HardwareProfile],
    physical_conditions: Sequence[RobustnessCondition],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference_name = str(experiment["evaluation"]["reference_controller"])
    reference = rollouts[reference_name]
    results: list[dict[str, Any]] = []
    conditions: list[dict[str, Any]] = []
    parameters = {
        str(item["id"]): dict(item.get("parameters", {}))
        for item in experiment["controllers"]
    }
    for controller_name, profile_map in rollouts.items():
        if controller_name == reference_name:
            continue
        for profile in profiles:
            candidate_rollouts = profile_map[profile.identifier]
            reference_rollouts = reference[profile.identifier]
            candidate_summary = summarize_rollouts(candidate_rollouts)
            reference_summary = summarize_rollouts(reference_rollouts)
            candidate_summary["paired_delta_vs_no_correction"] = {
                metric: paired_delta(candidate_rollouts, reference_rollouts, metric)
                for metric in ("strehl", "power_in_bucket", "phase_rmse")
            }
            one_conditions = _condition_records(
                controller_name,
                profile,
                candidate_rollouts,
                reference_rollouts,
                physical_conditions,
            )
            gate = sealed_closed_loop_gate(
                candidate_summary,
                reference_summary,
                one_conditions,
                **experiment["profile_gate"],
            )
            results.append({
                "controller": controller_name,
                "profile": profile.identifier,
                "profile_label": profile.label,
                "profile_required_for_selection": profile.identifier in experiment["required_profile_ids"],
                **{key: value for key, value in candidate_summary.items() if key != "controller"},
                "parameters": parameters[controller_name],
                "reference_power_in_bucket": reference_summary["power_in_bucket"]["mean"],
                "profile_gate": gate,
            })
            conditions.extend(one_conditions)
    return results, conditions


def _condition_records(
    controller_name: str,
    profile: HardwareProfile,
    candidates: Sequence[ControllerRollout],
    references: Sequence[ControllerRollout],
    physical_conditions: Sequence[RobustnessCondition],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for physical, candidate, reference in zip(
        physical_conditions, candidates, references, strict=True
    ):
        cand = summarize_rollouts([candidate])
        ref = summarize_rollouts([reference])
        records.append({
            "controller": controller_name,
            "profile": profile.identifier,
            "physical_condition": physical.identifier,
            "regime": physical.regime,
            "episodes": cand["episodes"],
            "reference_power_in_bucket": ref["power_in_bucket"]["mean"],
            "power_in_bucket": cand["power_in_bucket"]["mean"],
            "strehl": cand["strehl"]["mean"],
            "phase_rmse": cand["phase_rmse"]["mean"],
            "violation_fraction": cand["violation_fraction"]["mean"],
            "saturated_fraction": cand["saturated_fraction"]["mean"],
            "slew_limited_fraction": cand["slew_limited_fraction"]["mean"],
            "settling_limited_fraction": cand["settling_limited_fraction"]["mean"],
            "action_latency_ms": cand["action_latency_ms"]["mean"],
            "power_delta_vs_no_correction": paired_delta([candidate], [reference], "power_in_bucket")["mean"],
            "strehl_delta_vs_no_correction": paired_delta([candidate], [reference], "strehl")["mean"],
            "phase_rmse_delta_vs_no_correction": paired_delta([candidate], [reference], "phase_rmse")["mean"],
        })
    return records


def _controller_factories(
    experiment: dict[str, Any], base_config: S1EnvConfig
) -> dict[str, ControllerFactory]:
    factories: dict[str, ControllerFactory] = {}
    for definition in experiment["controllers"]:
        identifier = str(definition["id"])
        kind = str(definition["kind"])
        parameters = dict(definition.get("parameters", {}))

        def factory(
            identifier: str = identifier,
            kind: str = kind,
            parameters: dict[str, Any] = parameters,
        ) -> ModalController:
            controller = make_controller(
                kind,
                base_config.num_modes,
                base_config.modal_limit_rad,
                **parameters,
            )
            controller.name = identifier
            return controller

        if identifier in factories:
            raise ValueError(f"duplicate controller id: {identifier}")
        factories[identifier] = factory
    return factories


def _controller_definition(experiment: dict[str, Any], identifier: str) -> dict[str, Any]:
    matches = [item for item in experiment["controllers"] if str(item["id"]) == identifier]
    if len(matches) != 1:
        raise ValueError(f"controller definition is not unique: {identifier}")
    return matches[0]


def _preflight(
    experiment: dict[str, Any],
    experiment_path: Path,
    environment_path: Path,
    env_config: S1EnvConfig,
    profiles: Sequence[HardwareProfile],
) -> dict[str, Any]:
    if _file_sha256(environment_path) != str(experiment["environment_config_sha256"]):
        raise RuntimeError("environment config hash mismatch")
    profile_source = _project_path(experiment["hardware_profile_source"])
    if _file_sha256(profile_source) != str(experiment["hardware_profile_source_sha256"]):
        raise RuntimeError("hardware profile source hash mismatch")
    upstream_path = _project_path(experiment["upstream_s4c_summary"])
    if not upstream_path.exists():
        raise FileNotFoundError("S4-C0 summary is required")
    if _file_sha256(upstream_path) != str(experiment["upstream_s4c_summary_sha256"]):
        raise RuntimeError("S4-C0 summary hash mismatch")
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    if upstream["s4c0_gate"]["validation_gate"] != "FAIL":
        raise RuntimeError("S4-D0 expects the audited S4-C0 failure diagnosis")
    failed_required = {
        item["profile"]
        for item in upstream["profile_results"]
        if item["required_for_gate"] and item["profile_gate"]["validation_gate"] == "FAIL"
    }
    if failed_required != {"settling_050", "registration_severe"}:
        raise RuntimeError("S4-C0 failure profiles differ from the audited diagnosis")

    definitions = {str(item["id"]): item for item in experiment["controllers"]}
    if len(definitions) != len(experiment["controllers"]):
        raise ValueError("controller ids must be unique")
    evaluation = experiment["evaluation"]
    required_controllers = {
        str(evaluation["reference_controller"]),
        str(evaluation["incumbent_controller"]),
        *map(str, evaluation["candidate_controllers"]),
    }
    if not required_controllers.issubset(definitions):
        raise ValueError("evaluation references an undefined controller")

    profile_ids = {item.identifier for item in profiles}
    required_profile_ids = set(map(str, experiment["required_profile_ids"]))
    if not required_profile_ids.issubset(profile_ids):
        raise ValueError("required profiles are missing")
    if not set(map(str, experiment["quick"]["profile_ids"])).issubset(profile_ids):
        raise ValueError("quick profiles are missing")
    if env_config.batch_size != int(evaluation["episodes_per_physical_condition"]):
        raise ValueError("environment batch size does not match evaluation episodes")

    conditions = _physical_conditions(experiment)
    new_seeds = {
        condition.base_seed + offset
        for condition in conditions
        for offset in range(env_config.batch_size)
    }
    upstream_seed_sets = {
        tuple(map(int, item["episode_seeds"]))
        for item in upstream["profile_results"]
    }
    if len(upstream_seed_sets) != 1:
        raise RuntimeError("S4-C0 profiles do not share the expected paired seeds")
    upstream_seeds = set(next(iter(upstream_seed_sets)))
    expected_new_seed_count = len(conditions) * env_config.batch_size
    if len(new_seeds) != expected_new_seed_count:
        raise RuntimeError("S4-D0 physical seed ranges overlap each other")
    if new_seeds & upstream_seeds:
        raise RuntimeError("S4-D0 development seeds overlap S4-C0")

    expected_manifest = {
        str(key): str(value) for key, value in experiment["source_manifest"].items()
    }
    actual_manifest = _actual_source_manifest(experiment)
    if actual_manifest != expected_manifest:
        raise RuntimeError("S4-D0 source manifest mismatch")
    output_dir = _project_path(experiment["outputs"]["directory"])
    if output_dir.exists():
        raise FileExistsError(f"development output already exists: {output_dir}")

    return {
        "experiment_config": _relative(experiment_path),
        "experiment_config_sha256": _file_sha256(experiment_path),
        "upstream_s4c_gate": "FAIL",
        "audited_failure_profiles": sorted(failed_required),
        "controller_count": len(definitions),
        "candidate_count": len(evaluation["candidate_controllers"]),
        "physical_condition_count": len(conditions),
        "hardware_profile_count": len(profiles),
        "required_profile_count": len(required_profile_ids),
        "episodes_per_controller_profile": len(conditions) * env_config.batch_size,
        "development_cuda_steps": len(definitions) * len(profiles) * len(conditions) * int(evaluation["steps"]),
        "source_manifest_verified": True,
        "seed_isolation_verified": True,
        "output_directory_absent": True,
        "real_slm_actions": False,
        "rl_training": False,
    }


def _profiles(experiment: dict[str, Any]) -> list[HardwareProfile]:
    source = _load_yaml(_project_path(experiment["hardware_profile_source"]))
    by_id = {
        item.identifier: item
        for item in map(HardwareProfile.from_mapping, source["hardware_profiles"])
    }
    requested = list(map(str, experiment["hardware_profile_ids"]))
    missing = [item for item in requested if item not in by_id]
    if missing:
        raise ValueError(f"hardware profiles not found in source: {missing}")
    return [by_id[item] for item in requested]


def _physical_conditions(experiment: dict[str, Any]) -> list[RobustnessCondition]:
    return [RobustnessCondition.from_mapping(item) for item in experiment["physical_conditions"]]


def _append_progress(
    path: Path,
    controller: str,
    profile: HardwareProfile,
    physical: RobustnessCondition,
    trajectory: Path,
) -> None:
    record = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "controller": controller,
        "hardware_profile": profile.identifier,
        "physical_condition": physical.identifier,
        "trajectory": _relative(trajectory),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _profile_csv_rows(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in results:
        gate = item["profile_gate"]
        rows.append({
            "controller": item["controller"],
            "profile": item["profile"],
            "required_for_selection": item["profile_required_for_selection"],
            "episodes": item["episodes"],
            "reference_power_in_bucket": item["reference_power_in_bucket"],
            "power_in_bucket": item["power_in_bucket"]["mean"],
            "power_delta_ci95_low": item["paired_delta_vs_no_correction"]["power_in_bucket"]["ci95_low"],
            "mean_relative_power_gain": gate["mean_relative_power_gain"],
            "strehl": item["strehl"]["mean"],
            "phase_rmse": item["phase_rmse"]["mean"],
            "violation_fraction": item["violation_fraction"]["mean"],
            "profile_gate": gate["validation_gate"],
        })
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("CSV rows must not be empty")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _trajectory_manifest(directory: Path) -> dict[str, str]:
    return {
        _relative(path): _file_sha256(path)
        for path in sorted(directory.glob("*.h5"))
    }


def _actual_source_manifest(experiment: dict[str, Any]) -> dict[str, str]:
    return {
        str(relative): _file_sha256(_project_path(str(relative)))
        for relative in experiment["source_manifest"]
    }


def _runtime_versions() -> dict[str, str | None]:
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
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
        ).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _load_yaml(path: Path) -> dict[str, Any]:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return values


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


if __name__ == "__main__":
    main()
