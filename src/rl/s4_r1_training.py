"""S4-D2-R1小残差范围的CUDA训练与失败前置检查。"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import torch

from src.rl.s4_training import (
    _effective_settings,
    _file_sha256,
    _git_record,
    _load_yaml,
    _make_residual_controller,
    _project_path,
    _relative,
    _runtime_record,
    _source_manifest,
    _summarize_policy_seeds,
    _train_one_policy_seed,
    _write_json,
)
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.training_progress import progress_message


def run_s4_r1_residual_sac(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """准备或运行R1；正式训练仍由用户在IDE终端启动。"""
    experiment_path = _project_path(config_path)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight = preflight_s4_r1(
        experiment_path,
        experiment,
        settings,
        quick=quick,
    )
    if preflight_only:
        return preflight

    device = resolve_device("cuda")
    output_directory = _project_path(settings["output_directory"])
    if output_directory.exists():
        raise FileExistsError(
            "R1 output already exists; preserve it and diagnose before retrying: "
            f"{output_directory}"
        )
    output_directory.mkdir(parents=True)
    source_manifest = _source_manifest(experiment["tracked_source_files"])
    _write_json(output_directory / "preflight.json", preflight)
    _write_json(output_directory / "source_manifest.json", source_manifest)
    _write_json(output_directory / "effective_config.json", settings)

    started = time.perf_counter()
    seed_results: list[dict[str, Any]] = []
    for policy_index, (policy_seed, environment_seed_base) in enumerate(
        zip(
            settings["policy_seeds"],
            settings["training_environment_seed_bases"],
            strict=True,
        ),
        start=1,
    ):
        seed_directory = output_directory / f"policy_seed_{policy_seed}"
        seed_directory.mkdir()
        progress_message(
            f"S4-D2-R1：开始策略种子 {policy_index}/{len(settings['policy_seeds'])} "
            f"({policy_seed})；共享训练内核进度条沿用S4-D2标签。"
        )
        result = _train_one_policy_seed(
            experiment=experiment,
            settings=settings,
            base_config=preflight["base_environment_config"],
            policy_seed=int(policy_seed),
            environment_seed_base=int(environment_seed_base),
            device=device,
            output_directory=seed_directory,
            seed_index=policy_index,
            seed_count=len(settings["policy_seeds"]),
        )
        seed_results.append(result)
        _write_json(
            output_directory / "partial_summary.json",
            {"policy_seeds": seed_results},
        )

    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "s4d2_r1_small_residual_sac_v1",
        },
        "experiment": {
            "id": "AO-S4-D2-R1-SMALL-RESIDUAL-SAC",
            "type": "software_only_cuda_one_factor_rl_revision",
            "status": "completed_pending_audit",
            "quick": quick,
            "duration_seconds": time.perf_counter() - started,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "evidence_boundary": {
            "real_slm_actions": False,
            "s4d1_trajectories_used_for_training": False,
            "s4d2_trajectories_used_for_training": False,
            "diagnostic_trajectories_used_for_training": False,
            "sealed_s4d3_accessed": False,
            "only_planned_factor_changed": True,
            "interpretation": (
                "R1 tests a smaller physical residual-action range in CUDA simulation. "
                "It cannot establish real FSLM performance or authorize S4-D3 without audit."
            ),
        },
        "one_factor_revision": {
            "factor": "residual_action_limit_rad",
            "s4d2_value": float(preflight["one_factor_revision"]["s4d2_value"]),
            "r1_value": float(preflight["one_factor_revision"]["r1_value"]),
            "ratio": float(preflight["one_factor_revision"]["ratio"]),
            "unchanged": list(preflight["one_factor_revision"]["unchanged"]),
        },
        "inputs": {
            "experiment_config": _relative(experiment_path),
            "experiment_config_sha256": _file_sha256(experiment_path),
            "source_manifest": source_manifest,
            "upstream_s4d2_gate": "FAIL",
            "upstream_diagnostic_status": "ANALYZED",
            "user_authorized_revision": True,
        },
        "policy_seed_results": seed_results,
        "development_summary": _summarize_policy_seeds(seed_results, settings),
        "runtime": _runtime_record(),
        "git": _git_record(),
        "next_action": (
            "Stop and ask the assistant to audit R1. Do not retrain, change the gate, "
            "or open S4-D3."
        ),
    }
    _write_json(output_directory / "summary.json", summary)
    return summary


def preflight_s4_r1(
    experiment_path: Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    """验证R1只改变动作物理上限，并保持所有保护边界。"""
    metadata = experiment["metadata"]
    if str(metadata["stage"]) != "S4-D2-R1":
        raise ValueError("experiment metadata must identify S4-D2-R1")
    if not bool(metadata.get("user_authorized_revision", False)):
        raise RuntimeError("R1 requires explicit user authorization")
    for field in (
        "allow_real_hardware_claims",
        "allow_prior_trajectory_reuse",
        "allow_sealed_test_access",
        "allow_gate_relaxation",
        "allow_algorithm_change",
        "allow_reward_change",
    ):
        if bool(metadata.get(field, True)):
            raise RuntimeError(f"R1 protection flag must remain false: {field}")

    upstream = _verify_upstream_evidence(experiment["upstream_evidence"])
    original = upstream["s4d2_config"]
    _verify_one_factor_revision(experiment, original)

    environment_path = _project_path(experiment["environment_config"])
    if _file_sha256(environment_path) != str(experiment["environment_config_sha256"]):
        raise RuntimeError("environment configuration hash mismatch")
    hardware_path = _project_path(experiment["hardware_profile_source"])
    if _file_sha256(hardware_path) != str(experiment["hardware_profile_source_sha256"]):
        raise RuntimeError("hardware profile source hash mismatch")
    base_config, _ = load_s1_config(environment_path)
    controller = _make_residual_controller(experiment, base_config)
    if controller.state_size != 100 or base_config.num_modes != 10:
        raise RuntimeError("R1 policy dimensions changed")

    output_directory = _project_path(settings["output_directory"])
    if output_directory.exists():
        raise FileExistsError(f"R1 output already exists: {output_directory}")
    _validate_seed_namespaces(experiment, settings, quick=quick)

    full_episode_transitions = (
        int(settings["environment_batch_size"]) * int(settings["episode_length"])
    )
    if int(settings["total_transitions_per_seed"]) % full_episode_transitions:
        raise ValueError("R1 training budget must contain complete batched episodes")
    if int(settings["warmup_transitions"]) < int(settings["sac_batch_size"]):
        raise ValueError("R1 warmup must fill at least one SAC minibatch")
    if len(settings["policy_seeds"]) != len(settings["training_environment_seed_bases"]):
        raise ValueError("R1 policy and environment seed counts differ")
    if not all(_project_path(path).is_file() for path in experiment["tracked_source_files"]):
        raise FileNotFoundError("one or more R1 tracked source files are missing")

    return {
        "status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_TRAINING",
        "quick": quick,
        "experiment_config": _relative(experiment_path),
        "experiment_config_sha256": _file_sha256(experiment_path),
        "output_directory": _relative(output_directory),
        "base_environment_config": asdict(base_config),
        "state_size": controller.state_size,
        "action_size": base_config.num_modes,
        "policy_seed_count": len(settings["policy_seeds"]),
        "transitions_per_policy_seed": int(settings["total_transitions_per_seed"]),
        "complete_batched_episodes_per_policy_seed": (
            int(settings["total_transitions_per_seed"]) // full_episode_transitions
        ),
        "one_factor_revision": {
            "factor": "residual_action_limit_rad",
            "s4d2_value": float(original["action"]["residual_action_limit_rad"]),
            "r1_value": float(experiment["action"]["residual_action_limit_rad"]),
            "ratio": float(experiment["action"]["residual_action_limit_rad"])
            / float(original["action"]["residual_action_limit_rad"]),
            "unchanged": [
                "algorithm",
                "network",
                "policy_observation",
                "reward",
                "frozen_controller",
                "training_budget",
                "physical_conditions",
                "hardware_profiles",
                "development_gate",
            ],
        },
        "upstream": {
            "s4d2_gate": "FAIL",
            "diagnostic_wiring": "PASS",
            "diagnostic_supports_excessive_amplitude": True,
            "diagnostic_supports_wrong_direction": False,
            "diagnostic_audit": "ANALYZED",
        },
        "seed_isolation_verified": True,
        "cuda_required": True,
        "real_slm_actions": False,
        "sealed_s4d3_access": False,
        "gate_relaxation_allowed": False,
        "algorithm_changed": False,
        "reward_changed": False,
        "formal_command": (
            ".\\.venv\\Scripts\\python.exe scripts\\train_s4_r1_residual_sac.py "
            "--config configs\\experiments\\s4_residual_sac_r1_v1.yaml"
        ),
    }


def _verify_upstream_evidence(upstream: dict[str, Any]) -> dict[str, Any]:
    verified: dict[str, Path] = {}
    for field in (
        "s4d2_summary",
        "s4d2_config",
        "s4d2_audit",
        "diagnostic_summary",
        "diagnostic_source_manifest",
        "diagnostic_config",
        "diagnostic_audit",
    ):
        path = _project_path(upstream[field])
        if _file_sha256(path) != str(upstream[f"{field}_sha256"]):
            raise RuntimeError(f"upstream evidence hash mismatch: {field}")
        verified[field] = path

    s4d2_summary = json.loads(verified["s4d2_summary"].read_text(encoding="utf-8"))
    if (
        s4d2_summary["development_summary"]["status"] != "ANALYSIS_REQUIRED"
        or int(s4d2_summary["development_summary"]["pass_count"]) != 0
        or bool(s4d2_summary["development_summary"]["s4d3_authorized"])
    ):
        raise RuntimeError("S4-D2 failure state changed")
    if "Verification Status: `ANALYZED`" not in verified["s4d2_audit"].read_text(
        encoding="utf-8"
    ):
        raise RuntimeError("S4-D2 audit is not ANALYZED")

    diagnostic = json.loads(
        verified["diagnostic_summary"].read_text(encoding="utf-8")
    )
    if bool(diagnostic["experiment"]["quick"]):
        raise RuntimeError("R1 cannot use a quick diagnostic as evidence")
    if diagnostic["zero_residual_equivalence"]["status"] != "PASS":
        raise RuntimeError("diagnostic wiring check did not pass")
    diagnosis = diagnostic["diagnosis"]
    if not bool(diagnosis["amplitude_check"]["supports_excessive_action_amplitude"]):
        raise RuntimeError("diagnostic does not support the R1 amplitude hypothesis")
    if bool(diagnosis["direction_check"]["supports_wrong_direction"]):
        raise RuntimeError("diagnostic indicates a direction issue; R1 is not isolated")
    if bool(diagnosis["s4d3_authorized"]):
        raise RuntimeError("S4-D3 must remain closed")
    if "Verification Status: `ANALYZED`" not in verified[
        "diagnostic_audit"
    ].read_text(encoding="utf-8"):
        raise RuntimeError("diagnostic audit is not ANALYZED")

    recorded_manifest = json.loads(
        verified["diagnostic_source_manifest"].read_text(encoding="utf-8")
    )
    for relative, digest in recorded_manifest.items():
        path = _project_path(relative)
        if not path.is_file() or _file_sha256(path) != digest:
            raise RuntimeError(f"diagnostic source changed: {relative}")
    return {
        "s4d2_config": _load_yaml(verified["s4d2_config"]),
        "diagnostic": diagnostic,
    }


def _verify_one_factor_revision(
    experiment: dict[str, Any], original: dict[str, Any]
) -> None:
    exact_fields = (
        "environment_config",
        "environment_config_sha256",
        "hardware_profile_source",
        "hardware_profile_source_sha256",
        "frozen_controller",
        "policy_observation",
        "reward",
        "sac",
        "training_seed_span_per_policy",
        "training_physical_conditions",
        "training_hardware_profile_ids",
        "interval_validation_profile_ids",
        "final_validation_profile_ids",
        "validation",
    )
    for field in exact_fields:
        if experiment[field] != original[field]:
            raise RuntimeError(f"R1 changed forbidden field: {field}")
    original_action = dict(original["action"])
    revised_action = dict(experiment["action"])
    old_limit = float(original_action.pop("residual_action_limit_rad"))
    new_limit = float(revised_action.pop("residual_action_limit_rad"))
    if revised_action != original_action:
        raise RuntimeError("R1 changed another action constraint")
    if old_limit != 0.05 or new_limit != 0.0125:
        raise RuntimeError("R1 residual limit must change only from 0.05 to 0.0125 rad")

    old_validation = original["validation_physical_conditions"]
    new_validation = experiment["validation_physical_conditions"]
    if len(old_validation) != len(new_validation):
        raise RuntimeError("R1 changed the number of validation conditions")
    for old, new in zip(old_validation, new_validation, strict=True):
        old_copy = {key: value for key, value in old.items() if key not in {"id", "base_seed"}}
        new_copy = {key: value for key, value in new.items() if key not in {"id", "base_seed"}}
        if old_copy != new_copy:
            raise RuntimeError("R1 changed validation physics instead of only seeds")


def _validate_seed_namespaces(
    experiment: dict[str, Any], settings: dict[str, Any], *, quick: bool
) -> None:
    span = int(settings["training_seed_span_per_policy"])
    training_ranges = [
        range(int(base), int(base) + span)
        for base in settings["training_environment_seed_bases"]
    ]
    for index, left in enumerate(training_ranges):
        left_set = set(left)
        for right in training_ranges[index + 1 :]:
            if left_set.intersection(right):
                raise RuntimeError("R1 policy training seed ranges overlap")
    validation_seeds = {
        int(item["base_seed"]) + offset
        for item in settings["validation_physical_conditions"]
        for offset in range(int(settings["validation_batch_size"]))
    }
    if any(seed in training for training in training_ranges for seed in validation_seeds):
        raise RuntimeError("R1 training and validation seeds overlap")
    active = set(validation_seeds)
    for training in training_ranges:
        active.update(training)
    for protected in experiment["protected_seed_ranges"]:
        start = int(protected["start_inclusive"])
        end = int(protected["end_exclusive"])
        if any(start <= seed < end for seed in active):
            raise RuntimeError(f"R1 overlaps protected seeds: {protected['id']}")
    reserved = int(experiment["reserved_oracle_seed_base"])
    if any(seed >= reserved for seed in active):
        raise RuntimeError("R1 overlaps the reserved oracle namespace")

    # 正式与快速命名空间必须互相隔离，即使本次只运行其中一个。
    formal_ranges = [
        range(int(base), int(base) + int(experiment["training_seed_span_per_policy"]))
        for base in experiment["training_environment_seed_bases"]
    ]
    quick_base = int(experiment["quick"]["training_environment_seed_bases"][0])
    quick_range = range(quick_base, quick_base + int(experiment["training_seed_span_per_policy"]))
    if any(set(item).intersection(quick_range) for item in formal_ranges):
        raise RuntimeError("R1 formal and quick training seed ranges overlap")
    if quick and len(settings["policy_seeds"]) != 1:
        raise RuntimeError("R1 quick smoke must use exactly one policy seed")
