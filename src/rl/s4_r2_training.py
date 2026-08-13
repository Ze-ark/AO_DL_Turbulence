"""S4-D2-R2高维泽尼克残差SAC的准备、训练编排与证据边界。"""

from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import asdict, replace
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
from src.simulation.config import S1EnvConfig, load_s1_config
from src.training_progress import progress_message


def run_s4_r2_residual_sac(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """准备或运行R2；正式训练仍由用户在IDE终端启动。"""
    experiment_path = _project_path(config_path)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight = preflight_s4_r2(
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
            "R2 output already exists; preserve it and diagnose before retrying: "
            f"{output_directory}"
        )
    output_directory.mkdir(parents=True)
    source_manifest = _source_manifest(experiment["tracked_source_files"])
    _write_json(output_directory / "preflight.json", preflight)
    _write_json(output_directory / "source_manifest.json", source_manifest)
    _write_json(output_directory / "effective_config.json", settings)

    started = time.perf_counter()
    representation_results: list[dict[str, Any]] = []
    for representation_index, representation in enumerate(
        experiment["representations"], start=1
    ):
        representation_id = str(representation["id"])
        representation_directory = output_directory / representation_id
        representation_directory.mkdir()
        representation_experiment = deepcopy(experiment)
        representation_experiment["active_representation"] = deepcopy(representation)
        representation_experiment["action"]["num_modes"] = int(
            representation["num_modes"]
        )
        representation_settings = deepcopy(settings)
        representation_settings["progress_stage_label"] = (
            f"S4-D2-R2 {representation_id} "
        )
        base_config = dict(preflight["base_environment_config"])
        base_config["num_modes"] = int(representation["num_modes"])

        progress_message(
            f"S4-D2-R2：开始表示 {representation_index}/"
            f"{len(experiment['representations'])}（{representation_id}）。"
        )
        seed_results: list[dict[str, Any]] = []
        for policy_index, (policy_seed, environment_seed_base) in enumerate(
            zip(
                settings["policy_seeds"],
                settings["training_environment_seed_bases"],
                strict=True,
            ),
            start=1,
        ):
            seed_directory = representation_directory / f"policy_seed_{policy_seed}"
            seed_directory.mkdir()
            result = _train_one_policy_seed(
                experiment=representation_experiment,
                settings=representation_settings,
                base_config=base_config,
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
                {
                    "completed_representations": representation_results,
                    "active_representation": {
                        "representation": representation,
                        "policy_seed_results": seed_results,
                    },
                },
            )
        representation_results.append(
            {
                "representation": deepcopy(representation),
                "policy_seed_results": seed_results,
                "development_summary": _summarize_policy_seeds(seed_results, settings),
            }
        )

    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "s4d2_r2_high_dimensional_residual_sac_v1",
        },
        "experiment": {
            "id": "AO-S4-D2-R2-HIGH-DIMENSIONAL-RESIDUAL-SAC",
            "type": "software_only_cuda_representation_revision",
            "status": "completed_pending_audit",
            "quick": quick,
            "duration_seconds": time.perf_counter() - started,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "evidence_boundary": {
            "real_slm_actions": False,
            "prior_trajectories_used_for_training": False,
            "simulator_truth_in_policy_observation": False,
            "sealed_s4d3_accessed": False,
            "traditional_baseline_anchor_modes": int(
                experiment["frozen_controller"]["active_anchor_modes"]
            ),
            "total_action_budget_preserved": True,
            "interpretation": (
                "R2 trains and development-validates 21D and 36D residual SAC in CUDA "
                "simulation. It cannot establish real FSLM performance or authorize S4-D3 "
                "before an independent audit."
            ),
        },
        "inputs": {
            "experiment_config": _relative(experiment_path),
            "experiment_config_sha256": _file_sha256(experiment_path),
            "source_manifest": source_manifest,
            "representation_capacity_status": "ANALYZED",
            "user_authorized_next_stage": True,
        },
        "design": {
            "primary_representation_id": str(
                experiment["selection_rule"]["primary_representation_id"]
            ),
            "challenger_representation_id": str(
                experiment["selection_rule"]["challenger_representation_id"]
            ),
            "paired_training_and_validation_seeds": True,
            "per_episode_validation_records": True,
            "representation_selection_deferred_to_audit": True,
        },
        "representation_results": representation_results,
        "development_summary": {
            "status": "ANALYSIS_REQUIRED",
            "selected_representation": None,
            "s4d3_authorized": False,
        },
        "runtime": _runtime_record(),
        "git": _git_record(),
        "next_action": (
            "Stop and ask the assistant to audit R2. Do not retrain, select a representation, "
            "open S4-D3, or operate real SLM hardware."
        ),
    }
    _write_json(output_directory / "summary.json", summary)
    return summary


def preflight_s4_r2(
    experiment_path: Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    """验证R2确由容量证据触发，并保持算法、基线和安全预算不变。"""
    metadata = experiment["metadata"]
    if str(metadata["stage"]) != "S4-D2-R2":
        raise ValueError("experiment metadata must identify S4-D2-R2")
    if not bool(metadata.get("user_authorized_next_stage", False)):
        raise RuntimeError("R2 requires explicit user authorization")
    for field in (
        "allow_real_hardware_actions",
        "allow_prior_trajectory_reuse",
        "allow_sealed_test_access",
        "allow_gate_relaxation",
        "allow_algorithm_change",
        "allow_reward_change",
        "allow_total_action_budget_increase",
    ):
        if bool(metadata.get(field, True)):
            raise RuntimeError(f"R2 protection flag must remain false: {field}")

    capacity = _verify_capacity_evidence(experiment["upstream_capacity"])
    passing = set(map(str, capacity["interpretation"]["passing_representation_ids"]))
    representations = list(experiment["representations"])
    representation_ids = [str(item["id"]) for item in representations]
    if representation_ids != ["zernike_21", "zernike_36"]:
        raise RuntimeError("R2 must compare the predeclared 21D primary and 36D challenger")
    for item in representations:
        if str(item["id"]) not in passing:
            raise RuntimeError(f"representation did not pass capacity gate: {item['id']}")
        if str(item["kind"]) != "zernike":
            raise RuntimeError("R2 only permits the passing Zernike representations")
        if int(item["num_modes"]) not in {21, 36}:
            raise RuntimeError("R2 action dimension must be 21 or 36")

    environment_path = _project_path(experiment["environment_config"])
    if _file_sha256(environment_path) != str(experiment["environment_config_sha256"]):
        raise RuntimeError("environment configuration hash mismatch")
    hardware_path = _project_path(experiment["hardware_profile_source"])
    if _file_sha256(hardware_path) != str(experiment["hardware_profile_source_sha256"]):
        raise RuntimeError("hardware profile source hash mismatch")
    base_config, _ = load_s1_config(environment_path)
    if base_config.num_modes != int(experiment["frozen_controller"]["active_anchor_modes"]):
        raise RuntimeError("the frozen baseline anchor is no longer the original ten modes")
    _verify_action_and_learning_contract(experiment, capacity)

    controller_contracts: list[dict[str, Any]] = []
    for item in representations:
        representation_config = replace(base_config, num_modes=int(item["num_modes"]))
        representation_experiment = deepcopy(experiment)
        representation_experiment["action"]["num_modes"] = int(item["num_modes"])
        controller = _make_residual_controller(
            representation_experiment, representation_config
        )
        expected_state_size = 10 * int(item["num_modes"])
        if controller.state_size != expected_state_size:
            raise RuntimeError("R2 residual policy state contract is inconsistent")
        controller_contracts.append(
            {
                "id": str(item["id"]),
                "num_modes": int(item["num_modes"]),
                "state_size": controller.state_size,
                "action_size": int(item["num_modes"]),
            }
        )

    output_directory = _project_path(settings["output_directory"])
    if output_directory.exists():
        raise FileExistsError(f"R2 output already exists: {output_directory}")
    _validate_seed_namespaces(experiment, settings, quick=quick)
    full_episode_transitions = (
        int(settings["environment_batch_size"]) * int(settings["episode_length"])
    )
    if int(settings["total_transitions_per_seed"]) % full_episode_transitions:
        raise ValueError("R2 training budget must contain complete batched episodes")
    if int(settings["warmup_transitions"]) < int(settings["sac_batch_size"]):
        raise ValueError("R2 warmup must fill at least one SAC minibatch")
    if len(settings["policy_seeds"]) != len(settings["training_environment_seed_bases"]):
        raise ValueError("R2 policy and environment seed counts differ")
    if not bool(settings["persist_final_episode_records"]):
        raise RuntimeError("R2 must persist final per-episode validation records")
    if not all(_project_path(path).is_file() for path in experiment["tracked_source_files"]):
        raise FileNotFoundError("one or more R2 tracked source files are missing")

    return {
        "status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_TRAINING",
        "quick": quick,
        "experiment_config": _relative(experiment_path),
        "experiment_config_sha256": _file_sha256(experiment_path),
        "output_directory": _relative(output_directory),
        "base_environment_config": asdict(base_config),
        "representation_contracts": controller_contracts,
        "policy_seed_count_per_representation": len(settings["policy_seeds"]),
        "representation_count": len(representations),
        "total_policy_runs": len(representations) * len(settings["policy_seeds"]),
        "transitions_per_policy_seed": int(settings["total_transitions_per_seed"]),
        "complete_batched_episodes_per_policy_seed": (
            int(settings["total_transitions_per_seed"]) // full_episode_transitions
        ),
        "capacity_evidence": {
            "status": "ANALYZED",
            "passing_representation_ids": sorted(passing),
            "source_run_quick": False,
            "deployable_controller": False,
        },
        "scientific_contract": {
            "algorithm": "SAC",
            "network_hidden_size": int(experiment["sac"]["hidden_size"]),
            "frozen_baseline_anchor_modes": int(
                experiment["frozen_controller"]["active_anchor_modes"]
            ),
            "residual_component_limit_rad": float(
                experiment["action"]["residual_action_limit_rad"]
            ),
            "residual_l2_budget_rad": (
                int(experiment["action"]["total_budget_anchor_modes"]) ** 0.5
                * float(experiment["action"]["residual_action_limit_rad"])
            ),
            "final_l2_budget_rad": (
                int(experiment["action"]["total_budget_anchor_modes"]) ** 0.5
                * float(experiment["action"]["final_action_step_limit_rad"])
            ),
            "per_episode_validation_records": True,
        },
        "seed_isolation_verified": True,
        "cuda_required": True,
        "real_slm_actions": False,
        "sealed_s4d3_access": False,
        "formal_command": (
            ".\\.venv\\Scripts\\python.exe scripts\\train_s4_r2_residual_sac.py "
            "--config configs\\experiments\\s4_residual_sac_r2_v1.yaml"
        ),
    }


def _verify_capacity_evidence(upstream: dict[str, Any]) -> dict[str, Any]:
    paths: dict[str, Path] = {}
    for field in (
        "summary",
        "source_manifest",
        "scenario_records",
        "experiment_config",
        "audit_record",
    ):
        path = _project_path(upstream[field])
        if _file_sha256(path) != str(upstream[f"{field}_sha256"]):
            raise RuntimeError(f"capacity evidence hash mismatch: {field}")
        paths[field] = path
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    if bool(summary["experiment"]["quick"]):
        raise RuntimeError("R2 cannot use a quick capacity scan as evidence")
    if summary["interpretation"]["status"] != "REPRESENTATION_CAPACITY_FOUND":
        raise RuntimeError("capacity scan did not find a viable representation")
    if not bool(summary["interpretation"]["capacity_demonstrated"]):
        raise RuntimeError("capacity evidence no longer demonstrates capacity")
    if bool(summary["interpretation"]["s4d3_authorized"]):
        raise RuntimeError("S4-D3 must remain closed")
    if bool(summary["evidence_boundary"]["training_performed"]):
        raise RuntimeError("capacity evidence must remain non-learning")
    if summary["truth_alignment"]["status"] != "PASS":
        raise RuntimeError("capacity truth alignment did not pass")
    if "Verification Status: `ANALYZED`" not in paths["audit_record"].read_text(
        encoding="utf-8"
    ):
        raise RuntimeError("capacity audit is not ANALYZED")
    with paths["scenario_records"].open("r", encoding="utf-8", newline="") as handle:
        row_count = sum(1 for _ in csv.DictReader(handle))
    if row_count != 576:
        raise RuntimeError("capacity scenario record count changed")
    return summary


def _verify_action_and_learning_contract(
    experiment: dict[str, Any], capacity: dict[str, Any]
) -> None:
    action = experiment["action"]
    design = capacity["design"]
    if float(action["residual_action_limit_rad"]) != float(
        design["residual_component_limit_rad"]
    ):
        raise RuntimeError("R2 residual component limit differs from capacity scan")
    if float(action["final_action_step_limit_rad"]) != float(
        design["final_component_step_limit_rad"]
    ):
        raise RuntimeError("R2 final step limit differs from capacity scan")
    if not bool(action.get("preserve_total_phase_rms_budget", False)):
        raise RuntimeError("R2 must preserve the total phase RMS action budget")
    if int(action["total_budget_anchor_modes"]) != int(design["anchor_modes"]):
        raise RuntimeError("R2 total action budget anchor changed")
    if bool(experiment["policy_observation"]["include_oracle_quality_metrics"]):
        raise RuntimeError("oracle quality metrics must not enter the R2 policy")
    if int(experiment["sac"]["hidden_size"]) != 256:
        raise RuntimeError("R2 must keep the predeclared SAC hidden size")


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
                raise RuntimeError("R2 policy training seed ranges overlap")
    validation_seeds = {
        int(item["base_seed"]) + offset
        for item in settings["validation_physical_conditions"]
        for offset in range(int(settings["validation_batch_size"]))
    }
    if any(seed in training for training in training_ranges for seed in validation_seeds):
        raise RuntimeError("R2 training and validation seeds overlap")
    active = set(validation_seeds)
    for training in training_ranges:
        active.update(training)
    for protected in experiment["protected_seed_ranges"]:
        start = int(protected["start_inclusive"])
        end = int(protected["end_exclusive"])
        if any(start <= seed < end for seed in active):
            raise RuntimeError(f"R2 overlaps protected seeds: {protected['id']}")

    formal_ranges = [
        range(int(base), int(base) + int(experiment["training_seed_span_per_policy"]))
        for base in experiment["training_environment_seed_bases"]
    ]
    quick_range = range(
        int(experiment["quick"]["training_environment_seed_bases"][0]),
        int(experiment["quick"]["training_environment_seed_bases"][0])
        + int(experiment["training_seed_span_per_policy"]),
    )
    if any(set(item).intersection(quick_range) for item in formal_ranges):
        raise RuntimeError("R2 formal and quick training seed ranges overlap")
    if quick and len(settings["policy_seeds"]) != 1:
        raise RuntimeError("R2 quick smoke must use exactly one policy seed")
