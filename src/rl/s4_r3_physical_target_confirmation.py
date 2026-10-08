"""R3-D2-A5-R1固定16步纯物理目标的独立回合确认。"""

from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import torch

from src.rl.s4_r3_critic_source_diagnostic import _load_extended_policy
from src.rl.s4_r3_failure_diagnostic import _load_student
from src.rl.s4_r3_multistep_critic_training import (
    ProbeDataset,
    _collect_policy_split,
    _effective_settings as _repair_effective_settings,
    preflight_s4_r3_multistep_critic_training,
)
from src.rl.s4_r3_physical_target_stability import (
    horizon_stability_record,
    load_physical_target_pairs,
    move_physical_target_pairs,
    stability_gate_checks,
)
from src.rl.s4_representation_capacity import (
    ActionRepresentation,
    build_action_basis,
)
from src.rl.s4_training import (
    _file_sha256,
    _git_record,
    _load_yaml,
    _project_path,
    _relative,
    _runtime_record,
    _source_manifest,
    _write_json,
    json_safe,
)
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.training_progress import counted_progress, update_progress


REQUIRED_CANDIDATES = ("zero", "actor")
REQUIRED_STRATA = ("all", "high_magnitude")


def run_s4_r3_physical_target_confirmation(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """生成独立配对回合并确认固定的16步纯物理目标。"""
    experiment_path = _project_path(config_path)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight, d1_experiment, checkpoints = preflight_s4_r3_physical_target_confirmation(
        experiment_path,
        experiment,
        settings,
        quick=quick,
    )
    if preflight_only:
        return preflight

    device = resolve_device(str(experiment["runtime"]["device"]))
    if bool(experiment["runtime"]["deterministic_algorithms"]):
        torch.use_deterministic_algorithms(True)

    output_directory = _project_path(settings["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=False)
    dataset_directory = output_directory / "datasets"
    dataset_directory.mkdir()
    _write_json(output_directory / "preflight.json", preflight)
    _write_json(output_directory / "effective_config.json", settings)
    source_manifest = _source_manifest(experiment["tracked_source_files"])
    _write_json(output_directory / "source_manifest.json", source_manifest)

    base_config, _ = load_s1_config(_project_path(d1_experiment["environment_config"]))
    representation = ActionRepresentation.from_mapping(
        d1_experiment["representation"]
    )
    base_config = replace(base_config, num_modes=representation.num_modes)
    basis, _, basis_diagnostics = build_action_basis(
        base_config,
        representation,
        device,
    )

    started = time.perf_counter()
    collection_started = time.perf_counter()
    progress_path = output_directory / "collection_progress.jsonl"
    collection_bar = counted_progress(
        total=int(preflight["expected_branch_rollouts"]),
        description="A5-R1 独立物理回合",
        unit="分支",
    )
    records: list[dict[str, Any]] = []
    dataset_records: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        policy = _load_extended_policy(checkpoint, device=device)
        student = _load_student(
            checkpoint,
            experiment=d1_experiment,
            device=device,
        )
        dataset = _collect_policy_split(
            split_name="independent_confirmation",
            split=settings["confirmation_split"],
            experiment=d1_experiment,
            base_config=base_config,
            basis=basis,
            policy=policy,
            student=student,
            candidate_actions=settings["candidate_actions"],
            max_horizon=int(settings["maximum_horizon"]),
            collection_bar=collection_bar,
            collection_progress_path=progress_path,
            collection_started=collection_started,
            device=device,
        )
        dataset.validate(max_horizon=int(settings["maximum_horizon"]))
        dataset_path = (
            dataset_directory
            / f"seed_{policy.policy_seed}_independent_confirmation.pt"
        )
        torch.save(dataset.payload(), dataset_path)
        pair_dataset = load_physical_target_pairs(
            {"path": _relative(dataset_path)},
            maximum_horizon=int(settings["maximum_horizon"]),
            state_size=int(settings["state_size"]),
            action_size=int(settings["action_size"]),
        )
        pair_dataset = move_physical_target_pairs(pair_dataset, device)
        dataset_records.append(
            _dataset_record(
                policy_seed=policy.policy_seed,
                dataset=dataset,
                pair_count=len(pair_dataset.rows),
                path=dataset_path,
            )
        )
        for stratum in settings["strata"]:
            record = horizon_stability_record(
                pair_dataset,
                policy_seed=policy.policy_seed,
                split="independent_confirmation",
                horizon=int(settings["fixed_horizon"]),
                stratum=stratum,
                magnitude_threshold=float(
                    settings["magnitude_thresholds"][policy.policy_seed]
                ),
                terminal_horizon=int(settings["terminal_horizon"]),
                local_offsets=settings["local_offsets"],
                thresholds=settings,
                device=device,
            )
            checks = confirmation_gate_checks(
                record,
                thresholds=settings,
                stratum=stratum,
            )
            record["gate"] = "PASS" if all(checks.values()) else "FAIL"
            record["gate_checks"] = json.dumps(
                checks,
                ensure_ascii=False,
                sort_keys=True,
            )
            record["class_balance_gate_applied"] = stratum in set(
                settings["class_balance_gate_strata"]
            )
            record["gate_revision"] = (
                "class_balance_only_on_all_samples"
            )
            records.append(record)
            update_progress(
                collection_bar,
                device=device,
                metrics={
                    "种子": float(policy.policy_seed),
                    "同号": float(record["reward_power_sign_agreement"]),
                    "门槛": 1.0 if record["gate"] == "PASS" else 0.0,
                },
            )
    collection_bar.close()

    decision = independent_confirmation_decision(
        records,
        policy_seeds=settings["policy_seeds"],
        strata=settings["strata"],
        quick=quick,
    )
    metrics_path = output_directory / "confirmation_metrics.csv"
    decision_path = output_directory / "decision.json"
    _write_rows(metrics_path, records)
    _write_json(decision_path, decision)
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": str(experiment["metadata"]["version_label"]),
        },
        "experiment": {
            "id": str(experiment["metadata"]["experiment_id"]),
            "status": "quick_smoke_only" if quick else "completed_pending_audit",
            "quick": quick,
            "duration_seconds": time.perf_counter() - started,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "fixed_confirmation_contract": {
            "horizon": settings["fixed_horizon"],
            "terminal_horizon": settings["terminal_horizon"],
            "local_offsets": settings["local_offsets"],
            "strata": settings["strata"],
            "class_balance_gate_strata": settings[
                "class_balance_gate_strata"
            ],
            "magnitude_thresholds_from_a5_development": settings[
                "magnitude_thresholds"
            ],
            "horizon_reselection": False,
        },
        "independence": {
            "new_complete_turbulence_episodes": True,
            "confirmation_condition_ids": [
                item["id"] for item in settings["confirmation_split"]["conditions"]
            ],
            "confirmation_base_seeds": [
                int(item["base_seed"])
                for item in settings["confirmation_split"]["conditions"]
            ],
            "upstream_episode_seed_overlap": 0,
            "thresholds_reestimated_on_confirmation": False,
        },
        "basis_diagnostics": basis_diagnostics,
        "datasets": dataset_records,
        "decision": decision,
        "records": {
            "collection_progress": _relative(progress_path),
            "collection_progress_sha256": _file_sha256(progress_path),
            "confirmation_metrics": _relative(metrics_path),
            "confirmation_metrics_sha256": _file_sha256(metrics_path),
            "confirmation_metric_rows": len(records),
            "decision": _relative(decision_path),
            "decision_sha256": _file_sha256(decision_path),
        },
        "inputs": {
            "experiment_config": _relative(experiment_path),
            "experiment_config_sha256": _file_sha256(experiment_path),
            "upstream_a5_summary": experiment["upstream_a5"]["summary"],
            "upstream_a5_summary_sha256": experiment["upstream_a5"][
                "summary_sha256"
            ],
            "frozen_source_experiment": experiment["frozen_source"][
                "experiment_config"
            ],
            "frozen_checkpoints": [
                {
                    "policy_seed": int(item["policy_seed"]),
                    "path": item["path"],
                    "sha256": item["sha256"],
                }
                for item in checkpoints
            ],
            "source_manifest": source_manifest,
        },
        "evidence_boundary": {
            "new_simulation_episodes_generated": True,
            "gradient_updates": 0,
            "model_training": False,
            "actor_updates": 0,
            "student_updates": 0,
            "reward_changed": False,
            "action_budget_changed": False,
            "mechanism_audit_accessed": False,
            "full_rl_trained": False,
            "s4d3_accessed": False,
            "real_slm_actions": False,
        },
        "runtime": _runtime_record(),
        "git": _git_record(),
        "next_action": (
            "Quick output is software evidence only."
            if quick
            else "Stop for read-only audit; do not train RL or rerun automatically."
        ),
    }
    _write_json(output_directory / "summary.json", summary)
    return json_safe(summary)


def preflight_s4_r3_physical_target_confirmation(
    experiment_path: Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """锁定A5-R1的事先规则、独立种子和只读策略来源。"""
    metadata = experiment["metadata"]
    if str(metadata["stage"]) != "S4-D2-R3-D2-A5-R1":
        raise ValueError("A5-R1 metadata stage changed")
    if not bool(metadata["diagnostic_only"]):
        raise RuntimeError("A5-R1 must remain diagnostic-only")
    if not bool(metadata["allow_new_episode_generation"]):
        raise RuntimeError("A5-R1 must explicitly allow new simulation episodes")
    required_false = (
        "allow_gradient_updates",
        "allow_model_training",
        "allow_actor_updates",
        "allow_student_updates",
        "allow_reward_change",
        "allow_action_budget_change",
        "allow_horizon_reselection",
        "allow_mechanism_audit_access",
        "allow_s4d3_access",
        "allow_real_hardware_actions",
    )
    for field in required_false:
        if bool(metadata[field]):
            raise RuntimeError(f"A5-R1 safety guard was relaxed: {field}")
    if not bool(experiment["runtime"]["require_cuda"]):
        raise RuntimeError("A5-R1 requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("A5-R1 requires an available CUDA device")

    for field in ("audit_record", "plan"):
        _verify_hash(
            _project_path(experiment["design_contract"][field]),
            str(experiment["design_contract"][f"{field}_sha256"]),
        )
    for field in (
        "summary",
        "preflight",
        "effective_config",
        "source_manifest",
        "horizon_metrics",
        "selection",
    ):
        _verify_hash(
            _project_path(experiment["upstream_a5"][field]),
            str(experiment["upstream_a5"][f"{field}_sha256"]),
        )
    a5_summary = json.loads(
        _project_path(experiment["upstream_a5"]["summary"]).read_text(
            encoding="utf-8"
        )
    )
    if str(a5_summary["selection"]["status"]) != str(
        experiment["upstream_a5"]["required_status"]
    ):
        raise RuntimeError("A5-R1 upstream A5 status changed")
    if bool(a5_summary["experiment"]["quick"]):
        raise RuntimeError("A5-R1 cannot consume quick A5 evidence")
    if bool(a5_summary["evidence_boundary"]["new_physical_episodes_generated"]):
        raise RuntimeError("A5 upstream evidence boundary changed")

    if settings["fixed_horizon"] != 16:
        raise RuntimeError("A5-R1 fixed horizon changed")
    if settings["terminal_horizon"] != 32:
        raise RuntimeError("A5-R1 terminal horizon changed")
    if settings["local_offsets"] != [1, 2, 4]:
        raise RuntimeError("A5-R1 local offsets changed")
    if settings["strata"] != list(REQUIRED_STRATA):
        raise RuntimeError("A5-R1 strata changed")
    if settings["class_balance_gate_strata"] != ["all"]:
        raise RuntimeError("A5-R1 class-balance gate scope changed")
    if tuple(item["id"] for item in settings["candidate_actions"]) != REQUIRED_CANDIDATES:
        raise RuntimeError("A5-R1 candidate actions changed")
    if max(settings["confirmation_split"]["probe_steps"]) + settings[
        "maximum_horizon"
    ] > settings["confirmation_split"]["episode_length_steps"]:
        raise RuntimeError("A5-R1 probe exceeds the complete episode")

    if not quick:
        if settings["policy_seeds"] != [9301, 9302, 9303]:
            raise RuntimeError("A5-R1 formal policy seeds changed")
        if len(settings["confirmation_split"]["profile_ids"]) != 6:
            raise RuntimeError("A5-R1 formal profile coverage changed")
        if len(settings["confirmation_split"]["conditions"]) != 3:
            raise RuntimeError("A5-R1 formal condition coverage changed")
        if settings["confirmation_split"]["probe_steps"] != [0, 80, 160]:
            raise RuntimeError("A5-R1 formal probe steps changed")
        if settings["confirmation_split"]["episodes_per_condition"] != 16:
            raise RuntimeError("A5-R1 formal episode count changed")
        _verify_frozen_magnitude_thresholds(a5_summary, settings)

    frozen_source = experiment["frozen_source"]
    repair_path = _project_path(frozen_source["experiment_config"])
    _verify_hash(repair_path, str(frozen_source["experiment_config_sha256"]))
    repair_experiment = _load_yaml(repair_path)
    repair_settings = _repair_effective_settings(repair_experiment, quick=quick)
    _, d1_experiment, checkpoints = preflight_s4_r3_multistep_critic_training(
        repair_path,
        repair_experiment,
        repair_settings,
        quick=quick,
    )
    wanted = set(settings["policy_seeds"])
    checkpoints = [
        item
        for item in checkpoints
        if int(item["policy_seed"]) in wanted
        and str(item["arm"]) == str(frozen_source["required_policy_arm"])
    ]
    if len(checkpoints) != len(wanted):
        raise RuntimeError("A5-R1 requires one frozen actor per policy seed")

    independence = _verify_independent_seed_namespace(
        settings["confirmation_split"],
        repair_experiment=repair_experiment,
        repair_settings=repair_settings,
        reserved=int(experiment["confirmation_data"]["reserved_s4d3_seed_base"]),
    )
    missing = [
        item
        for item in experiment["tracked_source_files"]
        if not _project_path(item).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"A5-R1 tracked source files missing: {missing}")

    split = settings["confirmation_split"]
    expected_branches = (
        len(checkpoints)
        * len(split["profile_ids"])
        * len(split["conditions"])
        * len(split["probe_steps"])
        * len(settings["candidate_actions"])
    )
    expected_samples = expected_branches * int(split["episodes_per_condition"])
    output = _project_path(settings["output_directory"])
    if output.exists():
        raise FileExistsError(f"A5-R1 output already exists: {output}")
    preflight = {
        "status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_CONFIRMATION",
        "quick": quick,
        "fixed_horizon": settings["fixed_horizon"],
        "policy_seeds": settings["policy_seeds"],
        "profile_count": len(split["profile_ids"]),
        "condition_count": len(split["conditions"]),
        "probe_steps": split["probe_steps"],
        "episodes_per_condition": split["episodes_per_condition"],
        "expected_branch_rollouts": expected_branches,
        "expected_samples": expected_samples,
        "expected_action_pairs": expected_samples // 2,
        "expected_metric_rows": len(settings["policy_seeds"])
        * len(settings["strata"]),
        "new_episode_generation": True,
        "gradient_updates": 0,
        "model_training": False,
        "actor_updates": 0,
        "student_updates": 0,
        "horizon_reselection": False,
        "class_balance_gate_strata": settings["class_balance_gate_strata"],
        "seed_namespace": independence,
        "cuda_required": True,
        "output_directory": _relative(output),
        "s4d3_access": False,
        "real_slm_actions": False,
        "formal_command": (
            ".\\.venv\\Scripts\\python.exe "
            "scripts\\confirm_s4_r3_physical_target.py --config "
            f"{str(_relative(experiment_path)).replace('/', chr(92))}"
        ),
    }
    return preflight, d1_experiment, checkpoints


def confirmation_gate_checks(
    record: dict[str, Any],
    *,
    thresholds: dict[str, Any],
    stratum: str,
) -> dict[str, bool]:
    """把类别比例门槛只用于总体样本。"""
    if stratum not in REQUIRED_STRATA:
        raise ValueError(f"unknown A5-R1 stratum: {stratum}")
    checks = stability_gate_checks(record, thresholds=thresholds)
    if stratum not in set(thresholds["class_balance_gate_strata"]):
        checks.pop("class_balance")
    return checks


def independent_confirmation_decision(
    records: list[dict[str, Any]],
    *,
    policy_seeds: list[int],
    strata: list[str],
    quick: bool,
) -> dict[str, Any]:
    """正式确认只允许固定16步的六行全通过。"""
    if quick:
        return {
            "status": "QUICK_SMOKE_ONLY",
            "required_rows": len(policy_seeds) * len(strata),
            "observed_rows": len(records),
            "all_rows_pass": False,
            "supervised_probe_design_authorized": False,
            "full_rl_authorized": False,
            "s4d3_authorized": False,
            "real_hardware_authorized": False,
        }
    required_keys = {(seed, stratum) for seed in policy_seeds for stratum in strata}
    observed_keys = {
        (int(item["policy_seed"]), str(item["stratum"])) for item in records
    }
    if len(records) != len(required_keys) or observed_keys != required_keys:
        raise RuntimeError("A5-R1 confirmation metric coverage is incomplete")
    passed = all(str(item["gate"]) == "PASS" for item in records)
    return {
        "status": (
            "PHYSICAL_H16_TARGET_INDEPENDENTLY_CONFIRMED"
            if passed
            else "PHYSICAL_H16_TARGET_NOT_CONFIRMED"
        ),
        "required_rows": len(required_keys),
        "observed_rows": len(records),
        "passed_rows": sum(str(item["gate"]) == "PASS" for item in records),
        "all_rows_pass": passed,
        "decision_rule": "fixed_h16_all_policy_seeds_and_both_strata_must_pass",
        "supervised_probe_design_authorized": passed,
        "full_rl_authorized": False,
        "s4d3_authorized": False,
        "real_hardware_authorized": False,
    }


def _effective_settings(
    experiment: dict[str, Any], *, quick: bool
) -> dict[str, Any]:
    data = experiment["confirmation_data"]
    target = experiment["fixed_target"]
    thresholds = experiment["diagnostic_thresholds"]
    settings: dict[str, Any] = {
        "output_directory": experiment["outputs"]["directory"],
        "policy_seeds": [int(value) for value in data["policy_seeds"]],
        "candidate_actions": deepcopy(experiment["candidate_actions"]),
        "maximum_horizon": int(data["maximum_horizon"]),
        "state_size": int(data["state_size"]),
        "action_size": int(data["action_size"]),
        "fixed_horizon": int(target["horizon"]),
        "terminal_horizon": int(target["terminal_horizon"]),
        "local_offsets": [int(value) for value in target["local_offsets"]],
        "strata": [str(value) for value in target["strata"]],
        "magnitude_thresholds": {
            int(seed): float(value)
            for seed, value in target[
                "magnitude_thresholds_from_a5_development"
            ].items()
        },
        "minimum_sign_agreement": float(
            thresholds["minimum_sign_agreement"]
        ),
        "minimum_balanced_accuracy": float(
            thresholds["minimum_balanced_accuracy"]
        ),
        "minimum_actor_better_fraction": float(
            thresholds["minimum_actor_better_fraction"]
        ),
        "maximum_actor_better_fraction": float(
            thresholds["maximum_actor_better_fraction"]
        ),
        "class_balance_gate_strata": [
            str(value) for value in thresholds["class_balance_gate_strata"]
        ],
        "confirmation_split": {
            "profile_ids": list(data["profile_ids"]),
            "conditions": deepcopy(data["conditions"]),
            "probe_steps": [int(value) for value in data["probe_steps"]],
            "episodes_per_condition": int(data["episodes_per_condition"]),
            "episode_length_steps": int(data["episode_length_steps"]),
        },
    }
    if quick:
        quick_config = experiment["quick"]
        settings["output_directory"] = experiment["outputs"]["quick_directory"]
        settings["policy_seeds"] = [
            int(value) for value in quick_config["policy_seeds"]
        ]
        settings["confirmation_split"] = {
            "profile_ids": list(quick_config["profile_ids"]),
            "conditions": deepcopy(quick_config["conditions"]),
            "probe_steps": [int(value) for value in quick_config["probe_steps"]],
            "episodes_per_condition": int(
                quick_config["episodes_per_condition"]
            ),
            "episode_length_steps": int(quick_config["episode_length_steps"]),
        }
        settings["magnitude_thresholds"] = {
            int(seed): float(value)
            for seed, value in quick_config["magnitude_thresholds"].items()
        }
    return settings


def _verify_frozen_magnitude_thresholds(
    a5_summary: dict[str, Any], settings: dict[str, Any]
) -> None:
    upstream = a5_summary["scan_contract"][
        "magnitude_thresholds_from_development"
    ]
    for seed in settings["policy_seeds"]:
        expected = float(upstream[str(seed)][str(settings["fixed_horizon"])])
        actual = float(settings["magnitude_thresholds"][seed])
        if abs(actual - expected) > 1e-15:
            raise RuntimeError("A5-R1 frozen magnitude threshold changed")


def _verify_independent_seed_namespace(
    split: dict[str, Any],
    *,
    repair_experiment: dict[str, Any],
    repair_settings: dict[str, Any],
    reserved: int,
) -> dict[str, Any]:
    new_seeds: set[int] = set()
    for condition in split["conditions"]:
        base = int(condition["base_seed"])
        seeds = set(range(base, base + int(split["episodes_per_condition"])))
        if new_seeds & seeds:
            raise RuntimeError("A5-R1 confirmation seed ranges overlap")
        if max(seeds) >= reserved:
            raise RuntimeError("A5-R1 entered the reserved S4-D3 seed namespace")
        new_seeds.update(seeds)

    upstream_seeds: set[int] = set()
    for upstream_split in repair_settings["splits"].values():
        count = int(upstream_split["episodes_per_condition"])
        for condition in upstream_split["conditions"]:
            base = int(condition["base_seed"])
            upstream_seeds.update(range(base, base + count))
    quick = repair_experiment["quick"]
    for split_name in ("development", "validation", "mechanism_audit"):
        count = int(quick[split_name]["episodes_per_condition"])
        for condition in quick[split_name]["conditions"]:
            base = int(condition["base_seed"])
            upstream_seeds.update(range(base, base + count))
    overlap = sorted(new_seeds & upstream_seeds)
    if overlap:
        raise RuntimeError("A5-R1 confirmation seeds overlap upstream episodes")
    return {
        "minimum_episode_seed": min(new_seeds),
        "maximum_episode_seed": max(new_seeds),
        "unique_episode_seeds": len(new_seeds),
        "upstream_overlap": 0,
        "reserved_s4d3_seed_base": reserved,
    }


def _dataset_record(
    *,
    policy_seed: int,
    dataset: ProbeDataset,
    pair_count: int,
    path: Path,
) -> dict[str, Any]:
    return {
        "policy_seed": int(policy_seed),
        "split": "independent_confirmation",
        "path": _relative(path),
        "sha256": _file_sha256(path),
        "samples": int(dataset.states.shape[0]),
        "pairs": int(pair_count),
        "state_size": int(dataset.states.shape[1]),
        "action_size": int(dataset.actions.shape[1]),
        "target_horizons": int(dataset.n_step_targets.shape[1]),
        "episode_seeds": len({int(row["episode_seed"]) for row in dataset.rows}),
    }


def _verify_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if _file_sha256(path) != expected:
        raise RuntimeError(f"A5-R1 hash mismatch: {path}")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("A5-R1 confirmation records must not be empty")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

