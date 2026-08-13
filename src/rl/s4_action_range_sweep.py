"""S4-D2-R1理想动作范围扫描；不训练强化学习。"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Iterable

import torch

from src.rl.s4_oracle_bound import (
    _collect_metrics,
    _concatenate_metrics,
    _future_disturbance_modal_sequence,
    _load_json,
    _paired_summary,
    _record_progress,
    _rollout_baseline_and_modal_ceiling,
    _rollout_oracle_preview,
    _science_context_summary,
    _verify_matching_physics,
    select_hindsight_envelope,
)
from src.rl.s4_training import (
    _file_sha256,
    _load_yaml,
    _make_residual_controller,
    _profiles,
    _project_path,
    _relative,
    _runtime_record,
    _source_manifest,
    _write_json,
    _git_record,
    json_safe,
)
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.robust_control import RobustnessCondition
from src.training_progress import advance_to, counted_progress, update_progress


def run_s4_r1_action_range_sweep(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """扫描预声明残差范围；不训练、不读旧轨迹、不访问S4-D3。"""
    experiment_path = _project_path(config_path)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight = preflight_s4_r1_action_range_sweep(
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
            "action-range output already exists; preserve it and audit before retrying: "
            f"{output_directory}"
        )
    output_directory.mkdir(parents=True)
    _write_json(output_directory / "preflight.json", preflight)
    _write_json(output_directory / "effective_config.json", settings)
    _write_json(
        output_directory / "source_manifest.json",
        _source_manifest(experiment["tracked_source_files"]),
    )

    base_config, _ = load_s1_config(_project_path(experiment["environment_config"]))
    horizons = list(map(int, settings["preview_horizons_frames"]))
    limits = list(map(float, settings["residual_action_limits_rad"]))
    base_config = replace(
        base_config,
        batch_size=int(settings["batch_size"]),
        episode_length=max(base_config.episode_length, int(settings["steps"]) + max(horizons)),
    )
    profiles = _profiles(experiment, settings["profile_ids"])
    conditions = [
        RobustnessCondition.from_mapping(item)
        for item in settings["physical_conditions"]
    ]
    scenario_count = len(profiles) * len(conditions)
    total_rollouts = scenario_count * (1 + len(limits) * len(horizons))
    progress = counted_progress(
        total=total_rollouts,
        description="S4-D2-R1理想动作范围扫描",
        unit="轨迹组",
    )
    progress_path = output_directory / "progress.jsonl"
    completed = 0
    started = time.perf_counter()

    exact_candidates = _limit_metric_store(limits)
    exact_baselines = _limit_metric_store(limits)
    cumulative_candidates = _limit_metric_store(limits)
    cumulative_baselines = _limit_metric_store(limits)
    profile_cumulative_candidates = _profile_limit_metric_store(limits)
    profile_cumulative_baselines = _profile_limit_metric_store(limits)
    modal_ceiling: dict[str, list[torch.Tensor]] = defaultdict(list)
    modal_ceiling_baseline: dict[str, list[torch.Tensor]] = defaultdict(list)
    scenario_records: list[dict[str, Any]] = []
    truth_alignment_max_abs = 0.0

    for profile in profiles:
        for condition in conditions:
            config = profile.environment_config(condition.environment_config(base_config))
            future_truth = _future_disturbance_modal_sequence(
                config=config,
                condition=condition,
                profile=profile,
                length=int(settings["steps"]) + max(horizons),
                device=device,
            )
            baseline, ceiling = _rollout_baseline_and_modal_ceiling(
                experiment=_experiment_for_limit(experiment, limits[0]),
                config=config,
                condition=condition,
                profile=profile,
                steps=int(settings["steps"]),
                device=device,
            )
            _collect_metrics(modal_ceiling, ceiling)
            _collect_metrics(modal_ceiling_baseline, baseline)
            completed += 1
            _record_progress(
                progress_path,
                completed=completed,
                total=total_rollouts,
                profile=profile.identifier,
                condition=condition.identifier,
                controller="frozen_baseline",
                power=float(baseline["power_in_bucket"].mean()),
                started=started,
            )
            advance_to(progress, completed)
            update_progress(
                progress,
                device=device,
                metrics={"最近功率": float(baseline["power_in_bucket"].mean())},
            )

            exact_envelopes: dict[float, dict[str, torch.Tensor]] = {}
            for limit in limits:
                limit_experiment = _experiment_for_limit(experiment, limit)
                horizon_results: dict[int, dict[str, torch.Tensor]] = {}
                for horizon in horizons:
                    candidate, alignment = _rollout_oracle_preview(
                        experiment=limit_experiment,
                        config=config,
                        condition=condition,
                        profile=profile,
                        steps=int(settings["steps"]),
                        preview_horizon_frames=horizon,
                        future_disturbance_modal=future_truth,
                        device=device,
                    )
                    truth_alignment_max_abs = max(truth_alignment_max_abs, alignment)
                    _add_selected_limit(candidate, limit)
                    horizon_results[horizon] = candidate
                    scenario_records.append(
                        _scenario_record(
                            controller=f"limit_{limit:g}_preview_{horizon}",
                            profile=profile.identifier,
                            condition=condition.identifier,
                            candidate=candidate,
                            baseline=baseline,
                            allowed_limit=limit,
                        )
                    )
                    completed += 1
                    _record_progress(
                        progress_path,
                        completed=completed,
                        total=total_rollouts,
                        profile=profile.identifier,
                        condition=condition.identifier,
                        controller=f"limit_{limit:g}_preview_{horizon}",
                        power=float(candidate["power_in_bucket"].mean()),
                        started=started,
                    )
                    advance_to(progress, completed)
                    update_progress(
                        progress,
                        device=device,
                        metrics={
                            "动作上限": limit,
                            "预见帧": float(horizon),
                            "功率差": float(
                                (
                                    candidate["power_in_bucket"]
                                    - baseline["power_in_bucket"]
                                ).mean()
                            ),
                        },
                    )

                exact = select_hindsight_envelope(horizon_results)
                _add_selected_limit(exact, limit)
                exact_envelopes[limit] = exact
                _collect_metrics(exact_candidates[limit], exact)
                _collect_metrics(exact_baselines[limit], baseline)
                scenario_records.append(
                    _scenario_record(
                        controller=f"limit_{limit:g}_exact_hindsight",
                        profile=profile.identifier,
                        condition=condition.identifier,
                        candidate=exact,
                        baseline=baseline,
                        allowed_limit=limit,
                    )
                )

            for allowed_limit in limits:
                eligible = {
                    limit: exact_envelopes[limit]
                    for limit in limits
                    if limit <= allowed_limit
                }
                cumulative = select_limit_envelope(eligible)
                _collect_metrics(cumulative_candidates[allowed_limit], cumulative)
                _collect_metrics(cumulative_baselines[allowed_limit], baseline)
                _collect_metrics(
                    profile_cumulative_candidates[allowed_limit][profile.identifier],
                    cumulative,
                )
                _collect_metrics(
                    profile_cumulative_baselines[allowed_limit][profile.identifier],
                    baseline,
                )
                scenario_records.append(
                    _scenario_record(
                        controller=f"limit_{allowed_limit:g}_cumulative_hindsight",
                        profile=profile.identifier,
                        condition=condition.identifier,
                        candidate=cumulative,
                        baseline=baseline,
                        allowed_limit=allowed_limit,
                    )
                )
    progress.close()

    exact_summaries = [
        {
            "residual_action_limit_rad": limit,
            "overall": _range_paired_summary(
                f"exact_limit_{limit:g}",
                _concatenate_metrics(exact_candidates[limit]),
                _concatenate_metrics(exact_baselines[limit]),
                settings["gate"],
            ),
        }
        for limit in limits
    ]
    cumulative_summaries = []
    for allowed_limit in limits:
        overall = _range_paired_summary(
            f"cumulative_limit_{allowed_limit:g}",
            _concatenate_metrics(cumulative_candidates[allowed_limit]),
            _concatenate_metrics(cumulative_baselines[allowed_limit]),
            settings["gate"],
        )
        profile_summaries = [
            _range_paired_summary(
                profile.identifier,
                _concatenate_metrics(
                    profile_cumulative_candidates[allowed_limit][profile.identifier]
                ),
                _concatenate_metrics(
                    profile_cumulative_baselines[allowed_limit][profile.identifier]
                ),
                settings["gate"],
            )
            for profile in profiles
        ]
        all_profiles_pass = all(
            item["capacity_gate"] == "PASS" for item in profile_summaries
        )
        cumulative_summaries.append(
            {
                "maximum_allowed_residual_action_rad": allowed_limit,
                "eligible_residual_action_limits_rad": [
                    item for item in limits if item <= allowed_limit
                ],
                "overall": overall,
                "profiles": profile_summaries,
                "all_profiles_pass": all_profiles_pass,
                "capacity_gate": (
                    "PASS"
                    if overall["capacity_gate"] == "PASS" and all_profiles_pass
                    else "FAIL"
                ),
            }
        )

    diagnostic_limit = find_minimum_demonstrated_limit(cumulative_summaries)
    formal_limit = None if quick else diagnostic_limit
    status = (
        "QUICK_SMOKE_ONLY"
        if quick
        else (
            "ACTION_RANGE_FOUND"
            if formal_limit is not None
            else "NO_RANGE_DEMONSTRATED"
        )
    )
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run-diagnostic",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "s4d2_r1_action_range_sweep_v1",
        },
        "experiment": {
            "id": "AO-S4-D2-R1-ACTION-RANGE-SWEEP",
            "type": "software_only_cuda_oracle_action_range_diagnostic",
            "status": "completed_pending_audit",
            "quick": quick,
            "duration_seconds": time.perf_counter() - started,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "evidence_boundary": {
            "training_performed": False,
            "optimizer_updates": 0,
            "future_simulator_truth_accessed": True,
            "preview_hindsight_selection": True,
            "action_limit_hindsight_selection": True,
            "deployable_controller": False,
            "mathematical_global_optimum": False,
            "old_trajectories_used": False,
            "sealed_s4d3_accessed": False,
            "real_slm_actions": False,
            "interpretation": (
                "This is a nested empirical optimistic envelope over declared action "
                "limits and preview horizons, not a deployable controller or a global optimum."
            ),
        },
        "inputs": {
            "experiment_config": _relative(experiment_path),
            "experiment_config_sha256": _file_sha256(experiment_path),
            "upstream_oracle_summary_sha256": preflight[
                "upstream_oracle_summary_sha256"
            ],
            "upstream_oracle_audit_sha256": preflight[
                "upstream_oracle_audit_sha256"
            ],
        },
        "design": {
            "paired_episode_seeds": True,
            "nested_cumulative_limit_envelope": True,
            "same_hardware_chain_as_r1": True,
            "residual_action_limits_rad": limits,
            "final_action_step_limit_rad": float(
                experiment["action"]["final_action_step_limit_rad"]
            ),
            "preview_horizons_frames": horizons,
            "selection_metric": "power_in_bucket",
            "profiles": list(settings["profile_ids"]),
            "physical_conditions": deepcopy(settings["physical_conditions"]),
            "episodes_per_physical_condition": int(settings["batch_size"]),
            "steps": int(settings["steps"]),
            "progress_rollouts": total_rollouts,
        },
        "truth_alignment": {
            "max_absolute_modal_difference": truth_alignment_max_abs,
            "tolerance": float(settings["truth_alignment_tolerance"]),
            "status": (
                "PASS"
                if truth_alignment_max_abs
                <= float(settings["truth_alignment_tolerance"])
                else "FAIL"
            ),
        },
        "exact_limit_hindsight_envelopes": exact_summaries,
        "cumulative_limit_hindsight_envelopes": cumulative_summaries,
        "absolute_modal_ceiling_context": _science_context_summary(
            "unconstrained_instantaneous_modal_ceiling",
            _concatenate_metrics(modal_ceiling),
            _concatenate_metrics(modal_ceiling_baseline),
        ),
        "interpretation": {
            "status": status,
            "diagnostic_first_passing_limit_rad": diagnostic_limit,
            "minimum_demonstrated_limit_rad": formal_limit,
            "capacity_demonstrated": formal_limit is not None,
            "r2_authorized": False,
            "s4d3_authorized": False,
            "next_rule": (
                "Audit first. A formal passing range only supports designing an R2 "
                "candidate inside that range; it does not authorize training or S4-D3."
            ),
        },
        "runtime": _runtime_record(),
        "git": _git_record(),
        "next_action": (
            "Stop and ask the assistant to audit the action-range sweep. "
            "Do not start R2 or open S4-D3."
        ),
    }
    _write_json(output_directory / "summary.json", summary)
    _write_scenario_csv(output_directory / "scenario_records.csv", scenario_records)
    return summary


def preflight_s4_r1_action_range_sweep(
    experiment_path: Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    """在CUDA和输出创建前锁定上游负结果、扫描范围与种子。"""
    metadata = experiment["metadata"]
    if str(metadata["stage"]) != "S4-D2-R1-ACTION-RANGE":
        raise ValueError("action-range metadata must identify S4-D2-R1-ACTION-RANGE")
    forbidden = (
        "allow_training",
        "allow_optimizer_updates",
        "allow_checkpoint_updates",
        "allow_old_trajectory_access",
        "allow_s4d3_access",
        "allow_real_hardware_actions",
    )
    if any(bool(metadata.get(field, True)) for field in forbidden):
        raise RuntimeError("action-range mutation and protected-data flags must stay false")
    if not bool(metadata.get("allow_simulator_truth_access", False)):
        raise RuntimeError("action-range sweep must declare simulator-truth access")

    upstream = _verify_oracle_failure(experiment["upstream_oracle"])
    oracle_config = upstream["config"]
    for field in ("frozen_controller", "policy_observation"):
        if experiment[field] != oracle_config[field]:
            raise RuntimeError(f"action-range sweep changed oracle field: {field}")
    for field in (
        "num_modes",
        "final_action_step_limit_rad",
        "preserve_environment_modal_limit",
    ):
        if experiment["action"][field] != oracle_config["action"][field]:
            raise RuntimeError(f"action-range sweep changed action field: {field}")
    if list(experiment["evaluation"]["profile_ids"]) != list(
        oracle_config["evaluation"]["profile_ids"]
    ):
        raise RuntimeError("action-range hardware profiles differ from oracle bound")
    for field in ("steps", "episodes_per_physical_condition"):
        if experiment["evaluation"][field] != oracle_config["evaluation"][field]:
            raise RuntimeError(f"action-range sweep changed evaluation field: {field}")
    _verify_matching_physics(
        oracle_config["evaluation"]["physical_conditions"],
        experiment["evaluation"]["physical_conditions"],
    )
    if experiment["gate"] != oracle_config["gate"]:
        raise RuntimeError("action-range sweep changed the original R1 gate")
    if list(experiment["oracle"]["preview_horizons_frames"]) != list(
        oracle_config["oracle"]["preview_horizons_frames"]
    ):
        raise RuntimeError("action-range sweep changed formal preview horizons")
    if (
        str(experiment["oracle"]["selection_metric"]) != "power_in_bucket"
        or not bool(experiment["oracle"]["per_episode_hindsight_envelope"])
        or not bool(experiment["oracle"]["nested_cumulative_limit_envelope"])
    ):
        raise RuntimeError("action-range optimistic envelope settings changed")
    for field in ("compensate_phase_scale", "truth_alignment_tolerance"):
        if experiment["oracle"][field] != oracle_config["oracle"][field]:
            raise RuntimeError(f"action-range sweep changed oracle field: {field}")

    environment_path = _project_path(experiment["environment_config"])
    hardware_path = _project_path(experiment["hardware_profile_source"])
    if _file_sha256(environment_path) != str(experiment["environment_config_sha256"]):
        raise RuntimeError("environment config hash mismatch")
    if _file_sha256(hardware_path) != str(
        experiment["hardware_profile_source_sha256"]
    ):
        raise RuntimeError("hardware profile source hash mismatch")
    if experiment["environment_config_sha256"] != oracle_config[
        "environment_config_sha256"
    ] or experiment["hardware_profile_source_sha256"] != oracle_config[
        "hardware_profile_source_sha256"
    ]:
        raise RuntimeError("action-range environment or hardware source differs from oracle")

    formal_limits = _validate_limits(
        experiment["action"]["residual_action_limits_rad"],
        final_limit=float(experiment["action"]["final_action_step_limit_rad"]),
        require_anchors=True,
    )
    active_limits = _validate_limits(
        settings["residual_action_limits_rad"],
        final_limit=float(experiment["action"]["final_action_step_limit_rad"]),
        require_anchors=not quick,
    )
    if quick and any(limit not in formal_limits for limit in active_limits):
        raise RuntimeError("quick action limits must be a subset of formal limits")
    horizons = list(map(int, settings["preview_horizons_frames"]))
    if not horizons or horizons != sorted(set(horizons)) or horizons[0] < 0:
        raise ValueError("preview horizons must be sorted unique non-negative integers")
    if quick and any(
        horizon not in experiment["oracle"]["preview_horizons_frames"]
        for horizon in horizons
    ):
        raise RuntimeError("quick preview horizons must be a subset of formal horizons")

    base_config, _ = load_s1_config(environment_path)
    controller_experiment = _experiment_for_limit(experiment, active_limits[0])
    controller = _make_residual_controller(controller_experiment, base_config)
    if controller.state_size != 100 or base_config.num_modes != 10:
        raise RuntimeError("action-range residual controller dimensions changed")
    _validate_sweep_seeds(experiment, settings)

    output_directory = _project_path(settings["output_directory"])
    if output_directory.exists():
        raise FileExistsError(f"action-range output already exists: {output_directory}")
    if not all(_project_path(path).is_file() for path in experiment["tracked_source_files"]):
        raise FileNotFoundError("one or more action-range tracked source files are missing")
    scenario_count = len(settings["profile_ids"]) * len(settings["physical_conditions"])
    return {
        "status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_SIMULATION",
        "quick": quick,
        "experiment_config": _relative(experiment_path),
        "experiment_config_sha256": _file_sha256(experiment_path),
        "output_directory": _relative(output_directory),
        "base_environment_config": asdict(base_config),
        "state_size": controller.state_size,
        "action_size": base_config.num_modes,
        "residual_action_limits_rad": active_limits,
        "preview_horizons_frames": horizons,
        "scenario_count": scenario_count,
        "progress_rollouts": scenario_count * (1 + len(active_limits) * len(horizons)),
        "episodes_per_scenario": int(settings["batch_size"]),
        "steps": int(settings["steps"]),
        "upstream_oracle_gate": "FAIL",
        "upstream_oracle_summary_sha256": upstream["summary_sha256"],
        "upstream_oracle_audit_sha256": upstream["audit_sha256"],
        "seed_isolation_verified": True,
        "cuda_required": True,
        "training_allowed": False,
        "optimizer_updates": 0,
        "future_simulator_truth_accessed": True,
        "deployable_controller": False,
        "sealed_s4d3_accessed": False,
        "real_slm_actions": False,
        "formal_command": (
            ".\\.venv\\Scripts\\python.exe "
            "scripts\\run_s4_r1_action_range_sweep.py --config "
            "configs\\experiments\\s4_r1_action_range_sweep_v1.yaml"
        ),
    }


def select_limit_envelope(
    candidates: dict[float, dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """逐回合在所有不超过当前上限的候选中选桶内功率最高者。"""
    if not candidates:
        raise ValueError("limit envelope requires at least one candidate")
    limits = sorted(candidates)
    metric_keys = tuple(candidates[limits[0]])
    if any(tuple(candidates[item]) != metric_keys for item in limits[1:]):
        raise ValueError("action-limit candidates expose different metrics")
    powers = torch.stack(
        [candidates[limit]["power_in_bucket"] for limit in limits], dim=0
    )
    selected = powers.argmax(dim=0)
    result = {}
    for metric in metric_keys:
        stacked = torch.stack([candidates[limit][metric] for limit in limits], dim=0)
        result[metric] = stacked.gather(0, selected.unsqueeze(0)).squeeze(0)
    limit_values = torch.tensor(limits, dtype=powers.dtype, device=powers.device)
    result["selected_residual_action_limit_rad"] = limit_values[selected].cpu()
    return result


def find_minimum_demonstrated_limit(
    summaries: Iterable[dict[str, Any]],
) -> float | None:
    """返回第一个通过总体与全部档位门槛的累积动作上限。"""
    ordered = sorted(
        summaries,
        key=lambda item: float(item["maximum_allowed_residual_action_rad"]),
    )
    for item in ordered:
        if item["capacity_gate"] == "PASS":
            return float(item["maximum_allowed_residual_action_rad"])
    return None


def _verify_oracle_failure(upstream: dict[str, Any]) -> dict[str, Any]:
    paths: dict[str, Path] = {}
    for field in ("summary", "source_manifest", "experiment_config", "audit_record"):
        path = _project_path(upstream[field])
        if _file_sha256(path) != str(upstream[f"{field}_sha256"]):
            raise RuntimeError(f"oracle upstream hash mismatch: {field}")
        paths[field] = path
    summary = _load_json(paths["summary"])
    if (
        bool(summary["experiment"]["quick"])
        or summary["truth_alignment"]["status"] != "PASS"
        or summary["hindsight_envelope"]["capacity_gate"] != "FAIL"
        or summary["interpretation"]["status"] != "CAPACITY_NOT_DEMONSTRATED"
        or bool(summary["interpretation"]["capacity_demonstrated"])
        or bool(summary["interpretation"]["r2_authorized"])
        or bool(summary["interpretation"]["s4d3_authorized"])
        or bool(summary["evidence_boundary"]["sealed_s4d3_accessed"])
    ):
        raise RuntimeError("formal oracle failure state changed")
    audit = paths["audit_record"].read_text(encoding="utf-8")
    if "Verification Status: `ANALYZED`" not in audit or "门槛为 **FAIL**" not in audit:
        raise RuntimeError("oracle audit record is not an ANALYZED failure")
    recorded_manifest = _load_json(paths["source_manifest"])
    for relative, digest in recorded_manifest.items():
        source = _project_path(relative)
        if not source.is_file() or _file_sha256(source) != digest:
            raise RuntimeError(f"oracle tracked source changed: {relative}")
    return {
        "config": _load_yaml(paths["experiment_config"]),
        "summary_sha256": _file_sha256(paths["summary"]),
        "audit_sha256": _file_sha256(paths["audit_record"]),
    }


def _validate_limits(
    values: Iterable[float],
    *,
    final_limit: float,
    require_anchors: bool,
) -> list[float]:
    limits = list(map(float, values))
    if not limits or limits != sorted(set(limits)) or limits[0] <= 0:
        raise ValueError("action limits must be sorted unique positive values")
    if limits[-1] > final_limit:
        raise ValueError("residual action limit exceeds the final action step limit")
    if require_anchors and (
        not any(abs(item - 0.0125) < 1e-12 for item in limits)
        or not any(abs(item - 0.05) < 1e-12 for item in limits)
    ):
        raise ValueError("formal scan must include the R1 0.0125 and D2 0.05 anchors")
    return limits


def _validate_sweep_seeds(
    experiment: dict[str, Any], settings: dict[str, Any]
) -> None:
    batch = int(settings["batch_size"])
    active = {
        int(item["base_seed"]) + offset
        for item in settings["physical_conditions"]
        for offset in range(batch)
    }
    if len(active) != len(settings["physical_conditions"]) * batch:
        raise RuntimeError("action-range physical conditions reuse episode seeds")
    formal = {
        int(item["base_seed"]) + offset
        for item in experiment["evaluation"]["physical_conditions"]
        for offset in range(int(experiment["evaluation"]["episodes_per_physical_condition"]))
    }
    quick = {
        int(item["base_seed"]) + offset
        for item in experiment["quick"]["physical_conditions"]
        for offset in range(int(experiment["quick"]["batch_size"]))
    }
    if len(formal) != (
        len(experiment["evaluation"]["physical_conditions"])
        * int(experiment["evaluation"]["episodes_per_physical_condition"])
    ) or len(quick) != (
        len(experiment["quick"]["physical_conditions"])
        * int(experiment["quick"]["batch_size"])
    ):
        raise RuntimeError("action-range declared conditions reuse episode seeds")
    declared = formal | quick
    for item in experiment["protected_seed_ranges"]:
        start = int(item["start_inclusive"])
        end = int(item["end_exclusive"])
        if any(start <= seed < end for seed in declared):
            raise RuntimeError(f"action-range seeds overlap protected range: {item['id']}")
    reserved = int(experiment["reserved_future_algorithm_seed_base"])
    if any(seed >= reserved for seed in declared):
        raise RuntimeError("action-range overlaps the reserved future algorithm namespace")
    if formal & quick:
        raise RuntimeError("action-range formal and quick seeds overlap")


def _effective_settings(
    experiment: dict[str, Any], *, quick: bool
) -> dict[str, Any]:
    evaluation = experiment["evaluation"]
    settings = {
        "quick": quick,
        "output_directory": experiment["outputs"]["directory"],
        "profile_ids": list(evaluation["profile_ids"]),
        "physical_conditions": deepcopy(evaluation["physical_conditions"]),
        "batch_size": int(evaluation["episodes_per_physical_condition"]),
        "steps": int(evaluation["steps"]),
        "residual_action_limits_rad": list(
            experiment["action"]["residual_action_limits_rad"]
        ),
        "preview_horizons_frames": list(
            experiment["oracle"]["preview_horizons_frames"]
        ),
        "truth_alignment_tolerance": float(
            experiment["oracle"]["truth_alignment_tolerance"]
        ),
        "gate": deepcopy(experiment["gate"]),
    }
    if quick:
        quick_settings = experiment["quick"]
        settings.update(
            {
                "output_directory": experiment["outputs"]["quick_directory"],
                "profile_ids": list(quick_settings["profile_ids"]),
                "physical_conditions": deepcopy(quick_settings["physical_conditions"]),
                "batch_size": int(quick_settings["batch_size"]),
                "steps": int(quick_settings["steps"]),
                "residual_action_limits_rad": list(
                    quick_settings["residual_action_limits_rad"]
                ),
                "preview_horizons_frames": list(
                    quick_settings["preview_horizons_frames"]
                ),
            }
        )
    return settings


def _experiment_for_limit(
    experiment: dict[str, Any], limit: float
) -> dict[str, Any]:
    selected = deepcopy(experiment)
    selected["action"]["residual_action_limit_rad"] = float(limit)
    return selected


def _add_selected_limit(values: dict[str, torch.Tensor], limit: float) -> None:
    values["selected_residual_action_limit_rad"] = torch.full_like(
        values["power_in_bucket"], float(limit)
    )


def _range_paired_summary(
    identifier: str,
    candidate: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
    gate: dict[str, Any],
) -> dict[str, Any]:
    summary = _paired_summary(identifier, candidate, baseline, gate)
    summary["action_diagnostics"]["selected_residual_action_limit_rad"] = (
        _distribution_for_summary(candidate["selected_residual_action_limit_rad"])
    )
    return summary


def _distribution_for_summary(values: torch.Tensor) -> dict[str, float]:
    from src.rl.s4_training import _distribution

    return _distribution(values)


def _limit_metric_store(
    limits: Iterable[float],
) -> dict[float, dict[str, list[torch.Tensor]]]:
    return {float(limit): defaultdict(list) for limit in limits}


def _profile_limit_metric_store(
    limits: Iterable[float],
) -> dict[float, dict[str, dict[str, list[torch.Tensor]]]]:
    return {
        float(limit): defaultdict(lambda: defaultdict(list)) for limit in limits
    }


def _scenario_record(
    *,
    controller: str,
    profile: str,
    condition: str,
    candidate: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
    allowed_limit: float,
) -> dict[str, Any]:
    return {
        "controller": controller,
        "profile": profile,
        "physical_condition": condition,
        "maximum_allowed_residual_action_rad": allowed_limit,
        "episodes": int(candidate["power_in_bucket"].numel()),
        "candidate_power_mean": float(candidate["power_in_bucket"].mean()),
        "baseline_power_mean": float(baseline["power_in_bucket"].mean()),
        "relative_power_gain": float(
            (candidate["power_in_bucket"].mean() - baseline["power_in_bucket"].mean())
            / baseline["power_in_bucket"].mean()
        ),
        "power_delta_mean": float(
            (candidate["power_in_bucket"] - baseline["power_in_bucket"]).mean()
        ),
        "strehl_delta_mean": float((candidate["strehl"] - baseline["strehl"]).mean()),
        "phase_rmse_delta_mean": float(
            (candidate["phase_rmse"] - baseline["phase_rmse"]).mean()
        ),
        "violation_fraction_mean": float(candidate["violation_fraction"].mean()),
        "requested_residual_abs_mean_rad": float(
            candidate["requested_residual_abs_mean_rad"].mean()
        ),
        "selected_preview_horizon_mean": float(
            candidate["selected_preview_horizon_frames"].mean()
        ),
        "selected_residual_action_limit_mean_rad": float(
            candidate["selected_residual_action_limit_rad"].mean()
        ),
    }


def _write_scenario_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(json_safe(rows))
