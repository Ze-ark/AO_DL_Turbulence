"""S4-D2-R3负结果后的只读动作、评论家与奖励机制诊断。"""

from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch.nn import functional as functional

from src.rl.residual_sac import QNetwork, SacConfig, SquashedGaussianActor
from src.rl.s4_closed_loop_shift import _oracle_normalized_action
from src.rl.s4_representation_capacity import (
    ANCHOR_MODES,
    ActionRepresentation,
    _future_disturbance_sequence,
    build_action_basis,
    representation_registration_inverse,
)
from src.rl.s4_r3_student_anchored_sac import _make_controller, _normalize_state
from src.rl.s4_training import (
    _append_step_metrics,
    _distribution,
    _empty_step_metrics,
    _file_sha256,
    _git_record,
    _load_yaml,
    _mean_step_metrics,
    _noisy_observation,
    _profiles,
    _project_path,
    _relative,
    _residual_reward,
    _runtime_record,
    _source_manifest,
    _write_json,
)
from src.rl.student_anchored_control import FrozenStudentPolicy, load_frozen_student
from src.runtime import resolve_device
from src.simulation.config import S1EnvConfig, load_s1_config
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.hardware_effects import HardwareProfile
from src.simulation.robust_control import RobustnessCondition
from src.training_progress import advance_to, counted_progress, update_progress


SCIENCE_METRICS = (
    "power_in_bucket",
    "measured_power_in_bucket",
    "strehl",
    "phase_rmse",
    "violation_fraction",
)

TELEMETRY_METRICS = (
    "training_style_reward",
    "actor_abs_mean",
    "applied_correction_abs_mean",
    "correction_l2_rad",
    "combined_requested_added_l2_rad",
    "actor_ideal_cosine",
    "actor_ideal_same_direction_fraction",
    "actor_ideal_mse",
    "ideal_correction_abs_mean",
    "ideal_correction_saturation_fraction",
    "q_actor",
    "q_zero",
    "q_reverse",
    "q_ideal",
    "q_actor_minus_zero",
    "correction_projection_fraction",
    "final_projection_fraction",
)


@dataclass(frozen=True)
class FrozenR3Policy:
    arm: str
    policy_seed: int
    student_id: str
    actor: SquashedGaussianActor
    q1: QNetwork
    q2: QNetwork


@dataclass(frozen=True)
class ScenarioBundle:
    arm: str
    policy_seed: int
    student_id: str
    profile_id: str
    condition_id: str
    base_seed: int
    variants: dict[float, dict[str, torch.Tensor]]


def run_s4_r3_failure_diagnostic(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """只读运行R3失败诊断，不训练、不写检查点、不访问S4-D3。"""
    experiment_path = _project_path(config_path)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight = preflight_s4_r3_failure_diagnostic(
        experiment_path, experiment, settings, quick=quick
    )
    if preflight_only:
        return preflight

    device = resolve_device("cuda")
    output_directory = _project_path(settings["output_directory"])
    if output_directory.exists():
        raise FileExistsError(
            "R3-D1 output already exists; preserve it for audit: "
            f"{output_directory}"
        )
    output_directory.mkdir(parents=True)
    _write_json(output_directory / "preflight.json", preflight)
    _write_json(output_directory / "effective_config.json", settings)
    source_manifest = _source_manifest(experiment["tracked_source_files"])
    _write_json(output_directory / "source_manifest.json", source_manifest)

    base_config, _ = load_s1_config(_project_path(experiment["environment_config"]))
    representation = ActionRepresentation.from_mapping(experiment["representation"])
    base_config = replace(base_config, num_modes=representation.num_modes)
    basis, pupil, basis_diagnostics = build_action_basis(
        base_config, representation, device
    )
    profiles = _profiles(experiment, settings["profile_ids"])
    conditions = [
        RobustnessCondition.from_mapping(item)
        for item in settings["diagnostic_conditions"]
    ]
    mappings: dict[str, torch.Tensor] = {}
    mapping_diagnostics: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        mapping, diagnostics = representation_registration_inverse(
            basis=basis,
            pupil=pupil,
            profile=profile,
            rcond=float(experiment["representation"]["pseudoinverse_rcond"]),
        )
        mappings[profile.identifier] = mapping
        mapping_diagnostics[profile.identifier] = diagnostics

    started = time.perf_counter()
    bundles: list[ScenarioBundle] = []
    scenario_records: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    total = (
        len(settings["checkpoints"])
        * len(profiles)
        * len(conditions)
        * len(settings["variants"])
    )
    completed = 0
    bar = counted_progress(total=total, description="R3失败机制诊断", unit="场景")
    for checkpoint in settings["checkpoints"]:
        policy = _load_policy(checkpoint, device=device)
        student = _load_student(checkpoint, experiment=experiment, device=device)
        for profile in profiles:
            for condition in conditions:
                config = _scenario_config(
                    base_config,
                    condition,
                    profile,
                    episodes=int(settings["episodes_per_condition"]),
                    steps=int(settings["steps"]),
                    preview=int(experiment["teacher"]["preview_horizon_frames"]),
                )
                future_truth = _future_disturbance_sequence(
                    config=config,
                    condition=condition,
                    profile=profile,
                    length=int(settings["steps"])
                    + int(experiment["teacher"]["preview_horizon_frames"]),
                    basis=basis,
                    device=device,
                )
                variant_results: dict[float, dict[str, torch.Tensor]] = {}
                for variant in settings["variants"]:
                    scale = float(variant["scale"])
                    result = _rollout_variant(
                        experiment=experiment,
                        config=config,
                        condition=condition,
                        profile=profile,
                        steps=int(settings["steps"]),
                        basis=basis,
                        future_truth=future_truth,
                        registration_mapping=mappings[profile.identifier],
                        policy=policy,
                        student=student,
                        scale=scale,
                        device=device,
                    )
                    variant_results[scale] = result
                    completed += 1
                    advance_to(bar, completed)
                    update_progress(
                        bar,
                        device=device,
                        metrics={
                            "组": 0.0 if policy.arm == "student_backbone" else 1.0,
                            "种子": float(policy.policy_seed),
                            "缩放": scale,
                            "功率": float(result["power_in_bucket"].mean()),
                            "教师余弦": float(result["actor_ideal_cosine"].mean()),
                        },
                    )
                bundle = ScenarioBundle(
                    arm=policy.arm,
                    policy_seed=policy.policy_seed,
                    student_id=policy.student_id,
                    profile_id=profile.identifier,
                    condition_id=condition.identifier,
                    base_seed=condition.base_seed,
                    variants=variant_results,
                )
                bundles.append(bundle)
                zero = variant_results[0.0]
                for variant in settings["variants"]:
                    scale = float(variant["scale"])
                    result = variant_results[scale]
                    scenario_records.append(
                        _scenario_row(bundle, variant, result, zero)
                    )
                    episode_records.extend(
                        _episode_rows(bundle, variant, result, zero)
                    )
    bar.close()

    grouped = _grouped_summaries(bundles, settings["variants"])
    zero_alignment = _zero_alignment(bundles)
    interpretation = interpret_r3_failure_diagnostic(
        grouped,
        thresholds=experiment["interpretation_thresholds"],
        zero_alignment=zero_alignment,
        quick=quick,
    )
    scenario_path = output_directory / "scenario_records.csv"
    episode_path = output_directory / "episode_records.csv"
    _write_csv(scenario_path, scenario_records)
    _write_csv(episode_path, episode_records)
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "s4d2_r3_d1_failure_diagnostic_v1",
        },
        "experiment": {
            "id": "AO-S4-D2-R3-D1-FAILURE-DIAGNOSTIC",
            "status": "completed_pending_independent_audit",
            "quick": quick,
            "duration_seconds": time.perf_counter() - started,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "evidence_boundary": {
            "checkpoint_updates": False,
            "training_transitions": 0,
            "reward_changed": False,
            "algorithm_changed": False,
            "sealed_s4d3_accessed": False,
            "real_slm_actions": False,
            "diagnostic_only": True,
        },
        "inputs": {
            "experiment_config": _relative(experiment_path),
            "experiment_config_sha256": _file_sha256(experiment_path),
            "upstream_r3_summary": experiment["upstream_r3"]["summary"],
            "upstream_r3_summary_sha256": experiment["upstream_r3"]["summary_sha256"],
            "checkpoint_count": len(settings["checkpoints"]),
            "source_manifest": source_manifest,
        },
        "design": {
            "variants": settings["variants"],
            "profiles": settings["profile_ids"],
            "conditions": settings["diagnostic_conditions"],
            "episodes_per_condition": settings["episodes_per_condition"],
            "steps": settings["steps"],
            "paired_student_only_baseline": True,
            "teacher_truth_used_only_for_diagnostic_labels": True,
        },
        "basis_diagnostics": basis_diagnostics,
        "registration_diagnostics": mapping_diagnostics,
        "zero_alignment": zero_alignment,
        "grouped": grouped,
        "interpretation": interpretation,
        "records": {
            "scenario_records": _relative(scenario_path),
            "scenario_records_sha256": _file_sha256(scenario_path),
            "scenario_rows": len(scenario_records),
            "episode_records": _relative(episode_path),
            "episode_records_sha256": _file_sha256(episode_path),
            "episode_rows": len(episode_records),
        },
        "runtime": _runtime_record(),
        "git": _git_record(),
        "next_action": (
            "Stop and request a read-only audit. Do not retrain, change rewards or "
            "algorithms, open S4-D3, or operate real SLM hardware."
        ),
    }
    _write_json(output_directory / "summary.json", summary)
    return summary


def preflight_s4_r3_failure_diagnostic(
    experiment_path: Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    metadata = experiment["metadata"]
    if str(metadata["stage"]) != "S4-D2-R3-D1":
        raise ValueError("diagnostic metadata must identify S4-D2-R3-D1")
    if not bool(metadata.get("user_authorized_next_stage", False)):
        raise RuntimeError("R3-D1 requires explicit user authorization")
    for field in (
        "allow_training",
        "allow_checkpoint_updates",
        "allow_reward_change",
        "allow_algorithm_change",
        "allow_gate_relaxation",
        "allow_s4d3_access",
        "allow_real_hardware_actions",
    ):
        if bool(metadata.get(field, True)):
            raise RuntimeError(f"R3-D1 protection flag must remain false: {field}")
    upstream = _verify_upstream(experiment["upstream_r3"])
    if upstream["run_count"] != 6 or upstream["all_failed"] is not True:
        raise RuntimeError("R3-D1 requires the audited six-run R3 negative result")
    representation = experiment["representation"]
    if (
        str(representation["kind"]) != "zernike"
        or int(representation["num_modes"]) != 21
        or int(representation["anchor_modes"]) != ANCHOR_MODES
        or int(representation["output_modes"]) != 11
    ):
        raise RuntimeError("R3-D1 representation contract changed")
    if int(experiment["policy_observation"]["state_size"]) != 210:
        raise RuntimeError("R3-D1 state contract changed")
    if not math.isclose(float(experiment["student_anchor"]["deployment_scale"]), 0.5):
        raise RuntimeError("R3-D1 student scale changed")
    if not math.isclose(float(experiment["action"]["correction_component_limit_rad"]), 0.0125):
        raise RuntimeError("R3-D1 correction limit changed")
    formal_variants = [float(item["scale"]) for item in experiment["variants"]]
    if formal_variants != [-1.0, -0.5, 0.0, 0.25, 0.5, 1.0]:
        raise RuntimeError("R3-D1 locked direction/amplitude variants changed")
    checkpoint_contracts = _verify_checkpoints(
        experiment["checkpoints"], upstream["run_inventory"]
    )
    _validate_seed_namespace(experiment, settings)
    for field, hash_field in (
        ("environment_config", "environment_config_sha256"),
        ("hardware_profile_source", "hardware_profile_source_sha256"),
    ):
        if _file_sha256(_project_path(experiment[field])) != str(experiment[hash_field]):
            raise RuntimeError(f"R3-D1 input hash mismatch: {field}")
    missing = [
        path for path in experiment["tracked_source_files"] if not _project_path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"R3-D1 tracked source files missing: {missing}")
    expected_scenarios = (
        len(settings["checkpoints"])
        * len(settings["profile_ids"])
        * len(settings["diagnostic_conditions"])
        * len(settings["variants"])
    )
    return {
        "status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_DIAGNOSTIC",
        "quick": quick,
        "upstream_r3_status": upstream["status"],
        "upstream_all_six_runs_failed": upstream["all_failed"],
        "checkpoint_contracts": checkpoint_contracts,
        "checkpoint_count": len(settings["checkpoints"]),
        "variant_scales": [float(item["scale"]) for item in settings["variants"]],
        "expected_scenarios": expected_scenarios,
        "expected_episode_records": expected_scenarios
        * int(settings["episodes_per_condition"]),
        "checkpoint_writes": 0,
        "training_transitions": 0,
        "cuda_required": True,
        "sealed_s4d3_access": False,
        "real_slm_actions": False,
        "formal_command": (
            ".\\.venv\\Scripts\\python.exe scripts\\diagnose_s4_r3_failure.py "
            "--config configs\\experiments\\s4_r3_failure_diagnostic_v1.yaml"
        ),
    }


@torch.no_grad()
def _rollout_variant(
    *,
    experiment: dict[str, Any],
    config: S1EnvConfig,
    condition: RobustnessCondition,
    profile: HardwareProfile,
    steps: int,
    basis: torch.Tensor,
    future_truth: torch.Tensor,
    registration_mapping: torch.Tensor,
    policy: FrozenR3Policy,
    student: FrozenStudentPolicy,
    scale: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    environment = AdaptiveOpticsEnv(
        config, device, profile.effects_config(), basis_override=basis
    )
    observation, _ = environment.reset(seed=condition.base_seed)
    generator = torch.Generator(device=device).manual_seed(condition.base_seed + 40_000_000)
    noisy = _noisy_observation(
        observation, config.num_modes, profile.observation_noise_std_rad, generator
    )
    controller = _make_controller(experiment, config, student)
    state = controller.reset(noisy)
    science = _empty_step_metrics()
    telemetry: dict[str, list[torch.Tensor]] = {key: [] for key in TELEMETRY_METRICS}
    preview = int(experiment["teacher"]["preview_horizon_frames"])
    correction_limit = float(experiment["action"]["correction_component_limit_rad"])
    student_scale = float(experiment["student_anchor"]["deployment_scale"])
    correction_scale = correction_limit / float(experiment["action"]["residual_action_limit_rad"])
    for step in range(steps):
        normalized_state = _normalize_state(state, student)
        raw = policy.actor.deterministic(normalized_state)
        applied = (scale * raw).clamp(-1, 1)
        target_request = (
            -future_truth[step + preview] @ registration_mapping.transpose(0, 1)
        )
        oracle_added = _oracle_normalized_action(
            state,
            target_request,
            residual_component_limit_rad=float(
                experiment["action"]["residual_action_limit_rad"]
            ),
            residual_l2_budget_rad=controller.controller.residual_l2_budget_rad,
        )
        student_added = controller.student_action(state)
        ideal = (
            (oracle_added - student_scale * student_added) / correction_scale
        ).clamp(-1, 1)
        cosine = functional.cosine_similarity(raw, ideal, dim=-1, eps=1e-8)
        dot = (raw * ideal).sum(dim=-1)
        zeros = torch.zeros_like(raw)
        q_actor = torch.minimum(
            policy.q1(normalized_state, raw), policy.q2(normalized_state, raw)
        ).squeeze(-1)
        q_zero = torch.minimum(
            policy.q1(normalized_state, zeros), policy.q2(normalized_state, zeros)
        ).squeeze(-1)
        q_reverse = torch.minimum(
            policy.q1(normalized_state, -raw), policy.q2(normalized_state, -raw)
        ).squeeze(-1)
        q_ideal = torch.minimum(
            policy.q1(normalized_state, ideal), policy.q2(normalized_state, ideal)
        ).squeeze(-1)
        action = controller.compose_action(state, applied)
        observation, _, _, _, info = environment.step(action.composed.final_delta_rad)
        _append_step_metrics(science, info)
        reward = _residual_reward(
            measured_power=info["measured_power_in_bucket"],
            normalized_residual=applied,
            violation=info["violation_fraction"],
            reward_config=experiment["reward"],
        )
        telemetry["training_style_reward"].append(reward.cpu())
        telemetry["actor_abs_mean"].append(raw.abs().mean(dim=-1).cpu())
        telemetry["applied_correction_abs_mean"].append(
            applied.abs().mean(dim=-1).cpu()
        )
        telemetry["correction_l2_rad"].append(
            (applied * correction_limit).norm(dim=-1).cpu()
        )
        telemetry["combined_requested_added_l2_rad"].append(
            action.composed.requested_residual_rad[:, ANCHOR_MODES:].norm(dim=-1).cpu()
        )
        telemetry["actor_ideal_cosine"].append(cosine.cpu())
        telemetry["actor_ideal_same_direction_fraction"].append((dot > 0).float().cpu())
        telemetry["actor_ideal_mse"].append((raw - ideal).square().mean(dim=-1).cpu())
        telemetry["ideal_correction_abs_mean"].append(ideal.abs().mean(dim=-1).cpu())
        telemetry["ideal_correction_saturation_fraction"].append(
            ideal.abs().ge(0.999).float().mean(dim=-1).cpu()
        )
        telemetry["q_actor"].append(q_actor.cpu())
        telemetry["q_zero"].append(q_zero.cpu())
        telemetry["q_reverse"].append(q_reverse.cpu())
        telemetry["q_ideal"].append(q_ideal.cpu())
        telemetry["q_actor_minus_zero"].append((q_actor - q_zero).cpu())
        telemetry["correction_projection_fraction"].append(
            action.correction_projection_fraction.cpu()
        )
        telemetry["final_projection_fraction"].append(
            action.final_projection_fraction.cpu()
        )
        if step + 1 < steps:
            noisy = _noisy_observation(
                observation,
                config.num_modes,
                profile.observation_noise_std_rad,
                generator,
            )
            state = controller.advance_observation(noisy)
    result = _mean_step_metrics(science)
    result.update(
        {key: torch.stack(values, dim=1).mean(dim=1) for key, values in telemetry.items()}
    )
    if any(not bool(torch.isfinite(value).all()) for value in result.values()):
        raise RuntimeError("R3-D1 rollout produced non-finite metrics")
    return result


def _grouped_summaries(
    bundles: list[ScenarioBundle], variants: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    keys = sorted({(item.arm, item.policy_seed, item.student_id) for item in bundles})
    for arm, policy_seed, student_id in keys:
        selected = [
            item
            for item in bundles
            if (item.arm, item.policy_seed, item.student_id)
            == (arm, policy_seed, student_id)
        ]
        zero = _concat_results(selected, 0.0)
        variant_summaries: list[dict[str, Any]] = []
        for variant in variants:
            scale = float(variant["scale"])
            candidate = _concat_results(selected, scale)
            power_delta = candidate["power_in_bucket"] - zero["power_in_bucket"]
            reward_delta = candidate["training_style_reward"] - zero["training_style_reward"]
            variant_summaries.append(
                {
                    "id": str(variant["id"]),
                    "scale": scale,
                    "episodes": int(candidate["power_in_bucket"].numel()),
                    "relative_power_gain_vs_student": float(power_delta.mean())
                    / max(float(zero["power_in_bucket"].mean()), 1e-12),
                    "paired_power_delta": _distribution(power_delta),
                    "paired_reward_delta": _distribution(reward_delta),
                    "candidate_power": _distribution(candidate["power_in_bucket"]),
                    "candidate_strehl": _distribution(candidate["strehl"]),
                    "candidate_phase_rmse": _distribution(candidate["phase_rmse"]),
                    "candidate_violation": _distribution(candidate["violation_fraction"]),
                    "action": {
                        key: _distribution(candidate[key])
                        for key in (
                            "applied_correction_abs_mean",
                            "correction_l2_rad",
                            "combined_requested_added_l2_rad",
                            "correction_projection_fraction",
                            "final_projection_fraction",
                        )
                    },
                }
            )
        zero_alignment = {
            key: _distribution(zero[key])
            for key in (
                "actor_abs_mean",
                "actor_ideal_cosine",
                "actor_ideal_same_direction_fraction",
                "actor_ideal_mse",
                "ideal_correction_abs_mean",
                "ideal_correction_saturation_fraction",
                "q_actor",
                "q_zero",
                "q_reverse",
                "q_ideal",
                "q_actor_minus_zero",
            )
        }
        grouped.append(
            {
                "arm": arm,
                "policy_seed": policy_seed,
                "student_id": student_id,
                "scenarios": len(selected),
                "student_state_alignment": zero_alignment,
                "variants": variant_summaries,
            }
        )
    return grouped


def interpret_r3_failure_diagnostic(
    grouped: list[dict[str, Any]],
    *,
    thresholds: dict[str, Any],
    zero_alignment: dict[str, Any],
    quick: bool,
) -> dict[str, Any]:
    """机械生成原因标签，不授权重训、换奖励或换算法。"""
    results: list[dict[str, Any]] = []
    for item in grouped:
        by_scale = {float(row["scale"]): row for row in item["variants"]}
        full = by_scale[1.0]
        reverse = by_scale[-1.0]
        reduced = [by_scale[scale] for scale in (0.25, 0.5) if scale in by_scale]
        full_negative = (
            float(full["paired_power_delta"]["ci95_high"])
            < float(thresholds["negative_power_ci_high"])
        )
        reverse_positive = (
            float(reverse["paired_power_delta"]["ci95_low"])
            > float(thresholds["positive_power_ci_low"])
        )
        positive_reduced = [
            row
            for row in reduced
            if float(row["paired_power_delta"]["ci95_low"])
            > float(thresholds["positive_power_ci_low"])
        ]
        alignment = item["student_state_alignment"]
        low_alignment = (
            float(alignment["actor_ideal_cosine"]["mean"])
            < float(thresholds["low_actor_teacher_cosine"])
            or float(alignment["actor_ideal_same_direction_fraction"]["mean"])
            < float(thresholds["low_same_direction_fraction"])
        )
        critic_prefers_actor = (
            float(alignment["q_actor_minus_zero"]["mean"])
            > float(thresholds["critic_preference_margin"])
        )
        reward_conflict = (
            float(full["paired_reward_delta"]["mean"])
            > float(thresholds["reward_conflict_margin"])
            and full_negative
        )
        all_nonzero_nonpositive = all(
            float(row["paired_power_delta"]["ci95_high"]) <= 0
            for scale, row in by_scale.items()
            if not math.isclose(scale, 0.0, abs_tol=1e-12)
        )
        if reverse_positive and full_negative:
            label = "ACTION_DIRECTION_MISMATCH_SUPPORTED"
        elif positive_reduced and full_negative:
            label = "EXCESSIVE_ACTION_AMPLITUDE_SUPPORTED"
        elif critic_prefers_actor and full_negative:
            label = "CRITIC_RANKING_FAILURE_SUPPORTED"
        elif reward_conflict:
            label = "REWARD_METRIC_CONFLICT_SUPPORTED"
        elif low_alignment and full_negative:
            label = "POLICY_TEACHER_DIRECTION_MISMATCH_SUPPORTED"
        elif all_nonzero_nonpositive:
            label = "NO_TESTED_CORRECTION_IMPROVES_STUDENT"
        else:
            label = "CAUSE_NOT_ISOLATED"
        results.append(
            {
                "arm": item["arm"],
                "policy_seed": item["policy_seed"],
                "primary_label": label,
                "full_scale_negative": full_negative,
                "reverse_scale_positive": reverse_positive,
                "positive_reduced_scales": [row["scale"] for row in positive_reduced],
                "low_actor_teacher_alignment": low_alignment,
                "critic_prefers_actor_over_zero": critic_prefers_actor,
                "reward_improves_while_power_worsens": reward_conflict,
                "all_nonzero_scales_nonpositive": all_nonzero_nonpositive,
            }
        )
    main = [item for item in results if item["arm"] == "student_backbone"]
    labels = sorted({item["primary_label"] for item in main})
    if quick:
        status = "QUICK_SMOKE_ONLY"
    elif not bool(zero_alignment["pass"]):
        status = "ZERO_CORRECTION_ALIGNMENT_FAILED"
    elif len(labels) == 1:
        status = labels[0]
    else:
        status = "MIXED_FAILURE_MECHANISMS"
    return {
        "status": status,
        "checkpoint_results": results,
        "main_arm_labels": labels,
        "zero_alignment_pass": bool(zero_alignment["pass"]),
        "retraining_authorized": False,
        "reward_change_authorized": False,
        "algorithm_change_authorized": False,
        "s4d3_authorized": False,
        "real_hardware_authorized": False,
        "next_rule": (
            "Audit the diagnostic first. Only a consistent mechanism across all three "
            "main seeds may justify one separately designed revision."
        ),
    }


def _zero_alignment(bundles: list[ScenarioBundle]) -> dict[str, Any]:
    maximum = 0.0
    comparisons = 0
    by_key: dict[tuple[str, str, str], dict[str, torch.Tensor]] = {}
    for item in bundles:
        key = (item.student_id, item.profile_id, item.condition_id)
        zero = item.variants[0.0]
        if key in by_key:
            reference = by_key[key]
            for metric in SCIENCE_METRICS:
                maximum = max(
                    maximum,
                    float((zero[metric] - reference[metric]).abs().max()),
                )
            comparisons += 1
        else:
            by_key[key] = zero
    tolerance = 5e-6
    return {
        "cross_arm_comparisons": comparisons,
        "max_absolute_science_metric_difference": maximum,
        "tolerance": tolerance,
        "pass": comparisons > 0 and maximum <= tolerance,
    }


def _concat_results(
    bundles: list[ScenarioBundle], scale: float
) -> dict[str, torch.Tensor]:
    keys = (*SCIENCE_METRICS, *TELEMETRY_METRICS)
    return {
        key: torch.cat([item.variants[scale][key] for item in bundles])
        for key in keys
    }


def _scenario_row(
    bundle: ScenarioBundle,
    variant: dict[str, Any],
    result: dict[str, torch.Tensor],
    zero: dict[str, torch.Tensor],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "arm": bundle.arm,
        "policy_seed": bundle.policy_seed,
        "student_id": bundle.student_id,
        "profile_id": bundle.profile_id,
        "condition_id": bundle.condition_id,
        "variant": str(variant["id"]),
        "scale": float(variant["scale"]),
        "episodes": int(result["power_in_bucket"].numel()),
    }
    for metric in (*SCIENCE_METRICS, *TELEMETRY_METRICS):
        row[f"{metric}_mean"] = float(result[metric].mean())
    for metric in ("power_in_bucket", "strehl", "phase_rmse", "training_style_reward"):
        row[f"delta_vs_student_{metric}_mean"] = float(
            (result[metric] - zero[metric]).mean()
        )
    return row


def _episode_rows(
    bundle: ScenarioBundle,
    variant: dict[str, Any],
    result: dict[str, torch.Tensor],
    zero: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    count = int(result["power_in_bucket"].numel())
    for index in range(count):
        row: dict[str, Any] = {
            "arm": bundle.arm,
            "policy_seed": bundle.policy_seed,
            "student_id": bundle.student_id,
            "profile_id": bundle.profile_id,
            "condition_id": bundle.condition_id,
            "episode_index": index,
            "episode_seed": bundle.base_seed + index,
            "variant": str(variant["id"]),
            "scale": float(variant["scale"]),
        }
        for metric in (*SCIENCE_METRICS, *TELEMETRY_METRICS):
            row[metric] = float(result[metric][index])
        for metric in ("power_in_bucket", "strehl", "phase_rmse", "training_style_reward"):
            row[f"student_{metric}"] = float(zero[metric][index])
            row[f"delta_vs_student_{metric}"] = float(
                result[metric][index] - zero[metric][index]
            )
        rows.append(row)
    return rows


def _load_policy(item: dict[str, Any], *, device: torch.device) -> FrozenR3Policy:
    payload = torch.load(
        _project_path(item["path"]), map_location=device, weights_only=False
    )
    if payload.get("algorithm") != "student_anchored_residual_sac":
        raise RuntimeError("R3-D1 checkpoint algorithm mismatch")
    config = SacConfig(**payload["config"])
    if config.state_size != 210 or config.action_size != 11:
        raise RuntimeError("R3-D1 checkpoint dimensions changed")
    actor = SquashedGaussianActor(config).to(device)
    q1 = QNetwork(config).to(device)
    q2 = QNetwork(config).to(device)
    actor.load_state_dict(payload["actor"])
    q1.load_state_dict(payload["q1"])
    q2.load_state_dict(payload["q2"])
    for model in (actor, q1, q2):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return FrozenR3Policy(
        arm=str(item["arm"]),
        policy_seed=int(item["policy_seed"]),
        student_id=str(item["student_id"]),
        actor=actor,
        q1=q1,
        q2=q2,
    )


def _load_student(
    item: dict[str, Any],
    *,
    experiment: dict[str, Any],
    device: torch.device,
) -> FrozenStudentPolicy:
    return load_frozen_student(
        _project_path(item["student_path"]),
        identifier=str(item["student_id"]),
        state_size=210,
        hidden_size=int(experiment["student_anchor"]["hidden_size"]),
        output_size=11,
        device=device,
    )


def _verify_upstream(upstream: dict[str, Any]) -> dict[str, Any]:
    for field in (
        "summary",
        "preflight",
        "effective_config",
        "source_manifest",
        "experiment_config",
    ):
        path = _project_path(upstream[field])
        if _file_sha256(path) != str(upstream[f"{field}_sha256"]):
            raise RuntimeError(f"R3-D1 upstream hash mismatch: {field}")
    summary = json.loads(_project_path(upstream["summary"]).read_text(encoding="utf-8"))
    if bool(summary["experiment"]["quick"]):
        raise RuntimeError("R3-D1 cannot diagnose quick-smoke output")
    status = str(summary["experiment"]["status"])
    if status != str(upstream["required_status"]):
        raise RuntimeError("R3-D1 upstream completion status changed")
    runs = summary["run_results"]
    all_failed = all(
        item["final_development_validation"]["development_gate"]
        == str(upstream["required_development_gate"])
        for item in runs
    )
    inventory = {
        (str(item["arm"]), int(item["policy_seed"])): {
            "student_id": str(item["student_id"]),
            "best_checkpoint": str(item["best_checkpoint"]),
            "best_checkpoint_sha256": str(item["best_checkpoint_sha256"]),
        }
        for item in runs
    }
    return {
        "status": status,
        "run_count": len(runs),
        "all_failed": all_failed,
        "run_inventory": inventory,
    }


def _verify_checkpoints(
    checkpoints: list[dict[str, Any]],
    inventory: dict[tuple[str, int], dict[str, str]],
) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    expected = {
        (arm, seed)
        for arm in ("student_backbone", "random_backbone")
        for seed in (9301, 9302, 9303)
    }
    actual = {(str(item["arm"]), int(item["policy_seed"])) for item in checkpoints}
    if actual != expected or len(checkpoints) != 6:
        raise RuntimeError("R3-D1 checkpoint inventory must contain the locked six runs")
    for item in checkpoints:
        key = (str(item["arm"]), int(item["policy_seed"]))
        upstream = inventory[key]
        path = _project_path(item["path"])
        if _relative(path) != upstream["best_checkpoint"]:
            raise RuntimeError("R3-D1 checkpoint is not the recorded best checkpoint")
        digest = _file_sha256(path)
        if digest != str(item["sha256"]) or digest != upstream["best_checkpoint_sha256"]:
            raise RuntimeError("R3-D1 checkpoint hash mismatch")
        student_path = _project_path(item["student_path"])
        if _file_sha256(student_path) != str(item["student_sha256"]):
            raise RuntimeError("R3-D1 student checkpoint hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        tensors = [value for value in payload["actor"].values() if torch.is_tensor(value)]
        if (
            payload.get("algorithm") != "student_anchored_residual_sac"
            or str(payload.get("student_id")) != str(item["student_id"])
            or not all(bool(torch.isfinite(value).all()) for value in tensors)
        ):
            raise RuntimeError("R3-D1 checkpoint content contract failed")
        contracts.append(
            {
                "arm": key[0],
                "policy_seed": key[1],
                "student_id": str(item["student_id"]),
                "checkpoint_sha256": digest,
            }
        )
    return contracts


def _validate_seed_namespace(
    experiment: dict[str, Any], settings: dict[str, Any]
) -> None:
    episodes = int(settings["episodes_per_condition"])
    seeds = {
        int(item["base_seed"]) + offset
        for item in settings["diagnostic_conditions"]
        for offset in range(episodes)
    }
    if any(seed < 3_700_000 or seed >= 4_000_000 for seed in seeds):
        raise RuntimeError("R3-D1 seeds left the locked diagnostic namespace")
    if len(seeds) != len(settings["diagnostic_conditions"]) * episodes:
        raise RuntimeError("R3-D1 diagnostic episode seeds overlap")
    for protected in experiment["protected_seed_ranges"]:
        start = int(protected["start_inclusive"])
        stop = int(protected["end_exclusive"])
        if any(start <= seed < stop for seed in seeds):
            raise RuntimeError(f"R3-D1 seeds overlap protected range: {protected['id']}")


def _scenario_config(
    base: S1EnvConfig,
    condition: RobustnessCondition,
    profile: HardwareProfile,
    *,
    episodes: int,
    steps: int,
    preview: int,
) -> S1EnvConfig:
    config = replace(
        base,
        batch_size=episodes,
        episode_length=max(base.episode_length, steps + preview),
    )
    return profile.environment_config(condition.environment_config(config))


def _effective_settings(experiment: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "quick": quick,
        "output_directory": experiment["outputs"]["directory"],
        "checkpoints": deepcopy(experiment["checkpoints"]),
        "variants": deepcopy(experiment["variants"]),
        "profile_ids": list(experiment["profile_ids"]),
        "diagnostic_conditions": deepcopy(experiment["diagnostic_conditions"]),
        "episodes_per_condition": int(experiment["diagnostic"]["episodes_per_condition"]),
        "steps": int(experiment["diagnostic"]["steps"]),
    }
    if quick:
        quick_settings = experiment["quick"]
        arms = set(map(str, quick_settings["checkpoint_arms"]))
        seeds = set(map(int, quick_settings["policy_seeds"]))
        variant_ids = set(map(str, quick_settings["variants"]))
        settings["checkpoints"] = [
            item
            for item in settings["checkpoints"]
            if str(item["arm"]) in arms and int(item["policy_seed"]) in seeds
        ]
        settings["variants"] = [
            item for item in settings["variants"] if str(item["id"]) in variant_ids
        ]
        settings["profile_ids"] = list(quick_settings["profile_ids"])
        settings["diagnostic_conditions"] = deepcopy(
            quick_settings["diagnostic_conditions"]
        )
        settings["episodes_per_condition"] = int(
            quick_settings["episodes_per_condition"]
        )
        settings["steps"] = int(quick_settings["steps"])
        settings["output_directory"] = experiment["outputs"]["quick_directory"]
    return settings


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("R3-D1 records must not be empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
