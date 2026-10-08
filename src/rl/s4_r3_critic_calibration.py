"""R3-D1-A冻结评论家部署排序校准与奖励分解诊断。"""

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

from src.rl.s4_closed_loop_shift import _oracle_normalized_action
from src.rl.s4_r3_failure_diagnostic import (
    FrozenR3Policy,
    _effective_settings as _d1_effective_settings,
    _load_policy,
    _load_student,
    _scenario_config,
    preflight_s4_r3_failure_diagnostic,
)
from src.rl.s4_r3_student_anchored_sac import _make_controller, _normalize_state
from src.rl.s4_representation_capacity import (
    ANCHOR_MODES,
    ActionRepresentation,
    _future_disturbance_sequence,
    build_action_basis,
    representation_registration_inverse,
)
from src.rl.s4_training import (
    _distribution,
    _file_sha256,
    _git_record,
    _load_yaml,
    _noisy_observation,
    _profiles,
    _project_path,
    _relative,
    _runtime_record,
    _source_manifest,
    _write_json,
)
from src.rl.student_anchored_control import FrozenStudentPolicy
from src.runtime import resolve_device
from src.simulation.config import S1EnvConfig, load_s1_config
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.hardware_effects import HardwareProfile
from src.simulation.robust_control import RobustnessCondition
from src.training_progress import advance_to, counted_progress, update_progress


REQUIRED_CANDIDATES = (
    "reverse_actor",
    "zero",
    "quarter_actor",
    "actor",
    "ideal_teacher",
)

REWARD_ACCUMULATOR_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
}


@dataclass(frozen=True)
class ProbeBundle:
    arm: str
    policy_seed: int
    student_id: str
    profile_id: str
    condition_id: str
    base_seed: int
    probe_step: int
    candidates: dict[str, dict[str, torch.Tensor]]


def run_s4_r3_critic_calibration(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """冻结R3策略，校准评论家动作排序并拆解奖励，不执行训练。"""
    experiment_path = _project_path(config_path)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight, d1_experiment = preflight_s4_r3_critic_calibration(
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
            "R3-D1-A output already exists; preserve it for audit: "
            f"{output_directory}"
        )
    output_directory.mkdir(parents=True)
    _write_json(output_directory / "preflight.json", preflight)
    _write_json(output_directory / "effective_config.json", settings)
    source_manifest = _source_manifest(experiment["tracked_source_files"])
    _write_json(output_directory / "source_manifest.json", source_manifest)

    base_config, _ = load_s1_config(_project_path(d1_experiment["environment_config"]))
    representation = ActionRepresentation.from_mapping(d1_experiment["representation"])
    base_config = replace(base_config, num_modes=representation.num_modes)
    basis, pupil, basis_diagnostics = build_action_basis(
        base_config, representation, device
    )
    profiles = _profiles(d1_experiment, settings["profile_ids"])
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
            rcond=float(d1_experiment["representation"]["pseudoinverse_rcond"]),
        )
        mappings[profile.identifier] = mapping
        mapping_diagnostics[profile.identifier] = diagnostics

    started = time.perf_counter()
    bundles: list[ProbeBundle] = []
    total = (
        len(settings["checkpoints"])
        * len(profiles)
        * len(conditions)
        * len(settings["probe_steps"])
        * len(settings["candidate_actions"])
    )
    completed = 0
    bar = counted_progress(total=total, description="R3评论家校准", unit="分支")
    max_required_step = max(settings["probe_steps"]) + int(
        settings["return_horizon_steps"]
    )
    preview = int(d1_experiment["teacher"]["preview_horizon_frames"])
    for checkpoint in settings["checkpoints"]:
        policy = _load_policy(checkpoint, device=device)
        student = _load_student(
            checkpoint,
            experiment=d1_experiment,
            device=device,
        )
        checkpoint_payload = torch.load(
            _project_path(checkpoint["path"]),
            map_location="cpu",
            weights_only=False,
        )
        gamma = float(checkpoint_payload["config"]["gamma"])
        for profile in profiles:
            for condition in conditions:
                config = _scenario_config(
                    base_config,
                    condition,
                    profile,
                    episodes=int(settings["episodes_per_condition"]),
                    steps=max_required_step,
                    preview=preview,
                )
                future_truth = _future_disturbance_sequence(
                    config=config,
                    condition=condition,
                    profile=profile,
                    length=max_required_step + preview,
                    basis=basis,
                    device=device,
                )
                for probe_step in settings["probe_steps"]:
                    candidate_results: dict[str, dict[str, torch.Tensor]] = {}
                    for candidate in settings["candidate_actions"]:
                        result = _rollout_probe_branch(
                            d1_experiment=d1_experiment,
                            config=config,
                            condition=condition,
                            profile=profile,
                            basis=basis,
                            future_truth=future_truth,
                            registration_mapping=mappings[profile.identifier],
                            policy=policy,
                            student=student,
                            candidate=candidate,
                            probe_step=int(probe_step),
                            horizon=int(settings["return_horizon_steps"]),
                            gamma=gamma,
                            reward_accumulator_dtype=_resolve_reward_accumulator_dtype(
                                settings["reward_accumulator_dtype"]
                            ),
                            device=device,
                        )
                        candidate_results[str(candidate["id"])] = result
                        completed += 1
                        advance_to(bar, completed)
                        update_progress(
                            bar,
                            device=device,
                            metrics={
                                "种子": float(policy.policy_seed),
                                "探针": float(probe_step),
                                "Q": float(result["q_min"].mean()),
                                "回报": float(result["discounted_reward_return"].mean()),
                            },
                        )
                    bundles.append(
                        ProbeBundle(
                            arm=policy.arm,
                            policy_seed=policy.policy_seed,
                            student_id=policy.student_id,
                            profile_id=profile.identifier,
                            condition_id=condition.identifier,
                            base_seed=condition.base_seed,
                            probe_step=int(probe_step),
                            candidates=candidate_results,
                        )
                    )
    bar.close()

    grouped = summarize_critic_calibration(bundles, settings["candidate_actions"])
    interpretation = interpret_critic_calibration(
        grouped,
        thresholds=experiment["interpretation_thresholds"],
        quick=quick,
    )
    probe_rows = _probe_rows(bundles, settings["candidate_actions"])
    episode_rows = _episode_rows(bundles, settings["candidate_actions"])
    probe_path = output_directory / "probe_records.csv"
    episode_path = output_directory / "episode_records.csv"
    _write_csv(probe_path, probe_rows)
    _write_csv(episode_path, episode_rows)
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": str(
                experiment["metadata"].get(
                    "version_label", "s4d2_r3_d1a_critic_calibration_v1"
                )
            ),
        },
        "experiment": {
            "id": str(
                experiment["metadata"].get(
                    "experiment_id", "AO-S4-D2-R3-D1-A-CRITIC-CALIBRATION"
                )
            ),
            "status": "completed_pending_independent_audit",
            "quick": quick,
            "duration_seconds": time.perf_counter() - started,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "evidence_boundary": {
            "deployment_ranking_calibration": True,
            "exact_soft_q_target_calibration": False,
            "checkpoint_updates": False,
            "training_transitions": 0,
            "reward_changed": False,
            "algorithm_changed": False,
            "sealed_s4d3_accessed": False,
            "real_slm_actions": False,
        },
        "inputs": {
            "experiment_config": _relative(experiment_path),
            "experiment_config_sha256": _file_sha256(experiment_path),
            "upstream_d1_summary": experiment["upstream_d1"]["summary"],
            "upstream_d1_summary_sha256": experiment["upstream_d1"][
                "summary_sha256"
            ],
            "checkpoint_count": len(settings["checkpoints"]),
            "source_manifest": source_manifest,
            "repair_reference": (
                {
                    "summary": experiment["repair_reference"]["summary"],
                    "summary_sha256": experiment["repair_reference"][
                        "summary_sha256"
                    ],
                    "required_interpretation_status": experiment[
                        "repair_reference"
                    ]["required_interpretation_status"],
                }
                if "repair_reference" in experiment
                else None
            ),
        },
        "design": {
            "candidate_actions": settings["candidate_actions"],
            "profiles": settings["profile_ids"],
            "conditions": settings["diagnostic_conditions"],
            "episodes_per_condition": settings["episodes_per_condition"],
            "probe_steps": settings["probe_steps"],
            "return_horizon_steps": settings["return_horizon_steps"],
            "common_prefix_policy": "frozen_student_only",
            "candidate_applies_for_first_step_only": True,
            "continuation_policy": "deterministic_frozen_actor",
            "finite_horizon_return": True,
            "reward_accumulator_dtype": settings["reward_accumulator_dtype"],
            "decomposition_tolerance": float(
                experiment["interpretation_thresholds"]["decomposition_tolerance"]
            ),
            "precision_repair_only": str(experiment["metadata"]["stage"]).endswith(
                "-V2"
            ),
        },
        "basis_diagnostics": basis_diagnostics,
        "registration_diagnostics": mapping_diagnostics,
        "grouped": grouped,
        "interpretation": interpretation,
        "records": {
            "probe_records": _relative(probe_path),
            "probe_records_sha256": _file_sha256(probe_path),
            "probe_rows": len(probe_rows),
            "episode_records": _relative(episode_path),
            "episode_records_sha256": _file_sha256(episode_path),
            "episode_rows": len(episode_rows),
        },
        "runtime": _runtime_record(),
        "git": _git_record(),
        "next_action": (
            "Stop for a read-only audit. Do not retrain, change reward weights, "
            "change algorithms, open S4-D3, or operate real SLM hardware."
        ),
    }
    _write_json(output_directory / "summary.json", summary)
    return summary


def preflight_s4_r3_critic_calibration(
    experiment_path: Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = experiment["metadata"]
    stage = str(metadata["stage"])
    if stage not in {"S4-D2-R3-D1-A", "S4-D2-R3-D1-A-V2"}:
        raise ValueError(
            "critic-calibration metadata must identify S4-D2-R3-D1-A or its V2 precision repair"
        )
    if not bool(metadata.get("user_authorized_next_stage", False)):
        raise RuntimeError("R3-D1-A requires explicit user authorization")
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
            raise RuntimeError(f"R3-D1-A protection flag must remain false: {field}")

    upstream = experiment["upstream_d1"]
    for field in (
        "summary",
        "preflight",
        "effective_config",
        "source_manifest",
        "scenario_records",
        "episode_records",
        "experiment_config",
    ):
        if _file_sha256(_project_path(upstream[field])) != str(
            upstream[f"{field}_sha256"]
        ):
            raise RuntimeError(f"R3-D1-A upstream hash mismatch: {field}")
    d1_summary = json.loads(
        _project_path(upstream["summary"]).read_text(encoding="utf-8")
    )
    if bool(d1_summary["experiment"]["quick"]):
        raise RuntimeError("R3-D1-A cannot consume quick-smoke evidence")
    if str(d1_summary["experiment"]["status"]) != str(upstream["required_status"]):
        raise RuntimeError("R3-D1-A upstream completion status changed")
    if str(d1_summary["interpretation"]["status"]) != str(
        upstream["required_interpretation_status"]
    ):
        raise RuntimeError("R3-D1-A upstream interpretation status changed")
    if bool(d1_summary["zero_alignment"]["pass"]) != bool(
        upstream["required_zero_alignment_pass"]
    ):
        raise RuntimeError("R3-D1-A upstream zero alignment changed")
    if int(d1_summary["records"]["scenario_rows"]) != int(
        upstream["required_scenario_rows"]
    ) or int(d1_summary["records"]["episode_rows"]) != int(
        upstream["required_episode_rows"]
    ):
        raise RuntimeError("R3-D1-A upstream record count changed")

    repair_reference = experiment.get("repair_reference")
    if stage == "S4-D2-R3-D1-A-V2":
        if not isinstance(repair_reference, dict):
            raise RuntimeError("R3-D1-A-V2 requires the immutable V1 repair reference")
        for field in (
            "summary",
            "preflight",
            "effective_config",
            "source_manifest",
            "probe_records",
            "episode_records",
            "experiment_config",
        ):
            if _file_sha256(_project_path(repair_reference[field])) != str(
                repair_reference[f"{field}_sha256"]
            ):
                raise RuntimeError(f"R3-D1-A-V2 repair-reference hash mismatch: {field}")
        v1_summary = json.loads(
            _project_path(repair_reference["summary"]).read_text(encoding="utf-8")
        )
        if str(v1_summary["interpretation"]["status"]) != str(
            repair_reference["required_interpretation_status"]
        ):
            raise RuntimeError("R3-D1-A-V2 repair-reference status changed")
        if int(v1_summary["records"]["probe_rows"]) != int(
            repair_reference["required_probe_rows"]
        ) or int(v1_summary["records"]["episode_rows"]) != int(
            repair_reference["required_episode_rows"]
        ):
            raise RuntimeError("R3-D1-A-V2 repair-reference row count changed")

    d1_path = _project_path(upstream["experiment_config"])
    d1_experiment = _load_yaml(d1_path)
    d1_preflight = preflight_s4_r3_failure_diagnostic(
        d1_path,
        d1_experiment,
        _d1_effective_settings(d1_experiment, quick=False),
        quick=False,
    )
    if int(d1_preflight["checkpoint_count"]) != 6:
        raise RuntimeError("R3-D1-A requires all six frozen R3 checkpoints")
    if tuple(str(item["id"]) for item in experiment["candidate_actions"]) != (
        REQUIRED_CANDIDATES
    ):
        raise RuntimeError("R3-D1-A candidate action set changed")
    if str(experiment["diagnostic"]["continuation_policy"]) != (
        "deterministic_frozen_actor"
    ) or str(experiment["diagnostic"]["common_prefix_policy"]) != (
        "frozen_student_only"
    ):
        raise RuntimeError("R3-D1-A policy isolation contract changed")
    _validate_seed_namespace(experiment, settings)
    accumulator_dtype = str(settings["reward_accumulator_dtype"])
    _resolve_reward_accumulator_dtype(accumulator_dtype)
    if stage == "S4-D2-R3-D1-A-V2" and accumulator_dtype != "float64":
        raise RuntimeError("R3-D1-A-V2 must accumulate reward components in float64")
    max_step = max(settings["probe_steps"]) + int(settings["return_horizon_steps"])
    if max_step > 200:
        raise RuntimeError("R3-D1-A probe and return horizon exceed the locked episode")
    missing = [
        path
        for path in experiment["tracked_source_files"]
        if not _project_path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"R3-D1-A tracked source files missing: {missing}")
    expected_probes = (
        len(settings["checkpoints"])
        * len(settings["profile_ids"])
        * len(settings["diagnostic_conditions"])
        * len(settings["probe_steps"])
    )
    expected_branches = expected_probes * len(settings["candidate_actions"])
    preflight = {
        "status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_DIAGNOSTIC",
        "quick": quick,
        "upstream_d1_status": d1_summary["experiment"]["status"],
        "upstream_interpretation": d1_summary["interpretation"]["status"],
        "zero_alignment_pass": d1_summary["zero_alignment"]["pass"],
        "checkpoint_count": len(settings["checkpoints"]),
        "candidate_ids": [str(item["id"]) for item in settings["candidate_actions"]],
        "expected_probe_states": expected_probes,
        "expected_branch_rollouts": expected_branches,
        "expected_episode_records": expected_branches
        * int(settings["episodes_per_condition"]),
        "checkpoint_writes": 0,
        "training_transitions": 0,
        "cuda_required": True,
        "reward_accumulator_dtype": accumulator_dtype,
        "decomposition_tolerance": float(
            experiment["interpretation_thresholds"]["decomposition_tolerance"]
        ),
        "sealed_s4d3_access": False,
        "real_slm_actions": False,
        "formal_command": (
            ".\\.venv\\Scripts\\python.exe "
            "scripts\\diagnose_s4_r3_critic_calibration.py --config "
            f"{str(_relative(experiment_path)).replace('/', chr(92))}"
        ),
    }
    return preflight, d1_experiment


@torch.no_grad()
def _rollout_probe_branch(
    *,
    d1_experiment: dict[str, Any],
    config: S1EnvConfig,
    condition: RobustnessCondition,
    profile: HardwareProfile,
    basis: torch.Tensor,
    future_truth: torch.Tensor,
    registration_mapping: torch.Tensor,
    policy: FrozenR3Policy,
    student: FrozenStudentPolicy,
    candidate: dict[str, Any],
    probe_step: int,
    horizon: int,
    gamma: float,
    reward_accumulator_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    environment = AdaptiveOpticsEnv(
        config,
        device,
        profile.effects_config(),
        basis_override=basis,
    )
    observation, _ = environment.reset(seed=condition.base_seed)
    observation_generator = torch.Generator(device=device).manual_seed(
        condition.base_seed + 40_000_000
    )
    noisy = _noisy_observation(
        observation,
        config.num_modes,
        profile.observation_noise_std_rad,
        observation_generator,
    )
    controller = _make_controller(d1_experiment, config, student)
    state = controller.reset(noisy)
    zeros = torch.zeros(
        config.batch_size,
        11,
        device=device,
        dtype=torch.float32,
    )
    for _ in range(probe_step):
        action = controller.compose_action(state, zeros)
        observation, _, _, _, _ = environment.step(action.composed.final_delta_rad)
        noisy = _noisy_observation(
            observation,
            config.num_modes,
            profile.observation_noise_std_rad,
            observation_generator,
        )
        state = controller.advance_observation(noisy)

    normalized_state = _normalize_state(state, student)
    actor = policy.actor.deterministic(normalized_state)
    preview = int(d1_experiment["teacher"]["preview_horizon_frames"])
    target_request = -future_truth[probe_step + preview] @ registration_mapping.transpose(
        0, 1
    )
    oracle_added = _oracle_normalized_action(
        state,
        target_request,
        residual_component_limit_rad=float(
            d1_experiment["action"]["residual_action_limit_rad"]
        ),
        residual_l2_budget_rad=controller.controller.residual_l2_budget_rad,
    )
    student_added = controller.student_action(state)
    student_scale = float(d1_experiment["student_anchor"]["deployment_scale"])
    correction_scale = float(
        d1_experiment["action"]["correction_component_limit_rad"]
    ) / float(d1_experiment["action"]["residual_action_limit_rad"])
    ideal = ((oracle_added - student_scale * student_added) / correction_scale).clamp(
        -1, 1
    )
    candidate_action = _candidate_action(candidate, actor=actor, ideal=ideal)
    q1 = policy.q1(normalized_state, candidate_action).squeeze(-1)
    q2 = policy.q2(normalized_state, candidate_action).squeeze(-1)
    q_min = torch.minimum(q1, q2)

    batch = config.batch_size
    reward_return = torch.zeros(
        batch,
        device=device,
        dtype=reward_accumulator_dtype,
    )
    true_power_return = torch.zeros_like(reward_return)
    measured_power_return = torch.zeros_like(reward_return)
    action_penalty_return = torch.zeros_like(reward_return)
    violation_penalty_return = torch.zeros_like(reward_return)
    first_projection = torch.zeros(batch, device=device, dtype=torch.float32)
    final_projection = torch.zeros_like(first_projection)
    reward_config = d1_experiment["reward"]
    power_weight = float(reward_config["measured_power_weight"])
    action_weight = float(reward_config["residual_action_weight"])
    violation_weight = float(reward_config["violation_weight"])
    for local_step in range(horizon):
        if local_step == 0:
            correction = candidate_action
        else:
            correction = policy.actor.deterministic(_normalize_state(state, student))
        action = controller.compose_action(state, correction)
        observation, _, _, _, info = environment.step(action.composed.final_delta_rad)
        power_term = (
            info["measured_power_in_bucket"].to(reward_accumulator_dtype)
            * power_weight
        )
        action_penalty = (
            action.correction_normalized.square()
            .mean(dim=-1)
            .to(reward_accumulator_dtype)
            * action_weight
        )
        violation_penalty = (
            info["violation_fraction"].to(reward_accumulator_dtype)
            * violation_weight
        )
        reward = power_term - action_penalty - violation_penalty
        discount = gamma**local_step
        reward_return += discount * reward
        true_power_return += discount * info["reward_power_in_bucket"].to(
            reward_accumulator_dtype
        )
        measured_power_return += discount * power_term
        action_penalty_return += discount * action_penalty
        violation_penalty_return += discount * violation_penalty
        if local_step == 0:
            first_projection = action.correction_projection_fraction.clone()
            final_projection = action.final_projection_fraction.clone()
        if local_step + 1 < horizon:
            noisy = _noisy_observation(
                observation,
                config.num_modes,
                profile.observation_noise_std_rad,
                observation_generator,
            )
            state = controller.advance_observation(noisy)

    decomposition_error = (
        reward_return
        - (measured_power_return - action_penalty_return - violation_penalty_return)
    ).abs()
    result = {
        "q1": q1.cpu(),
        "q2": q2.cpu(),
        "q_min": q_min.cpu(),
        "q_disagreement": (q1 - q2).abs().cpu(),
        "discounted_reward_return": reward_return.cpu(),
        "discounted_true_power_return": true_power_return.cpu(),
        "discounted_measured_power_term": measured_power_return.cpu(),
        "discounted_action_penalty": action_penalty_return.cpu(),
        "discounted_violation_penalty": violation_penalty_return.cpu(),
        "decomposition_error": decomposition_error.cpu(),
        "first_action_abs_mean": candidate_action.abs().mean(dim=-1).cpu(),
        "first_action_l2": candidate_action.norm(dim=-1).cpu(),
        "actor_ideal_cosine": functional.cosine_similarity(
            actor, ideal, dim=-1, eps=1e-8
        ).cpu(),
        "first_correction_projection_fraction": first_projection.cpu(),
        "first_final_projection_fraction": final_projection.cpu(),
    }
    if any(not bool(torch.isfinite(value).all()) for value in result.values()):
        raise RuntimeError("R3-D1-A branch produced non-finite metrics")
    return result


def summarize_critic_calibration(
    bundles: list[ProbeBundle],
    candidate_actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate_ids = [str(item["id"]) for item in candidate_actions]
    grouped: list[dict[str, Any]] = []
    keys = sorted({(item.arm, item.policy_seed, item.student_id) for item in bundles})
    for arm, policy_seed, student_id in keys:
        selected = [
            item
            for item in bundles
            if (item.arm, item.policy_seed, item.student_id)
            == (arm, policy_seed, student_id)
        ]
        concatenated = {
            candidate_id: {
                metric: torch.cat(
                    [item.candidates[candidate_id][metric] for item in selected]
                )
                for metric in selected[0].candidates[candidate_id]
            }
            for candidate_id in candidate_ids
        }
        q_matrix = torch.stack(
            [concatenated[item]["q_min"] for item in candidate_ids], dim=1
        )
        return_matrix = torch.stack(
            [
                concatenated[item]["discounted_reward_return"]
                for item in candidate_ids
            ],
            dim=1,
        )
        rank_accuracy = _pairwise_rank_accuracy(q_matrix, return_matrix)
        top_agreement = q_matrix.argmax(dim=1).eq(return_matrix.argmax(dim=1)).float()
        centered_q = q_matrix - q_matrix.mean(dim=1, keepdim=True)
        centered_return = return_matrix - return_matrix.mean(dim=1, keepdim=True)
        centered_correlation = _correlation(
            centered_q.flatten(), centered_return.flatten()
        )
        actor = concatenated["actor"]
        zero = concatenated["zero"]
        quarter = concatenated["quarter_actor"]
        actor_q_advantage = actor["q_min"] - zero["q_min"]
        actor_return_advantage = (
            actor["discounted_reward_return"] - zero["discounted_reward_return"]
        )
        actor_power_advantage = (
            actor["discounted_true_power_return"]
            - zero["discounted_true_power_return"]
        )
        quarter_reward_advantage = (
            quarter["discounted_reward_return"] - zero["discounted_reward_return"]
        )
        quarter_power_advantage = (
            quarter["discounted_true_power_return"]
            - zero["discounted_true_power_return"]
        )
        false_positive = actor_q_advantage.gt(0) & actor_return_advantage.lt(0)
        decomposition_max = max(
            float(concatenated[item]["decomposition_error"].max())
            for item in candidate_ids
        )
        grouped.append(
            {
                "arm": arm,
                "policy_seed": policy_seed,
                "student_id": student_id,
                "probe_states": len(selected),
                "episode_probes": int(q_matrix.shape[0]),
                "candidate_summaries": [
                    {
                        "id": candidate_id,
                        "q_min": _distribution(concatenated[candidate_id]["q_min"]),
                        "discounted_reward_return": _distribution(
                            concatenated[candidate_id]["discounted_reward_return"]
                        ),
                        "discounted_true_power_return": _distribution(
                            concatenated[candidate_id]["discounted_true_power_return"]
                        ),
                        "discounted_measured_power_term": _distribution(
                            concatenated[candidate_id]["discounted_measured_power_term"]
                        ),
                        "discounted_action_penalty": _distribution(
                            concatenated[candidate_id]["discounted_action_penalty"]
                        ),
                        "discounted_violation_penalty": _distribution(
                            concatenated[candidate_id]["discounted_violation_penalty"]
                        ),
                    }
                    for candidate_id in candidate_ids
                ],
                "ranking": {
                    "pairwise_rank_accuracy": _distribution(rank_accuracy),
                    "top_action_agreement": _distribution(top_agreement),
                    "centered_q_return_correlation": centered_correlation,
                    "actor_q_advantage_vs_zero": _distribution(actor_q_advantage),
                    "actor_reward_return_advantage_vs_zero": _distribution(
                        actor_return_advantage
                    ),
                    "actor_true_power_return_advantage_vs_zero": _distribution(
                        actor_power_advantage
                    ),
                    "actor_false_positive_fraction": float(false_positive.float().mean()),
                },
                "quarter_actor_vs_zero": {
                    "reward_return_advantage": _distribution(quarter_reward_advantage),
                    "true_power_return_advantage": _distribution(
                        quarter_power_advantage
                    ),
                    "measured_power_term_advantage": _distribution(
                        quarter["discounted_measured_power_term"]
                        - zero["discounted_measured_power_term"]
                    ),
                    "action_penalty_increase": _distribution(
                        quarter["discounted_action_penalty"]
                        - zero["discounted_action_penalty"]
                    ),
                    "violation_penalty_increase": _distribution(
                        quarter["discounted_violation_penalty"]
                        - zero["discounted_violation_penalty"]
                    ),
                },
                "reward_decomposition_max_abs_error": decomposition_max,
            }
        )
    return grouped


def interpret_critic_calibration(
    grouped: list[dict[str, Any]],
    *,
    thresholds: dict[str, Any],
    quick: bool,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for item in grouped:
        ranking = item["ranking"]
        quarter = item["quarter_actor_vs_zero"]
        q_positive = float(ranking["actor_q_advantage_vs_zero"]["ci95_low"]) > float(
            thresholds["positive_ci_low"]
        )
        return_negative = float(
            ranking["actor_reward_return_advantage_vs_zero"]["ci95_high"]
        ) < float(thresholds["negative_ci_high"])
        poor_pairwise_ranking = float(
            ranking["pairwise_rank_accuracy"]["mean"]
        ) < float(thresholds["maximum_rank_accuracy_for_failure"])
        poor_top_ranking = float(ranking["top_action_agreement"]["mean"]) < float(
            thresholds["maximum_top_action_agreement_for_failure"]
        )
        critic_failure = q_positive and return_negative and (
            poor_pairwise_ranking or poor_top_ranking
        )
        reward_suppression = (
            float(quarter["true_power_return_advantage"]["ci95_low"])
            > float(thresholds["positive_ci_low"])
            and float(quarter["reward_return_advantage"]["ci95_high"])
            < float(thresholds["negative_ci_high"])
        )
        decomposition_pass = float(item["reward_decomposition_max_abs_error"]) <= float(
            thresholds["decomposition_tolerance"]
        )
        if not decomposition_pass:
            label = "REWARD_DECOMPOSITION_FAILED"
        elif critic_failure and reward_suppression:
            label = "CRITIC_FAILURE_WITH_REWARD_SUPPRESSION"
        elif critic_failure:
            label = "CRITIC_DEPLOYMENT_RANKING_FAILURE_CONFIRMED"
        elif reward_suppression:
            label = "REWARD_PENALTY_SUPPRESSION_CONFIRMED"
        else:
            label = "MECHANISM_NOT_CONFIRMED"
        results.append(
            {
                "arm": item["arm"],
                "policy_seed": item["policy_seed"],
                "primary_label": label,
                "critic_failure_confirmed": critic_failure,
                "reward_penalty_suppression_confirmed": reward_suppression,
                "q_actor_advantage_positive": q_positive,
                "actor_return_advantage_negative": return_negative,
                "poor_pairwise_ranking": poor_pairwise_ranking,
                "poor_top_action_agreement": poor_top_ranking,
                "reward_decomposition_pass": decomposition_pass,
            }
        )
    main = [item for item in results if item["arm"] == "student_backbone"]
    all_main_critic = len(main) == 3 and all(
        item["critic_failure_confirmed"] for item in main
    )
    all_main_reward = len(main) == 3 and all(
        item["reward_penalty_suppression_confirmed"] for item in main
    )
    if quick:
        status = "QUICK_SMOKE_ONLY"
    elif not all(item["reward_decomposition_pass"] for item in results):
        status = "REWARD_DECOMPOSITION_FAILED"
    elif all_main_critic and all_main_reward:
        status = "CRITIC_FAILURE_AND_REWARD_SUPPRESSION_CONFIRMED"
    elif all_main_critic:
        status = "CRITIC_DEPLOYMENT_RANKING_FAILURE_CONFIRMED"
    elif all_main_reward:
        status = "REWARD_PENALTY_SUPPRESSION_CONFIRMED"
    else:
        status = "MIXED_OR_UNCONFIRMED"
    return {
        "status": status,
        "checkpoint_results": results,
        "all_main_seeds_critic_failure": all_main_critic,
        "all_main_seeds_reward_suppression": all_main_reward,
        "deployment_ranking_only": True,
        "exact_soft_q_target_claim_authorized": False,
        "retraining_authorized": False,
        "reward_change_authorized": False,
        "algorithm_change_authorized": False,
        "s4d3_authorized": False,
        "real_hardware_authorized": False,
        "next_rule": (
            "Audit first. A revision may target only a mechanism consistently "
            "confirmed across all three main seeds."
        ),
    }


def _candidate_action(
    candidate: dict[str, Any],
    *,
    actor: torch.Tensor,
    ideal: torch.Tensor,
) -> torch.Tensor:
    kind = str(candidate["kind"])
    if kind == "actor_scale":
        return (float(candidate["scale"]) * actor).clamp(-1, 1)
    if kind == "ideal_teacher":
        return ideal
    raise ValueError(f"unsupported R3-D1-A candidate kind: {kind}")


def _pairwise_rank_accuracy(
    predicted: torch.Tensor,
    observed: torch.Tensor,
) -> torch.Tensor:
    if predicted.shape != observed.shape or predicted.ndim != 2:
        raise ValueError("rank matrices must share shape [samples, candidates]")
    comparisons: list[torch.Tensor] = []
    for left in range(predicted.shape[1]):
        for right in range(left + 1, predicted.shape[1]):
            predicted_sign = torch.sign(predicted[:, left] - predicted[:, right])
            observed_sign = torch.sign(observed[:, left] - observed[:, right])
            comparisons.append(predicted_sign.eq(observed_sign).float())
    return torch.stack(comparisons, dim=1).mean(dim=1)


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float()
    right = right.float()
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if float(denominator) <= 1e-12:
        return 0.0
    return float((left * right).sum() / denominator)


def _resolve_reward_accumulator_dtype(value: str) -> torch.dtype:
    try:
        return REWARD_ACCUMULATOR_DTYPES[str(value)]
    except KeyError as error:
        allowed = ", ".join(REWARD_ACCUMULATOR_DTYPES)
        raise ValueError(
            f"unknown reward accumulator dtype {value!r}; expected one of: {allowed}"
        ) from error


def _effective_settings(experiment: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    d1_experiment = _load_yaml(_project_path(experiment["upstream_d1"]["experiment_config"]))
    settings: dict[str, Any] = {
        "quick": quick,
        "output_directory": experiment["outputs"]["directory"],
        "checkpoints": deepcopy(d1_experiment["checkpoints"]),
        "candidate_actions": deepcopy(experiment["candidate_actions"]),
        "profile_ids": list(experiment["profile_ids"]),
        "diagnostic_conditions": deepcopy(experiment["diagnostic_conditions"]),
        "episodes_per_condition": int(experiment["diagnostic"]["episodes_per_condition"]),
        "probe_steps": [int(value) for value in experiment["diagnostic"]["probe_steps"]],
        "return_horizon_steps": int(experiment["diagnostic"]["return_horizon_steps"]),
        "reward_accumulator_dtype": str(
            experiment["diagnostic"].get("reward_accumulator_dtype", "float32")
        ),
    }
    if quick:
        quick_settings = experiment["quick"]
        arms = set(map(str, quick_settings["checkpoint_arms"]))
        seeds = set(map(int, quick_settings["policy_seeds"]))
        settings["checkpoints"] = [
            item
            for item in settings["checkpoints"]
            if str(item["arm"]) in arms and int(item["policy_seed"]) in seeds
        ]
        settings["profile_ids"] = list(quick_settings["profile_ids"])
        settings["diagnostic_conditions"] = deepcopy(
            quick_settings["diagnostic_conditions"]
        )
        settings["episodes_per_condition"] = int(
            quick_settings["episodes_per_condition"]
        )
        settings["probe_steps"] = [
            int(value) for value in quick_settings["probe_steps"]
        ]
        settings["return_horizon_steps"] = int(
            quick_settings["return_horizon_steps"]
        )
        settings["output_directory"] = experiment["outputs"]["quick_directory"]
    return settings


def _validate_seed_namespace(
    experiment: dict[str, Any],
    settings: dict[str, Any],
) -> None:
    ranges = [
        (int(item["start_inclusive"]), int(item["end_exclusive"]))
        for item in experiment["protected_seed_ranges"]
    ]
    episode_count = int(settings["episodes_per_condition"])
    intervals: list[tuple[int, int]] = []
    for condition in settings["diagnostic_conditions"]:
        start = int(condition["base_seed"])
        stop = start + episode_count
        if any(start < protected_stop and stop > protected_start for protected_start, protected_stop in ranges):
            raise RuntimeError("R3-D1-A seed interval overlaps a protected namespace")
        if stop > 4_000_000:
            raise RuntimeError("R3-D1-A diagnostic seeds enter the sealed S4-D3 range")
        intervals.append((start, stop))
    for index, (start, stop) in enumerate(intervals):
        for other_start, other_stop in intervals[index + 1 :]:
            if start < other_stop and stop > other_start:
                raise RuntimeError("R3-D1-A diagnostic seed intervals overlap")


def _probe_rows(
    bundles: list[ProbeBundle],
    candidate_actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        for candidate in candidate_actions:
            identifier = str(candidate["id"])
            result = bundle.candidates[identifier]
            row: dict[str, Any] = {
                "arm": bundle.arm,
                "policy_seed": bundle.policy_seed,
                "student_id": bundle.student_id,
                "profile_id": bundle.profile_id,
                "condition_id": bundle.condition_id,
                "probe_step": bundle.probe_step,
                "candidate": identifier,
                "episodes": int(result["q_min"].numel()),
            }
            for metric, value in result.items():
                row[f"{metric}_mean"] = float(value.mean())
            rows.append(row)
    return rows


def _episode_rows(
    bundles: list[ProbeBundle],
    candidate_actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        for candidate in candidate_actions:
            identifier = str(candidate["id"])
            result = bundle.candidates[identifier]
            for episode_index in range(int(result["q_min"].numel())):
                row: dict[str, Any] = {
                    "arm": bundle.arm,
                    "policy_seed": bundle.policy_seed,
                    "student_id": bundle.student_id,
                    "profile_id": bundle.profile_id,
                    "condition_id": bundle.condition_id,
                    "probe_step": bundle.probe_step,
                    "episode_index": episode_index,
                    "episode_seed": bundle.base_seed + episode_index,
                    "candidate": identifier,
                }
                for metric, value in result.items():
                    row[metric] = float(value[episode_index])
                rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("R3-D1-A records must not be empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
