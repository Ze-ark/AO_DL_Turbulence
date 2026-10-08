"""R3-D1-B评论家误差来源定位：软贝尔曼目标、部署目标与动作支持代理。"""

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
from torch.distributions import Normal

from src.rl.residual_sac import QNetwork, SacConfig, SquashedGaussianActor
from src.rl.s4_closed_loop_shift import _oracle_normalized_action
from src.rl.s4_r3_critic_calibration import (
    REQUIRED_CANDIDATES,
    _candidate_action,
    _pairwise_rank_accuracy,
)
from src.rl.s4_r3_failure_diagnostic import (
    _effective_settings as _d1_effective_settings,
    _load_policy,
    _load_student,
    _scenario_config,
    preflight_s4_r3_failure_diagnostic,
)
from src.rl.s4_r3_student_anchored_sac import _make_controller, _normalize_state
from src.rl.s4_representation_capacity import (
    ActionRepresentation,
    _future_disturbance_sequence,
    build_action_basis,
    representation_registration_inverse,
)
from src.rl.s4_training import (
    _file_sha256,
    _git_record,
    _load_yaml,
    _noisy_observation,
    _profiles,
    _project_path,
    _relative,
    _residual_reward,
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


@dataclass(frozen=True)
class ExtendedPolicy:
    arm: str
    policy_seed: int
    student_id: str
    actor: SquashedGaussianActor
    q1: QNetwork
    q2: QNetwork
    target_q1: QNetwork
    target_q2: QNetwork
    alpha: float
    gamma: float


@dataclass(frozen=True)
class SourceProbeBundle:
    arm: str
    policy_seed: int
    student_id: str
    profile_id: str
    condition_id: str
    base_seed: int
    probe_step: int
    candidates: dict[str, dict[str, torch.Tensor]]


def run_s4_r3_critic_source_diagnostic(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """只读定位R3评论家误差来源；不训练、不更新检查点。"""
    experiment_path = _project_path(config_path)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight, d1_experiment = preflight_s4_r3_critic_source_diagnostic(
        experiment_path, experiment, settings, quick=quick
    )
    if preflight_only:
        return preflight

    device = resolve_device("cuda")
    output_directory = _project_path(settings["output_directory"])
    if output_directory.exists():
        raise FileExistsError(
            f"R3-D1-B output already exists; preserve it for audit: {output_directory}"
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
    bundles: list[SourceProbeBundle] = []
    history_manifest: dict[str, str] = {}
    total = (
        len(settings["checkpoints"])
        * len(profiles)
        * len(conditions)
        * len(settings["probe_steps"])
        * len(settings["candidate_actions"])
    )
    completed = 0
    bar = counted_progress(total=total, description="R3评论家来源定位", unit="分支")
    preview = int(d1_experiment["teacher"]["preview_horizon_frames"])
    for checkpoint in settings["checkpoints"]:
        policy = _load_extended_policy(checkpoint, device=device)
        student = _load_student(checkpoint, experiment=d1_experiment, device=device)
        history_paths = _history_checkpoint_paths(
            checkpoint, settings["history_checkpoint_names"]
        )
        history_actors = []
        for history_path in history_paths:
            history_manifest[str(_relative(history_path))] = _file_sha256(history_path)
            history_actors.append(_load_history_actor(history_path, device=device))
        for profile_index, profile in enumerate(profiles):
            for condition_index, condition in enumerate(conditions):
                config = _scenario_config(
                    base_config,
                    condition,
                    profile,
                    episodes=int(settings["episodes_per_condition"]),
                    steps=int(settings["episode_length_steps"]),
                    preview=preview,
                )
                future_truth = _future_disturbance_sequence(
                    config=config,
                    condition=condition,
                    profile=profile,
                    length=int(settings["episode_length_steps"]) + preview,
                    basis=basis,
                    device=device,
                )
                for probe_step in settings["probe_steps"]:
                    candidate_results: dict[str, dict[str, torch.Tensor]] = {}
                    for candidate_index, candidate in enumerate(
                        settings["candidate_actions"]
                    ):
                        target_seed = (
                            int(settings["target_sampling_seed_base"])
                            + policy.policy_seed * 1_000_000
                            + profile_index * 100_000
                            + condition_index * 10_000
                            + int(probe_step) * 10
                            + candidate_index
                        )
                        result = _one_step_probe_branch(
                            d1_experiment=d1_experiment,
                            config=config,
                            condition=condition,
                            profile=profile,
                            basis=basis,
                            future_truth=future_truth,
                            registration_mapping=mappings[profile.identifier],
                            policy=policy,
                            history_actors=history_actors,
                            student=student,
                            candidate=candidate,
                            probe_step=int(probe_step),
                            target_samples=int(settings["target_samples"]),
                            target_seed=target_seed,
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
                                "贝尔曼差": float(result["soft_bellman_residual"].mean()),
                            },
                        )
                    bundles.append(
                        SourceProbeBundle(
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

    grouped = summarize_critic_sources(bundles, settings["candidate_actions"])
    interpretation = interpret_critic_sources(
        grouped, thresholds=experiment["interpretation_thresholds"], quick=quick
    )
    probe_rows = _probe_rows(bundles, settings["candidate_actions"])
    episode_rows = _episode_rows(bundles, settings["candidate_actions"])
    probe_path = output_directory / "probe_records.csv"
    episode_path = output_directory / "episode_records.csv"
    _write_csv(probe_path, probe_rows)
    _write_csv(episode_path, episode_rows)
    _write_json(output_directory / "history_checkpoint_manifest.json", history_manifest)
    alignment = _verify_d1a_alignment(
        episode_rows,
        _project_path(experiment["upstream_d1a"]["episode_records"]),
        tolerance=float(experiment["alignment"]["q_tolerance"]),
        quick=quick,
    )
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
            "status": "completed_pending_independent_audit",
            "quick": quick,
            "duration_seconds": time.perf_counter() - started,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "evidence_boundary": {
            "one_step_sac_soft_bellman_target": True,
            "exact_replay_support_reconstructed": False,
            "historical_checkpoint_support_proxy_only": True,
            "deployment_32_step_evidence_reused_from_d1a_v2": not quick,
            "causal_support_claim_authorized": False,
            "training_transitions": 0,
            "checkpoint_updates": False,
            "algorithm_changed": False,
            "sealed_s4d3_accessed": False,
            "real_slm_actions": False,
        },
        "design": {
            "candidate_actions": settings["candidate_actions"],
            "profiles": settings["profile_ids"],
            "conditions": settings["diagnostic_conditions"],
            "probe_steps": settings["probe_steps"],
            "episodes_per_condition": settings["episodes_per_condition"],
            "target_samples": settings["target_samples"],
            "target_sampling": "antithetic_gaussian_reparameterization",
            "history_checkpoint_names": settings["history_checkpoint_names"],
            "history_support_kind": "saved_policy_snapshot_proxy_not_replay_support",
        },
        "basis_diagnostics": basis_diagnostics,
        "registration_diagnostics": mapping_diagnostics,
        "d1a_alignment": alignment,
        "grouped": grouped,
        "interpretation": interpretation,
        "records": {
            "probe_records": _relative(probe_path),
            "probe_records_sha256": _file_sha256(probe_path),
            "probe_rows": len(probe_rows),
            "episode_records": _relative(episode_path),
            "episode_records_sha256": _file_sha256(episode_path),
            "episode_rows": len(episode_rows),
            "history_checkpoint_manifest": _relative(
                output_directory / "history_checkpoint_manifest.json"
            ),
            "history_checkpoint_count": len(history_manifest),
        },
        "inputs": {
            "experiment_config": _relative(experiment_path),
            "experiment_config_sha256": _file_sha256(experiment_path),
            "upstream_d1a_summary": experiment["upstream_d1a"]["summary"],
            "upstream_d1a_summary_sha256": experiment["upstream_d1a"][
                "summary_sha256"
            ],
            "source_manifest": source_manifest,
        },
        "runtime": _runtime_record(),
        "git": _git_record(),
        "next_action": (
            "Stop for a read-only audit. Do not retrain or change SAC, reward, "
            "gates, S4-D3, or real hardware before the mechanism result is audited."
        ),
    }
    _write_json(output_directory / "summary.json", summary)
    return summary


def preflight_s4_r3_critic_source_diagnostic(
    experiment_path: Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = experiment["metadata"]
    if str(metadata["stage"]) != "S4-D2-R3-D1-B":
        raise ValueError("critic-source metadata must identify S4-D2-R3-D1-B")
    if not bool(metadata.get("user_authorized_next_stage", False)):
        raise RuntimeError("R3-D1-B requires explicit user authorization")
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
            raise RuntimeError(f"R3-D1-B protection flag must remain false: {field}")

    upstream = experiment["upstream_d1a"]
    for field in (
        "summary",
        "preflight",
        "effective_config",
        "source_manifest",
        "probe_records",
        "episode_records",
        "experiment_config",
    ):
        if _file_sha256(_project_path(upstream[field])) != str(
            upstream[f"{field}_sha256"]
        ):
            raise RuntimeError(f"R3-D1-B upstream hash mismatch: {field}")
    upstream_summary = json.loads(
        _project_path(upstream["summary"]).read_text(encoding="utf-8")
    )
    if bool(upstream_summary["experiment"]["quick"]):
        raise RuntimeError("R3-D1-B cannot consume quick-smoke evidence")
    if str(upstream_summary["interpretation"]["status"]) != str(
        upstream["required_interpretation_status"]
    ):
        raise RuntimeError("R3-D1-B upstream interpretation changed")
    if int(upstream_summary["records"]["probe_rows"]) != int(
        upstream["required_probe_rows"]
    ) or int(upstream_summary["records"]["episode_rows"]) != int(
        upstream["required_episode_rows"]
    ):
        raise RuntimeError("R3-D1-B upstream record counts changed")

    d1a_experiment = _load_yaml(_project_path(upstream["experiment_config"]))
    d1_path = _project_path(d1a_experiment["upstream_d1"]["experiment_config"])
    d1_experiment = _load_yaml(d1_path)
    d1_preflight = preflight_s4_r3_failure_diagnostic(
        d1_path,
        d1_experiment,
        _d1_effective_settings(d1_experiment, quick=False),
        quick=False,
    )
    if int(d1_preflight["checkpoint_count"]) != 6:
        raise RuntimeError("R3-D1-B requires all six frozen R3 checkpoints")
    if tuple(str(item["id"]) for item in settings["candidate_actions"]) != (
        REQUIRED_CANDIDATES
    ):
        raise RuntimeError("R3-D1-B candidate action set changed")
    if int(settings["target_samples"]) <= 0 or int(settings["target_samples"]) % 2:
        raise ValueError("R3-D1-B target_samples must be a positive even number")
    if max(settings["probe_steps"]) >= int(settings["episode_length_steps"]):
        raise RuntimeError("R3-D1-B probe reaches the terminal transition")
    history_paths = [
        path
        for checkpoint in settings["checkpoints"]
        for path in _history_checkpoint_paths(
            checkpoint, settings["history_checkpoint_names"]
        )
    ]
    missing = [path for path in history_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"R3-D1-B history checkpoints missing: {missing}")
    tracked_missing = [
        path
        for path in experiment["tracked_source_files"]
        if not _project_path(path).is_file()
    ]
    if tracked_missing:
        raise FileNotFoundError(
            f"R3-D1-B tracked source files missing: {tracked_missing}"
        )
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
        "upstream_interpretation": upstream_summary["interpretation"]["status"],
        "checkpoint_count": len(settings["checkpoints"]),
        "history_checkpoint_count": len(history_paths),
        "target_samples": int(settings["target_samples"]),
        "expected_probe_states": expected_probes,
        "expected_branch_rollouts": expected_branches,
        "expected_episode_records": expected_branches
        * int(settings["episodes_per_condition"]),
        "checkpoint_writes": 0,
        "training_transitions": 0,
        "replay_buffer_reconstructed": False,
        "cuda_required": True,
        "sealed_s4d3_access": False,
        "real_slm_actions": False,
        "formal_command": (
            ".\\.venv\\Scripts\\python.exe "
            "scripts\\diagnose_s4_r3_critic_source.py --config "
            f"{str(_relative(experiment_path)).replace('/', chr(92))}"
        ),
    }
    return preflight, d1_experiment


@torch.no_grad()
def _one_step_probe_branch(
    *,
    d1_experiment: dict[str, Any],
    config: S1EnvConfig,
    condition: RobustnessCondition,
    profile: HardwareProfile,
    basis: torch.Tensor,
    future_truth: torch.Tensor,
    registration_mapping: torch.Tensor,
    policy: ExtendedPolicy,
    history_actors: list[SquashedGaussianActor],
    student: FrozenStudentPolicy,
    candidate: dict[str, Any],
    probe_step: int,
    target_samples: int,
    target_seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    environment = AdaptiveOpticsEnv(
        config, device, profile.effects_config(), basis_override=basis
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
    zeros = torch.zeros(config.batch_size, 11, device=device)
    for _ in range(probe_step):
        prefix = controller.compose_action(state, zeros)
        observation, _, _, _, _ = environment.step(prefix.composed.final_delta_rad)
        noisy = _noisy_observation(
            observation,
            config.num_modes,
            profile.observation_noise_std_rad,
            observation_generator,
        )
        state = controller.advance_observation(noisy)

    normalized_state = _normalize_state(state, student)
    actor_action = policy.actor.deterministic(normalized_state)
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
    ideal_action = (
        (oracle_added - student_scale * student_added) / correction_scale
    ).clamp(-1, 1)
    action = _candidate_action(candidate, actor=actor_action, ideal=ideal_action)
    q1 = policy.q1(normalized_state, action).squeeze(-1)
    q2 = policy.q2(normalized_state, action).squeeze(-1)
    q_min = torch.minimum(q1, q2)
    support = _history_support_metrics(normalized_state, action, history_actors)

    composed = controller.compose_action(state, action)
    next_observation, _, terminated, truncated, info = environment.step(
        composed.composed.final_delta_rad
    )
    reward = _residual_reward(
        measured_power=info["measured_power_in_bucket"],
        normalized_residual=composed.correction_normalized,
        violation=info["violation_fraction"],
        reward_config=d1_experiment["reward"],
    )
    done = terminated | truncated
    next_noisy = _noisy_observation(
        next_observation,
        config.num_modes,
        profile.observation_noise_std_rad,
        observation_generator,
    )
    next_state = _normalize_state(controller.advance_observation(next_noisy), student)
    soft_target, soft_se, entropy_bonus = _soft_target_mc(
        next_state=next_state,
        reward=reward,
        done=done,
        actor=policy.actor,
        target_q1=policy.target_q1,
        target_q2=policy.target_q2,
        alpha=policy.alpha,
        gamma=policy.gamma,
        samples=target_samples,
        seed=target_seed,
        device=device,
    )
    deterministic_next_action = policy.actor.deterministic(next_state)
    deterministic_next_q = torch.minimum(
        policy.target_q1(next_state, deterministic_next_action),
        policy.target_q2(next_state, deterministic_next_action),
    ).squeeze(-1)
    deterministic_target = reward + policy.gamma * (~done).float() * deterministic_next_q
    result = {
        "q1": q1.cpu(),
        "q2": q2.cpu(),
        "q_min": q_min.cpu(),
        "one_step_reward": reward.cpu(),
        "soft_target": soft_target.cpu(),
        "soft_target_mc_se": soft_se.cpu(),
        "soft_bellman_residual": (q_min - soft_target).cpu(),
        "deterministic_target": deterministic_target.cpu(),
        "deterministic_bellman_residual": (q_min - deterministic_target).cpu(),
        "entropy_bonus": entropy_bonus.cpu(),
        "terminal_fraction": done.float().cpu(),
        "first_action_abs_mean": action.abs().mean(dim=-1).cpu(),
        "first_action_l2": action.norm(dim=-1).cpu(),
        "warmup_range_exceed_fraction": action.abs().gt(0.25).float().mean(dim=-1).cpu(),
        **{key: value.cpu() for key, value in support.items()},
    }
    if any(not bool(torch.isfinite(value).all()) for value in result.values()):
        raise RuntimeError("R3-D1-B branch produced non-finite metrics")
    return result


@torch.no_grad()
def _soft_target_mc(
    *,
    next_state: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    actor: SquashedGaussianActor,
    target_q1: QNetwork,
    target_q2: QNetwork,
    alpha: float,
    gamma: float,
    samples: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if samples <= 0 or samples % 2:
        raise ValueError("soft target samples must be a positive even number")
    mean, log_std = actor.distribution_parameters(next_state)
    generator = torch.Generator(device=device).manual_seed(seed)
    half_noise = torch.randn(
        samples // 2,
        *mean.shape,
        device=device,
        generator=generator,
        dtype=mean.dtype,
    )
    noise = torch.cat((half_noise, -half_noise), dim=0)
    pre_tanh = mean.unsqueeze(0) + log_std.exp().unsqueeze(0) * noise
    actions = torch.tanh(pre_tanh)
    distribution = Normal(mean.unsqueeze(0), log_std.exp().unsqueeze(0))
    log_probability = (
        distribution.log_prob(pre_tanh)
        - torch.log(1 - actions.square() + 1e-6)
    ).sum(dim=-1)
    flat_state = next_state.unsqueeze(0).expand(samples, -1, -1).reshape(
        -1, next_state.shape[-1]
    )
    flat_actions = actions.reshape(-1, actions.shape[-1])
    target_q = torch.minimum(
        target_q1(flat_state, flat_actions), target_q2(flat_state, flat_actions)
    ).reshape(samples, next_state.shape[0])
    soft_value = target_q - float(alpha) * log_probability
    continuation = (~done).float()
    targets = reward.unsqueeze(0) + float(gamma) * continuation.unsqueeze(0) * soft_value
    target_mean = targets.mean(dim=0)
    target_se = targets.std(dim=0, unbiased=True) / math.sqrt(samples)
    entropy_bonus = (
        float(gamma)
        * continuation
        * (-float(alpha) * log_probability).mean(dim=0)
    )
    return target_mean, target_se, entropy_bonus


@torch.no_grad()
def _history_support_metrics(
    state: torch.Tensor,
    action: torch.Tensor,
    history_actors: list[SquashedGaussianActor],
) -> dict[str, torch.Tensor]:
    if not history_actors:
        raise ValueError("at least one history actor is required")
    deterministic = torch.stack(
        [actor.deterministic(state) for actor in history_actors], dim=0
    )
    rms = (deterministic - action.unsqueeze(0)).square().mean(dim=-1).sqrt()
    log_probabilities = torch.stack(
        [_squashed_log_probability(actor, state, action) for actor in history_actors],
        dim=0,
    )
    return {
        "history_nearest_action_rms": rms.min(dim=0).values,
        "history_mean_action_rms": rms.mean(dim=0),
        "history_max_log_probability": log_probabilities.max(dim=0).values,
        "history_mean_log_probability": log_probabilities.mean(dim=0),
    }


def _squashed_log_probability(
    actor: SquashedGaussianActor,
    state: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    clipped = action.clamp(-1 + 1e-6, 1 - 1e-6)
    pre_tanh = torch.atanh(clipped)
    mean, log_std = actor.distribution_parameters(state)
    distribution = Normal(mean, log_std.exp())
    return (
        distribution.log_prob(pre_tanh)
        - torch.log(1 - clipped.square() + 1e-6)
    ).sum(dim=-1)


def summarize_critic_sources(
    bundles: list[SourceProbeBundle],
    candidate_actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate_ids = [str(item["id"]) for item in candidate_actions]
    groups = sorted({(item.arm, item.policy_seed, item.student_id) for item in bundles})
    grouped: list[dict[str, Any]] = []
    for arm, policy_seed, student_id in groups:
        selected = [
            item
            for item in bundles
            if (item.arm, item.policy_seed, item.student_id)
            == (arm, policy_seed, student_id)
        ]
        concatenated = {
            candidate_id: {
                metric: torch.cat(
                    [bundle.candidates[candidate_id][metric] for bundle in selected]
                )
                for metric in selected[0].candidates[candidate_id]
            }
            for candidate_id in candidate_ids
        }
        cluster_ids = [
            f"{bundle.condition_id}:{episode_index}"
            for bundle in selected
            for episode_index in range(
                int(bundle.candidates[candidate_ids[0]]["q_min"].numel())
            )
        ]
        q_matrix = torch.stack(
            [concatenated[item]["q_min"] for item in candidate_ids], dim=1
        )
        target_matrix = torch.stack(
            [concatenated[item]["soft_target"] for item in candidate_ids], dim=1
        )
        actor = concatenated["actor"]
        zero = concatenated["zero"]
        actor_q = actor["q_min"] - zero["q_min"]
        actor_target = actor["soft_target"] - zero["soft_target"]
        actor_residual = (
            actor["soft_bellman_residual"] - zero["soft_bellman_residual"]
        )
        distance_delta = (
            actor["history_nearest_action_rms"]
            - zero["history_nearest_action_rms"]
        )
        logp_delta = (
            actor["history_max_log_probability"]
            - zero["history_max_log_probability"]
        )
        rank_accuracy = _pairwise_rank_accuracy(q_matrix, target_matrix)
        top_agreement = q_matrix.argmax(dim=1).eq(target_matrix.argmax(dim=1)).float()
        grouped.append(
            {
                "arm": arm,
                "policy_seed": policy_seed,
                "student_id": student_id,
                "probe_states": len(selected),
                "episode_probes": int(q_matrix.shape[0]),
                "independent_trajectory_clusters": len(set(cluster_ids)),
                "candidate_summaries": [
                    {
                        "id": item,
                        "q_min": _distribution(concatenated[item]["q_min"]),
                        "soft_target": _distribution(
                            concatenated[item]["soft_target"]
                        ),
                        "soft_bellman_residual": _distribution(
                            concatenated[item]["soft_bellman_residual"]
                        ),
                        "history_nearest_action_rms": _distribution(
                            concatenated[item]["history_nearest_action_rms"]
                        ),
                        "history_max_log_probability": _distribution(
                            concatenated[item]["history_max_log_probability"]
                        ),
                    }
                    for item in candidate_ids
                ],
                "ranking": {
                    "pairwise_q_soft_target_accuracy": _clustered_distribution(
                        rank_accuracy, cluster_ids
                    ),
                    "top_q_soft_target_agreement": _clustered_distribution(
                        top_agreement, cluster_ids
                    ),
                    "actor_q_advantage_vs_zero": _clustered_distribution(
                        actor_q, cluster_ids
                    ),
                    "actor_soft_target_advantage_vs_zero": _clustered_distribution(
                        actor_target, cluster_ids
                    ),
                    "actor_excess_bellman_residual_vs_zero": _clustered_distribution(
                        actor_residual, cluster_ids
                    ),
                },
                "support_proxy": {
                    "actor_minus_zero_nearest_rms": _clustered_distribution(
                        distance_delta, cluster_ids
                    ),
                    "actor_minus_zero_max_log_probability": _clustered_distribution(
                        logp_delta, cluster_ids
                    ),
                },
                "target_mc_se_max": max(
                    float(concatenated[item]["soft_target_mc_se"].max())
                    for item in candidate_ids
                ),
                "terminal_fraction_max": max(
                    float(concatenated[item]["terminal_fraction"].max())
                    for item in candidate_ids
                ),
            }
        )
    return grouped


def interpret_critic_sources(
    grouped: list[dict[str, Any]],
    *,
    thresholds: dict[str, Any],
    quick: bool,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for item in grouped:
        ranking = item["ranking"]
        support = item["support_proxy"]
        q_positive = float(
            ranking["actor_q_advantage_vs_zero"]["ci95_low"]
        ) > float(thresholds["positive_ci_low"])
        target_negative = float(
            ranking["actor_soft_target_advantage_vs_zero"]["ci95_high"]
        ) < float(thresholds["negative_ci_high"])
        excess_residual = float(
            ranking["actor_excess_bellman_residual_vs_zero"]["ci95_low"]
        ) > float(thresholds["positive_ci_low"])
        poor_rank = float(
            ranking["pairwise_q_soft_target_accuracy"]["mean"]
        ) < float(thresholds["minimum_consistent_rank_accuracy"])
        poor_top = float(
            ranking["top_q_soft_target_agreement"]["mean"]
        ) < float(thresholds["minimum_consistent_top_agreement"])
        bellman_misfit = q_positive and (
            target_negative or excess_residual or poor_rank or poor_top
        )
        support_proxy = (
            float(support["actor_minus_zero_nearest_rms"]["ci95_low"])
            > float(thresholds["positive_ci_low"])
            and float(
                support["actor_minus_zero_max_log_probability"]["ci95_high"]
            )
            < float(thresholds["negative_ci_high"])
        )
        results.append(
            {
                "arm": item["arm"],
                "policy_seed": item["policy_seed"],
                "soft_bellman_misfit": bellman_misfit,
                "support_proxy_associated": support_proxy,
                "actor_q_advantage_positive": q_positive,
                "actor_soft_target_advantage_negative": target_negative,
                "actor_excess_residual_positive": excess_residual,
                "poor_q_soft_target_pairwise_ranking": poor_rank,
                "poor_q_soft_target_top_agreement": poor_top,
            }
        )
    main = [item for item in results if item["arm"] == "student_backbone"]
    all_main_bellman = len(main) == 3 and all(
        item["soft_bellman_misfit"] for item in main
    )
    all_main_support = len(main) == 3 and all(
        item["support_proxy_associated"] for item in main
    )
    if quick:
        status = "QUICK_SMOKE_ONLY"
    elif all_main_bellman and all_main_support:
        status = "SOFT_BELLMAN_MISFIT_WITH_SUPPORT_PROXY_ASSOCIATION"
    elif all_main_bellman:
        status = "SOFT_BELLMAN_MISFIT_CONFIRMED"
    elif all_main_support:
        status = "DEPLOYMENT_MISMATCH_WITH_SUPPORT_PROXY_ASSOCIATION"
    else:
        status = "DEPLOYMENT_OBJECTIVE_MISMATCH_OR_MIXED"
    return {
        "status": status,
        "checkpoint_results": results,
        "all_main_seeds_soft_bellman_misfit": all_main_bellman,
        "all_main_seeds_support_proxy_association": all_main_support,
        "support_proxy_is_causal_proof": False,
        "exact_replay_support_claim_authorized": False,
        "retraining_authorized": False,
        "algorithm_change_authorized": False,
        "reward_change_authorized": False,
        "s4d3_authorized": False,
        "real_hardware_authorized": False,
        "next_rule": (
            "Audit the frozen diagnosis first. A repair may target only a mechanism "
            "confirmed across all three student-backbone seeds."
        ),
    }


def _clustered_distribution(
    values: torch.Tensor, cluster_ids: list[str]
) -> dict[str, float | int]:
    if values.ndim != 1 or values.numel() != len(cluster_ids):
        raise ValueError("clustered values and identifiers must align")
    grouped: dict[str, list[float]] = {}
    for value, identifier in zip(values.tolist(), cluster_ids, strict=True):
        grouped.setdefault(identifier, []).append(float(value))
    cluster_means = torch.tensor(
        [sum(items) / len(items) for items in grouped.values()], dtype=torch.float64
    )
    result = _distribution(cluster_means)
    result["clusters"] = len(grouped)
    result["raw_observations"] = int(values.numel())
    return result


def _distribution(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().double().flatten()
    mean = float(values.mean())
    if values.numel() <= 1:
        margin = 0.0
    else:
        margin = 1.96 * float(values.std(unbiased=True)) / math.sqrt(values.numel())
    return {
        "mean": mean,
        "median": float(values.median()),
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def _load_extended_policy(
    item: dict[str, Any], *, device: torch.device
) -> ExtendedPolicy:
    frozen = _load_policy(item, device=device)
    payload = torch.load(_project_path(item["path"]), map_location=device, weights_only=False)
    config = SacConfig(**payload["config"])
    target_q1 = QNetwork(config).to(device)
    target_q2 = QNetwork(config).to(device)
    target_q1.load_state_dict(payload["target_q1"])
    target_q2.load_state_dict(payload["target_q2"])
    for model in (target_q1, target_q2):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    alpha = float(torch.as_tensor(payload["log_alpha"]).exp())
    return ExtendedPolicy(
        arm=frozen.arm,
        policy_seed=frozen.policy_seed,
        student_id=frozen.student_id,
        actor=frozen.actor,
        q1=frozen.q1,
        q2=frozen.q2,
        target_q1=target_q1,
        target_q2=target_q2,
        alpha=alpha,
        gamma=float(config.gamma),
    )


def _load_history_actor(
    path: Path, *, device: torch.device
) -> SquashedGaussianActor:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("algorithm") != "student_anchored_residual_sac":
        raise RuntimeError(f"history checkpoint algorithm mismatch: {path}")
    config = SacConfig(**payload["config"])
    actor = SquashedGaussianActor(config).to(device)
    actor.load_state_dict(payload["actor"])
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor


def _history_checkpoint_paths(
    checkpoint: dict[str, Any], names: list[str]
) -> list[Path]:
    directory = _project_path(checkpoint["path"]).parent
    return [directory / name for name in names]


def _effective_settings(experiment: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    d1a = _load_yaml(_project_path(experiment["upstream_d1a"]["experiment_config"]))
    d1 = _load_yaml(_project_path(d1a["upstream_d1"]["experiment_config"]))
    settings: dict[str, Any] = {
        "quick": quick,
        "output_directory": experiment["outputs"]["directory"],
        "checkpoints": deepcopy(d1["checkpoints"]),
        "candidate_actions": deepcopy(d1a["candidate_actions"]),
        "profile_ids": list(d1a["profile_ids"]),
        "diagnostic_conditions": deepcopy(d1a["diagnostic_conditions"]),
        "episodes_per_condition": int(d1a["diagnostic"]["episodes_per_condition"]),
        "probe_steps": [int(item) for item in d1a["diagnostic"]["probe_steps"]],
        "episode_length_steps": int(experiment["diagnostic"]["episode_length_steps"]),
        "target_samples": int(experiment["diagnostic"]["target_samples"]),
        "target_sampling_seed_base": int(
            experiment["diagnostic"]["target_sampling_seed_base"]
        ),
        "history_checkpoint_names": list(
            experiment["diagnostic"]["history_checkpoint_names"]
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
        settings["probe_steps"] = [int(item) for item in quick_settings["probe_steps"]]
        settings["target_samples"] = int(quick_settings["target_samples"])
        settings["output_directory"] = experiment["outputs"]["quick_directory"]
    return settings


def _verify_d1a_alignment(
    rows: list[dict[str, Any]],
    upstream_path: Path,
    *,
    tolerance: float,
    quick: bool,
) -> dict[str, Any]:
    if quick:
        return {"checked": False, "reason": "quick smoke uses a separate seed"}
    keys = (
        "arm",
        "policy_seed",
        "profile_id",
        "condition_id",
        "probe_step",
        "episode_index",
        "candidate",
    )
    with upstream_path.open("r", encoding="utf-8", newline="") as handle:
        upstream = {
            tuple(str(row[key]) for key in keys): float(row["q_min"])
            for row in csv.DictReader(handle)
        }
    errors = []
    for row in rows:
        key = tuple(str(row[item]) for item in keys)
        if key not in upstream:
            raise RuntimeError(f"R3-D1-B missing upstream alignment key: {key}")
        errors.append(abs(float(row["q_min"]) - upstream[key]))
    maximum = max(errors, default=0.0)
    if maximum > tolerance or len(rows) != len(upstream):
        raise RuntimeError("R3-D1-B truth alignment with D1-A-V2 failed")
    return {
        "checked": True,
        "rows": len(rows),
        "q_max_abs_error": maximum,
        "tolerance": tolerance,
        "pass": True,
    }


def _probe_rows(
    bundles: list[SourceProbeBundle], candidate_actions: list[dict[str, Any]]
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
    bundles: list[SourceProbeBundle], candidate_actions: list[dict[str, Any]]
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
        raise ValueError("R3-D1-B records must not be empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
