"""R3-D1-C确定性部署多步目标偏差曲线诊断。"""

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

from src.rl.s4_r3_critic_source_diagnostic import (
    ExtendedPolicy,
    _clustered_distribution,
    _distribution,
    _load_extended_policy,
)
from src.rl.s4_r3_failure_diagnostic import (
    _effective_settings as _d1_effective_settings,
    _load_student,
    _scenario_config,
    preflight_s4_r3_failure_diagnostic,
)
from src.rl.s4_r3_student_anchored_sac import _make_controller, _normalize_state
from src.rl.s4_representation_capacity import (
    ActionRepresentation,
    build_action_basis,
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


REQUIRED_CANDIDATES = ("zero", "actor")


@dataclass(frozen=True)
class MultiStepBundle:
    arm: str
    policy_seed: int
    student_id: str
    profile_id: str
    condition_id: str
    base_seed: int
    probe_step: int
    candidates: dict[str, dict[str, torch.Tensor]]


def run_s4_r3_multistep_target_diagnostic(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """只读生成多步目标偏差曲线，不训练、不更新检查点。"""
    experiment_path = _project_path(config_path)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight, d1_experiment = preflight_s4_r3_multistep_target_diagnostic(
        experiment_path, experiment, settings, quick=quick
    )
    if preflight_only:
        return preflight

    device = resolve_device("cuda")
    output_directory = _project_path(settings["output_directory"])
    if output_directory.exists():
        raise FileExistsError(
            f"R3-D1-C output already exists; preserve it for audit: {output_directory}"
        )
    output_directory.mkdir(parents=True)
    _write_json(output_directory / "preflight.json", preflight)
    _write_json(output_directory / "effective_config.json", settings)
    source_manifest = _source_manifest(experiment["tracked_source_files"])
    _write_json(output_directory / "source_manifest.json", source_manifest)

    base_config, _ = load_s1_config(_project_path(d1_experiment["environment_config"]))
    representation = ActionRepresentation.from_mapping(d1_experiment["representation"])
    base_config = replace(base_config, num_modes=representation.num_modes)
    basis, _, basis_diagnostics = build_action_basis(
        base_config, representation, device
    )
    profiles = _profiles(d1_experiment, settings["profile_ids"])
    conditions = [
        RobustnessCondition.from_mapping(item)
        for item in settings["diagnostic_conditions"]
    ]

    started = time.perf_counter()
    bundles: list[MultiStepBundle] = []
    total = (
        len(settings["checkpoints"])
        * len(profiles)
        * len(conditions)
        * len(settings["probe_steps"])
        * len(settings["candidate_actions"])
    )
    completed = 0
    bar = counted_progress(total=total, description="R3多步目标曲线", unit="分支")
    preview = int(d1_experiment["teacher"]["preview_horizon_frames"])
    for checkpoint in settings["checkpoints"]:
        policy = _load_extended_policy(checkpoint, device=device)
        student = _load_student(checkpoint, experiment=d1_experiment, device=device)
        for profile in profiles:
            for condition in conditions:
                config = _scenario_config(
                    base_config,
                    condition,
                    profile,
                    episodes=int(settings["episodes_per_condition"]),
                    steps=int(settings["episode_length_steps"]),
                    preview=preview,
                )
                for probe_step in settings["probe_steps"]:
                    candidate_results: dict[str, dict[str, torch.Tensor]] = {}
                    for candidate in settings["candidate_actions"]:
                        result = _rollout_multistep_branch(
                            d1_experiment=d1_experiment,
                            config=config,
                            condition=condition,
                            profile=profile,
                            basis=basis,
                            policy=policy,
                            student=student,
                            candidate_id=str(candidate["id"]),
                            probe_step=int(probe_step),
                            horizons=settings["horizons"],
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
                                "32步": float(
                                    result["empirical_reward_return"][:, -1].mean()
                                ),
                            },
                        )
                    bundles.append(
                        MultiStepBundle(
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

    grouped = summarize_multistep_targets(
        bundles, horizons=settings["horizons"]
    )
    interpretation = interpret_multistep_targets(grouped, quick=quick)
    probe_rows = _probe_rows(bundles, settings["horizons"])
    episode_rows = _episode_rows(bundles, settings["horizons"])
    probe_path = output_directory / "probe_records.csv"
    episode_path = output_directory / "episode_records.csv"
    _write_csv(probe_path, probe_rows)
    _write_csv(episode_path, episode_rows)
    alignment = _verify_upstream_alignment(
        episode_rows,
        experiment=experiment,
        tolerance=float(experiment["alignment"]["tolerance"]),
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
            "deterministic_deployment_multistep_curve": True,
            "exact_stochastic_soft_q_curve": False,
            "reuses_d1a_d1b_physical_states": not quick,
            "training_transitions": 0,
            "checkpoint_updates": False,
            "reward_changed": False,
            "algorithm_changed": False,
            "sealed_s4d3_accessed": False,
            "real_slm_actions": False,
        },
        "design": {
            "candidate_actions": settings["candidate_actions"],
            "horizons": settings["horizons"],
            "profiles": settings["profile_ids"],
            "conditions": settings["diagnostic_conditions"],
            "probe_steps": settings["probe_steps"],
            "episodes_per_condition": settings["episodes_per_condition"],
            "first_action": "zero_or_frozen_actor",
            "continuation_policy": "deterministic_frozen_actor",
            "bootstrap": "frozen_target_critic_with_deterministic_actor",
            "reward_accumulator_dtype": "float64",
        },
        "basis_diagnostics": basis_diagnostics,
        "upstream_alignment": alignment,
        "grouped": grouped,
        "interpretation": interpretation,
        "records": {
            "probe_records": _relative(probe_path),
            "probe_records_sha256": _file_sha256(probe_path),
            "probe_rows": len(probe_rows),
            "episode_records": _relative(episode_path),
            "episode_records_sha256": _file_sha256(episode_path),
            "episode_rows": len(episode_rows),
            "horizon_records": len(episode_rows) * len(settings["horizons"]),
        },
        "inputs": {
            "experiment_config": _relative(experiment_path),
            "experiment_config_sha256": _file_sha256(experiment_path),
            "upstream_d1b_summary": experiment["upstream_d1b"]["summary"],
            "upstream_d1b_summary_sha256": experiment["upstream_d1b"][
                "summary_sha256"
            ],
            "source_manifest": source_manifest,
        },
        "runtime": _runtime_record(),
        "git": _git_record(),
        "next_action": (
            "Stop for read-only audit. Do not retrain or select a critic repair "
            "until the multistep sign-change curve is independently checked."
        ),
    }
    _write_json(output_directory / "summary.json", summary)
    return summary


def preflight_s4_r3_multistep_target_diagnostic(
    experiment_path: Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = experiment["metadata"]
    if str(metadata["stage"]) != "S4-D2-R3-D1-C":
        raise ValueError("multistep-target metadata must identify S4-D2-R3-D1-C")
    if not bool(metadata.get("user_authorized_next_stage", False)):
        raise RuntimeError("R3-D1-C requires explicit user authorization")
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
            raise RuntimeError(f"R3-D1-C protection flag must remain false: {field}")

    upstream = experiment["upstream_d1b"]
    for field in (
        "summary",
        "preflight",
        "effective_config",
        "source_manifest",
        "probe_records",
        "episode_records",
        "history_checkpoint_manifest",
        "experiment_config",
        "audit_record",
    ):
        if _file_sha256(_project_path(upstream[field])) != str(
            upstream[f"{field}_sha256"]
        ):
            raise RuntimeError(f"R3-D1-C upstream hash mismatch: {field}")
    upstream_summary = json.loads(
        _project_path(upstream["summary"]).read_text(encoding="utf-8")
    )
    if bool(upstream_summary["experiment"]["quick"]):
        raise RuntimeError("R3-D1-C cannot consume quick-smoke evidence")
    if str(upstream_summary["interpretation"]["status"]) != str(
        upstream["required_interpretation_status"]
    ):
        raise RuntimeError("R3-D1-C upstream interpretation changed")
    if not bool(upstream_summary["d1a_alignment"]["pass"]):
        raise RuntimeError("R3-D1-C requires passing D1-A alignment")
    if int(upstream_summary["records"]["probe_rows"]) != int(
        upstream["required_probe_rows"]
    ) or int(upstream_summary["records"]["episode_rows"]) != int(
        upstream["required_episode_rows"]
    ):
        raise RuntimeError("R3-D1-C upstream record counts changed")
    audit_text = _project_path(upstream["audit_record"]).read_text(encoding="utf-8")
    if "Verification Status: `ANALYZED / SOFT-BELLMAN-MISFIT-CONFIRMED`" not in audit_text:
        raise RuntimeError("R3-D1-C upstream audit is not analyzed")

    d1b = _load_yaml(_project_path(upstream["experiment_config"]))
    d1a = _load_yaml(_project_path(d1b["upstream_d1a"]["experiment_config"]))
    d1_path = _project_path(d1a["upstream_d1"]["experiment_config"])
    d1_experiment = _load_yaml(d1_path)
    d1_preflight = preflight_s4_r3_failure_diagnostic(
        d1_path,
        d1_experiment,
        _d1_effective_settings(d1_experiment, quick=False),
        quick=False,
    )
    if int(d1_preflight["checkpoint_count"]) != 6:
        raise RuntimeError("R3-D1-C requires all six frozen R3 checkpoints")
    candidate_ids = tuple(str(item["id"]) for item in settings["candidate_actions"])
    if candidate_ids != REQUIRED_CANDIDATES:
        raise RuntimeError("R3-D1-C must compare only zero and actor")
    horizons = settings["horizons"]
    if horizons != sorted(set(horizons)) or horizons[0] != 1:
        raise RuntimeError("R3-D1-C horizons must be sorted, unique, and start at one")
    if max(settings["probe_steps"]) + max(horizons) > int(
        settings["episode_length_steps"]
    ):
        raise RuntimeError("R3-D1-C curve exceeds the locked episode")
    missing = [
        path
        for path in experiment["tracked_source_files"]
        if not _project_path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"R3-D1-C tracked source files missing: {missing}")
    expected_probes = (
        len(settings["checkpoints"])
        * len(settings["profile_ids"])
        * len(settings["diagnostic_conditions"])
        * len(settings["probe_steps"])
    )
    expected_branches = expected_probes * len(settings["candidate_actions"])
    expected_episodes = expected_branches * int(settings["episodes_per_condition"])
    preflight = {
        "status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_DIAGNOSTIC",
        "quick": quick,
        "upstream_interpretation": upstream_summary["interpretation"]["status"],
        "checkpoint_count": len(settings["checkpoints"]),
        "candidate_ids": list(candidate_ids),
        "horizons": horizons,
        "expected_probe_states": expected_probes,
        "expected_branch_rollouts": expected_branches,
        "expected_episode_records": expected_episodes,
        "expected_horizon_records": expected_episodes * len(horizons),
        "checkpoint_writes": 0,
        "training_transitions": 0,
        "cuda_required": True,
        "sealed_s4d3_access": False,
        "real_slm_actions": False,
        "formal_command": (
            ".\\.venv\\Scripts\\python.exe "
            "scripts\\diagnose_s4_r3_multistep_target.py --config "
            f"{str(_relative(experiment_path)).replace('/', chr(92))}"
        ),
    }
    return preflight, d1_experiment


@torch.no_grad()
def _rollout_multistep_branch(
    *,
    d1_experiment: dict[str, Any],
    config: S1EnvConfig,
    condition: RobustnessCondition,
    profile: HardwareProfile,
    basis: torch.Tensor,
    policy: ExtendedPolicy,
    student: FrozenStudentPolicy,
    candidate_id: str,
    probe_step: int,
    horizons: list[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if candidate_id not in REQUIRED_CANDIDATES:
        raise ValueError(f"unsupported R3-D1-C candidate: {candidate_id}")
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
    first_action = zeros if candidate_id == "zero" else actor_action
    q1 = policy.q1(normalized_state, first_action).squeeze(-1)
    q2 = policy.q2(normalized_state, first_action).squeeze(-1)
    q_min = torch.minimum(q1, q2)
    reward_return = torch.zeros(config.batch_size, device=device, dtype=torch.float64)
    power_return = torch.zeros_like(reward_return)
    reward_curves: list[torch.Tensor] = []
    power_curves: list[torch.Tensor] = []
    target_curves: list[torch.Tensor] = []
    bootstrap_curves: list[torch.Tensor] = []
    horizon_set = set(horizons)
    terminal_max = torch.zeros(config.batch_size, device=device)
    for local_step in range(1, max(horizons) + 1):
        correction = (
            first_action
            if local_step == 1
            else policy.actor.deterministic(_normalize_state(state, student))
        )
        action = controller.compose_action(state, correction)
        observation, _, terminated, truncated, info = environment.step(
            action.composed.final_delta_rad
        )
        reward = _residual_reward(
            measured_power=info["measured_power_in_bucket"],
            normalized_residual=action.correction_normalized,
            violation=info["violation_fraction"],
            reward_config=d1_experiment["reward"],
        )
        discount = policy.gamma ** (local_step - 1)
        reward_return += discount * reward.double()
        power_return += discount * info["reward_power_in_bucket"].double()
        terminal_max = torch.maximum(
            terminal_max, (terminated | truncated).float()
        )
        next_noisy = _noisy_observation(
            observation,
            config.num_modes,
            profile.observation_noise_std_rad,
            observation_generator,
        )
        next_state = controller.advance_observation(next_noisy)
        if local_step in horizon_set:
            normalized_next = _normalize_state(next_state, student)
            bootstrap_action = policy.actor.deterministic(normalized_next)
            bootstrap_q = torch.minimum(
                policy.target_q1(normalized_next, bootstrap_action),
                policy.target_q2(normalized_next, bootstrap_action),
            ).squeeze(-1)
            bootstrap = (policy.gamma**local_step) * bootstrap_q.double()
            reward_curves.append(reward_return.clone())
            power_curves.append(power_return.clone())
            bootstrap_curves.append(bootstrap)
            target_curves.append(reward_return + bootstrap)
        state = next_state
    result = {
        "q1": q1.cpu(),
        "q2": q2.cpu(),
        "q_min": q_min.cpu(),
        "empirical_reward_return": torch.stack(reward_curves, dim=1).cpu(),
        "empirical_true_power_return": torch.stack(power_curves, dim=1).cpu(),
        "bootstrap_component": torch.stack(bootstrap_curves, dim=1).cpu(),
        "deterministic_bootstrap_target": torch.stack(target_curves, dim=1).cpu(),
        "q_minus_bootstrap_target": (
            q_min.double().unsqueeze(1) - torch.stack(target_curves, dim=1)
        ).cpu(),
        "terminal_fraction": terminal_max.cpu(),
    }
    if any(not bool(torch.isfinite(value).all()) for value in result.values()):
        raise RuntimeError("R3-D1-C branch produced non-finite metrics")
    return result


def summarize_multistep_targets(
    bundles: list[MultiStepBundle], *, horizons: list[int]
) -> list[dict[str, Any]]:
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
            candidate: {
                metric: torch.cat(
                    [bundle.candidates[candidate][metric] for bundle in selected]
                )
                for metric in selected[0].candidates[candidate]
            }
            for candidate in REQUIRED_CANDIDATES
        }
        cluster_ids = [
            f"{bundle.condition_id}:{episode_index}"
            for bundle in selected
            for episode_index in range(
                int(bundle.candidates["zero"]["q_min"].numel())
            )
        ]
        actor = concatenated["actor"]
        zero = concatenated["zero"]
        q_advantage = actor["q_min"] - zero["q_min"]
        horizons_summary: list[dict[str, Any]] = []
        for index, horizon in enumerate(horizons):
            target_advantage = (
                actor["deterministic_bootstrap_target"][:, index]
                - zero["deterministic_bootstrap_target"][:, index]
            )
            empirical_advantage = (
                actor["empirical_reward_return"][:, index]
                - zero["empirical_reward_return"][:, index]
            )
            power_advantage = (
                actor["empirical_true_power_return"][:, index]
                - zero["empirical_true_power_return"][:, index]
            )
            bootstrap_advantage = (
                actor["bootstrap_component"][:, index]
                - zero["bootstrap_component"][:, index]
            )
            horizons_summary.append(
                {
                    "horizon": horizon,
                    "q_advantage_vs_zero": _clustered_distribution(
                        q_advantage, cluster_ids
                    ),
                    "deterministic_target_advantage_vs_zero": _clustered_distribution(
                        target_advantage, cluster_ids
                    ),
                    "empirical_reward_advantage_vs_zero": _clustered_distribution(
                        empirical_advantage, cluster_ids
                    ),
                    "empirical_true_power_advantage_vs_zero": _clustered_distribution(
                        power_advantage, cluster_ids
                    ),
                    "bootstrap_component_advantage_vs_zero": _clustered_distribution(
                        bootstrap_advantage, cluster_ids
                    ),
                    "online_q_excess_vs_target": _clustered_distribution(
                        q_advantage - target_advantage, cluster_ids
                    ),
                }
            )
        point_flip = next(
            (
                item["horizon"]
                for item in horizons_summary
                if float(item["deterministic_target_advantage_vs_zero"]["mean"])
                <= 0
            ),
            None,
        )
        confirmed_flip = next(
            (
                item["horizon"]
                for item in horizons_summary
                if float(
                    item["deterministic_target_advantage_vs_zero"]["ci95_high"]
                )
                < 0
            ),
            None,
        )
        grouped.append(
            {
                "arm": arm,
                "policy_seed": policy_seed,
                "student_id": student_id,
                "probe_states": len(selected),
                "episode_probes": int(q_advantage.numel()),
                "independent_trajectory_clusters": len(set(cluster_ids)),
                "q_advantage_vs_zero": _clustered_distribution(
                    q_advantage, cluster_ids
                ),
                "horizons": horizons_summary,
                "point_estimate_target_sign_flip_horizon": point_flip,
                "confirmed_negative_target_horizon": confirmed_flip,
                "terminal_fraction_max": max(
                    float(concatenated[item]["terminal_fraction"].max())
                    for item in REQUIRED_CANDIDATES
                ),
            }
        )
    return grouped


def interpret_multistep_targets(
    grouped: list[dict[str, Any]], *, quick: bool
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for item in grouped:
        first = item["horizons"][0]
        last = item["horizons"][-1]
        q_positive = float(item["q_advantage_vs_zero"]["ci95_low"]) > 0
        first_empirical_negative = float(
            first["empirical_reward_advantage_vs_zero"]["ci95_high"]
        ) < 0
        first_target_positive = float(
            first["deterministic_target_advantage_vs_zero"]["ci95_low"]
        ) > 0
        last_empirical_negative = float(
            last["empirical_reward_advantage_vs_zero"]["ci95_high"]
        ) < 0
        last_target_positive = float(
            last["deterministic_target_advantage_vs_zero"]["ci95_low"]
        ) > 0
        one_step_optimism = q_positive and first_empirical_negative and first_target_positive
        persistent = one_step_optimism and last_empirical_negative and last_target_positive
        corrected = one_step_optimism and item["confirmed_negative_target_horizon"] is not None
        results.append(
            {
                "arm": item["arm"],
                "policy_seed": item["policy_seed"],
                "one_step_bootstrap_optimism": one_step_optimism,
                "persistent_through_max_horizon": persistent,
                "confirmed_finite_horizon_correction": corrected,
                "point_estimate_target_sign_flip_horizon": item[
                    "point_estimate_target_sign_flip_horizon"
                ],
                "confirmed_negative_target_horizon": item[
                    "confirmed_negative_target_horizon"
                ],
            }
        )
    main = [item for item in results if item["arm"] == "student_backbone"]
    all_one_step = len(main) == 3 and all(
        item["one_step_bootstrap_optimism"] for item in main
    )
    all_persistent = len(main) == 3 and all(
        item["persistent_through_max_horizon"] for item in main
    )
    all_corrected = len(main) == 3 and all(
        item["confirmed_finite_horizon_correction"] for item in main
    )
    if quick:
        status = "QUICK_SMOKE_ONLY"
    elif all_persistent:
        status = "PERSISTENT_BOOTSTRAP_OPTIMISM_CONFIRMED"
    elif all_one_step and all_corrected:
        status = "BOOTSTRAP_OPTIMISM_WITH_FINITE_HORIZON_CORRECTION"
    elif all_one_step:
        status = "BOOTSTRAP_OPTIMISM_DECAYS_WITHOUT_UNIFORM_SIGN_FLIP"
    else:
        status = "MULTISTEP_MECHANISM_MIXED_OR_UNCONFIRMED"
    return {
        "status": status,
        "checkpoint_results": results,
        "all_main_seeds_one_step_bootstrap_optimism": all_one_step,
        "all_main_seeds_persistent_through_max_horizon": all_persistent,
        "all_main_seeds_confirmed_finite_horizon_correction": all_corrected,
        "exact_stochastic_soft_q_claim_authorized": False,
        "retraining_authorized": False,
        "algorithm_change_authorized": False,
        "reward_change_authorized": False,
        "s4d3_authorized": False,
        "real_hardware_authorized": False,
        "next_rule": (
            "Audit first. A future critic repair may use the earliest consistently "
            "correcting horizon, but only after all three main seeds agree."
        ),
    }


def _effective_settings(experiment: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    d1b = _load_yaml(_project_path(experiment["upstream_d1b"]["experiment_config"]))
    d1a = _load_yaml(_project_path(d1b["upstream_d1a"]["experiment_config"]))
    d1 = _load_yaml(_project_path(d1a["upstream_d1"]["experiment_config"]))
    settings: dict[str, Any] = {
        "quick": quick,
        "output_directory": experiment["outputs"]["directory"],
        "checkpoints": deepcopy(d1["checkpoints"]),
        "candidate_actions": deepcopy(experiment["candidate_actions"]),
        "profile_ids": list(d1a["profile_ids"]),
        "diagnostic_conditions": deepcopy(d1a["diagnostic_conditions"]),
        "episodes_per_condition": int(d1a["diagnostic"]["episodes_per_condition"]),
        "probe_steps": [int(item) for item in d1a["diagnostic"]["probe_steps"]],
        "episode_length_steps": int(experiment["diagnostic"]["episode_length_steps"]),
        "horizons": [int(item) for item in experiment["diagnostic"]["horizons"]],
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
        settings["horizons"] = [int(item) for item in quick_settings["horizons"]]
        settings["output_directory"] = experiment["outputs"]["quick_directory"]
    return settings


def _verify_upstream_alignment(
    rows: list[dict[str, Any]],
    *,
    experiment: dict[str, Any],
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
    d1b_path = _project_path(experiment["upstream_d1b"]["episode_records"])
    with d1b_path.open("r", encoding="utf-8", newline="") as handle:
        d1b = {
            tuple(str(row[key]) for key in keys): row
            for row in csv.DictReader(handle)
            if row["candidate"] in REQUIRED_CANDIDATES
        }
    d1b_config = _load_yaml(
        _project_path(experiment["upstream_d1b"]["experiment_config"])
    )
    d1a_path = _project_path(d1b_config["upstream_d1a"]["episode_records"])
    with d1a_path.open("r", encoding="utf-8", newline="") as handle:
        d1a = {
            tuple(str(row[key]) for key in keys): row
            for row in csv.DictReader(handle)
            if row["candidate"] in REQUIRED_CANDIDATES
        }
    errors = {"q": [], "reward_h1": [], "target_h1": [], "reward_h32": [], "power_h32": []}
    for row in rows:
        key = tuple(str(row[item]) for item in keys)
        if key not in d1a or key not in d1b:
            raise RuntimeError(f"R3-D1-C missing upstream alignment key: {key}")
        errors["q"].append(abs(float(row["q_min"]) - float(d1b[key]["q_min"])))
        errors["reward_h1"].append(
            abs(
                float(row["empirical_reward_return_h1"])
                - float(d1b[key]["one_step_reward"])
            )
        )
        errors["target_h1"].append(
            abs(
                float(row["deterministic_bootstrap_target_h1"])
                - float(d1b[key]["deterministic_target"])
            )
        )
        errors["reward_h32"].append(
            abs(
                float(row["empirical_reward_return_h32"])
                - float(d1a[key]["discounted_reward_return"])
            )
        )
        errors["power_h32"].append(
            abs(
                float(row["empirical_true_power_return_h32"])
                - float(d1a[key]["discounted_true_power_return"])
            )
        )
    maxima = {key: max(values, default=0.0) for key, values in errors.items()}
    if len(rows) != len(d1a) or len(rows) != len(d1b) or max(maxima.values()) > tolerance:
        raise RuntimeError("R3-D1-C alignment with D1-A/D1-B failed")
    return {"checked": True, "rows": len(rows), "max_abs_errors": maxima, "tolerance": tolerance, "pass": True}


def _probe_rows(
    bundles: list[MultiStepBundle], horizons: list[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        for candidate in REQUIRED_CANDIDATES:
            result = bundle.candidates[candidate]
            row: dict[str, Any] = {
                "arm": bundle.arm,
                "policy_seed": bundle.policy_seed,
                "student_id": bundle.student_id,
                "profile_id": bundle.profile_id,
                "condition_id": bundle.condition_id,
                "probe_step": bundle.probe_step,
                "candidate": candidate,
                "episodes": int(result["q_min"].numel()),
                "q_min_mean": float(result["q_min"].mean()),
                "terminal_fraction_max": float(result["terminal_fraction"].max()),
            }
            for index, horizon in enumerate(horizons):
                for metric in (
                    "empirical_reward_return",
                    "empirical_true_power_return",
                    "bootstrap_component",
                    "deterministic_bootstrap_target",
                    "q_minus_bootstrap_target",
                ):
                    row[f"{metric}_h{horizon}_mean"] = float(
                        result[metric][:, index].mean()
                    )
            rows.append(row)
    return rows


def _episode_rows(
    bundles: list[MultiStepBundle], horizons: list[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        for candidate in REQUIRED_CANDIDATES:
            result = bundle.candidates[candidate]
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
                    "candidate": candidate,
                    "q_min": float(result["q_min"][episode_index]),
                    "terminal_fraction": float(result["terminal_fraction"][episode_index]),
                }
                for index, horizon in enumerate(horizons):
                    for metric in (
                        "empirical_reward_return",
                        "empirical_true_power_return",
                        "bootstrap_component",
                        "deterministic_bootstrap_target",
                        "q_minus_bootstrap_target",
                    ):
                        row[f"{metric}_h{horizon}"] = float(
                            result[metric][episode_index, index]
                        )
                rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("R3-D1-C records must not be empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
