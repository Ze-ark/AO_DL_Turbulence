"""S4-D2-R3学生锚定残差SAC的训练、配对评估与证据封存。"""

from __future__ import annotations

import csv
from collections import deque
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterable

import torch

from src.rl.residual_control import AnchoredResidualTrackingController
from src.rl.residual_sac import ResidualSacAgent, SacConfig, TransitionReplayBuffer
from src.rl.s4_closed_loop_shift import _rollout_teacher
from src.rl.s4_high_order_learnability import _paired_science_summary
from src.rl.s4_representation_capacity import (
    ANCHOR_MODES,
    ActionRepresentation,
    _future_disturbance_sequence,
    build_action_basis,
    representation_registration_inverse,
)
from src.rl.s4_training import (
    _append_jsonl,
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
from src.rl.student_anchored_control import (
    FrozenStudentPolicy,
    StudentAnchoredResidualController,
    initialize_actor_from_student,
    load_frozen_student,
)
from src.runtime import resolve_device
from src.simulation.config import S1EnvConfig, load_s1_config
from src.simulation.controllers import TrackingLeakyIntegratorController
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


def run_s4_r3_student_anchored_sac(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """运行R3；正式模式必须由用户在IDE终端手动启动。"""
    experiment_path = _project_path(config_path)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight = preflight_s4_r3(experiment_path, experiment, settings, quick=quick)
    if preflight_only:
        return preflight

    device = resolve_device("cuda")
    output_directory = _project_path(settings["output_directory"])
    if output_directory.exists():
        raise FileExistsError(
            "R3 output already exists; preserve it and diagnose before retrying: "
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
    profiles = _profiles(experiment, settings["training_hardware_profile_ids"])
    evaluation_profiles = _profiles(
        experiment, settings["final_validation_profile_ids"]
    )
    registration_mappings: dict[str, torch.Tensor] = {}
    registration_diagnostics: dict[str, dict[str, Any]] = {}
    for profile in evaluation_profiles:
        mapping, diagnostics = representation_registration_inverse(
            basis=basis,
            pupil=pupil,
            profile=profile,
            rcond=float(experiment["representation"]["pseudoinverse_rcond"]),
        )
        registration_mappings[profile.identifier] = mapping
        registration_diagnostics[profile.identifier] = diagnostics

    started = time.perf_counter()
    run_results: list[dict[str, Any]] = []
    total_runs = len(settings["arms"]) * len(settings["policy_seeds"])
    run_index = 0
    for arm in settings["arms"]:
        arm_directory = output_directory / str(arm)
        arm_directory.mkdir()
        for pairing_index, (policy_seed, environment_seed_base, student_item) in enumerate(
            zip(
                settings["policy_seeds"],
                settings["training_environment_seed_bases"],
                settings["student_checkpoints"],
                strict=True,
            )
        ):
            run_index += 1
            seed_directory = arm_directory / f"policy_seed_{policy_seed}"
            seed_directory.mkdir()
            student = _load_student_item(
                student_item,
                experiment=experiment,
                device=device,
            )
            result = _train_one_run(
                arm=str(arm),
                experiment=experiment,
                settings=settings,
                base_config=base_config,
                basis=basis,
                training_profiles=profiles,
                evaluation_profiles=evaluation_profiles,
                registration_mappings=registration_mappings,
                student=student,
                policy_seed=int(policy_seed),
                environment_seed_base=int(environment_seed_base),
                output_directory=seed_directory,
                run_index=run_index,
                total_runs=total_runs,
                device=device,
            )
            result["student_pairing_index"] = pairing_index
            run_results.append(result)
            _write_json(
                output_directory / "partial_summary.json",
                {"completed_runs": run_results},
            )

    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "s4d2_r3_student_anchored_residual_sac_v1",
        },
        "experiment": {
            "id": "AO-S4-D2-R3-STUDENT-ANCHORED-RESIDUAL-SAC",
            "status": "completed_pending_independent_audit",
            "quick": quick,
            "duration_seconds": time.perf_counter() - started,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "evidence_boundary": {
            "software_simulation_only": True,
            "real_slm_actions": False,
            "sealed_s4d3_accessed": False,
            "student_model_frozen": True,
            "traditional_anchor_modes": ANCHOR_MODES,
            "sac_correction_modes": representation.num_modes - ANCHOR_MODES,
            "algorithm_changed": False,
            "reward_changed": False,
            "interpretation": (
                "R3只验证冻结学生基座上的小幅SAC增量。正式结果仍需独立审计，"
                "不能直接外推到真实SLM或自然大气。"
            ),
        },
        "inputs": {
            "experiment_config": _relative(experiment_path),
            "experiment_config_sha256": _file_sha256(experiment_path),
            "source_manifest": source_manifest,
            "upstream_status": preflight["upstream_status"],
        },
        "design": {
            "arms": settings["arms"],
            "policy_seeds": settings["policy_seeds"],
            "student_scale": float(experiment["student_anchor"]["deployment_scale"]),
            "correction_component_limit_rad": float(
                experiment["action"]["correction_component_limit_rad"]
            ),
            "paired_trajectories": True,
            "main_arm_required": True,
        },
        "basis_diagnostics": basis_diagnostics,
        "registration_diagnostics": registration_diagnostics,
        "run_results": run_results,
        "development_summary": _summarize_runs(run_results, quick=quick),
        "runtime": _runtime_record(),
        "git": _git_record(),
        "next_action": (
            "停止并让助手只读审计R3结果；不要自动重训、打开S4-D3或操作真实SLM。"
        ),
    }
    _write_json(output_directory / "summary.json", summary)
    return summary


def preflight_s4_r3(
    experiment_path: Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    metadata = experiment["metadata"]
    if str(metadata["stage"]) != "S4-D2-R3":
        raise ValueError("experiment metadata must identify S4-D2-R3")
    if not bool(metadata.get("user_authorized_next_stage", False)):
        raise RuntimeError("R3 requires explicit user authorization")
    for field in (
        "allow_real_hardware_actions",
        "allow_sealed_test_access",
        "allow_algorithm_change",
        "allow_reward_change",
        "allow_total_action_budget_increase",
        "allow_student_finetuning",
        "allow_second_aggregation_round",
    ):
        if bool(metadata.get(field, True)):
            raise RuntimeError(f"R3 protection flag must remain false: {field}")

    upstream = _verify_upstream(experiment["upstream_student_aggregation"])
    representation = ActionRepresentation.from_mapping(experiment["representation"])
    if (
        representation.kind != "zernike"
        or representation.num_modes != 21
        or int(experiment["representation"]["anchor_modes"]) != ANCHOR_MODES
        or int(experiment["representation"]["output_modes"]) != 11
    ):
        raise RuntimeError("R3 representation must remain Zernike 21 with 10+11 split")
    if int(experiment["policy_observation"]["state_size"]) != 210:
        raise RuntimeError("R3 policy state must remain the normalized 210D state")
    if not math.isclose(
        float(experiment["student_anchor"]["deployment_scale"]), 0.50, abs_tol=1e-12
    ):
        raise RuntimeError("R3 student deployment scale must remain 0.50")
    action = experiment["action"]
    if not math.isclose(float(action["residual_action_limit_rad"]), 0.05):
        raise RuntimeError("R3 shared residual component limit changed")
    if not math.isclose(float(action["correction_component_limit_rad"]), 0.0125):
        raise RuntimeError("R3 SAC correction limit must remain 0.0125 rad")
    if int(action["total_budget_anchor_modes"]) != ANCHOR_MODES:
        raise RuntimeError("R3 total action budget anchor changed")
    sac = experiment["sac"]
    required_sac = {
        "hidden_size": 256,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "tau": 0.005,
        "initial_alpha": 0.05,
        "total_transitions_per_seed": 499200,
    }
    for field, expected in required_sac.items():
        if not math.isclose(float(sac[field]), float(expected), abs_tol=1e-12):
            raise RuntimeError(f"R3 locked SAC setting changed: {field}")
    if list(experiment["arms"]) != ["student_backbone", "random_backbone"]:
        raise RuntimeError("R3 requires the locked main and mechanism-ablation arms")
    if list(map(int, experiment["policy_seeds"])) != [9301, 9302, 9303]:
        raise RuntimeError("R3 formal policy seeds changed")
    if len(experiment["student_checkpoints"]) != 3:
        raise RuntimeError("R3 requires three paired student checkpoints")
    for item in experiment["student_checkpoints"]:
        path = _project_path(item["path"])
        if _file_sha256(path) != str(item["sha256"]):
            raise RuntimeError(f"student checkpoint hash mismatch: {item['id']}")
    _validate_seed_partitions(experiment, settings)
    environment_path = _project_path(experiment["environment_config"])
    profile_path = _project_path(experiment["hardware_profile_source"])
    if _file_sha256(environment_path) != str(experiment["environment_config_sha256"]):
        raise RuntimeError("environment configuration hash mismatch")
    if _file_sha256(profile_path) != str(experiment["hardware_profile_source_sha256"]):
        raise RuntimeError("hardware profile source hash mismatch")
    missing = [
        path for path in experiment["tracked_source_files"] if not _project_path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"R3 tracked source files missing: {missing}")
    return {
        "status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_TRAINING",
        "quick": quick,
        "upstream_status": upstream["status"],
        "student_checkpoint_hashes_verified": True,
        "state_size": 210,
        "action_size": 11,
        "student_frozen": True,
        "deterministic_initial_correction_zero": True,
        "arms": settings["arms"],
        "policy_seeds": settings["policy_seeds"],
        "total_runs": len(settings["arms"]) * len(settings["policy_seeds"]),
        "transitions_per_run": settings["total_transitions_per_seed"],
        "total_transitions": len(settings["arms"])
        * len(settings["policy_seeds"])
        * int(settings["total_transitions_per_seed"]),
        "cuda_required": True,
        "sealed_s4d3_access": False,
        "real_slm_actions": False,
        "formal_command": (
            ".\\.venv\\Scripts\\python.exe "
            "scripts\\train_s4_r3_student_anchored_sac.py --config "
            "configs\\experiments\\s4_student_anchored_sac_r3_v1.yaml"
        ),
    }


def _train_one_run(
    *,
    arm: str,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    base_config: S1EnvConfig,
    basis: torch.Tensor,
    training_profiles: list[HardwareProfile],
    evaluation_profiles: list[HardwareProfile],
    registration_mappings: dict[str, torch.Tensor],
    student: FrozenStudentPolicy,
    policy_seed: int,
    environment_seed_base: int,
    output_directory: Path,
    run_index: int,
    total_runs: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(policy_seed)
    torch.cuda.manual_seed_all(policy_seed)
    chooser = random.Random(policy_seed)
    action_generator = torch.Generator(device=device).manual_seed(policy_seed + 10_000_000)
    config = replace(
        base_config,
        batch_size=int(settings["environment_batch_size"]),
        episode_length=int(settings["episode_length"]),
    )
    sac_config = SacConfig(
        state_size=210,
        action_size=11,
        hidden_size=int(experiment["sac"]["hidden_size"]),
        learning_rate=float(experiment["sac"]["learning_rate"]),
        gamma=float(experiment["sac"]["gamma"]),
        tau=float(experiment["sac"]["tau"]),
        initial_alpha=float(experiment["sac"]["initial_alpha"]),
    )
    agent = ResidualSacAgent(sac_config, device)
    if arm == "student_backbone":
        initialize_actor_from_student(agent.actor, student)
    elif arm != "random_backbone":
        raise ValueError(f"unknown R3 arm: {arm}")
    if torch.count_nonzero(agent.actor.mean_head.weight) or torch.count_nonzero(
        agent.actor.mean_head.bias
    ):
        raise RuntimeError("R3 actor mean head must start at exact zero")
    replay = TransitionReplayBuffer(
        int(settings["replay_buffer_size"]),
        210,
        11,
        storage_device=str(experiment["sac"]["replay_storage_device"]),
        seed=policy_seed,
    )
    schedule = [
        (condition, profile)
        for condition in settings["training_physical_conditions"]
        for profile in training_profiles
    ]
    chooser.shuffle(schedule)
    environment_cache: dict[tuple[str, str], AdaptiveOpticsEnv] = {}
    total = int(settings["total_transitions_per_seed"])
    warmup = int(settings["warmup_transitions"])
    transitions = environment_steps = completed_episodes = episode_index = 0
    next_log = int(settings["log_interval_transitions"])
    next_validation = int(settings["validation_interval_transitions"])
    next_checkpoint = int(settings["checkpoint_interval_transitions"])
    best_score = -float("inf")
    best_validation: dict[str, Any] | None = None
    latest_losses = {
        "actor_loss": float("nan"),
        "critic_loss": float("nan"),
        "alpha_loss": float("nan"),
        "alpha": float(agent.alpha.detach()),
        "mean_q": float("nan"),
    }
    recent_reward: deque[float] = deque(maxlen=100)
    recent_power: deque[float] = deque(maxlen=100)
    recent_violation: deque[float] = deque(maxlen=100)
    recent_correction_projection: deque[float] = deque(maxlen=100)
    recent_final_projection: deque[float] = deque(maxlen=100)
    loss_path = output_directory / "loss_history.csv"
    progress_path = output_directory / "progress.jsonl"
    validation_path = output_directory / "validation_history.jsonl"
    fieldnames = [
        "arm", "policy_seed", "transitions", "environment_steps", "episodes",
        "buffer_size", "mean_reward", "mean_measured_power", "mean_violation",
        "mean_correction_projection", "mean_final_projection",
        "latest_rl_minus_student_power_gain", "actor_loss", "critic_loss",
        "alpha_loss", "alpha", "mean_q",
    ]
    best_checkpoint = output_directory / "checkpoint_best.pt"
    latest_dev_increment = float("nan")
    with loss_path.open("w", encoding="utf-8", newline="") as loss_handle:
        writer = csv.DictWriter(loss_handle, fieldnames=fieldnames)
        writer.writeheader()
        bar = counted_progress(
            total=total,
            description=f"R3 {arm} 运行{run_index}/{total_runs} 种子{policy_seed}",
            unit="转移",
        )
        while transitions < total:
            if episode_index > 0 and episode_index % len(schedule) == 0:
                chooser.shuffle(schedule)
            condition_item, profile = schedule[episode_index % len(schedule)]
            condition = RobustnessCondition.from_mapping(
                {
                    **condition_item,
                    "base_seed": environment_seed_base
                    + episode_index * config.batch_size,
                }
            )
            if condition.base_seed + config.batch_size > (
                environment_seed_base + int(settings["training_seed_span_per_policy"])
            ):
                raise RuntimeError("R3 training environment seed span exhausted")
            cache_key = (condition.identifier, profile.identifier)
            environment = environment_cache.get(cache_key)
            if environment is None:
                configured = profile.environment_config(
                    condition.environment_config(config)
                )
                environment = AdaptiveOpticsEnv(
                    configured,
                    device,
                    hardware_effects=profile.effects_config(),
                    basis_override=basis,
                )
                environment_cache[cache_key] = environment
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
            controller = _make_controller(experiment, config, student)
            raw_state = controller.reset(noisy)
            for _ in range(config.episode_length):
                state = _normalize_state(raw_state, student)
                if transitions < warmup:
                    correction = 0.5 * torch.rand(
                        config.batch_size,
                        11,
                        device=device,
                        generator=action_generator,
                    ) - 0.25
                else:
                    correction = agent.act(state, deterministic=False)
                action = controller.compose_action(raw_state, correction)
                next_observation, _, terminated, _, info = environment.step(
                    action.composed.final_delta_rad
                )
                reward = _residual_reward(
                    measured_power=info["measured_power_in_bucket"],
                    normalized_residual=action.correction_normalized,
                    violation=info["violation_fraction"],
                    reward_config=experiment["reward"],
                )
                if bool(terminated.all()):
                    next_raw_state = torch.zeros_like(raw_state)
                    next_state = torch.zeros_like(state)
                else:
                    next_noisy = _noisy_observation(
                        next_observation,
                        config.num_modes,
                        profile.observation_noise_std_rad,
                        observation_generator,
                    )
                    next_raw_state = controller.advance_observation(next_noisy)
                    next_state = _normalize_state(next_raw_state, student)
                replay.add_batch(state, correction, reward, next_state, terminated)
                raw_state = next_raw_state
                transitions += config.batch_size
                environment_steps += 1
                recent_reward.append(float(reward.mean()))
                recent_power.append(float(info["measured_power_in_bucket"].mean()))
                recent_violation.append(float(info["violation_fraction"].mean()))
                recent_correction_projection.append(
                    float(action.correction_projection_fraction.mean())
                )
                recent_final_projection.append(
                    float(action.final_projection_fraction.mean())
                )
                if transitions >= warmup and len(replay) >= int(settings["sac_batch_size"]):
                    for _ in range(int(settings["updates_per_environment_step"])):
                        latest_losses = agent.update(
                            replay.sample(int(settings["sac_batch_size"]), device)
                        )
                advance_to(bar, min(transitions, total))
                if transitions >= next_validation or transitions >= total:
                    validation = evaluate_r3_policy(
                        agent=agent,
                        experiment=experiment,
                        settings=settings,
                        base_config=config,
                        basis=basis,
                        profiles=_profiles(
                            experiment, settings["interval_validation_profile_ids"]
                        ),
                        registration_mappings=registration_mappings,
                        student=student,
                        device=device,
                        include_teacher=False,
                        include_episode_records=False,
                    )
                    validation["transitions"] = min(transitions, total)
                    latest_dev_increment = float(
                        validation["overall"]["rl_vs_student"]["relative_power_gain"]
                    )
                    _append_jsonl(validation_path, validation)
                    score = latest_dev_increment - 10 * max(
                        0.0,
                        float(validation["overall"]["candidate_violation_mean"])
                        - float(experiment["gate"]["max_violation_fraction"]),
                    )
                    if score > best_score:
                        best_score = score
                        best_validation = validation
                        torch.save(_checkpoint(agent, student, arm), best_checkpoint)
                    next_validation += int(settings["validation_interval_transitions"])
                if transitions >= next_log or transitions >= total:
                    record = {
                        "arm": arm,
                        "policy_seed": policy_seed,
                        "transitions": min(transitions, total),
                        "environment_steps": environment_steps,
                        "episodes": completed_episodes,
                        "buffer_size": len(replay),
                        "mean_reward": _mean(recent_reward),
                        "mean_measured_power": _mean(recent_power),
                        "mean_violation": _mean(recent_violation),
                        "mean_correction_projection": _mean(recent_correction_projection),
                        "mean_final_projection": _mean(recent_final_projection),
                        "latest_rl_minus_student_power_gain": latest_dev_increment,
                        **latest_losses,
                    }
                    writer.writerow(record)
                    loss_handle.flush()
                    _append_jsonl(progress_path, record)
                    update_progress(
                        bar,
                        device=device,
                        metrics={
                            "奖励": record["mean_reward"],
                            "功率": record["mean_measured_power"],
                            "违规": record["mean_violation"],
                            "RL-学生": record["latest_rl_minus_student_power_gain"],
                            "策略损失": record["actor_loss"],
                            "评价损失": record["critic_loss"],
                            "温度": record["alpha"],
                        },
                    )
                    next_log += int(settings["log_interval_transitions"])
                if transitions >= next_checkpoint or transitions >= total:
                    torch.save(
                        _checkpoint(agent, student, arm),
                        output_directory / f"checkpoint_{min(transitions, total):09d}.pt",
                    )
                    next_checkpoint += int(settings["checkpoint_interval_transitions"])
                if transitions >= total:
                    break
            completed_episodes += config.batch_size
            episode_index += 1
        bar.close()
    if not best_checkpoint.exists():
        raise RuntimeError("R3 training finished without a development checkpoint")
    final_checkpoint = output_directory / "checkpoint_final.pt"
    torch.save(_checkpoint(agent, student, arm), final_checkpoint)
    best_payload = torch.load(best_checkpoint, map_location=device, weights_only=False)
    agent.load_actor(best_payload)
    final_validation = evaluate_r3_policy(
        agent=agent,
        experiment=experiment,
        settings=settings,
        base_config=config,
        basis=basis,
        profiles=evaluation_profiles,
        registration_mappings=registration_mappings,
        student=student,
        device=device,
        include_teacher=True,
        include_episode_records=True,
    )
    episode_records = final_validation.pop("episode_records")
    episode_path = output_directory / "final_episode_records.csv"
    _write_rows(episode_path, episode_records)
    result = {
        "arm": arm,
        "policy_seed": policy_seed,
        "student_id": student.identifier,
        "training_environment_seed_base": environment_seed_base,
        "transitions": total,
        "environment_steps": environment_steps,
        "completed_episodes": completed_episodes,
        "gradient_updates": agent.update_count,
        "best_interval_validation": best_validation,
        "final_development_validation": final_validation,
        "best_checkpoint": _relative(best_checkpoint),
        "best_checkpoint_sha256": _file_sha256(best_checkpoint),
        "final_checkpoint": _relative(final_checkpoint),
        "final_checkpoint_sha256": _file_sha256(final_checkpoint),
        "loss_history": _relative(loss_path),
        "progress_log": _relative(progress_path),
        "validation_history": _relative(validation_path),
        "final_episode_records": _relative(episode_path),
        "final_episode_records_sha256": _file_sha256(episode_path),
        "final_episode_record_rows": len(episode_records),
    }
    _write_json(output_directory / "seed_summary.json", result)
    return result


@torch.no_grad()
def evaluate_r3_policy(
    *,
    agent: ResidualSacAgent,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    base_config: S1EnvConfig,
    basis: torch.Tensor,
    profiles: list[HardwareProfile],
    registration_mappings: dict[str, torch.Tensor],
    student: FrozenStudentPolicy,
    device: torch.device,
    include_teacher: bool,
    include_episode_records: bool,
) -> dict[str, Any]:
    conditions = [
        RobustnessCondition.from_mapping(item)
        for item in settings["validation_physical_conditions"]
    ]
    steps = int(settings["validation_steps"])
    batch = int(settings["validation_batch_size"])
    base = replace(base_config, batch_size=batch, episode_length=max(base_config.episode_length, steps + 2))
    all_scenarios: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        for condition in conditions:
            config = profile.environment_config(condition.environment_config(base))
            baseline = _rollout_baseline(
                experiment, config, condition, profile, steps, basis, device
            )
            student_metrics, student_telemetry = _rollout_student_or_rl(
                agent=None,
                experiment=experiment,
                config=config,
                condition=condition,
                profile=profile,
                steps=steps,
                basis=basis,
                student=student,
                device=device,
            )
            candidate, telemetry = _rollout_student_or_rl(
                agent=agent,
                experiment=experiment,
                config=config,
                condition=condition,
                profile=profile,
                steps=steps,
                basis=basis,
                student=student,
                device=device,
            )
            teacher = None
            if include_teacher:
                future_truth = _future_disturbance_sequence(
                    config=config,
                    condition=condition,
                    profile=profile,
                    length=steps + int(experiment["teacher"]["preview_horizon_frames"]),
                    basis=basis,
                    device=device,
                )
                teacher, _ = _rollout_teacher(
                    experiment=experiment,
                    config=config,
                    condition=condition,
                    profile=profile,
                    steps=steps,
                    basis=basis,
                    future_truth=future_truth,
                    registration_mapping=registration_mappings[profile.identifier],
                    device=device,
                )
            scenario = {
                "profile_id": profile.identifier,
                "condition_id": condition.identifier,
                "base_seed": condition.base_seed,
                "baseline": baseline,
                "student": student_metrics,
                "candidate": candidate,
                "teacher": teacher,
                "telemetry": telemetry,
                "student_telemetry": student_telemetry,
            }
            all_scenarios.append(scenario)
            if include_episode_records:
                rows.extend(_episode_rows(scenario))
    result = _summarize_evaluation(all_scenarios, experiment, include_teacher)
    if include_episode_records:
        result["episode_records"] = rows
    return result


def _rollout_student_or_rl(
    *,
    agent: ResidualSacAgent | None,
    experiment: dict[str, Any],
    config: S1EnvConfig,
    condition: RobustnessCondition,
    profile: HardwareProfile,
    steps: int,
    basis: torch.Tensor,
    student: FrozenStudentPolicy,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
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
    metrics = _empty_step_metrics()
    correction_projection: list[torch.Tensor] = []
    final_projection: list[torch.Tensor] = []
    for step in range(steps):
        correction = (
            torch.zeros(config.batch_size, 11, device=device)
            if agent is None
            else agent.act(_normalize_state(state, student), deterministic=True)
        )
        action = controller.compose_action(state, correction)
        observation, _, _, _, info = environment.step(action.composed.final_delta_rad)
        _append_step_metrics(metrics, info)
        correction_projection.append(action.correction_projection_fraction.cpu())
        final_projection.append(action.final_projection_fraction.cpu())
        if step + 1 < steps:
            noisy = _noisy_observation(
                observation,
                config.num_modes,
                profile.observation_noise_std_rad,
                generator,
            )
            state = controller.advance_observation(noisy)
    return _mean_step_metrics(metrics), {
        "correction_projection_fraction": torch.stack(correction_projection, dim=1).mean(dim=1),
        "final_projection_fraction": torch.stack(final_projection, dim=1).mean(dim=1),
    }


def _rollout_baseline(
    experiment: dict[str, Any],
    config: S1EnvConfig,
    condition: RobustnessCondition,
    profile: HardwareProfile,
    steps: int,
    basis: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    environment = AdaptiveOpticsEnv(
        config, device, profile.effects_config(), basis_override=basis
    )
    observation, _ = environment.reset(seed=condition.base_seed)
    generator = torch.Generator(device=device).manual_seed(condition.base_seed + 40_000_000)
    observation = _noisy_observation(
        observation, config.num_modes, profile.observation_noise_std_rad, generator
    )
    params = experiment["frozen_controller"]["parameters"]
    controller = TrackingLeakyIntegratorController(
        num_modes=ANCHOR_MODES,
        modal_limit_rad=config.modal_limit_rad,
        gain=float(params["gain"]),
        leak=float(params["leak"]),
        tracking_gain=float(params["tracking_gain"]),
        max_request_step_rad=float(params["max_request_step_rad"]),
    )
    controller.reset(config.batch_size, device, observation.dtype)
    metrics = _empty_step_metrics()
    for step in range(steps):
        anchor_observation = torch.cat(
            (
                observation[:, :ANCHOR_MODES],
                observation[:, config.num_modes : config.num_modes + ANCHOR_MODES],
                observation[:, -2:],
            ),
            dim=-1,
        )
        action = torch.zeros(config.batch_size, config.num_modes, device=device)
        action[:, :ANCHOR_MODES] = controller.action(anchor_observation)
        observation, _, _, _, info = environment.step(action)
        _append_step_metrics(metrics, info)
        if step + 1 < steps:
            observation = _noisy_observation(
                observation,
                config.num_modes,
                profile.observation_noise_std_rad,
                generator,
            )
    return _mean_step_metrics(metrics)


def _summarize_evaluation(
    scenarios: list[dict[str, Any]],
    experiment: dict[str, Any],
    include_teacher: bool,
) -> dict[str, Any]:
    profile_ids = sorted({item["profile_id"] for item in scenarios})
    grouped: dict[str, Any] = {}
    for profile_id in ["all_profiles", *profile_ids]:
        selected = (
            scenarios
            if profile_id == "all_profiles"
            else [item for item in scenarios if item["profile_id"] == profile_id]
        )
        grouped[profile_id] = _summarize_selected(
            selected, experiment, include_teacher, profile_id
        )
    profile_pass = all(grouped[item]["gate"] == "PASS" for item in profile_ids)
    development_gate = (
        "PASS"
        if grouped["all_profiles"]["gate"] == "PASS" and profile_pass
        else "FAIL"
    )
    return {
        "overall": grouped.pop("all_profiles"),
        "profiles": grouped,
        "all_profiles_pass": profile_pass,
        "development_gate": development_gate,
        "teacher_included": include_teacher,
        "sealed_test_accessed": False,
    }


def _summarize_selected(
    scenarios: list[dict[str, Any]],
    experiment: dict[str, Any],
    include_teacher: bool,
    identifier: str,
) -> dict[str, Any]:
    candidate = _concat_scenario_metrics(scenarios, "candidate")
    student = _concat_scenario_metrics(scenarios, "student")
    baseline = _concat_scenario_metrics(scenarios, "baseline")
    gate = experiment["gate"]
    zero_gain_gate = {
        "proposed_min_relative_power_gain": 0.0,
        "min_power_delta_ci95_low": gate["min_power_delta_ci95_low"],
        "min_strehl_delta_ci95_low": gate["min_strehl_delta_ci95_low"],
        "max_phase_rmse_delta_ci95_high": gate["max_phase_rmse_delta_ci95_high"],
        "max_violation_fraction": gate["max_violation_fraction"],
    }
    baseline_gate = {**zero_gain_gate, "proposed_min_relative_power_gain": gate["min_relative_power_gain_vs_traditional"]}
    rl_vs_student = _paired_science_summary(
        f"{identifier}_rl_vs_student", candidate, student, zero_gain_gate
    )
    rl_vs_baseline = _paired_science_summary(
        f"{identifier}_rl_vs_baseline", candidate, baseline, baseline_gate
    )
    correction_projection = torch.cat(
        [item["telemetry"]["correction_projection_fraction"] for item in scenarios]
    )
    final_projection = torch.cat(
        [item["telemetry"]["final_projection_fraction"] for item in scenarios]
    )
    result: dict[str, Any] = {
        "id": identifier,
        "rl_vs_student": rl_vs_student,
        "rl_vs_traditional": rl_vs_baseline,
        "candidate_violation_mean": float(candidate["violation_fraction"].mean()),
        "correction_projection_fraction": _distribution(correction_projection),
        "final_projection_fraction": _distribution(final_projection),
    }
    passed = (
        rl_vs_student["gate"] == "PASS"
        and rl_vs_baseline["gate"] == "PASS"
        and float(correction_projection.mean()) <= float(gate["max_projection_fraction"])
        and float(final_projection.mean()) <= float(gate["max_projection_fraction"])
    )
    if include_teacher:
        teacher = _concat_scenario_metrics(scenarios, "teacher")
        teacher_vs_baseline = _paired_science_summary(
            f"{identifier}_teacher_vs_baseline", teacher, baseline, baseline_gate
        )
        teacher_minus_student = teacher["power_in_bucket"] - student["power_in_bucket"]
        rl_minus_student = candidate["power_in_bucket"] - student["power_in_bucket"]
        headroom = float(teacher_minus_student.mean())
        recovery = float(rl_minus_student.mean()) / max(headroom, 1e-12)
        teacher_positive = (
            _distribution(teacher_minus_student)["ci95_low"] > 0
            and teacher_vs_baseline["gate"] == "PASS"
        )
        result.update(
            {
                "teacher_vs_traditional": teacher_vs_baseline,
                "teacher_minus_student_power": _distribution(teacher_minus_student),
                "rl_recovery_of_teacher_student_headroom": recovery,
                "teacher_positive_control_pass": teacher_positive,
            }
        )
        passed = (
            passed
            and teacher_positive
            and recovery >= float(gate["min_teacher_headroom_recovery_fraction"])
        )
    result["gate"] = "PASS" if passed else "FAIL"
    return result


def _concat_scenario_metrics(
    scenarios: list[dict[str, Any]], key: str
) -> dict[str, torch.Tensor]:
    if any(item[key] is None for item in scenarios):
        raise RuntimeError(f"missing evaluation metrics: {key}")
    return {
        metric: torch.cat([item[key][metric] for item in scenarios])
        for metric in SCIENCE_METRICS
    }


def _episode_rows(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    count = int(scenario["candidate"]["power_in_bucket"].numel())
    for index in range(count):
        row: dict[str, Any] = {
            "profile_id": scenario["profile_id"],
            "condition_id": scenario["condition_id"],
            "episode_index": index,
            "episode_seed": int(scenario["base_seed"]) + index,
        }
        for source in ("baseline", "student", "candidate", "teacher"):
            values = scenario[source]
            if values is not None:
                for metric in SCIENCE_METRICS:
                    row[f"{source}_{metric}"] = float(values[metric][index])
        row["delta_rl_minus_student_power_in_bucket"] = (
            row["candidate_power_in_bucket"] - row["student_power_in_bucket"]
        )
        row["delta_rl_minus_traditional_power_in_bucket"] = (
            row["candidate_power_in_bucket"] - row["baseline_power_in_bucket"]
        )
        row["correction_projection_fraction"] = float(
            scenario["telemetry"]["correction_projection_fraction"][index]
        )
        row["final_projection_fraction"] = float(
            scenario["telemetry"]["final_projection_fraction"][index]
        )
        rows.append(row)
    return rows


def _make_controller(
    experiment: dict[str, Any],
    config: S1EnvConfig,
    student: FrozenStudentPolicy,
) -> StudentAnchoredResidualController:
    params = experiment["frozen_controller"]["parameters"]
    base = AnchoredResidualTrackingController(
        num_modes=config.num_modes,
        anchor_modes=ANCHOR_MODES,
        modal_limit_rad=config.modal_limit_rad,
        history_frames=int(experiment["policy_observation"]["history_frames"]),
        residual_action_limit_rad=float(experiment["action"]["residual_action_limit_rad"]),
        final_action_step_limit_rad=float(experiment["action"]["final_action_step_limit_rad"]),
        gain=float(params["gain"]),
        leak=float(params["leak"]),
        tracking_gain=float(params["tracking_gain"]),
    )
    return StudentAnchoredResidualController(
        base,
        student,
        student_scale=float(experiment["student_anchor"]["deployment_scale"]),
        correction_component_limit_rad=float(experiment["action"]["correction_component_limit_rad"]),
        projection_tolerance_rad=float(experiment["gate"]["projection_tolerance_rad"]),
    )


def _normalize_state(state: torch.Tensor, student: FrozenStudentPolicy) -> torch.Tensor:
    return (state - student.state_mean) / student.state_scale


def _load_student_item(
    item: dict[str, Any],
    *,
    experiment: dict[str, Any],
    device: torch.device,
) -> FrozenStudentPolicy:
    return load_frozen_student(
        _project_path(item["path"]),
        identifier=str(item["id"]),
        state_size=210,
        hidden_size=int(experiment["student_anchor"]["hidden_size"]),
        output_size=11,
        device=device,
    )


def _checkpoint(
    agent: ResidualSacAgent,
    student: FrozenStudentPolicy,
    arm: str,
) -> dict[str, Any]:
    payload = agent.checkpoint()
    payload.update(
        {
            "algorithm": "student_anchored_residual_sac",
            "arm": arm,
            "student_id": student.identifier,
            "state_mean": student.state_mean.detach().cpu(),
            "state_scale": student.state_scale.detach().cpu(),
        }
    )
    return payload


def _verify_upstream(upstream: dict[str, Any]) -> dict[str, Any]:
    for field in ("summary", "preflight", "effective_config", "source_manifest", "audit_record"):
        path = _project_path(upstream[field])
        if _file_sha256(path) != str(upstream[f"{field}_sha256"]):
            raise RuntimeError(f"R3 upstream evidence hash mismatch: {field}")
    summary = json.loads(_project_path(upstream["summary"]).read_text(encoding="utf-8"))
    status = str(summary["interpretation"]["status"])
    if status != "SINGLE_ROUND_STUDENT_AGGREGATION_PASS":
        raise RuntimeError("R3 upstream student aggregation did not pass")
    audit = _project_path(upstream["audit_record"]).read_text(encoding="utf-8")
    if "Verification Status: `ANALYZED`" not in audit:
        raise RuntimeError("R3 upstream audit is not ANALYZED")
    return {"status": status}


def _validate_seed_partitions(
    experiment: dict[str, Any], settings: dict[str, Any]
) -> None:
    span = int(settings["training_seed_span_per_policy"])
    training = [
        set(range(int(base), int(base) + span))
        for base in settings["training_environment_seed_bases"]
    ]
    for index, left in enumerate(training):
        for right in training[index + 1 :]:
            if left & right:
                raise RuntimeError("R3 policy training seed ranges overlap")
    validation = {
        int(item["base_seed"]) + offset
        for item in settings["validation_physical_conditions"]
        for offset in range(int(settings["validation_batch_size"]))
    }
    if any(validation & item for item in training):
        raise RuntimeError("R3 training and validation seeds overlap")
    if any(seed < 3_500_000 or seed >= 4_000_000 for item in training for seed in item):
        raise RuntimeError("R3 training seeds left the locked development namespace")
    if any(seed < 3_600_000 or seed >= 3_700_000 for seed in validation):
        raise RuntimeError("R3 validation seeds left the locked validation namespace")
    if int(experiment["reserved_s4d3_seed_base"]) < 4_000_000:
        raise RuntimeError("sealed S4-D3 seed base must start at or above 4,000,000")


def _effective_settings(experiment: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    sac = experiment["sac"]
    settings: dict[str, Any] = {
        "quick": quick,
        "output_directory": experiment["outputs"]["directory"],
        "arms": list(experiment["arms"]),
        "policy_seeds": list(experiment["policy_seeds"]),
        "student_checkpoints": deepcopy(experiment["student_checkpoints"]),
        "training_environment_seed_bases": list(experiment["training_environment_seed_bases"]),
        "training_seed_span_per_policy": int(experiment["training_seed_span_per_policy"]),
        "training_physical_conditions": deepcopy(experiment["training_physical_conditions"]),
        "training_hardware_profile_ids": list(experiment["training_hardware_profile_ids"]),
        "validation_physical_conditions": deepcopy(experiment["validation_physical_conditions"]),
        "interval_validation_profile_ids": list(experiment["interval_validation_profile_ids"]),
        "final_validation_profile_ids": list(experiment["final_validation_profile_ids"]),
        "environment_batch_size": 32,
        "episode_length": int(experiment["validation"]["steps"]),
        "validation_batch_size": int(experiment["validation"]["episodes_per_physical_condition"]),
        "validation_steps": int(experiment["validation"]["steps"]),
        "replay_buffer_size": int(sac["replay_buffer_size"]),
        "sac_batch_size": int(sac["batch_size"]),
        "warmup_transitions": int(sac["warmup_transitions"]),
        "total_transitions_per_seed": int(sac["total_transitions_per_seed"]),
        "updates_per_environment_step": int(sac["updates_per_environment_step"]),
        "validation_interval_transitions": int(sac["validation_interval_transitions"]),
        "checkpoint_interval_transitions": int(sac["checkpoint_interval_transitions"]),
        "log_interval_transitions": int(sac["log_interval_transitions"]),
    }
    if quick:
        quick_settings = experiment["quick"]
        for field in (
            "total_transitions_per_seed", "warmup_transitions", "replay_buffer_size",
            "updates_per_environment_step", "validation_interval_transitions",
            "checkpoint_interval_transitions", "log_interval_transitions",
            "environment_batch_size", "episode_length", "validation_steps",
        ):
            settings[field] = int(quick_settings[field])
        settings["sac_batch_size"] = int(quick_settings["batch_size"])
        settings["validation_batch_size"] = int(quick_settings["environment_batch_size"])
        settings["policy_seeds"] = list(quick_settings["policy_seeds"])
        indices = [list(map(int, experiment["policy_seeds"])).index(int(seed)) for seed in settings["policy_seeds"]]
        settings["student_checkpoints"] = [settings["student_checkpoints"][index] for index in indices]
        settings["training_environment_seed_bases"] = list(quick_settings["training_environment_seed_bases"])
        settings["training_hardware_profile_ids"] = list(quick_settings["training_hardware_profile_ids"])
        settings["interval_validation_profile_ids"] = list(quick_settings["validation_hardware_profile_ids"])
        settings["final_validation_profile_ids"] = list(quick_settings["validation_hardware_profile_ids"])
        settings["validation_physical_conditions"] = deepcopy(quick_settings["validation_physical_conditions"])
        settings["output_directory"] = experiment["outputs"]["quick_directory"]
    return settings


def _summarize_runs(results: list[dict[str, Any]], *, quick: bool) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        by_arm.setdefault(str(item["arm"]), []).append(item)
    arm_status = {
        arm: all(
            item["final_development_validation"]["development_gate"] == "PASS"
            for item in items
        )
        for arm, items in by_arm.items()
    }
    return {
        "status": "QUICK_SMOKE_ONLY" if quick else "ANALYSIS_REQUIRED",
        "arm_all_seed_gate_pass": arm_status,
        "main_arm_required": True,
        "main_arm_pass_pending_audit": arm_status.get("student_backbone", False),
        "s4d3_authorized": False,
        "real_hardware_authorized": False,
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("episode records must not be empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else float("nan")
