"""S4-D2残差SAC的CUDA训练、开发验证与可追溯输出。"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import time
from typing import Any, Iterable

import torch
import yaml

from src.rl.residual_control import (
    AnchoredResidualTrackingController,
    ResidualTrackingController,
)
from src.rl.residual_sac import ResidualSacAgent, SacConfig, TransitionReplayBuffer
from src.runtime import resolve_device
from src.simulation.config import S1EnvConfig, load_s1_config
from src.simulation.controllers import TrackingLeakyIntegratorController
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.hardware_effects import HardwareProfile
from src.simulation.robust_control import RobustnessCondition
from src.training_progress import advance_to, counted_progress, update_progress


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_s4_residual_sac(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """准备或运行S4-D2；正式训练入口由用户在IDE中调用。"""
    experiment_path = _project_path(config_path)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight = preflight_s4_residual_sac(
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
            f"output directory already exists; preserve it and diagnose before retrying: {output_directory}"
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
        _write_json(output_directory / "partial_summary.json", {"policy_seeds": seed_results})

    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "s4_residual_sac_v1",
        },
        "experiment": {
            "id": "AO-S4-D2-RESIDUAL-SAC-TRAINING",
            "type": "software_only_cuda_residual_reinforcement_learning",
            "status": "completed_pending_audit",
            "quick": quick,
            "duration_seconds": time.perf_counter() - started,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "evidence_boundary": {
            "real_slm_actions": False,
            "d1_trajectories_used_for_training": False,
            "sealed_s4d3_accessed": False,
            "policy_observation_contains_oracle_quality_metrics": False,
            "interpretation": (
                "S4-D2 only trains and development-validates Residual SAC in CUDA simulation. "
                "It cannot establish real FSLM performance or sealed-test superiority."
            ),
        },
        "inputs": {
            "experiment_config": _relative(experiment_path),
            "experiment_config_sha256": _file_sha256(experiment_path),
            "source_manifest": source_manifest,
            "upstream_s4d1_gate": "PASS",
        },
        "policy_seed_results": seed_results,
        "development_summary": _summarize_policy_seeds(seed_results, settings),
        "runtime": _runtime_record(),
        "git": _git_record(),
        "next_action": (
            "Stop and ask the assistant to audit S4-D2. Do not open or generate S4-D3."
        ),
    }
    _write_json(output_directory / "summary.json", summary)
    return summary


def preflight_s4_residual_sac(
    experiment_path: Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    """在创建输出、占用CUDA或读取D1轨迹内容前完成失败前置检查。"""
    if str(experiment["metadata"]["stage"]) != "S4-D2":
        raise ValueError("experiment metadata must identify S4-D2")
    if bool(experiment["metadata"].get("allow_d1_data_reuse", True)):
        raise RuntimeError("S4-D1 data reuse must remain disabled")
    if bool(experiment["metadata"].get("allow_sealed_test_access", True)):
        raise RuntimeError("sealed-test access must remain disabled")

    environment_path = _project_path(experiment["environment_config"])
    if _file_sha256(environment_path) != str(experiment["environment_config_sha256"]):
        raise RuntimeError("environment configuration changed after S4-D1")
    hardware_path = _project_path(experiment["hardware_profile_source"])
    if _file_sha256(hardware_path) != str(experiment["hardware_profile_source_sha256"]):
        raise RuntimeError("hardware profile source hash mismatch")
    base_config, _ = load_s1_config(environment_path)
    if base_config.num_modes != int(experiment["action"]["num_modes"]):
        raise RuntimeError("SAC action dimension differs from the environment modal basis")

    upstream = _verify_s4d1(experiment["upstream_s4d1"])
    frozen = experiment["frozen_controller"]
    if frozen["id"] != upstream["frozen_controller"]:
        raise RuntimeError("frozen controller id differs from S4-D1")
    upstream_definition = next(
        item
        for item in upstream["summary"]["design"]["controllers"]
        if item["id"] == frozen["id"]
    )
    if upstream_definition["parameters"] != frozen["parameters"]:
        raise RuntimeError("frozen controller parameters differ from S4-D1")

    output_directory = _project_path(settings["output_directory"])
    if output_directory.exists():
        raise FileExistsError(f"output directory already exists: {output_directory}")
    _validate_seed_partitions(settings, upstream["episode_seeds"])
    full_episode_transitions = (
        int(settings["environment_batch_size"]) * int(settings["episode_length"])
    )
    if int(settings["total_transitions_per_seed"]) % full_episode_transitions:
        raise ValueError("training budget must contain only complete batched episodes")
    if int(settings["warmup_transitions"]) < int(settings["sac_batch_size"]):
        raise ValueError("warmup must fill at least one SAC minibatch")

    controller = _make_residual_controller(experiment, base_config)
    expected_state_size = (
        int(experiment["policy_observation"]["history_frames"])
        * 2
        * base_config.num_modes
        + 2 * base_config.num_modes
    )
    if controller.state_size != expected_state_size:
        raise RuntimeError("residual policy state contract is inconsistent")
    if bool(experiment["policy_observation"]["include_oracle_quality_metrics"]):
        raise RuntimeError("oracle quality metrics must not enter the formal policy")

    return {
        "status": "READY_FOR_USER_TRAINING" if not quick else "READY_FOR_QUICK_SMOKE",
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
        "upstream_s4d1": {
            "gate": "PASS",
            "summary_sha256": upstream["summary_sha256"],
            "trajectory_count": upstream["trajectory_count"],
            "episode_seed_count": len(upstream["episode_seeds"]),
            "d1_data_allowed_in_training": False,
        },
        "seed_isolation_verified": True,
        "source_files_present": all(
            _project_path(path).is_file() for path in experiment["tracked_source_files"]
        ),
        "cuda_required": True,
        "real_slm_actions": False,
        "sealed_s4d3_access": False,
        "formal_command": (
            ".\\.venv\\Scripts\\python.exe scripts\\train_s4_residual_sac.py "
            "--config configs\\experiments\\s4_residual_sac_v1.yaml"
        ),
    }


def _train_one_policy_seed(
    *,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    base_config: dict[str, Any],
    policy_seed: int,
    environment_seed_base: int,
    device: torch.device,
    output_directory: Path,
    seed_index: int,
    seed_count: int,
) -> dict[str, Any]:
    torch.manual_seed(policy_seed)
    torch.cuda.manual_seed_all(policy_seed)
    chooser = random.Random(policy_seed)
    action_generator = torch.Generator(device=device).manual_seed(policy_seed + 10_000_000)
    environment_config = S1EnvConfig(**base_config)
    environment_config = replace(
        environment_config,
        batch_size=int(settings["environment_batch_size"]),
        episode_length=int(settings["episode_length"]),
    )
    environment_config.validate()

    residual_controller = _make_residual_controller(experiment, environment_config)
    sac_config = SacConfig(
        state_size=residual_controller.state_size,
        action_size=environment_config.num_modes,
        hidden_size=int(experiment["sac"]["hidden_size"]),
        learning_rate=float(experiment["sac"]["learning_rate"]),
        gamma=float(experiment["sac"]["gamma"]),
        tau=float(experiment["sac"]["tau"]),
        initial_alpha=float(experiment["sac"]["initial_alpha"]),
    )
    agent = ResidualSacAgent(sac_config, device)
    replay = TransitionReplayBuffer(
        int(settings["replay_buffer_size"]),
        sac_config.state_size,
        sac_config.action_size,
        storage_device=str(experiment["sac"]["replay_storage_device"]),
        seed=policy_seed,
    )
    profiles = _profiles(experiment, settings["training_hardware_profile_ids"])
    conditions = list(settings["training_physical_conditions"])
    scenario_schedule = [
        (condition, profile) for condition in conditions for profile in profiles
    ]
    chooser.shuffle(scenario_schedule)
    environment_cache: dict[tuple[str, str], AdaptiveOpticsEnv] = {}
    loss_path = output_directory / "loss_history.csv"
    progress_path = output_directory / "progress.jsonl"
    validation_path = output_directory / "validation_history.jsonl"
    loss_handle = loss_path.open("w", encoding="utf-8", newline="")
    loss_writer = csv.DictWriter(
        loss_handle,
        fieldnames=[
            "transitions",
            "environment_steps",
            "episodes",
            "buffer_size",
            "mean_reward",
            "mean_measured_power",
            "mean_violation",
            "actor_loss",
            "critic_loss",
            "alpha_loss",
            "alpha",
            "mean_q",
        ],
    )
    loss_writer.writeheader()

    total = int(settings["total_transitions_per_seed"])
    warmup = int(settings["warmup_transitions"])
    recent_rewards: deque[float] = deque(maxlen=100)
    recent_power: deque[float] = deque(maxlen=100)
    recent_violations: deque[float] = deque(maxlen=100)
    latest_losses = {
        "actor_loss": float("nan"),
        "critic_loss": float("nan"),
        "alpha_loss": float("nan"),
        "alpha": float(agent.alpha.detach()),
        "mean_q": float("nan"),
    }
    transitions = 0
    environment_steps = 0
    completed_episodes = 0
    next_log = int(settings["log_interval_transitions"])
    next_validation = int(settings["validation_interval_transitions"])
    next_checkpoint = int(settings["checkpoint_interval_transitions"])
    best_score = -float("inf")
    best_validation: dict[str, Any] | None = None
    best_checkpoint = output_directory / "checkpoint_best.pt"
    bar = counted_progress(
        total=total,
        description=(
            f"{settings.get('progress_stage_label', 'S4-D2')}策略种子 "
            f"{seed_index}/{seed_count}"
        ),
        unit="转移",
    )

    episode_index = 0
    while transitions < total:
        if episode_index > 0 and episode_index % len(scenario_schedule) == 0:
            chooser.shuffle(scenario_schedule)
        condition_mapping, profile = scenario_schedule[
            episode_index % len(scenario_schedule)
        ]
        condition = RobustnessCondition.from_mapping(
            {
                **condition_mapping,
                "base_seed": environment_seed_base
                + episode_index * environment_config.batch_size,
            }
        )
        if condition.base_seed + environment_config.batch_size > (
            environment_seed_base + int(settings["training_seed_span_per_policy"])
        ):
            raise RuntimeError("training environment seed span exhausted")
        cache_key = (condition.identifier, profile.identifier)
        environment = environment_cache.get(cache_key)
        if environment is None:
            configured = profile.environment_config(
                condition.environment_config(environment_config)
            )
            environment = AdaptiveOpticsEnv(
                configured,
                device,
                hardware_effects=profile.effects_config(),
            )
            environment_cache[cache_key] = environment
        observation, _ = environment.reset(seed=condition.base_seed)
        observation_generator = torch.Generator(device=device).manual_seed(
            condition.base_seed + 40_000_000
        )
        noisy_observation = _noisy_observation(
            observation,
            environment_config.num_modes,
            profile.observation_noise_std_rad,
            observation_generator,
        )
        state = residual_controller.reset(noisy_observation)

        for _ in range(environment_config.episode_length):
            if transitions < warmup:
                normalized_action = 0.5 * torch.rand(
                    environment_config.batch_size,
                    environment_config.num_modes,
                    device=device,
                    generator=action_generator,
                ) - 0.25
            else:
                normalized_action = agent.act(state, deterministic=False)
            composed = residual_controller.compose_action(normalized_action)
            next_observation, _, terminated, _, info = environment.step(
                composed.final_delta_rad
            )
            reward = _residual_reward(
                measured_power=info["measured_power_in_bucket"],
                normalized_residual=composed.normalized_request,
                violation=info["violation_fraction"],
                reward_config=experiment["reward"],
            )
            done = terminated
            if bool(done.all()):
                next_state = torch.zeros_like(state)
            else:
                next_noisy_observation = _noisy_observation(
                    next_observation,
                    environment_config.num_modes,
                    profile.observation_noise_std_rad,
                    observation_generator,
                )
                next_state = residual_controller.advance_observation(
                    next_noisy_observation
                )
            replay.add_batch(
                state,
                composed.normalized_request,
                reward,
                next_state,
                done,
            )
            state = next_state
            transitions += environment_config.batch_size
            environment_steps += 1
            recent_rewards.append(float(reward.mean()))
            recent_power.append(float(info["measured_power_in_bucket"].mean()))
            recent_violations.append(float(info["violation_fraction"].mean()))

            if transitions >= warmup and len(replay) >= int(settings["sac_batch_size"]):
                for _ in range(int(settings["updates_per_environment_step"])):
                    latest_losses = agent.update(
                        replay.sample(int(settings["sac_batch_size"]), device)
                    )

            advance_to(bar, min(transitions, total))
            if transitions >= next_log or transitions >= total:
                record = {
                    "transitions": min(transitions, total),
                    "environment_steps": environment_steps,
                    "episodes": completed_episodes,
                    "buffer_size": len(replay),
                    "mean_reward": _mean(recent_rewards),
                    "mean_measured_power": _mean(recent_power),
                    "mean_violation": _mean(recent_violations),
                    **latest_losses,
                }
                loss_writer.writerow(record)
                loss_handle.flush()
                _append_jsonl(progress_path, record)
                update_progress(
                    bar,
                    device=device,
                    metrics={
                        "奖励": record["mean_reward"],
                        "功率": record["mean_measured_power"],
                        "违规": record["mean_violation"],
                        "策略损失": record["actor_loss"],
                        "评价损失": record["critic_loss"],
                    },
                )
                next_log += int(settings["log_interval_transitions"])

            if transitions >= next_validation or transitions >= total:
                validation = evaluate_residual_policy(
                    agent=agent,
                    experiment=experiment,
                    settings=settings,
                    base_config=environment_config,
                    device=device,
                    profile_ids=settings["interval_validation_profile_ids"],
                )
                validation["transitions"] = min(transitions, total)
                _append_jsonl(validation_path, validation)
                score = float(validation["selection_score"])
                if score > best_score:
                    best_score = score
                    best_validation = validation
                    torch.save(agent.checkpoint(), best_checkpoint)
                next_validation += int(settings["validation_interval_transitions"])

            if transitions >= next_checkpoint or transitions >= total:
                torch.save(
                    agent.checkpoint(),
                    output_directory / f"checkpoint_{min(transitions, total):09d}.pt",
                )
                next_checkpoint += int(settings["checkpoint_interval_transitions"])
            if transitions >= total:
                break
        completed_episodes += environment_config.batch_size
        episode_index += 1

    bar.close()
    loss_handle.close()
    if not best_checkpoint.exists():
        raise RuntimeError("training finished without a development checkpoint")
    final_checkpoint = output_directory / "checkpoint_final.pt"
    torch.save(agent.checkpoint(), final_checkpoint)
    best_payload = torch.load(best_checkpoint, map_location=device, weights_only=False)
    agent.load_actor(best_payload)
    final_validation = evaluate_residual_policy(
        agent=agent,
        experiment=experiment,
        settings=settings,
        base_config=environment_config,
        device=device,
        profile_ids=settings["final_validation_profile_ids"],
        include_episode_records=bool(settings["persist_final_episode_records"]),
    )
    episode_records = final_validation.pop("episode_records", [])
    episode_records_path: Path | None = None
    if episode_records:
        episode_records_path = output_directory / "final_episode_records.csv"
        _write_episode_records_csv(
            episode_records_path,
            episode_records,
            representation_id=str(experiment.get("active_representation", {}).get("id", "zernike_10")),
            policy_seed=policy_seed,
        )
    result = {
        "policy_seed": policy_seed,
        "training_environment_seed_base": environment_seed_base,
        "transitions": total,
        "environment_steps": environment_steps,
        "completed_episodes": completed_episodes,
        "gradient_updates": agent.update_count,
        "best_interval_validation": best_validation,
        "final_development_validation": final_validation,
        "best_checkpoint": _relative(best_checkpoint),
        "final_checkpoint": _relative(final_checkpoint),
        "loss_history": _relative(loss_path),
        "progress_log": _relative(progress_path),
        "validation_history": _relative(validation_path),
    }
    if episode_records_path is not None:
        result["final_episode_records"] = {
            "path": _relative(episode_records_path),
            "sha256": _file_sha256(episode_records_path),
            "rows": len(episode_records),
        }
    _write_json(output_directory / "seed_summary.json", result)
    return result


@torch.no_grad()
def evaluate_residual_policy(
    *,
    agent: ResidualSacAgent,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    base_config: S1EnvConfig,
    device: torch.device,
    profile_ids: Iterable[str],
    include_episode_records: bool = False,
) -> dict[str, Any]:
    """在相同回合种子上配对比较冻结基线与确定性残差策略。"""
    profiles = _profiles(experiment, profile_ids)
    conditions = [
        RobustnessCondition.from_mapping(item)
        for item in settings["validation_physical_conditions"]
    ]
    evaluation_batch = int(settings["validation_batch_size"])
    evaluation_steps = int(settings["validation_steps"])
    configured_base = replace(
        base_config,
        batch_size=evaluation_batch,
        episode_length=max(base_config.episode_length, evaluation_steps),
    )
    profile_records: list[dict[str, Any]] = []
    overall_candidate: dict[str, list[torch.Tensor]] = _empty_metric_lists()
    overall_baseline: dict[str, list[torch.Tensor]] = _empty_metric_lists()
    episode_records: list[dict[str, Any]] = []
    for profile in profiles:
        candidate_metrics: dict[str, list[torch.Tensor]] = _empty_metric_lists()
        baseline_metrics: dict[str, list[torch.Tensor]] = _empty_metric_lists()
        for condition in conditions:
            condition_config = profile.environment_config(
                condition.environment_config(configured_base)
            )
            candidate = _rollout_candidate(
                agent,
                experiment,
                condition_config,
                condition,
                profile,
                evaluation_steps,
                device,
            )
            baseline = _rollout_baseline(
                experiment,
                condition_config,
                condition,
                profile,
                evaluation_steps,
                device,
            )
            _extend_metrics(candidate_metrics, candidate)
            _extend_metrics(baseline_metrics, baseline)
            _extend_metrics(overall_candidate, candidate)
            _extend_metrics(overall_baseline, baseline)
            if include_episode_records:
                episode_records.extend(
                    _paired_episode_records(
                        profile.identifier,
                        condition.identifier,
                        condition.base_seed,
                        candidate,
                        baseline,
                    )
                )
        profile_records.append(
            _paired_evaluation_record(
                profile.identifier,
                candidate_metrics,
                baseline_metrics,
                experiment["validation"],
            )
        )
    overall = _paired_evaluation_record(
        "all_profiles",
        overall_candidate,
        overall_baseline,
        experiment["validation"],
    )
    all_profiles_pass = all(
        item["development_gate"] == "PASS" for item in profile_records
    )
    development_gate = (
        "PASS"
        if overall["development_gate"] == "PASS" and all_profiles_pass
        else "FAIL"
    )
    result = {
        "profiles": profile_records,
        "overall": overall,
        "selection_score": (
            float(overall["relative_power_gain"])
            - 10
            * max(
                0.0,
                float(overall["candidate"]["violation_fraction"]["mean"])
                - float(experiment["validation"]["max_violation_fraction"]),
            )
        ),
        "development_gate": development_gate,
        "all_profiles_pass": all_profiles_pass,
        "sealed_test_accessed": False,
    }
    if include_episode_records:
        result["episode_records"] = episode_records
    return result


def _rollout_candidate(
    agent: ResidualSacAgent,
    experiment: dict[str, Any],
    config: S1EnvConfig,
    condition: RobustnessCondition,
    profile: HardwareProfile,
    steps: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    environment = AdaptiveOpticsEnv(config, device, profile.effects_config())
    observation, _ = environment.reset(seed=condition.base_seed)
    generator = torch.Generator(device=device).manual_seed(condition.base_seed + 40_000_000)
    noisy = _noisy_observation(
        observation, config.num_modes, profile.observation_noise_std_rad, generator
    )
    controller = _make_residual_controller(experiment, config)
    state = controller.reset(noisy)
    metrics = _empty_step_metrics()
    for step in range(steps):
        normalized = agent.act(state, deterministic=True)
        action = controller.compose_action(normalized)
        observation, _, _, _, info = environment.step(action.final_delta_rad)
        _append_step_metrics(metrics, info)
        if step + 1 < steps:
            noisy = _noisy_observation(
                observation,
                config.num_modes,
                profile.observation_noise_std_rad,
                generator,
            )
            state = controller.advance_observation(noisy)
    return _mean_step_metrics(metrics)


def _rollout_baseline(
    experiment: dict[str, Any],
    config: S1EnvConfig,
    condition: RobustnessCondition,
    profile: HardwareProfile,
    steps: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    environment = AdaptiveOpticsEnv(config, device, profile.effects_config())
    observation, _ = environment.reset(seed=condition.base_seed)
    generator = torch.Generator(device=device).manual_seed(condition.base_seed + 40_000_000)
    observation = _noisy_observation(
        observation, config.num_modes, profile.observation_noise_std_rad, generator
    )
    parameters = experiment["frozen_controller"]["parameters"]
    anchor_modes = int(
        experiment["frozen_controller"].get("active_anchor_modes", config.num_modes)
    )
    controller = TrackingLeakyIntegratorController(
        num_modes=anchor_modes,
        modal_limit_rad=config.modal_limit_rad,
        gain=float(parameters["gain"]),
        leak=float(parameters["leak"]),
        tracking_gain=float(parameters["tracking_gain"]),
        max_request_step_rad=float(parameters["max_request_step_rad"]),
    )
    controller.reset(config.batch_size, device, observation.dtype)
    metrics = _empty_step_metrics()
    for step in range(steps):
        if anchor_modes == config.num_modes:
            action = controller.action(observation)
        else:
            anchor_observation = _extract_anchor_observation(
                observation,
                num_modes=config.num_modes,
                anchor_modes=anchor_modes,
            )
            action = torch.zeros(
                config.batch_size,
                config.num_modes,
                device=device,
                dtype=observation.dtype,
            )
            action[:, :anchor_modes] = controller.action(anchor_observation)
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


def _paired_evaluation_record(
    identifier: str,
    candidate_lists: dict[str, list[torch.Tensor]],
    baseline_lists: dict[str, list[torch.Tensor]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    candidate = {key: torch.cat(value) for key, value in candidate_lists.items()}
    baseline = {key: torch.cat(value) for key, value in baseline_lists.items()}
    paired = {
        key: _distribution(candidate[key] - baseline[key])
        for key in candidate
        if key != "measured_power_in_bucket"
    }
    candidate_summary = {key: _distribution(value) for key, value in candidate.items()}
    baseline_summary = {key: _distribution(value) for key, value in baseline.items()}
    relative_gain = (
        candidate_summary["power_in_bucket"]["mean"]
        - baseline_summary["power_in_bucket"]["mean"]
    ) / baseline_summary["power_in_bucket"]["mean"]
    passed = (
        relative_gain >= float(gate["proposed_min_relative_power_gain"])
        and paired["power_in_bucket"]["ci95_low"]
        > float(gate["min_power_delta_ci95_low"])
        and paired["strehl"]["ci95_low"] > float(gate["min_strehl_delta_ci95_low"])
        and paired["phase_rmse"]["ci95_high"]
        < float(gate["max_phase_rmse_delta_ci95_high"])
        and candidate_summary["violation_fraction"]["mean"]
        <= float(gate["max_violation_fraction"])
    )
    return {
        "id": identifier,
        "episodes": int(candidate["power_in_bucket"].numel()),
        "candidate": candidate_summary,
        "baseline": baseline_summary,
        "paired_delta_candidate_minus_baseline": paired,
        "relative_power_gain": relative_gain,
        "development_gate": "PASS" if passed else "FAIL",
    }


def _residual_reward(
    *,
    measured_power: torch.Tensor,
    normalized_residual: torch.Tensor,
    violation: torch.Tensor,
    reward_config: dict[str, Any],
) -> torch.Tensor:
    return (
        float(reward_config["measured_power_weight"]) * measured_power
        - float(reward_config["residual_action_weight"])
        * normalized_residual.square().mean(dim=-1)
        - float(reward_config["violation_weight"]) * violation
    )


def _noisy_observation(
    observation: torch.Tensor,
    num_modes: int,
    noise_std_rad: float,
    generator: torch.Generator,
) -> torch.Tensor:
    noisy = observation.clone()
    if noise_std_rad > 0:
        noise = torch.randn(
            observation.shape[0],
            num_modes,
            device=observation.device,
            dtype=observation.dtype,
            generator=generator,
        )
        noisy[:, :num_modes] += noise_std_rad * noise
    return noisy


def _extract_anchor_observation(
    observation: torch.Tensor,
    *,
    num_modes: int,
    anchor_modes: int,
) -> torch.Tensor:
    """从高维观测中提取冻结传统控制器使用的低维观测。"""
    if not 0 < anchor_modes <= num_modes:
        raise ValueError("anchor_modes must be in [1, num_modes]")
    expected = 2 * num_modes + 2
    if observation.ndim != 2 or observation.shape[1] != expected:
        raise ValueError(f"observation must have shape [batch, {expected}]")
    return torch.cat(
        (
            observation[:, :anchor_modes],
            observation[:, num_modes : num_modes + anchor_modes],
            observation[:, -2:],
        ),
        dim=-1,
    )


def _make_residual_controller(
    experiment: dict[str, Any], config: S1EnvConfig
) -> ResidualTrackingController | AnchoredResidualTrackingController:
    parameters = experiment["frozen_controller"]["parameters"]
    anchor_modes = int(
        experiment["frozen_controller"].get("active_anchor_modes", config.num_modes)
    )
    if anchor_modes < config.num_modes:
        if not bool(experiment["action"].get("preserve_total_phase_rms_budget", False)):
            raise RuntimeError(
                "higher-dimensional residual control must preserve the ten-mode total action budget"
            )
        budget_anchor_modes = int(
            experiment["action"].get("total_budget_anchor_modes", anchor_modes)
        )
        if budget_anchor_modes != anchor_modes:
            raise RuntimeError("action budget anchor differs from frozen controller anchor")
        return AnchoredResidualTrackingController(
            num_modes=config.num_modes,
            anchor_modes=anchor_modes,
            modal_limit_rad=config.modal_limit_rad,
            history_frames=int(experiment["policy_observation"]["history_frames"]),
            residual_action_limit_rad=float(
                experiment["action"]["residual_action_limit_rad"]
            ),
            final_action_step_limit_rad=float(
                experiment["action"]["final_action_step_limit_rad"]
            ),
            gain=float(parameters["gain"]),
            leak=float(parameters["leak"]),
            tracking_gain=float(parameters["tracking_gain"]),
        )
    return ResidualTrackingController(
        num_modes=config.num_modes,
        modal_limit_rad=config.modal_limit_rad,
        history_frames=int(experiment["policy_observation"]["history_frames"]),
        residual_action_limit_rad=float(experiment["action"]["residual_action_limit_rad"]),
        final_action_step_limit_rad=float(
            experiment["action"]["final_action_step_limit_rad"]
        ),
        gain=float(parameters["gain"]),
        leak=float(parameters["leak"]),
        tracking_gain=float(parameters["tracking_gain"]),
    )


def _verify_s4d1(upstream: dict[str, Any]) -> dict[str, Any]:
    for field in ("summary", "source_manifest", "trajectory_manifest", "audit_record"):
        path = _project_path(upstream[field])
        if _file_sha256(path) != str(upstream[f"{field}_sha256"]):
            raise RuntimeError(f"S4-D1 {field} hash mismatch")
    audit = _project_path(upstream["audit_record"]).read_text(encoding="utf-8")
    if "Verification Status: `ANALYZED`" not in audit:
        raise RuntimeError("S4-D1 audit record is not ANALYZED")
    summary_path = _project_path(upstream["summary"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    gate = summary["s4d1_unseen_validation_gate"]
    if gate["validation_gate"] != upstream["required_gate"]:
        raise RuntimeError("S4-D1 gate is not PASS")
    if gate["frozen_controller"] != upstream["frozen_controller"]:
        raise RuntimeError("S4-D1 frozen controller changed")
    manifest = json.loads(
        _project_path(upstream["trajectory_manifest"]).read_text(encoding="utf-8")
    )
    for relative, digest in manifest.items():
        path = _project_path(relative)
        if not path.is_file() or _file_sha256(path) != digest:
            raise RuntimeError(f"S4-D1 trajectory integrity mismatch: {relative}")
    episode_seeds = {
        int(seed)
        for item in summary["controller_profile_results"]
        for seed in item["episode_seeds"]
    }
    return {
        "summary": summary,
        "summary_sha256": _file_sha256(summary_path),
        "frozen_controller": gate["frozen_controller"],
        "episode_seeds": episode_seeds,
        "trajectory_count": len(manifest),
    }


def _validate_seed_partitions(
    settings: dict[str, Any], d1_seeds: set[int]
) -> None:
    span = int(settings["training_seed_span_per_policy"])
    training_sets = [
        set(range(int(base), int(base) + span))
        for base in settings["training_environment_seed_bases"]
    ]
    for index, left in enumerate(training_sets):
        if left & d1_seeds:
            raise RuntimeError("S4-D2 training seeds overlap S4-D1")
        for right in training_sets[index + 1 :]:
            if left & right:
                raise RuntimeError("policy training seed ranges overlap")
    validation_seeds = {
        int(item["base_seed"]) + offset
        for item in settings["validation_physical_conditions"]
        for offset in range(int(settings["validation_batch_size"]))
    }
    if validation_seeds & d1_seeds:
        raise RuntimeError("S4-D2 validation seeds overlap S4-D1")
    if any(validation_seeds & training for training in training_sets):
        raise RuntimeError("S4-D2 training and validation seeds overlap")
    reserved = int(settings["reserved_s4d3_seed_base"])
    if reserved in validation_seeds or any(reserved in training for training in training_sets):
        raise RuntimeError("reserved S4-D3 seed namespace overlaps S4-D2")


def _effective_settings(experiment: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    sac = experiment["sac"]
    settings: dict[str, Any] = {
        "quick": quick,
        "output_directory": experiment["outputs"]["directory"],
        "policy_seeds": list(experiment["policy_seeds"]),
        "training_environment_seed_bases": list(
            experiment["training_environment_seed_bases"]
        ),
        "training_seed_span_per_policy": int(experiment["training_seed_span_per_policy"]),
        "training_physical_conditions": deepcopy(experiment["training_physical_conditions"]),
        "training_hardware_profile_ids": list(experiment["training_hardware_profile_ids"]),
        "validation_physical_conditions": deepcopy(
            experiment["validation_physical_conditions"]
        ),
        "interval_validation_profile_ids": list(
            experiment["interval_validation_profile_ids"]
        ),
        "final_validation_profile_ids": list(experiment["final_validation_profile_ids"]),
        "reserved_s4d3_seed_base": int(experiment["reserved_s4d3_seed_base"]),
        "environment_batch_size": 32,
        "episode_length": int(experiment["validation"]["steps"]),
        "validation_batch_size": int(
            experiment["validation"]["episodes_per_physical_condition"]
        ),
        "validation_steps": int(experiment["validation"]["steps"]),
        "replay_buffer_size": int(sac["replay_buffer_size"]),
        "sac_batch_size": int(sac["batch_size"]),
        "warmup_transitions": int(sac["warmup_transitions"]),
        "total_transitions_per_seed": int(sac["total_transitions_per_seed"]),
        "updates_per_environment_step": int(sac["updates_per_environment_step"]),
        "validation_interval_transitions": int(sac["validation_interval_transitions"]),
        "checkpoint_interval_transitions": int(sac["checkpoint_interval_transitions"]),
        "log_interval_transitions": int(sac["log_interval_transitions"]),
        "persist_final_episode_records": bool(
            experiment["validation"].get("persist_per_episode_records", False)
        ),
        "progress_stage_label": str(
            experiment.get("metadata", {}).get("progress_stage_label", "S4-D2")
        ),
    }
    if quick:
        quick_settings = experiment["quick"]
        for field in (
            "total_transitions_per_seed",
            "warmup_transitions",
            "replay_buffer_size",
            "updates_per_environment_step",
            "validation_interval_transitions",
            "checkpoint_interval_transitions",
            "log_interval_transitions",
        ):
            settings[field] = int(quick_settings[field])
        settings["sac_batch_size"] = int(quick_settings["batch_size"])
        settings["environment_batch_size"] = int(quick_settings["environment_batch_size"])
        settings["episode_length"] = int(quick_settings["episode_length"])
        settings["validation_batch_size"] = int(quick_settings["environment_batch_size"])
        settings["validation_steps"] = int(quick_settings["validation_steps"])
        settings["policy_seeds"] = list(quick_settings["policy_seeds"])
        settings["training_environment_seed_bases"] = list(
            quick_settings["training_environment_seed_bases"]
        )
        settings["training_hardware_profile_ids"] = list(
            quick_settings["training_hardware_profile_ids"]
        )
        settings["validation_physical_conditions"] = deepcopy(
            quick_settings["validation_physical_conditions"]
        )
        settings["interval_validation_profile_ids"] = list(
            quick_settings["validation_hardware_profile_ids"]
        )
        settings["final_validation_profile_ids"] = list(
            quick_settings["validation_hardware_profile_ids"]
        )
        settings["output_directory"] = experiment["outputs"]["quick_directory"]
    return settings


def _profiles(
    experiment: dict[str, Any], identifiers: Iterable[str]
) -> list[HardwareProfile]:
    source = _load_yaml(_project_path(experiment["hardware_profile_source"]))
    by_id = {
        str(item["id"]): HardwareProfile.from_mapping(item)
        for item in source["hardware_profiles"]
    }
    requested = list(map(str, identifiers))
    missing = [identifier for identifier in requested if identifier not in by_id]
    if missing:
        raise KeyError(f"unknown hardware profiles: {missing}")
    return [by_id[identifier] for identifier in requested]


def _empty_step_metrics() -> dict[str, list[torch.Tensor]]:
    return {
        "power_in_bucket": [],
        "measured_power_in_bucket": [],
        "strehl": [],
        "phase_rmse": [],
        "violation_fraction": [],
    }


def _empty_metric_lists() -> dict[str, list[torch.Tensor]]:
    return _empty_step_metrics()


def _append_step_metrics(
    metrics: dict[str, list[torch.Tensor]], info: dict[str, torch.Tensor]
) -> None:
    metrics["power_in_bucket"].append(info["reward_power_in_bucket"].detach().cpu())
    metrics["measured_power_in_bucket"].append(
        info["measured_power_in_bucket"].detach().cpu()
    )
    metrics["strehl"].append(info["reward_strehl"].detach().cpu())
    metrics["phase_rmse"].append(info["reward_phase_rmse"].detach().cpu())
    metrics["violation_fraction"].append(info["violation_fraction"].detach().cpu())


def _mean_step_metrics(
    metrics: dict[str, list[torch.Tensor]]
) -> dict[str, torch.Tensor]:
    return {key: torch.stack(values, dim=1).mean(dim=1) for key, values in metrics.items()}


def _extend_metrics(
    target: dict[str, list[torch.Tensor]], values: dict[str, torch.Tensor]
) -> None:
    for key, value in values.items():
        target[key].append(value)


def _paired_episode_records(
    profile_id: str,
    condition_id: str,
    base_seed: int,
    candidate: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    """保留每个配对验证回合，供独立统计与表示间比较重算。"""
    episode_count = int(candidate["power_in_bucket"].numel())
    if any(int(values.numel()) != episode_count for values in candidate.values()):
        raise RuntimeError("candidate validation metrics have inconsistent episode counts")
    if any(int(values.numel()) != episode_count for values in baseline.values()):
        raise RuntimeError("baseline validation metrics have inconsistent episode counts")
    records: list[dict[str, Any]] = []
    for index in range(episode_count):
        record: dict[str, Any] = {
            "profile_id": profile_id,
            "condition_id": condition_id,
            "episode_index": index,
            "episode_seed": int(base_seed) + index,
        }
        for metric in candidate:
            candidate_value = float(candidate[metric][index])
            baseline_value = float(baseline[metric][index])
            record[f"candidate_{metric}"] = candidate_value
            record[f"baseline_{metric}"] = baseline_value
            record[f"delta_{metric}"] = candidate_value - baseline_value
        records.append(record)
    return records


def _write_episode_records_csv(
    path: Path,
    records: list[dict[str, Any]],
    *,
    representation_id: str,
    policy_seed: int,
) -> None:
    if not records:
        raise ValueError("episode records must not be empty")
    fieldnames = ["representation_id", "policy_seed", *records[0].keys()]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "representation_id": representation_id,
                    "policy_seed": policy_seed,
                    **record,
                }
            )


def _distribution(values: torch.Tensor) -> dict[str, float]:
    values = values.to(torch.float64)
    mean = values.mean()
    if values.numel() > 1:
        half_width = 1.96 * values.std(unbiased=True) / values.numel() ** 0.5
    else:
        half_width = torch.tensor(float("nan"), dtype=torch.float64)
    return {
        "mean": float(mean),
        "median": float(values.median()),
        "ci95_low": float(mean - half_width),
        "ci95_high": float(mean + half_width),
    }


def _summarize_policy_seeds(
    results: list[dict[str, Any]], settings: dict[str, Any]
) -> dict[str, Any]:
    gains = torch.tensor(
        [
            item["final_development_validation"]["overall"]["relative_power_gain"]
            for item in results
        ],
        dtype=torch.float64,
    )
    passes = sum(
        item["final_development_validation"]["development_gate"] == "PASS"
        for item in results
    )
    return {
        "policy_seed_count": len(results),
        "pass_count": passes,
        "relative_power_gain_across_policy_seeds": _distribution(gains),
        "status": "ANALYSIS_REQUIRED",
        "s4d3_authorized": False,
        "training_transitions_per_seed": settings["total_transitions_per_seed"],
    }


def _source_manifest(paths: Iterable[str]) -> dict[str, str]:
    return {str(path): _file_sha256(_project_path(path)) for path in paths}


def _runtime_record() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }


def _git_record() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False) + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def json_safe(value: Any) -> Any:
    """把未定义的早期损失转换为JSON null，保持JSON/JSONL严格可解析。"""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())
