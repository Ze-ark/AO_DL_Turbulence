"""S4-D2-R1失败后的受限理想控制能力上限诊断。"""

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

from src.rl.s4_training import (
    _append_jsonl,
    _append_step_metrics,
    _distribution,
    _empty_step_metrics,
    _file_sha256,
    _git_record,
    _load_yaml,
    _make_residual_controller,
    _mean_step_metrics,
    _noisy_observation,
    _profiles,
    _project_path,
    _relative,
    _runtime_record,
    _source_manifest,
    _write_json,
    json_safe,
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
ACTION_METRICS = (
    "requested_residual_abs_mean_rad",
    "realized_residual_abs_mean_rad",
    "normalized_saturation_fraction",
    "target_request_error_abs_mean_rad",
    "selected_preview_horizon_frames",
)


def run_s4_r1_oracle_bound(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """运行受限真值预见控制器；不训练、不读取S4-D3。"""
    experiment_path = _project_path(config_path)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight = preflight_s4_r1_oracle_bound(
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
            "oracle-bound output already exists; preserve it and audit before retrying: "
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
    max_preview = max(map(int, settings["preview_horizons_frames"]))
    base_config = replace(
        base_config,
        batch_size=int(settings["batch_size"]),
        episode_length=max(
            base_config.episode_length,
            int(settings["steps"]) + max_preview,
        ),
    )
    profiles = _profiles(experiment, settings["profile_ids"])
    conditions = [
        RobustnessCondition.from_mapping(item)
        for item in settings["physical_conditions"]
    ]
    horizons = list(map(int, settings["preview_horizons_frames"]))
    scenario_count = len(profiles) * len(conditions)
    total_rollouts = scenario_count * (1 + len(horizons))
    progress = counted_progress(
        total=total_rollouts,
        description="S4-D2-R1理想上限",
        unit="轨迹组",
    )
    progress_path = output_directory / "progress.jsonl"
    completed = 0
    started = time.perf_counter()

    aggregate_candidates: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    aggregate_baselines: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    aggregate_modal_ceiling: dict[str, list[torch.Tensor]] = defaultdict(list)
    profile_envelopes: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    profile_baselines: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    scenario_records: list[dict[str, Any]] = []
    truth_alignment_max_abs = 0.0

    for profile in profiles:
        for condition in conditions:
            config = profile.environment_config(condition.environment_config(base_config))
            future_truth = _future_disturbance_modal_sequence(
                config=config,
                condition=condition,
                profile=profile,
                length=int(settings["steps"]) + max_preview,
                device=device,
            )
            baseline, modal_ceiling = _rollout_baseline_and_modal_ceiling(
                experiment=experiment,
                config=config,
                condition=condition,
                profile=profile,
                steps=int(settings["steps"]),
                device=device,
            )
            for metric, values in modal_ceiling.items():
                aggregate_modal_ceiling[metric].append(values)
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

            horizon_results: dict[int, dict[str, torch.Tensor]] = {}
            for horizon in horizons:
                candidate, alignment = _rollout_oracle_preview(
                    experiment=experiment,
                    config=config,
                    condition=condition,
                    profile=profile,
                    steps=int(settings["steps"]),
                    preview_horizon_frames=horizon,
                    future_disturbance_modal=future_truth,
                    device=device,
                )
                truth_alignment_max_abs = max(truth_alignment_max_abs, alignment)
                horizon_results[horizon] = candidate
                identifier = f"preview_{horizon}"
                _collect_metrics(aggregate_candidates[identifier], candidate)
                _collect_metrics(aggregate_baselines[identifier], baseline)
                scenario_records.append(
                    _scenario_record(
                        controller=identifier,
                        profile=profile.identifier,
                        condition=condition.identifier,
                        candidate=candidate,
                        baseline=baseline,
                    )
                )
                completed += 1
                _record_progress(
                    progress_path,
                    completed=completed,
                    total=total_rollouts,
                    profile=profile.identifier,
                    condition=condition.identifier,
                    controller=identifier,
                    power=float(candidate["power_in_bucket"].mean()),
                    started=started,
                )
                advance_to(progress, completed)
                update_progress(
                    progress,
                    device=device,
                    metrics={
                        "预见帧": float(horizon),
                        "功率差": float(
                            (candidate["power_in_bucket"] - baseline["power_in_bucket"]).mean()
                        ),
                    },
                )

            envelope = select_hindsight_envelope(horizon_results)
            _collect_metrics(aggregate_candidates["hindsight_envelope"], envelope)
            _collect_metrics(aggregate_baselines["hindsight_envelope"], baseline)
            _collect_metrics(profile_envelopes[profile.identifier], envelope)
            _collect_metrics(profile_baselines[profile.identifier], baseline)
            scenario_records.append(
                _scenario_record(
                    controller="hindsight_envelope",
                    profile=profile.identifier,
                    condition=condition.identifier,
                    candidate=envelope,
                    baseline=baseline,
                )
            )
    progress.close()

    fixed_preview_summaries = [
        _paired_summary(
            identifier,
            _concatenate_metrics(aggregate_candidates[identifier]),
            _concatenate_metrics(aggregate_baselines[identifier]),
            settings["gate"],
        )
        for identifier in (f"preview_{horizon}" for horizon in horizons)
    ]
    envelope_summary = _paired_summary(
        "hindsight_envelope",
        _concatenate_metrics(aggregate_candidates["hindsight_envelope"]),
        _concatenate_metrics(aggregate_baselines["hindsight_envelope"]),
        settings["gate"],
    )
    profile_summaries = [
        _paired_summary(
            profile.identifier,
            _concatenate_metrics(profile_envelopes[profile.identifier]),
            _concatenate_metrics(profile_baselines[profile.identifier]),
            settings["gate"],
        )
        for profile in profiles
    ]
    all_profiles_pass = all(item["capacity_gate"] == "PASS" for item in profile_summaries)
    capacity_gate = (
        "PASS"
        if envelope_summary["capacity_gate"] == "PASS" and all_profiles_pass
        else "FAIL"
    )
    modal_ceiling_summary = _science_context_summary(
        "unconstrained_instantaneous_modal_ceiling",
        _concatenate_metrics(aggregate_modal_ceiling),
        _concatenate_metrics(aggregate_baselines["hindsight_envelope"]),
    )
    best_fixed = max(
        fixed_preview_summaries,
        key=lambda item: float(item["relative_power_gain"]),
    )
    status = "QUICK_SMOKE_ONLY" if quick else (
        "CAPACITY_DEMONSTRATED" if capacity_gate == "PASS" else "CAPACITY_NOT_DEMONSTRATED"
    )
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run-diagnostic",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "s4d2_r1_oracle_capacity_bound_v1",
        },
        "experiment": {
            "id": "AO-S4-D2-R1-ORACLE-CAPACITY-BOUND",
            "type": "software_only_cuda_oracle_capability_diagnostic",
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
            "per_episode_hindsight_selection": True,
            "deployable_controller": False,
            "old_trajectories_used": False,
            "sealed_s4d3_accessed": False,
            "real_slm_actions": False,
            "interpretation": (
                "This is an optimistic empirical capability bound over a declared "
                "truth-preview family, not a mathematically global optimum or a deployable policy."
            ),
        },
        "inputs": {
            "experiment_config": _relative(experiment_path),
            "experiment_config_sha256": _file_sha256(experiment_path),
            "upstream_r1_summary_sha256": preflight["upstream_r1_summary_sha256"],
        },
        "design": {
            "paired_episode_seeds": True,
            "same_hardware_chain_as_r1": True,
            "residual_action_limit_rad": float(experiment["action"]["residual_action_limit_rad"]),
            "final_action_step_limit_rad": float(
                experiment["action"]["final_action_step_limit_rad"]
            ),
            "preview_horizons_frames": horizons,
            "selection_metric": "power_in_bucket",
            "profiles": list(settings["profile_ids"]),
            "physical_conditions": deepcopy(settings["physical_conditions"]),
            "episodes_per_physical_condition": int(settings["batch_size"]),
            "steps": int(settings["steps"]),
        },
        "truth_alignment": {
            "max_absolute_modal_difference": truth_alignment_max_abs,
            "tolerance": float(settings["truth_alignment_tolerance"]),
            "status": (
                "PASS"
                if truth_alignment_max_abs <= float(settings["truth_alignment_tolerance"])
                else "FAIL"
            ),
        },
        "fixed_preview_summaries": fixed_preview_summaries,
        "best_fixed_preview": {
            "controller": best_fixed["controller"],
            "relative_power_gain": best_fixed["relative_power_gain"],
            "capacity_gate": best_fixed["capacity_gate"],
        },
        "hindsight_envelope": {
            "overall": envelope_summary,
            "profiles": profile_summaries,
            "all_profiles_pass": all_profiles_pass,
            "capacity_gate": capacity_gate,
        },
        "absolute_modal_ceiling_context": modal_ceiling_summary,
        "interpretation": {
            "status": status,
            "capacity_demonstrated": capacity_gate == "PASS" and not quick,
            "r2_authorized": False,
            "s4d3_authorized": False,
            "next_rule": (
                "Audit first. A formal PASS only supports designing R2; a formal FAIL "
                "stops the current small-residual RL route unless a stronger bound is justified."
            ),
        },
        "runtime": _runtime_record(),
        "git": _git_record(),
        "next_action": (
            "Stop and ask the assistant to audit the oracle bound. Do not run R2 or open S4-D3."
        ),
    }
    _write_json(output_directory / "summary.json", summary)
    _write_scenario_csv(output_directory / "scenario_records.csv", scenario_records)
    return summary


def preflight_s4_r1_oracle_bound(
    experiment_path: Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    """在占用CUDA和创建输出前核对R1负结果、约束和种子隔离。"""
    metadata = experiment["metadata"]
    if str(metadata["stage"]) != "S4-D2-R1-UB":
        raise ValueError("oracle-bound metadata must identify S4-D2-R1-UB")
    forbidden = (
        "allow_training",
        "allow_optimizer_updates",
        "allow_checkpoint_updates",
        "allow_old_trajectory_access",
        "allow_s4d3_access",
        "allow_real_hardware_actions",
    )
    if any(bool(metadata.get(field, True)) for field in forbidden):
        raise RuntimeError("oracle-bound mutation and protected-data flags must stay false")
    if not bool(metadata.get("allow_simulator_truth_access", False)):
        raise RuntimeError("oracle-bound must explicitly declare simulator-truth access")

    upstream = _verify_r1_failure(experiment["upstream_r1"])
    r1_config = upstream["config"]
    for field in ("frozen_controller", "policy_observation", "action"):
        if experiment[field] != r1_config[field]:
            raise RuntimeError(f"oracle-bound changed R1 field: {field}")
    if list(experiment["evaluation"]["profile_ids"]) != list(
        r1_config["final_validation_profile_ids"]
    ):
        raise RuntimeError("oracle-bound hardware profiles differ from R1 final development")
    _verify_matching_physics(
        r1_config["validation_physical_conditions"],
        experiment["evaluation"]["physical_conditions"],
    )
    expected_gate = {
        "proposed_min_relative_power_gain": r1_config["validation"][
            "proposed_min_relative_power_gain"
        ],
        "min_power_delta_ci95_low": r1_config["validation"][
            "min_power_delta_ci95_low"
        ],
        "min_strehl_delta_ci95_low": r1_config["validation"][
            "min_strehl_delta_ci95_low"
        ],
        "max_phase_rmse_delta_ci95_high": r1_config["validation"][
            "max_phase_rmse_delta_ci95_high"
        ],
        "max_violation_fraction": r1_config["validation"]["max_violation_fraction"],
    }
    if experiment["gate"] != expected_gate:
        raise RuntimeError("oracle-bound changed the R1 development gate")

    environment_path = _project_path(experiment["environment_config"])
    hardware_path = _project_path(experiment["hardware_profile_source"])
    if _file_sha256(environment_path) != str(experiment["environment_config_sha256"]):
        raise RuntimeError("environment config hash mismatch")
    if _file_sha256(hardware_path) != str(experiment["hardware_profile_source_sha256"]):
        raise RuntimeError("hardware profile source hash mismatch")
    base_config, _ = load_s1_config(environment_path)
    controller = _make_residual_controller(experiment, base_config)
    if controller.state_size != 100 or base_config.num_modes != 10:
        raise RuntimeError("oracle-bound residual controller dimensions changed")

    horizons = list(map(int, settings["preview_horizons_frames"]))
    if not horizons or horizons != sorted(set(horizons)) or horizons[0] < 0:
        raise ValueError("preview horizons must be sorted unique non-negative integers")
    if not bool(experiment["oracle"]["per_episode_hindsight_envelope"]):
        raise RuntimeError("oracle-bound must keep the optimistic hindsight envelope")
    if str(experiment["oracle"]["selection_metric"]) != "power_in_bucket":
        raise RuntimeError("oracle-bound selection metric must stay power_in_bucket")
    _validate_oracle_seeds(experiment, settings)

    output_directory = _project_path(settings["output_directory"])
    if output_directory.exists():
        raise FileExistsError(f"oracle-bound output already exists: {output_directory}")
    if not all(_project_path(path).is_file() for path in experiment["tracked_source_files"]):
        raise FileNotFoundError("one or more oracle-bound tracked source files are missing")
    return {
        "status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_SIMULATION",
        "quick": quick,
        "experiment_config": _relative(experiment_path),
        "experiment_config_sha256": _file_sha256(experiment_path),
        "output_directory": _relative(output_directory),
        "base_environment_config": asdict(base_config),
        "state_size": controller.state_size,
        "action_size": base_config.num_modes,
        "residual_action_limit_rad": float(experiment["action"]["residual_action_limit_rad"]),
        "preview_horizons_frames": horizons,
        "scenario_count": len(settings["profile_ids"]) * len(settings["physical_conditions"]),
        "episodes_per_scenario": int(settings["batch_size"]),
        "steps": int(settings["steps"]),
        "upstream_r1_gate": "FAIL",
        "upstream_r1_summary_sha256": upstream["summary_sha256"],
        "seed_isolation_verified": True,
        "cuda_required": True,
        "training_allowed": False,
        "optimizer_updates": 0,
        "future_simulator_truth_accessed": True,
        "deployable_controller": False,
        "sealed_s4d3_accessed": False,
        "real_slm_actions": False,
        "formal_command": (
            ".\\.venv\\Scripts\\python.exe scripts\\run_s4_r1_oracle_bound.py "
            "--config configs\\experiments\\s4_r1_oracle_bound_v1.yaml"
        ),
    }


@torch.no_grad()
def _future_disturbance_modal_sequence(
    *,
    config: S1EnvConfig,
    condition: RobustnessCondition,
    profile: HardwareProfile,
    length: int,
    device: torch.device,
) -> torch.Tensor:
    environment = AdaptiveOpticsEnv(config, device, profile.effects_config())
    observation, _ = environment.reset(seed=condition.base_seed)
    zero = torch.zeros(
        config.batch_size,
        config.num_modes,
        device=device,
        dtype=observation.dtype,
    )
    values = []
    for index in range(length):
        values.append(_disturbance_modal(observation, config.num_modes).clone())
        if index + 1 < length:
            observation, _, _, _, _ = environment.step(zero)
    return torch.stack(values, dim=0)


@torch.no_grad()
def _rollout_baseline_and_modal_ceiling(
    *,
    experiment: dict[str, Any],
    config: S1EnvConfig,
    condition: RobustnessCondition,
    profile: HardwareProfile,
    steps: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    environment = AdaptiveOpticsEnv(config, device, profile.effects_config())
    observation, _ = environment.reset(seed=condition.base_seed)
    generator = torch.Generator(device=device).manual_seed(condition.base_seed + 40_000_000)
    observation = _noisy_observation(
        observation, config.num_modes, profile.observation_noise_std_rad, generator
    )
    parameters = experiment["frozen_controller"]["parameters"]
    controller = TrackingLeakyIntegratorController(
        num_modes=config.num_modes,
        modal_limit_rad=config.modal_limit_rad,
        gain=float(parameters["gain"]),
        leak=float(parameters["leak"]),
        tracking_gain=float(parameters["tracking_gain"]),
        max_request_step_rad=float(parameters["max_request_step_rad"]),
    )
    controller.reset(config.batch_size, device, observation.dtype)
    baseline = _empty_step_metrics()
    ceiling = _empty_step_metrics()
    for step in range(steps):
        ideal = environment.oracle_modal_upper_bound()
        ceiling["power_in_bucket"].append(ideal["power_in_bucket"].detach().cpu())
        ceiling["measured_power_in_bucket"].append(
            ideal["power_in_bucket"].detach().cpu()
        )
        ceiling["strehl"].append(ideal["strehl"].detach().cpu())
        ceiling["phase_rmse"].append(ideal["phase_rmse"].detach().cpu())
        ceiling["violation_fraction"].append(
            torch.zeros_like(ideal["power_in_bucket"]).cpu()
        )
        action = controller.action(observation)
        observation, _, _, _, info = environment.step(action)
        _append_step_metrics(baseline, info)
        if step + 1 < steps:
            observation = _noisy_observation(
                observation,
                config.num_modes,
                profile.observation_noise_std_rad,
                generator,
            )
    return _mean_step_metrics(baseline), _mean_step_metrics(ceiling)


@torch.no_grad()
def _rollout_oracle_preview(
    *,
    experiment: dict[str, Any],
    config: S1EnvConfig,
    condition: RobustnessCondition,
    profile: HardwareProfile,
    steps: int,
    preview_horizon_frames: int,
    future_disturbance_modal: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], float]:
    environment = AdaptiveOpticsEnv(config, device, profile.effects_config())
    observation, _ = environment.reset(seed=condition.base_seed)
    generator = torch.Generator(device=device).manual_seed(condition.base_seed + 40_000_000)
    raw_observation = observation
    noisy = _noisy_observation(
        observation, config.num_modes, profile.observation_noise_std_rad, generator
    )
    controller = _make_residual_controller(experiment, config)
    state = controller.reset(noisy)
    science = _empty_step_metrics()
    actions: dict[str, list[torch.Tensor]] = {
        key: [] for key in ACTION_METRICS if key != "selected_preview_horizon_frames"
    }
    alignment_max = 0.0
    for step in range(steps):
        actual_disturbance = _disturbance_modal(raw_observation, config.num_modes)
        alignment_max = max(
            alignment_max,
            float((actual_disturbance - future_disturbance_modal[step]).abs().max()),
        )
        future_index = min(
            step + preview_horizon_frames,
            future_disturbance_modal.shape[0] - 1,
        )
        normalized, target = oracle_normalized_residual(
            state=state,
            future_disturbance_modal=future_disturbance_modal[future_index],
            num_modes=config.num_modes,
            residual_action_limit_rad=float(
                experiment["action"]["residual_action_limit_rad"]
            ),
            modal_limit_rad=config.modal_limit_rad,
            phase_scale=profile.phase_scale,
        )
        action = controller.compose_action(normalized)
        raw_observation, _, _, _, info = environment.step(action.final_delta_rad)
        _append_step_metrics(science, info)
        actions["requested_residual_abs_mean_rad"].append(
            action.requested_residual_rad.abs().mean(dim=-1).cpu()
        )
        actions["realized_residual_abs_mean_rad"].append(
            action.realized_residual_rad.abs().mean(dim=-1).cpu()
        )
        actions["normalized_saturation_fraction"].append(
            normalized.abs().ge(0.999).float().mean(dim=-1).cpu()
        )
        actions["target_request_error_abs_mean_rad"].append(
            controller.requested_modal.sub(target).abs().mean(dim=-1).cpu()
        )
        if step + 1 < steps:
            noisy = _noisy_observation(
                raw_observation,
                config.num_modes,
                profile.observation_noise_std_rad,
                generator,
            )
            state = controller.advance_observation(noisy)
    result = _mean_step_metrics(science)
    result.update(
        {key: torch.stack(values, dim=1).mean(dim=1) for key, values in actions.items()}
    )
    result["selected_preview_horizon_frames"] = torch.full(
        (config.batch_size,),
        float(preview_horizon_frames),
        dtype=result["power_in_bucket"].dtype,
    )
    return result, alignment_max


def oracle_normalized_residual(
    *,
    state: torch.Tensor,
    future_disturbance_modal: torch.Tensor,
    num_modes: int,
    residual_action_limit_rad: float,
    modal_limit_rad: float,
    phase_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把未来真值目标投影到R1允许的逐模态残差盒中。"""
    if residual_action_limit_rad <= 0 or phase_scale <= 0:
        raise ValueError("residual limit and phase scale must be positive")
    if state.ndim != 2 or state.shape[1] < 2 * num_modes:
        raise ValueError("state does not contain requested-modal and baseline-delta fields")
    if future_disturbance_modal.shape != (state.shape[0], num_modes):
        raise ValueError("future disturbance modal shape does not match state")
    prior = state[:, -2 * num_modes : -num_modes]
    baseline_delta = state[:, -num_modes:]
    target = (-future_disturbance_modal / phase_scale).clamp(
        -modal_limit_rad,
        modal_limit_rad,
    )
    requested_residual = target - prior - baseline_delta
    normalized = (requested_residual / residual_action_limit_rad).clamp(-1, 1)
    return normalized, target


def select_hindsight_envelope(
    candidates: dict[int, dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """逐回合选择平均桶内功率最高的预见时域，并保留同一候选的全部指标。"""
    if not candidates:
        raise ValueError("hindsight envelope requires at least one candidate")
    horizons = sorted(candidates)
    metric_keys = tuple(candidates[horizons[0]])
    if any(tuple(candidates[item]) != metric_keys for item in horizons[1:]):
        raise ValueError("oracle candidates expose different metrics")
    powers = torch.stack(
        [candidates[horizon]["power_in_bucket"] for horizon in horizons],
        dim=0,
    )
    selected = powers.argmax(dim=0)
    result = {}
    for metric in metric_keys:
        stacked = torch.stack(
            [candidates[horizon][metric] for horizon in horizons],
            dim=0,
        )
        result[metric] = stacked.gather(0, selected.unsqueeze(0)).squeeze(0)
    horizon_values = torch.tensor(horizons, dtype=powers.dtype, device=powers.device)
    result["selected_preview_horizon_frames"] = horizon_values[selected].cpu()
    return result


def _paired_summary(
    identifier: str,
    candidate: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
    gate: dict[str, Any],
) -> dict[str, Any]:
    candidate_summary = {key: _distribution(candidate[key]) for key in SCIENCE_METRICS}
    baseline_summary = {key: _distribution(baseline[key]) for key in SCIENCE_METRICS}
    paired = {
        key: _distribution(candidate[key] - baseline[key])
        for key in SCIENCE_METRICS
        if key != "measured_power_in_bucket"
    }
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
        "controller": identifier,
        "episodes": int(candidate["power_in_bucket"].numel()),
        "candidate": candidate_summary,
        "baseline": baseline_summary,
        "paired_delta_candidate_minus_baseline": paired,
        "relative_power_gain": relative_gain,
        "action_diagnostics": {
            key: _distribution(candidate[key]) for key in ACTION_METRICS
        },
        "capacity_gate": "PASS" if passed else "FAIL",
    }


def _science_context_summary(
    identifier: str,
    candidate: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
) -> dict[str, Any]:
    science = {key: _distribution(candidate[key]) for key in SCIENCE_METRICS}
    reference = {key: _distribution(baseline[key]) for key in SCIENCE_METRICS}
    paired = {
        key: _distribution(candidate[key] - baseline[key])
        for key in SCIENCE_METRICS
        if key != "measured_power_in_bucket"
    }
    return {
        "controller": identifier,
        "episodes": int(candidate["power_in_bucket"].numel()),
        "candidate": science,
        "baseline": reference,
        "paired_delta_candidate_minus_baseline": paired,
        "relative_power_gain": (
            science["power_in_bucket"]["mean"] - reference["power_in_bucket"]["mean"]
        )
        / reference["power_in_bucket"]["mean"],
        "constraint_note": "Ignores residual limit, SLM delay, quantization and registration.",
    }


def _verify_r1_failure(upstream: dict[str, Any]) -> dict[str, Any]:
    paths = {}
    for field in ("summary", "source_manifest", "experiment_config", "audit_record"):
        path = _project_path(upstream[field])
        if _file_sha256(path) != str(upstream[f"{field}_sha256"]):
            raise RuntimeError(f"R1 upstream hash mismatch: {field}")
        paths[field] = path
    summary = _load_json(paths["summary"])
    development = summary["development_summary"]
    if (
        bool(summary["experiment"]["quick"])
        or int(development["pass_count"]) != 0
        or bool(development["s4d3_authorized"])
        or any(
            item["final_development_validation"]["development_gate"] != "FAIL"
            for item in summary["policy_seed_results"]
        )
    ):
        raise RuntimeError("R1 failure state changed")
    if bool(summary["evidence_boundary"]["sealed_s4d3_accessed"]):
        raise RuntimeError("R1 unexpectedly accessed S4-D3")
    audit = paths["audit_record"].read_text(encoding="utf-8")
    if "Verification Status: `ANALYZED`" not in audit or "门槛为 **FAIL**" not in audit:
        raise RuntimeError("R1 audit record is not an ANALYZED failure")
    recorded_manifest = _load_json(paths["source_manifest"])
    for relative, digest in recorded_manifest.items():
        path = _project_path(relative)
        if not path.is_file() or _file_sha256(path) != digest:
            raise RuntimeError(f"R1 tracked source changed: {relative}")
    return {
        "config": _load_yaml(paths["experiment_config"]),
        "summary_sha256": _file_sha256(paths["summary"]),
    }


def _verify_matching_physics(
    original: Iterable[dict[str, Any]], revised: Iterable[dict[str, Any]]
) -> None:
    original_items = list(original)
    revised_items = list(revised)
    if len(original_items) != len(revised_items):
        raise RuntimeError("oracle-bound changed the number of physical conditions")
    for left, right in zip(original_items, revised_items, strict=True):
        left_copy = {key: value for key, value in left.items() if key not in {"id", "base_seed"}}
        right_copy = {
            key: value for key, value in right.items() if key not in {"id", "base_seed"}
        }
        if left_copy != right_copy:
            raise RuntimeError("oracle-bound changed R1 validation physics")


def _validate_oracle_seeds(
    experiment: dict[str, Any], settings: dict[str, Any]
) -> None:
    batch = int(settings["batch_size"])
    active = {
        int(item["base_seed"]) + offset
        for item in settings["physical_conditions"]
        for offset in range(batch)
    }
    if len(active) != len(settings["physical_conditions"]) * batch:
        raise RuntimeError("oracle-bound physical conditions reuse episode seeds")
    for item in experiment["protected_seed_ranges"]:
        start = int(item["start_inclusive"])
        end = int(item["end_exclusive"])
        if any(start <= seed < end for seed in active):
            raise RuntimeError(f"oracle-bound seeds overlap protected range: {item['id']}")
    reserved_r2 = int(experiment["reserved_r2_seed_base"])
    if any(seed >= reserved_r2 for seed in active):
        raise RuntimeError("oracle-bound overlaps the reserved R2 namespace")
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
    if formal & quick:
        raise RuntimeError("oracle-bound formal and quick seeds overlap")


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
        "preview_horizons_frames": list(experiment["oracle"]["preview_horizons_frames"]),
        "truth_alignment_tolerance": float(experiment["oracle"]["truth_alignment_tolerance"]),
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
                "preview_horizons_frames": list(
                    quick_settings["preview_horizons_frames"]
                ),
            }
        )
    return settings


def _disturbance_modal(observation: torch.Tensor, num_modes: int) -> torch.Tensor:
    return observation[:, :num_modes] - observation[:, num_modes : 2 * num_modes]


def _collect_metrics(
    target: dict[str, list[torch.Tensor]], values: dict[str, torch.Tensor]
) -> None:
    for metric, tensor in values.items():
        target[metric].append(tensor.detach().cpu())


def _concatenate_metrics(
    values: dict[str, list[torch.Tensor]],
) -> dict[str, torch.Tensor]:
    return {metric: torch.cat(parts) for metric, parts in values.items()}


def _scenario_record(
    *,
    controller: str,
    profile: str,
    condition: str,
    candidate: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
) -> dict[str, Any]:
    return {
        "controller": controller,
        "profile": profile,
        "physical_condition": condition,
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
    }


def _record_progress(
    path: Path,
    *,
    completed: int,
    total: int,
    profile: str,
    condition: str,
    controller: str,
    power: float,
    started: float,
) -> None:
    _append_jsonl(
        path,
        {
            "completed_rollouts": completed,
            "total_rollouts": total,
            "profile": profile,
            "physical_condition": condition,
            "controller": controller,
            "mean_power_in_bucket": power,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


def _write_scenario_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(json_safe(rows))


def _load_json(path: Path) -> dict[str, Any]:
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON mapping: {path}")
    return value
