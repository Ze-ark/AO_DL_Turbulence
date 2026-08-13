"""S4-D2-R2冻结监督策略的闭环分布漂移诊断。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from src.rl.residual_control import project_box_and_l2
from src.rl.s4_high_order_learnability import (
    HighOrderImitationPolicy,
    Predictor,
    TeacherScenario,
    _append_jsonl,
    _collect_science,
    _concatenate,
    _episode_rows,
    _model_predictor,
    _paired_science_summary,
    _ridge_predictor,
    _scenario_row,
    _student_controller,
    _summarize_teacher_scenarios,
    _teacher_controller,
    _write_csv,
)
from src.rl.s4_oracle_bound import SCIENCE_METRICS
from src.rl.s4_r2_diagnostic import _rollout_anchored_baseline
from src.rl.s4_representation_capacity import (
    ANCHOR_MODES,
    ActionRepresentation,
    _future_disturbance_sequence,
    build_action_basis,
    representation_registration_inverse,
)
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
    _runtime_record,
    _source_manifest,
    _write_json,
    json_safe,
)
from src.runtime import resolve_device
from src.simulation.config import S1EnvConfig, load_s1_config
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.hardware_effects import HardwareProfile
from src.simulation.robust_control import RobustnessCondition
from src.training_progress import advance_to, counted_progress, update_progress


TELEMETRY_METRICS = (
    "raw_oracle_mse",
    "applied_oracle_mse",
    "raw_oracle_cosine",
    "state_abs_z_mean",
    "state_abs_z_gt3_fraction",
    "prior_added_request_rms_rad",
    "post_added_request_rms_rad",
    "requested_added_residual_rms_rad",
    "realized_added_residual_rms_rad",
    "residual_projection_fraction",
    "final_projection_fraction",
    "violation_fraction",
)


@dataclass(frozen=True)
class FrozenPredictor:
    """一个只读预测器及其训练状态归一化。"""

    identifier: str
    predict: Predictor
    state_mean: torch.Tensor
    state_scale: torch.Tensor


@dataclass(frozen=True)
class ShiftScenario:
    """一个预测器、缩放、硬件档位和动态条件的完整回合结果。"""

    predictor_id: str
    scale: float
    profile_id: str
    condition_id: str
    base_seed: int
    candidate: dict[str, torch.Tensor]
    baseline: dict[str, torch.Tensor]
    temporal: dict[str, dict[str, torch.Tensor]]


def run_s4_closed_loop_shift(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """冻结全部检查点，运行缩放和策略状态教师误差诊断。"""
    experiment_path = _project_path(config_path)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight = preflight_s4_closed_loop_shift(
        experiment_path,
        experiment,
        settings,
        quick=quick,
    )
    device = resolve_device("cuda")
    preflight["cuda_device"] = str(device)
    preflight["gpu_name"] = torch.cuda.get_device_name(device)
    if preflight_only:
        return preflight

    output_directory = _project_path(settings["output_directory"])
    if output_directory.exists():
        raise FileExistsError(
            "closed-loop-shift output already exists; preserve it for audit: "
            f"{output_directory}"
        )
    output_directory.mkdir(parents=True)
    _write_json(output_directory / "preflight.json", preflight)
    _write_json(output_directory / "effective_config.json", settings)
    _write_json(
        output_directory / "source_manifest.json",
        _source_manifest(experiment["tracked_source_files"]),
    )

    started = time.perf_counter()
    predictors = _load_predictors(experiment, device=device)
    offline_references = _upstream_offline_references(experiment)
    base_config, _ = load_s1_config(_project_path(experiment["environment_config"]))
    representation = ActionRepresentation.from_mapping(experiment["representation"])
    base_config = replace(base_config, num_modes=representation.num_modes)
    basis, pupil, basis_diagnostics = build_action_basis(
        base_config,
        representation,
        device,
    )
    profiles = _profiles(experiment, settings["profile_ids"])
    conditions = [
        RobustnessCondition.from_mapping(item) for item in settings["conditions"]
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

    progress_path = output_directory / "progress.jsonl"
    total_scenarios = len(profiles) * len(conditions) * (
        2 + len(predictors) * len(settings["scales"])
    )
    bar = counted_progress(
        total=total_scenarios,
        description="闭环分布漂移诊断",
        unit="场景",
    )
    completed = 0
    baselines: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    futures: dict[tuple[str, str], torch.Tensor] = {}
    configs: dict[tuple[str, str], S1EnvConfig] = {}
    teacher_scenarios: list[TeacherScenario] = []
    control_episode_records: list[dict[str, Any]] = []

    for profile in profiles:
        for condition in conditions:
            key = (profile.identifier, condition.identifier)
            config = _scenario_config(
                base_config,
                condition,
                profile,
                episodes=int(settings["episodes_per_condition"]),
                steps=int(settings["steps_per_episode"]),
                preview=int(experiment["teacher"]["preview_horizon_frames"]),
            )
            configs[key] = config
            future_truth = _future_disturbance_sequence(
                config=config,
                condition=condition,
                profile=profile,
                length=int(settings["steps_per_episode"])
                + int(experiment["teacher"]["preview_horizon_frames"]),
                basis=basis,
                device=device,
            )
            futures[key] = future_truth
            baseline = _rollout_anchored_baseline(
                experiment,
                config,
                condition,
                profile,
                int(settings["steps_per_episode"]),
                device,
            )
            baselines[key] = baseline
            completed += 1
            _progress_event(
                bar,
                progress_path,
                completed,
                total_scenarios,
                phase="baseline",
                profile=profile.identifier,
                condition=condition.identifier,
                device=device,
            )

            teacher, label_delta = _rollout_teacher(
                experiment=experiment,
                config=config,
                condition=condition,
                profile=profile,
                steps=int(settings["steps_per_episode"]),
                basis=basis,
                future_truth=future_truth,
                registration_mapping=mappings[profile.identifier],
                device=device,
            )
            teacher_scenarios.append(
                TeacherScenario(
                    profile_id=profile.identifier,
                    condition_id=condition.identifier,
                    base_seed=condition.base_seed,
                    candidate=teacher,
                    baseline=baseline,
                )
            )
            control_episode_records.extend(
                _control_episode_rows(
                    profile.identifier,
                    condition.identifier,
                    condition.base_seed,
                    teacher,
                    baseline,
                    label_delta,
                )
            )
            completed += 1
            _progress_event(
                bar,
                progress_path,
                completed,
                total_scenarios,
                phase="teacher_positive_control",
                profile=profile.identifier,
                condition=condition.identifier,
                device=device,
                metric=float(
                    (teacher["power_in_bucket"] - baseline["power_in_bucket"]).mean()
                ),
            )

    scenarios: list[ShiftScenario] = []
    scenario_records: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    temporal_records: list[dict[str, Any]] = []
    temporal_episode_records: list[dict[str, Any]] = []
    projection_tolerance = float(
        experiment["interpretation_thresholds"][
            "projection_comparison_tolerance_rad"
        ]
    )

    for predictor in predictors:
        for scale in settings["scales"]:
            for profile in profiles:
                for condition in conditions:
                    key = (profile.identifier, condition.identifier)
                    candidate, temporal = _rollout_predictor(
                        frozen=predictor,
                        scale=float(scale),
                        experiment=experiment,
                        config=configs[key],
                        condition=condition,
                        profile=profile,
                        steps=int(settings["steps_per_episode"]),
                        time_bins=settings["time_bins"],
                        basis=basis,
                        future_truth=futures[key],
                        registration_mapping=mappings[profile.identifier],
                        projection_tolerance=projection_tolerance,
                        device=device,
                    )
                    baseline = baselines[key]
                    item = ShiftScenario(
                        predictor_id=predictor.identifier,
                        scale=float(scale),
                        profile_id=profile.identifier,
                        condition_id=condition.identifier,
                        base_seed=condition.base_seed,
                        candidate=candidate,
                        baseline=baseline,
                        temporal=temporal,
                    )
                    scenarios.append(item)
                    scenario_records.append(_shift_scenario_row(item))
                    episode_records.extend(_shift_episode_rows(item))
                    temporal_records.extend(_temporal_scenario_rows(item))
                    temporal_episode_records.extend(_temporal_episode_rows(item))
                    completed += 1
                    _progress_event(
                        bar,
                        progress_path,
                        completed,
                        total_scenarios,
                        phase="predictor_scale_rollout",
                        profile=profile.identifier,
                        condition=condition.identifier,
                        predictor=predictor.identifier,
                        scale=float(scale),
                        device=device,
                        metric=float(
                            (
                                candidate["power_in_bucket"]
                                - baseline["power_in_bucket"]
                            ).mean()
                        ),
                    )
    bar.close()

    teacher_summary = _summarize_teacher_scenarios(
        teacher_scenarios,
        gate=experiment["science_gate"],
    )
    grouped_results: dict[str, dict[str, Any]] = {}
    for predictor in predictors:
        grouped_results[predictor.identifier] = {}
        for scale in settings["scales"]:
            selected = [
                item
                for item in scenarios
                if item.predictor_id == predictor.identifier
                and math.isclose(item.scale, float(scale), abs_tol=1e-12)
            ]
            grouped_results[predictor.identifier][_scale_key(float(scale))] = (
                _summarize_shift_scenarios(
                    selected,
                    gate=experiment["science_gate"],
                    time_bins=settings["time_bins"],
                    thresholds=experiment["interpretation_thresholds"],
                    offline_reference=offline_references[predictor.identifier],
                )
            )

    zero_alignment = _zero_alignment(
        scenarios,
        tolerance=float(
            experiment["interpretation_thresholds"]["zero_alignment_max_abs_error"]
        ),
    )
    if zero_alignment["status"] != "PASS":
        raise RuntimeError("zero-scale alignment with the frozen baseline failed")
    interpretation = _interpretation(
        grouped_results,
        teacher_summary=teacher_summary,
        thresholds=experiment["interpretation_thresholds"],
        quick=quick,
    )

    _write_csv(output_directory / "control_episode_records.csv", control_episode_records)
    _write_csv(output_directory / "scenario_records.csv", scenario_records)
    _write_csv(output_directory / "episode_records.csv", episode_records)
    _write_csv(output_directory / "temporal_records.csv", temporal_records)
    _write_csv(
        output_directory / "temporal_episode_records.csv", temporal_episode_records
    )
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run-diagnostic",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "s4d2_r2_closed_loop_shift_v1",
        },
        "experiment": {
            "id": "AO-S4-D2-R2-CLOSED-LOOP-SHIFT",
            "type": "software_only_cuda_frozen_checkpoint_diagnostic",
            "status": "completed_pending_audit",
            "quick": quick,
            "duration_seconds": time.perf_counter() - started,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "evidence_boundary": {
            "training_performed": False,
            "optimizer_updates": 0,
            "checkpoint_writes": 0,
            "frozen_checkpoint_reads": len(predictors),
            "future_truth_used_for_diagnostic_teacher": True,
            "future_truth_used_by_predictors": False,
            "new_development_seeds": True,
            "s4d3_accessed": False,
            "real_slm_actions": False,
        },
        "inputs": {
            "experiment_config": _relative(experiment_path),
            "experiment_config_sha256": _file_sha256(experiment_path),
            "upstream_summary_sha256": experiment["upstream_learnability"][
                "summary_sha256"
            ],
            "upstream_audit_sha256": experiment["upstream_learnability"][
                "audit_record_sha256"
            ],
            "checkpoint_sha256": {
                item["id"]: item["checkpoint_sha256"]
                for item in experiment["predictors"]
            },
        },
        "design": {
            "representation": experiment["representation"],
            "predictor_ids": [item.identifier for item in predictors],
            "scales": settings["scales"],
            "steps_per_episode": settings["steps_per_episode"],
            "episodes_per_condition": settings["episodes_per_condition"],
            "profile_ids": settings["profile_ids"],
            "condition_ids": [item.identifier for item in conditions],
            "time_bins": settings["time_bins"],
            "upstream_offline_references": offline_references,
        },
        "basis_diagnostics": basis_diagnostics,
        "mapping_diagnostics": mapping_diagnostics,
        "integrity": {
            "zero_scale_alignment": zero_alignment,
            "label_semantics": {
                "stored_upstream_target": "requested_added_residual",
                "diagnostic_teacher_comparison": "requested_added_residual",
                "requested_and_realized_are_recorded_separately": True,
            },
        },
        "teacher_positive_control": teacher_summary,
        "predictor_scale_results": grouped_results,
        "interpretation": interpretation,
        "record_counts": {
            "control_episode_records": len(control_episode_records),
            "scenario_records": len(scenario_records),
            "episode_records": len(episode_records),
            "temporal_records": len(temporal_records),
            "temporal_episode_records": len(temporal_episode_records),
        },
        "runtime": _runtime_record(),
        "git": _git_record(),
        "next_action": (
            "Stop and ask the assistant to audit the frozen-checkpoint diagnostic. "
            "Do not train RL, open S4-D3, or operate real SLM hardware."
        ),
    }
    _write_json(output_directory / "summary.json", json_safe(summary))
    return summary


def preflight_s4_closed_loop_shift(
    experiment_path: Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    """锁定封存证据、冻结检查点、新种子和硬件安全边界。"""
    metadata = experiment["metadata"]
    if str(metadata["stage"]) != "S4-D2-R2-CLOSED-LOOP-SHIFT":
        raise ValueError("closed-loop-shift stage metadata is invalid")
    expected_flags = {
        "allow_training": False,
        "allow_optimizer_updates": False,
        "allow_checkpoint_writes": False,
        "allow_checkpoint_changes": False,
        "allow_reward_change": False,
        "allow_algorithm_comparison": False,
        "allow_future_truth_for_diagnostic_teacher": True,
        "allow_future_truth_in_predictor_inputs": False,
        "allow_hardware_profile_id_in_predictor_inputs": False,
        "allow_s4d3_access": False,
        "allow_real_hardware_actions": False,
    }
    for key, expected in expected_flags.items():
        if bool(metadata.get(key)) is not expected:
            raise RuntimeError(f"invalid closed-loop-shift safety flag: {key}")

    upstream = _verify_upstream_learnability(experiment["upstream_learnability"])
    for field in ("environment_config", "hardware_profile_source"):
        path = _project_path(experiment[field])
        if _file_sha256(path) != str(experiment[f"{field}_sha256"]):
            raise RuntimeError(f"closed-loop-shift {field} hash mismatch")

    representation = experiment["representation"]
    if (
        str(representation["id"]) != "zernike_21_added_11"
        or int(representation["num_modes"]) != 21
        or int(representation["anchor_modes"]) != ANCHOR_MODES
        or int(representation["output_modes"]) != 11
        or not 0 < float(representation["pseudoinverse_rcond"]) < 1
    ):
        raise RuntimeError("closed-loop-shift must keep the added 11 mode branch")
    if int(experiment["teacher"]["preview_horizon_frames"]) != 2:
        raise RuntimeError("diagnostic teacher must keep the fixed two-frame preview")
    observation = experiment["policy_observation"]
    if (
        int(observation["history_frames"]) != 4
        or int(observation["state_size"]) != 210
        or bool(observation["include_future_truth"])
        or bool(observation["include_oracle_quality_metrics"])
        or bool(observation["include_hardware_profile_identifier"])
    ):
        raise RuntimeError("frozen predictor observation contract changed")

    predictor_ids = [str(item["id"]) for item in experiment["predictors"]]
    if predictor_ids != [
        "ridge",
        "mlp_seed_8201",
        "mlp_seed_8202",
        "mlp_seed_8203",
    ]:
        raise RuntimeError("frozen predictor inventory changed")
    inventory = _checkpoint_inventory(experiment["predictors"])
    _validate_settings(experiment, settings)
    output_directory = _project_path(settings["output_directory"])
    if output_directory.exists():
        raise FileExistsError(f"closed-loop-shift output already exists: {output_directory}")

    scenario_count = (
        len(settings["profile_ids"])
        * len(settings["conditions"])
        * len(predictor_ids)
        * len(settings["scales"])
    )
    episode_count = scenario_count * int(settings["episodes_per_condition"])
    temporal_episode_count = episode_count * len(settings["time_bins"])
    return {
        "status": "READY_FOR_USER_FROZEN_CHECKPOINT_DIAGNOSTIC",
        "quick": quick,
        "experiment_config": _relative(experiment_path),
        "experiment_config_sha256": _file_sha256(experiment_path),
        "output_directory": settings["output_directory"],
        "upstream_summary_sha256": upstream["summary_sha256"],
        "upstream_audit_sha256": upstream["audit_sha256"],
        "predictor_inventory": inventory,
        "scales": settings["scales"],
        "time_bins": settings["time_bins"],
        "profile_ids": settings["profile_ids"],
        "condition_ids": [str(item["id"]) for item in settings["conditions"]],
        "steps_per_episode": settings["steps_per_episode"],
        "episodes_per_condition": settings["episodes_per_condition"],
        "planned_records": {
            "scenario_records": scenario_count,
            "episode_records": episode_count,
            "temporal_records": scenario_count * len(settings["time_bins"]),
            "temporal_episode_records": temporal_episode_count,
        },
        "cuda_required": True,
        "training_allowed": False,
        "optimizer_updates": 0,
        "checkpoint_writes": 0,
        "future_truth_in_predictor_inputs": False,
        "s4d3_accessed": False,
        "real_slm_actions": False,
        "formal_command": (
            ".\\.venv\\Scripts\\python.exe scripts\\diagnose_s4_closed_loop_shift.py "
            "--config configs\\experiments\\s4_closed_loop_shift_v1.yaml"
        ),
    }


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


@torch.no_grad()
def _rollout_teacher(
    *,
    experiment: dict[str, Any],
    config: S1EnvConfig,
    condition: RobustnessCondition,
    profile: HardwareProfile,
    steps: int,
    basis: torch.Tensor,
    future_truth: torch.Tensor,
    registration_mapping: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    environment = AdaptiveOpticsEnv(
        config,
        device,
        profile.effects_config(),
        basis_override=basis,
    )
    observation, _ = environment.reset(seed=condition.base_seed)
    generator = torch.Generator(device=device).manual_seed(condition.base_seed + 40_000_000)
    noisy = _noisy_observation(
        observation,
        config.num_modes,
        profile.observation_noise_std_rad,
        generator,
    )
    controller = _teacher_controller(experiment, config)
    controller.reset(noisy)
    science = _empty_step_metrics()
    requested_realized_delta: list[torch.Tensor] = []
    preview = int(experiment["teacher"]["preview_horizon_frames"])
    for step in range(steps):
        target_request = (
            -future_truth[step + preview] @ registration_mapping.transpose(0, 1)
        )
        action, _, _ = controller.compose_added_modes_to_target(target_request)
        requested_realized_delta.append(
            (
                action.requested_residual_rad[:, ANCHOR_MODES:]
                - action.realized_residual_rad[:, ANCHOR_MODES:]
            )
            .abs()
            .amax(dim=-1)
            .cpu()
        )
        observation, _, _, _, info = environment.step(action.final_delta_rad)
        _append_step_metrics(science, info)
        if step + 1 < steps:
            noisy = _noisy_observation(
                observation,
                config.num_modes,
                profile.observation_noise_std_rad,
                generator,
            )
            controller.advance_observation(noisy)
    return _mean_step_metrics(science), torch.stack(requested_realized_delta, dim=1).amax(
        dim=1
    )


@torch.no_grad()
def _rollout_predictor(
    *,
    frozen: FrozenPredictor,
    scale: float,
    experiment: dict[str, Any],
    config: S1EnvConfig,
    condition: RobustnessCondition,
    profile: HardwareProfile,
    steps: int,
    time_bins: list[dict[str, Any]],
    basis: torch.Tensor,
    future_truth: torch.Tensor,
    registration_mapping: torch.Tensor,
    projection_tolerance: float,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    environment = AdaptiveOpticsEnv(
        config,
        device,
        profile.effects_config(),
        basis_override=basis,
    )
    observation, _ = environment.reset(seed=condition.base_seed)
    generator = torch.Generator(device=device).manual_seed(condition.base_seed + 40_000_000)
    noisy = _noisy_observation(
        observation,
        config.num_modes,
        profile.observation_noise_std_rad,
        generator,
    )
    controller = _student_controller(experiment, config)
    state = controller.reset(noisy)
    science = _empty_step_metrics()
    telemetry: dict[str, list[torch.Tensor]] = {
        key: [] for key in TELEMETRY_METRICS
    }
    limit = float(experiment["action_budget"]["residual_component_limit_rad"])
    preview = int(experiment["teacher"]["preview_horizon_frames"])
    state_mean = frozen.state_mean.to(device)
    state_scale = frozen.state_scale.to(device)

    for step in range(steps):
        target_request = (
            -future_truth[step + preview] @ registration_mapping.transpose(0, 1)
        )
        oracle = _oracle_normalized_action(
            state,
            target_request,
            residual_component_limit_rad=limit,
            residual_l2_budget_rad=controller.residual_l2_budget_rad,
        )
        raw = frozen.predict(state).clamp(-1, 1)
        applied = (raw * scale).clamp(-1, 1)
        state_z = (state - state_mean) / state_scale
        prior = controller.requested_modal[:, ANCHOR_MODES:].clone()
        normalized = torch.zeros(
            config.batch_size,
            config.num_modes,
            device=device,
            dtype=state.dtype,
        )
        normalized[:, ANCHOR_MODES:] = applied
        intended = normalized * limit
        action = controller.compose_action(normalized)
        post = controller.requested_modal[:, ANCHOR_MODES:].clone()
        requested_added = action.requested_residual_rad[:, ANCHOR_MODES:]
        realized_added = action.realized_residual_rad[:, ANCHOR_MODES:]

        telemetry["raw_oracle_mse"].append((raw - oracle).square().mean(dim=-1).cpu())
        telemetry["applied_oracle_mse"].append(
            (applied - oracle).square().mean(dim=-1).cpu()
        )
        telemetry["raw_oracle_cosine"].append(
            torch.nn.functional.cosine_similarity(raw, oracle, dim=-1, eps=1e-8).cpu()
        )
        telemetry["state_abs_z_mean"].append(state_z.abs().mean(dim=-1).cpu())
        telemetry["state_abs_z_gt3_fraction"].append(
            (state_z.abs() > 3).float().mean(dim=-1).cpu()
        )
        telemetry["prior_added_request_rms_rad"].append(
            prior.square().mean(dim=-1).sqrt().cpu()
        )
        telemetry["post_added_request_rms_rad"].append(
            post.square().mean(dim=-1).sqrt().cpu()
        )
        telemetry["requested_added_residual_rms_rad"].append(
            requested_added.square().mean(dim=-1).sqrt().cpu()
        )
        telemetry["realized_added_residual_rms_rad"].append(
            realized_added.square().mean(dim=-1).sqrt().cpu()
        )
        telemetry["residual_projection_fraction"].append(
            (
                (
                    action.requested_residual_rad[:, ANCHOR_MODES:]
                    - intended[:, ANCHOR_MODES:]
                ).abs()
                > projection_tolerance
            )
            .float()
            .mean(dim=-1)
            .cpu()
        )
        telemetry["final_projection_fraction"].append(
            ((requested_added - realized_added).abs() > projection_tolerance)
            .float()
            .mean(dim=-1)
            .cpu()
        )

        observation, _, _, _, info = environment.step(action.final_delta_rad)
        _append_step_metrics(science, info)
        telemetry["violation_fraction"].append(info["violation_fraction"].cpu())
        if step + 1 < steps:
            noisy = _noisy_observation(
                observation,
                config.num_modes,
                profile.observation_noise_std_rad,
                generator,
            )
            state = controller.advance_observation(noisy)

    temporal: dict[str, dict[str, torch.Tensor]] = {}
    for item in time_bins:
        start = int(item["start"])
        stop = int(item["stop"])
        temporal[str(item["id"])] = {
            metric: torch.stack(values[start:stop], dim=1).mean(dim=1)
            for metric, values in telemetry.items()
        }
    return _mean_step_metrics(science), temporal


def _oracle_normalized_action(
    state: torch.Tensor,
    target_request: torch.Tensor,
    *,
    residual_component_limit_rad: float,
    residual_l2_budget_rad: float,
) -> torch.Tensor:
    """在不改变控制器状态的情况下复算当前学生状态上的教师请求。"""
    num_modes = target_request.shape[1]
    prior = state[:, -2 * num_modes : -num_modes]
    baseline_delta = state[:, -num_modes:]
    desired = target_request.to(state.device, state.dtype) - prior - baseline_delta
    desired = desired.clone()
    desired[:, :ANCHOR_MODES] = 0
    projected, _ = project_box_and_l2(
        desired,
        component_limit=residual_component_limit_rad,
        l2_limit=residual_l2_budget_rad,
    )
    return (projected[:, ANCHOR_MODES:] / residual_component_limit_rad).clamp(-1, 1)


def _summarize_shift_scenarios(
    scenarios: list[ShiftScenario],
    *,
    gate: dict[str, Any],
    time_bins: list[dict[str, Any]],
    thresholds: dict[str, Any],
    offline_reference: dict[str, float],
) -> dict[str, Any]:
    candidates: dict[str, list[torch.Tensor]] = defaultdict(list)
    baselines: dict[str, list[torch.Tensor]] = defaultdict(list)
    by_profile_c: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_profile_b: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_condition_c: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_condition_b: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    temporal_values: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item in scenarios:
        _collect_science(candidates, item.candidate)
        _collect_science(baselines, item.baseline)
        _collect_science(by_profile_c[item.profile_id], item.candidate)
        _collect_science(by_profile_b[item.profile_id], item.baseline)
        _collect_science(by_condition_c[item.condition_id], item.candidate)
        _collect_science(by_condition_b[item.condition_id], item.baseline)
        for bin_id, values in item.temporal.items():
            for metric, tensor in values.items():
                temporal_values[bin_id][metric].append(tensor)

    identifier = f"{scenarios[0].predictor_id}_scale_{_scale_key(scenarios[0].scale)}"
    overall = _paired_science_summary(
        identifier,
        _concatenate(candidates),
        _concatenate(baselines),
        gate,
    )
    profiles = [
        _paired_science_summary(
            profile_id,
            _concatenate(by_profile_c[profile_id]),
            _concatenate(by_profile_b[profile_id]),
            gate,
        )
        for profile_id in sorted(by_profile_c)
    ]
    conditions = [
        _paired_science_summary(
            condition_id,
            _concatenate(by_condition_c[condition_id]),
            _concatenate(by_condition_b[condition_id]),
            gate,
        )
        for condition_id in sorted(by_condition_c)
    ]
    temporal = {
        str(item["id"]): {
            metric: _distribution(
                torch.cat(temporal_values[str(item["id"])][metric])
            )
            for metric in TELEMETRY_METRICS
        }
        for item in time_bins
    }
    first = str(time_bins[0]["id"])
    last = str(time_bins[-1]["id"])
    mse_ratio = temporal[last]["raw_oracle_mse"]["mean"] / max(
        temporal[first]["raw_oracle_mse"]["mean"], 1e-12
    )
    ood_increase = (
        temporal[last]["state_abs_z_gt3_fraction"]["mean"]
        - temporal[first]["state_abs_z_gt3_fraction"]["mean"]
    )
    request_ratio = temporal[last]["post_added_request_rms_rad"]["mean"] / max(
        temporal[first]["post_added_request_rms_rad"]["mean"], 1e-12
    )
    early_over_offline = temporal[first]["raw_oracle_mse"]["mean"] / max(
        float(offline_reference["mse"]), 1e-12
    )
    late_over_offline = temporal[last]["raw_oracle_mse"]["mean"] / max(
        float(offline_reference["mse"]), 1e-12
    )
    distribution_gap = min(early_over_offline, late_over_offline)
    distribution_gap_threshold = float(
        thresholds["on_policy_mse_over_offline_reference_min"]
    )
    shift_signals = {
        "upstream_offline_mse": float(offline_reference["mse"]),
        "upstream_offline_cosine": float(offline_reference["mean_cosine_similarity"]),
        "early_on_policy_mse_over_offline": early_over_offline,
        "late_on_policy_mse_over_offline": late_over_offline,
        "raw_oracle_mse_ratio_late_over_early": mse_ratio,
        "state_ood_fraction_late_minus_early": ood_increase,
        "cumulative_added_request_ratio_late_over_early": request_ratio,
        "mse_growth": mse_ratio
        >= float(thresholds["on_policy_mse_ratio_late_over_early_min"]),
        "state_ood_growth": ood_increase
        >= float(thresholds["state_ood_fraction_increase_min"]),
        "cumulative_request_growth": request_ratio
        >= float(thresholds["cumulative_added_request_ratio_late_over_early_min"]),
        "teacher_distribution_gap": distribution_gap >= distribution_gap_threshold
        or math.isclose(
            distribution_gap,
            distribution_gap_threshold,
            rel_tol=1.0e-6,
            abs_tol=1.0e-9,
        ),
    }
    shift_signals["status"] = (
        "SUPPORTED"
        if shift_signals["teacher_distribution_gap"]
        or (
            shift_signals["mse_growth"]
            and (
                shift_signals["state_ood_growth"]
                or shift_signals["cumulative_request_growth"]
            )
        )
        else "NOT_CONFIRMED"
    )
    all_profiles_pass = all(item["gate"] == "PASS" for item in profiles)
    closed_loop_gate = (
        "PASS"
        if overall["gate"] == "PASS"
        and (all_profiles_pass if bool(gate["require_all_profiles_pass"]) else True)
        else "FAIL"
    )
    return {
        "overall": overall,
        "profiles": profiles,
        "conditions": conditions,
        "all_profiles_pass": all_profiles_pass,
        "closed_loop_gate": closed_loop_gate,
        "temporal": temporal,
        "shift_signals": shift_signals,
    }


def _interpretation(
    grouped: dict[str, dict[str, Any]],
    *,
    teacher_summary: dict[str, Any],
    thresholds: dict[str, Any],
    quick: bool,
) -> dict[str, Any]:
    mlp_ids = [item for item in grouped if item.startswith("mlp_seed_")]
    per_mlp: list[dict[str, Any]] = []
    for predictor_id in mlp_ids:
        values = grouped[predictor_id]
        full = values[_scale_key(1.0)]
        reduced = [
            (float(key), value)
            for key, value in values.items()
            if 0 < float(key) < 1
        ]
        best_scale, best = max(
            reduced,
            key=lambda item: float(item[1]["overall"]["relative_power_gain"]),
        )
        improvement = float(best["overall"]["relative_power_gain"]) - float(
            full["overall"]["relative_power_gain"]
        )
        per_mlp.append(
            {
                "predictor_id": predictor_id,
                "full_scale_relative_power_gain": full["overall"][
                    "relative_power_gain"
                ],
                "best_reduced_scale": best_scale,
                "best_reduced_scale_relative_power_gain": best["overall"][
                    "relative_power_gain"
                ],
                "improvement_over_full_scale": improvement,
                "scale_sensitive": improvement
                >= float(
                    thresholds["reduced_scale_improvement_over_full_scale_min"]
                ),
                "full_scale_shift_status": full["shift_signals"]["status"],
                "safe_reduced_scale_recovery": any(
                    value["closed_loop_gate"] == "PASS"
                    for key, value in values.items()
                    if 0 < float(key) < 1
                ),
            }
        )
    shift_count = sum(
        item["full_scale_shift_status"] == "SUPPORTED" for item in per_mlp
    )
    scale_sensitive_count = sum(bool(item["scale_sensitive"]) for item in per_mlp)
    recovery_count = sum(bool(item["safe_reduced_scale_recovery"]) for item in per_mlp)
    if quick:
        status = "QUICK_SMOKE_ONLY"
    elif teacher_summary["closed_loop_gate"] != "PASS":
        status = "TEACHER_POSITIVE_CONTROL_FAILED"
    elif recovery_count == len(per_mlp):
        status = "SAFE_REDUCED_SCALE_RECOVERY"
    elif shift_count == len(per_mlp):
        status = "CLOSED_LOOP_DISTRIBUTION_SHIFT_SUPPORTED"
    elif scale_sensitive_count == len(per_mlp):
        status = "ACTION_SCALE_SENSITIVE"
    else:
        status = "MIXED_OR_UNRESOLVED"
    return {
        "status": status,
        "teacher_positive_control_pass": teacher_summary["closed_loop_gate"] == "PASS",
        "mlp_count": len(per_mlp),
        "full_scale_shift_supported_count": shift_count,
        "scale_sensitive_count": scale_sensitive_count,
        "safe_reduced_scale_recovery_count": recovery_count,
        "per_mlp": per_mlp,
        "new_training_authorized": False,
        "rl_algorithm_change_authorized": False,
        "s4d3_authorized": False,
        "real_hardware_authorized": False,
        "next_rule": "Audit the frozen-checkpoint diagnosis before any model change.",
    }


def _zero_alignment(
    scenarios: list[ShiftScenario], *, tolerance: float
) -> dict[str, Any]:
    values = []
    for item in scenarios:
        if math.isclose(item.scale, 0.0, abs_tol=1e-12):
            for metric in SCIENCE_METRICS:
                values.append(float((item.candidate[metric] - item.baseline[metric]).abs().max()))
    maximum = max(values, default=float("inf"))
    return {
        "max_abs_metric_error": maximum,
        "tolerance": tolerance,
        "status": "PASS" if maximum <= tolerance else "FAIL",
    }


def _load_predictors(
    experiment: dict[str, Any], *, device: torch.device
) -> list[FrozenPredictor]:
    values: list[FrozenPredictor] = []
    for item in experiment["predictors"]:
        checkpoint = torch.load(
            _project_path(item["checkpoint"]),
            map_location=device,
            weights_only=False,
        )
        if str(item["kind"]) == "ridge":
            predict = _ridge_predictor(
                checkpoint["weight"],
                checkpoint["state_mean"],
                checkpoint["state_scale"],
                device,
            )
        else:
            model = HighOrderImitationPolicy(
                210,
                int(item["hidden_size"]),
                11,
            ).to(device)
            model.load_state_dict(checkpoint["model"])
            predict = _model_predictor(
                model,
                checkpoint["state_mean"],
                checkpoint["state_scale"],
                device,
            )
        values.append(
            FrozenPredictor(
                identifier=str(item["id"]),
                predict=predict,
                state_mean=checkpoint["state_mean"],
                state_scale=checkpoint["state_scale"],
            )
        )
    return values


def _checkpoint_inventory(predictors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inventory = []
    reference_mean: torch.Tensor | None = None
    reference_scale: torch.Tensor | None = None
    for item in predictors:
        path = _project_path(item["checkpoint"])
        if _file_sha256(path) != str(item["checkpoint_sha256"]):
            raise RuntimeError(f"frozen checkpoint hash mismatch: {item['id']}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        tensors: list[torch.Tensor] = []

        def collect(value: Any) -> None:
            if isinstance(value, torch.Tensor):
                tensors.append(value)
            elif isinstance(value, dict):
                for child in value.values():
                    collect(child)

        collect(checkpoint)
        if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
            raise RuntimeError(f"non-finite frozen checkpoint: {item['id']}")
        mean = checkpoint["state_mean"]
        scale = checkpoint["state_scale"]
        if mean.shape != (210,) or scale.shape != (210,) or not bool((scale > 0).all()):
            raise RuntimeError(f"invalid checkpoint normalization: {item['id']}")
        if reference_mean is None:
            reference_mean = mean
            reference_scale = scale
        elif not torch.equal(mean, reference_mean) or not torch.equal(scale, reference_scale):
            raise RuntimeError("frozen predictors do not share one training normalization")
        if str(item["kind"]) == "mlp":
            if int(checkpoint["seed"]) != int(item["initialization_seed"]):
                raise RuntimeError(f"checkpoint seed mismatch: {item['id']}")
        inventory.append(
            {
                "id": str(item["id"]),
                "kind": str(item["kind"]),
                "checkpoint": str(item["checkpoint"]),
                "checkpoint_sha256": str(item["checkpoint_sha256"]),
                "seed": checkpoint.get("seed"),
                "step": checkpoint.get("step"),
                "finite": True,
            }
        )
    return inventory


def _upstream_offline_references(
    experiment: dict[str, Any],
) -> dict[str, dict[str, float]]:
    summary = json.loads(
        _project_path(experiment["upstream_learnability"]["summary"]).read_text(
            encoding="utf-8"
        )
    )
    values = {
        "ridge": summary["ridge"]["offline_diagnostic_test"],
    }
    for item in summary["mlp_initializations"]:
        values[f"mlp_seed_{int(item['initialization_seed'])}"] = item[
            "offline_diagnostic_test"
        ]
    return {
        identifier: {
            "mse": float(metrics["mse"]),
            "mean_cosine_similarity": float(metrics["mean_cosine_similarity"]),
        }
        for identifier, metrics in values.items()
    }


def _verify_upstream_learnability(upstream: dict[str, Any]) -> dict[str, str]:
    fields = (
        "summary",
        "preflight",
        "effective_config",
        "source_manifest",
        "teacher_dataset",
        "scenario_records",
        "episode_records",
        "experiment_config",
        "audit_record",
    )
    paths: dict[str, Path] = {}
    for field in fields:
        path = _project_path(upstream[field])
        if _file_sha256(path) != str(upstream[f"{field}_sha256"]):
            raise RuntimeError(f"learnability evidence hash mismatch: {field}")
        paths[field] = path
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    interpretation = summary["interpretation"]
    if (
        bool(summary["experiment"]["quick"])
        or summary["experiment"]["status"] != "completed_pending_audit"
        or interpretation["status"] != "OFFLINE_LEARNABLE_BUT_CLOSED_LOOP_FAILED"
        or int(interpretation["offline_pass_count"]) != 3
        or int(interpretation["closed_loop_pass_count"]) != 0
        or not bool(interpretation["teacher_positive_control_pass"])
    ):
        raise RuntimeError("formal learnability result is not usable")
    audit = paths["audit_record"].read_text(encoding="utf-8")
    if (
        "Verification Status: `ANALYZED`" not in audit
        or "闭环分布漂移诊断：`AUTHORIZED`" not in audit
        or "新RL训练或算法更换：`NOT AUTHORIZED`" not in audit
    ):
        raise RuntimeError("learnability audit is not finalized")
    manifest = json.loads(paths["source_manifest"].read_text(encoding="utf-8"))
    for relative, digest in manifest.items():
        path = _project_path(relative)
        if not path.is_file() or _file_sha256(path) != str(digest):
            raise RuntimeError(f"learnability source changed: {relative}")
    return {
        "summary_sha256": _file_sha256(paths["summary"]),
        "audit_sha256": _file_sha256(paths["audit_record"]),
    }


def _validate_settings(experiment: dict[str, Any], settings: dict[str, Any]) -> None:
    steps = int(settings["steps_per_episode"])
    episodes = int(settings["episodes_per_condition"])
    if steps <= 0 or episodes <= 0:
        raise ValueError("diagnostic steps and episodes must be positive")
    scales = list(map(float, settings["scales"]))
    if scales != sorted(set(scales)) or scales[0] != 0.0 or scales[-1] != 1.0:
        raise RuntimeError("diagnostic scales must be unique, sorted, and include 0 and 1")
    if any(not 0 <= value <= 1 for value in scales):
        raise ValueError("diagnostic scales must be in [0, 1]")
    cursor = 0
    for item in settings["time_bins"]:
        if int(item["start"]) != cursor or int(item["stop"]) <= cursor:
            raise RuntimeError("time bins must be contiguous and non-empty")
        cursor = int(item["stop"])
    if cursor != steps:
        raise RuntimeError("time bins must cover the complete episode")
    protected = [
        (int(item["start_inclusive"]), int(item["end_exclusive"]))
        for item in experiment["protected_seed_ranges"]
    ]
    seen: set[int] = set()
    for item in settings["conditions"]:
        start = int(item["base_seed"])
        current = set(range(start, start + episodes))
        if seen & current:
            raise RuntimeError("diagnostic episode seeds overlap")
        if any(start < stop and start + episodes > begin for begin, stop in protected):
            raise RuntimeError("diagnostic episode seeds overlap a protected range")
        seen.update(current)


def _effective_settings(experiment: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    diagnostic = experiment["diagnostic"]
    settings: dict[str, Any] = {
        "output_directory": experiment["outputs"]["directory"],
        "steps_per_episode": int(diagnostic["steps_per_episode"]),
        "episodes_per_condition": int(diagnostic["episodes_per_condition"]),
        "scales": list(map(float, diagnostic["scales"])),
        "time_bins": list(diagnostic["time_bins"]),
        "profile_ids": list(diagnostic["profile_ids"]),
        "conditions": list(diagnostic["conditions"]),
    }
    if quick:
        quick_settings = experiment["quick"]
        settings.update(
            {
                "output_directory": experiment["outputs"]["quick_directory"],
                "steps_per_episode": int(quick_settings["steps_per_episode"]),
                "episodes_per_condition": int(
                    quick_settings["episodes_per_condition"]
                ),
                "scales": list(map(float, quick_settings["scales"])),
                "time_bins": list(quick_settings["time_bins"]),
                "profile_ids": list(quick_settings["profile_ids"]),
                "conditions": list(quick_settings["conditions"]),
            }
        )
    return settings


def _shift_scenario_row(item: ShiftScenario) -> dict[str, Any]:
    row = _scenario_row(
        item.predictor_id,
        item.profile_id,
        item.condition_id,
        item.candidate,
        item.baseline,
    )
    return {"scale": item.scale, **row}


def _shift_episode_rows(item: ShiftScenario) -> list[dict[str, Any]]:
    rows = _episode_rows(
        item.predictor_id,
        item.profile_id,
        item.condition_id,
        item.base_seed,
        item.candidate,
        item.baseline,
    )
    return [{"scale": item.scale, **row} for row in rows]


def _temporal_scenario_rows(item: ShiftScenario) -> list[dict[str, Any]]:
    rows = []
    for bin_id, values in item.temporal.items():
        row: dict[str, Any] = {
            "scale": item.scale,
            "predictor_id": item.predictor_id,
            "profile": item.profile_id,
            "physical_condition": item.condition_id,
            "time_bin": bin_id,
            "episodes": int(next(iter(values.values())).numel()),
        }
        for metric, tensor in values.items():
            row[f"{metric}_mean"] = float(tensor.mean())
        rows.append(row)
    return rows


def _temporal_episode_rows(item: ShiftScenario) -> list[dict[str, Any]]:
    rows = []
    for bin_id, values in item.temporal.items():
        count = int(next(iter(values.values())).numel())
        for index in range(count):
            row: dict[str, Any] = {
                "scale": item.scale,
                "predictor_id": item.predictor_id,
                "profile": item.profile_id,
                "physical_condition": item.condition_id,
                "time_bin": bin_id,
                "episode_index": index,
                "episode_seed": item.base_seed + index,
            }
            for metric, tensor in values.items():
                row[metric] = float(tensor[index])
            rows.append(row)
    return rows


def _control_episode_rows(
    profile_id: str,
    condition_id: str,
    base_seed: int,
    teacher: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
    label_delta: torch.Tensor,
) -> list[dict[str, Any]]:
    rows = _episode_rows(
        "teacher_fixed_preview_2",
        profile_id,
        condition_id,
        base_seed,
        teacher,
        baseline,
    )
    for index, row in enumerate(rows):
        row["max_requested_minus_realized_added_abs_rad"] = float(label_delta[index])
    return rows


def _progress_event(
    bar: Any,
    progress_path: Path,
    completed: int,
    total: int,
    *,
    phase: str,
    profile: str,
    condition: str,
    device: torch.device,
    predictor: str | None = None,
    scale: float | None = None,
    metric: float | None = None,
) -> None:
    advance_to(bar, completed)
    record: dict[str, Any] = {
        "phase": phase,
        "profile": profile,
        "condition": condition,
        "completed_scenarios": completed,
        "total_scenarios": total,
    }
    if predictor is not None:
        record["predictor"] = predictor
    if scale is not None:
        record["scale"] = scale
    if metric is not None:
        record["power_delta"] = metric
    _append_jsonl(progress_path, record)
    metrics = {"功率差": metric} if metric is not None else {}
    update_progress(bar, device=device, metrics=metrics)


def _scale_key(value: float) -> str:
    return f"{value:.2f}"
