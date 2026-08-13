"""S4-D2-R2新增高阶动作子空间的非学习理想容量诊断。"""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from src.rl.residual_control import AnchoredResidualTrackingController, ResidualAction
from src.rl.s4_oracle_bound import (
    ACTION_METRICS,
    SCIENCE_METRICS,
    _collect_metrics,
    _concatenate_metrics,
    _paired_summary,
    _record_progress,
    select_hindsight_envelope,
)
from src.rl.s4_r2_diagnostic import _rollout_anchored_baseline
from src.rl.s4_representation_capacity import (
    ANCHOR_MODES,
    ActionRepresentation,
    _extract_anchor_observation,
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


HIGH_ORDER_ACTION_METRICS = (
    "requested_residual_phase_rms_rad",
    "realized_residual_phase_rms_rad",
    "residual_projection_fraction",
    "final_projection_fraction",
    "request_projection_fraction",
    "requested_anchor_residual_abs_mean_rad",
    "realized_anchor_residual_abs_mean_rad",
    "requested_added_residual_phase_rms_rad",
    "realized_added_residual_phase_rms_rad",
)


class AddedModesOracleController(AnchoredResidualTrackingController):
    """把理想目标转换为只在新增模式上提出的残差请求。"""

    def compose_added_modes_to_target(
        self,
        target_request: torch.Tensor,
    ) -> tuple[ResidualAction, torch.Tensor, torch.Tensor]:
        prior = self._pending_prior_requested
        baseline_delta = self._pending_baseline_delta
        if prior is None or baseline_delta is None:
            raise RuntimeError("reset or advance_observation must prepare a decision first")
        if target_request.shape != prior.shape:
            raise ValueError("target request shape does not match controller")
        prior_copy = prior.clone()
        desired_residual = target_request.to(prior.device, prior.dtype) - prior - baseline_delta
        desired_residual = desired_residual.clone()
        desired_residual[:, : self.anchor_modes] = 0
        normalized = desired_residual / self.residual_action_limit_rad
        action = self.compose_action(normalized)
        return action, desired_residual, prior_copy


def run_s4_high_order_capacity(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """运行只允许第11维以后理想残差的纯软件容量诊断。"""
    experiment_path = _project_path(config_path)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight = preflight_s4_high_order_capacity(
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
            "high-order-capacity output already exists; preserve it for audit: "
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
    base_config = replace(
        base_config,
        batch_size=int(settings["batch_size"]),
        episode_length=max(
            base_config.episode_length,
            int(settings["steps"]) + max(horizons),
        ),
    )
    representations = [
        ActionRepresentation.from_mapping(item) for item in settings["representations"]
    ]
    profiles = _profiles(experiment, settings["profile_ids"])
    conditions = [
        RobustnessCondition.from_mapping(item)
        for item in settings["physical_conditions"]
    ]
    basis_data = {
        item.identifier: build_action_basis(base_config, item, device)
        for item in representations
    }
    bases = {identifier: values[0] for identifier, values in basis_data.items()}
    pupils = {identifier: values[1] for identifier, values in basis_data.items()}
    mappings: dict[tuple[str, str], torch.Tensor] = {}
    mapping_diagnostics: dict[str, dict[str, dict[str, Any]]] = {}
    for representation in representations:
        mapping_diagnostics[representation.identifier] = {}
        for profile in profiles:
            mapping, diagnostics = representation_registration_inverse(
                basis=bases[representation.identifier],
                pupil=pupils[representation.identifier],
                profile=profile,
                rcond=float(experiment["registration_inverse"]["pseudoinverse_rcond"]),
            )
            mappings[(representation.identifier, profile.identifier)] = mapping
            mapping_diagnostics[representation.identifier][profile.identifier] = diagnostics

    scenario_count = len(profiles) * len(conditions)
    total_rollouts = len(representations) * scenario_count * (1 + len(horizons))
    progress = counted_progress(
        total=total_rollouts,
        description="S4-D2-R2新增高阶子空间容量",
        unit="轨迹组",
    )
    progress_path = output_directory / "progress.jsonl"
    started = time.perf_counter()
    completed = 0
    truth_alignment_max = 0.0
    requested_anchor_max = 0.0
    scenario_records: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    candidates = {
        item.identifier: defaultdict(list) for item in representations
    }
    baselines = {
        item.identifier: defaultdict(list) for item in representations
    }
    profile_candidates = {
        item.identifier: defaultdict(lambda: defaultdict(list))
        for item in representations
    }
    profile_baselines = {
        item.identifier: defaultdict(lambda: defaultdict(list))
        for item in representations
    }

    for representation in representations:
        representation_id = representation.identifier
        basis = bases[representation_id]
        for profile in profiles:
            for condition in conditions:
                config = replace(base_config, num_modes=representation.num_modes)
                config = profile.environment_config(condition.environment_config(config))
                baseline = _rollout_anchored_baseline(
                    experiment,
                    config,
                    condition,
                    profile,
                    int(settings["steps"]),
                    device,
                )
                completed += 1
                _record_progress(
                    progress_path,
                    completed=completed,
                    total=total_rollouts,
                    profile=profile.identifier,
                    condition=condition.identifier,
                    controller=f"{representation_id}_frozen_baseline",
                    power=float(baseline["power_in_bucket"].mean()),
                    started=started,
                )
                advance_to(progress, completed)

                future_truth = _future_disturbance_sequence(
                    config=config,
                    condition=condition,
                    profile=profile,
                    length=int(settings["steps"]) + max(horizons),
                    basis=basis,
                    device=device,
                )
                horizon_results: dict[int, dict[str, torch.Tensor]] = {}
                for horizon in horizons:
                    candidate, alignment, requested_anchor = _rollout_added_modes_preview(
                        experiment=experiment,
                        config=config,
                        condition=condition,
                        profile=profile,
                        steps=int(settings["steps"]),
                        preview_horizon_frames=horizon,
                        future_disturbance=future_truth,
                        registration_mapping=mappings[
                            (representation_id, profile.identifier)
                        ],
                        basis=basis,
                        device=device,
                    )
                    truth_alignment_max = max(truth_alignment_max, alignment)
                    requested_anchor_max = max(requested_anchor_max, requested_anchor)
                    horizon_results[horizon] = candidate
                    variant = f"added_only_preview_{horizon}"
                    scenario_records.append(
                        _scenario_record(
                            representation=representation,
                            variant=variant,
                            profile=profile,
                            condition=condition,
                            candidate=candidate,
                            baseline=baseline,
                        )
                    )
                    episode_records.extend(
                        _episode_rows(
                            representation=representation,
                            variant=variant,
                            profile=profile,
                            condition=condition,
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
                        controller=f"{representation_id}_{variant}",
                        power=float(candidate["power_in_bucket"].mean()),
                        started=started,
                    )
                    advance_to(progress, completed)
                    update_progress(
                        progress,
                        device=device,
                        metrics={
                            "动作维度": float(representation.num_modes),
                            "预见帧": float(horizon),
                            "功率差": float(
                                candidate["power_in_bucket"].mean()
                                - baseline["power_in_bucket"].mean()
                            ),
                        },
                    )

                envelope = select_hindsight_envelope(horizon_results)
                _collect_metrics(candidates[representation_id], envelope)
                _collect_metrics(baselines[representation_id], baseline)
                _collect_metrics(
                    profile_candidates[representation_id][profile.identifier],
                    envelope,
                )
                _collect_metrics(
                    profile_baselines[representation_id][profile.identifier],
                    baseline,
                )
                scenario_records.append(
                    _scenario_record(
                        representation=representation,
                        variant="added_only_hindsight_envelope",
                        profile=profile,
                        condition=condition,
                        candidate=envelope,
                        baseline=baseline,
                    )
                )
                episode_records.extend(
                    _episode_rows(
                        representation=representation,
                        variant="added_only_hindsight_envelope",
                        profile=profile,
                        condition=condition,
                        candidate=envelope,
                        baseline=baseline,
                    )
                )
    progress.close()

    summaries = []
    passing = []
    for representation in representations:
        identifier = representation.identifier
        overall = _high_order_capacity_summary(
            identifier,
            _concatenate_metrics(candidates[identifier]),
            _concatenate_metrics(baselines[identifier]),
            experiment["gate"],
        )
        per_profile = [
            _high_order_capacity_summary(
                profile.identifier,
                _concatenate_metrics(
                    profile_candidates[identifier][profile.identifier]
                ),
                _concatenate_metrics(
                    profile_baselines[identifier][profile.identifier]
                ),
                experiment["gate"],
            )
            for profile in profiles
        ]
        all_profiles_pass = all(item["capacity_gate"] == "PASS" for item in per_profile)
        target_ids = set(map(str, experiment["gate"]["target_profile_ids"]))
        target_profiles_pass = target_ids <= {
            item["controller"]
            for item in per_profile
            if item["capacity_gate"] == "PASS"
        }
        final_gate = (
            "PASS"
            if overall["capacity_gate"] == "PASS"
            and (
                all_profiles_pass
                if bool(experiment["gate"]["require_all_profiles_pass"])
                else True
            )
            and target_profiles_pass
            else "FAIL"
        )
        summaries.append(
            {
                "representation": asdict(representation),
                "added_modes": representation.num_modes - ANCHOR_MODES,
                "overall": overall,
                "profiles": per_profile,
                "all_profiles_pass": all_profiles_pass,
                "target_profiles_pass": target_profiles_pass,
                "capacity_gate": final_gate,
            }
        )
        if final_gate == "PASS":
            passing.append(identifier)

    truth_pass = truth_alignment_max <= float(
        experiment["oracle"]["truth_alignment_tolerance"]
    )
    anchor_request_pass = requested_anchor_max <= float(
        experiment["action_budget"]["requested_anchor_residual_tolerance_rad"]
    )
    integrity_pass = truth_pass and anchor_request_pass
    expected_scenario_rows = len(representations) * scenario_count * (len(horizons) + 1)
    expected_episode_rows = expected_scenario_rows * int(settings["batch_size"])
    if len(scenario_records) != expected_scenario_rows:
        raise RuntimeError("high-order-capacity scenario record count mismatch")
    if len(episode_records) != expected_episode_rows:
        raise RuntimeError("high-order-capacity episode record count mismatch")

    if quick:
        status = "QUICK_SMOKE_ONLY"
    elif not integrity_pass:
        status = "INTEGRITY_CHECK_FAILED"
    elif passing:
        status = "HIGH_ORDER_CAPACITY_FOUND"
    else:
        status = "NO_HIGH_ORDER_CAPACITY"
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run-diagnostic",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "s4d2_r2_high_order_capacity_v1",
        },
        "experiment": {
            "id": "AO-S4-D2-R2-HIGH-ORDER-CAPACITY",
            "type": "software_only_cuda_nonlearning_added_mode_oracle_capacity",
            "status": "completed_pending_audit" if integrity_pass else "failed_integrity_check",
            "quick": quick,
            "duration_seconds": time.perf_counter() - started,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "evidence_boundary": {
            "training_performed": False,
            "optimizer_updates": 0,
            "future_simulator_truth_accessed": True,
            "exact_simulator_registration_parameters_accessed": True,
            "per_episode_hindsight_selection": True,
            "deployable_controller": False,
            "prior_trajectory_files_reused": False,
            "s4d3_accessed": False,
            "real_slm_actions": False,
            "interpretation": (
                "The frozen controller owns the first ten requested residual coordinates; "
                "oracle residual requests are nonzero only in added modes."
            ),
        },
        "inputs": {
            "experiment_config": _relative(experiment_path),
            "experiment_config_sha256": _file_sha256(experiment_path),
            "upstream_modal_summary_sha256": preflight[
                "upstream_modal_summary_sha256"
            ],
            "upstream_modal_audit_sha256": preflight["upstream_modal_audit_sha256"],
        },
        "design": {
            "paired_episode_seeds": True,
            "anchor_modes": ANCHOR_MODES,
            "representations": [asdict(item) for item in representations],
            "preview_horizons_frames": horizons,
            "profiles": [profile.identifier for profile in profiles],
            "physical_conditions": settings["physical_conditions"],
            "episodes_per_physical_condition": int(settings["batch_size"]),
            "steps": int(settings["steps"]),
            "progress_rollouts": total_rollouts,
        },
        "basis_diagnostics": preflight["basis_checks"],
        "mapping_diagnostics": mapping_diagnostics,
        "integrity": {
            "truth_alignment": {
                "max_absolute_modal_difference": truth_alignment_max,
                "tolerance": float(experiment["oracle"]["truth_alignment_tolerance"]),
                "status": "PASS" if truth_pass else "FAIL",
            },
            "requested_anchor_residual": {
                "max_absolute_rad": requested_anchor_max,
                "tolerance_rad": float(
                    experiment["action_budget"]["requested_anchor_residual_tolerance_rad"]
                ),
                "status": "PASS" if anchor_request_pass else "FAIL",
            },
        },
        "high_order_capacity": summaries,
        "interpretation": {
            "status": status,
            "passing_representation_ids": passing if not quick and integrity_pass else [],
            "capacity_demonstrated": bool(passing) and not quick and integrity_pass,
            "new_rl_training_authorized": False,
            "algorithm_change_authorized": False,
            "s4d3_authorized": False,
            "real_hardware_authorized": False,
            "next_rule": (
                "Audit first. A PASS only authorizes designing an added-mode-only RL "
                "candidate; it does not authorize training or sealed validation."
            ),
        },
        "record_counts": {
            "scenario_records": len(scenario_records),
            "episode_records": len(episode_records),
        },
        "runtime": _runtime_record(),
        "git": _git_record(),
        "next_action": (
            "Stop and ask the assistant to audit the high-order capacity result. "
            "Do not train, change algorithms, open S4-D3, or operate real SLM hardware."
        ),
    }
    _write_csv(output_directory / "scenario_records.csv", scenario_records)
    _write_csv(output_directory / "episode_records.csv", episode_records)
    _write_json(output_directory / "summary.json", json_safe(summary))
    if not integrity_pass:
        raise RuntimeError("high-order-capacity integrity check failed; output preserved")
    return summary


def preflight_s4_high_order_capacity(
    experiment_path: Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    """锁定上游审计、动作子空间、公平门槛和安全边界。"""
    metadata = experiment["metadata"]
    if str(metadata["stage"]) != "S4-D2-R2-HIGH-ORDER-CAPACITY":
        raise ValueError("high-order-capacity stage metadata is invalid")
    forbidden = (
        "allow_training",
        "allow_optimizer_updates",
        "allow_checkpoint_updates",
        "allow_prior_trajectory_reuse",
        "allow_s4d3_access",
        "allow_real_hardware_actions",
        "allow_algorithm_change",
        "allow_reward_change",
        "allow_gate_relaxation",
    )
    if any(bool(metadata.get(field, True)) for field in forbidden):
        raise RuntimeError("high-order-capacity mutation and scope flags must be false")
    if not bool(metadata.get("allow_simulator_truth_access")):
        raise RuntimeError("high-order-capacity requires explicit simulator truth access")
    if not bool(metadata.get("allow_exact_registration_parameter_access")):
        raise RuntimeError("high-order-capacity requires exact registration access")

    upstream = _verify_upstream_modal_ablation(experiment["upstream_modal_ablation"])
    for field in ("environment_config", "hardware_profile_source"):
        path = _project_path(experiment[field])
        if _file_sha256(path) != str(experiment[f"{field}_sha256"]):
            raise RuntimeError(f"high-order-capacity input hash mismatch: {field}")
    base_config, _ = load_s1_config(_project_path(experiment["environment_config"]))
    if base_config.num_modes != ANCHOR_MODES:
        raise RuntimeError("frozen baseline must remain ten-dimensional")
    if int(experiment["frozen_controller"]["active_anchor_modes"]) != ANCHOR_MODES:
        raise RuntimeError("high-order-capacity anchor count changed")

    action = experiment["action_budget"]
    if (
        float(action["residual_component_limit_rad"]) != 0.05
        or float(action["final_component_step_limit_rad"]) != 0.15
        or int(action["total_budget_anchor_modes"]) != ANCHOR_MODES
        or not bool(action["preserve_total_phase_rms_budget"])
        or not bool(action["preserve_environment_modal_limit"])
    ):
        raise RuntimeError("high-order-capacity changed the audited action budget")
    gate = experiment["gate"]
    if float(gate["proposed_min_relative_power_gain"]) != 0.02:
        raise RuntimeError("high-order-capacity changed the two-percent gate")
    if not bool(gate["require_all_profiles_pass"]):
        raise RuntimeError("formal high-order capacity must require every profile")

    representations = [
        ActionRepresentation.from_mapping(item) for item in settings["representations"]
    ]
    expected = {("zernike_21_added_11", 21), ("zernike_36_added_26", 36)}
    actual = {(item.identifier, item.num_modes) for item in representations}
    if actual != expected:
        raise RuntimeError("high-order-capacity must compare the audited 21/36D bases")
    for raw in settings["representations"]:
        if int(raw["added_modes"]) != int(raw["num_modes"]) - ANCHOR_MODES:
            raise RuntimeError("added-mode count is inconsistent")

    horizons = list(map(int, settings["preview_horizons_frames"]))
    if not horizons or horizons != sorted(set(horizons)) or horizons[0] < 0:
        raise ValueError("preview horizons must be sorted unique non-negative values")
    _validate_seeds(experiment, settings)
    profiles = _profiles(experiment, settings["profile_ids"])
    basis_checks: dict[str, dict[str, Any]] = {}
    mapping_checks: dict[str, dict[str, dict[str, Any]]] = {}
    for representation in representations:
        basis, pupil, diagnostics = build_action_basis(
            base_config,
            representation,
            torch.device("cpu"),
        )
        basis_checks[representation.identifier] = diagnostics
        mapping_checks[representation.identifier] = {}
        for profile in profiles:
            _, mapping = representation_registration_inverse(
                basis=basis,
                pupil=pupil,
                profile=profile,
                rcond=float(experiment["registration_inverse"]["pseudoinverse_rcond"]),
            )
            mapping_checks[representation.identifier][profile.identifier] = mapping
            required_rank = math.ceil(
                representation.num_modes
                * float(experiment["registration_inverse"]["minimum_rank_fraction"])
            )
            if int(mapping["effective_rank"]) < required_rank:
                raise RuntimeError("high-order registration mapping lost too much rank")

    if not all(_project_path(path).is_file() for path in experiment["tracked_source_files"]):
        raise FileNotFoundError("one or more high-order-capacity sources are missing")
    output_directory = _project_path(settings["output_directory"])
    if output_directory.exists():
        raise FileExistsError(f"high-order-capacity output already exists: {output_directory}")
    scenario_count = len(settings["profile_ids"]) * len(settings["physical_conditions"])
    rollout_count = len(representations) * scenario_count * (1 + len(horizons))
    scenario_rows = len(representations) * scenario_count * (len(horizons) + 1)
    episode_rows = scenario_rows * int(settings["batch_size"])
    return {
        "status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_SIMULATION",
        "quick": quick,
        "experiment_config": _relative(experiment_path),
        "experiment_config_sha256": _file_sha256(experiment_path),
        "output_directory": _relative(output_directory),
        "upstream_modal_status": "ANALYZED",
        "upstream_modal_summary_sha256": upstream["summary_sha256"],
        "upstream_modal_audit_sha256": upstream["audit_sha256"],
        "representations": [asdict(item) for item in representations],
        "basis_checks": basis_checks,
        "mapping_checks": mapping_checks,
        "preview_horizons_frames": horizons,
        "scenario_count_per_representation": scenario_count,
        "rollout_count": rollout_count,
        "expected_scenario_record_rows": scenario_rows,
        "expected_episode_record_rows": episode_rows,
        "episodes_per_scenario": int(settings["batch_size"]),
        "steps": int(settings["steps"]),
        "cuda_required": True,
        "training_allowed": False,
        "optimizer_updates": 0,
        "future_simulator_truth_accessed": True,
        "requested_rl_anchor_coordinates_forced_zero": True,
        "prior_trajectory_files_reused": False,
        "s4d3_accessed": False,
        "real_slm_actions": False,
        "formal_command": (
            ".\\.venv\\Scripts\\python.exe scripts\\run_s4_high_order_capacity.py "
            "--config configs\\experiments\\s4_high_order_capacity_v1.yaml"
        ),
    }


def _rollout_added_modes_preview(
    *,
    experiment: dict[str, Any],
    config: S1EnvConfig,
    condition: RobustnessCondition,
    profile: HardwareProfile,
    steps: int,
    preview_horizon_frames: int,
    future_disturbance: torch.Tensor,
    registration_mapping: torch.Tensor,
    basis: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], float, float]:
    environment = AdaptiveOpticsEnv(
        config,
        device,
        profile.effects_config(),
        basis_override=basis,
    )
    observation, _ = environment.reset(seed=condition.base_seed)
    raw_observation = observation
    generator = torch.Generator(device=device).manual_seed(
        condition.base_seed + 40_000_000
    )
    noisy = _noisy_observation(
        observation,
        config.num_modes,
        profile.observation_noise_std_rad,
        generator,
    )
    params = experiment["frozen_controller"]["parameters"]
    controller = AddedModesOracleController(
        num_modes=config.num_modes,
        anchor_modes=ANCHOR_MODES,
        modal_limit_rad=config.modal_limit_rad,
        history_frames=4,
        residual_action_limit_rad=float(
            experiment["action_budget"]["residual_component_limit_rad"]
        ),
        final_action_step_limit_rad=float(
            experiment["action_budget"]["final_component_step_limit_rad"]
        ),
        gain=float(params["gain"]),
        leak=float(params["leak"]),
        tracking_gain=float(params["tracking_gain"]),
    )
    controller.reset(noisy)
    science = _empty_step_metrics()
    action_values: dict[str, list[torch.Tensor]] = {
        key: []
        for key in tuple(ACTION_METRICS) + HIGH_ORDER_ACTION_METRICS
        if key != "selected_preview_horizon_frames"
    }
    alignment_max = 0.0
    requested_anchor_max = 0.0
    residual_limit = float(experiment["action_budget"]["residual_component_limit_rad"])
    for step in range(steps):
        actual = environment.oracle_disturbance_modal()
        alignment_max = max(
            alignment_max,
            float((actual - future_disturbance[step]).abs().max()),
        )
        future_index = min(
            step + preview_horizon_frames,
            future_disturbance.shape[0] - 1,
        )
        desired_applied = -future_disturbance[future_index]
        target_request = desired_applied @ registration_mapping.transpose(0, 1)
        action, desired_residual, prior = controller.compose_added_modes_to_target(
            target_request
        )
        requested_anchor_max = max(
            requested_anchor_max,
            float(action.requested_residual_rad[:, :ANCHOR_MODES].abs().max()),
        )
        raw_observation, _, _, _, info = environment.step(action.final_delta_rad)
        _append_step_metrics(science, info)

        normalized_clipped = (desired_residual / residual_limit).clamp(-1, 1)
        unprojected = normalized_clipped * residual_limit
        nominal_final = action.baseline_delta_rad + action.requested_residual_rad
        nominal_target = prior + nominal_final
        action_values["requested_residual_abs_mean_rad"].append(
            action.requested_residual_rad.abs().mean(dim=-1).cpu()
        )
        action_values["realized_residual_abs_mean_rad"].append(
            action.realized_residual_rad.abs().mean(dim=-1).cpu()
        )
        action_values["normalized_saturation_fraction"].append(
            normalized_clipped.abs().ge(1 - 1e-7).float().mean(dim=-1).cpu()
        )
        action_values["target_request_error_abs_mean_rad"].append(
            (controller.requested_modal - target_request).abs().mean(dim=-1).cpu()
        )
        action_values["requested_residual_phase_rms_rad"].append(
            torch.linalg.vector_norm(action.requested_residual_rad, dim=-1).cpu()
        )
        action_values["realized_residual_phase_rms_rad"].append(
            torch.linalg.vector_norm(action.realized_residual_rad, dim=-1).cpu()
        )
        action_values["residual_projection_fraction"].append(
            action.requested_residual_rad.sub(unprojected)
            .abs()
            .amax(dim=-1)
            .gt(1e-7)
            .float()
            .cpu()
        )
        action_values["final_projection_fraction"].append(
            action.final_delta_rad.sub(nominal_final)
            .abs()
            .amax(dim=-1)
            .gt(1e-7)
            .float()
            .cpu()
        )
        action_values["request_projection_fraction"].append(
            controller.requested_modal.sub(nominal_target)
            .abs()
            .amax(dim=-1)
            .gt(1e-7)
            .float()
            .cpu()
        )
        action_values["requested_anchor_residual_abs_mean_rad"].append(
            action.requested_residual_rad[:, :ANCHOR_MODES].abs().mean(dim=-1).cpu()
        )
        action_values["realized_anchor_residual_abs_mean_rad"].append(
            action.realized_residual_rad[:, :ANCHOR_MODES].abs().mean(dim=-1).cpu()
        )
        action_values["requested_added_residual_phase_rms_rad"].append(
            torch.linalg.vector_norm(
                action.requested_residual_rad[:, ANCHOR_MODES:], dim=-1
            ).cpu()
        )
        action_values["realized_added_residual_phase_rms_rad"].append(
            torch.linalg.vector_norm(
                action.realized_residual_rad[:, ANCHOR_MODES:], dim=-1
            ).cpu()
        )
        if step + 1 < steps:
            noisy = _noisy_observation(
                raw_observation,
                config.num_modes,
                profile.observation_noise_std_rad,
                generator,
            )
            controller.advance_observation(noisy)

    result = _mean_step_metrics(science)
    result.update(
        {
            key: torch.stack(values, dim=1).mean(dim=1)
            for key, values in action_values.items()
        }
    )
    result["selected_preview_horizon_frames"] = torch.full(
        (config.batch_size,),
        float(preview_horizon_frames),
        dtype=result["power_in_bucket"].dtype,
    )
    return result, alignment_max, requested_anchor_max


def _high_order_capacity_summary(
    identifier: str,
    candidate: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
    gate: dict[str, Any],
) -> dict[str, Any]:
    result = _paired_summary(identifier, candidate, baseline, gate)
    result["high_order_action_diagnostics"] = {
        key: _distribution(candidate[key]) for key in HIGH_ORDER_ACTION_METRICS
    }
    return result


def _scenario_record(
    *,
    representation: ActionRepresentation,
    variant: str,
    profile: HardwareProfile,
    condition: RobustnessCondition,
    candidate: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "representation_id": representation.identifier,
        "num_modes": representation.num_modes,
        "added_modes": representation.num_modes - ANCHOR_MODES,
        "variant": variant,
        "profile": profile.identifier,
        "physical_condition": condition.identifier,
        "episodes": int(candidate["power_in_bucket"].numel()),
    }
    for metric in SCIENCE_METRICS:
        row[f"candidate_{metric}_mean"] = float(candidate[metric].mean())
        row[f"baseline_{metric}_mean"] = float(baseline[metric].mean())
        row[f"delta_{metric}_mean"] = float(
            (candidate[metric] - baseline[metric]).mean()
        )
    for metric in tuple(ACTION_METRICS) + HIGH_ORDER_ACTION_METRICS:
        row[f"action_{metric}_mean"] = float(candidate[metric].mean())
    return row


def _episode_rows(
    *,
    representation: ActionRepresentation,
    variant: str,
    profile: HardwareProfile,
    condition: RobustnessCondition,
    candidate: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    rows = []
    count = int(candidate["power_in_bucket"].numel())
    for index in range(count):
        row: dict[str, Any] = {
            "representation_id": representation.identifier,
            "num_modes": representation.num_modes,
            "added_modes": representation.num_modes - ANCHOR_MODES,
            "variant": variant,
            "profile": profile.identifier,
            "physical_condition": condition.identifier,
            "episode_index": index,
            "episode_seed": int(condition.base_seed) + index,
        }
        for metric in SCIENCE_METRICS:
            candidate_value = float(candidate[metric][index])
            baseline_value = float(baseline[metric][index])
            row[f"candidate_{metric}"] = candidate_value
            row[f"baseline_{metric}"] = baseline_value
            row[f"delta_{metric}"] = candidate_value - baseline_value
        for metric in tuple(ACTION_METRICS) + HIGH_ORDER_ACTION_METRICS:
            row[f"action_{metric}"] = float(candidate[metric][index])
        rows.append(row)
    return rows


def _verify_upstream_modal_ablation(upstream: dict[str, Any]) -> dict[str, str]:
    fields = (
        "summary",
        "scenario_records",
        "episode_records",
        "preflight",
        "effective_config",
        "source_manifest",
        "experiment_config",
        "audit_record",
    )
    paths = {}
    for field in fields:
        path = _project_path(upstream[field])
        if _file_sha256(path) != str(upstream[f"{field}_sha256"]):
            raise RuntimeError(f"modal-ablation evidence hash mismatch: {field}")
        paths[field] = path
    with paths["summary"].open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if (
        bool(summary["experiment"]["quick"])
        or summary["experiment"]["status"] != "completed_pending_audit"
        or summary["zero_residual_equivalence"]["status"] != "PASS"
        or summary["upstream_all_modes_quarter_reproduction"]["status"] != "PASS"
        or int(summary["record_counts"]["scenario_records"]) != 360
        or int(summary["record_counts"]["episode_records"]) != 5760
        or bool(summary["interpretation"]["training_authorized"])
        or bool(summary["interpretation"]["s4d3_authorized"])
    ):
        raise RuntimeError("formal modal-ablation state changed")
    audit = paths["audit_record"].read_text(encoding="utf-8")
    if (
        "Verification Status: `ANALYZED`" not in audit
        or "新增高阶策略动作是主要损失来源：`SUPPORTED`" not in audit
    ):
        raise RuntimeError("modal-ablation audit is not finalized")
    with paths["source_manifest"].open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    for relative, digest in manifest.items():
        path = _project_path(relative)
        if not path.is_file() or _file_sha256(path) != str(digest):
            raise RuntimeError(f"modal-ablation source changed: {relative}")
    return {
        "summary_sha256": _file_sha256(paths["summary"]),
        "audit_sha256": _file_sha256(paths["audit_record"]),
    }


def _validate_seeds(experiment: dict[str, Any], settings: dict[str, Any]) -> None:
    ranges = [
        (int(item["start_inclusive"]), int(item["end_exclusive"]))
        for item in experiment["protected_seed_ranges"]
    ]
    batch_size = int(settings["batch_size"])
    active_ranges = []
    for condition in settings["physical_conditions"]:
        start = int(condition["base_seed"])
        stop = start + batch_size
        active_ranges.append((start, stop))
        if any(start < old_stop and stop > old_start for old_start, old_stop in ranges):
            raise RuntimeError("high-order-capacity seeds overlap a protected range")
    for index, left in enumerate(active_ranges):
        for right in active_ranges[index + 1 :]:
            if left[0] < right[1] and left[1] > right[0]:
                raise RuntimeError("high-order physical conditions reuse episode seeds")


def _effective_settings(experiment: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    evaluation = experiment["evaluation"]
    settings = {
        "output_directory": experiment["outputs"]["directory"],
        "steps": int(evaluation["steps"]),
        "batch_size": int(evaluation["episodes_per_physical_condition"]),
        "preview_horizons_frames": list(experiment["oracle"]["preview_horizons_frames"]),
        "profile_ids": list(evaluation["profile_ids"]),
        "physical_conditions": list(evaluation["physical_conditions"]),
        "representations": list(experiment["representations"]),
    }
    if quick:
        quick_settings = experiment["quick"]
        ids = set(map(str, quick_settings["representation_ids"]))
        settings.update(
            {
                "output_directory": experiment["outputs"]["quick_directory"],
                "steps": int(quick_settings["steps"]),
                "batch_size": int(
                    quick_settings["episodes_per_physical_condition"]
                ),
                "preview_horizons_frames": list(
                    quick_settings["preview_horizons_frames"]
                ),
                "profile_ids": list(quick_settings["profile_ids"]),
                "physical_conditions": list(quick_settings["physical_conditions"]),
                "representations": [
                    item for item in experiment["representations"] if item["id"] in ids
                ],
            }
        )
    return settings


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty high-order-capacity CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(json_safe(rows))
