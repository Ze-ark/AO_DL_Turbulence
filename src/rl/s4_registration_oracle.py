"""S4-D2-R1配准感知理想控制诊断；不训练强化学习。"""

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
    ACTION_METRICS,
    SCIENCE_METRICS,
    _collect_metrics,
    _concatenate_metrics,
    _disturbance_modal,
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
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.hardware_effects import (
    HardwareProfile,
    apply_registration_error,
)
from src.simulation.modes import make_low_order_zernike_basis
from src.simulation.robust_control import RobustnessCondition
from src.training_progress import advance_to, counted_progress, update_progress


BLIND = "registration_blind"
AWARE = "registration_aware_spatial_lstsq"


def run_s4_registration_oracle(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """比较盲配准与已知配准逆映射理想控制器。"""
    experiment_path = _project_path(config_path)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight = preflight_s4_registration_oracle(
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
            "registration-oracle output already exists; preserve and audit it: "
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
        episode_length=max(base_config.episode_length, int(settings["steps"]) + max(horizons)),
    )
    profiles = _profiles(experiment, settings["profile_ids"])
    conditions = [
        RobustnessCondition.from_mapping(item)
        for item in settings["physical_conditions"]
    ]
    scenario_count = len(profiles) * len(conditions)
    total_rollouts = scenario_count * (1 + 2 * len(horizons))
    progress = counted_progress(
        total=total_rollouts,
        description="S4-D2-R1配准感知理想控制",
        unit="轨迹组",
    )
    progress_path = output_directory / "progress.jsonl"
    completed = 0
    started = time.perf_counter()

    variant_candidates = {
        BLIND: defaultdict(list),
        AWARE: defaultdict(list),
    }
    variant_baselines = {
        BLIND: defaultdict(list),
        AWARE: defaultdict(list),
    }
    profile_candidates = {
        BLIND: defaultdict(lambda: defaultdict(list)),
        AWARE: defaultdict(lambda: defaultdict(list)),
    }
    profile_baselines = {
        BLIND: defaultdict(lambda: defaultdict(list)),
        AWARE: defaultdict(lambda: defaultdict(list)),
    }
    modal_ceiling: dict[str, list[torch.Tensor]] = defaultdict(list)
    modal_ceiling_baseline: dict[str, list[torch.Tensor]] = defaultdict(list)
    scenario_records: list[dict[str, Any]] = []
    truth_alignment_max_abs = 0.0

    mappings: dict[str, torch.Tensor] = {}
    mapping_diagnostics: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        mapping, diagnostics = registration_inverse_modal_map(
            config=base_config,
            profile=profile,
            rcond=float(experiment["registration_inverse"]["pseudoinverse_rcond"]),
            device=device,
        )
        mappings[profile.identifier] = mapping
        mapping_diagnostics[profile.identifier] = diagnostics

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
                experiment=experiment,
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

            envelopes: dict[str, dict[str, torch.Tensor]] = {}
            for variant in (BLIND, AWARE):
                horizon_results: dict[int, dict[str, torch.Tensor]] = {}
                for horizon in horizons:
                    if variant == BLIND or not bool(
                        mapping_diagnostics[profile.identifier]["registration_present"]
                    ):
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
                    else:
                        candidate, alignment = _rollout_registration_aware_preview(
                            experiment=experiment,
                            config=config,
                            condition=condition,
                            profile=profile,
                            steps=int(settings["steps"]),
                            preview_horizon_frames=horizon,
                            future_disturbance_modal=future_truth,
                            registration_mapping=mappings[profile.identifier],
                            device=device,
                        )
                    truth_alignment_max_abs = max(truth_alignment_max_abs, alignment)
                    horizon_results[horizon] = candidate
                    scenario_records.append(
                        _scenario_record(
                            controller=f"{variant}_preview_{horizon}",
                            variant=variant,
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
                        controller=f"{variant}_preview_{horizon}",
                        power=float(candidate["power_in_bucket"].mean()),
                        started=started,
                    )
                    advance_to(progress, completed)
                    update_progress(
                        progress,
                        device=device,
                        metrics={
                            "感知配准": float(variant == AWARE),
                            "预见帧": float(horizon),
                            "功率差": float(
                                (
                                    candidate["power_in_bucket"]
                                    - baseline["power_in_bucket"]
                                ).mean()
                            ),
                        },
                    )

                envelope = select_hindsight_envelope(horizon_results)
                envelopes[variant] = envelope
                _collect_metrics(variant_candidates[variant], envelope)
                _collect_metrics(variant_baselines[variant], baseline)
                _collect_metrics(
                    profile_candidates[variant][profile.identifier], envelope
                )
                _collect_metrics(
                    profile_baselines[variant][profile.identifier], baseline
                )
                scenario_records.append(
                    _scenario_record(
                        controller=f"{variant}_hindsight_envelope",
                        variant=variant,
                        profile=profile.identifier,
                        condition=condition.identifier,
                        candidate=envelope,
                        baseline=baseline,
                    )
                )
    progress.close()

    variant_summaries: dict[str, dict[str, Any]] = {}
    for variant in (BLIND, AWARE):
        overall = _paired_summary(
            variant,
            _concatenate_metrics(variant_candidates[variant]),
            _concatenate_metrics(variant_baselines[variant]),
            settings["gate"],
        )
        profile_summaries = [
            _paired_summary(
                profile.identifier,
                _concatenate_metrics(
                    profile_candidates[variant][profile.identifier]
                ),
                _concatenate_metrics(
                    profile_baselines[variant][profile.identifier]
                ),
                settings["gate"],
            )
            for profile in profiles
        ]
        all_profiles_pass = all(
            item["capacity_gate"] == "PASS" for item in profile_summaries
        )
        variant_summaries[variant] = {
            "overall": overall,
            "profiles": profile_summaries,
            "all_profiles_pass": all_profiles_pass,
            "capacity_gate": (
                "PASS"
                if overall["capacity_gate"] == "PASS" and all_profiles_pass
                else "FAIL"
            ),
        }

    blind_all = _concatenate_metrics(variant_candidates[BLIND])
    aware_all = _concatenate_metrics(variant_candidates[AWARE])
    comparison_overall = _comparison_summary(
        "aware_minus_blind",
        aware_all,
        blind_all,
        settings["comparison_gate"],
    )
    comparison_profiles = [
        _comparison_summary(
            profile.identifier,
            _concatenate_metrics(profile_candidates[AWARE][profile.identifier]),
            _concatenate_metrics(profile_candidates[BLIND][profile.identifier]),
            settings["comparison_gate"],
        )
        for profile in profiles
    ]
    comparison_by_profile = {
        item["comparison"]: item for item in comparison_profiles
    }
    aware_profiles = {
        item["controller"]: item
        for item in variant_summaries[AWARE]["profiles"]
    }
    declared_target_profiles = list(
        experiment["registration_inverse"]["target_profile_ids"]
    )
    evaluated_target_profiles = _available_profile_ids(
        declared_target_profiles,
        comparison_by_profile,
    )
    recovery_profiles = []
    for identifier in evaluated_target_profiles:
        comparison = comparison_by_profile[identifier]
        capacity = aware_profiles[identifier]
        passed = (
            comparison["improvement_gate"] == "PASS"
            and capacity["capacity_gate"] == "PASS"
        )
        recovery_profiles.append(
            {
                "profile": identifier,
                "aware_capacity_gate": capacity["capacity_gate"],
                "aware_vs_blind_improvement_gate": comparison["improvement_gate"],
                "recovery_gate": "PASS" if passed else "FAIL",
            }
        )

    declared_invariant_profiles = list(
        experiment["registration_inverse"]["invariant_profile_ids"]
    )
    evaluated_invariant_profiles = _available_profile_ids(
        declared_invariant_profiles,
        comparison_by_profile,
    )
    invariance = [
        _invariance_summary(
            identifier,
            _concatenate_metrics(profile_candidates[AWARE][identifier]),
            _concatenate_metrics(profile_candidates[BLIND][identifier]),
            tolerance=float(experiment["registration_inverse"]["identity_tolerance"]),
        )
        for identifier in evaluated_invariant_profiles
    ]
    recovery_gate = (
        "PASS"
        if recovery_profiles
        and all(item["recovery_gate"] == "PASS" for item in recovery_profiles)
        and all(item["status"] == "PASS" for item in invariance)
        else "FAIL"
    )
    status = (
        "QUICK_SMOKE_ONLY"
        if quick
        else (
            "REGISTRATION_RECOVERABLE"
            if recovery_gate == "PASS"
            else "REGISTRATION_NOT_RECOVERED"
        )
    )
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run-diagnostic",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "s4d2_r1_registration_oracle_v1",
        },
        "experiment": {
            "id": "AO-S4-D2-R1-REGISTRATION-ORACLE",
            "type": "software_only_cuda_registration_aware_oracle_diagnostic",
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
            "exact_simulator_registration_parameters_accessed": True,
            "per_episode_hindsight_selection": True,
            "deployable_controller": False,
            "old_trajectories_used": False,
            "sealed_s4d3_accessed": False,
            "real_slm_actions": False,
            "interpretation": (
                "Uses exact simulated registration and future truth to isolate action "
                "parameterization; it is not an estimator, policy, or hardware controller."
            ),
        },
        "inputs": {
            "experiment_config": _relative(experiment_path),
            "experiment_config_sha256": _file_sha256(experiment_path),
            "upstream_action_range_summary_sha256": preflight[
                "upstream_action_range_summary_sha256"
            ],
            "upstream_action_range_audit_sha256": preflight[
                "upstream_action_range_audit_sha256"
            ],
        },
        "design": {
            "paired_episode_seeds": True,
            "controller_variants": [BLIND, AWARE],
            "residual_action_limit_rad": float(
                experiment["action"]["residual_action_limit_rad"]
            ),
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
        "mapping_diagnostics": mapping_diagnostics,
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
        "variant_capacity": variant_summaries,
        "aware_vs_blind": {
            "overall": comparison_overall,
            "profiles": comparison_profiles,
        },
        "registration_recovery": {
            "declared_target_profile_ids": declared_target_profiles,
            "evaluated_target_profile_ids": evaluated_target_profiles,
            "declared_invariant_profile_ids": declared_invariant_profiles,
            "evaluated_invariant_profile_ids": evaluated_invariant_profiles,
            "target_profiles": recovery_profiles,
            "nonregistration_invariance": invariance,
            "recovery_gate": recovery_gate,
        },
        "absolute_modal_ceiling_context": _science_context_summary(
            "unconstrained_instantaneous_modal_ceiling",
            _concatenate_metrics(modal_ceiling),
            _concatenate_metrics(modal_ceiling_baseline),
        ),
        "interpretation": {
            "status": status,
            "registration_recoverable": recovery_gate == "PASS" and not quick,
            "all_profile_capacity_demonstrated": (
                variant_summaries[AWARE]["capacity_gate"] == "PASS" and not quick
            ),
            "r2_authorized": False,
            "s4d3_authorized": False,
            "next_rule": (
                "Audit first. PASS only identifies a registration-aware action "
                "parameterization worth designing; it does not authorize RL or S4-D3."
            ),
        },
        "runtime": _runtime_record(),
        "git": _git_record(),
        "next_action": (
            "Stop and ask the assistant to audit the registration oracle. "
            "Do not train R2 or open S4-D3."
        ),
    }
    _write_json(output_directory / "summary.json", summary)
    _write_scenario_csv(output_directory / "scenario_records.csv", scenario_records)
    return summary


def preflight_s4_registration_oracle(
    experiment_path: Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    """锁定动作范围FAIL、配准对照、源码和种子后再占用CUDA。"""
    metadata = experiment["metadata"]
    if str(metadata["stage"]) != "S4-D2-R1-REG-ORACLE":
        raise ValueError("registration-oracle stage must be S4-D2-R1-REG-ORACLE")
    forbidden = (
        "allow_training",
        "allow_optimizer_updates",
        "allow_checkpoint_updates",
        "allow_old_trajectory_access",
        "allow_s4d3_access",
        "allow_real_hardware_actions",
    )
    if any(bool(metadata.get(field, True)) for field in forbidden):
        raise RuntimeError("registration-oracle mutation flags must stay false")
    for field in (
        "allow_simulator_truth_access",
        "allow_exact_registration_parameter_access",
    ):
        if not bool(metadata.get(field, False)):
            raise RuntimeError(f"registration-oracle must explicitly declare {field}")

    # 先检查当前配置本身，再检查已封存上游源码。阶段推进后公共源码发生合法
    # 变化时，旧入口仍会拒绝重跑；但配置错误应优先给出更直接的原因。
    range_config = _load_yaml(
        _project_path(experiment["upstream_action_range"]["experiment_config"])
    )
    for field in ("frozen_controller", "policy_observation"):
        if experiment[field] != range_config[field]:
            raise RuntimeError(f"registration-oracle changed upstream field: {field}")
    for field in ("num_modes", "final_action_step_limit_rad", "preserve_environment_modal_limit"):
        if experiment["action"][field] != range_config["action"][field]:
            raise RuntimeError(f"registration-oracle changed action field: {field}")
    if float(experiment["action"]["residual_action_limit_rad"]) != max(
        map(float, range_config["action"]["residual_action_limits_rad"])
    ):
        raise RuntimeError("registration-oracle must keep the scanned 0.05 rad maximum")
    if list(experiment["evaluation"]["profile_ids"]) != list(
        range_config["evaluation"]["profile_ids"]
    ):
        raise RuntimeError("registration-oracle hardware profiles changed")
    for field in ("steps", "episodes_per_physical_condition"):
        if experiment["evaluation"][field] != range_config["evaluation"][field]:
            raise RuntimeError(f"registration-oracle changed evaluation field: {field}")
    _verify_matching_physics(
        range_config["evaluation"]["physical_conditions"],
        experiment["evaluation"]["physical_conditions"],
    )
    if experiment["gate"] != range_config["gate"]:
        raise RuntimeError("registration-oracle changed the original capacity gate")
    if list(experiment["oracle"]["preview_horizons_frames"]) != list(
        range_config["oracle"]["preview_horizons_frames"]
    ):
        raise RuntimeError("registration-oracle changed formal preview horizons")
    if (
        str(experiment["oracle"]["selection_metric"]) != "power_in_bucket"
        or not bool(experiment["oracle"]["per_episode_hindsight_envelope"])
    ):
        raise RuntimeError("registration-oracle optimistic preview settings changed")

    environment_path = _project_path(experiment["environment_config"])
    hardware_path = _project_path(experiment["hardware_profile_source"])
    if _file_sha256(environment_path) != str(experiment["environment_config_sha256"]):
        raise RuntimeError("environment config hash mismatch")
    if _file_sha256(hardware_path) != str(experiment["hardware_profile_source_sha256"]):
        raise RuntimeError("hardware profile source hash mismatch")
    if experiment["environment_config_sha256"] != range_config[
        "environment_config_sha256"
    ] or experiment["hardware_profile_source_sha256"] != range_config[
        "hardware_profile_source_sha256"
    ]:
        raise RuntimeError("registration-oracle environment or hardware source changed")

    base_config, _ = load_s1_config(environment_path)
    controller = _make_residual_controller(experiment, base_config)
    if controller.state_size != 100 or base_config.num_modes != 10:
        raise RuntimeError("registration-oracle controller dimensions changed")
    profiles = _profiles(experiment, experiment["evaluation"]["profile_ids"])
    by_id = {profile.identifier: profile for profile in profiles}
    targets = list(experiment["registration_inverse"]["target_profile_ids"])
    invariant = list(experiment["registration_inverse"]["invariant_profile_ids"])
    if targets != ["registration_moderate", "registration_severe"]:
        raise RuntimeError("registration target profiles must stay predeclared")
    if invariant != ["nominal", "delay_3", "settling_050"]:
        raise RuntimeError("nonregistration invariant profiles changed")
    if any(
        by_id[item].shift_x_pixels == 0
        and by_id[item].shift_y_pixels == 0
        and by_id[item].rotation_deg == 0
        for item in targets
    ):
        raise RuntimeError("a registration target profile has no registration error")
    if any(
        by_id[item].shift_x_pixels != 0
        or by_id[item].shift_y_pixels != 0
        or by_id[item].rotation_deg != 0
        for item in invariant
    ):
        raise RuntimeError("an invariant profile unexpectedly has registration error")

    mapping_checks = {}
    for profile in profiles:
        _, diagnostics = registration_inverse_modal_map(
            config=base_config,
            profile=profile,
            rcond=float(experiment["registration_inverse"]["pseudoinverse_rcond"]),
            device=torch.device("cpu"),
        )
        mapping_checks[profile.identifier] = diagnostics
        if float(diagnostics["condition_number"]) > float(
            experiment["registration_inverse"]["max_condition_number"]
        ):
            raise RuntimeError(f"registration inverse is ill-conditioned: {profile.identifier}")

    horizons = list(map(int, settings["preview_horizons_frames"]))
    if not horizons or horizons != sorted(set(horizons)) or horizons[0] < 0:
        raise ValueError("preview horizons must be sorted unique non-negative integers")
    if quick and any(
        horizon not in experiment["oracle"]["preview_horizons_frames"]
        for horizon in horizons
    ):
        raise RuntimeError("quick preview horizons must be a subset of formal horizons")
    _validate_registration_seeds(experiment, settings)
    upstream = _verify_action_range_failure(experiment["upstream_action_range"])

    output_directory = _project_path(settings["output_directory"])
    if output_directory.exists():
        raise FileExistsError(f"registration-oracle output already exists: {output_directory}")
    if not all(_project_path(path).is_file() for path in experiment["tracked_source_files"]):
        raise FileNotFoundError("one or more registration-oracle sources are missing")
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
        "residual_action_limit_rad": float(experiment["action"]["residual_action_limit_rad"]),
        "preview_horizons_frames": horizons,
        "scenario_count": scenario_count,
        "progress_rollouts": scenario_count * (1 + 2 * len(horizons)),
        "episodes_per_scenario": int(settings["batch_size"]),
        "steps": int(settings["steps"]),
        "mapping_checks": mapping_checks,
        "upstream_action_range_gate": "FAIL",
        "upstream_action_range_summary_sha256": upstream["summary_sha256"],
        "upstream_action_range_audit_sha256": upstream["audit_sha256"],
        "seed_isolation_verified": True,
        "cuda_required": True,
        "training_allowed": False,
        "optimizer_updates": 0,
        "future_simulator_truth_accessed": True,
        "exact_registration_parameters_accessed": True,
        "deployable_controller": False,
        "sealed_s4d3_accessed": False,
        "real_slm_actions": False,
        "formal_command": (
            ".\\.venv\\Scripts\\python.exe scripts\\run_s4_registration_oracle.py "
            "--config configs\\experiments\\s4_registration_oracle_v1.yaml"
        ),
    }


def registration_inverse_modal_map(
    *,
    config: S1EnvConfig,
    profile: HardwareProfile,
    rcond: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """构造使配准后瞳面相位最接近目标低阶相位的线性逆映射。"""
    if not 0 < rcond < 1:
        raise ValueError("pseudoinverse rcond must be in (0, 1)")
    basis, pupil = make_low_order_zernike_basis(
        config.grid_size,
        config.pupil_radius_fraction,
        config.num_modes,
        device,
    )
    registration_present = (
        profile.shift_x_pixels != 0
        or profile.shift_y_pixels != 0
        or profile.rotation_deg != 0
    )
    if not registration_present:
        mapping = torch.eye(config.num_modes, device=device, dtype=basis.dtype)
        mapping = mapping / profile.phase_scale
        return mapping, {
            "registration_present": False,
            "condition_number": 1.0,
            "relative_pupil_reconstruction_rmse": 0.0,
            "mapping_spectral_norm": float(1 / profile.phase_scale),
        }

    response = apply_registration_error(
        basis * profile.phase_scale,
        shift_x_pixels=profile.shift_x_pixels,
        shift_y_pixels=profile.shift_y_pixels,
        rotation_deg=profile.rotation_deg,
    )
    response_matrix = response[:, pupil].transpose(0, 1)
    target_matrix = basis[:, pupil].transpose(0, 1)
    singular_values = torch.linalg.svdvals(response_matrix)
    condition_number = singular_values.max() / singular_values.min()
    mapping = torch.linalg.pinv(response_matrix, rcond=rcond) @ target_matrix
    fitted = response_matrix @ mapping
    reconstruction_rmse = torch.sqrt((fitted - target_matrix).square().mean())
    target_rms = torch.sqrt(target_matrix.square().mean()).clamp_min(1e-12)
    return mapping, {
        "registration_present": True,
        "condition_number": float(condition_number),
        "relative_pupil_reconstruction_rmse": float(reconstruction_rmse / target_rms),
        "mapping_spectral_norm": float(torch.linalg.svdvals(mapping).max()),
    }


def registration_aware_normalized_residual(
    *,
    state: torch.Tensor,
    future_disturbance_modal: torch.Tensor,
    registration_mapping: torch.Tensor,
    num_modes: int,
    residual_action_limit_rad: float,
    modal_limit_rad: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把期望的实际加载模态反变换为SLM请求模态，再投影残差动作。"""
    if residual_action_limit_rad <= 0:
        raise ValueError("residual action limit must be positive")
    if state.ndim != 2 or state.shape[1] < 2 * num_modes:
        raise ValueError("state does not contain requested-modal and baseline-delta fields")
    if future_disturbance_modal.shape != (state.shape[0], num_modes):
        raise ValueError("future disturbance modal shape does not match state")
    if registration_mapping.shape != (num_modes, num_modes):
        raise ValueError("registration mapping has the wrong shape")
    prior = state[:, -2 * num_modes : -num_modes]
    baseline_delta = state[:, -num_modes:]
    desired_applied = -future_disturbance_modal
    target_request = (desired_applied @ registration_mapping.transpose(0, 1)).clamp(
        -modal_limit_rad,
        modal_limit_rad,
    )
    requested_residual = target_request - prior - baseline_delta
    normalized = (requested_residual / residual_action_limit_rad).clamp(-1, 1)
    return normalized, target_request


@torch.no_grad()
def _rollout_registration_aware_preview(
    *,
    experiment: dict[str, Any],
    config: S1EnvConfig,
    condition: RobustnessCondition,
    profile: HardwareProfile,
    steps: int,
    preview_horizon_frames: int,
    future_disturbance_modal: torch.Tensor,
    registration_mapping: torch.Tensor,
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
        normalized, target = registration_aware_normalized_residual(
            state=state,
            future_disturbance_modal=future_disturbance_modal[future_index],
            registration_mapping=registration_mapping,
            num_modes=config.num_modes,
            residual_action_limit_rad=float(
                experiment["action"]["residual_action_limit_rad"]
            ),
            modal_limit_rad=config.modal_limit_rad,
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


def _comparison_summary(
    identifier: str,
    aware: dict[str, torch.Tensor],
    blind: dict[str, torch.Tensor],
    gate: dict[str, Any],
) -> dict[str, Any]:
    aware_summary = {key: _distribution(aware[key]) for key in SCIENCE_METRICS}
    blind_summary = {key: _distribution(blind[key]) for key in SCIENCE_METRICS}
    paired = {
        key: _distribution(aware[key] - blind[key])
        for key in SCIENCE_METRICS
        if key != "measured_power_in_bucket"
    }
    passed = (
        paired["power_in_bucket"]["ci95_low"]
        > float(gate["min_power_delta_ci95_low"])
        and paired["strehl"]["ci95_low"]
        > float(gate["min_strehl_delta_ci95_low"])
        and paired["phase_rmse"]["ci95_high"]
        < float(gate["max_phase_rmse_delta_ci95_high"])
        and aware_summary["violation_fraction"]["mean"]
        <= float(gate["max_violation_fraction"])
    )
    return {
        "comparison": identifier,
        "episodes": int(aware["power_in_bucket"].numel()),
        "aware": aware_summary,
        "blind": blind_summary,
        "paired_delta_aware_minus_blind": paired,
        "relative_power_improvement_over_blind": (
            aware_summary["power_in_bucket"]["mean"]
            - blind_summary["power_in_bucket"]["mean"]
        )
        / blind_summary["power_in_bucket"]["mean"],
        "improvement_gate": "PASS" if passed else "FAIL",
    }


def _invariance_summary(
    identifier: str,
    aware: dict[str, torch.Tensor],
    blind: dict[str, torch.Tensor],
    *,
    tolerance: float,
) -> dict[str, Any]:
    maxima = {
        key: float((aware[key] - blind[key]).abs().max())
        for key in SCIENCE_METRICS + ACTION_METRICS
    }
    return {
        "profile": identifier,
        "max_absolute_differences": maxima,
        "tolerance": tolerance,
        "status": "PASS" if max(maxima.values()) <= tolerance else "FAIL",
    }


def _available_profile_ids(
    requested: Iterable[str], available: dict[str, Any]
) -> list[str]:
    """保持预声明顺序，只返回当前正式或快速设置实际评估的档位。"""
    return [str(identifier) for identifier in requested if str(identifier) in available]


def _verify_action_range_failure(upstream: dict[str, Any]) -> dict[str, Any]:
    paths: dict[str, Path] = {}
    for field in ("summary", "source_manifest", "experiment_config", "audit_record"):
        path = _project_path(upstream[field])
        if _file_sha256(path) != str(upstream[f"{field}_sha256"]):
            raise RuntimeError(f"action-range upstream hash mismatch: {field}")
        paths[field] = path
    summary = _load_json(paths["summary"])
    if (
        bool(summary["experiment"]["quick"])
        or summary["truth_alignment"]["status"] != "PASS"
        or summary["interpretation"]["status"] != "NO_RANGE_DEMONSTRATED"
        or summary["interpretation"]["minimum_demonstrated_limit_rad"] is not None
        or bool(summary["interpretation"]["capacity_demonstrated"])
        or bool(summary["interpretation"]["r2_authorized"])
        or bool(summary["interpretation"]["s4d3_authorized"])
        or any(
            item["capacity_gate"] != "FAIL"
            for item in summary["cumulative_limit_hindsight_envelopes"]
        )
    ):
        raise RuntimeError("formal action-range failure state changed")
    audit = paths["audit_record"].read_text(encoding="utf-8")
    if "Verification Status: `ANALYZED`" not in audit or "正式结论为 **FAIL**" not in audit:
        raise RuntimeError("action-range audit is not an ANALYZED failure")
    recorded_manifest = _load_json(paths["source_manifest"])
    for relative, digest in recorded_manifest.items():
        source = _project_path(relative)
        if not source.is_file() or _file_sha256(source) != digest:
            raise RuntimeError(f"action-range tracked source changed: {relative}")
    return {
        "config": _load_yaml(paths["experiment_config"]),
        "summary_sha256": _file_sha256(paths["summary"]),
        "audit_sha256": _file_sha256(paths["audit_record"]),
    }


def _validate_registration_seeds(
    experiment: dict[str, Any], settings: dict[str, Any]
) -> None:
    formal = _declared_seeds(
        experiment["evaluation"]["physical_conditions"],
        int(experiment["evaluation"]["episodes_per_physical_condition"]),
    )
    quick = _declared_seeds(
        experiment["quick"]["physical_conditions"],
        int(experiment["quick"]["batch_size"]),
    )
    active = _declared_seeds(settings["physical_conditions"], int(settings["batch_size"]))
    if len(active) != len(settings["physical_conditions"]) * int(settings["batch_size"]):
        raise RuntimeError("registration-oracle active conditions reuse seeds")
    declared = formal | quick
    for item in experiment["protected_seed_ranges"]:
        start = int(item["start_inclusive"])
        end = int(item["end_exclusive"])
        if any(start <= seed < end for seed in declared):
            raise RuntimeError(f"registration-oracle overlaps protected seeds: {item['id']}")
    reserved = int(experiment["reserved_future_learning_seed_base"])
    if any(seed >= reserved for seed in declared):
        raise RuntimeError("registration-oracle overlaps future learning seeds")
    if formal & quick:
        raise RuntimeError("registration-oracle formal and quick seeds overlap")


def _declared_seeds(
    conditions: Iterable[dict[str, Any]], batch_size: int
) -> set[int]:
    items = list(conditions)
    values = {
        int(item["base_seed"]) + offset
        for item in items
        for offset in range(batch_size)
    }
    if len(values) != len(items) * batch_size:
        raise RuntimeError("registration-oracle declared conditions reuse seeds")
    return values


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
        "comparison_gate": deepcopy(experiment["comparison_gate"]),
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


def _scenario_record(
    *,
    controller: str,
    variant: str,
    profile: str,
    condition: str,
    candidate: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
) -> dict[str, Any]:
    return {
        "controller": controller,
        "variant": variant,
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
        "normalized_saturation_fraction": float(
            candidate["normalized_saturation_fraction"].mean()
        ),
        "target_request_error_abs_mean_rad": float(
            candidate["target_request_error_abs_mean_rad"].mean()
        ),
        "selected_preview_horizon_mean": float(
            candidate["selected_preview_horizon_frames"].mean()
        ),
    }


def _write_scenario_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(json_safe(rows))
