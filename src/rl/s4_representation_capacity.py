"""S4-D2-R1多维动作表示能力扫描；不训练强化学习。"""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import math
from pathlib import Path
import time
from typing import Any, Iterable

import torch

from src.rl.s4_oracle_bound import (
    ACTION_METRICS,
    SCIENCE_METRICS,
    _collect_metrics,
    _concatenate_metrics,
    _load_json,
    _paired_summary,
    _record_progress,
    _science_context_summary,
    select_hindsight_envelope,
)
from src.rl.s4_registration_oracle import _comparison_summary
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
from src.simulation.controllers import TrackingLeakyIntegratorController
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.hardware_effects import HardwareProfile, apply_registration_error
from src.simulation.modes import (
    make_hybrid_spatial_basis,
    make_low_order_zernike_basis,
)
from src.simulation.robust_control import RobustnessCondition
from src.training_progress import advance_to, counted_progress, update_progress


ANCHOR_MODES = 10
FAIR_ACTION_METRICS = (
    "requested_residual_phase_rms_rad",
    "realized_residual_phase_rms_rad",
    "residual_global_budget_limited_fraction",
    "final_global_budget_limited_fraction",
    "request_global_budget_limited_fraction",
    "target_request_phase_rmse_rad",
)


@dataclass(frozen=True)
class ActionRepresentation:
    """一个预声明动作基。"""

    identifier: str
    kind: str
    num_modes: int

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "ActionRepresentation":
        representation = cls(
            identifier=str(values["id"]),
            kind=str(values["kind"]),
            num_modes=int(values["num_modes"]),
        )
        representation.validate()
        return representation

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("representation id must not be empty")
        if self.kind not in {"zernike", "hybrid_spatial"}:
            raise ValueError(f"unsupported representation kind: {self.kind}")
        if self.kind == "zernike" and not ANCHOR_MODES <= self.num_modes <= 36:
            raise ValueError("zernike representation must contain 10 to 36 modes")
        if self.kind == "hybrid_spatial" and not ANCHOR_MODES < self.num_modes <= 256:
            raise ValueError("hybrid spatial representation must contain 11 to 256 modes")


@dataclass(frozen=True)
class AnchoredAction:
    """保留10维传统控制器、同时应用高维残差后的动作记录。"""

    requested_residual_rad: torch.Tensor
    realized_residual_rad: torch.Tensor
    baseline_delta_rad: torch.Tensor
    final_delta_rad: torch.Tensor
    target_request_rad: torch.Tensor
    coordinate_saturation_fraction: torch.Tensor
    residual_global_limited: torch.Tensor
    final_global_limited: torch.Tensor
    request_global_limited: torch.Tensor


class AnchoredResidualController:
    """传统控制器固定在前10个模态，高维部分只属于诊断残差。"""

    def __init__(
        self,
        *,
        num_modes: int,
        modal_limit_rad: float,
        residual_limit_rad: float,
        final_step_limit_rad: float,
        parameters: dict[str, Any],
    ) -> None:
        if num_modes < ANCHOR_MODES:
            raise ValueError("representation must contain the ten anchor modes")
        self.num_modes = num_modes
        self.modal_limit_rad = modal_limit_rad
        self.residual_limit_rad = residual_limit_rad
        self.final_step_limit_rad = final_step_limit_rad
        self.residual_l2_budget = math.sqrt(ANCHOR_MODES) * residual_limit_rad
        self.final_l2_budget = math.sqrt(ANCHOR_MODES) * final_step_limit_rad
        self.request_l2_budget = math.sqrt(ANCHOR_MODES) * modal_limit_rad
        self.baseline = TrackingLeakyIntegratorController(
            num_modes=ANCHOR_MODES,
            modal_limit_rad=modal_limit_rad,
            gain=float(parameters["gain"]),
            leak=float(parameters["leak"]),
            tracking_gain=float(parameters["tracking_gain"]),
            max_request_step_rad=float(parameters["max_request_step_rad"]),
        )
        self.requested_coefficients: torch.Tensor | None = None
        self._prior: torch.Tensor | None = None
        self._baseline_delta: torch.Tensor | None = None

    def reset(self, anchor_observation: torch.Tensor) -> None:
        self._validate_anchor_observation(anchor_observation)
        self.baseline.reset(
            anchor_observation.shape[0],
            anchor_observation.device,
            anchor_observation.dtype,
        )
        self.requested_coefficients = torch.zeros(
            anchor_observation.shape[0],
            self.num_modes,
            device=anchor_observation.device,
            dtype=anchor_observation.dtype,
        )
        self.prepare(anchor_observation)

    def prepare(self, anchor_observation: torch.Tensor) -> None:
        self._validate_anchor_observation(anchor_observation)
        if self.requested_coefficients is None:
            raise RuntimeError("reset must be called before prepare")
        prior = self.requested_coefficients.clone()
        baseline_delta = torch.zeros_like(prior)
        baseline_delta[:, :ANCHOR_MODES] = self.baseline.action(anchor_observation)
        self._prior = prior
        self._baseline_delta = baseline_delta

    def compose_to_target(self, target_request: torch.Tensor) -> AnchoredAction:
        if self._prior is None or self._baseline_delta is None:
            raise RuntimeError("prepare must be called before compose_to_target")
        if target_request.shape != self._prior.shape:
            raise ValueError("target request shape does not match representation")
        prior = self._prior
        baseline_delta = self._baseline_delta
        target_request, request_limited = clip_box_and_l2(
            target_request,
            component_limit=self.modal_limit_rad,
            l2_limit=self.request_l2_budget,
        )
        requested_residual = target_request - prior - baseline_delta
        coordinate_saturation = requested_residual.abs().gt(self.residual_limit_rad)
        residual, residual_limited = clip_box_and_l2(
            requested_residual,
            component_limit=self.residual_limit_rad,
            l2_limit=self.residual_l2_budget,
        )
        combined, final_limited = clip_box_and_l2(
            baseline_delta + residual,
            component_limit=self.final_step_limit_rad,
            l2_limit=self.final_l2_budget,
        )
        requested, accumulated_limited = clip_box_and_l2(
            prior + combined,
            component_limit=self.modal_limit_rad,
            l2_limit=self.request_l2_budget,
        )
        final_delta = requested - prior
        realized_residual = final_delta - baseline_delta
        self.requested_coefficients = requested
        self.baseline.requested_modal = requested[:, :ANCHOR_MODES].clone()
        self._prior = None
        self._baseline_delta = None
        return AnchoredAction(
            requested_residual_rad=requested_residual,
            realized_residual_rad=realized_residual,
            baseline_delta_rad=baseline_delta,
            final_delta_rad=final_delta,
            target_request_rad=target_request,
            coordinate_saturation_fraction=coordinate_saturation.float().mean(dim=-1),
            residual_global_limited=residual_limited,
            final_global_limited=final_limited,
            request_global_limited=torch.maximum(request_limited, accumulated_limited),
        )

    @staticmethod
    def _validate_anchor_observation(observation: torch.Tensor) -> None:
        expected = 2 * ANCHOR_MODES + 2
        if observation.ndim != 2 or observation.shape[1] != expected:
            raise ValueError(f"anchor observation must have shape [batch, {expected}]")


def clip_box_and_l2(
    values: torch.Tensor,
    *,
    component_limit: float,
    l2_limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """先做逐坐标盒约束，再做每个样本的总L2预算投影。"""
    if component_limit <= 0 or l2_limit <= 0:
        raise ValueError("action limits must be positive")
    clipped = values.clamp(-component_limit, component_limit)
    norm = torch.linalg.vector_norm(clipped, dim=-1, keepdim=True)
    scale = (l2_limit / norm.clamp_min(1e-12)).clamp(max=1.0)
    projected = clipped * scale
    return projected, scale.squeeze(-1).lt(1 - 1e-7).to(values.dtype)


def run_s4_representation_capacity(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """比较10/21/36/256维动作表示的不可部署理想能力。"""
    experiment_path = _project_path(config_path)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight = preflight_s4_representation_capacity(
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
            "representation-capacity output already exists; preserve and audit it: "
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
    basis_data = {
        item.identifier: build_action_basis(base_config, item, device)
        for item in representations
    }
    bases = {identifier: values[0] for identifier, values in basis_data.items()}
    pupils = {identifier: values[1] for identifier, values in basis_data.items()}
    profiles = _profiles(experiment, settings["profile_ids"])
    conditions = [
        RobustnessCondition.from_mapping(item)
        for item in settings["physical_conditions"]
    ]
    scenario_count = len(profiles) * len(conditions)
    total_rollouts = scenario_count * (
        1 + len(representations) * len(horizons)
    )
    progress = counted_progress(
        total=total_rollouts,
        description="S4-D2-R1动作表示能力扫描",
        unit="轨迹组",
    )
    progress_path = output_directory / "progress.jsonl"
    completed = 0
    started = time.perf_counter()

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
    modal_ceiling: dict[str, list[torch.Tensor]] = defaultdict(list)
    modal_ceiling_baseline: dict[str, list[torch.Tensor]] = defaultdict(list)
    scenario_records: list[dict[str, Any]] = []
    truth_alignment_max = 0.0

    mappings: dict[tuple[str, str], torch.Tensor] = {}
    mapping_diagnostics: dict[str, dict[str, dict[str, Any]]] = {}
    for representation in representations:
        mapping_diagnostics[representation.identifier] = {}
        basis = bases[representation.identifier]
        pupil = pupils[representation.identifier]
        for profile in profiles:
            mapping, diagnostics = representation_registration_inverse(
                basis=basis,
                pupil=pupil,
                profile=profile,
                rcond=float(experiment["registration_inverse"]["pseudoinverse_rcond"]),
            )
            mappings[(representation.identifier, profile.identifier)] = mapping
            mapping_diagnostics[representation.identifier][profile.identifier] = diagnostics

    anchor_representation = _representation_by_id(
        representations,
        str(experiment["comparison"]["anchor_representation_id"]),
    )
    anchor_basis = bases[anchor_representation.identifier]

    for profile in profiles:
        for condition in conditions:
            anchor_config = replace(base_config, num_modes=ANCHOR_MODES)
            anchor_config = profile.environment_config(
                condition.environment_config(anchor_config)
            )
            baseline, ceiling = _rollout_anchor_baseline_and_ceiling(
                experiment=experiment,
                config=anchor_config,
                condition=condition,
                profile=profile,
                steps=int(settings["steps"]),
                basis=anchor_basis[:ANCHOR_MODES],
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
                controller="frozen_anchor_baseline",
                power=float(baseline["power_in_bucket"].mean()),
                started=started,
            )
            advance_to(progress, completed)

            for representation in representations:
                config = replace(base_config, num_modes=representation.num_modes)
                config = profile.environment_config(condition.environment_config(config))
                basis = bases[representation.identifier]
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
                    candidate, alignment = _rollout_representation_preview(
                        experiment=experiment,
                        config=config,
                        condition=condition,
                        profile=profile,
                        steps=int(settings["steps"]),
                        preview_horizon_frames=horizon,
                        future_disturbance=future_truth,
                        registration_mapping=mappings[
                            (representation.identifier, profile.identifier)
                        ],
                        basis=basis,
                        device=device,
                    )
                    truth_alignment_max = max(truth_alignment_max, alignment)
                    horizon_results[horizon] = candidate
                    scenario_records.append(
                        _scenario_record(
                            controller=f"{representation.identifier}_preview_{horizon}",
                            representation=representation,
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
                        controller=f"{representation.identifier}_preview_{horizon}",
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
                _collect_metrics(candidates[representation.identifier], envelope)
                _collect_metrics(baselines[representation.identifier], baseline)
                _collect_metrics(
                    profile_candidates[representation.identifier][profile.identifier],
                    envelope,
                )
                _collect_metrics(
                    profile_baselines[representation.identifier][profile.identifier],
                    baseline,
                )
                scenario_records.append(
                    _scenario_record(
                        controller=f"{representation.identifier}_hindsight_envelope",
                        representation=representation,
                        profile=profile.identifier,
                        condition=condition.identifier,
                        candidate=envelope,
                        baseline=baseline,
                    )
                )
    progress.close()

    representation_summaries: list[dict[str, Any]] = []
    concatenated_candidates: dict[str, dict[str, torch.Tensor]] = {}
    for representation in representations:
        identifier = representation.identifier
        concatenated_candidates[identifier] = _concatenate_metrics(candidates[identifier])
        overall = _capacity_summary(
            identifier,
            concatenated_candidates[identifier],
            _concatenate_metrics(baselines[identifier]),
            experiment["gate"],
        )
        per_profile = [
            _capacity_summary(
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
        target_ids = list(experiment["comparison"]["target_profile_ids"])
        target_profiles_pass = all(
            item["capacity_gate"] == "PASS"
            for item in per_profile
            if item["controller"] in target_ids
        ) and all(target in {item["controller"] for item in per_profile} for target in target_ids)
        representation_summaries.append(
            {
                "representation": asdict(representation),
                "overall": overall,
                "profiles": per_profile,
                "all_profiles_pass": all_profiles_pass,
                "target_profiles_pass": target_profiles_pass,
                "capacity_gate": (
                    "PASS"
                    if overall["capacity_gate"] == "PASS"
                    and all_profiles_pass
                    and target_profiles_pass
                    else "FAIL"
                ),
            }
        )

    anchor_candidates = concatenated_candidates[anchor_representation.identifier]
    comparisons = []
    for representation in representations:
        identifier = representation.identifier
        if identifier == anchor_representation.identifier:
            continue
        comparisons.append(
            {
                "representation": identifier,
                "overall": _comparison_summary(
                    identifier,
                    concatenated_candidates[identifier],
                    anchor_candidates,
                    experiment["comparison_gate"],
                ),
                "profiles": [
                    _comparison_summary(
                        profile.identifier,
                        _concatenate_metrics(
                            profile_candidates[identifier][profile.identifier]
                        ),
                        _concatenate_metrics(
                            profile_candidates[anchor_representation.identifier][
                                profile.identifier
                            ]
                        ),
                        experiment["comparison_gate"],
                    )
                    for profile in profiles
                ],
            }
        )

    raw_passing = [
        item["representation"]["identifier"]
        for item in representation_summaries
        if item["capacity_gate"] == "PASS"
    ]
    truth_alignment_pass = truth_alignment_max <= float(
        experiment["oracle"]["truth_alignment_tolerance"]
    )
    passing = raw_passing if truth_alignment_pass else []
    if not truth_alignment_pass:
        status = "TRUTH_ALIGNMENT_FAILED"
    elif quick:
        status = "QUICK_SMOKE_ONLY"
    else:
        status = (
            "REPRESENTATION_CAPACITY_FOUND"
            if passing
            else "NO_REPRESENTATION_CAPACITY"
        )
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run-diagnostic",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "s4d2_r1_representation_capacity_v1",
        },
        "experiment": {
            "id": "AO-S4-D2-R1-REPRESENTATION-CAPACITY",
            "type": "software_only_cuda_nonlearning_action_representation_capacity",
            "status": (
                "completed_pending_audit"
                if truth_alignment_pass
                else "failed_integrity_check"
            ),
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
                "Action representations share the same ten-mode baseline and physical "
                "coefficient budgets; this is an optimistic capacity diagnostic, not RL."
            ),
        },
        "inputs": {
            "experiment_config": _relative(experiment_path),
            "experiment_config_sha256": _file_sha256(experiment_path),
            "upstream_registration_summary_sha256": preflight[
                "upstream_registration_summary_sha256"
            ],
            "upstream_registration_audit_sha256": preflight[
                "upstream_registration_audit_sha256"
            ],
        },
        "design": {
            "paired_episode_seeds": True,
            "anchor_modes": ANCHOR_MODES,
            "representations": [asdict(item) for item in representations],
            "residual_component_limit_rad": float(
                experiment["action_budget"]["residual_component_limit_rad"]
            ),
            "residual_l2_budget_rad": math.sqrt(ANCHOR_MODES)
            * float(experiment["action_budget"]["residual_component_limit_rad"]),
            "final_component_step_limit_rad": float(
                experiment["action_budget"]["final_component_step_limit_rad"]
            ),
            "final_l2_budget_rad": math.sqrt(ANCHOR_MODES)
            * float(experiment["action_budget"]["final_component_step_limit_rad"]),
            "preview_horizons_frames": horizons,
            "profiles": [profile.identifier for profile in profiles],
            "physical_conditions": settings["physical_conditions"],
            "episodes_per_physical_condition": int(settings["batch_size"]),
            "steps": int(settings["steps"]),
            "progress_rollouts": total_rollouts,
        },
        "basis_diagnostics": preflight["basis_checks"],
        "mapping_diagnostics": mapping_diagnostics,
        "truth_alignment": {
            "method": "direct_turbulence_modal_projection",
            "max_absolute_modal_difference": truth_alignment_max,
            "tolerance": float(experiment["oracle"]["truth_alignment_tolerance"]),
            "status": "PASS" if truth_alignment_pass else "FAIL",
        },
        "representation_capacity": representation_summaries,
        "representation_vs_anchor": comparisons,
        "absolute_anchor_modal_ceiling_context": _science_context_summary(
            "unconstrained_instantaneous_ten_mode_ceiling",
            _concatenate_metrics(modal_ceiling),
            _concatenate_metrics(modal_ceiling_baseline),
        ),
        "interpretation": {
            "status": status,
            "passing_representation_ids": passing if not quick else [],
            "capacity_demonstrated": bool(passing) and not quick,
            "r2_authorized": False,
            "s4d3_authorized": False,
            "next_rule": (
                "Audit first. A passing representation only supports designing an R2 "
                "candidate; it does not authorize training, S4-D3, or hardware."
            ),
        },
        "runtime": _runtime_record(),
        "git": _git_record(),
        "next_action": (
            "Stop and ask the assistant to audit the representation-capacity result. "
            "Do not train R2 or open S4-D3."
        ),
    }
    _write_csv(output_directory / "scenario_records.csv", scenario_records)
    _write_json(output_directory / "summary.json", json_safe(summary))
    if summary["truth_alignment"]["status"] != "PASS":
        raise RuntimeError(
            "representation-capacity truth alignment failed: "
            f"max={truth_alignment_max:.10e}, "
            f"tolerance={float(experiment['oracle']['truth_alignment_tolerance']):.10e}; "
            "diagnostic summary was preserved"
        )
    return summary


def preflight_s4_representation_capacity(
    experiment_path: Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    """在创建输出目录前锁定上游FAIL、动作公平性和种子隔离。"""
    upstream = _verify_registration_failure(experiment["upstream_registration"])
    upstream_config = _load_yaml(
        _project_path(experiment["upstream_registration"]["experiment_config"])
    )
    _verify_fair_design(experiment, upstream_config)
    metadata = experiment["metadata"]
    forbidden = (
        bool(metadata.get("allow_training"))
        or bool(metadata.get("allow_optimizer_updates"))
        or bool(metadata.get("allow_old_trajectory_access"))
        or bool(metadata.get("allow_s4d3_access"))
        or bool(metadata.get("allow_real_hardware_actions"))
    )
    if forbidden:
        raise RuntimeError("representation-capacity metadata widened the safety boundary")
    if not bool(metadata.get("allow_simulator_truth_access")):
        raise RuntimeError("representation-capacity diagnostic requires explicit truth access")

    base_config, _ = load_s1_config(_project_path(experiment["environment_config"]))
    for field in ("environment_config", "hardware_profile_source"):
        if _file_sha256(_project_path(experiment[field])) != str(
            experiment[f"{field}_sha256"]
        ):
            raise RuntimeError(f"representation-capacity input hash mismatch: {field}")
    if base_config.num_modes != ANCHOR_MODES:
        raise RuntimeError("the frozen baseline must remain ten-dimensional")
    representations = [
        ActionRepresentation.from_mapping(item) for item in settings["representations"]
    ]
    identifiers = [item.identifier for item in representations]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("representation ids must be unique")
    anchor_id = str(experiment["comparison"]["anchor_representation_id"])
    anchor = _representation_by_id(representations, anchor_id)
    if anchor.kind != "zernike" or anchor.num_modes != ANCHOR_MODES:
        raise RuntimeError("comparison anchor must be the ten-mode Zernike representation")

    residual_limit = float(experiment["action_budget"]["residual_component_limit_rad"])
    final_limit = float(experiment["action_budget"]["final_component_step_limit_rad"])
    if residual_limit != 0.05 or final_limit != 0.15:
        raise RuntimeError("representation scan changed the frozen residual or final-step anchor")
    if not bool(experiment["action_budget"]["preserve_total_phase_rms_budget"]):
        raise RuntimeError("higher-dimensional actions must preserve the total phase RMS budget")

    basis_checks: dict[str, dict[str, Any]] = {}
    mapping_checks: dict[str, dict[str, dict[str, Any]]] = {}
    profiles = _profiles(experiment, settings["profile_ids"])
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
            if int(mapping["effective_rank"]) < int(
                math.ceil(
                    representation.num_modes
                    * float(experiment["registration_inverse"]["minimum_rank_fraction"])
                )
            ):
                raise RuntimeError(
                    f"registration mapping lost too much rank: {representation.identifier}/"
                    f"{profile.identifier}"
                )

    horizons = list(map(int, settings["preview_horizons_frames"]))
    if not horizons or horizons != sorted(set(horizons)) or horizons[0] < 0:
        raise ValueError("preview horizons must be sorted unique non-negative integers")
    _validate_seeds(experiment, settings)
    output_directory = _project_path(settings["output_directory"])
    if output_directory.exists():
        raise FileExistsError(f"representation-capacity output already exists: {output_directory}")
    if not all(_project_path(path).is_file() for path in experiment["tracked_source_files"]):
        raise FileNotFoundError("one or more representation-capacity sources are missing")
    scenario_count = len(settings["profile_ids"]) * len(settings["physical_conditions"])
    total = scenario_count * (1 + len(representations) * len(horizons))
    return {
        "status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_SIMULATION",
        "quick": quick,
        "experiment_config": _relative(experiment_path),
        "experiment_config_sha256": _file_sha256(experiment_path),
        "output_directory": _relative(output_directory),
        "base_environment_config": asdict(base_config),
        "representations": [asdict(item) for item in representations],
        "basis_checks": basis_checks,
        "mapping_checks": mapping_checks,
        "residual_component_limit_rad": residual_limit,
        "residual_l2_budget_rad": math.sqrt(ANCHOR_MODES) * residual_limit,
        "final_component_step_limit_rad": final_limit,
        "final_l2_budget_rad": math.sqrt(ANCHOR_MODES) * final_limit,
        "preview_horizons_frames": horizons,
        "scenario_count": scenario_count,
        "progress_rollouts": total,
        "scenario_record_rows": scenario_count
        * len(representations)
        * (len(horizons) + 1),
        "episodes_per_scenario": int(settings["batch_size"]),
        "steps": int(settings["steps"]),
        "upstream_registration_gate": "FAIL",
        "upstream_registration_summary_sha256": upstream["summary_sha256"],
        "upstream_registration_audit_sha256": upstream["audit_sha256"],
        "seed_isolation_verified": True,
        "cuda_required": True,
        "training_allowed": False,
        "optimizer_updates": 0,
        "future_simulator_truth_accessed": True,
        "truth_alignment_method": "direct_turbulence_modal_projection",
        "sealed_s4d3_accessed": False,
        "real_slm_actions": False,
        "formal_command": (
            ".\\.venv\\Scripts\\python.exe scripts\\run_s4_representation_capacity.py "
            "--config configs\\experiments\\s4_representation_capacity_v1.yaml"
        ),
    }


def build_action_basis(
    config: S1EnvConfig,
    representation: ActionRepresentation,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """构造动作基并检查光瞳内正交性与十维锚点一致性。"""
    cpu = torch.device("cpu")
    if representation.kind == "zernike":
        basis, pupil = make_low_order_zernike_basis(
            config.grid_size,
            config.pupil_radius_fraction,
            representation.num_modes,
            cpu,
            torch.float64,
        )
    else:
        basis, pupil = make_hybrid_spatial_basis(
            config.grid_size,
            config.pupil_radius_fraction,
            representation.num_modes,
            cpu,
            torch.float64,
            anchor_modes=ANCHOR_MODES,
        )
    anchor, _ = make_low_order_zernike_basis(
        config.grid_size,
        config.pupil_radius_fraction,
        ANCHOR_MODES,
        cpu,
        torch.float64,
    )
    gram = basis[:, pupil] @ basis[:, pupil].transpose(0, 1) / int(pupil.sum())
    error = float((gram - torch.eye(representation.num_modes, dtype=gram.dtype)).abs().max())
    anchor_difference = float((basis[:ANCHOR_MODES] - anchor).abs().max())
    if error > 1e-8:
        raise RuntimeError(
            f"action basis is not orthonormal: {representation.identifier}, error={error}"
        )
    if anchor_difference > 1e-10:
        raise RuntimeError("action basis changed the ten-mode anchor")
    return basis.to(device=device, dtype=torch.float32), pupil.to(device), {
        "kind": representation.kind,
        "num_modes": representation.num_modes,
        "pupil_samples": int(pupil.sum()),
        "max_orthonormality_error": error,
        "anchor_max_absolute_difference": anchor_difference,
    }


def representation_registration_inverse(
    *,
    basis: torch.Tensor,
    pupil: torch.Tensor,
    profile: HardwareProfile,
    rcond: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """计算给定动作基在已知配准误差下的最小二乘请求逆映射。"""
    if not 0 < rcond < 1:
        raise ValueError("pseudoinverse rcond must be in (0, 1)")
    num_modes = basis.shape[0]
    registration_present = (
        profile.shift_x_pixels != 0
        or profile.shift_y_pixels != 0
        or profile.rotation_deg != 0
    )
    if not registration_present:
        mapping = torch.eye(num_modes, device=basis.device, dtype=basis.dtype)
        mapping = mapping / profile.phase_scale
        return mapping, {
            "registration_present": False,
            "effective_rank": num_modes,
            "effective_rank_fraction": 1.0,
            "retained_condition_number": 1.0,
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
    retained = singular_values > rcond * singular_values.max()
    effective_rank = int(retained.sum())
    retained_condition = float(
        singular_values.max() / singular_values[retained].min()
    )
    mapping = torch.linalg.pinv(response_matrix, rcond=rcond) @ target_matrix
    fitted = response_matrix @ mapping
    reconstruction_rmse = torch.sqrt((fitted - target_matrix).square().mean())
    target_rms = torch.sqrt(target_matrix.square().mean()).clamp_min(1e-12)
    return mapping, {
        "registration_present": True,
        "effective_rank": effective_rank,
        "effective_rank_fraction": effective_rank / num_modes,
        "retained_condition_number": retained_condition,
        "relative_pupil_reconstruction_rmse": float(reconstruction_rmse / target_rms),
        "mapping_spectral_norm": float(torch.linalg.svdvals(mapping).max()),
    }


@torch.no_grad()
def _future_disturbance_sequence(
    *,
    config: S1EnvConfig,
    condition: RobustnessCondition,
    profile: HardwareProfile,
    length: int,
    basis: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    environment = AdaptiveOpticsEnv(
        config,
        device,
        profile.effects_config(),
        basis_override=basis,
    )
    environment.reset(seed=condition.base_seed)
    zero = torch.zeros(config.batch_size, config.num_modes, device=device)
    values = []
    for index in range(length):
        values.append(environment.oracle_disturbance_modal().clone())
        if index + 1 < length:
            environment.step(zero)
    return torch.stack(values, dim=0)


@torch.no_grad()
def _rollout_anchor_baseline_and_ceiling(
    *,
    experiment: dict[str, Any],
    config: S1EnvConfig,
    condition: RobustnessCondition,
    profile: HardwareProfile,
    steps: int,
    basis: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    environment = AdaptiveOpticsEnv(
        config,
        device,
        profile.effects_config(),
        basis_override=basis,
    )
    observation, _ = environment.reset(seed=condition.base_seed)
    generator = torch.Generator(device=device).manual_seed(condition.base_seed + 40_000_000)
    observation = _noisy_observation(
        observation,
        ANCHOR_MODES,
        profile.observation_noise_std_rad,
        generator,
    )
    parameters = experiment["frozen_controller"]["parameters"]
    controller = TrackingLeakyIntegratorController(
        num_modes=ANCHOR_MODES,
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
        for key in ("power_in_bucket", "strehl", "phase_rmse"):
            ceiling[key].append(ideal[key].detach().cpu())
        ceiling["measured_power_in_bucket"].append(
            ideal["power_in_bucket"].detach().cpu()
        )
        ceiling["violation_fraction"].append(
            torch.zeros_like(ideal["power_in_bucket"]).cpu()
        )
        action = controller.action(observation)
        observation, _, _, _, info = environment.step(action)
        _append_step_metrics(baseline, info)
        if step + 1 < steps:
            observation = _noisy_observation(
                observation,
                ANCHOR_MODES,
                profile.observation_noise_std_rad,
                generator,
            )
    return _mean_step_metrics(baseline), _mean_step_metrics(ceiling)


@torch.no_grad()
def _rollout_representation_preview(
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
) -> tuple[dict[str, torch.Tensor], float]:
    environment = AdaptiveOpticsEnv(
        config,
        device,
        profile.effects_config(),
        basis_override=basis,
    )
    observation, _ = environment.reset(seed=condition.base_seed)
    raw_observation = observation
    generator = torch.Generator(device=device).manual_seed(condition.base_seed + 40_000_000)
    anchor_observation = _extract_anchor_observation(observation, config.num_modes)
    anchor_observation = _noisy_observation(
        anchor_observation,
        ANCHOR_MODES,
        profile.observation_noise_std_rad,
        generator,
    )
    controller = AnchoredResidualController(
        num_modes=config.num_modes,
        modal_limit_rad=config.modal_limit_rad,
        residual_limit_rad=float(
            experiment["action_budget"]["residual_component_limit_rad"]
        ),
        final_step_limit_rad=float(
            experiment["action_budget"]["final_component_step_limit_rad"]
        ),
        parameters=experiment["frozen_controller"]["parameters"],
    )
    controller.reset(anchor_observation)
    science = _empty_step_metrics()
    actions: dict[str, list[torch.Tensor]] = {
        key: []
        for key in tuple(ACTION_METRICS) + FAIR_ACTION_METRICS
        if key != "selected_preview_horizon_frames"
    }
    alignment_max = 0.0
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
        action = controller.compose_to_target(target_request)
        raw_observation, _, _, _, info = environment.step(action.final_delta_rad)
        _append_step_metrics(science, info)

        requested_rms = torch.linalg.vector_norm(
            action.requested_residual_rad,
            dim=-1,
        )
        realized_rms = torch.linalg.vector_norm(
            action.realized_residual_rad,
            dim=-1,
        )
        target_error = torch.linalg.vector_norm(
            controller.requested_coefficients - action.target_request_rad,
            dim=-1,
        )
        actions["requested_residual_abs_mean_rad"].append(
            action.requested_residual_rad.abs().mean(dim=-1).cpu()
        )
        actions["realized_residual_abs_mean_rad"].append(
            action.realized_residual_rad.abs().mean(dim=-1).cpu()
        )
        actions["normalized_saturation_fraction"].append(
            action.coordinate_saturation_fraction.cpu()
        )
        actions["target_request_error_abs_mean_rad"].append(
            (controller.requested_coefficients - action.target_request_rad)
            .abs()
            .mean(dim=-1)
            .cpu()
        )
        actions["requested_residual_phase_rms_rad"].append(requested_rms.cpu())
        actions["realized_residual_phase_rms_rad"].append(realized_rms.cpu())
        actions["residual_global_budget_limited_fraction"].append(
            action.residual_global_limited.cpu()
        )
        actions["final_global_budget_limited_fraction"].append(
            action.final_global_limited.cpu()
        )
        actions["request_global_budget_limited_fraction"].append(
            action.request_global_limited.cpu()
        )
        actions["target_request_phase_rmse_rad"].append(target_error.cpu())
        if step + 1 < steps:
            anchor_observation = _extract_anchor_observation(
                raw_observation,
                config.num_modes,
            )
            anchor_observation = _noisy_observation(
                anchor_observation,
                ANCHOR_MODES,
                profile.observation_noise_std_rad,
                generator,
            )
            controller.prepare(anchor_observation)
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


def _extract_anchor_observation(
    observation: torch.Tensor,
    num_modes: int,
) -> torch.Tensor:
    if observation.ndim != 2 or observation.shape[1] != 2 * num_modes + 2:
        raise ValueError("representation observation has the wrong shape")
    return torch.cat(
        (
            observation[:, :ANCHOR_MODES],
            observation[:, num_modes : num_modes + ANCHOR_MODES],
            observation[:, -2:],
        ),
        dim=-1,
    )


def _capacity_summary(
    identifier: str,
    candidate: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
    gate: dict[str, Any],
) -> dict[str, Any]:
    result = _paired_summary(identifier, candidate, baseline, gate)
    result["fair_action_diagnostics"] = {
        key: _distribution(candidate[key]) for key in FAIR_ACTION_METRICS
    }
    return result


def _scenario_record(
    *,
    controller: str,
    representation: ActionRepresentation,
    profile: str,
    condition: str,
    candidate: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
) -> dict[str, Any]:
    candidate_power = float(candidate["power_in_bucket"].mean())
    baseline_power = float(baseline["power_in_bucket"].mean())
    return {
        "controller": controller,
        "representation": representation.identifier,
        "representation_kind": representation.kind,
        "action_dimension": representation.num_modes,
        "profile": profile,
        "physical_condition": condition,
        "episodes": int(candidate["power_in_bucket"].numel()),
        "candidate_power_mean": candidate_power,
        "baseline_power_mean": baseline_power,
        "relative_power_gain": (candidate_power - baseline_power) / baseline_power,
        "power_delta_mean": float(
            (candidate["power_in_bucket"] - baseline["power_in_bucket"]).mean()
        ),
        "strehl_delta_mean": float(
            (candidate["strehl"] - baseline["strehl"]).mean()
        ),
        "phase_rmse_delta_mean": float(
            (candidate["phase_rmse"] - baseline["phase_rmse"]).mean()
        ),
        "violation_fraction_mean": float(candidate["violation_fraction"].mean()),
        "requested_residual_phase_rms_rad": float(
            candidate["requested_residual_phase_rms_rad"].mean()
        ),
        "residual_global_budget_limited_fraction": float(
            candidate["residual_global_budget_limited_fraction"].mean()
        ),
        "final_global_budget_limited_fraction": float(
            candidate["final_global_budget_limited_fraction"].mean()
        ),
        "target_request_phase_rmse_rad": float(
            candidate["target_request_phase_rmse_rad"].mean()
        ),
        "selected_preview_horizon_mean": float(
            candidate["selected_preview_horizon_frames"].mean()
        ),
    }


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
        settings.update(
            {
                "output_directory": experiment["outputs"]["quick_directory"],
                "steps": int(quick_settings["steps"]),
                "batch_size": int(quick_settings["batch_size"]),
                "preview_horizons_frames": list(
                    quick_settings["preview_horizons_frames"]
                ),
                "profile_ids": list(quick_settings["profile_ids"]),
                "physical_conditions": list(quick_settings["physical_conditions"]),
                "representations": [
                    item
                    for item in experiment["representations"]
                    if item["id"] in quick_settings["representation_ids"]
                ],
            }
        )
    return settings


def _verify_registration_failure(upstream: dict[str, Any]) -> dict[str, Any]:
    paths: dict[str, Path] = {}
    for field in ("summary", "source_manifest", "experiment_config", "audit_record"):
        path = _project_path(upstream[field])
        if _file_sha256(path) != str(upstream[f"{field}_sha256"]):
            raise RuntimeError(f"registration-oracle upstream hash mismatch: {field}")
        paths[field] = path
    summary = _load_json(paths["summary"])
    if (
        bool(summary["experiment"]["quick"])
        or summary["truth_alignment"]["status"] != "PASS"
        or summary["registration_recovery"]["recovery_gate"] != "FAIL"
        or summary["interpretation"]["status"] != "REGISTRATION_NOT_RECOVERED"
        or bool(summary["interpretation"]["registration_recoverable"])
        or bool(summary["interpretation"]["r2_authorized"])
        or bool(summary["interpretation"]["s4d3_authorized"])
    ):
        raise RuntimeError("formal registration-oracle failure state changed")
    audit = paths["audit_record"].read_text(encoding="utf-8")
    if "Verification Status: `ANALYZED`" not in audit or "正式结论为 **FAIL**" not in audit:
        raise RuntimeError("registration-oracle audit record is not finalized as FAIL")
    return {
        "summary_sha256": _file_sha256(paths["summary"]),
        "audit_sha256": _file_sha256(paths["audit_record"]),
    }


def _validate_seeds(experiment: dict[str, Any], settings: dict[str, Any]) -> None:
    protected = [
        (int(item["start_inclusive"]), int(item["end_exclusive"]))
        for item in experiment["protected_seed_ranges"]
    ]
    batch = int(settings["batch_size"])
    seeds = []
    for condition in settings["physical_conditions"]:
        start = int(condition["base_seed"])
        stop = start + batch
        seeds.append((start, stop))
        if any(start < old_stop and stop > old_start for old_start, old_stop in protected):
            raise RuntimeError("representation-capacity seeds overlap protected prior ranges")
    for index, first in enumerate(seeds):
        for second in seeds[index + 1 :]:
            if first[0] < second[1] and first[1] > second[0]:
                raise RuntimeError("representation-capacity physical conditions overlap seeds")


def _verify_fair_design(
    experiment: dict[str, Any],
    upstream: dict[str, Any],
) -> None:
    """除动作表示和新种子外，锁定上一阶段的正式物理与门槛。"""
    if experiment["environment_config"] != upstream["environment_config"]:
        raise RuntimeError("representation scan changed the environment config path")
    if experiment["hardware_profile_source"] != upstream["hardware_profile_source"]:
        raise RuntimeError("representation scan changed the hardware profile source")
    if experiment["frozen_controller"]["id"] != upstream["frozen_controller"]["id"]:
        raise RuntimeError("representation scan changed the frozen controller")
    if experiment["frozen_controller"]["parameters"] != upstream["frozen_controller"]["parameters"]:
        raise RuntimeError("representation scan changed frozen controller parameters")
    if experiment["oracle"]["preview_horizons_frames"] != upstream["oracle"]["preview_horizons_frames"]:
        raise RuntimeError("representation scan changed formal preview horizons")
    if float(experiment["oracle"]["truth_alignment_tolerance"]) != float(
        upstream["oracle"]["truth_alignment_tolerance"]
    ):
        raise RuntimeError("representation scan changed truth-alignment tolerance")
    if experiment["gate"] != upstream["gate"]:
        raise RuntimeError("representation scan changed the capacity gate")
    if experiment["evaluation"]["profile_ids"] != upstream["evaluation"]["profile_ids"]:
        raise RuntimeError("representation scan changed formal hardware profiles")

    old_conditions = upstream["evaluation"]["physical_conditions"]
    new_conditions = experiment["evaluation"]["physical_conditions"]
    if len(old_conditions) != len(new_conditions):
        raise RuntimeError("representation scan changed physical-condition count")
    for old, new in zip(old_conditions, new_conditions, strict=True):
        old_physics = {
            key: value for key, value in old.items() if key not in {"id", "base_seed"}
        }
        new_physics = {
            key: value for key, value in new.items() if key not in {"id", "base_seed"}
        }
        if old_physics != new_physics:
            raise RuntimeError("representation scan changed formal physical conditions")
    if float(experiment["action_budget"]["residual_component_limit_rad"]) != float(
        upstream["action"]["residual_action_limit_rad"]
    ):
        raise RuntimeError("representation scan changed residual component limit")
    if float(experiment["action_budget"]["final_component_step_limit_rad"]) != float(
        upstream["action"]["final_action_step_limit_rad"]
    ):
        raise RuntimeError("representation scan changed final component step limit")


def _representation_by_id(
    representations: Iterable[ActionRepresentation],
    identifier: str,
) -> ActionRepresentation:
    matches = [item for item in representations if item.identifier == identifier]
    if len(matches) != 1:
        raise ValueError(f"expected one representation named {identifier}")
    return matches[0]


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("cannot write empty representation-capacity records")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
