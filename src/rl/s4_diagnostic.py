"""S4-D2残差SAC失败后的只读CUDA诊断。"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Iterable

import torch

from src.rl.residual_sac import SacConfig, SquashedGaussianActor
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
    _residual_reward,
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
    "training_style_reward",
    "raw_normalized_abs_mean",
    "scaled_normalized_abs_mean",
    "requested_residual_abs_mean_rad",
    "realized_residual_abs_mean_rad",
    "baseline_delta_abs_mean_rad",
    "final_delta_abs_mean_rad",
    "baseline_residual_cosine",
    "cancellation_fraction",
    "same_direction_fraction",
    "residual_projection_fraction",
    "normalized_saturation_fraction",
)


def run_s4_residual_diagnostic(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """运行残差缩放与方向诊断，不训练、不更新检查点。"""
    experiment_path = _project_path(config_path)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight = preflight_s4_residual_diagnostic(
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
            "diagnostic output already exists; preserve it and audit before retrying: "
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
    base_config = replace(
        base_config,
        batch_size=int(settings["batch_size"]),
        episode_length=max(base_config.episode_length, int(settings["steps"])),
    )
    profiles = _profiles(experiment, settings["profile_ids"])
    conditions = [
        RobustnessCondition.from_mapping(item)
        for item in settings["physical_conditions"]
    ]
    checkpoints = settings["checkpoints"]
    nonzero_variants = [
        item for item in settings["variants"] if float(item["scale"]) != 0.0
    ]
    scenario_count = len(profiles) * len(conditions)
    total_rollouts = scenario_count * (
        2 + len(checkpoints) * len(nonzero_variants)
    )
    progress = counted_progress(
        total=total_rollouts,
        description="S4-D2失败诊断",
        unit="轨迹组",
    )
    completed = 0
    started = time.perf_counter()
    scenario_records: list[dict[str, Any]] = []
    aggregate_candidate: dict[tuple[str, int | None], dict[str, list[torch.Tensor]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    aggregate_baseline: dict[tuple[str, int | None], dict[str, list[torch.Tensor]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    zero_max_abs = 0.0

    configured_scenarios: list[
        tuple[HardwareProfile, RobustnessCondition, S1EnvConfig]
    ] = []
    for profile in profiles:
        for condition in conditions:
            config = profile.environment_config(condition.environment_config(base_config))
            configured_scenarios.append((profile, condition, config))

    baseline_cache: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    zero_variant = next(
        item for item in settings["variants"] if float(item["scale"]) == 0.0
    )
    for profile, condition, config in configured_scenarios:
        scenario_key = (profile.identifier, condition.identifier)
        baseline = _rollout_baseline(
            experiment, config, condition, profile, int(settings["steps"]), device
        )
        baseline_cache[scenario_key] = baseline
        completed += 1
        advance_to(progress, completed)
        update_progress(
            progress,
            device=device,
            metrics={"最近功率": float(baseline["power_in_bucket"].mean())},
        )

        zero = _rollout_residual_variant(
            actor=None,
            scale=0.0,
            experiment=experiment,
            config=config,
            condition=condition,
            profile=profile,
            steps=int(settings["steps"]),
            device=device,
        )
        zero_diff = max(
            float((zero[key] - baseline[key]).abs().max())
            for key in SCIENCE_METRICS
        )
        zero_max_abs = max(zero_max_abs, zero_diff)
        _collect_pair(
            aggregate_candidate,
            aggregate_baseline,
            (str(zero_variant["id"]), None),
            zero,
            baseline,
        )
        scenario_records.append(
            _scenario_record(
                policy_seed=None,
                variant=zero_variant,
                profile=profile,
                condition=condition,
                candidate=zero,
                baseline=baseline,
            )
        )
        completed += 1
        advance_to(progress, completed)
        update_progress(
            progress,
            device=device,
            metrics={"零残差误差": zero_max_abs},
        )

    for checkpoint in checkpoints:
        policy_seed = int(checkpoint["policy_seed"])
        actor = _load_actor(checkpoint, device)
        for variant in nonzero_variants:
            key = (str(variant["id"]), policy_seed)
            for profile, condition, config in configured_scenarios:
                baseline = baseline_cache[(profile.identifier, condition.identifier)]
                candidate = _rollout_residual_variant(
                    actor=actor,
                    scale=float(variant["scale"]),
                    experiment=experiment,
                    config=config,
                    condition=condition,
                    profile=profile,
                    steps=int(settings["steps"]),
                    device=device,
                )
                _collect_pair(
                    aggregate_candidate,
                    aggregate_baseline,
                    key,
                    candidate,
                    baseline,
                )
                scenario_records.append(
                    _scenario_record(
                        policy_seed=policy_seed,
                        variant=variant,
                        profile=profile,
                        condition=condition,
                        candidate=candidate,
                        baseline=baseline,
                    )
                )
                completed += 1
                advance_to(progress, completed)
                update_progress(
                    progress,
                    device=device,
                    metrics={
                        "种子": float(policy_seed),
                        "缩放": float(variant["scale"]),
                        "功率差": float(
                            (candidate["power_in_bucket"] - baseline["power_in_bucket"]).mean()
                        ),
                    },
                )
        del actor
    progress.close()

    policy_variant_summaries = _policy_variant_summaries(
        aggregate_candidate,
        aggregate_baseline,
        settings["variants"],
    )
    grouped = _group_variant_summaries(
        aggregate_candidate,
        aggregate_baseline,
        settings["variants"],
    )
    diagnosis = interpret_diagnostic(
        grouped,
        zero_max_abs=zero_max_abs,
        zero_tolerance=float(settings["zero_equivalence_tolerance"]),
    )
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run-diagnostic",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "s4d2_residual_sac_diagnostic_v1",
        },
        "experiment": {
            "id": "AO-S4-D2-RESIDUAL-SAC-FAILURE-DIAGNOSTIC",
            "type": "software_only_cuda_post_hoc_diagnostic",
            "status": "completed_pending_audit",
            "quick": quick,
            "duration_seconds": time.perf_counter() - started,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "evidence_boundary": {
            "training_performed": False,
            "optimizer_updates": 0,
            "checkpoints_modified": False,
            "s4d1_trajectories_accessed": False,
            "sealed_s4d3_accessed": False,
            "real_slm_actions": False,
            "post_hoc_diagnostic_only": True,
            "interpretation": (
                "Residual scaling and sign reversal are diagnostic interventions, "
                "not pre-registered candidate algorithms or S4-D3 evidence."
            ),
        },
        "inputs": {
            "diagnostic_config": _relative(experiment_path),
            "diagnostic_config_sha256": _file_sha256(experiment_path),
            "upstream_s4d2_summary_sha256": preflight[
                "upstream_s4d2_summary_sha256"
            ],
            "checkpoint_count": len(checkpoints),
        },
        "design": {
            "paired_episode_seeds": True,
            "deterministic_policy": True,
            "batch_size": int(settings["batch_size"]),
            "steps": int(settings["steps"]),
            "profiles": list(settings["profile_ids"]),
            "physical_conditions": deepcopy(settings["physical_conditions"]),
            "variants": deepcopy(settings["variants"]),
        },
        "zero_residual_equivalence": {
            "max_absolute_science_metric_difference": zero_max_abs,
            "tolerance": float(settings["zero_equivalence_tolerance"]),
            "status": (
                "PASS"
                if zero_max_abs <= float(settings["zero_equivalence_tolerance"])
                else "FAIL"
            ),
        },
        "policy_variant_summaries": policy_variant_summaries,
        "grouped_variant_summaries": grouped,
        "diagnosis": diagnosis,
        "runtime": _runtime_record(),
        "git": _git_record(),
        "next_action": (
            "Stop and ask the assistant to audit the diagnostic. Do not retrain or open S4-D3."
        ),
    }
    _write_json(output_directory / "summary.json", summary)
    _write_scenario_csv(output_directory / "scenario_records.csv", scenario_records)
    return summary


def preflight_s4_residual_diagnostic(
    experiment_path: Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    """在占用CUDA和创建输出前核对失败结论、检查点与种子隔离。"""
    metadata = experiment["metadata"]
    if str(metadata["stage"]) != "S4-D2-DIAGNOSTIC":
        raise ValueError("diagnostic metadata must identify S4-D2-DIAGNOSTIC")
    forbidden_flags = (
        "allow_training",
        "allow_checkpoint_updates",
        "allow_d1_trajectory_access",
        "allow_s4d3_access",
        "allow_real_hardware_actions",
    )
    if any(bool(metadata.get(field, True)) for field in forbidden_flags):
        raise RuntimeError("diagnostic mutation and protected-data flags must stay false")

    upstream = experiment["upstream_s4d2"]
    upstream_paths = {
        field: _project_path(upstream[field])
        for field in ("summary", "source_manifest", "experiment_config", "audit_record")
    }
    for field, path in upstream_paths.items():
        if _file_sha256(path) != str(upstream[f"{field}_sha256"]):
            raise RuntimeError(f"S4-D2 {field} hash mismatch")
    audit_text = upstream_paths["audit_record"].read_text(encoding="utf-8")
    if "Verification Status: `ANALYZED`" not in audit_text or "**FAIL**" not in audit_text:
        raise RuntimeError("S4-D2 audit must be ANALYZED and record a FAIL")
    training_summary = json.loads(upstream_paths["summary"].read_text(encoding="utf-8"))
    development = training_summary["development_summary"]
    if (
        training_summary["experiment"]["status"] != "completed_pending_audit"
        or development["status"] != "ANALYSIS_REQUIRED"
        or int(development["pass_count"]) != 0
        or bool(development["s4d3_authorized"])
    ):
        raise RuntimeError("S4-D2 failure state or S4-D3 lock changed")

    recorded_source = json.loads(
        upstream_paths["source_manifest"].read_text(encoding="utf-8")
    )
    if recorded_source != training_summary["inputs"]["source_manifest"]:
        raise RuntimeError("S4-D2 source manifests disagree")
    for relative, digest in recorded_source.items():
        path = _project_path(relative)
        if not path.is_file() or _file_sha256(path) != digest:
            raise RuntimeError(f"S4-D2 training source changed: {relative}")

    training_config = _load_yaml(upstream_paths["experiment_config"])
    for field in (
        "environment_config",
        "environment_config_sha256",
        "hardware_profile_source",
        "hardware_profile_source_sha256",
        "frozen_controller",
        "policy_observation",
        "action",
        "reward",
    ):
        if experiment[field] != training_config[field]:
            raise RuntimeError(f"diagnostic changed frozen S4-D2 field: {field}")
    if int(experiment["sac_actor"]["hidden_size"]) != int(
        training_config["sac"]["hidden_size"]
    ):
        raise RuntimeError("diagnostic actor width differs from training")

    summary_by_seed = {
        int(item["policy_seed"]): item for item in training_summary["policy_seed_results"]
    }
    for checkpoint in experiment["checkpoints"]:
        seed = int(checkpoint["policy_seed"])
        path = _project_path(checkpoint["path"])
        if seed not in summary_by_seed:
            raise RuntimeError(f"unknown policy seed: {seed}")
        if summary_by_seed[seed]["best_checkpoint"] != _relative(path):
            raise RuntimeError(f"checkpoint is not the recorded best for seed {seed}")
        if _file_sha256(path) != str(checkpoint["sha256"]):
            raise RuntimeError(f"checkpoint hash mismatch for seed {seed}")

    variants = settings["variants"]
    ids = [str(item["id"]) for item in variants]
    scales = [float(item["scale"]) for item in variants]
    if len(ids) != len(set(ids)) or any(abs(value) > 1 for value in scales):
        raise ValueError("diagnostic variants must have unique ids and scales within [-1, 1]")
    for required in (0.0, 0.25, 0.5, 1.0, -1.0):
        if required not in scales and not quick:
            raise ValueError(f"formal diagnostic is missing required scale {required}")

    _validate_diagnostic_seeds(experiment, settings)
    environment_path = _project_path(experiment["environment_config"])
    if _file_sha256(environment_path) != str(experiment["environment_config_sha256"]):
        raise RuntimeError("environment configuration hash mismatch")
    hardware_path = _project_path(experiment["hardware_profile_source"])
    if _file_sha256(hardware_path) != str(experiment["hardware_profile_source_sha256"]):
        raise RuntimeError("hardware profile source hash mismatch")
    base_config, _ = load_s1_config(environment_path)
    controller = _make_residual_controller(experiment, base_config)
    expected_state_size = int(experiment["sac_actor"]["state_size"])
    expected_action_size = int(experiment["sac_actor"]["action_size"])
    if controller.state_size != expected_state_size or base_config.num_modes != expected_action_size:
        raise RuntimeError("diagnostic actor dimensions differ from the environment contract")

    output_directory = _project_path(settings["output_directory"])
    if output_directory.exists():
        raise FileExistsError(f"diagnostic output already exists: {output_directory}")
    return {
        "status": "READY_FOR_QUICK_DIAGNOSTIC" if quick else "READY_FOR_USER_DIAGNOSTIC",
        "quick": quick,
        "experiment_config": _relative(experiment_path),
        "experiment_config_sha256": _file_sha256(experiment_path),
        "output_directory": _relative(output_directory),
        "upstream_s4d2_gate": "FAIL",
        "upstream_s4d2_summary_sha256": _file_sha256(upstream_paths["summary"]),
        "checkpoint_count": len(settings["checkpoints"]),
        "variant_count": len(settings["variants"]),
        "scenario_count": len(settings["profile_ids"]) * len(settings["physical_conditions"]),
        "episode_count_per_rollout": int(settings["batch_size"]),
        "steps_per_episode": int(settings["steps"]),
        "seed_isolation_verified": True,
        "cuda_required": True,
        "training_allowed": False,
        "optimizer_updates": 0,
        "checkpoints_are_read_only": True,
        "s4d1_trajectories_accessed": False,
        "sealed_s4d3_accessed": False,
        "real_slm_actions": False,
        "formal_command": (
            ".\\.venv\\Scripts\\python.exe scripts\\diagnose_s4_residual_sac.py "
            "--config configs\\experiments\\s4_residual_sac_diagnostic_v1.yaml"
        ),
    }


@torch.no_grad()
def _rollout_residual_variant(
    *,
    actor: SquashedGaussianActor | None,
    scale: float,
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
    science = _empty_step_metrics()
    actions: dict[str, list[torch.Tensor]] = {key: [] for key in ACTION_METRICS}
    for step in range(steps):
        if actor is None:
            raw = torch.zeros(
                config.batch_size,
                config.num_modes,
                device=device,
                dtype=state.dtype,
            )
        else:
            raw = actor.deterministic(state)
        scaled = (scale * raw).clamp(-1, 1)
        action = controller.compose_action(scaled)
        observation, _, _, _, info = environment.step(action.final_delta_rad)
        _append_step_metrics(science, info)
        baseline = action.baseline_delta_rad
        requested = action.requested_residual_rad
        denominator = baseline.norm(dim=-1) * requested.norm(dim=-1)
        cosine = torch.where(
            denominator > 1e-12,
            (baseline * requested).sum(dim=-1) / denominator.clamp_min(1e-12),
            torch.zeros_like(denominator),
        )
        dot = (baseline * requested).sum(dim=-1)
        actions["training_style_reward"].append(
            _residual_reward(
                measured_power=info["measured_power_in_bucket"],
                normalized_residual=scaled,
                violation=info["violation_fraction"],
                reward_config=experiment["reward"],
            ).detach().cpu()
        )
        actions["raw_normalized_abs_mean"].append(raw.abs().mean(dim=-1).cpu())
        actions["scaled_normalized_abs_mean"].append(scaled.abs().mean(dim=-1).cpu())
        actions["requested_residual_abs_mean_rad"].append(
            requested.abs().mean(dim=-1).cpu()
        )
        actions["realized_residual_abs_mean_rad"].append(
            action.realized_residual_rad.abs().mean(dim=-1).cpu()
        )
        actions["baseline_delta_abs_mean_rad"].append(
            baseline.abs().mean(dim=-1).cpu()
        )
        actions["final_delta_abs_mean_rad"].append(
            action.final_delta_rad.abs().mean(dim=-1).cpu()
        )
        actions["baseline_residual_cosine"].append(cosine.cpu())
        actions["cancellation_fraction"].append((dot < 0).float().cpu())
        actions["same_direction_fraction"].append((dot > 0).float().cpu())
        actions["residual_projection_fraction"].append(
            action.realized_residual_rad.sub(action.requested_residual_rad)
            .abs()
            .gt(1e-7)
            .float()
            .mean(dim=-1)
            .cpu()
        )
        actions["normalized_saturation_fraction"].append(
            scaled.abs().ge(0.999).float().mean(dim=-1).cpu()
        )
        if step + 1 < steps:
            noisy = _noisy_observation(
                observation,
                config.num_modes,
                profile.observation_noise_std_rad,
                generator,
            )
            state = controller.advance_observation(noisy)
    result = _mean_step_metrics(science)
    result.update(
        {key: torch.stack(values, dim=1).mean(dim=1) for key, values in actions.items()}
    )
    return result


@torch.no_grad()
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
    controller = TrackingLeakyIntegratorController(
        num_modes=config.num_modes,
        modal_limit_rad=config.modal_limit_rad,
        gain=float(parameters["gain"]),
        leak=float(parameters["leak"]),
        tracking_gain=float(parameters["tracking_gain"]),
        max_request_step_rad=float(parameters["max_request_step_rad"]),
    )
    controller.reset(config.batch_size, device, observation.dtype)
    metrics = _empty_step_metrics()
    for step in range(steps):
        action = controller.action(observation)
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


def _load_actor(
    checkpoint: dict[str, Any], device: torch.device
) -> SquashedGaussianActor:
    payload = torch.load(
        _project_path(checkpoint["path"]),
        map_location=device,
        weights_only=True,
    )
    if payload.get("algorithm") != "residual_sac":
        raise RuntimeError("checkpoint is not a residual SAC checkpoint")
    config = SacConfig(**payload["config"])
    actor = SquashedGaussianActor(config).to(device)
    actor.load_state_dict(payload["actor"])
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor


def _collect_pair(
    candidates: dict[tuple[str, int | None], dict[str, list[torch.Tensor]]],
    baselines: dict[tuple[str, int | None], dict[str, list[torch.Tensor]]],
    key: tuple[str, int | None],
    candidate: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
) -> None:
    for metric, values in candidate.items():
        candidates[key][metric].append(values)
    for metric, values in baseline.items():
        baselines[key][metric].append(values)


def _summary_record(
    variant_id: str,
    scale: float,
    policy_seed: int | None,
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
        "variant": variant_id,
        "scale": scale,
        "policy_seed": policy_seed,
        "episodes": int(candidate["power_in_bucket"].numel()),
        "candidate": science,
        "baseline": reference,
        "paired_delta_candidate_minus_baseline": paired,
        "relative_power_gain": (
            science["power_in_bucket"]["mean"] - reference["power_in_bucket"]["mean"]
        )
        / reference["power_in_bucket"]["mean"],
        "action_diagnostics": {
            key: _distribution(candidate[key]) for key in ACTION_METRICS
        },
    }


def _policy_variant_summaries(
    candidates: dict[tuple[str, int | None], dict[str, list[torch.Tensor]]],
    baselines: dict[tuple[str, int | None], dict[str, list[torch.Tensor]]],
    variants: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    scale_by_id = {str(item["id"]): float(item["scale"]) for item in variants}
    records = []
    for (variant_id, policy_seed), values in sorted(
        candidates.items(), key=lambda item: (item[0][0], item[0][1] or -1)
    ):
        candidate = {key: torch.cat(parts) for key, parts in values.items()}
        baseline = {
            key: torch.cat(parts) for key, parts in baselines[(variant_id, policy_seed)].items()
        }
        records.append(
            _summary_record(
                variant_id,
                scale_by_id[variant_id],
                policy_seed,
                candidate,
                baseline,
            )
        )
    return records


def _group_variant_summaries(
    candidates: dict[tuple[str, int | None], dict[str, list[torch.Tensor]]],
    baselines: dict[tuple[str, int | None], dict[str, list[torch.Tensor]]],
    variants: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for variant in variants:
        variant_id = str(variant["id"])
        matching = [key for key in candidates if key[0] == variant_id]
        if not matching:
            continue
        candidate = {
            metric: torch.cat(
                [part for key in matching for part in candidates[key][metric]]
            )
            for metric in candidates[matching[0]]
        }
        baseline = {
            metric: torch.cat(
                [part for key in matching for part in baselines[key][metric]]
            )
            for metric in SCIENCE_METRICS
        }
        records.append(
            _summary_record(
                variant_id,
                float(variant["scale"]),
                None,
                candidate,
                baseline,
            )
        )
    return records


def interpret_diagnostic(
    grouped: list[dict[str, Any]],
    *,
    zero_max_abs: float,
    zero_tolerance: float,
) -> dict[str, Any]:
    """把预声明比较转换为保守的诊断标签，不作算法优越性结论。"""
    by_scale = {float(item["scale"]): item for item in grouped}
    gains = {scale: float(item["relative_power_gain"]) for scale, item in by_scale.items()}
    full = gains.get(1.0)
    half = gains.get(0.5)
    quarter = gains.get(0.25)
    flipped = gains.get(-1.0)
    smaller_outperform_full = (
        full is not None
        and half is not None
        and quarter is not None
        and half > full
        and quarter > full
    )
    sign_flip_outperforms_full = (
        full is not None and flipped is not None and flipped > full
    )
    sign_flip_positive = flipped is not None and flipped > 0
    full_record = by_scale.get(1.0)
    cancellation = (
        None
        if full_record is None
        else float(
            full_record["action_diagnostics"]["cancellation_fraction"]["mean"]
        )
    )
    cosine = (
        None
        if full_record is None
        else float(
            full_record["action_diagnostics"]["baseline_residual_cosine"]["mean"]
        )
    )
    nonzero = [item for item in grouped if float(item["scale"]) != 0.0]
    reward_power_alignment: bool | None = None
    if nonzero:
        reward_best = max(
            nonzero,
            key=lambda item: item["action_diagnostics"]["training_style_reward"]["mean"],
        )["variant"]
        power_best = max(
            nonzero,
            key=lambda item: item["candidate"]["power_in_bucket"]["mean"],
        )["variant"]
        reward_power_alignment = reward_best == power_best
    return {
        "wiring_check": {
            "status": "PASS" if zero_max_abs <= zero_tolerance else "FAIL",
            "supports_wiring_bug": zero_max_abs > zero_tolerance,
        },
        "amplitude_check": {
            "relative_power_gain_by_scale": {str(key): value for key, value in gains.items()},
            "both_smaller_scales_outperform_full_scale": smaller_outperform_full,
            "supports_excessive_action_amplitude": smaller_outperform_full,
        },
        "direction_check": {
            "full_scale_cancellation_fraction": cancellation,
            "full_scale_mean_cosine_with_baseline": cosine,
            "sign_flip_outperforms_full_scale": sign_flip_outperforms_full,
            "sign_flip_has_positive_power_gain": sign_flip_positive,
            "supports_wrong_direction": sign_flip_outperforms_full and sign_flip_positive,
        },
        "reward_metric_check": {
            "highest_mean_diagnostic_reward_also_has_highest_mean_true_power": (
                reward_power_alignment
            ),
            "note": (
                "This compares deterministic post-hoc variants only; it does not prove "
                "that the training reward is causal."
            ),
        },
        "s4d3_authorized": False,
        "retraining_authorized": False,
    }


def _scenario_record(
    *,
    policy_seed: int | None,
    variant: dict[str, Any],
    profile: HardwareProfile,
    condition: RobustnessCondition,
    candidate: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
) -> dict[str, Any]:
    return {
        "policy_seed": policy_seed,
        "variant": str(variant["id"]),
        "scale": float(variant["scale"]),
        "profile": profile.identifier,
        "physical_condition": condition.identifier,
        "episodes": int(candidate["power_in_bucket"].numel()),
        "candidate_power_mean": float(candidate["power_in_bucket"].mean()),
        "baseline_power_mean": float(baseline["power_in_bucket"].mean()),
        "power_delta_mean": float(
            (candidate["power_in_bucket"] - baseline["power_in_bucket"]).mean()
        ),
        "strehl_delta_mean": float((candidate["strehl"] - baseline["strehl"]).mean()),
        "phase_rmse_delta_mean": float(
            (candidate["phase_rmse"] - baseline["phase_rmse"]).mean()
        ),
        "violation_delta_mean": float(
            (candidate["violation_fraction"] - baseline["violation_fraction"]).mean()
        ),
        "training_style_reward_mean": float(candidate["training_style_reward"].mean()),
        "requested_residual_abs_mean_rad": float(
            candidate["requested_residual_abs_mean_rad"].mean()
        ),
        "realized_residual_abs_mean_rad": float(
            candidate["realized_residual_abs_mean_rad"].mean()
        ),
        "baseline_residual_cosine_mean": float(
            candidate["baseline_residual_cosine"].mean()
        ),
        "cancellation_fraction_mean": float(
            candidate["cancellation_fraction"].mean()
        ),
        "residual_projection_fraction_mean": float(
            candidate["residual_projection_fraction"].mean()
        ),
    }


def _write_scenario_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(json_safe(rows))


def _validate_diagnostic_seeds(
    experiment: dict[str, Any], settings: dict[str, Any]
) -> None:
    batch = int(settings["batch_size"])
    diagnostic = {
        int(item["base_seed"]) + offset
        for item in settings["physical_conditions"]
        for offset in range(batch)
    }
    if len(diagnostic) != len(settings["physical_conditions"]) * batch:
        raise RuntimeError("diagnostic physical conditions reuse episode seeds")
    for item in experiment["forbidden_seed_ranges"]:
        start = int(item["start_inclusive"])
        end = int(item["end_exclusive"])
        if any(start <= seed < end for seed in diagnostic):
            raise RuntimeError(f"diagnostic seeds overlap protected range: {item['id']}")


def _effective_settings(
    experiment: dict[str, Any], *, quick: bool
) -> dict[str, Any]:
    evaluation = experiment["evaluation"]
    settings = {
        "quick": quick,
        "output_directory": experiment["outputs"]["directory"],
        "checkpoints": deepcopy(experiment["checkpoints"]),
        "variants": deepcopy(experiment["variants"]),
        "profile_ids": list(experiment["diagnostic_profile_ids"]),
        "physical_conditions": deepcopy(experiment["diagnostic_physical_conditions"]),
        "batch_size": int(evaluation["episodes_per_physical_condition"]),
        "steps": int(evaluation["steps"]),
        "zero_equivalence_tolerance": float(
            evaluation["zero_equivalence_tolerance"]
        ),
    }
    if quick:
        quick_settings = experiment["quick"]
        allowed_seeds = set(map(int, quick_settings["policy_seeds"]))
        allowed_variants = set(map(str, quick_settings["variant_ids"]))
        settings.update(
            {
                "output_directory": experiment["outputs"]["quick_directory"],
                "checkpoints": [
                    item
                    for item in experiment["checkpoints"]
                    if int(item["policy_seed"]) in allowed_seeds
                ],
                "variants": [
                    item
                    for item in experiment["variants"]
                    if str(item["id"]) in allowed_variants
                ],
                "profile_ids": list(quick_settings["profile_ids"]),
                "physical_conditions": deepcopy(quick_settings["physical_conditions"]),
                "batch_size": int(quick_settings["batch_size"]),
                "steps": int(quick_settings["steps"]),
            }
        )
    return settings
