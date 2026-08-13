"""S4-D2-R2正式负结果后的只读动作缩放与安全投影诊断。"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import csv
from dataclasses import replace
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
    "requested_residual_l2_rad",
    "realized_residual_abs_mean_rad",
    "realized_residual_l2_rad",
    "baseline_delta_l2_rad",
    "final_delta_l2_rad",
    "baseline_residual_cosine",
    "cancellation_fraction",
    "same_direction_fraction",
    "residual_l2_projection_fraction",
    "final_projection_fraction",
    "normalized_saturation_fraction",
    "requested_modal_l2_budget_ratio",
    "final_delta_l2_budget_ratio",
)


def run_s4_r2_residual_diagnostic(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """用现有R2最优检查点做诊断，不训练也不更新权重。"""
    experiment_path = _project_path(config_path)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight = preflight_s4_r2_residual_diagnostic(
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
            "diagnostic output already exists; preserve it for audit: "
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
    profiles = _profiles(experiment, settings["profile_ids"])
    conditions = [
        RobustnessCondition.from_mapping(item)
        for item in settings["physical_conditions"]
    ]
    nonzero_variants = [
        item for item in settings["variants"] if float(item["scale"]) != 0.0
    ]
    zero_variant = next(
        item for item in settings["variants"] if float(item["scale"]) == 0.0
    )
    checkpoint_counts = {
        str(rep["id"]): sum(
            str(item["representation_id"]) == str(rep["id"])
            for item in settings["checkpoints"]
        )
        for rep in settings["representations"]
    }
    scenarios_per_representation = len(profiles) * len(conditions)
    total_rollouts = sum(
        scenarios_per_representation
        * (2 + checkpoint_counts[str(rep["id"])] * len(nonzero_variants))
        for rep in settings["representations"]
    )
    progress = counted_progress(
        total=total_rollouts,
        description="S4-D2-R2失败诊断",
        unit="轨迹组",
    )
    completed = 0
    started = time.perf_counter()

    candidates: dict[
        tuple[str, str, int | None], dict[str, list[torch.Tensor]]
    ] = defaultdict(lambda: defaultdict(list))
    baselines: dict[
        tuple[str, str, int | None], dict[str, list[torch.Tensor]]
    ] = defaultdict(lambda: defaultdict(list))
    scenario_records: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    zero_max_abs_by_representation: dict[str, float] = {}

    for representation in settings["representations"]:
        representation_id = str(representation["id"])
        num_modes = int(representation["num_modes"])
        representation_experiment = deepcopy(experiment)
        representation_experiment["active_representation"] = deepcopy(representation)
        representation_experiment["action"]["num_modes"] = num_modes
        representation_base = replace(
            base_config,
            num_modes=num_modes,
            batch_size=int(settings["batch_size"]),
            episode_length=max(base_config.episode_length, int(settings["steps"])),
        )
        configured_scenarios: list[
            tuple[HardwareProfile, RobustnessCondition, S1EnvConfig]
        ] = []
        for profile in profiles:
            for condition in conditions:
                configured_scenarios.append(
                    (
                        profile,
                        condition,
                        profile.environment_config(
                            condition.environment_config(representation_base)
                        ),
                    )
                )

        baseline_cache: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
        zero_max_abs = 0.0
        for profile, condition, config in configured_scenarios:
            scenario_key = (profile.identifier, condition.identifier)
            baseline = _rollout_anchored_baseline(
                representation_experiment,
                config,
                condition,
                profile,
                int(settings["steps"]),
                device,
            )
            baseline_cache[scenario_key] = baseline
            completed += 1
            advance_to(progress, completed)
            update_progress(
                progress,
                device=device,
                metrics={
                    "表示维数": float(num_modes),
                    "基线功率": float(baseline["power_in_bucket"].mean()),
                },
            )

            zero = _rollout_variant(
                actor=None,
                scale=0.0,
                experiment=representation_experiment,
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
            key = (representation_id, str(zero_variant["id"]), None)
            _collect_pair(candidates, baselines, key, zero, baseline)
            scenario_records.append(
                _scenario_record(
                    representation_id=representation_id,
                    num_modes=num_modes,
                    policy_seed=None,
                    variant=zero_variant,
                    profile=profile,
                    condition=condition,
                    candidate=zero,
                    baseline=baseline,
                )
            )
            episode_records.extend(
                _episode_rows(
                    representation_id=representation_id,
                    num_modes=num_modes,
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
        zero_max_abs_by_representation[representation_id] = zero_max_abs

        representation_checkpoints = [
            item
            for item in settings["checkpoints"]
            if str(item["representation_id"]) == representation_id
        ]
        for checkpoint in representation_checkpoints:
            policy_seed = int(checkpoint["policy_seed"])
            actor = _load_actor(checkpoint, device)
            for variant in nonzero_variants:
                key = (representation_id, str(variant["id"]), policy_seed)
                for profile, condition, config in configured_scenarios:
                    baseline = baseline_cache[
                        (profile.identifier, condition.identifier)
                    ]
                    candidate = _rollout_variant(
                        actor=actor,
                        scale=float(variant["scale"]),
                        experiment=representation_experiment,
                        config=config,
                        condition=condition,
                        profile=profile,
                        steps=int(settings["steps"]),
                        device=device,
                    )
                    _collect_pair(
                        candidates,
                        baselines,
                        key,
                        candidate,
                        baseline,
                    )
                    scenario_records.append(
                        _scenario_record(
                            representation_id=representation_id,
                            num_modes=num_modes,
                            policy_seed=policy_seed,
                            variant=variant,
                            profile=profile,
                            condition=condition,
                            candidate=candidate,
                            baseline=baseline,
                        )
                    )
                    episode_records.extend(
                        _episode_rows(
                            representation_id=representation_id,
                            num_modes=num_modes,
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
                            "表示维数": float(num_modes),
                            "策略种子": float(policy_seed),
                            "缩放": float(variant["scale"]),
                            "功率差": float(
                                (
                                    candidate["power_in_bucket"]
                                    - baseline["power_in_bucket"]
                                ).mean()
                            ),
                        },
                    )
            del actor
    progress.close()

    policy_variant_summaries = _policy_variant_summaries(
        candidates,
        baselines,
        settings["variants"],
    )
    grouped_variant_summaries = _group_variant_summaries(
        candidates,
        baselines,
        settings["variants"],
    )
    representation_comparisons = _representation_comparisons(
        candidates,
        baselines,
        settings["variants"],
    )
    diagnosis = interpret_r2_diagnostic(
        grouped_variant_summaries,
        zero_max_abs_by_representation=zero_max_abs_by_representation,
        zero_tolerance=float(settings["zero_equivalence_tolerance"]),
        projection_warning_fraction=float(settings["projection_warning_fraction"]),
        violation_limit=float(settings["violation_limit"]),
    )
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run-diagnostic",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "s4d2_r2_action_diagnostic_v1",
        },
        "experiment": {
            "id": "AO-S4-D2-R2-FAILURE-DIAGNOSTIC",
            "type": "software_only_cuda_post_hoc_read_only_diagnostic",
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
            "prior_trajectories_reused": False,
            "s4d3_accessed": False,
            "real_slm_actions": False,
            "post_hoc_diagnostic_only": True,
            "interpretation": (
                "Action scaling, sign reversal and projection telemetry diagnose an "
                "existing failure; they are not a new trained algorithm or S4-D3 evidence."
            ),
        },
        "inputs": {
            "diagnostic_config": _relative(experiment_path),
            "diagnostic_config_sha256": _file_sha256(experiment_path),
            "upstream_r2_summary_sha256": preflight["upstream_r2_summary_sha256"],
            "checkpoint_count": len(settings["checkpoints"]),
        },
        "design": {
            "paired_episode_seeds": True,
            "deterministic_policy": True,
            "batch_size": int(settings["batch_size"]),
            "steps": int(settings["steps"]),
            "representations": deepcopy(settings["representations"]),
            "profiles": list(settings["profile_ids"]),
            "physical_conditions": deepcopy(settings["physical_conditions"]),
            "variants": deepcopy(settings["variants"]),
        },
        "zero_residual_equivalence": {
            "max_absolute_science_metric_difference_by_representation": (
                zero_max_abs_by_representation
            ),
            "tolerance": float(settings["zero_equivalence_tolerance"]),
            "status": (
                "PASS"
                if all(
                    value <= float(settings["zero_equivalence_tolerance"])
                    for value in zero_max_abs_by_representation.values()
                )
                else "FAIL"
            ),
        },
        "policy_variant_summaries": policy_variant_summaries,
        "grouped_variant_summaries": grouped_variant_summaries,
        "representation_comparisons": representation_comparisons,
        "diagnosis": diagnosis,
        "record_counts": {
            "scenario_records": len(scenario_records),
            "episode_records": len(episode_records),
        },
        "runtime": _runtime_record(),
        "git": _git_record(),
        "next_action": (
            "Stop and ask the assistant to audit the diagnostic. Do not retrain, "
            "change algorithms, open S4-D3, or operate real SLM hardware."
        ),
    }
    _write_json(output_directory / "summary.json", summary)
    _write_csv(output_directory / "scenario_records.csv", scenario_records)
    _write_csv(output_directory / "episode_records.csv", episode_records)
    return summary


def preflight_s4_r2_residual_diagnostic(
    experiment_path: Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    """在创建输出和占用CUDA前核对R2负结果与只读边界。"""
    metadata = experiment["metadata"]
    if str(metadata["stage"]) != "S4-D2-R2-DIAGNOSTIC":
        raise ValueError("diagnostic metadata must identify S4-D2-R2-DIAGNOSTIC")
    forbidden_flags = (
        "allow_training",
        "allow_checkpoint_updates",
        "allow_prior_trajectory_reuse",
        "allow_s4d3_access",
        "allow_real_hardware_actions",
        "allow_algorithm_change",
        "allow_reward_change",
        "allow_gate_relaxation",
    )
    if any(bool(metadata.get(field, True)) for field in forbidden_flags):
        raise RuntimeError("all diagnostic mutation and scope-expansion flags must be false")

    upstream = experiment["upstream_r2"]
    upstream_fields = (
        "summary",
        "preflight",
        "source_manifest",
        "effective_config",
        "experiment_config",
        "audit_record",
    )
    upstream_paths = {
        field: _project_path(upstream[field]) for field in upstream_fields
    }
    for field, path in upstream_paths.items():
        if _file_sha256(path) != str(upstream[f"{field}_sha256"]):
            raise RuntimeError(f"R2 evidence hash mismatch: {field}")
    audit_text = upstream_paths["audit_record"].read_text(encoding="utf-8")
    if "Verification Status: `ANALYZED`" not in audit_text or "**FAIL**" not in audit_text:
        raise RuntimeError("R2 audit must be ANALYZED and record a FAIL")

    training_summary = json.loads(
        upstream_paths["summary"].read_text(encoding="utf-8")
    )
    if (
        bool(training_summary["experiment"]["quick"])
        or training_summary["experiment"]["status"] != "completed_pending_audit"
        or training_summary["development_summary"]["status"] != "ANALYSIS_REQUIRED"
        or training_summary["development_summary"]["selected_representation"] is not None
        or bool(training_summary["development_summary"]["s4d3_authorized"])
    ):
        raise RuntimeError("R2 formal failure state or S4-D3 lock changed")

    recorded_source = json.loads(
        upstream_paths["source_manifest"].read_text(encoding="utf-8")
    )
    if recorded_source != training_summary["inputs"]["source_manifest"]:
        raise RuntimeError("R2 source manifests disagree")
    for relative, digest in recorded_source.items():
        path = _project_path(relative)
        if not path.is_file() or _file_sha256(path) != digest:
            raise RuntimeError(f"R2 training source changed: {relative}")

    training_config = _load_yaml(upstream_paths["experiment_config"])
    if training_summary["inputs"]["experiment_config_sha256"] != _file_sha256(
        upstream_paths["experiment_config"]
    ):
        raise RuntimeError("R2 summary and experiment configuration hash disagree")
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
            raise RuntimeError(f"diagnostic changed frozen R2 field: {field}")
    expected_representations = [
        {
            "id": str(item["id"]),
            "kind": str(item["kind"]),
            "num_modes": int(item["num_modes"]),
        }
        for item in training_config["representations"]
    ]
    if list(experiment["representations"]) != expected_representations:
        raise RuntimeError("diagnostic representation contract differs from R2")

    summary_by_representation = {
        str(item["representation"]["id"]): item
        for item in training_summary["representation_results"]
    }
    if set(summary_by_representation) != {"zernike_21", "zernike_36"}:
        raise RuntimeError("R2 summary no longer contains the two declared representations")
    for item in summary_by_representation.values():
        if (
            int(item["development_summary"]["pass_count"]) != 0
            or bool(item["development_summary"]["s4d3_authorized"])
        ):
            raise RuntimeError("an R2 representation no longer records the audited failure")

    checkpoint_contracts: list[dict[str, Any]] = []
    for checkpoint in experiment["checkpoints"]:
        representation_id = str(checkpoint["representation_id"])
        policy_seed = int(checkpoint["policy_seed"])
        num_modes = int(checkpoint["num_modes"])
        result_by_seed = {
            int(item["policy_seed"]): item
            for item in summary_by_representation[representation_id][
                "policy_seed_results"
            ]
        }
        if policy_seed not in result_by_seed:
            raise RuntimeError(f"unknown R2 policy seed: {policy_seed}")
        result = result_by_seed[policy_seed]
        path = _project_path(checkpoint["path"])
        if result["best_checkpoint"] != _relative(path):
            raise RuntimeError("diagnostic checkpoint is not the recorded R2 best actor")
        if _file_sha256(path) != str(checkpoint["sha256"]):
            raise RuntimeError("R2 best checkpoint hash mismatch")
        episode_record = result["final_episode_records"]
        episode_path = _project_path(episode_record["path"])
        if (
            int(episode_record["rows"]) != 288
            or _file_sha256(episode_path) != str(episode_record["sha256"])
        ):
            raise RuntimeError("R2 final per-episode evidence changed")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("algorithm") != "residual_sac":
            raise RuntimeError("R2 checkpoint is not residual SAC")
        actor_config = SacConfig(**payload["config"])
        if (
            actor_config.state_size != 10 * num_modes
            or actor_config.action_size != num_modes
            or actor_config.hidden_size != int(training_config["sac"]["hidden_size"])
        ):
            raise RuntimeError("R2 checkpoint dimensions differ from the declared representation")
        if not all(torch.isfinite(value).all() for value in payload["actor"].values()):
            raise RuntimeError("R2 actor checkpoint contains non-finite tensors")
        checkpoint_contracts.append(
            {
                "representation_id": representation_id,
                "policy_seed": policy_seed,
                "state_size": actor_config.state_size,
                "action_size": actor_config.action_size,
            }
        )

    representation_ids = [str(item["id"]) for item in settings["representations"]]
    if len(representation_ids) != len(set(representation_ids)):
        raise ValueError("diagnostic representation ids must be unique")
    checkpoint_keys = [
        (str(item["representation_id"]), int(item["policy_seed"]))
        for item in settings["checkpoints"]
    ]
    if len(checkpoint_keys) != len(set(checkpoint_keys)):
        raise ValueError("diagnostic checkpoints must be unique by representation and seed")
    if not quick:
        expected_keys = {
            (representation_id, seed)
            for representation_id in ("zernike_21", "zernike_36")
            for seed in (7101, 7102, 7103)
        }
        if set(checkpoint_keys) != expected_keys:
            raise RuntimeError("formal diagnostic must use all six audited R2 best actors")

    variants = settings["variants"]
    variant_ids = [str(item["id"]) for item in variants]
    scales = [float(item["scale"]) for item in variants]
    if len(variant_ids) != len(set(variant_ids)) or any(abs(value) > 1 for value in scales):
        raise ValueError("diagnostic variants must be unique and stay within [-1, 1]")
    if scales.count(0.0) != 1:
        raise ValueError("diagnostic requires exactly one zero-residual variant")
    if not quick and set(scales) != {0.0, 0.25, 0.5, 1.0, -1.0}:
        raise ValueError("formal diagnostic requires zero, quarter, half, full and reverse")

    _validate_diagnostic_seeds(experiment, settings)
    environment_path = _project_path(experiment["environment_config"])
    if _file_sha256(environment_path) != str(experiment["environment_config_sha256"]):
        raise RuntimeError("environment configuration hash mismatch")
    hardware_path = _project_path(experiment["hardware_profile_source"])
    if _file_sha256(hardware_path) != str(experiment["hardware_profile_source_sha256"]):
        raise RuntimeError("hardware profile source hash mismatch")
    base_config, _ = load_s1_config(environment_path)
    for representation in settings["representations"]:
        representation_config = replace(
            base_config, num_modes=int(representation["num_modes"])
        )
        representation_experiment = deepcopy(experiment)
        representation_experiment["action"]["num_modes"] = int(
            representation["num_modes"]
        )
        controller = _make_residual_controller(
            representation_experiment,
            representation_config,
        )
        if controller.state_size != 10 * int(representation["num_modes"]):
            raise RuntimeError("diagnostic controller state contract is inconsistent")

    if not all(_project_path(path).is_file() for path in experiment["tracked_source_files"]):
        raise FileNotFoundError("one or more diagnostic tracked source files are missing")
    output_directory = _project_path(settings["output_directory"])
    if output_directory.exists():
        raise FileExistsError(f"diagnostic output already exists: {output_directory}")

    scenario_count = len(settings["profile_ids"]) * len(settings["physical_conditions"])
    nonzero_count = sum(float(item["scale"]) != 0.0 for item in settings["variants"])
    rollout_count = sum(
        scenario_count
        * (
            2
            + sum(
                str(item["representation_id"]) == str(rep["id"])
                for item in settings["checkpoints"]
            )
            * nonzero_count
        )
        for rep in settings["representations"]
    )
    return {
        "status": "READY_FOR_QUICK_DIAGNOSTIC" if quick else "READY_FOR_USER_DIAGNOSTIC",
        "quick": quick,
        "experiment_config": _relative(experiment_path),
        "experiment_config_sha256": _file_sha256(experiment_path),
        "output_directory": _relative(output_directory),
        "upstream_r2_gate": "FAIL",
        "upstream_r2_summary_sha256": _file_sha256(upstream_paths["summary"]),
        "representations": representation_ids,
        "checkpoint_contracts": checkpoint_contracts,
        "checkpoint_count": len(settings["checkpoints"]),
        "variant_count": len(settings["variants"]),
        "scenario_count_per_representation": scenario_count,
        "rollout_count": rollout_count,
        "episode_count_per_rollout": int(settings["batch_size"]),
        "steps_per_episode": int(settings["steps"]),
        "seed_isolation_verified": True,
        "cuda_required": True,
        "training_allowed": False,
        "optimizer_updates": 0,
        "checkpoints_are_read_only": True,
        "prior_trajectories_reused": False,
        "s4d3_accessed": False,
        "real_slm_actions": False,
        "formal_command": (
            ".\\.venv\\Scripts\\python.exe scripts\\diagnose_s4_r2_residual_sac.py "
            "--config configs\\experiments\\s4_residual_sac_r2_diagnostic_v1.yaml"
        ),
    }


@torch.no_grad()
def _rollout_anchored_baseline(
    experiment: dict[str, Any],
    config: S1EnvConfig,
    condition: RobustnessCondition,
    profile: HardwareProfile,
    steps: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """运行独立的10维传统控制器，并把动作零填充到当前表示维数。"""
    environment = AdaptiveOpticsEnv(config, device, profile.effects_config())
    observation, _ = environment.reset(seed=condition.base_seed)
    generator = torch.Generator(device=device).manual_seed(
        condition.base_seed + 40_000_000
    )
    observation = _noisy_observation(
        observation,
        config.num_modes,
        profile.observation_noise_std_rad,
        generator,
    )
    anchor_modes = int(experiment["frozen_controller"]["active_anchor_modes"])
    parameters = experiment["frozen_controller"]["parameters"]
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
        anchor_action = controller.action(
            _anchor_observation(observation, config.num_modes, anchor_modes)
        )
        action = torch.zeros(
            config.batch_size,
            config.num_modes,
            device=device,
            dtype=observation.dtype,
        )
        action[:, :anchor_modes] = anchor_action
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


@torch.no_grad()
def _rollout_variant(
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
    generator = torch.Generator(device=device).manual_seed(
        condition.base_seed + 40_000_000
    )
    noisy = _noisy_observation(
        observation,
        config.num_modes,
        profile.observation_noise_std_rad,
        generator,
    )
    controller = _make_residual_controller(experiment, config)
    state = controller.reset(noisy)
    science = _empty_step_metrics()
    actions: dict[str, list[torch.Tensor]] = {key: [] for key in ACTION_METRICS}
    residual_limit = float(experiment["action"]["residual_action_limit_rad"])
    anchor_modes = int(experiment["frozen_controller"]["active_anchor_modes"])
    residual_l2_budget = anchor_modes**0.5 * residual_limit
    final_l2_budget = (
        anchor_modes**0.5
        * float(experiment["action"]["final_action_step_limit_rad"])
    )
    request_l2_budget = anchor_modes**0.5 * float(config.modal_limit_rad)
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
        unprojected_residual = scaled * residual_limit
        action = controller.compose_action(scaled)
        requested_modal = controller.requested_modal.clone()
        observation, _, _, _, info = environment.step(action.final_delta_rad)
        _append_step_metrics(science, info)

        baseline = action.baseline_delta_rad
        requested = action.requested_residual_rad
        realized = action.realized_residual_rad
        denominator = baseline.norm(dim=-1) * requested.norm(dim=-1)
        cosine = torch.where(
            denominator > 1e-12,
            (baseline * requested).sum(dim=-1) / denominator.clamp_min(1e-12),
            torch.zeros_like(denominator),
        )
        dot = (baseline * requested).sum(dim=-1)
        residual_projected = (
            requested.sub(unprojected_residual).abs().amax(dim=-1).gt(1e-7).float()
        )
        nominal_final = baseline + requested
        final_projected = (
            action.final_delta_rad.sub(nominal_final)
            .abs()
            .amax(dim=-1)
            .gt(1e-7)
            .float()
        )
        actions["training_style_reward"].append(
            _residual_reward(
                measured_power=info["measured_power_in_bucket"],
                normalized_residual=scaled,
                violation=info["violation_fraction"],
                reward_config=experiment["reward"],
            )
            .detach()
            .cpu()
        )
        actions["raw_normalized_abs_mean"].append(raw.abs().mean(dim=-1).cpu())
        actions["scaled_normalized_abs_mean"].append(
            scaled.abs().mean(dim=-1).cpu()
        )
        actions["requested_residual_abs_mean_rad"].append(
            requested.abs().mean(dim=-1).cpu()
        )
        actions["requested_residual_l2_rad"].append(
            requested.norm(dim=-1).cpu()
        )
        actions["realized_residual_abs_mean_rad"].append(
            realized.abs().mean(dim=-1).cpu()
        )
        actions["realized_residual_l2_rad"].append(realized.norm(dim=-1).cpu())
        actions["baseline_delta_l2_rad"].append(baseline.norm(dim=-1).cpu())
        actions["final_delta_l2_rad"].append(
            action.final_delta_rad.norm(dim=-1).cpu()
        )
        actions["baseline_residual_cosine"].append(cosine.cpu())
        actions["cancellation_fraction"].append((dot < 0).float().cpu())
        actions["same_direction_fraction"].append((dot > 0).float().cpu())
        actions["residual_l2_projection_fraction"].append(
            residual_projected.cpu()
        )
        actions["final_projection_fraction"].append(final_projected.cpu())
        actions["normalized_saturation_fraction"].append(
            scaled.abs().ge(0.999).float().mean(dim=-1).cpu()
        )
        actions["requested_modal_l2_budget_ratio"].append(
            requested_modal.norm(dim=-1).div(request_l2_budget).cpu()
        )
        actions["final_delta_l2_budget_ratio"].append(
            action.final_delta_rad.norm(dim=-1).div(final_l2_budget).cpu()
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
    if any(
        not torch.isfinite(value).all()
        for value in result.values()
    ):
        raise RuntimeError("diagnostic rollout produced non-finite metrics")
    if float(result["requested_residual_l2_rad"].max()) > residual_l2_budget + 1e-5:
        raise RuntimeError("diagnostic residual exceeded the shared L2 budget")
    return result


def _anchor_observation(
    observation: torch.Tensor,
    num_modes: int,
    anchor_modes: int,
) -> torch.Tensor:
    return torch.cat(
        (
            observation[:, :anchor_modes],
            observation[:, num_modes : num_modes + anchor_modes],
            observation[:, -2:],
        ),
        dim=-1,
    )


def _load_actor(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> SquashedGaussianActor:
    payload = torch.load(
        _project_path(checkpoint["path"]),
        map_location=device,
        weights_only=True,
    )
    if payload.get("algorithm") != "residual_sac":
        raise RuntimeError("checkpoint is not a residual SAC checkpoint")
    config = SacConfig(**payload["config"])
    if (
        config.state_size != 10 * int(checkpoint["num_modes"])
        or config.action_size != int(checkpoint["num_modes"])
    ):
        raise RuntimeError("checkpoint actor dimensions do not match its representation")
    actor = SquashedGaussianActor(config).to(device)
    actor.load_state_dict(payload["actor"])
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor


def _collect_pair(
    candidates: dict[
        tuple[str, str, int | None], dict[str, list[torch.Tensor]]
    ],
    baselines: dict[
        tuple[str, str, int | None], dict[str, list[torch.Tensor]]
    ],
    key: tuple[str, str, int | None],
    candidate: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
) -> None:
    for metric, values in candidate.items():
        candidates[key][metric].append(values)
    for metric, values in baseline.items():
        baselines[key][metric].append(values)


def _summary_record(
    representation_id: str,
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
    }
    baseline_power = reference["power_in_bucket"]["mean"]
    return {
        "representation_id": representation_id,
        "variant": variant_id,
        "scale": scale,
        "policy_seed": policy_seed,
        "episodes": int(candidate["power_in_bucket"].numel()),
        "candidate": science,
        "baseline": reference,
        "paired_delta_candidate_minus_baseline": paired,
        "relative_power_gain": (
            science["power_in_bucket"]["mean"] - baseline_power
        )
        / baseline_power,
        "action_diagnostics": {
            key: _distribution(candidate[key]) for key in ACTION_METRICS
        },
    }


def _policy_variant_summaries(
    candidates: dict[
        tuple[str, str, int | None], dict[str, list[torch.Tensor]]
    ],
    baselines: dict[
        tuple[str, str, int | None], dict[str, list[torch.Tensor]]
    ],
    variants: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    scale_by_id = {str(item["id"]): float(item["scale"]) for item in variants}
    records: list[dict[str, Any]] = []
    for key in sorted(
        candidates,
        key=lambda item: (item[0], item[1], item[2] if item[2] is not None else -1),
    ):
        representation_id, variant_id, policy_seed = key
        candidate = {
            metric: torch.cat(parts) for metric, parts in candidates[key].items()
        }
        baseline = {
            metric: torch.cat(parts) for metric, parts in baselines[key].items()
        }
        records.append(
            _summary_record(
                representation_id,
                variant_id,
                scale_by_id[variant_id],
                policy_seed,
                candidate,
                baseline,
            )
        )
    return records


def _group_variant_summaries(
    candidates: dict[
        tuple[str, str, int | None], dict[str, list[torch.Tensor]]
    ],
    baselines: dict[
        tuple[str, str, int | None], dict[str, list[torch.Tensor]]
    ],
    variants: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    representation_ids = sorted({key[0] for key in candidates})
    for representation_id in representation_ids:
        for variant in variants:
            variant_id = str(variant["id"])
            matching = [
                key
                for key in candidates
                if key[0] == representation_id and key[1] == variant_id
            ]
            if not matching:
                continue
            first = matching[0]
            candidate = {
                metric: torch.cat(
                    [part for key in matching for part in candidates[key][metric]]
                )
                for metric in candidates[first]
            }
            baseline = {
                metric: torch.cat(
                    [part for key in matching for part in baselines[key][metric]]
                )
                for metric in SCIENCE_METRICS
            }
            records.append(
                _summary_record(
                    representation_id,
                    variant_id,
                    float(variant["scale"]),
                    None,
                    candidate,
                    baseline,
                )
            )
    return records


def _representation_comparisons(
    candidates: dict[
        tuple[str, str, int | None], dict[str, list[torch.Tensor]]
    ],
    baselines: dict[
        tuple[str, str, int | None], dict[str, list[torch.Tensor]]
    ],
    variants: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按相同策略种子和相同回合比较36维与21维，不把均值误当配对数据。"""
    records: list[dict[str, Any]] = []
    for variant in variants:
        variant_id = str(variant["id"])
        keys_21 = {
            key[2]: key
            for key in candidates
            if key[0] == "zernike_21" and key[1] == variant_id
        }
        keys_36 = {
            key[2]: key
            for key in candidates
            if key[0] == "zernike_36" and key[1] == variant_id
        }
        shared_seeds = sorted(
            set(keys_21).intersection(keys_36),
            key=lambda value: value if value is not None else -1,
        )
        if not shared_seeds:
            continue
        candidate_differences: list[torch.Tensor] = []
        baseline_differences: list[torch.Tensor] = []
        improvement_differences: list[torch.Tensor] = []
        for seed in shared_seeds:
            key_21 = keys_21[seed]
            key_36 = keys_36[seed]
            candidate_21 = torch.cat(candidates[key_21]["power_in_bucket"])
            candidate_36 = torch.cat(candidates[key_36]["power_in_bucket"])
            baseline_21 = torch.cat(baselines[key_21]["power_in_bucket"])
            baseline_36 = torch.cat(baselines[key_36]["power_in_bucket"])
            if not (
                candidate_21.shape
                == candidate_36.shape
                == baseline_21.shape
                == baseline_36.shape
            ):
                raise RuntimeError("representation comparison lost paired episode alignment")
            candidate_differences.append(candidate_36 - candidate_21)
            baseline_differences.append(baseline_36 - baseline_21)
            improvement_differences.append(
                (candidate_36 - baseline_36) - (candidate_21 - baseline_21)
            )
        records.append(
            {
                "variant": variant_id,
                "scale": float(variant["scale"]),
                "policy_seeds": shared_seeds,
                "episodes": int(sum(item.numel() for item in candidate_differences)),
                "candidate_power_delta_36d_minus_21d": _distribution(
                    torch.cat(candidate_differences)
                ),
                "baseline_power_delta_36d_minus_21d": _distribution(
                    torch.cat(baseline_differences)
                ),
                "paired_improvement_delta_36d_minus_21d": _distribution(
                    torch.cat(improvement_differences)
                ),
            }
        )
    return records


def interpret_r2_diagnostic(
    grouped: list[dict[str, Any]],
    *,
    zero_max_abs_by_representation: dict[str, float],
    zero_tolerance: float,
    projection_warning_fraction: float,
    violation_limit: float,
) -> dict[str, Any]:
    """生成保守的原因标签；诊断本身不授权重新训练或更换算法。"""
    results: dict[str, Any] = {}
    for representation_id in sorted({str(item["representation_id"]) for item in grouped}):
        records = [
            item for item in grouped if item["representation_id"] == representation_id
        ]
        by_scale = {float(item["scale"]): item for item in records}
        gains = {
            scale: float(item["relative_power_gain"])
            for scale, item in by_scale.items()
        }
        full = gains.get(1.0)
        half = gains.get(0.5)
        quarter = gains.get(0.25)
        reverse = gains.get(-1.0)
        smaller_outperform_full = (
            full is not None
            and half is not None
            and quarter is not None
            and half > full
            and quarter > full
        )
        best_smaller_positive = (
            half is not None
            and quarter is not None
            and max(half, quarter) > 0
        )
        reverse_outperforms_full = (
            reverse is not None and full is not None and reverse > full
        )
        reverse_positive = reverse is not None and reverse > 0
        full_record = by_scale.get(1.0)
        projection_fraction = (
            None
            if full_record is None
            else float(
                full_record["action_diagnostics"]["final_projection_fraction"][
                    "mean"
                ]
            )
        )
        residual_projection_fraction = (
            None
            if full_record is None
            else float(
                full_record["action_diagnostics"][
                    "residual_l2_projection_fraction"
                ]["mean"]
            )
        )
        violation = (
            None
            if full_record is None
            else float(full_record["candidate"]["violation_fraction"]["mean"])
        )
        projection_conflict = (
            projection_fraction is not None
            and residual_projection_fraction is not None
            and max(projection_fraction, residual_projection_fraction)
            > projection_warning_fraction
        )
        wiring_failed = (
            zero_max_abs_by_representation[representation_id] > zero_tolerance
        )
        if wiring_failed:
            primary_label = "WIRING_OR_BASELINE_MISMATCH_SUSPECTED"
        elif reverse_outperforms_full and reverse_positive:
            primary_label = "ACTION_DIRECTION_MISMATCH_SUPPORTED"
        elif smaller_outperform_full and best_smaller_positive:
            primary_label = "EXCESSIVE_ACTION_AMPLITUDE_SUPPORTED"
        elif projection_conflict:
            primary_label = "SAFETY_PROJECTION_CONFLICT_SUPPORTED"
        elif smaller_outperform_full:
            primary_label = "AMPLITUDE_SENSITIVITY_WITHOUT_POSITIVE_GAIN"
        else:
            primary_label = "CAUSE_NOT_ISOLATED"
        results[representation_id] = {
            "primary_label": primary_label,
            "wiring_check": {
                "status": "FAIL" if wiring_failed else "PASS",
                "zero_max_absolute_difference": zero_max_abs_by_representation[
                    representation_id
                ],
                "supports_wiring_bug": wiring_failed,
            },
            "amplitude_check": {
                "relative_power_gain_by_scale": {
                    str(key): value for key, value in gains.items()
                },
                "both_smaller_scales_outperform_full_scale": smaller_outperform_full,
                "a_smaller_scale_has_positive_power_gain": best_smaller_positive,
                "supports_excessive_action_amplitude": (
                    smaller_outperform_full and best_smaller_positive
                ),
            },
            "direction_check": {
                "reverse_outperforms_full_scale": reverse_outperforms_full,
                "reverse_has_positive_power_gain": reverse_positive,
                "supports_wrong_direction": (
                    reverse_outperforms_full and reverse_positive
                ),
            },
            "projection_and_safety_check": {
                "full_scale_residual_l2_projection_fraction": (
                    residual_projection_fraction
                ),
                "full_scale_final_projection_fraction": projection_fraction,
                "projection_warning_fraction": projection_warning_fraction,
                "full_scale_violation_fraction": violation,
                "violation_limit": violation_limit,
                "supports_projection_conflict": projection_conflict,
                "safety_limit_failed": violation is not None and violation > violation_limit,
            },
        }
    return {
        "by_representation": results,
        "retraining_authorized": False,
        "algorithm_change_authorized": False,
        "s4d3_authorized": False,
        "real_hardware_authorized": False,
        "note": (
            "Formal diagnostic results require an independent audit before choosing "
            "a code fix, reward revision or algorithm revision."
        ),
    }


def _scenario_record(
    *,
    representation_id: str,
    num_modes: int,
    policy_seed: int | None,
    variant: dict[str, Any],
    profile: HardwareProfile,
    condition: RobustnessCondition,
    candidate: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "representation_id": representation_id,
        "num_modes": num_modes,
        "policy_seed": policy_seed,
        "variant": str(variant["id"]),
        "scale": float(variant["scale"]),
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
    for metric in ACTION_METRICS:
        row[f"action_{metric}_mean"] = float(candidate[metric].mean())
    return row


def _episode_rows(
    *,
    representation_id: str,
    num_modes: int,
    policy_seed: int | None,
    variant: dict[str, Any],
    profile: HardwareProfile,
    condition: RobustnessCondition,
    candidate: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    count = int(candidate["power_in_bucket"].numel())
    for index in range(count):
        row: dict[str, Any] = {
            "representation_id": representation_id,
            "num_modes": num_modes,
            "policy_seed": policy_seed,
            "variant": str(variant["id"]),
            "scale": float(variant["scale"]),
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
        for metric in ACTION_METRICS:
            row[f"action_{metric}"] = float(candidate[metric][index])
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty diagnostic CSV: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(json_safe(rows))


def _validate_diagnostic_seeds(
    experiment: dict[str, Any],
    settings: dict[str, Any],
) -> None:
    batch_size = int(settings["batch_size"])
    seed_sets = [
        {
            int(item["base_seed"]) + offset
            for offset in range(batch_size)
        }
        for item in settings["physical_conditions"]
    ]
    if any(
        left.intersection(right)
        for index, left in enumerate(seed_sets)
        for right in seed_sets[index + 1 :]
    ):
        raise RuntimeError("diagnostic physical conditions reuse episode seeds")
    active = set().union(*seed_sets)
    for item in experiment["forbidden_seed_ranges"]:
        start = int(item["start_inclusive"])
        end = int(item["end_exclusive"])
        if any(start <= seed < end for seed in active):
            raise RuntimeError(f"diagnostic seeds overlap protected range: {item['id']}")


def _effective_settings(
    experiment: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    evaluation = experiment["evaluation"]
    settings = {
        "quick": quick,
        "output_directory": experiment["outputs"]["directory"],
        "representations": deepcopy(experiment["representations"]),
        "checkpoints": deepcopy(experiment["checkpoints"]),
        "variants": deepcopy(experiment["variants"]),
        "profile_ids": list(experiment["diagnostic_profile_ids"]),
        "physical_conditions": deepcopy(experiment["diagnostic_physical_conditions"]),
        "batch_size": int(evaluation["episodes_per_physical_condition"]),
        "steps": int(evaluation["steps"]),
        "zero_equivalence_tolerance": float(
            evaluation["zero_equivalence_tolerance"]
        ),
        "projection_warning_fraction": float(
            evaluation["projection_warning_fraction"]
        ),
        "violation_limit": float(evaluation["violation_limit"]),
    }
    if quick:
        quick_settings = experiment["quick"]
        representation_ids = set(map(str, quick_settings["representation_ids"]))
        policy_seeds = set(map(int, quick_settings["policy_seeds"]))
        variant_ids = set(map(str, quick_settings["variant_ids"]))
        settings.update(
            {
                "output_directory": experiment["outputs"]["quick_directory"],
                "representations": [
                    item
                    for item in experiment["representations"]
                    if str(item["id"]) in representation_ids
                ],
                "checkpoints": [
                    item
                    for item in experiment["checkpoints"]
                    if str(item["representation_id"]) in representation_ids
                    and int(item["policy_seed"]) in policy_seeds
                ],
                "variants": [
                    item
                    for item in experiment["variants"]
                    if str(item["id"]) in variant_ids
                ],
                "profile_ids": list(quick_settings["profile_ids"]),
                "physical_conditions": deepcopy(
                    quick_settings["physical_conditions"]
                ),
                "batch_size": int(quick_settings["batch_size"]),
                "steps": int(quick_settings["steps"]),
            }
        )
    return settings
