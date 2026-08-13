"""S4-D2-R2新增11维高阶动作的观测可学习性诊断。"""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.rl.residual_control import AnchoredResidualTrackingController
from src.rl.s4_high_order_capacity import AddedModesOracleController
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
from src.training_progress import advance_to, counted_progress, progress_bar, update_progress


Predictor = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class TeacherDataset:
    """同一完整回合拆分内的可部署状态与新增11维教师动作。"""

    states: torch.Tensor
    targets: torch.Tensor
    episodes: int
    steps_per_episode: int


@dataclass(frozen=True)
class TeacherScenario:
    """诊断测试中一个硬件档位和动态条件的配对教师结果。"""

    profile_id: str
    condition_id: str
    base_seed: int
    candidate: dict[str, torch.Tensor]
    baseline: dict[str, torch.Tensor]


class HighOrderImitationPolicy(nn.Module):
    """与旧SAC演员主干同宽的两层MLP，只输出新增11维。"""

    def __init__(self, state_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        if min(state_size, hidden_size, output_size) <= 0:
            raise ValueError("model dimensions must be positive")
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


def run_s4_high_order_learnability(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """生成教师数据、训练监督模型，并在新完整回合中闭环诊断。"""
    experiment_path = _project_path(config_path)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight = preflight_s4_high_order_learnability(
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
            "high-order-learnability output already exists; preserve it for audit: "
            f"{output_directory}"
        )
    checkpoint_directory = output_directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True)
    _write_json(output_directory / "preflight.json", preflight)
    _write_json(output_directory / "effective_config.json", settings)
    _write_json(
        output_directory / "source_manifest.json",
        _source_manifest(experiment["tracked_source_files"]),
    )

    started = time.perf_counter()
    base_config, _ = load_s1_config(_project_path(experiment["environment_config"]))
    representation = ActionRepresentation.from_mapping(experiment["representation"])
    base_config = replace(base_config, num_modes=representation.num_modes)
    basis, pupil, basis_diagnostics = build_action_basis(
        base_config,
        representation,
        device,
    )
    profiles = _profiles(experiment, settings["profile_ids"])
    registration_mappings: dict[str, torch.Tensor] = {}
    mapping_diagnostics: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        mapping, diagnostics = representation_registration_inverse(
            basis=basis,
            pupil=pupil,
            profile=profile,
            rcond=float(experiment["upstream_capacity"].get("pseudoinverse_rcond", 1e-5)),
        )
        registration_mappings[profile.identifier] = mapping
        mapping_diagnostics[profile.identifier] = diagnostics

    progress_path = output_directory / "progress.jsonl"
    train_split, _ = _generate_teacher_dataset(
        split_name="training",
        experiment=experiment,
        settings=settings,
        base_config=base_config,
        profiles=profiles,
        conditions=settings["training_conditions"],
        episodes_per_condition=int(settings["training_episodes_per_condition"]),
        basis=basis,
        registration_mappings=registration_mappings,
        device=device,
        progress_path=progress_path,
        collect_control_metrics=False,
    )
    validation_split, _ = _generate_teacher_dataset(
        split_name="validation",
        experiment=experiment,
        settings=settings,
        base_config=base_config,
        profiles=profiles,
        conditions=settings["validation_conditions"],
        episodes_per_condition=int(settings["validation_episodes_per_condition"]),
        basis=basis,
        registration_mappings=registration_mappings,
        device=device,
        progress_path=progress_path,
        collect_control_metrics=False,
    )
    test_split, teacher_scenarios = _generate_teacher_dataset(
        split_name="diagnostic_test",
        experiment=experiment,
        settings=settings,
        base_config=base_config,
        profiles=profiles,
        conditions=settings["diagnostic_test_conditions"],
        episodes_per_condition=int(settings["diagnostic_test_episodes_per_condition"]),
        basis=basis,
        registration_mappings=registration_mappings,
        device=device,
        progress_path=progress_path,
        collect_control_metrics=True,
    )

    state_mean = train_split.states.mean(dim=0)
    state_scale = train_split.states.std(dim=0, unbiased=False).clamp_min(
        float(settings["normalization_min_scale"])
    )
    dataset_path = output_directory / "teacher_dataset.pt"
    torch.save(
        {
            "train": _dataset_payload(train_split),
            "validation": _dataset_payload(validation_split),
            "diagnostic_test": _dataset_payload(test_split),
            "state_mean": state_mean,
            "state_scale": state_scale,
            "teacher_preview_horizon_frames": int(settings["preview_horizon_frames"]),
            "evidence_boundary": {
                "future_truth_in_targets": True,
                "future_truth_in_inputs": False,
                "hardware_profile_id_in_inputs": False,
                "complete_episode_splits": True,
            },
        },
        dataset_path,
    )

    ridge_weight = _fit_ridge(
        train_split,
        state_mean,
        state_scale,
        alpha=float(experiment["model"]["ridge_alpha"]),
        device=device,
    )
    ridge_path = checkpoint_directory / "ridge.pt"
    torch.save(
        {
            "weight": ridge_weight,
            "state_mean": state_mean,
            "state_scale": state_scale,
        },
        ridge_path,
    )
    ridge_predictor = _ridge_predictor(ridge_weight, state_mean, state_scale, device)
    ridge_offline = _offline_metrics(ridge_predictor, test_split, device=device)

    model_results: list[dict[str, Any]] = []
    trained_models: dict[int, HighOrderImitationPolicy] = {}
    for seed in settings["initialization_seeds"]:
        model, record = _train_one_policy(
            seed=int(seed),
            train=train_split,
            validation=validation_split,
            test=test_split,
            state_mean=state_mean,
            state_scale=state_scale,
            settings=settings,
            device=device,
            checkpoint_directory=checkpoint_directory,
            output_directory=output_directory,
            progress_path=progress_path,
        )
        trained_models[int(seed)] = model
        model_results.append(record)

    baseline_lookup = {
        (item.profile_id, item.condition_id): item.baseline for item in teacher_scenarios
    }
    teacher_closed_loop = _summarize_teacher_scenarios(
        teacher_scenarios,
        gate=experiment["closed_loop_gate"],
    )
    scenario_records: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []

    ridge_closed_loop = _evaluate_predictor(
        predictor_id="ridge",
        predictor=ridge_predictor,
        experiment=experiment,
        settings=settings,
        base_config=base_config,
        profiles=profiles,
        conditions=settings["diagnostic_test_conditions"],
        episodes_per_condition=int(settings["diagnostic_test_episodes_per_condition"]),
        basis=basis,
        baseline_lookup=baseline_lookup,
        device=device,
        progress_path=progress_path,
        scenario_records=scenario_records,
        episode_records=episode_records,
    )

    for record in model_results:
        seed = int(record["initialization_seed"])
        model = trained_models[seed]
        predictor = _model_predictor(model, state_mean, state_scale, device)
        record["closed_loop_diagnostic_test"] = _evaluate_predictor(
            predictor_id=f"mlp_seed_{seed}",
            predictor=predictor,
            experiment=experiment,
            settings=settings,
            base_config=base_config,
            profiles=profiles,
            conditions=settings["diagnostic_test_conditions"],
            episodes_per_condition=int(settings["diagnostic_test_episodes_per_condition"]),
            basis=basis,
            baseline_lookup=baseline_lookup,
            device=device,
            progress_path=progress_path,
            scenario_records=scenario_records,
            episode_records=episode_records,
        )

    teacher_pass = teacher_closed_loop["closed_loop_gate"] == "PASS"
    offline_passes = [
        _offline_gate(item["offline_diagnostic_test"], experiment["offline_gate"])
        for item in model_results
    ]
    closed_loop_passes = [
        item["closed_loop_diagnostic_test"]["closed_loop_gate"] == "PASS"
        for item in model_results
    ]
    if quick:
        interpretation_status = "QUICK_SMOKE_ONLY"
    elif not teacher_pass:
        interpretation_status = "TEACHER_POSITIVE_CONTROL_FAILED"
    elif all(offline_passes) and all(closed_loop_passes):
        interpretation_status = "OBSERVATION_LEARNABLE"
    elif all(offline_passes):
        interpretation_status = "OFFLINE_LEARNABLE_BUT_CLOSED_LOOP_FAILED"
    else:
        interpretation_status = "OBSERVATION_OR_MODEL_LIMITED"

    _write_csv(output_directory / "scenario_records.csv", scenario_records)
    _write_csv(output_directory / "episode_records.csv", episode_records)
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run-diagnostic",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "s4d2_r2_high_order_learnability_v1",
        },
        "experiment": {
            "id": "AO-S4-D2-R2-HIGH-ORDER-LEARNABILITY",
            "type": "software_only_cuda_supervised_observation_learnability",
            "status": "completed_pending_audit",
            "quick": quick,
            "duration_seconds": time.perf_counter() - started,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "evidence_boundary": {
            "supervised_training_performed": True,
            "rl_training_performed": False,
            "teacher_future_truth_accessed": True,
            "model_future_truth_accessed": False,
            "teacher_exact_registration_accessed": True,
            "model_hardware_profile_identifier_accessed": False,
            "complete_episode_splits": True,
            "s4d3_accessed": False,
            "real_slm_actions": False,
        },
        "inputs": {
            "experiment_config": _relative(experiment_path),
            "experiment_config_sha256": _file_sha256(experiment_path),
            "upstream_capacity_summary_sha256": experiment["upstream_capacity"][
                "summary_sha256"
            ],
            "upstream_capacity_audit_sha256": experiment["upstream_capacity"][
                "audit_record_sha256"
            ],
            "dataset": _relative(dataset_path),
            "dataset_sha256": _file_sha256(dataset_path),
        },
        "design": {
            "representation": experiment["representation"],
            "teacher_preview_horizon_frames": int(settings["preview_horizon_frames"]),
            "policy_observation": experiment["policy_observation"],
            "model": {
                **experiment["model"],
                "hidden_size": int(settings["hidden_size"]),
            },
            "split_samples": {
                "training": len(train_split.states),
                "validation": len(validation_split.states),
                "diagnostic_test": len(test_split.states),
            },
            "split_episodes": {
                "training": train_split.episodes,
                "validation": validation_split.episodes,
                "diagnostic_test": test_split.episodes,
            },
        },
        "basis_diagnostics": basis_diagnostics,
        "mapping_diagnostics": mapping_diagnostics,
        "teacher_positive_control": teacher_closed_loop,
        "ridge": {
            "checkpoint": _relative(ridge_path),
            "offline_diagnostic_test": ridge_offline,
            "closed_loop_diagnostic_test": ridge_closed_loop,
        },
        "mlp_initializations": model_results,
        "interpretation": {
            "status": interpretation_status,
            "teacher_positive_control_pass": teacher_pass,
            "offline_pass_count": sum(offline_passes),
            "closed_loop_pass_count": sum(closed_loop_passes),
            "initialization_count": len(model_results),
            "new_rl_training_authorized": False,
            "algorithm_change_authorized": False,
            "s4d3_authorized": False,
            "real_hardware_authorized": False,
            "next_rule": (
                "Audit first. The result distinguishes observation learnability from "
                "SAC reward/optimization; it does not automatically authorize RL training."
            ),
        },
        "record_counts": {
            "scenario_records": len(scenario_records),
            "episode_records": len(episode_records),
        },
        "runtime": _runtime_record(),
        "git": _git_record(),
        "next_action": (
            "Stop and ask the assistant to audit the observation-learnability result. "
            "Do not train RL, open S4-D3, or operate real SLM hardware."
        ),
    }
    _write_json(output_directory / "summary.json", json_safe(summary))
    return summary


def preflight_s4_high_order_learnability(
    experiment_path: Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    """锁定上游容量证据、可部署输入、完整回合拆分和硬件边界。"""
    metadata = experiment["metadata"]
    if str(metadata["stage"]) != "S4-D2-R2-HIGH-ORDER-LEARNABILITY":
        raise ValueError("high-order-learnability stage metadata is invalid")
    expected_flags = {
        "allow_supervised_training": True,
        "allow_rl_training": False,
        "allow_reward_change": False,
        "allow_algorithm_comparison": False,
        "allow_prior_trajectory_reuse": False,
        "allow_s4d3_access": False,
        "allow_real_hardware_actions": False,
        "allow_future_truth_in_teacher_labels": True,
        "allow_future_truth_in_model_inputs": False,
        "allow_exact_registration_in_teacher_labels": True,
        "allow_hardware_profile_id_in_model_inputs": False,
    }
    for key, expected in expected_flags.items():
        if bool(metadata.get(key)) is not expected:
            raise RuntimeError(f"invalid high-order-learnability safety flag: {key}")

    upstream = _verify_upstream_capacity(experiment["upstream_capacity"])
    for field in ("environment_config", "hardware_profile_source"):
        path = _project_path(experiment[field])
        if _file_sha256(path) != str(experiment[f"{field}_sha256"]):
            raise RuntimeError(f"high-order-learnability {field} hash mismatch")

    representation = experiment["representation"]
    if (
        str(representation["id"]) != "zernike_21_added_11"
        or str(representation["kind"]) != "zernike"
        or int(representation["num_modes"]) != 21
        or int(representation["anchor_modes"]) != ANCHOR_MODES
        or int(representation["output_modes"]) != 11
    ):
        raise RuntimeError("learnability diagnosis must use the added 11 modes only")
    observation = experiment["policy_observation"]
    expected_state_size = int(observation["history_frames"]) * 2 * 21 + 2 * 21
    if (
        int(observation["history_frames"]) != 4
        or int(observation["state_size"]) != expected_state_size
        or bool(observation["include_future_truth"])
        or bool(observation["include_oracle_quality_metrics"])
        or bool(observation["include_hardware_profile_identifier"])
    ):
        raise RuntimeError("model input no longer matches the old deployable SAC observation")
    if int(experiment["teacher"]["preview_horizon_frames"]) != 2:
        raise RuntimeError("teacher preview horizon must remain fixed at two frames")
    if int(experiment["model"]["output_size"]) != 11:
        raise RuntimeError("supervised model must output exactly the added 11 modes")

    _validate_training_settings(settings)
    split_evidence = _validate_seed_splits(experiment, settings)
    output_directory = _project_path(settings["output_directory"])
    if output_directory.exists():
        raise FileExistsError(
            f"high-order-learnability output already exists: {output_directory}"
        )
    model = HighOrderImitationPolicy(
        int(observation["state_size"]),
        int(settings["hidden_size"]),
        11,
    )
    parameter_count = sum(item.numel() for item in model.parameters())
    samples = {
        split: (
            len(settings[f"{split}_conditions"])
            * len(settings["profile_ids"])
            * int(settings[f"{split}_episodes_per_condition"])
            * int(settings["steps_per_episode"])
        )
        for split in ("training", "validation", "diagnostic_test")
    }
    return {
        "status": "READY_FOR_USER_SUPERVISED_TRAINING",
        "quick": quick,
        "experiment_config": _relative(experiment_path),
        "experiment_config_sha256": _file_sha256(experiment_path),
        "output_directory": settings["output_directory"],
        "upstream_capacity_summary_sha256": upstream["summary_sha256"],
        "upstream_capacity_audit_sha256": upstream["audit_sha256"],
        "representation": representation,
        "state_size": int(observation["state_size"]),
        "output_size": 11,
        "hidden_size": int(settings["hidden_size"]),
        "parameter_count": parameter_count,
        "teacher_preview_horizon_frames": 2,
        "sample_counts": samples,
        "split_evidence": split_evidence,
        "profile_ids": settings["profile_ids"],
        "initialization_seeds": settings["initialization_seeds"],
        "cuda_required": True,
        "supervised_training_allowed": True,
        "rl_training_allowed": False,
        "future_truth_in_teacher_labels": True,
        "future_truth_in_model_inputs": False,
        "hardware_profile_id_in_model_inputs": False,
        "s4d3_accessed": False,
        "real_slm_actions": False,
        "formal_command": (
            ".\\.venv\\Scripts\\python.exe scripts\\train_s4_high_order_learnability.py "
            "--config configs\\experiments\\s4_high_order_learnability_v1.yaml"
        ),
    }


def _generate_teacher_dataset(
    *,
    split_name: str,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    base_config: S1EnvConfig,
    profiles: list[HardwareProfile],
    conditions: Iterable[dict[str, Any]],
    episodes_per_condition: int,
    basis: torch.Tensor,
    registration_mappings: dict[str, torch.Tensor],
    device: torch.device,
    progress_path: Path,
    collect_control_metrics: bool,
) -> tuple[TeacherDataset, list[TeacherScenario]]:
    condition_list = [RobustnessCondition.from_mapping(item) for item in conditions]
    steps = int(settings["steps_per_episode"])
    preview = int(settings["preview_horizon_frames"])
    total = len(profiles) * len(condition_list) * steps
    bar = counted_progress(
        total=total,
        description=f"生成{split_name}教师回合",
        unit="步",
    )
    all_states: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    scenarios: list[TeacherScenario] = []
    completed = 0
    for profile in profiles:
        for condition in condition_list:
            config = replace(
                base_config,
                batch_size=episodes_per_condition,
                episode_length=max(steps + preview, base_config.episode_length),
            )
            config = profile.environment_config(condition.environment_config(config))
            future_truth = _future_disturbance_sequence(
                config=config,
                condition=condition,
                profile=profile,
                length=steps + preview,
                basis=basis,
                device=device,
            )
            environment = AdaptiveOpticsEnv(
                config,
                device,
                profile.effects_config(),
                basis_override=basis,
            )
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
            controller = _teacher_controller(experiment, config)
            state = controller.reset(noisy)
            scenario_states: list[torch.Tensor] = []
            scenario_targets: list[torch.Tensor] = []
            science = _empty_step_metrics()
            for step in range(steps):
                desired_applied = -future_truth[step + preview]
                target_request = desired_applied @ registration_mappings[
                    profile.identifier
                ].transpose(0, 1)
                action, _, _ = controller.compose_added_modes_to_target(target_request)
                scenario_states.append(state.detach().cpu())
                scenario_targets.append(
                    (
                        action.requested_residual_rad[:, ANCHOR_MODES:]
                        / float(experiment["action_budget"]["residual_component_limit_rad"])
                    )
                    .detach()
                    .cpu()
                )
                observation, _, _, _, info = environment.step(action.final_delta_rad)
                _append_step_metrics(science, info)
                completed += 1
                advance_to(bar, completed)
                if step + 1 < steps:
                    noisy = _noisy_observation(
                        observation,
                        config.num_modes,
                        profile.observation_noise_std_rad,
                        generator,
                    )
                    state = controller.advance_observation(noisy)
            states = torch.stack(scenario_states, dim=1).flatten(0, 1)
            targets = torch.stack(scenario_targets, dim=1).flatten(0, 1)
            all_states.append(states)
            all_targets.append(targets)
            _append_jsonl(
                progress_path,
                {
                    "phase": "teacher_data",
                    "split": split_name,
                    "profile": profile.identifier,
                    "condition": condition.identifier,
                    "completed_steps": completed,
                    "total_steps": total,
                    "samples": len(states),
                },
            )
            update_progress(
                bar,
                device=device,
                metrics={"样本": float(sum(len(item) for item in all_states))},
            )
            if collect_control_metrics:
                candidate = _mean_step_metrics(science)
                baseline = _rollout_anchored_baseline(
                    experiment,
                    config,
                    condition,
                    profile,
                    steps,
                    device,
                )
                scenarios.append(
                    TeacherScenario(
                        profile_id=profile.identifier,
                        condition_id=condition.identifier,
                        base_seed=condition.base_seed,
                        candidate=candidate,
                        baseline=baseline,
                    )
                )
    bar.close()
    dataset = TeacherDataset(
        states=torch.cat(all_states),
        targets=torch.cat(all_targets),
        episodes=len(profiles) * len(condition_list) * episodes_per_condition,
        steps_per_episode=steps,
    )
    if len(dataset.states) != dataset.episodes * steps:
        raise RuntimeError("teacher dataset sample count does not match complete episodes")
    return dataset, scenarios


def _teacher_controller(
    experiment: dict[str, Any], config: S1EnvConfig
) -> AddedModesOracleController:
    params = experiment["frozen_controller"]["parameters"]
    return AddedModesOracleController(
        num_modes=config.num_modes,
        anchor_modes=ANCHOR_MODES,
        modal_limit_rad=config.modal_limit_rad,
        history_frames=int(experiment["policy_observation"]["history_frames"]),
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


def _student_controller(
    experiment: dict[str, Any], config: S1EnvConfig
) -> AnchoredResidualTrackingController:
    params = experiment["frozen_controller"]["parameters"]
    return AnchoredResidualTrackingController(
        num_modes=config.num_modes,
        anchor_modes=ANCHOR_MODES,
        modal_limit_rad=config.modal_limit_rad,
        history_frames=int(experiment["policy_observation"]["history_frames"]),
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


def _fit_ridge(
    train: TeacherDataset,
    state_mean: torch.Tensor,
    state_scale: torch.Tensor,
    *,
    alpha: float,
    device: torch.device,
) -> torch.Tensor:
    if alpha <= 0:
        raise ValueError("ridge alpha must be positive")
    x = ((train.states - state_mean) / state_scale).to(device)
    y = train.targets.to(device)
    x = torch.cat((x, torch.ones(len(x), 1, device=device)), dim=1)
    gram = x.transpose(0, 1) @ x
    regularizer = torch.eye(gram.shape[0], device=device) * alpha
    regularizer[-1, -1] = 0
    weight = torch.linalg.solve(gram + regularizer, x.transpose(0, 1) @ y)
    return weight.detach().cpu()


def _ridge_predictor(
    weight: torch.Tensor,
    state_mean: torch.Tensor,
    state_scale: torch.Tensor,
    device: torch.device,
) -> Predictor:
    weight_device = weight.to(device)
    mean_device = state_mean.to(device)
    scale_device = state_scale.to(device)

    def predict(state: torch.Tensor) -> torch.Tensor:
        normalized = (state.to(device) - mean_device) / scale_device
        augmented = torch.cat(
            (normalized, torch.ones(len(normalized), 1, device=device)), dim=1
        )
        return (augmented @ weight_device).clamp(-1, 1)

    return predict


def _model_predictor(
    model: HighOrderImitationPolicy,
    state_mean: torch.Tensor,
    state_scale: torch.Tensor,
    device: torch.device,
) -> Predictor:
    mean_device = state_mean.to(device)
    scale_device = state_scale.to(device)
    model.eval()

    def predict(state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model((state.to(device) - mean_device) / scale_device)

    return predict


def _train_one_policy(
    *,
    seed: int,
    train: TeacherDataset,
    validation: TeacherDataset,
    test: TeacherDataset,
    state_mean: torch.Tensor,
    state_scale: torch.Tensor,
    settings: dict[str, Any],
    device: torch.device,
    checkpoint_directory: Path,
    output_directory: Path,
    progress_path: Path,
) -> tuple[HighOrderImitationPolicy, dict[str, Any]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    model = HighOrderImitationPolicy(
        train.states.shape[1], int(settings["hidden_size"]), train.targets.shape[1]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    loader = DataLoader(
        TensorDataset(train.states, train.targets),
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    iterator = iter(loader)
    mean_device = state_mean.to(device)
    scale_device = state_scale.to(device)
    max_steps = int(settings["max_optimizer_steps"])
    validation_interval = int(settings["validation_interval_steps"])
    min_steps = int(settings["min_optimizer_steps"])
    patience = int(settings["early_stopping_patience_checks"])
    relative_delta = float(settings["early_stopping_relative_min_delta"])
    checkpoint_path = checkpoint_directory / f"mlp_seed_{seed}_best.pt"
    final_path = checkpoint_directory / f"mlp_seed_{seed}_final.pt"
    history_path = output_directory / f"mlp_seed_{seed}_loss.csv"
    history: list[dict[str, Any]] = []
    best_validation = float("inf")
    best_step = 0
    stale_checks = 0
    interval_loss = 0.0
    interval_batches = 0
    bar = progress_bar(
        range(1, max_steps + 1),
        description=f"监督MLP种子{seed}",
        unit="步",
        leave=True,
    )
    stopped_early = False
    for step in bar:
        try:
            states, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            states, targets = next(iterator)
        states = states.to(device)
        targets = targets.to(device)
        prediction = model((states - mean_device) / scale_device)
        loss = torch.mean((prediction - targets).square())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(settings["gradient_clip_norm"])
        )
        optimizer.step()
        interval_loss += float(loss.detach())
        interval_batches += 1

        if step % validation_interval == 0 or step == max_steps:
            validation_metrics = _offline_metrics(
                _model_predictor(model, state_mean, state_scale, device),
                validation,
                device=device,
            )
            validation_mse = float(validation_metrics["mse"])
            train_mse = interval_loss / max(interval_batches, 1)
            significant = (
                math.isinf(best_validation)
                or (best_validation - validation_mse) / max(best_validation, 1e-12)
                >= relative_delta
            )
            if validation_mse < best_validation:
                best_validation = validation_mse
                best_step = step
                torch.save(
                    {
                        "model": model.state_dict(),
                        "state_mean": state_mean,
                        "state_scale": state_scale,
                        "seed": seed,
                        "step": step,
                        "validation_mse": validation_mse,
                    },
                    checkpoint_path,
                )
            stale_checks = 0 if significant else stale_checks + 1
            row = {
                "step": step,
                "train_mse": train_mse,
                "validation_mse": validation_mse,
                "validation_skill": validation_metrics["skill_score_against_zero"],
                "validation_cosine": validation_metrics["mean_cosine_similarity"],
                "best_validation_mse": best_validation,
                "stale_checks": stale_checks,
            }
            history.append(row)
            _write_csv(history_path, history)
            _append_jsonl(progress_path, {"phase": "training", "seed": seed, **row})
            update_progress(
                bar,
                device=device,
                metrics={
                    "训练误差": train_mse,
                    "验证误差": validation_mse,
                    "技能分数": float(validation_metrics["skill_score_against_zero"]),
                },
            )
            interval_loss = 0.0
            interval_batches = 0
            if step >= min_steps and stale_checks >= patience:
                stopped_early = True
                break
    torch.save(
        {
            "model": model.state_dict(),
            "state_mean": state_mean,
            "state_scale": state_scale,
            "seed": seed,
            "step": int(history[-1]["step"]),
        },
        final_path,
    )
    best = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    predictor = _model_predictor(model, state_mean, state_scale, device)
    return model, {
        "initialization_seed": seed,
        "best_step": best_step,
        "stopped_early": stopped_early,
        "best_validation_mse": best_validation,
        "checkpoint_best": _relative(checkpoint_path),
        "checkpoint_final": _relative(final_path),
        "loss_history": _relative(history_path),
        "offline_validation": _offline_metrics(
            predictor, validation, device=device
        ),
        "offline_diagnostic_test": _offline_metrics(
            predictor, test, device=device
        ),
    }


def _offline_metrics(
    predictor: Predictor,
    data: TeacherDataset,
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> dict[str, float]:
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    loader = DataLoader(
        TensorDataset(data.states, data.targets), batch_size=batch_size, shuffle=False
    )
    with torch.no_grad():
        for states, target in loader:
            predictions.append(predictor(states.to(device)).detach().cpu())
            targets.append(target)
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    mse = float((prediction - target).square().mean())
    zero_mse = float(target.square().mean())
    skill = 1 - mse / max(zero_mse, 1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        prediction,
        target,
        dim=-1,
        eps=1e-8,
    )
    return {
        "samples": len(target),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float((prediction - target).abs().mean()),
        "zero_predictor_mse": zero_mse,
        "skill_score_against_zero": skill,
        "mean_cosine_similarity": float(cosine.mean()),
    }


def _evaluate_predictor(
    *,
    predictor_id: str,
    predictor: Predictor,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    base_config: S1EnvConfig,
    profiles: list[HardwareProfile],
    conditions: Iterable[dict[str, Any]],
    episodes_per_condition: int,
    basis: torch.Tensor,
    baseline_lookup: dict[tuple[str, str], dict[str, torch.Tensor]],
    device: torch.device,
    progress_path: Path,
    scenario_records: list[dict[str, Any]],
    episode_records: list[dict[str, Any]],
) -> dict[str, Any]:
    condition_list = [RobustnessCondition.from_mapping(item) for item in conditions]
    steps = int(settings["steps_per_episode"])
    bar = counted_progress(
        total=len(profiles) * len(condition_list),
        description=f"闭环诊断{predictor_id}",
        unit="场景",
    )
    candidates: dict[str, list[torch.Tensor]] = defaultdict(list)
    baselines: dict[str, list[torch.Tensor]] = defaultdict(list)
    profile_candidates: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    profile_baselines: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    completed = 0
    for profile in profiles:
        for condition in condition_list:
            config = replace(
                base_config,
                batch_size=episodes_per_condition,
                episode_length=max(steps, base_config.episode_length),
            )
            config = profile.environment_config(condition.environment_config(config))
            candidate = _rollout_student(
                predictor,
                experiment,
                config,
                condition,
                profile,
                steps,
                basis,
                device,
            )
            baseline = baseline_lookup[(profile.identifier, condition.identifier)]
            _collect_science(candidates, candidate)
            _collect_science(baselines, baseline)
            _collect_science(profile_candidates[profile.identifier], candidate)
            _collect_science(profile_baselines[profile.identifier], baseline)
            scenario_records.append(
                _scenario_row(
                    predictor_id,
                    profile.identifier,
                    condition.identifier,
                    candidate,
                    baseline,
                )
            )
            episode_records.extend(
                _episode_rows(
                    predictor_id,
                    profile.identifier,
                    condition.identifier,
                    condition.base_seed,
                    candidate,
                    baseline,
                )
            )
            completed += 1
            advance_to(bar, completed)
            _append_jsonl(
                progress_path,
                {
                    "phase": "closed_loop_evaluation",
                    "predictor": predictor_id,
                    "profile": profile.identifier,
                    "condition": condition.identifier,
                    "completed_scenarios": completed,
                    "total_scenarios": len(profiles) * len(condition_list),
                },
            )
            update_progress(
                bar,
                device=device,
                metrics={
                    "功率差": float(
                        candidate["power_in_bucket"].mean()
                        - baseline["power_in_bucket"].mean()
                    )
                },
            )
    bar.close()
    overall = _paired_science_summary(
        predictor_id,
        _concatenate(candidates),
        _concatenate(baselines),
        experiment["closed_loop_gate"],
    )
    per_profile = [
        _paired_science_summary(
            profile.identifier,
            _concatenate(profile_candidates[profile.identifier]),
            _concatenate(profile_baselines[profile.identifier]),
            experiment["closed_loop_gate"],
        )
        for profile in profiles
    ]
    all_profiles_pass = all(item["gate"] == "PASS" for item in per_profile)
    final_gate = (
        "PASS"
        if overall["gate"] == "PASS"
        and (
            all_profiles_pass
            if bool(experiment["closed_loop_gate"]["require_all_profiles_pass"])
            else True
        )
        else "FAIL"
    )
    return {
        "predictor": predictor_id,
        "overall": overall,
        "profiles": per_profile,
        "all_profiles_pass": all_profiles_pass,
        "closed_loop_gate": final_gate,
    }


@torch.no_grad()
def _rollout_student(
    predictor: Predictor,
    experiment: dict[str, Any],
    config: S1EnvConfig,
    condition: RobustnessCondition,
    profile: HardwareProfile,
    steps: int,
    basis: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    environment = AdaptiveOpticsEnv(
        config,
        device,
        profile.effects_config(),
        basis_override=basis,
    )
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
    controller = _student_controller(experiment, config)
    state = controller.reset(noisy)
    science = _empty_step_metrics()
    for step in range(steps):
        added = predictor(state).clamp(-1, 1)
        normalized = torch.zeros(
            config.batch_size,
            config.num_modes,
            device=device,
            dtype=state.dtype,
        )
        normalized[:, ANCHOR_MODES:] = added
        action = controller.compose_action(normalized)
        observation, _, _, _, info = environment.step(action.final_delta_rad)
        _append_step_metrics(science, info)
        if step + 1 < steps:
            noisy = _noisy_observation(
                observation,
                config.num_modes,
                profile.observation_noise_std_rad,
                generator,
            )
            state = controller.advance_observation(noisy)
    return _mean_step_metrics(science)


def _summarize_teacher_scenarios(
    scenarios: list[TeacherScenario], *, gate: dict[str, Any]
) -> dict[str, Any]:
    candidates: dict[str, list[torch.Tensor]] = defaultdict(list)
    baselines: dict[str, list[torch.Tensor]] = defaultdict(list)
    profile_candidates: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    profile_baselines: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item in scenarios:
        _collect_science(candidates, item.candidate)
        _collect_science(baselines, item.baseline)
        _collect_science(profile_candidates[item.profile_id], item.candidate)
        _collect_science(profile_baselines[item.profile_id], item.baseline)
    overall = _paired_science_summary(
        "teacher_fixed_preview_2",
        _concatenate(candidates),
        _concatenate(baselines),
        gate,
    )
    profiles = [
        _paired_science_summary(
            profile_id,
            _concatenate(profile_candidates[profile_id]),
            _concatenate(profile_baselines[profile_id]),
            gate,
        )
        for profile_id in sorted(profile_candidates)
    ]
    all_profiles_pass = all(item["gate"] == "PASS" for item in profiles)
    return {
        "overall": overall,
        "profiles": profiles,
        "all_profiles_pass": all_profiles_pass,
        "closed_loop_gate": (
            "PASS" if overall["gate"] == "PASS" and all_profiles_pass else "FAIL"
        ),
    }


def _paired_science_summary(
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
        and paired["strehl"]["ci95_low"]
        > float(gate["min_strehl_delta_ci95_low"])
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
        "gate": "PASS" if passed else "FAIL",
    }


def _collect_science(
    destination: dict[str, list[torch.Tensor]], source: dict[str, torch.Tensor]
) -> None:
    for metric in SCIENCE_METRICS:
        destination[metric].append(source[metric].detach().cpu())


def _concatenate(values: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.cat(items) for key, items in values.items()}


def _scenario_row(
    predictor_id: str,
    profile_id: str,
    condition_id: str,
    candidate: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "predictor_id": predictor_id,
        "profile": profile_id,
        "physical_condition": condition_id,
        "episodes": int(candidate["power_in_bucket"].numel()),
    }
    for metric in SCIENCE_METRICS:
        row[f"candidate_{metric}_mean"] = float(candidate[metric].mean())
        row[f"baseline_{metric}_mean"] = float(baseline[metric].mean())
        row[f"delta_{metric}_mean"] = float(
            (candidate[metric] - baseline[metric]).mean()
        )
    return row


def _episode_rows(
    predictor_id: str,
    profile_id: str,
    condition_id: str,
    base_seed: int,
    candidate: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    rows = []
    count = int(candidate["power_in_bucket"].numel())
    for index in range(count):
        row: dict[str, Any] = {
            "predictor_id": predictor_id,
            "profile": profile_id,
            "physical_condition": condition_id,
            "episode_index": index,
            "episode_seed": base_seed + index,
        }
        for metric in SCIENCE_METRICS:
            candidate_value = float(candidate[metric][index])
            baseline_value = float(baseline[metric][index])
            row[f"candidate_{metric}"] = candidate_value
            row[f"baseline_{metric}"] = baseline_value
            row[f"delta_{metric}"] = candidate_value - baseline_value
        rows.append(row)
    return rows


def _offline_gate(metrics: dict[str, Any], gate: dict[str, Any]) -> bool:
    return (
        float(metrics["skill_score_against_zero"])
        >= float(gate["min_skill_score_against_zero"])
        and float(metrics["mean_cosine_similarity"])
        >= float(gate["min_mean_cosine_similarity"])
    )


def _verify_upstream_capacity(upstream: dict[str, Any]) -> dict[str, str]:
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
    paths: dict[str, Path] = {}
    for field in fields:
        path = _project_path(upstream[field])
        if _file_sha256(path) != str(upstream[f"{field}_sha256"]):
            raise RuntimeError(f"high-order-capacity evidence hash mismatch: {field}")
        paths[field] = path
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    passing = set(summary["interpretation"]["passing_representation_ids"])
    if (
        bool(summary["experiment"]["quick"])
        or summary["experiment"]["status"] != "completed_pending_audit"
        or not bool(summary["interpretation"]["capacity_demonstrated"])
        or "zernike_21_added_11" not in passing
        or summary["integrity"]["truth_alignment"]["status"] != "PASS"
        or summary["integrity"]["requested_anchor_residual"]["status"] != "PASS"
    ):
        raise RuntimeError("formal high-order-capacity state is not usable")
    audit = paths["audit_record"].read_text(encoding="utf-8")
    if (
        "Verification Status: `ANALYZED`" not in audit
        or "新增11维作为下一步主方案：`SUPPORTED`" not in audit
        or "4帧观测可学习性诊断：`AUTHORIZED`" not in audit
    ):
        raise RuntimeError("high-order-capacity audit is not finalized")
    manifest = json.loads(paths["source_manifest"].read_text(encoding="utf-8"))
    for relative, digest in manifest.items():
        path = _project_path(relative)
        if not path.is_file() or _file_sha256(path) != str(digest):
            raise RuntimeError(f"high-order-capacity source changed: {relative}")
    return {
        "summary_sha256": _file_sha256(paths["summary"]),
        "audit_sha256": _file_sha256(paths["audit_record"]),
    }


def _validate_seed_splits(
    experiment: dict[str, Any], settings: dict[str, Any]
) -> dict[str, Any]:
    protected = [
        (int(item["start_inclusive"]), int(item["end_exclusive"]))
        for item in experiment["protected_seed_ranges"]
    ]
    split_sets: dict[str, set[int]] = {}
    for split in ("training", "validation", "diagnostic_test"):
        episode_count = int(settings[f"{split}_episodes_per_condition"])
        seeds: set[int] = set()
        for item in settings[f"{split}_conditions"]:
            start = int(item["base_seed"])
            current = set(range(start, start + episode_count))
            if seeds & current:
                raise RuntimeError(f"episode seeds overlap within {split}")
            if any(start < stop and start + episode_count > begin for begin, stop in protected):
                raise RuntimeError(f"{split} episode seeds overlap a protected range")
            seeds.update(current)
        split_sets[split] = seeds
    names = list(split_sets)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            if split_sets[left_name] & split_sets[right_name]:
                raise RuntimeError(f"episode seeds overlap across {left_name} and {right_name}")
    return {
        name: {
            "episode_seed_count": len(values),
            "minimum_seed": min(values),
            "maximum_seed": max(values),
        }
        for name, values in split_sets.items()
    }


def _validate_training_settings(settings: dict[str, Any]) -> None:
    positive = (
        "steps_per_episode",
        "training_episodes_per_condition",
        "validation_episodes_per_condition",
        "diagnostic_test_episodes_per_condition",
        "hidden_size",
        "batch_size",
        "max_optimizer_steps",
        "min_optimizer_steps",
        "validation_interval_steps",
        "early_stopping_patience_checks",
    )
    if any(int(settings[key]) <= 0 for key in positive):
        raise ValueError("training sizes and intervals must be positive")
    if int(settings["min_optimizer_steps"]) > int(settings["max_optimizer_steps"]):
        raise ValueError("minimum optimizer steps exceed maximum")
    if not settings["initialization_seeds"]:
        raise ValueError("at least one initialization seed is required")
    if len(settings["initialization_seeds"]) != len(set(settings["initialization_seeds"])):
        raise ValueError("initialization seeds must be unique")
    if float(settings["learning_rate"]) <= 0 or float(settings["gradient_clip_norm"]) <= 0:
        raise ValueError("learning rate and gradient clipping must be positive")


def _effective_settings(experiment: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    dataset = experiment["dataset"]
    training = experiment["training"]
    settings: dict[str, Any] = {
        "output_directory": experiment["outputs"]["directory"],
        "steps_per_episode": int(dataset["steps_per_episode"]),
        "training_episodes_per_condition": int(
            dataset["training_episodes_per_condition"]
        ),
        "validation_episodes_per_condition": int(
            dataset["validation_episodes_per_condition"]
        ),
        "diagnostic_test_episodes_per_condition": int(
            dataset["diagnostic_test_episodes_per_condition"]
        ),
        "profile_ids": list(dataset["profile_ids"]),
        "training_conditions": list(dataset["training_conditions"]),
        "validation_conditions": list(dataset["validation_conditions"]),
        "diagnostic_test_conditions": list(dataset["diagnostic_test_conditions"]),
        "preview_horizon_frames": int(experiment["teacher"]["preview_horizon_frames"]),
        "initialization_seeds": list(map(int, training["initialization_seeds"])),
        "hidden_size": int(experiment["model"]["hidden_size"]),
        "batch_size": int(training["batch_size"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "max_optimizer_steps": int(training["max_optimizer_steps"]),
        "min_optimizer_steps": int(training["min_optimizer_steps"]),
        "validation_interval_steps": int(training["validation_interval_steps"]),
        "early_stopping_patience_checks": int(
            training["early_stopping_patience_checks"]
        ),
        "early_stopping_relative_min_delta": float(
            training["early_stopping_relative_min_delta"]
        ),
        "gradient_clip_norm": float(training["gradient_clip_norm"]),
        "normalization_min_scale": float(training["normalization_min_scale"]),
    }
    if quick:
        quick_settings = experiment["quick"]
        settings.update(
            {
                "output_directory": experiment["outputs"]["quick_directory"],
                "steps_per_episode": int(quick_settings["steps_per_episode"]),
                "training_episodes_per_condition": int(
                    quick_settings["training_episodes_per_condition"]
                ),
                "validation_episodes_per_condition": int(
                    quick_settings["validation_episodes_per_condition"]
                ),
                "diagnostic_test_episodes_per_condition": int(
                    quick_settings["diagnostic_test_episodes_per_condition"]
                ),
                "profile_ids": list(quick_settings["profile_ids"]),
                "training_conditions": list(quick_settings["training_conditions"]),
                "validation_conditions": list(quick_settings["validation_conditions"]),
                "diagnostic_test_conditions": list(
                    quick_settings["diagnostic_test_conditions"]
                ),
                "initialization_seeds": list(
                    map(int, quick_settings["initialization_seeds"])
                ),
                "hidden_size": int(quick_settings["hidden_size"]),
                "batch_size": int(quick_settings["batch_size"]),
                "max_optimizer_steps": int(quick_settings["max_optimizer_steps"]),
                "min_optimizer_steps": int(quick_settings["min_optimizer_steps"]),
                "validation_interval_steps": int(
                    quick_settings["validation_interval_steps"]
                ),
                "early_stopping_patience_checks": int(
                    quick_settings["early_stopping_patience_checks"]
                ),
            }
        )
    return settings


def _dataset_payload(dataset: TeacherDataset) -> dict[str, Any]:
    return {
        "states": dataset.states,
        "targets": dataset.targets,
        "episodes": dataset.episodes,
        "steps_per_episode": dataset.steps_per_episode,
    }


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(value), ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
