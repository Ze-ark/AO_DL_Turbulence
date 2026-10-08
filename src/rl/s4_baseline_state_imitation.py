"""S4-D2-R2基线闭环状态监督学习与同种子部署诊断。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

import torch

from src.rl.s4_closed_loop_shift import (
    FrozenPredictor,
    ShiftScenario,
    _checkpoint_inventory,
    _load_predictors,
    _oracle_normalized_action,
    _progress_event,
    _rollout_predictor,
    _rollout_teacher,
    _scale_key,
    _scenario_config,
    _shift_episode_rows,
    _shift_scenario_row,
    _summarize_shift_scenarios,
    _temporal_episode_rows,
    _temporal_scenario_rows,
    _zero_alignment,
)
from src.rl.s4_high_order_learnability import (
    HighOrderImitationPolicy,
    TeacherDataset,
    TeacherScenario,
    _append_jsonl,
    _dataset_payload,
    _episode_rows,
    _fit_ridge,
    _model_predictor,
    _offline_gate,
    _offline_metrics,
    _ridge_predictor,
    _student_controller,
    _summarize_teacher_scenarios,
    _train_one_policy,
    _validate_seed_splits,
    _validate_training_settings,
    _write_csv,
)
from src.rl.s4_representation_capacity import (
    ANCHOR_MODES,
    ActionRepresentation,
    _future_disturbance_sequence,
    build_action_basis,
    representation_registration_inverse,
)
from src.rl.s4_training import (
    _append_step_metrics,
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


@dataclass(frozen=True)
class BaselineStateSplit:
    """传统基线轨迹上的监督数据和可选闭环上下文。"""

    dataset: TeacherDataset
    baselines: dict[tuple[str, str], dict[str, torch.Tensor]]
    future_truth: dict[tuple[str, str], torch.Tensor]
    configs: dict[tuple[str, str], S1EnvConfig]


def run_s4_baseline_state_imitation(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """在传统基线轨迹上生成标签、训练模型并进行配对闭环评估。"""
    experiment_path = _project_path(config_path)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight = preflight_s4_baseline_state_imitation(
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
            "baseline-state-imitation output already exists; preserve it for audit: "
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
    base_config = _replace_num_modes(base_config, representation.num_modes)
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
            rcond=float(experiment["representation"]["pseudoinverse_rcond"]),
        )
        registration_mappings[profile.identifier] = mapping
        mapping_diagnostics[profile.identifier] = diagnostics

    progress_path = output_directory / "progress.jsonl"
    train_split = _generate_baseline_state_split(
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
        keep_context=False,
    )
    validation_split = _generate_baseline_state_split(
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
        keep_context=False,
    )
    test_split = _generate_baseline_state_split(
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
        keep_context=True,
    )

    state_mean = train_split.dataset.states.mean(dim=0)
    state_scale = train_split.dataset.states.std(dim=0, unbiased=False).clamp_min(
        float(settings["normalization_min_scale"])
    )
    dataset_path = output_directory / "baseline_state_dataset.pt"
    torch.save(
        {
            "train": _dataset_payload(train_split.dataset),
            "validation": _dataset_payload(validation_split.dataset),
            "diagnostic_test": _dataset_payload(test_split.dataset),
            "state_mean": state_mean,
            "state_scale": state_scale,
            "source_policy": "tracking_conservative_with_zero_added_modes",
            "teacher_preview_horizon_frames": int(settings["preview_horizon_frames"]),
            "evidence_boundary": {
                "future_truth_in_targets": True,
                "future_truth_in_inputs": False,
                "hardware_profile_id_in_inputs": False,
                "complete_episode_splits": True,
                "teacher_trajectory_samples": False,
                "student_action_samples": False,
            },
        },
        dataset_path,
    )

    ridge_weight = _fit_ridge(
        train_split.dataset,
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
            "data_source": "traditional_baseline_trajectory",
        },
        ridge_path,
    )
    ridge_predict = _ridge_predictor(ridge_weight, state_mean, state_scale, device)

    model_results: list[dict[str, Any]] = []
    trained_models: dict[int, HighOrderImitationPolicy] = {}
    for seed in settings["initialization_seeds"]:
        model, record = _train_one_policy(
            seed=int(seed),
            train=train_split.dataset,
            validation=validation_split.dataset,
            test=test_split.dataset,
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

    reference_items = _selected_reference_predictors(experiment, settings)
    reference_predictors = _load_predictors(
        {"predictors": reference_items},
        device=device,
    )
    new_predictors = [
        FrozenPredictor(
            identifier="baseline_ridge",
            predict=ridge_predict,
            state_mean=state_mean,
            state_scale=state_scale,
        )
    ]
    for record in model_results:
        seed = int(record["initialization_seed"])
        new_predictors.append(
            FrozenPredictor(
                identifier=f"baseline_mlp_seed_{seed}",
                predict=_model_predictor(
                    trained_models[seed], state_mean, state_scale, device
                ),
                state_mean=state_mean,
                state_scale=state_scale,
            )
        )
    predictors = reference_predictors + new_predictors
    offline_results = {
        item.identifier: _offline_metrics(
            item.predict,
            test_split.dataset,
            device=device,
        )
        for item in predictors
    }

    evaluation = _evaluate_all(
        experiment=experiment,
        settings=settings,
        profiles=profiles,
        predictors=predictors,
        offline_results=offline_results,
        registration_mappings=registration_mappings,
        basis=basis,
        test_context=test_split,
        device=device,
        progress_path=progress_path,
    )
    interpretation = _interpretation(
        grouped=evaluation["grouped"],
        offline_results=offline_results,
        teacher_summary=evaluation["teacher_summary"],
        experiment=experiment,
        settings=settings,
        quick=quick,
    )

    _write_csv(
        output_directory / "control_episode_records.csv",
        evaluation["control_episode_records"],
    )
    _write_csv(
        output_directory / "scenario_records.csv",
        evaluation["scenario_records"],
    )
    _write_csv(
        output_directory / "episode_records.csv",
        evaluation["episode_records"],
    )
    _write_csv(
        output_directory / "temporal_records.csv",
        evaluation["temporal_records"],
    )
    _write_csv(
        output_directory / "temporal_episode_records.csv",
        evaluation["temporal_episode_records"],
    )

    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run-diagnostic",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "s4d2_r2_baseline_state_imitation_v1",
        },
        "experiment": {
            "id": "AO-S4-D2-R2-BASELINE-STATE-IMITATION",
            "type": "software_only_cuda_supervised_distribution_correction",
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
            "training_state_source": "traditional_baseline_trajectory",
            "student_state_aggregation_rounds": 0,
            "complete_episode_splits": True,
            "s4d3_accessed": False,
            "real_slm_actions": False,
        },
        "inputs": {
            "experiment_config": _relative(experiment_path),
            "experiment_config_sha256": _file_sha256(experiment_path),
            "upstream_shift_summary_sha256": experiment["upstream_shift"][
                "summary_sha256"
            ],
            "upstream_shift_audit_canonical_sha256": experiment["upstream_shift"][
                "audit_record_sha256"
            ],
            "dataset": _relative(dataset_path),
            "dataset_sha256": _file_sha256(dataset_path),
        },
        "design": {
            "representation": experiment["representation"],
            "policy_observation": experiment["policy_observation"],
            "training_data_source": experiment["dataset"]["state_source"],
            "comparison_data_source": "teacher_trajectory_frozen_checkpoints",
            "deployment_scales": settings["deployment_scales"],
            "primary_scale": settings["primary_scale"],
            "model": {
                **experiment["model"],
                "hidden_size": int(settings["hidden_size"]),
            },
            "split_samples": {
                "training": len(train_split.dataset.states),
                "validation": len(validation_split.dataset.states),
                "diagnostic_test": len(test_split.dataset.states),
            },
            "split_episodes": {
                "training": train_split.dataset.episodes,
                "validation": validation_split.dataset.episodes,
                "diagnostic_test": test_split.dataset.episodes,
            },
        },
        "basis_diagnostics": basis_diagnostics,
        "mapping_diagnostics": mapping_diagnostics,
        "offline_diagnostic_test": offline_results,
        "baseline_ridge": {
            "checkpoint": _relative(ridge_path),
            "offline_diagnostic_test": offline_results["baseline_ridge"],
        },
        "baseline_mlp_initializations": model_results,
        "teacher_positive_control": evaluation["teacher_summary"],
        "zero_scale_alignment": evaluation["zero_alignment"],
        "predictor_scale_results": evaluation["grouped"],
        "interpretation": interpretation,
        "record_counts": {
            "control_episode_records": len(evaluation["control_episode_records"]),
            "scenario_records": len(evaluation["scenario_records"]),
            "episode_records": len(evaluation["episode_records"]),
            "temporal_records": len(evaluation["temporal_records"]),
            "temporal_episode_records": len(
                evaluation["temporal_episode_records"]
            ),
        },
        "runtime": _runtime_record(),
        "git": _git_record(),
        "next_action": (
            "Stop and ask the assistant to audit the baseline-state imitation result. "
            "Do not start student-state aggregation, RL, S4-D3, or real SLM actions."
        ),
    }
    _write_json(output_directory / "summary.json", json_safe(summary))
    return summary


def preflight_s4_baseline_state_imitation(
    experiment_path: Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    """锁定单因素数据源变更、完整回合拆分和硬件安全边界。"""
    metadata = experiment["metadata"]
    if str(metadata["stage"]) != "S4-D2-R2-BASELINE-STATE-IMITATION":
        raise ValueError("baseline-state-imitation stage metadata is invalid")
    expected_flags = {
        "allow_supervised_training": True,
        "allow_rl_training": False,
        "allow_reward_change": False,
        "allow_model_architecture_change": False,
        "allow_teacher_trajectory_training_samples": False,
        "allow_student_action_training_samples": False,
        "allow_future_truth_in_teacher_labels": True,
        "allow_future_truth_in_model_inputs": False,
        "allow_hardware_profile_id_in_model_inputs": False,
        "allow_s4d3_access": False,
        "allow_real_hardware_actions": False,
    }
    for key, expected in expected_flags.items():
        if bool(metadata.get(key)) is not expected:
            raise RuntimeError(f"invalid baseline-state-imitation safety flag: {key}")

    upstream = _verify_upstream_shift(experiment["upstream_shift"])
    for field in ("environment_config", "hardware_profile_source"):
        path = _project_path(experiment[field])
        if not _matches_exact_or_lf_canonical(
            path, str(experiment[f"{field}_sha256"])
        ):
            raise RuntimeError(f"baseline-state-imitation {field} hash mismatch")

    representation = experiment["representation"]
    if (
        str(representation["id"]) != "zernike_21_added_11"
        or str(representation["kind"]) != "zernike"
        or int(representation["num_modes"]) != 21
        or int(representation["anchor_modes"]) != ANCHOR_MODES
        or int(representation["output_modes"]) != 11
        or not 0 < float(representation["pseudoinverse_rcond"]) < 1
    ):
        raise RuntimeError("baseline-state imitation must keep the added 11 modes")
    observation = experiment["policy_observation"]
    if (
        int(observation["history_frames"]) != 4
        or int(observation["state_size"]) != 210
        or bool(observation["include_future_truth"])
        or bool(observation["include_oracle_quality_metrics"])
        or bool(observation["include_hardware_profile_identifier"])
    ):
        raise RuntimeError("baseline-state observation contract changed")
    if int(experiment["teacher"]["preview_horizon_frames"]) != 2:
        raise RuntimeError("baseline-state teacher must keep two-frame preview")
    if str(experiment["dataset"]["state_source"]) != "traditional_baseline_trajectory":
        raise RuntimeError("training data source is not the traditional baseline trajectory")
    if int(experiment["model"]["hidden_size"]) != 256:
        raise RuntimeError("formal model architecture must remain the 256-wide MLP")
    if int(experiment["model"]["output_size"]) != 11:
        raise RuntimeError("model must output exactly the added 11 modes")

    _validate_training_settings(settings)
    split_evidence = _validate_seed_splits(experiment, settings)
    _validate_deployment_settings(experiment, settings)
    reference_items = _selected_reference_predictors(experiment, settings)
    inventory = _checkpoint_inventory(reference_items)
    output_directory = _project_path(settings["output_directory"])
    if output_directory.exists():
        raise FileExistsError(
            f"baseline-state-imitation output already exists: {output_directory}"
        )

    model = HighOrderImitationPolicy(
        int(observation["state_size"]),
        int(settings["hidden_size"]),
        11,
    )
    parameter_count = sum(value.numel() for value in model.parameters())
    samples = {
        split: (
            len(settings[f"{split}_conditions"])
            * len(settings["profile_ids"])
            * int(settings[f"{split}_episodes_per_condition"])
            * int(settings["steps_per_episode"])
        )
        for split in ("training", "validation", "diagnostic_test")
    }
    predictor_count = len(reference_items) + 1 + len(settings["initialization_seeds"])
    scenario_count = (
        predictor_count
        * len(settings["deployment_scales"])
        * len(settings["profile_ids"])
        * len(settings["diagnostic_test_conditions"])
    )
    episode_count = scenario_count * int(
        settings["diagnostic_test_episodes_per_condition"]
    )
    return {
        "status": "READY_FOR_USER_BASELINE_STATE_SUPERVISED_TRAINING",
        "quick": quick,
        "experiment_config": _relative(experiment_path),
        "experiment_config_sha256": _file_sha256(experiment_path),
        "output_directory": settings["output_directory"],
        "upstream_shift_summary_sha256": upstream["summary_sha256"],
        "upstream_shift_audit_canonical_sha256": upstream["audit_sha256"],
        "text_evidence_hash_mode": "exact_or_lf_canonical",
        "reference_predictor_inventory": inventory,
        "training_state_source": "traditional_baseline_trajectory",
        "student_state_aggregation_rounds": 0,
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
        "deployment_scales": settings["deployment_scales"],
        "primary_scale": settings["primary_scale"],
        "planned_records": {
            "scenario_records": scenario_count,
            "episode_records": episode_count,
            "temporal_records": scenario_count * len(settings["time_bins"]),
            "temporal_episode_records": episode_count * len(settings["time_bins"]),
            "control_episode_records": 2
            * len(settings["profile_ids"])
            * len(settings["diagnostic_test_conditions"])
            * int(settings["diagnostic_test_episodes_per_condition"]),
        },
        "cuda_required": True,
        "supervised_training_allowed": True,
        "rl_training_allowed": False,
        "future_truth_in_teacher_labels": True,
        "future_truth_in_model_inputs": False,
        "s4d3_accessed": False,
        "real_slm_actions": False,
        "formal_command": (
            ".\\.venv\\Scripts\\python.exe "
            "scripts\\train_s4_baseline_state_imitation.py "
            "--config configs\\experiments\\s4_baseline_state_imitation_v1.yaml"
        ),
    }


def _generate_baseline_state_split(
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
    keep_context: bool,
) -> BaselineStateSplit:
    condition_list = [RobustnessCondition.from_mapping(item) for item in conditions]
    steps = int(settings["steps_per_episode"])
    preview = int(settings["preview_horizon_frames"])
    total = len(profiles) * len(condition_list) * steps
    bar = counted_progress(
        total=total,
        description=f"生成{split_name}基线状态回合",
        unit="步",
    )
    all_states: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    baselines: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    futures: dict[tuple[str, str], torch.Tensor] = {}
    configs: dict[tuple[str, str], S1EnvConfig] = {}
    completed = 0
    limit = float(experiment["action_budget"]["residual_component_limit_rad"])
    for profile in profiles:
        for condition in condition_list:
            config = _scenario_config(
                base_config,
                condition,
                profile,
                episodes=episodes_per_condition,
                steps=steps,
                preview=preview,
            )
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
            controller = _student_controller(experiment, config)
            state = controller.reset(noisy)
            scenario_states: list[torch.Tensor] = []
            scenario_targets: list[torch.Tensor] = []
            science = _empty_step_metrics()
            for step in range(steps):
                target_request = (
                    -future_truth[step + preview]
                    @ registration_mappings[profile.identifier].transpose(0, 1)
                )
                oracle = _oracle_normalized_action(
                    state,
                    target_request,
                    residual_component_limit_rad=limit,
                    residual_l2_budget_rad=controller.residual_l2_budget_rad,
                )
                scenario_states.append(state.detach().cpu())
                scenario_targets.append(oracle.detach().cpu())
                normalized = torch.zeros(
                    config.batch_size,
                    config.num_modes,
                    device=device,
                    dtype=state.dtype,
                )
                action = controller.compose_action(normalized)
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
            key = (profile.identifier, condition.identifier)
            if keep_context:
                baselines[key] = _mean_step_metrics(science)
                futures[key] = future_truth
                configs[key] = config
            _append_jsonl(
                progress_path,
                {
                    "phase": "baseline_state_data",
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
    bar.close()
    dataset = TeacherDataset(
        states=torch.cat(all_states),
        targets=torch.cat(all_targets),
        episodes=len(profiles) * len(condition_list) * episodes_per_condition,
        steps_per_episode=steps,
    )
    if len(dataset.states) != dataset.episodes * steps:
        raise RuntimeError("baseline-state sample count does not match complete episodes")
    if not bool(torch.isfinite(dataset.states).all()) or not bool(
        torch.isfinite(dataset.targets).all()
    ):
        raise RuntimeError("baseline-state dataset contains non-finite values")
    return BaselineStateSplit(dataset, baselines, futures, configs)


def _evaluate_all(
    *,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    profiles: list[HardwareProfile],
    predictors: list[FrozenPredictor],
    offline_results: dict[str, dict[str, float]],
    registration_mappings: dict[str, torch.Tensor],
    basis: torch.Tensor,
    test_context: BaselineStateSplit,
    device: torch.device,
    progress_path: Path,
) -> dict[str, Any]:
    conditions = [
        RobustnessCondition.from_mapping(item)
        for item in settings["diagnostic_test_conditions"]
    ]
    steps = int(settings["steps_per_episode"])
    total = len(profiles) * len(conditions) * (
        2 + len(predictors) * len(settings["deployment_scales"])
    )
    bar = counted_progress(total=total, description="基线状态监督闭环评估", unit="场景")
    completed = 0
    teacher_scenarios: list[TeacherScenario] = []
    zero_scenarios: list[ShiftScenario] = []
    scenarios: list[ShiftScenario] = []
    control_episode_records: list[dict[str, Any]] = []
    scenario_records: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    temporal_records: list[dict[str, Any]] = []
    temporal_episode_records: list[dict[str, Any]] = []
    projection_tolerance = float(
        experiment["interpretation_thresholds"]["projection_comparison_tolerance_rad"]
    )

    zero_predictor = next(
        item for item in predictors if item.identifier == "baseline_ridge"
    )
    for profile in profiles:
        for condition in conditions:
            key = (profile.identifier, condition.identifier)
            baseline = test_context.baselines[key]
            teacher, label_delta = _rollout_teacher(
                experiment=experiment,
                config=test_context.configs[key],
                condition=condition,
                profile=profile,
                steps=steps,
                basis=basis,
                future_truth=test_context.future_truth[key],
                registration_mapping=registration_mappings[profile.identifier],
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
            rows = _episode_rows(
                "teacher_fixed_preview_2",
                profile.identifier,
                condition.identifier,
                condition.base_seed,
                teacher,
                baseline,
            )
            for index, row in enumerate(rows):
                row["max_requested_minus_realized_added_abs_rad"] = float(
                    label_delta[index]
                )
            control_episode_records.extend(rows)
            completed += 1
            _progress_event(
                bar,
                progress_path,
                completed,
                total,
                phase="teacher_positive_control",
                profile=profile.identifier,
                condition=condition.identifier,
                device=device,
                metric=float(
                    (teacher["power_in_bucket"] - baseline["power_in_bucket"]).mean()
                ),
            )

            zero_candidate, zero_temporal = _rollout_predictor(
                frozen=zero_predictor,
                scale=0.0,
                experiment=experiment,
                config=test_context.configs[key],
                condition=condition,
                profile=profile,
                steps=steps,
                time_bins=settings["time_bins"],
                basis=basis,
                future_truth=test_context.future_truth[key],
                registration_mapping=registration_mappings[profile.identifier],
                projection_tolerance=projection_tolerance,
                device=device,
            )
            zero_item = ShiftScenario(
                predictor_id="zero_alignment",
                scale=0.0,
                profile_id=profile.identifier,
                condition_id=condition.identifier,
                base_seed=condition.base_seed,
                candidate=zero_candidate,
                baseline=baseline,
                temporal=zero_temporal,
            )
            zero_scenarios.append(zero_item)
            zero_rows = _episode_rows(
                "zero_alignment",
                profile.identifier,
                condition.identifier,
                condition.base_seed,
                zero_candidate,
                baseline,
            )
            for row in zero_rows:
                row["max_requested_minus_realized_added_abs_rad"] = 0.0
            control_episode_records.extend(zero_rows)
            completed += 1
            _progress_event(
                bar,
                progress_path,
                completed,
                total,
                phase="zero_alignment",
                profile=profile.identifier,
                condition=condition.identifier,
                device=device,
                metric=float(
                    (
                        zero_candidate["power_in_bucket"]
                        - baseline["power_in_bucket"]
                    ).mean()
                ),
            )

    for predictor in predictors:
        for scale in settings["deployment_scales"]:
            for profile in profiles:
                for condition in conditions:
                    key = (profile.identifier, condition.identifier)
                    candidate, temporal = _rollout_predictor(
                        frozen=predictor,
                        scale=float(scale),
                        experiment=experiment,
                        config=test_context.configs[key],
                        condition=condition,
                        profile=profile,
                        steps=steps,
                        time_bins=settings["time_bins"],
                        basis=basis,
                        future_truth=test_context.future_truth[key],
                        registration_mapping=registration_mappings[profile.identifier],
                        projection_tolerance=projection_tolerance,
                        device=device,
                    )
                    item = ShiftScenario(
                        predictor_id=predictor.identifier,
                        scale=float(scale),
                        profile_id=profile.identifier,
                        condition_id=condition.identifier,
                        base_seed=condition.base_seed,
                        candidate=candidate,
                        baseline=test_context.baselines[key],
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
                        total,
                        phase="predictor_scale_rollout",
                        profile=profile.identifier,
                        condition=condition.identifier,
                        predictor=predictor.identifier,
                        scale=float(scale),
                        device=device,
                        metric=float(
                            (
                                candidate["power_in_bucket"]
                                - test_context.baselines[key]["power_in_bucket"]
                            ).mean()
                        ),
                    )
    bar.close()

    grouped: dict[str, dict[str, Any]] = {}
    for predictor in predictors:
        grouped[predictor.identifier] = {}
        for scale in settings["deployment_scales"]:
            selected = [
                item
                for item in scenarios
                if item.predictor_id == predictor.identifier
                and math.isclose(item.scale, float(scale), abs_tol=1e-12)
            ]
            grouped[predictor.identifier][_scale_key(float(scale))] = (
                _summarize_shift_scenarios(
                    selected,
                    gate=experiment["closed_loop_gate"],
                    time_bins=settings["time_bins"],
                    thresholds=experiment["interpretation_thresholds"],
                    offline_reference=offline_results[predictor.identifier],
                )
            )
    return {
        "teacher_summary": _summarize_teacher_scenarios(
            teacher_scenarios,
            gate=experiment["closed_loop_gate"],
        ),
        "zero_alignment": _zero_alignment(
            zero_scenarios,
            tolerance=float(
                experiment["interpretation_thresholds"][
                    "zero_alignment_max_abs_error"
                ]
            ),
        ),
        "grouped": grouped,
        "control_episode_records": control_episode_records,
        "scenario_records": scenario_records,
        "episode_records": episode_records,
        "temporal_records": temporal_records,
        "temporal_episode_records": temporal_episode_records,
    }


def _interpretation(
    *,
    grouped: dict[str, dict[str, Any]],
    offline_results: dict[str, dict[str, float]],
    teacher_summary: dict[str, Any],
    experiment: dict[str, Any],
    settings: dict[str, Any],
    quick: bool,
) -> dict[str, Any]:
    threshold = float(experiment["data_source_gate"]["max_new_to_old_mse_ratio"])
    improvement_threshold = float(
        experiment["data_source_gate"]["min_primary_gain_improvement_over_old"]
    )
    primary_key = _scale_key(float(settings["primary_scale"]))
    comparisons: list[dict[str, Any]] = []
    for seed in settings["initialization_seeds"]:
        old_id = f"teacher_mlp_seed_{int(seed)}"
        new_id = f"baseline_mlp_seed_{int(seed)}"
        old_offline = offline_results[old_id]
        new_offline = offline_results[new_id]
        old_primary = grouped[old_id][primary_key]
        new_primary = grouped[new_id][primary_key]
        comparisons.append(
            {
                "initialization_seed": int(seed),
                "old_predictor_id": old_id,
                "new_predictor_id": new_id,
                "old_baseline_state_test_mse": old_offline["mse"],
                "new_baseline_state_test_mse": new_offline["mse"],
                "new_to_old_mse_ratio": float(new_offline["mse"])
                / max(float(old_offline["mse"]), 1e-12),
                "offline_gate": _offline_gate(
                    new_offline, experiment["offline_gate"]
                ),
                "old_primary_relative_power_gain": old_primary["overall"][
                    "relative_power_gain"
                ],
                "new_primary_relative_power_gain": new_primary["overall"][
                    "relative_power_gain"
                ],
                "primary_gain_improvement_over_old": float(
                    new_primary["overall"]["relative_power_gain"]
                )
                - float(old_primary["overall"]["relative_power_gain"]),
                "new_primary_closed_loop_gate": new_primary["closed_loop_gate"],
            }
        )
    offline_corrected = all(
        item["offline_gate"]
        and float(item["new_to_old_mse_ratio"]) <= threshold
        for item in comparisons
    )
    primary_improved = all(
        float(item["primary_gain_improvement_over_old"]) >= improvement_threshold
        for item in comparisons
    )
    primary_pass = all(
        item["new_primary_closed_loop_gate"] == "PASS" for item in comparisons
    )
    safe_scales = [
        float(scale)
        for scale in settings["deployment_scales"]
        if all(
            grouped[f"baseline_mlp_seed_{int(seed)}"][_scale_key(float(scale))][
                "closed_loop_gate"
            ]
            == "PASS"
            for seed in settings["initialization_seeds"]
        )
    ]
    if quick:
        status = "QUICK_SMOKE_ONLY"
    elif teacher_summary["closed_loop_gate"] != "PASS":
        status = "TEACHER_POSITIVE_CONTROL_FAILED"
    elif offline_corrected and primary_improved and primary_pass:
        status = "BASELINE_STATE_CORRECTION_PASS"
    elif offline_corrected and safe_scales:
        status = "NONPRIMARY_SAFE_SCALE_FOUND_REQUIRES_CONFIRMATION"
    elif offline_corrected:
        status = "OFFLINE_CORRECTED_BUT_CLOSED_LOOP_FAILED"
    else:
        status = "BASELINE_STATE_DATA_NOT_LEARNABLE"
    return {
        "status": status,
        "teacher_positive_control_pass": teacher_summary["closed_loop_gate"]
        == "PASS",
        "baseline_state_offline_correction_pass": offline_corrected,
        "primary_scale": float(settings["primary_scale"]),
        "primary_scale_improvement_pass": primary_improved,
        "primary_scale_closed_loop_pass": primary_pass,
        "safe_scales_all_mlp": safe_scales,
        "initialization_comparisons": comparisons,
        "new_rl_training_authorized": False,
        "student_state_aggregation_authorized": False,
        "s4d3_authorized": False,
        "real_hardware_authorized": False,
        "next_rule": (
            "Audit first. Only an audited offline correction without closed-loop "
            "recovery may authorize one bounded student-state aggregation round."
        ),
    }


def _verify_upstream_shift(upstream: dict[str, Any]) -> dict[str, str]:
    exact_fields = (
        "summary",
        "preflight",
        "effective_config",
        "source_manifest",
        "control_episode_records",
        "scenario_records",
        "episode_records",
        "temporal_records",
        "temporal_episode_records",
    )
    paths: dict[str, Path] = {}
    for field in exact_fields:
        path = _project_path(upstream[field])
        if _file_sha256(path) != str(upstream[f"{field}_sha256"]):
            raise RuntimeError(f"closed-loop-shift evidence hash mismatch: {field}")
        paths[field] = path
    for field in ("experiment_config", "audit_record"):
        path = _project_path(upstream[field])
        if not _matches_exact_or_lf_canonical(
            path, str(upstream[f"{field}_sha256"])
        ):
            raise RuntimeError(f"closed-loop-shift text evidence mismatch: {field}")
        paths[field] = path
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    interpretation = summary["interpretation"]
    if (
        bool(summary["experiment"]["quick"])
        or summary["experiment"]["status"] != "completed_pending_audit"
        or interpretation["status"]
        != "CLOSED_LOOP_DISTRIBUTION_SHIFT_SUPPORTED"
        or int(interpretation["safe_reduced_scale_recovery_count"]) != 0
        or not bool(interpretation["teacher_positive_control_pass"])
        or summary["integrity"]["zero_scale_alignment"]["status"] != "PASS"
    ):
        raise RuntimeError("formal closed-loop-shift result is not usable")
    audit = paths["audit_record"].read_text(encoding="utf-8")
    if (
        "Verification Status: `ANALYZED`" not in audit
        or "CLOSED_LOOP_DISTRIBUTION_SHIFT_SUPPORTED / NO_SAFE_SCALE_RECOVERY"
        not in audit
        or "闭环状态数据聚合实验设计：`AUTHORIZED`" not in audit
    ):
        raise RuntimeError("closed-loop-shift audit is not finalized")
    manifest = json.loads(paths["source_manifest"].read_text(encoding="utf-8"))
    for relative, digest in manifest.items():
        path = _project_path(relative)
        if not path.is_file() or not _matches_exact_or_lf_canonical(
            path, str(digest)
        ):
            raise RuntimeError(f"closed-loop-shift source changed: {relative}")
    return {
        "summary_sha256": _file_sha256(paths["summary"]),
        "audit_sha256": _lf_canonical_sha256(paths["audit_record"]),
    }


def _validate_deployment_settings(
    experiment: dict[str, Any], settings: dict[str, Any]
) -> None:
    scales = list(map(float, settings["deployment_scales"]))
    if scales != sorted(set(scales)) or any(not 0 < value <= 1 for value in scales):
        raise RuntimeError("deployment scales must be unique, sorted, and in (0, 1]")
    if float(settings["primary_scale"]) not in scales:
        raise RuntimeError("primary scale must be one of the deployment scales")
    steps = int(settings["steps_per_episode"])
    cursor = 0
    for item in settings["time_bins"]:
        if int(item["start"]) != cursor or int(item["stop"]) <= cursor:
            raise RuntimeError("time bins must be contiguous and increasing")
        cursor = int(item["stop"])
    if cursor != steps:
        raise RuntimeError("time bins must cover the complete episode")
    if float(experiment["data_source_gate"]["max_new_to_old_mse_ratio"]) <= 0:
        raise ValueError("offline MSE ratio threshold must be positive")


def _effective_settings(experiment: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    dataset = experiment["dataset"]
    training = experiment["training"]
    evaluation = experiment["evaluation"]
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
        "deployment_scales": list(map(float, evaluation["deployment_scales"])),
        "primary_scale": float(evaluation["primary_scale"]),
        "time_bins": list(evaluation["time_bins"]),
        "reference_predictor_ids": [
            str(item["id"]) for item in experiment["reference_predictors"]
        ],
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
                "validation_conditions": list(
                    quick_settings["validation_conditions"]
                ),
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
                "deployment_scales": list(
                    map(float, quick_settings["deployment_scales"])
                ),
                "primary_scale": float(quick_settings["primary_scale"]),
                "time_bins": list(quick_settings["time_bins"]),
                "reference_predictor_ids": list(
                    map(str, quick_settings["reference_predictor_ids"])
                ),
            }
        )
    return settings


def _selected_reference_predictors(
    experiment: dict[str, Any], settings: dict[str, Any]
) -> list[dict[str, Any]]:
    by_id = {str(item["id"]): item for item in experiment["reference_predictors"]}
    identifiers = list(settings["reference_predictor_ids"])
    if any(identifier not in by_id for identifier in identifiers):
        raise RuntimeError("unknown reference predictor identifier")
    return [by_id[identifier] for identifier in identifiers]


def _matches_exact_or_lf_canonical(path: Path, expected: str) -> bool:
    return _file_sha256(path) == expected or _lf_canonical_sha256(path) == expected


def _lf_canonical_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def _replace_num_modes(config: S1EnvConfig, num_modes: int) -> S1EnvConfig:
    from dataclasses import replace

    return replace(config, num_modes=num_modes)
