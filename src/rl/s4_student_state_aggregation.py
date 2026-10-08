"""S4-D2-R2单轮受限学生状态数据聚合监督实验。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

import torch

from src.rl.s4_baseline_state_imitation import (
    BaselineStateSplit,
    _evaluate_all,
    _generate_baseline_state_split,
    _matches_exact_or_lf_canonical,
    _replace_num_modes,
)
from src.rl.s4_closed_loop_shift import (
    FrozenPredictor,
    _checkpoint_inventory,
    _load_predictors,
    _oracle_normalized_action,
    _scenario_config,
    _scale_key,
)
from src.rl.s4_high_order_learnability import (
    HighOrderImitationPolicy,
    TeacherDataset,
    _append_jsonl,
    _dataset_payload,
    _fit_ridge,
    _model_predictor,
    _offline_gate,
    _offline_metrics,
    _ridge_predictor,
    _student_controller,
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
class StudentAggregationSplit:
    """学生行为策略走访状态、教师标签和逐回合采集记录。"""

    dataset: TeacherDataset
    collection_records: list[dict[str, Any]]


@dataclass(frozen=True)
class TrainedArm:
    """一个同数据预算训练分支的冻结预测器与训练记录。"""

    predictors: list[FrozenPredictor]
    ridge_record: dict[str, Any]
    model_records: list[dict[str, Any]]
    state_mean: torch.Tensor
    state_scale: torch.Tensor


def run_s4_student_state_aggregation(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """执行一次学生状态采集、同预算监督训练和配对闭环评估。"""
    experiment_path = _project_path(config_path)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight = preflight_s4_student_state_aggregation(
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
            "student-state-aggregation output already exists; preserve it for audit: "
            f"{output_directory}"
        )
    output_directory.mkdir(parents=True)
    progress_path = output_directory / "progress.jsonl"
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

    upstream_data = torch.load(
        _project_path(experiment["upstream_baseline"]["dataset"]),
        map_location="cpu",
        weights_only=False,
    )
    original = {
        "training": _dataset_from_payload(upstream_data["train"]),
        "validation": _dataset_from_payload(upstream_data["validation"]),
        "diagnostic_test": _dataset_from_payload(
            upstream_data["diagnostic_test"]
        ),
    }
    if quick:
        original = {
            key: _subset_complete_episodes(
                value,
                episodes=int(settings["upstream_quick_episodes_per_split"]),
                steps=int(settings["steps_per_episode"]),
            )
            for key, value in original.items()
        }
    reference_items = _selected_predictors(
        experiment["reference_predictors"],
        settings["reference_predictor_ids"],
    )
    references = _load_predictors({"predictors": reference_items}, device=device)
    behavior_by_id = {
        item.identifier: item
        for item in references
        if item.identifier in settings["behavior_predictor_ids"]
    }
    if set(behavior_by_id) != set(settings["behavior_predictor_ids"]):
        raise RuntimeError("student behavior predictor inventory is incomplete")
    behavior_predictors = [
        behavior_by_id[identifier]
        for identifier in settings["behavior_predictor_ids"]
    ]

    matched: dict[str, BaselineStateSplit] = {}
    student: dict[str, StudentAggregationSplit] = {}
    for split in ("training", "validation", "diagnostic_test"):
        conditions = settings[f"{split}_conditions"]
        episodes = int(settings[f"{split}_episodes_per_condition"])
        matched[split] = _generate_baseline_state_split(
            split_name=f"{split}_matched_baseline",
            experiment=experiment,
            settings=settings,
            base_config=base_config,
            profiles=profiles,
            conditions=conditions,
            episodes_per_condition=episodes,
            basis=basis,
            registration_mappings=mappings,
            device=device,
            progress_path=progress_path,
            keep_context=False,
        )
        student[split] = _generate_student_aggregation_split(
            split_name=split,
            experiment=experiment,
            settings=settings,
            base_config=base_config,
            profiles=profiles,
            conditions=conditions,
            episodes_per_condition=episodes,
            basis=basis,
            registration_mappings=mappings,
            behavior_predictors=behavior_predictors,
            device=device,
            progress_path=progress_path,
        )

    dataset_path = output_directory / "student_state_aggregation_dataset.pt"
    torch.save(
        {
            "matched_baseline_extra": {
                key: _dataset_payload(value.dataset)
                for key, value in matched.items()
            },
            "student_aggregation_round_1": {
                key: _dataset_payload(value.dataset)
                for key, value in student.items()
            },
            "upstream_dataset": experiment["upstream_baseline"]["dataset"],
            "upstream_dataset_sha256": experiment["upstream_baseline"][
                "dataset_sha256"
            ],
            "aggregation_rounds": 1,
            "behavior_scales": settings["behavior_scales"],
            "behavior_predictor_ids": settings["behavior_predictor_ids"],
            "future_truth_in_targets": True,
            "future_truth_in_inputs": False,
            "complete_episode_splits": True,
        },
        dataset_path,
    )
    collection_records = [
        row for split in student.values() for row in split.collection_records
    ]
    _write_csv(output_directory / "collection_records.csv", collection_records)

    control_data = {
        split: _concatenate_datasets(original[split], matched[split].dataset)
        for split in original
    }
    aggregate_data = {
        split: _concatenate_datasets(original[split], student[split].dataset)
        for split in original
    }
    control_arm = _train_arm(
        arm_id="matched_control",
        train=control_data["training"],
        validation=control_data["validation"],
        diagnostic_test=student["diagnostic_test"].dataset,
        experiment=experiment,
        settings=settings,
        device=device,
        output_directory=output_directory,
        progress_path=progress_path,
    )
    aggregate_arm = _train_arm(
        arm_id="student_aggregate",
        train=aggregate_data["training"],
        validation=aggregate_data["validation"],
        diagnostic_test=student["diagnostic_test"].dataset,
        experiment=experiment,
        settings=settings,
        device=device,
        output_directory=output_directory,
        progress_path=progress_path,
    )

    all_predictors = references + control_arm.predictors + aggregate_arm.predictors
    offline_student = {
        item.identifier: _offline_metrics(
            item.predict,
            student["diagnostic_test"].dataset,
            device=device,
        )
        for item in all_predictors
    }
    offline_matched = {
        item.identifier: _offline_metrics(
            item.predict,
            matched["diagnostic_test"].dataset,
            device=device,
        )
        for item in all_predictors
    }
    offline_original = {
        item.identifier: _offline_metrics(
            item.predict,
            original["diagnostic_test"],
            device=device,
        )
        for item in all_predictors
    }

    evaluation_context = _generate_baseline_state_split(
        split_name="paired_closed_loop_evaluation",
        experiment=experiment,
        settings=settings,
        base_config=base_config,
        profiles=profiles,
        conditions=settings["evaluation_conditions"],
        episodes_per_condition=int(settings["evaluation_episodes_per_condition"]),
        basis=basis,
        registration_mappings=mappings,
        device=device,
        progress_path=progress_path,
        keep_context=True,
    )
    evaluation_settings = {
        **settings,
        "diagnostic_test_conditions": settings["evaluation_conditions"],
        "diagnostic_test_episodes_per_condition": settings[
            "evaluation_episodes_per_condition"
        ],
    }
    evaluation = _evaluate_all(
        experiment=experiment,
        settings=evaluation_settings,
        profiles=profiles,
        predictors=all_predictors,
        offline_results=offline_student,
        registration_mappings=mappings,
        basis=basis,
        test_context=evaluation_context,
        device=device,
        progress_path=progress_path,
    )
    coverage = {
        "matched_baseline_extra_training": _state_coverage(
            matched["training"].dataset
        ),
        "student_aggregation_training": _state_coverage(
            student["training"].dataset
        ),
        "matched_control_combined_training": _state_coverage(
            control_data["training"]
        ),
        "student_aggregate_combined_training": _state_coverage(
            aggregate_data["training"]
        ),
    }
    interpretation = _interpretation(
        grouped=evaluation["grouped"],
        offline_student=offline_student,
        coverage=coverage,
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
            "version_label": "s4d2_r2_student_state_aggregation_v1",
        },
        "experiment": {
            "id": "AO-S4-D2-R2-STUDENT-STATE-AGGREGATION-R1",
            "type": "software_only_cuda_single_round_supervised_aggregation",
            "status": "completed_pending_audit",
            "quick": quick,
            "duration_seconds": time.perf_counter() - started,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "evidence_boundary": {
            "supervised_training_performed": True,
            "rl_training_performed": False,
            "aggregation_rounds": 1,
            "future_truth_in_teacher_labels": True,
            "future_truth_in_model_inputs": False,
            "behavior_checkpoints_frozen": True,
            "matched_data_budget_control": True,
            "complete_episode_splits": True,
            "s4d3_accessed": False,
            "real_slm_actions": False,
        },
        "inputs": {
            "experiment_config": _relative(experiment_path),
            "experiment_config_sha256": _file_sha256(experiment_path),
            "upstream_summary_sha256": experiment["upstream_baseline"][
                "summary_sha256"
            ],
            "upstream_audit_sha256": experiment["upstream_baseline"][
                "audit_record_sha256"
            ],
            "upstream_dataset_sha256": experiment["upstream_baseline"][
                "dataset_sha256"
            ],
            "aggregation_dataset": _relative(dataset_path),
            "aggregation_dataset_sha256": _file_sha256(dataset_path),
        },
        "design": {
            "representation": experiment["representation"],
            "policy_observation": experiment["policy_observation"],
            "behavior_scales": settings["behavior_scales"],
            "behavior_predictor_ids": settings["behavior_predictor_ids"],
            "deployment_scales": settings["deployment_scales"],
            "primary_scale": settings["primary_scale"],
            "matched_control_samples": {
                key: len(value.states) for key, value in control_data.items()
            },
            "student_aggregate_samples": {
                key: len(value.states) for key, value in aggregate_data.items()
            },
        },
        "basis_diagnostics": basis_diagnostics,
        "mapping_diagnostics": mapping_diagnostics,
        "state_coverage": coverage,
        "training": {
            "matched_control": {
                "ridge": control_arm.ridge_record,
                "mlp_initializations": control_arm.model_records,
            },
            "student_aggregate": {
                "ridge": aggregate_arm.ridge_record,
                "mlp_initializations": aggregate_arm.model_records,
            },
        },
        "offline_diagnostics": {
            "student_state_test": offline_student,
            "matched_baseline_test": offline_matched,
            "original_baseline_test": offline_original,
        },
        "teacher_positive_control": evaluation["teacher_summary"],
        "zero_scale_alignment": evaluation["zero_alignment"],
        "predictor_scale_results": evaluation["grouped"],
        "interpretation": interpretation,
        "record_counts": {
            "collection_records": len(collection_records),
            "control_episode_records": len(
                evaluation["control_episode_records"]
            ),
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
            "Stop and ask the assistant to audit this single aggregation round. "
            "Do not run another aggregation round, RL, S4-D3, or real SLM actions."
        ),
    }
    _write_json(output_directory / "summary.json", json_safe(summary))
    return summary


def preflight_s4_student_state_aggregation(
    experiment_path: Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    """在创建输出前锁定单轮、同预算、低动作采集和安全边界。"""
    metadata = experiment["metadata"]
    if str(metadata["stage"]) != "S4-D2-R2-STUDENT-STATE-AGGREGATION-R1":
        raise ValueError("student-state aggregation stage metadata is invalid")
    expected_flags = {
        "allow_supervised_training": True,
        "allow_rl_training": False,
        "allow_more_than_one_aggregation_round": False,
        "allow_reward_change": False,
        "allow_model_architecture_change": False,
        "allow_future_truth_in_teacher_labels": True,
        "allow_future_truth_in_model_inputs": False,
        "allow_s4d3_access": False,
        "allow_real_hardware_actions": False,
    }
    for key, expected in expected_flags.items():
        if bool(metadata.get(key)) is not expected:
            raise RuntimeError(f"invalid student-state aggregation flag: {key}")
    if int(experiment["aggregation"]["rounds"]) != 1:
        raise RuntimeError("exactly one student-state aggregation round is allowed")

    upstream = _verify_upstream_baseline(experiment["upstream_baseline"])
    for field in ("environment_config", "hardware_profile_source"):
        path = _project_path(experiment[field])
        if not _matches_exact_or_lf_canonical(
            path, str(experiment[f"{field}_sha256"])
        ):
            raise RuntimeError(f"student-state aggregation {field} hash mismatch")
    representation = experiment["representation"]
    if (
        str(representation["id"]) != "zernike_21_added_11"
        or int(representation["num_modes"]) != 21
        or int(representation["anchor_modes"]) != ANCHOR_MODES
        or int(representation["output_modes"]) != 11
    ):
        raise RuntimeError("student-state aggregation must keep the added 11 modes")
    observation = experiment["policy_observation"]
    if (
        int(observation["history_frames"]) != 4
        or int(observation["state_size"]) != 210
        or bool(observation["include_future_truth"])
        or bool(observation["include_hardware_profile_identifier"])
    ):
        raise RuntimeError("student-state aggregation observation contract changed")
    if int(experiment["model"]["hidden_size"]) != 256:
        raise RuntimeError("formal model must remain the 256-wide MLP")
    if int(experiment["model"]["output_size"]) != 11:
        raise RuntimeError("student-state model must output 11 added modes")

    _validate_training_settings(settings)
    split_evidence = _validate_seed_splits(experiment, settings)
    evaluation_evidence = _validate_evaluation_seeds(experiment, settings)
    _validate_scales(experiment, settings)
    reference_items = _selected_predictors(
        experiment["reference_predictors"],
        settings["reference_predictor_ids"],
    )
    inventory = _checkpoint_inventory(reference_items)
    if set(settings["behavior_predictor_ids"]) - {
        str(item["id"]) for item in reference_items
    }:
        raise RuntimeError("behavior predictors must be frozen references")
    output_directory = _project_path(settings["output_directory"])
    if output_directory.exists():
        raise FileExistsError(
            f"student-state aggregation output already exists: {output_directory}"
        )

    model = HighOrderImitationPolicy(210, int(settings["hidden_size"]), 11)
    parameter_count = sum(value.numel() for value in model.parameters())
    extra_samples = {
        split: (
            len(settings[f"{split}_conditions"])
            * len(settings["profile_ids"])
            * int(settings[f"{split}_episodes_per_condition"])
            * int(settings["steps_per_episode"])
        )
        for split in ("training", "validation", "diagnostic_test")
    }
    upstream_samples = (
        {
            split: int(settings["upstream_quick_episodes_per_split"])
            * int(settings["steps_per_episode"])
            for split in ("training", "validation", "diagnostic_test")
        }
        if quick
        else upstream["dataset_samples"]
    )
    combined_fit_samples = {
        split: int(upstream_samples[split]) + extra_samples[split]
        for split in ("training", "validation")
    }
    diagnostic_panel_samples = {
        "student_state": extra_samples["diagnostic_test"],
        "matched_extra_baseline_state": extra_samples["diagnostic_test"],
        "original_baseline_state": int(upstream_samples["diagnostic_test"]),
    }
    predictor_count = len(reference_items) + 2 * len(
        settings["initialization_seeds"]
    )
    scenario_count = (
        predictor_count
        * len(settings["deployment_scales"])
        * len(settings["profile_ids"])
        * len(settings["evaluation_conditions"])
    )
    episode_count = scenario_count * int(
        settings["evaluation_episodes_per_condition"]
    )
    control_count = (
        2
        * len(settings["profile_ids"])
        * len(settings["evaluation_conditions"])
        * int(settings["evaluation_episodes_per_condition"])
    )
    return {
        "status": "READY_FOR_USER_SINGLE_ROUND_STUDENT_STATE_TRAINING",
        "quick": quick,
        "experiment_config": _relative(experiment_path),
        "experiment_config_sha256": _file_sha256(experiment_path),
        "output_directory": settings["output_directory"],
        "upstream_summary_sha256": upstream["summary_sha256"],
        "upstream_audit_sha256": upstream["audit_sha256"],
        "reference_predictor_inventory": inventory,
        "aggregation_rounds": 1,
        "behavior_predictor_ids": settings["behavior_predictor_ids"],
        "behavior_scales": settings["behavior_scales"],
        "matched_data_budget_control": True,
        "extra_sample_counts_per_arm": extra_samples,
        "combined_fit_sample_counts_per_arm": combined_fit_samples,
        "diagnostic_panel_sample_counts": diagnostic_panel_samples,
        "split_evidence": split_evidence,
        "evaluation_evidence": evaluation_evidence,
        "initialization_seeds": settings["initialization_seeds"],
        "hidden_size": settings["hidden_size"],
        "parameter_count": parameter_count,
        "deployment_scales": settings["deployment_scales"],
        "primary_scale": settings["primary_scale"],
        "planned_records": {
            "collection_records": sum(
                len(settings[f"{split}_conditions"])
                * len(settings["profile_ids"])
                * int(settings[f"{split}_episodes_per_condition"])
                for split in ("training", "validation", "diagnostic_test")
            ),
            "scenario_records": scenario_count,
            "episode_records": episode_count,
            "temporal_records": scenario_count * len(settings["time_bins"]),
            "temporal_episode_records": episode_count
            * len(settings["time_bins"]),
            "control_episode_records": control_count,
        },
        "cuda_required": True,
        "supervised_training_allowed": True,
        "rl_training_allowed": False,
        "s4d3_accessed": False,
        "real_slm_actions": False,
        "formal_command": (
            ".\\.venv\\Scripts\\python.exe "
            "scripts\\train_s4_student_state_aggregation.py "
            "--config configs\\experiments\\s4_student_state_aggregation_v1.yaml"
        ),
    }


def _generate_student_aggregation_split(
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
    behavior_predictors: list[FrozenPredictor],
    device: torch.device,
    progress_path: Path,
) -> StudentAggregationSplit:
    condition_list = [RobustnessCondition.from_mapping(item) for item in conditions]
    steps = int(settings["steps_per_episode"])
    preview = int(settings["preview_horizon_frames"])
    total = len(profiles) * len(condition_list) * steps
    bar = counted_progress(
        total=total,
        description=f"生成{split_name}学生状态回合",
        unit="步",
    )
    all_states: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    collection_records: list[dict[str, Any]] = []
    completed = 0
    limit = float(experiment["action_budget"]["residual_component_limit_rad"])
    scales = list(map(float, settings["behavior_scales"]))
    predictor_ids = [item.identifier for item in behavior_predictors]
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
            assignments = [
                (
                    episode % len(behavior_predictors),
                    (episode // len(behavior_predictors)) % len(scales),
                )
                for episode in range(episodes_per_condition)
            ]
            scenario_states: list[torch.Tensor] = []
            scenario_targets: list[torch.Tensor] = []
            scenario_requested: list[torch.Tensor] = []
            scenario_science = _empty_step_metrics()
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
                for predictor_index, predictor in enumerate(behavior_predictors):
                    for scale_index, scale in enumerate(scales):
                        indices = [
                            index
                            for index, assignment in enumerate(assignments)
                            if assignment == (predictor_index, scale_index)
                        ]
                        if not indices:
                            continue
                        index_tensor = torch.tensor(indices, device=device)
                        raw = predictor.predict(state[index_tensor]).clamp(-1, 1)
                        normalized[index_tensor, ANCHOR_MODES:] = raw * scale
                action = controller.compose_action(normalized)
                scenario_requested.append(
                    controller.requested_modal[:, ANCHOR_MODES:].detach().cpu()
                )
                observation, _, _, _, info = environment.step(action.final_delta_rad)
                _append_step_metrics(scenario_science, info)
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
            stacked_states = torch.stack(scenario_states, dim=1)
            stacked_targets = torch.stack(scenario_targets, dim=1)
            stacked_requested = torch.stack(scenario_requested, dim=1)
            all_states.append(stacked_states.flatten(0, 1))
            all_targets.append(stacked_targets.flatten(0, 1))
            science = _mean_step_metrics(scenario_science)
            for episode, (predictor_index, scale_index) in enumerate(assignments):
                requested = stacked_requested[episode]
                collection_records.append(
                    {
                        "split": split_name,
                        "profile": profile.identifier,
                        "physical_condition": condition.identifier,
                        "episode_index": episode,
                        "episode_seed": condition.base_seed + episode,
                        "behavior_predictor_id": predictor_ids[predictor_index],
                        "behavior_scale": scales[scale_index],
                        "requested_added_rms_mean_rad": float(
                            requested.square().mean(dim=-1).sqrt().mean()
                        ),
                        "requested_added_rms_max_rad": float(
                            requested.square().mean(dim=-1).sqrt().max()
                        ),
                        "target_abs_mean": float(
                            stacked_targets[episode].abs().mean()
                        ),
                        "violation_fraction": float(
                            science["violation_fraction"][episode]
                        ),
                    }
                )
            _append_jsonl(
                progress_path,
                {
                    "phase": "student_state_collection",
                    "split": split_name,
                    "profile": profile.identifier,
                    "condition": condition.identifier,
                    "completed_steps": completed,
                    "total_steps": total,
                    "samples": int(stacked_states.shape[0] * steps),
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
        raise RuntimeError("student aggregation samples do not form complete episodes")
    if not bool(torch.isfinite(dataset.states).all()) or not bool(
        torch.isfinite(dataset.targets).all()
    ):
        raise RuntimeError("student aggregation dataset contains non-finite values")
    return StudentAggregationSplit(dataset, collection_records)


def _train_arm(
    *,
    arm_id: str,
    train: TeacherDataset,
    validation: TeacherDataset,
    diagnostic_test: TeacherDataset,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    device: torch.device,
    output_directory: Path,
    progress_path: Path,
) -> TrainedArm:
    arm_directory = output_directory / arm_id
    checkpoint_directory = arm_directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True)
    state_mean = train.states.mean(dim=0)
    state_scale = train.states.std(dim=0, unbiased=False).clamp_min(
        float(settings["normalization_min_scale"])
    )
    ridge_weight = _fit_ridge(
        train,
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
            "arm_id": arm_id,
        },
        ridge_path,
    )
    ridge_id = f"{arm_id}_ridge"
    predictors = [
        FrozenPredictor(
            identifier=ridge_id,
            predict=_ridge_predictor(ridge_weight, state_mean, state_scale, device),
            state_mean=state_mean,
            state_scale=state_scale,
        )
    ]
    records: list[dict[str, Any]] = []
    for seed in settings["initialization_seeds"]:
        model, record = _train_one_policy(
            seed=int(seed),
            train=train,
            validation=validation,
            test=diagnostic_test,
            state_mean=state_mean,
            state_scale=state_scale,
            settings=settings,
            device=device,
            checkpoint_directory=checkpoint_directory,
            output_directory=arm_directory,
            progress_path=progress_path,
        )
        record["arm_id"] = arm_id
        records.append(record)
        predictors.append(
            FrozenPredictor(
                identifier=f"{arm_id}_mlp_seed_{int(seed)}",
                predict=_model_predictor(model, state_mean, state_scale, device),
                state_mean=state_mean,
                state_scale=state_scale,
            )
        )
    ridge_record = {
        "checkpoint": _relative(ridge_path),
        "checkpoint_sha256": _file_sha256(ridge_path),
        "offline_student_state_test": _offline_metrics(
            predictors[0].predict,
            diagnostic_test,
            device=device,
        ),
    }
    return TrainedArm(predictors, ridge_record, records, state_mean, state_scale)


def _interpretation(
    *,
    grouped: dict[str, dict[str, Any]],
    offline_student: dict[str, dict[str, float]],
    coverage: dict[str, dict[str, Any]],
    teacher_summary: dict[str, Any],
    experiment: dict[str, Any],
    settings: dict[str, Any],
    quick: bool,
) -> dict[str, Any]:
    primary = _scale_key(float(settings["primary_scale"]))
    mse_ratio_limit = float(
        experiment["aggregation_gate"]["max_aggregate_to_control_mse_ratio"]
    )
    gain_threshold = float(
        experiment["aggregation_gate"][
            "min_primary_gain_improvement_over_control"
        ]
    )
    coverage_threshold = float(
        experiment["aggregation_gate"]["min_requested_added_std_rad"]
    )
    comparisons = []
    for seed in settings["initialization_seeds"]:
        control_id = f"matched_control_mlp_seed_{int(seed)}"
        aggregate_id = f"student_aggregate_mlp_seed_{int(seed)}"
        control_primary = grouped[control_id][primary]
        aggregate_primary = grouped[aggregate_id][primary]
        comparisons.append(
            {
                "initialization_seed": int(seed),
                "control_student_state_test_mse": offline_student[control_id][
                    "mse"
                ],
                "aggregate_student_state_test_mse": offline_student[aggregate_id][
                    "mse"
                ],
                "aggregate_to_control_mse_ratio": float(
                    offline_student[aggregate_id]["mse"]
                )
                / max(float(offline_student[control_id]["mse"]), 1e-12),
                "aggregate_offline_gate": _offline_gate(
                    offline_student[aggregate_id], experiment["offline_gate"]
                ),
                "control_primary_relative_power_gain": control_primary[
                    "overall"
                ]["relative_power_gain"],
                "aggregate_primary_relative_power_gain": aggregate_primary[
                    "overall"
                ]["relative_power_gain"],
                "primary_gain_improvement_over_control": float(
                    aggregate_primary["overall"]["relative_power_gain"]
                )
                - float(control_primary["overall"]["relative_power_gain"]),
                "aggregate_primary_closed_loop_gate": aggregate_primary[
                    "closed_loop_gate"
                ],
            }
        )
    coverage_value = float(
        coverage["student_aggregate_combined_training"][
            "requested_added_std_min_rad"
        ]
    )
    coverage_pass = coverage_value >= coverage_threshold
    offline_pass = all(
        item["aggregate_offline_gate"]
        and float(item["aggregate_to_control_mse_ratio"]) <= mse_ratio_limit
        for item in comparisons
    )
    improvement_pass = all(
        float(item["primary_gain_improvement_over_control"]) >= gain_threshold
        for item in comparisons
    )
    primary_pass = all(
        item["aggregate_primary_closed_loop_gate"] == "PASS"
        for item in comparisons
    )
    safe_scales = [
        float(scale)
        for scale in settings["deployment_scales"]
        if all(
            grouped[f"student_aggregate_mlp_seed_{int(seed)}"][
                _scale_key(float(scale))
            ]["closed_loop_gate"]
            == "PASS"
            for seed in settings["initialization_seeds"]
        )
    ]
    if quick:
        status = "QUICK_SMOKE_ONLY"
    elif teacher_summary["closed_loop_gate"] != "PASS":
        status = "TEACHER_POSITIVE_CONTROL_FAILED"
    elif coverage_pass and offline_pass and improvement_pass and primary_pass:
        status = "SINGLE_ROUND_STUDENT_AGGREGATION_PASS"
    elif coverage_pass and offline_pass:
        status = "SINGLE_ROUND_AGGREGATION_OFFLINE_ONLY_STOP"
    else:
        status = "SINGLE_ROUND_STUDENT_AGGREGATION_FAILED_STOP"
    return {
        "status": status,
        "teacher_positive_control_pass": teacher_summary["closed_loop_gate"]
        == "PASS",
        "requested_added_coverage_pass": coverage_pass,
        "student_state_offline_improvement_pass": offline_pass,
        "primary_gain_improvement_pass": improvement_pass,
        "primary_closed_loop_pass": primary_pass,
        "primary_scale": float(settings["primary_scale"]),
        "safe_scales_all_aggregate_mlp": safe_scales,
        "initialization_comparisons": comparisons,
        "aggregation_rounds_used": 1,
        "additional_aggregation_round_authorized": False,
        "rl_training_authorized": False,
        "s4d3_authorized": False,
        "real_hardware_authorized": False,
        "next_rule": (
            "Audit this only aggregation round before any algorithm decision. "
            "A second aggregation round is not pre-authorized."
        ),
    }


def _state_coverage(dataset: TeacherDataset) -> dict[str, Any]:
    requested_added = dataset.states[:, 168 + ANCHOR_MODES : 189]
    structural_zero = dataset.states[:, 189 + ANCHOR_MODES : 210]
    requested_std = requested_added.std(dim=0, unbiased=False)
    structural_std = structural_zero.std(dim=0, unbiased=False)
    return {
        "samples": len(dataset.states),
        "requested_added_std_min_rad": float(requested_std.min()),
        "requested_added_std_max_rad": float(requested_std.max()),
        "requested_added_nonzero_fraction": float(
            (requested_added.abs() > 1e-8).float().mean()
        ),
        "requested_added_abs_max_rad": float(requested_added.abs().max()),
        "structural_added_baseline_std_max_rad": float(structural_std.max()),
        "structural_added_baseline_abs_max_rad": float(
            structural_zero.abs().max()
        ),
    }


def _dataset_from_payload(payload: dict[str, Any]) -> TeacherDataset:
    return TeacherDataset(
        states=payload["states"],
        targets=payload["targets"],
        episodes=int(payload["episodes"]),
        steps_per_episode=int(payload["steps_per_episode"]),
    )


def _concatenate_datasets(left: TeacherDataset, right: TeacherDataset) -> TeacherDataset:
    if left.steps_per_episode != right.steps_per_episode:
        raise RuntimeError("cannot concatenate datasets with different episode lengths")
    return TeacherDataset(
        states=torch.cat((left.states, right.states)),
        targets=torch.cat((left.targets, right.targets)),
        episodes=left.episodes + right.episodes,
        steps_per_episode=left.steps_per_episode,
    )


def _subset_complete_episodes(
    dataset: TeacherDataset, *, episodes: int, steps: int
) -> TeacherDataset:
    if episodes <= 0 or episodes > dataset.episodes:
        raise ValueError("quick upstream episode count is invalid")
    if steps <= 0 or steps > dataset.steps_per_episode:
        raise ValueError("quick upstream step count is invalid")
    states = dataset.states.reshape(
        dataset.episodes, dataset.steps_per_episode, -1
    )[:episodes, :steps]
    targets = dataset.targets.reshape(
        dataset.episodes, dataset.steps_per_episode, -1
    )[:episodes, :steps]
    return TeacherDataset(
        states=states.flatten(0, 1).clone(),
        targets=targets.flatten(0, 1).clone(),
        episodes=episodes,
        steps_per_episode=steps,
    )


def _verify_upstream_baseline(upstream: dict[str, Any]) -> dict[str, Any]:
    exact_fields = (
        "summary",
        "preflight",
        "effective_config",
        "source_manifest",
        "dataset",
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
            raise RuntimeError(f"baseline-state upstream hash mismatch: {field}")
        paths[field] = path
    for field in ("experiment_config", "audit_record"):
        path = _project_path(upstream[field])
        if not _matches_exact_or_lf_canonical(
            path, str(upstream[f"{field}_sha256"])
        ):
            raise RuntimeError(f"baseline-state text evidence mismatch: {field}")
        paths[field] = path
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    if (
        bool(summary["experiment"]["quick"])
        or summary["experiment"]["status"] != "completed_pending_audit"
        or summary["interpretation"]["status"]
        != "OFFLINE_CORRECTED_BUT_CLOSED_LOOP_FAILED"
        or not bool(summary["interpretation"]["baseline_state_offline_correction_pass"])
        or bool(summary["interpretation"]["primary_scale_closed_loop_pass"])
        or summary["zero_scale_alignment"]["status"] != "PASS"
    ):
        raise RuntimeError("baseline-state formal result is not eligible")
    audit = paths["audit_record"].read_text(encoding="utf-8")
    if (
        "Verification Status: `ANALYZED`" not in audit
        or "ONE_BOUNDED_STUDENT_STATE_AGGREGATION_ROUND: AUTHORIZED" not in audit
    ):
        raise RuntimeError("baseline-state audit does not authorize one round")
    manifest = json.loads(paths["source_manifest"].read_text(encoding="utf-8"))
    for relative, digest in manifest.items():
        path = _project_path(relative)
        if not path.is_file() or not _matches_exact_or_lf_canonical(
            path, str(digest)
        ):
            raise RuntimeError(f"baseline-state source changed: {relative}")
    data = torch.load(paths["dataset"], map_location="cpu", weights_only=False)
    samples = {
        "training": len(data["train"]["states"]),
        "validation": len(data["validation"]["states"]),
        "diagnostic_test": len(data["diagnostic_test"]["states"]),
    }
    if samples != {
        "training": 115200,
        "validation": 57600,
        "diagnostic_test": 57600,
    }:
        raise RuntimeError("baseline-state upstream dataset shape changed")
    return {
        "summary_sha256": _file_sha256(paths["summary"]),
        "audit_sha256": _file_sha256(paths["audit_record"]),
        "dataset_samples": samples,
    }


def _validate_scales(experiment: dict[str, Any], settings: dict[str, Any]) -> None:
    behavior = list(map(float, settings["behavior_scales"]))
    maximum = float(experiment["aggregation"]["max_behavior_scale"])
    if (
        behavior != sorted(set(behavior))
        or any(value <= 0 or value > maximum for value in behavior)
        or maximum > 0.02
    ):
        raise RuntimeError("behavior scales exceed the bounded collection contract")
    deployment = list(map(float, settings["deployment_scales"]))
    if deployment != sorted(set(deployment)) or any(
        value <= 0 or value > 1 for value in deployment
    ):
        raise RuntimeError("deployment scales must be sorted and in (0, 1]")
    if float(settings["primary_scale"]) not in deployment:
        raise RuntimeError("primary scale must be predeclared in deployment scales")
    cursor = 0
    for item in settings["time_bins"]:
        if int(item["start"]) != cursor or int(item["stop"]) <= cursor:
            raise RuntimeError("time bins must be contiguous and increasing")
        cursor = int(item["stop"])
    if cursor != int(settings["steps_per_episode"]):
        raise RuntimeError("time bins must cover the complete episode")


def _validate_evaluation_seeds(
    experiment: dict[str, Any], settings: dict[str, Any]
) -> dict[str, Any]:
    count = int(settings["evaluation_episodes_per_condition"])
    evaluation: set[int] = set()
    for item in settings["evaluation_conditions"]:
        start = int(item["base_seed"])
        current = set(range(start, start + count))
        if evaluation & current:
            raise RuntimeError("evaluation episode seeds overlap")
        evaluation.update(current)
    development: set[int] = set()
    for split in ("training", "validation", "diagnostic_test"):
        split_count = int(settings[f"{split}_episodes_per_condition"])
        for item in settings[f"{split}_conditions"]:
            start = int(item["base_seed"])
            development.update(range(start, start + split_count))
    if evaluation & development:
        raise RuntimeError("evaluation seeds overlap aggregation data")
    protected = [
        (int(item["start_inclusive"]), int(item["end_exclusive"]))
        for item in experiment["protected_seed_ranges"]
    ]
    if any(
        any(begin <= seed < stop for begin, stop in protected)
        for seed in evaluation
    ):
        raise RuntimeError("evaluation seeds overlap a protected range")
    return {
        "episode_seed_count": len(evaluation),
        "minimum_seed": min(evaluation),
        "maximum_seed": max(evaluation),
    }


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
        "evaluation_episodes_per_condition": int(
            evaluation["episodes_per_condition"]
        ),
        "profile_ids": list(dataset["profile_ids"]),
        "training_conditions": list(dataset["training_conditions"]),
        "validation_conditions": list(dataset["validation_conditions"]),
        "diagnostic_test_conditions": list(dataset["diagnostic_test_conditions"]),
        "evaluation_conditions": list(evaluation["conditions"]),
        "preview_horizon_frames": int(experiment["teacher"]["preview_horizon_frames"]),
        "behavior_scales": list(map(float, experiment["aggregation"]["behavior_scales"])),
        "behavior_predictor_ids": list(
            map(str, experiment["aggregation"]["behavior_predictor_ids"])
        ),
        "reference_predictor_ids": [
            str(item["id"]) for item in experiment["reference_predictors"]
        ],
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
        "upstream_quick_episodes_per_split": 0,
    }
    if quick:
        q = experiment["quick"]
        settings.update(
            {
                "output_directory": experiment["outputs"]["quick_directory"],
                "steps_per_episode": int(q["steps_per_episode"]),
                "training_episodes_per_condition": int(
                    q["training_episodes_per_condition"]
                ),
                "validation_episodes_per_condition": int(
                    q["validation_episodes_per_condition"]
                ),
                "diagnostic_test_episodes_per_condition": int(
                    q["diagnostic_test_episodes_per_condition"]
                ),
                "evaluation_episodes_per_condition": int(
                    q["evaluation_episodes_per_condition"]
                ),
                "profile_ids": list(q["profile_ids"]),
                "training_conditions": list(q["training_conditions"]),
                "validation_conditions": list(q["validation_conditions"]),
                "diagnostic_test_conditions": list(q["diagnostic_test_conditions"]),
                "evaluation_conditions": list(q["evaluation_conditions"]),
                "behavior_scales": list(map(float, q["behavior_scales"])),
                "behavior_predictor_ids": list(map(str, q["behavior_predictor_ids"])),
                "reference_predictor_ids": list(map(str, q["reference_predictor_ids"])),
                "initialization_seeds": list(map(int, q["initialization_seeds"])),
                "hidden_size": int(q["hidden_size"]),
                "batch_size": int(q["batch_size"]),
                "max_optimizer_steps": int(q["max_optimizer_steps"]),
                "min_optimizer_steps": int(q["min_optimizer_steps"]),
                "validation_interval_steps": int(q["validation_interval_steps"]),
                "early_stopping_patience_checks": int(
                    q["early_stopping_patience_checks"]
                ),
                "deployment_scales": list(map(float, q["deployment_scales"])),
                "primary_scale": float(q["primary_scale"]),
                "time_bins": list(q["time_bins"]),
                "upstream_quick_episodes_per_split": int(
                    q["upstream_episodes_per_split"]
                ),
            }
        )
    return settings


def _selected_predictors(
    predictors: list[dict[str, Any]], identifiers: list[str]
) -> list[dict[str, Any]]:
    by_id = {str(item["id"]): item for item in predictors}
    if any(identifier not in by_id for identifier in identifiers):
        raise RuntimeError("unknown reference predictor identifier")
    return [by_id[identifier] for identifier in identifiers]
