from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from src.rl.s4_high_order_learnability import TeacherDataset
from src.rl.s4_student_state_aggregation import (
    _concatenate_datasets,
    _effective_settings,
    _interpretation,
    _state_coverage,
    _subset_complete_episodes,
    _validate_evaluation_seeds,
    _validate_scales,
    preflight_s4_student_state_aggregation,
)
from src.rl.s4_training import _load_yaml, _project_path


CONFIG = Path("configs/experiments/s4_student_state_aggregation_v1.yaml")


def _dataset(value: float, *, episodes: int = 2, steps: int = 3) -> TeacherDataset:
    return TeacherDataset(
        states=torch.full((episodes * steps, 210), value),
        targets=torch.full((episodes * steps, 11), value),
        episodes=episodes,
        steps_per_episode=steps,
    )


def test_formal_preflight_locks_one_round_and_matched_budget(tmp_path: Path) -> None:
    experiment_path = _project_path(CONFIG)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=False)
    settings["output_directory"] = str(tmp_path / "formal")
    result = preflight_s4_student_state_aggregation(
        experiment_path,
        experiment,
        settings,
        quick=False,
    )
    assert result["status"] == "READY_FOR_USER_SINGLE_ROUND_STUDENT_STATE_TRAINING"
    assert result["aggregation_rounds"] == 1
    assert result["matched_data_budget_control"]
    assert result["extra_sample_counts_per_arm"] == {
        "training": 115200,
        "validation": 57600,
        "diagnostic_test": 57600,
    }
    assert result["combined_fit_sample_counts_per_arm"] == {
        "training": 230400,
        "validation": 115200,
    }
    assert result["diagnostic_panel_sample_counts"] == {
        "student_state": 57600,
        "matched_extra_baseline_state": 57600,
        "original_baseline_state": 57600,
    }
    assert result["planned_records"] == {
        "collection_records": 1152,
        "scenario_records": 720,
        "episode_records": 11520,
        "temporal_records": 3600,
        "temporal_episode_records": 57600,
        "control_episode_records": 576,
    }
    assert not result["rl_training_allowed"]
    assert not result["real_slm_actions"]


def test_preflight_rejects_a_second_aggregation_round(tmp_path: Path) -> None:
    experiment_path = _project_path(CONFIG)
    experiment = _load_yaml(experiment_path)
    broken = deepcopy(experiment)
    broken["aggregation"]["rounds"] = 2
    settings = _effective_settings(broken, quick=False)
    settings["output_directory"] = str(tmp_path / "formal")
    with pytest.raises(RuntimeError, match="exactly one"):
        preflight_s4_student_state_aggregation(
            experiment_path,
            broken,
            settings,
            quick=False,
        )


def test_collection_scale_cannot_exceed_two_percent() -> None:
    experiment = _load_yaml(_project_path(CONFIG))
    settings = _effective_settings(experiment, quick=False)
    broken = deepcopy(settings)
    broken["behavior_scales"] = [0.01, 0.03]
    with pytest.raises(RuntimeError, match="bounded collection"):
        _validate_scales(experiment, broken)


def test_evaluation_seeds_must_not_overlap_aggregation_data() -> None:
    experiment = _load_yaml(_project_path(CONFIG))
    settings = _effective_settings(experiment, quick=False)
    broken = deepcopy(settings)
    broken["evaluation_conditions"][0]["base_seed"] = broken[
        "training_conditions"
    ][0]["base_seed"]
    with pytest.raises(RuntimeError, match="overlap aggregation"):
        _validate_evaluation_seeds(experiment, broken)


def test_state_coverage_distinguishes_deployable_request_from_structural_zero() -> None:
    data = _dataset(0.0)
    data.states[:, 178:189] = torch.linspace(0.0, 0.05, len(data.states)).unsqueeze(1)
    coverage = _state_coverage(data)
    assert coverage["requested_added_std_min_rad"] > 0
    assert coverage["requested_added_nonzero_fraction"] > 0
    assert coverage["structural_added_baseline_std_max_rad"] == 0
    assert coverage["structural_added_baseline_abs_max_rad"] == 0


def test_concatenate_datasets_preserves_complete_episode_accounting() -> None:
    result = _concatenate_datasets(_dataset(0.0), _dataset(1.0))
    assert result.episodes == 4
    assert len(result.states) == result.episodes * result.steps_per_episode
    assert result.states.shape == (12, 210)


def test_quick_subset_keeps_complete_shortened_episodes() -> None:
    source = _dataset(1.0, episodes=4, steps=5)
    result = _subset_complete_episodes(source, episodes=2, steps=3)
    assert result.episodes == 2
    assert result.steps_per_episode == 3
    assert result.states.shape == (6, 210)


def test_interpretation_requires_coverage_offline_and_closed_loop_pass() -> None:
    experiment = _load_yaml(_project_path(CONFIG))
    settings = _effective_settings(experiment, quick=True)
    offline = {
        "matched_control_mlp_seed_8201": {
            "mse": 0.4,
            "skill_score_against_zero": 0.5,
            "mean_cosine_similarity": 0.5,
        },
        "student_aggregate_mlp_seed_8201": {
            "mse": 0.2,
            "skill_score_against_zero": 0.7,
            "mean_cosine_similarity": 0.7,
        },
    }
    grouped = {
        "matched_control_mlp_seed_8201": {
            "0.02": {
                "overall": {"relative_power_gain": -0.01},
                "closed_loop_gate": "FAIL",
            },
            "0.10": {
                "overall": {"relative_power_gain": -0.10},
                "closed_loop_gate": "FAIL",
            }
        },
        "student_aggregate_mlp_seed_8201": {
            "0.02": {
                "overall": {"relative_power_gain": 0.01},
                "closed_loop_gate": "FAIL",
            },
            "0.10": {
                "overall": {"relative_power_gain": 0.04},
                "closed_loop_gate": "PASS",
            }
        },
    }
    coverage = {
        "student_aggregate_combined_training": {
            "requested_added_std_min_rad": 0.001
        }
    }
    result = _interpretation(
        grouped=grouped,
        offline_student=offline,
        coverage=coverage,
        teacher_summary={"closed_loop_gate": "PASS"},
        experiment=experiment,
        settings=settings,
        quick=False,
    )
    assert result["status"] == "SINGLE_ROUND_STUDENT_AGGREGATION_PASS"
    assert result["requested_added_coverage_pass"]
    assert result["student_state_offline_improvement_pass"]
    assert result["primary_closed_loop_pass"]
    assert not result["additional_aggregation_round_authorized"]
