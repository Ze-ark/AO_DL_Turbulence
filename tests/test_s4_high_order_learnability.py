from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from src.rl.s4_high_order_learnability import (
    HighOrderImitationPolicy,
    TeacherDataset,
    _effective_settings,
    _offline_metrics,
    _paired_science_summary,
    _validate_seed_splits,
    preflight_s4_high_order_learnability,
)
from src.rl.s4_training import _load_yaml, _project_path


CONFIG = Path("configs/experiments/s4_high_order_learnability_v1.yaml")


def test_imitation_policy_outputs_only_bounded_added_modes() -> None:
    model = HighOrderImitationPolicy(210, 32, 11)
    output = model(torch.randn(5, 210))
    assert output.shape == (5, 11)
    assert torch.all(output <= 1)
    assert torch.all(output >= -1)


def test_offline_metrics_detect_perfect_observable_predictor() -> None:
    states = torch.randn(20, 4)
    targets = states[:, :2].tanh()
    data = TeacherDataset(states=states, targets=targets, episodes=4, steps_per_episode=5)
    metrics = _offline_metrics(lambda value: value[:, :2].tanh(), data, device=torch.device("cpu"))
    assert metrics["mse"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["skill_score_against_zero"] == pytest.approx(1.0)
    assert metrics["mean_cosine_similarity"] == pytest.approx(1.0, abs=1e-6)


def test_paired_science_gate_uses_independent_metrics() -> None:
    baseline = {
        "power_in_bucket": torch.full((16,), 0.5),
        "measured_power_in_bucket": torch.full((16,), 0.5),
        "strehl": torch.full((16,), 0.4),
        "phase_rmse": torch.full((16,), 0.8),
        "violation_fraction": torch.full((16,), 0.01),
    }
    candidate = {
        "power_in_bucket": torch.full((16,), 0.55),
        "measured_power_in_bucket": torch.full((16,), 0.55),
        "strehl": torch.full((16,), 0.45),
        "phase_rmse": torch.full((16,), 0.7),
        "violation_fraction": torch.full((16,), 0.02),
    }
    gate = {
        "proposed_min_relative_power_gain": 0.02,
        "min_power_delta_ci95_low": 0.0,
        "min_strehl_delta_ci95_low": 0.0,
        "max_phase_rmse_delta_ci95_high": 0.0,
        "max_violation_fraction": 0.05,
    }
    result = _paired_science_summary("student", candidate, baseline, gate)
    assert result["gate"] == "PASS"
    assert result["relative_power_gain"] == pytest.approx(0.1)


def test_formal_preflight_locks_observation_and_complete_episode_splits(
    tmp_path: Path,
) -> None:
    experiment_path = _project_path(CONFIG)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=False)
    settings["output_directory"] = str(tmp_path / "formal")
    result = preflight_s4_high_order_learnability(
        experiment_path,
        experiment,
        settings,
        quick=False,
    )
    assert result["status"] == "READY_FOR_USER_SUPERVISED_TRAINING"
    assert result["state_size"] == 210
    assert result["output_size"] == 11
    assert result["sample_counts"] == {
        "training": 115200,
        "validation": 57600,
        "diagnostic_test": 57600,
    }
    assert not result["future_truth_in_model_inputs"]
    assert not result["hardware_profile_id_in_model_inputs"]
    assert not result["rl_training_allowed"]


def test_seed_validation_rejects_cross_split_episode_reuse() -> None:
    experiment = _load_yaml(_project_path(CONFIG))
    settings = _effective_settings(experiment, quick=False)
    broken = deepcopy(settings)
    broken["validation_conditions"][0]["base_seed"] = broken[
        "training_conditions"
    ][0]["base_seed"]
    with pytest.raises(RuntimeError, match="overlap across"):
        _validate_seed_splits(experiment, broken)


def test_preflight_rejects_oracle_information_in_model_input() -> None:
    experiment_path = _project_path(CONFIG)
    experiment = _load_yaml(experiment_path)
    broken = deepcopy(experiment)
    broken["policy_observation"]["include_future_truth"] = True
    settings = _effective_settings(broken, quick=False)
    with pytest.raises(RuntimeError, match="deployable SAC observation"):
        preflight_s4_high_order_learnability(
            experiment_path,
            broken,
            settings,
            quick=False,
        )
