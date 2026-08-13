"""S3收敛监视与128/256维复查判断测试。"""

from pathlib import Path

import pytest
import yaml

from src.simulation.convergence import (
    ConvergenceTracker,
    classify_larger_model,
    classify_more_data,
    convergence_label,
)


def test_tracker_waits_for_minimum_steps_and_patience():
    tracker = ConvergenceTracker(
        min_steps=360,
        patience_checks=3,
        relative_min_delta=0.01,
    )

    assert not tracker.update(120, 1.0)
    assert not tracker.update(240, 0.995)
    assert not tracker.update(360, 0.994)
    assert tracker.update(480, 0.993)


def test_tracker_resets_after_meaningful_improvement():
    tracker = ConvergenceTracker(
        min_steps=360,
        patience_checks=2,
        relative_min_delta=0.01,
    )

    assert not tracker.update(120, 1.0)
    assert not tracker.update(240, 0.98)
    assert not tracker.update(360, 0.979)
    assert tracker.update(480, 0.978)


def test_convergence_label_distinguishes_plateau_and_improvement():
    plateau = convergence_label(
        [1.0, 0.999, 0.998, 0.997, 0.996],
        early_stopped=False,
        final_window_checks=5,
        max_final_window_improvement=0.01,
    )
    improving = convergence_label(
        [1.0, 0.98, 0.96, 0.94, 0.92],
        early_stopped=False,
        final_window_checks=5,
        max_final_window_improvement=0.01,
    )

    assert plateau["label"] == "NEAR_PLATEAU_AT_MAX_STEPS"
    assert improving["label"] == "STILL_IMPROVING_AT_MAX_STEPS"


def test_capacity_and_data_classifiers_use_both_settings():
    summaries = {
        (192, 128): {"gru_rmse_mean": 0.0100},
        (192, 256): {"gru_rmse_mean": 0.0090},
        (384, 128): {"gru_rmse_mean": 0.0085},
        (384, 256): {"gru_rmse_mean": 0.0075},
    }

    capacity = classify_larger_model(
        summaries,
        episode_scales=[192, 384],
        reference_hidden_size=128,
        candidate_hidden_size=256,
        min_improvement=0.05,
    )
    data = classify_more_data(
        summaries,
        smaller_episode_scale=192,
        larger_episode_scale=384,
        hidden_sizes=[128, 256],
        min_improvement=0.10,
    )

    assert capacity["verdict"] == "LARGER_MODEL_SUPPORTED"
    assert capacity["rmse_improvement_by_episode_scale"]["192"] == pytest.approx(0.1)
    assert data["verdict"] == "MORE_DATA_SUPPORTED"


def test_capacity_classifier_reports_mixed_direction():
    summaries = {
        (192, 128): {"gru_rmse_mean": 0.010},
        (192, 256): {"gru_rmse_mean": 0.009},
        (384, 128): {"gru_rmse_mean": 0.008},
        (384, 256): {"gru_rmse_mean": 0.0082},
    }

    result = classify_larger_model(
        summaries,
        episode_scales=[192, 384],
        reference_hidden_size=128,
        candidate_hidden_size=256,
        min_improvement=0.05,
    )

    assert result["verdict"] == "LARGER_MODEL_MIXED"


def test_followup_config_declares_12_runs_and_convergence_budget():
    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (project_root / "configs/experiments/s3_convergence_capacity_v1.yaml").read_text(
            encoding="utf-8"
        )
    )

    experiment = config["experiment"]
    run_count = (
        len(experiment["episode_scales"])
        * len(experiment["hidden_sizes"])
        * len(experiment["initialization_seeds"])
    )
    assert experiment["episode_scales"] == [192, 384]
    assert experiment["hidden_sizes"] == [128, 256]
    assert run_count == 12
    assert config["training"]["min_optimizer_steps"] == 3600
    assert config["training"]["max_optimizer_steps"] > 3600
    assert config["dataset"]["metric_scale_source"] == "full_train_pool_history"
    assert config["metadata"]["allow_scientific_claims"] is False
