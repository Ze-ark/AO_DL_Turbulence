"""S3学习曲线完整回合抽样与机械诊断测试。"""

from collections import Counter
from pathlib import Path

import pytest
import torch
import yaml

from src.simulation.learning_curve import (
    balanced_episode_subset,
    classify_capacity,
    classify_learning_curve,
    summarize_replicates,
)
from src.simulation.temporal_dynamics import TemporalDynamicsData


def _pool_data() -> TemporalDynamicsData:
    seeds = torch.tensor(
        [10, 10, 11, 11, 20, 20, 21, 21, 30, 30, 31, 31, 40, 40, 41, 41]
    )
    conditions = torch.tensor([0] * 4 + [1] * 4 + [2] * 4 + [3] * 4)
    samples = len(seeds)
    return TemporalDynamicsData(
        histories=torch.arange(samples * 2, dtype=torch.float32).reshape(samples, 2, 1),
        targets=torch.zeros(samples, 1),
        episode_seed=seeds,
        condition_index=conditions,
        target_step=torch.tensor([2, 3] * 8),
    )


def test_balanced_subset_selects_complete_episodes_per_physical_condition():
    subset = balanced_episode_subset(
        _pool_data(),
        physical_ids_by_condition=["a", "a", "b", "b"],
        total_episodes=4,
    )

    pairs = set(zip(subset.condition_index.tolist(), subset.episode_seed.tolist()))
    assert pairs == {(0, 10), (1, 20), (2, 30), (3, 40)}
    assert len(subset.histories) == 8
    for condition, seed in pairs:
        assert int(((subset.condition_index == condition) & (subset.episode_seed == seed)).sum()) == 2


def test_learning_curve_classification_supports_clear_gap_closure():
    summaries = {
        48: {"gru_rmse_mean": 0.020, "ridge_rmse_mean": 0.010},
        192: {"gru_rmse_mean": 0.015, "ridge_rmse_mean": 0.010},
        384: {"gru_rmse_mean": 0.012, "ridge_rmse_mean": 0.009},
    }

    result = classify_learning_curve(
        summaries,
        supported_min_192_to_384_rmse_reduction=0.10,
        supported_min_gap_ratio_reduction_48_to_384=0.25,
        not_supported_max_192_to_384_rmse_reduction=0.05,
    )

    assert result["verdict"] == "DATA_LIMITED_SUPPORTED"
    assert result["gru_rmse_reduction_192_to_384"] == pytest.approx(0.2)
    assert result["gap_ratio_reduction_48_to_384"] == pytest.approx(1 / 3)


def test_learning_curve_classification_rejects_plateau():
    summaries = {
        48: {"gru_rmse_mean": 0.020, "ridge_rmse_mean": 0.010},
        192: {"gru_rmse_mean": 0.015, "ridge_rmse_mean": 0.010},
        384: {"gru_rmse_mean": 0.0147, "ridge_rmse_mean": 0.0095},
    }

    result = classify_learning_curve(
        summaries,
        supported_min_192_to_384_rmse_reduction=0.10,
        supported_min_gap_ratio_reduction_48_to_384=0.25,
        not_supported_max_192_to_384_rmse_reduction=0.05,
    )

    assert result["verdict"] == "DATA_LIMITED_NOT_SUPPORTED"


def test_capacity_classification_detects_better_smaller_model():
    summaries = {
        32: {"gru_rmse_mean": 0.008},
        64: {"gru_rmse_mean": 0.009},
        128: {"gru_rmse_mean": 0.010},
    }

    result = classify_capacity(
        summaries,
        reference_hidden_size=128,
        min_smaller_model_improvement=0.05,
    )

    assert result["verdict"] == "OVERPARAMETERIZATION_SUPPORTED"
    assert result["best_smaller_hidden_size"] == 32
    assert result["best_smaller_model_improvement"] == pytest.approx(0.2)


def test_replicate_summary_keeps_scale_groups_separate():
    records = [
        {"episode_scale": 48, "gru_rmse": 0.02, "ridge_rmse": 0.01, "skill_score": -3.0},
        {"episode_scale": 48, "gru_rmse": 0.018, "ridge_rmse": 0.01, "skill_score": -2.2},
        {"episode_scale": 96, "gru_rmse": 0.015, "ridge_rmse": 0.009, "skill_score": -1.8},
    ]

    result = summarize_replicates(records, "episode_scale")

    assert result[48]["runs"] == 2
    assert result[48]["gru_rmse_mean"] == pytest.approx(0.019)
    assert result[96]["runs"] == 1


def test_diagnostic_config_has_balanced_fresh_seeds_and_18_runs():
    project_root = Path(__file__).resolve().parents[1]
    diagnostic = yaml.safe_load(
        (project_root / "configs/experiments/s3_data_scaling_diagnostic_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    protected = yaml.safe_load(
        (project_root / diagnostic["protected_original_experiment"]).read_text(
            encoding="utf-8"
        )
    )
    pool = diagnostic["dataset"]["train_pool_conditions"]
    validation = diagnostic["dataset"]["diagnostic_validation_conditions"]
    physical_counts = Counter(item["physical_id"] for item in pool)
    diagnostic_seeds = {item["base_seed"] for item in pool + validation}
    protected_seeds = {
        item["base_seed"]
        for split in (
            "train_conditions",
            "validation_conditions",
            "sealed_test_conditions",
        )
        for item in protected["dataset"][split]
    }

    assert len(pool) == 12
    assert set(physical_counts.values()) == {2}
    assert len(diagnostic_seeds) == len(pool) + len(validation)
    assert diagnostic_seeds.isdisjoint(protected_seeds)
    assert all(scale % len(physical_counts) == 0 for scale in diagnostic["diagnostic"]["episode_scales"])

    primary_runs = (
        len(diagnostic["diagnostic"]["episode_scales"])
        * len(diagnostic["diagnostic"]["initialization_seeds"])
    )
    extra_capacity_runs = (
        len(diagnostic["diagnostic"]["capacity_hidden_sizes"]) - 1
    ) * len(diagnostic["diagnostic"]["initialization_seeds"])
    assert primary_runs + extra_capacity_runs == 18
