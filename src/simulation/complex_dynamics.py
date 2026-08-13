"""S3-B复杂动态实验的分类型线性基线与初始化种子门槛。"""

from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Any, Sequence

import torch

from src.simulation.temporal_dynamics import (
    DynamicsCondition,
    TemporalDynamicsData,
    fit_ridge_autoregression,
    modal_normalization,
    ridge_autoregressive_forecast,
)


def fit_regime_conditioned_ridge(
    train_data: TemporalDynamicsData,
    validation_data: TemporalDynamicsData,
    train_conditions: Sequence[DynamicsCondition],
    validation_conditions: Sequence[DynamicsCondition],
    ridge_alpha: float,
) -> dict[str, Any]:
    """按已知动态类型分别拟合岭回归，作为比单一岭回归更强的基线。"""
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha must be non-negative")
    _validate_condition_indices(train_data, train_conditions, "train")
    _validate_condition_indices(validation_data, validation_conditions, "validation")
    normalization_mean, normalization_scale = modal_normalization(train_data)
    prediction = torch.empty_like(validation_data.targets)
    models: dict[str, dict[str, torch.Tensor]] = {}
    validation_regimes = {condition.regime for condition in validation_conditions}
    for regime in sorted(validation_regimes):
        train_indices = [
            index for index, condition in enumerate(train_conditions) if condition.regime == regime
        ]
        validation_indices = [
            index
            for index, condition in enumerate(validation_conditions)
            if condition.regime == regime
        ]
        if not train_indices:
            raise ValueError(f"validation regime {regime!r} has no matching training regime")
        train_mask = _condition_mask(train_data.condition_index, train_indices)
        validation_mask = _condition_mask(
            validation_data.condition_index, validation_indices
        )
        weight, bias = fit_ridge_autoregression(
            train_data.histories[train_mask],
            train_data.targets[train_mask],
            normalization_mean,
            normalization_scale,
            ridge_alpha,
        )
        prediction[validation_mask] = ridge_autoregressive_forecast(
            validation_data.histories[validation_mask],
            normalization_mean,
            normalization_scale,
            weight,
            bias,
        )
        models[regime] = {"weight": weight, "bias": bias}
    return {
        "normalization_mean": normalization_mean,
        "normalization_scale": normalization_scale,
        "prediction": prediction,
        "models": models,
    }


def initialization_seed_gate(
    run_records: Sequence[dict[str, Any]],
    condition_records: Sequence[dict[str, Any]],
    *,
    min_mean_skill_score: float,
    min_ci95_low: float,
    require_every_run_positive: bool,
    require_every_condition_positive: bool,
) -> dict[str, Any]:
    """以模型初始化种子为独立复现单位，而不是把相邻时间窗当独立样本。"""
    if len(run_records) < 2:
        raise ValueError("at least two initialization runs are required")
    run_skills = [float(record["skill_score"]) for record in run_records]
    run_summary = student_t_summary(run_skills)
    by_condition: dict[str, list[float]] = {}
    by_regime: dict[str, list[float]] = {}
    for record in condition_records:
        by_condition.setdefault(str(record["condition_id"]), []).append(
            float(record["skill_score"])
        )
        by_regime.setdefault(str(record["regime"]), []).append(
            float(record["skill_score"])
        )
    condition_means = {key: mean(values) for key, values in sorted(by_condition.items())}
    regime_means = {key: mean(values) for key, values in sorted(by_regime.items())}
    every_run_positive = all(value > 0 for value in run_skills)
    every_condition_positive = all(value > 0 for value in condition_means.values())
    passed = (
        run_summary["mean"] >= min_mean_skill_score
        and run_summary["ci95_low"] > min_ci95_low
        and (every_run_positive or not require_every_run_positive)
        and (every_condition_positive or not require_every_condition_positive)
    )
    return {
        "validation_gate": "PASS" if passed else "FAIL",
        "independent_unit": "model_initialization_seed",
        "run_skill_score": run_summary,
        "min_mean_skill_score": min_mean_skill_score,
        "min_ci95_low": min_ci95_low,
        "require_every_run_positive": require_every_run_positive,
        "require_every_condition_positive": require_every_condition_positive,
        "every_run_positive": every_run_positive,
        "every_condition_positive": every_condition_positive,
        "condition_mean_skill_scores": condition_means,
        "regime_mean_skill_scores": regime_means,
    }


def student_t_summary(values: Sequence[float]) -> dict[str, float]:
    """返回小样本均值、样本标准差和双侧95% Student-t区间。"""
    materialized = [float(value) for value in values]
    count = len(materialized)
    if count < 2 or count > 31:
        raise ValueError("student_t_summary supports 2 to 31 values")
    average = mean(materialized)
    sample_std = stdev(materialized)
    critical = _TWO_SIDED_T_95[count - 1]
    half_width = critical * sample_std / math.sqrt(count)
    return {
        "count": float(count),
        "mean": average,
        "sample_std": sample_std,
        "ci95_low": average - half_width,
        "ci95_high": average + half_width,
        "minimum": min(materialized),
        "maximum": max(materialized),
    }


def _validate_condition_indices(
    data: TemporalDynamicsData,
    conditions: Sequence[DynamicsCondition],
    split_name: str,
) -> None:
    if not conditions:
        raise ValueError(f"{split_name} conditions must not be empty")
    observed = set(int(value) for value in torch.unique(data.condition_index).tolist())
    expected = set(range(len(conditions)))
    if observed != expected:
        raise ValueError(f"{split_name} condition indices do not match condition metadata")


def _condition_mask(condition_index: torch.Tensor, selected: Sequence[int]) -> torch.Tensor:
    mask = torch.zeros_like(condition_index, dtype=torch.bool)
    for index in selected:
        mask |= condition_index == index
    return mask


_TWO_SIDED_T_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}
