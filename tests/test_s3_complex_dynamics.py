"""S3-B复杂动态基线与初始化种子门槛测试。"""

import pytest
import torch

from src.simulation.complex_dynamics import (
    fit_regime_conditioned_ridge,
    initialization_seed_gate,
    student_t_summary,
)
from src.simulation.temporal_dynamics import DynamicsCondition, TemporalDynamicsData


def _data(
    histories: torch.Tensor,
    targets: torch.Tensor,
    condition_index: torch.Tensor,
) -> TemporalDynamicsData:
    samples = len(histories)
    return TemporalDynamicsData(
        histories=histories,
        targets=targets,
        episode_seed=torch.arange(samples),
        condition_index=condition_index,
        target_step=torch.full((samples,), 2),
    )


def test_regime_conditioned_ridge_recovers_two_opposite_linear_rules():
    generator = torch.Generator().manual_seed(14)
    histories = torch.randn(160, 2, 1, generator=generator)
    condition_index = torch.cat(
        (torch.zeros(80, dtype=torch.int64), torch.ones(80, dtype=torch.int64))
    )
    targets = histories[:, -1].clone()
    targets[condition_index == 1] *= -1
    train = _data(histories, targets, condition_index)
    validation = _data(histories.clone(), targets.clone(), condition_index.clone())
    conditions = [
        DynamicsCondition("a", 0.3, 0.0, 1, regime="frozen"),
        DynamicsCondition("b", 0.3, 0.0, 2, regime="boiling"),
    ]

    result = fit_regime_conditioned_ridge(
        train,
        validation,
        conditions,
        conditions,
        ridge_alpha=1e-8,
    )

    assert set(result["models"]) == {"boiling", "frozen"}
    assert torch.allclose(result["prediction"], targets, atol=1e-5)


def test_initialization_seed_gate_uses_seed_level_student_t_interval():
    run_records = [
        {"run_id": "a", "skill_score": 0.10},
        {"run_id": "b", "skill_score": 0.11},
        {"run_id": "c", "skill_score": 0.12},
    ]
    condition_records = [
        {"run_id": run, "condition_id": condition, "regime": regime, "skill_score": value}
        for run, value in (("a", 0.10), ("b", 0.11), ("c", 0.12))
        for condition, regime in (("v1", "frozen"), ("v2", "boiling"))
    ]

    gate = initialization_seed_gate(
        run_records,
        condition_records,
        min_mean_skill_score=0.05,
        min_ci95_low=0.0,
        require_every_run_positive=True,
        require_every_condition_positive=True,
    )

    assert gate["validation_gate"] == "PASS"
    assert gate["run_skill_score"]["count"] == 3
    assert gate["run_skill_score"]["ci95_low"] > 0


def test_student_t_summary_rejects_one_value():
    with pytest.raises(ValueError, match="2 to 31"):
        student_t_summary([0.1])
