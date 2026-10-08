from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.rl.s4_r3_multistep_critic_repair import (
    CriticTargetSpec,
    preflight_s4_r3_multistep_critic_repair_design,
    select_critic_target,
    truncated_lambda_weights,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "s4_r3_multistep_critic_repair_design_v1.yaml"
)


def test_truncated_lambda_weights_sum_to_one_and_keep_tail_mass() -> None:
    weights = truncated_lambda_weights(32, 0.95)

    assert weights.dtype == torch.float64
    assert weights.shape == (32,)
    assert torch.all(weights >= 0)
    assert torch.isclose(weights.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert weights[-1] == pytest.approx(0.95**31)


def test_lambda_endpoints_reduce_to_one_step_or_longest_target() -> None:
    targets = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    one = CriticTargetSpec("lambda_zero", "truncated_td_lambda", 3, 0.0)
    longest = CriticTargetSpec("lambda_one", "truncated_td_lambda", 3, 1.0)

    assert select_critic_target(targets, one).item() == pytest.approx(1.0)
    assert select_critic_target(targets, longest).item() == pytest.approx(3.0)


def test_target_selection_keeps_batch_shape_and_expected_mixture() -> None:
    targets = torch.tensor(
        [[1.0, 2.0, 4.0], [2.0, 4.0, 8.0]], dtype=torch.float64
    )
    n_step = CriticTargetSpec("n3", "n_step", 3)
    mixed = CriticTargetSpec("lambda_half", "truncated_td_lambda", 3, 0.5)

    assert torch.equal(select_critic_target(targets, n_step), targets[:, 2])
    expected = targets @ torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64)
    assert torch.allclose(select_critic_target(targets, mixed), expected)


def test_target_selection_rejects_short_or_non_finite_curves() -> None:
    spec = CriticTargetSpec("n16", "n_step", 16)
    with pytest.raises(ValueError, match="does not cover"):
        select_critic_target(torch.zeros(2, 15), spec)
    with pytest.raises(ValueError, match="non-finite"):
        select_critic_target(
            torch.tensor([[float("nan")]]), CriticTargetSpec("n1", "n_step", 1)
        )


def test_r3_d2a_design_preflight_locks_upstream_targets_and_safety() -> None:
    result = preflight_s4_r3_multistep_critic_repair_design(CONFIG)

    assert result["status"] == "READY_FOR_TRAINER_IMPLEMENTATION"
    assert result["upstream_interpretation"] == (
        "BOOTSTRAP_OPTIMISM_WITH_FINITE_HORIZON_CORRECTION"
    )
    assert result["target_ids"] == [
        "one_step_control",
        "n_step_16",
        "n_step_32",
        "td_lambda_095",
    ]
    assert result["planned_critic_fits"] == 12
    assert result["actor_updates"] == 0
    assert result["alpha_updates"] == 0
    assert result["formal_training_authorized"] is False
    assert result["sealed_s4d3_access"] is False
    assert result["real_slm_actions"] is False

