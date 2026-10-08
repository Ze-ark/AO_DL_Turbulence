from __future__ import annotations

from pathlib import Path

import torch

from src.rl.s4_r3_critic_calibration import (
    _candidate_action,
    _pairwise_rank_accuracy,
    _resolve_reward_accumulator_dtype,
    interpret_critic_calibration,
    preflight_s4_r3_critic_calibration,
    run_s4_r3_critic_calibration,
)
from src.rl.s4_training import _load_yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "s4_r3_critic_calibration_v1.yaml"
CONFIG_V2 = ROOT / "configs" / "experiments" / "s4_r3_critic_calibration_v2.yaml"


def _distribution(mean: float, low: float, high: float) -> dict[str, float]:
    return {"mean": mean, "median": mean, "ci95_low": low, "ci95_high": high}


def _group(
    *,
    arm: str = "student_backbone",
    seed: int = 9301,
    q_advantage: tuple[float, float, float] = (0.3, 0.2, 0.4),
    return_advantage: tuple[float, float, float] = (-0.2, -0.3, -0.1),
    rank_accuracy: float = 0.3,
    top_agreement: float = 0.2,
    quarter_power: tuple[float, float, float] = (0.1, 0.05, 0.15),
    quarter_reward: tuple[float, float, float] = (-0.1, -0.15, -0.05),
    decomposition_error: float = 1e-7,
) -> dict[str, object]:
    return {
        "arm": arm,
        "policy_seed": seed,
        "ranking": {
            "actor_q_advantage_vs_zero": _distribution(*q_advantage),
            "actor_reward_return_advantage_vs_zero": _distribution(
                *return_advantage
            ),
            "pairwise_rank_accuracy": _distribution(
                rank_accuracy, rank_accuracy, rank_accuracy
            ),
            "top_action_agreement": _distribution(
                top_agreement, top_agreement, top_agreement
            ),
        },
        "quarter_actor_vs_zero": {
            "true_power_return_advantage": _distribution(*quarter_power),
            "reward_return_advantage": _distribution(*quarter_reward),
        },
        "reward_decomposition_max_abs_error": decomposition_error,
    }


def _thresholds() -> dict[str, float]:
    return {
        "positive_ci_low": 0.0,
        "negative_ci_high": 0.0,
        "maximum_rank_accuracy_for_failure": 0.5,
        "maximum_top_action_agreement_for_failure": 0.5,
        "decomposition_tolerance": 5e-6,
    }


def test_preflight_locks_read_only_calibration_scope() -> None:
    result = run_s4_r3_critic_calibration(CONFIG, preflight_only=True)

    assert result["status"] == "READY_FOR_USER_DIAGNOSTIC"
    assert result["upstream_interpretation"] == "MIXED_FAILURE_MECHANISMS"
    assert result["zero_alignment_pass"] is True
    assert result["checkpoint_count"] == 6
    assert result["expected_probe_states"] == 324
    assert result["expected_branch_rollouts"] == 1_620
    assert result["expected_episode_records"] == 12_960
    assert result["checkpoint_writes"] == 0
    assert result["training_transitions"] == 0
    assert result["sealed_s4d3_access"] is False


def test_quick_preflight_uses_separate_small_scope() -> None:
    experiment = _load_yaml(CONFIG)
    from src.rl.s4_r3_critic_calibration import _effective_settings

    settings = _effective_settings(experiment, quick=True)
    result, _ = preflight_s4_r3_critic_calibration(
        CONFIG,
        experiment,
        settings,
        quick=True,
    )

    assert result["status"] == "READY_FOR_QUICK_SMOKE"
    assert result["checkpoint_count"] == 2
    assert result["expected_probe_states"] == 4
    assert result["expected_branch_rollouts"] == 20
    assert result["expected_episode_records"] == 40


def test_v2_preflight_keeps_gate_and_uses_float64_accumulation() -> None:
    experiment = _load_yaml(CONFIG_V2)
    from src.rl.s4_r3_critic_calibration import _effective_settings

    settings = _effective_settings(experiment, quick=False)
    result, _ = preflight_s4_r3_critic_calibration(
        CONFIG_V2,
        experiment,
        settings,
        quick=False,
    )

    assert result["status"] == "READY_FOR_USER_DIAGNOSTIC"
    assert result["reward_accumulator_dtype"] == "float64"
    assert settings["reward_accumulator_dtype"] == "float64"
    assert experiment["interpretation_thresholds"]["decomposition_tolerance"] == 5e-6
    assert experiment["outputs"]["directory"].endswith("_v2")


def test_reward_accumulator_dtype_is_explicit_and_rejects_unknown_values() -> None:
    assert _resolve_reward_accumulator_dtype("float32") is torch.float32
    assert _resolve_reward_accumulator_dtype("float64") is torch.float64

    try:
        _resolve_reward_accumulator_dtype("automatic")
    except ValueError as error:
        assert "reward accumulator dtype" in str(error)
    else:
        raise AssertionError("unknown reward accumulator dtype must fail closed")


def test_candidate_actions_preserve_locked_definitions() -> None:
    actor = torch.tensor([[0.8, -0.4]])
    ideal = torch.tensor([[-0.2, 0.6]])

    assert torch.equal(
        _candidate_action(
            {"kind": "actor_scale", "scale": 0.25}, actor=actor, ideal=ideal
        ),
        torch.tensor([[0.2, -0.1]]),
    )
    assert torch.equal(
        _candidate_action({"kind": "ideal_teacher"}, actor=actor, ideal=ideal),
        ideal,
    )


def test_pairwise_rank_accuracy_detects_reversed_order() -> None:
    predicted = torch.tensor([[1.0, 2.0, 3.0], [1.0, 3.0, 2.0]])
    observed = torch.tensor([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]])
    result = _pairwise_rank_accuracy(predicted, observed)

    assert torch.allclose(result, torch.tensor([0.0, 1.0]))


def test_interpretation_requires_all_main_seeds_for_confirmation() -> None:
    grouped = [_group(seed=seed) for seed in (9301, 9302, 9303)]
    result = interpret_critic_calibration(
        grouped,
        thresholds=_thresholds(),
        quick=False,
    )

    assert result["status"] == "CRITIC_FAILURE_AND_REWARD_SUPPRESSION_CONFIRMED"
    assert result["all_main_seeds_critic_failure"] is True
    assert result["all_main_seeds_reward_suppression"] is True
    assert result["retraining_authorized"] is False
    assert result["algorithm_change_authorized"] is False


def test_interpretation_stays_mixed_when_one_main_seed_disagrees() -> None:
    grouped = [_group(seed=9301), _group(seed=9302)]
    grouped.append(
        _group(
            seed=9303,
            q_advantage=(-0.1, -0.2, -0.05),
            return_advantage=(-0.2, -0.3, -0.1),
            rank_accuracy=0.8,
            top_agreement=0.8,
            quarter_power=(-0.1, -0.2, -0.05),
            quarter_reward=(-0.1, -0.2, -0.05),
        )
    )
    result = interpret_critic_calibration(
        grouped,
        thresholds=_thresholds(),
        quick=False,
    )

    assert result["status"] == "MIXED_OR_UNCONFIRMED"
    assert result["all_main_seeds_critic_failure"] is False
    assert result["reward_change_authorized"] is False
