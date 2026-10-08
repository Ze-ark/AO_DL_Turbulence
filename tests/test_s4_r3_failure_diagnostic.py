from __future__ import annotations

from pathlib import Path

import torch

from src.rl.s4_r3_failure_diagnostic import (
    _load_policy,
    interpret_r3_failure_diagnostic,
    run_s4_r3_failure_diagnostic,
)
from src.rl.s4_training import _load_yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "s4_r3_failure_diagnostic_v1.yaml"


def _distribution(mean: float, low: float, high: float) -> dict[str, float]:
    return {"mean": mean, "median": mean, "ci95_low": low, "ci95_high": high}


def _group(
    *,
    reverse: tuple[float, float, float],
    quarter: tuple[float, float, float],
    half: tuple[float, float, float],
    full: tuple[float, float, float],
    q_advantage: float,
    cosine: float = 0.5,
) -> dict[str, object]:
    variants = []
    for identifier, scale, values in (
        ("reverse_full", -1.0, reverse),
        ("student_only", 0.0, (0.0, 0.0, 0.0)),
        ("forward_quarter", 0.25, quarter),
        ("forward_half", 0.5, half),
        ("forward_full", 1.0, full),
    ):
        variants.append(
            {
                "id": identifier,
                "scale": scale,
                "paired_power_delta": _distribution(*values),
                "paired_reward_delta": _distribution(values[0], values[1], values[2]),
            }
        )
    return {
        "arm": "student_backbone",
        "policy_seed": 9301,
        "student_id": "aggregate_mlp_seed_8201",
        "student_state_alignment": {
            "actor_ideal_cosine": _distribution(cosine, cosine, cosine),
            "actor_ideal_same_direction_fraction": _distribution(0.7, 0.7, 0.7),
            "q_actor_minus_zero": _distribution(q_advantage, q_advantage, q_advantage),
        },
        "variants": variants,
    }


def _thresholds() -> dict[str, float]:
    return {
        "positive_power_ci_low": 0.0,
        "negative_power_ci_high": 0.0,
        "low_actor_teacher_cosine": 0.1,
        "low_same_direction_fraction": 0.5,
        "critic_preference_margin": 0.0,
        "reward_conflict_margin": 0.0,
    }


def test_preflight_locks_six_read_only_checkpoints_and_new_seed_namespace() -> None:
    result = run_s4_r3_failure_diagnostic(CONFIG, preflight_only=True)

    assert result["status"] == "READY_FOR_USER_DIAGNOSTIC"
    assert result["upstream_all_six_runs_failed"] is True
    assert result["checkpoint_count"] == 6
    assert result["variant_scales"] == [-1.0, -0.5, 0.0, 0.25, 0.5, 1.0]
    assert result["expected_scenarios"] == 648
    assert result["expected_episode_records"] == 10_368
    assert result["checkpoint_writes"] == 0
    assert result["training_transitions"] == 0
    assert result["sealed_s4d3_access"] is False


def test_best_r3_actor_and_critics_load_frozen_on_cpu() -> None:
    experiment = _load_yaml(CONFIG)
    policy = _load_policy(experiment["checkpoints"][0], device=torch.device("cpu"))
    state = torch.zeros(2, 210)
    action = policy.actor.deterministic(state)
    q_value = torch.minimum(policy.q1(state, action), policy.q2(state, action))

    assert action.shape == (2, 11)
    assert q_value.shape == (2, 1)
    assert torch.isfinite(action).all()
    assert torch.isfinite(q_value).all()
    assert all(not parameter.requires_grad for parameter in policy.actor.parameters())
    assert all(not parameter.requires_grad for parameter in policy.q1.parameters())


def test_interpretation_detects_wrong_direction_before_critic_label() -> None:
    grouped = [
        _group(
            reverse=(0.01, 0.005, 0.015),
            quarter=(-0.002, -0.003, -0.001),
            half=(-0.004, -0.005, -0.003),
            full=(-0.008, -0.009, -0.007),
            q_advantage=1.0,
        )
    ]
    result = interpret_r3_failure_diagnostic(
        grouped,
        thresholds=_thresholds(),
        zero_alignment={"pass": True},
        quick=False,
    )

    assert result["status"] == "ACTION_DIRECTION_MISMATCH_SUPPORTED"
    assert result["checkpoint_results"][0]["reverse_scale_positive"] is True
    assert result["retraining_authorized"] is False


def test_interpretation_detects_excessive_amplitude_when_half_scale_is_positive() -> None:
    grouped = [
        _group(
            reverse=(-0.003, -0.004, -0.002),
            quarter=(0.003, 0.001, 0.005),
            half=(0.004, 0.002, 0.006),
            full=(-0.008, -0.009, -0.007),
            q_advantage=-0.5,
        )
    ]
    result = interpret_r3_failure_diagnostic(
        grouped,
        thresholds=_thresholds(),
        zero_alignment={"pass": True},
        quick=False,
    )

    assert result["status"] == "EXCESSIVE_ACTION_AMPLITUDE_SUPPORTED"
    assert result["checkpoint_results"][0]["positive_reduced_scales"] == [0.25, 0.5]


def test_interpretation_detects_critic_ranking_failure() -> None:
    grouped = [
        _group(
            reverse=(-0.003, -0.004, -0.002),
            quarter=(-0.002, -0.003, -0.001),
            half=(-0.004, -0.005, -0.003),
            full=(-0.008, -0.009, -0.007),
            q_advantage=0.5,
        )
    ]
    result = interpret_r3_failure_diagnostic(
        grouped,
        thresholds=_thresholds(),
        zero_alignment={"pass": True},
        quick=False,
    )

    assert result["status"] == "CRITIC_RANKING_FAILURE_SUPPORTED"
    assert result["algorithm_change_authorized"] is False
