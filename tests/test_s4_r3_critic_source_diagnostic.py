from __future__ import annotations

from pathlib import Path

import torch

from src.rl.residual_sac import QNetwork, SacConfig, SquashedGaussianActor
from src.rl.s4_r3_critic_source_diagnostic import (
    _clustered_distribution,
    _history_support_metrics,
    _soft_target_mc,
    interpret_critic_sources,
    run_s4_r3_critic_source_diagnostic,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "s4_r3_critic_source_v1.yaml"


def _network_config() -> SacConfig:
    return SacConfig(state_size=4, action_size=2, hidden_size=8)


def _ci(mean: float, low: float, high: float) -> dict[str, float]:
    return {"mean": mean, "median": mean, "ci95_low": low, "ci95_high": high}


def _group(
    seed: int,
    *,
    bellman_misfit: bool = True,
    support_proxy: bool = True,
) -> dict[str, object]:
    return {
        "arm": "student_backbone",
        "policy_seed": seed,
        "ranking": {
            "actor_q_advantage_vs_zero": _ci(0.3, 0.2, 0.4),
            "actor_soft_target_advantage_vs_zero": (
                _ci(-0.2, -0.3, -0.1)
                if bellman_misfit
                else _ci(0.2, 0.1, 0.3)
            ),
            "actor_excess_bellman_residual_vs_zero": (
                _ci(0.5, 0.4, 0.6)
                if bellman_misfit
                else _ci(0.0, -0.1, 0.1)
            ),
            "pairwise_q_soft_target_accuracy": _ci(
                0.3 if bellman_misfit else 0.8,
                0.3 if bellman_misfit else 0.8,
                0.3 if bellman_misfit else 0.8,
            ),
            "top_q_soft_target_agreement": _ci(
                0.2 if bellman_misfit else 0.8,
                0.2 if bellman_misfit else 0.8,
                0.2 if bellman_misfit else 0.8,
            ),
        },
        "support_proxy": {
            "actor_minus_zero_nearest_rms": (
                _ci(0.2, 0.1, 0.3)
                if support_proxy
                else _ci(0.0, -0.1, 0.1)
            ),
            "actor_minus_zero_max_log_probability": (
                _ci(-1.0, -1.2, -0.8)
                if support_proxy
                else _ci(0.0, -0.1, 0.1)
            ),
        },
    }


def _thresholds() -> dict[str, float]:
    return {
        "positive_ci_low": 0.0,
        "negative_ci_high": 0.0,
        "minimum_consistent_rank_accuracy": 0.5,
        "minimum_consistent_top_agreement": 0.5,
    }


def test_formal_preflight_locks_read_only_scope() -> None:
    result = run_s4_r3_critic_source_diagnostic(CONFIG, preflight_only=True)

    assert result["status"] == "READY_FOR_USER_DIAGNOSTIC"
    assert result["upstream_interpretation"] == (
        "CRITIC_DEPLOYMENT_RANKING_FAILURE_CONFIRMED"
    )
    assert result["checkpoint_count"] == 6
    assert result["history_checkpoint_count"] == 30
    assert result["target_samples"] == 64
    assert result["expected_probe_states"] == 324
    assert result["expected_branch_rollouts"] == 1_620
    assert result["expected_episode_records"] == 12_960
    assert result["training_transitions"] == 0
    assert result["checkpoint_writes"] == 0
    assert result["replay_buffer_reconstructed"] is False
    assert result["sealed_s4d3_access"] is False


def test_soft_target_sampling_is_repeatable_and_antithetic() -> None:
    torch.manual_seed(3)
    config = _network_config()
    actor = SquashedGaussianActor(config)
    q1 = QNetwork(config)
    q2 = QNetwork(config)
    next_state = torch.randn(3, 4)
    reward = torch.tensor([0.1, 0.2, 0.3])
    done = torch.zeros(3, dtype=torch.bool)

    first = _soft_target_mc(
        next_state=next_state,
        reward=reward,
        done=done,
        actor=actor,
        target_q1=q1,
        target_q2=q2,
        alpha=0.05,
        gamma=0.99,
        samples=8,
        seed=77,
        device=torch.device("cpu"),
    )
    second = _soft_target_mc(
        next_state=next_state,
        reward=reward,
        done=done,
        actor=actor,
        target_q1=q1,
        target_q2=q2,
        alpha=0.05,
        gamma=0.99,
        samples=8,
        seed=77,
        device=torch.device("cpu"),
    )

    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert first[0].shape == (3,)
    assert bool(torch.isfinite(first[1]).all())


def test_history_support_proxy_prefers_matching_snapshot_action() -> None:
    torch.manual_seed(5)
    actor = SquashedGaussianActor(_network_config())
    state = torch.randn(4, 4)
    matching = actor.deterministic(state)
    result = _history_support_metrics(state, matching, [actor])

    assert torch.allclose(
        result["history_nearest_action_rms"], torch.zeros(4), atol=1e-7
    )
    assert bool(torch.isfinite(result["history_max_log_probability"]).all())


def test_clustered_distribution_uses_trajectory_means() -> None:
    values = torch.tensor([1.0, 3.0, 10.0, 14.0])
    result = _clustered_distribution(values, ["a", "a", "b", "b"])

    assert result["clusters"] == 2
    assert result["raw_observations"] == 4
    assert result["mean"] == 7.0


def test_interpretation_requires_all_three_main_seeds() -> None:
    grouped = [_group(seed) for seed in (9301, 9302, 9303)]
    result = interpret_critic_sources(
        grouped, thresholds=_thresholds(), quick=False
    )

    assert result["status"] == (
        "SOFT_BELLMAN_MISFIT_WITH_SUPPORT_PROXY_ASSOCIATION"
    )
    assert result["all_main_seeds_soft_bellman_misfit"] is True
    assert result["all_main_seeds_support_proxy_association"] is True
    assert result["retraining_authorized"] is False


def test_interpretation_stays_mixed_when_one_seed_disagrees() -> None:
    grouped = [_group(9301), _group(9302), _group(9303, bellman_misfit=False)]
    result = interpret_critic_sources(
        grouped, thresholds=_thresholds(), quick=False
    )

    assert result["status"] == "DEPLOYMENT_MISMATCH_WITH_SUPPORT_PROXY_ASSOCIATION"
    assert result["all_main_seeds_soft_bellman_misfit"] is False
    assert result["algorithm_change_authorized"] is False
