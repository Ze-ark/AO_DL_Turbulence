from __future__ import annotations

from pathlib import Path

from src.rl.s4_r3_multistep_target_diagnostic import (
    interpret_multistep_targets,
    run_s4_r3_multistep_target_diagnostic,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "s4_r3_multistep_target_v1.yaml"


def _ci(mean: float, low: float, high: float) -> dict[str, float]:
    return {"mean": mean, "median": mean, "ci95_low": low, "ci95_high": high}


def _group(
    seed: int,
    *,
    last_target: tuple[float, float, float] = (0.05, 0.02, 0.08),
    flip_horizon: int | None = None,
) -> dict[str, object]:
    return {
        "arm": "student_backbone",
        "policy_seed": seed,
        "q_advantage_vs_zero": _ci(0.3, 0.2, 0.4),
        "horizons": [
            {
                "horizon": 1,
                "deterministic_target_advantage_vs_zero": _ci(0.04, 0.02, 0.06),
                "empirical_reward_advantage_vs_zero": _ci(-0.01, -0.02, -0.005),
            },
            {
                "horizon": 32,
                "deterministic_target_advantage_vs_zero": _ci(*last_target),
                "empirical_reward_advantage_vs_zero": _ci(-0.02, -0.03, -0.01),
            },
        ],
        "point_estimate_target_sign_flip_horizon": flip_horizon,
        "confirmed_negative_target_horizon": flip_horizon,
    }


def test_formal_preflight_locks_multistep_scope() -> None:
    result = run_s4_r3_multistep_target_diagnostic(CONFIG, preflight_only=True)

    assert result["status"] == "READY_FOR_USER_DIAGNOSTIC"
    assert result["upstream_interpretation"] == "SOFT_BELLMAN_MISFIT_CONFIRMED"
    assert result["checkpoint_count"] == 6
    assert result["candidate_ids"] == ["zero", "actor"]
    assert result["horizons"] == [1, 2, 4, 8, 16, 32]
    assert result["expected_probe_states"] == 324
    assert result["expected_branch_rollouts"] == 648
    assert result["expected_episode_records"] == 5_184
    assert result["expected_horizon_records"] == 31_104
    assert result["training_transitions"] == 0
    assert result["checkpoint_writes"] == 0
    assert result["sealed_s4d3_access"] is False


def test_quick_preflight_uses_small_independent_scope() -> None:
    result = run_s4_r3_multistep_target_diagnostic(
        CONFIG, quick=True, preflight_only=True
    )

    assert result["status"] == "READY_FOR_QUICK_SMOKE"
    assert result["checkpoint_count"] == 2
    assert result["horizons"] == [1, 2, 4]
    assert result["expected_probe_states"] == 4
    assert result["expected_branch_rollouts"] == 8
    assert result["expected_episode_records"] == 16
    assert result["expected_horizon_records"] == 48


def test_interpretation_confirms_persistent_bootstrap_optimism() -> None:
    result = interpret_multistep_targets(
        [_group(seed) for seed in (9301, 9302, 9303)], quick=False
    )

    assert result["status"] == "PERSISTENT_BOOTSTRAP_OPTIMISM_CONFIRMED"
    assert result["all_main_seeds_one_step_bootstrap_optimism"] is True
    assert result["all_main_seeds_persistent_through_max_horizon"] is True
    assert result["retraining_authorized"] is False


def test_interpretation_reports_uniform_finite_horizon_correction() -> None:
    groups = [
        _group(seed, last_target=(-0.03, -0.05, -0.01), flip_horizon=32)
        for seed in (9301, 9302, 9303)
    ]
    result = interpret_multistep_targets(groups, quick=False)

    assert result["status"] == "BOOTSTRAP_OPTIMISM_WITH_FINITE_HORIZON_CORRECTION"
    assert result["all_main_seeds_confirmed_finite_horizon_correction"] is True
    assert result["algorithm_change_authorized"] is False


def test_interpretation_stays_mixed_when_one_seed_lacks_initial_optimism() -> None:
    groups = [_group(9301), _group(9302), _group(9303)]
    groups[-1]["horizons"][0]["deterministic_target_advantage_vs_zero"] = _ci(
        -0.01, -0.02, -0.005
    )
    result = interpret_multistep_targets(groups, quick=False)

    assert result["status"] == "MULTISTEP_MECHANISM_MIXED_OR_UNCONFIRMED"
    assert result["all_main_seeds_one_step_bootstrap_optimism"] is False
