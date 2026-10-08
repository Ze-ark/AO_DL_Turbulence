from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.rl.s4_r3_physical_target_confirmation import (
    _effective_settings,
    confirmation_gate_checks,
    independent_confirmation_decision,
    preflight_s4_r3_physical_target_confirmation,
)
from src.rl.s4_training import _load_yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "s4_r3_physical_target_confirmation_v1.yaml"
)


def _stable_record(*, actor_better_fraction: float) -> dict[str, float]:
    return {
        "reward_power_sign_agreement": 0.90,
        "reward_power_balanced_accuracy": 0.80,
        "reward_terminal_sign_agreement": 0.90,
        "reward_terminal_balanced_accuracy": 0.80,
        "power_terminal_sign_agreement": 0.90,
        "power_terminal_balanced_accuracy": 0.80,
        "reward_local_min_sign_agreement": 0.90,
        "reward_local_min_balanced_accuracy": 0.80,
        "power_local_min_sign_agreement": 0.90,
        "power_local_min_balanced_accuracy": 0.80,
        "reward_actor_better_fraction": actor_better_fraction,
    }


def _thresholds() -> dict[str, object]:
    return {
        "minimum_sign_agreement": 0.80,
        "minimum_balanced_accuracy": 0.65,
        "minimum_actor_better_fraction": 0.15,
        "maximum_actor_better_fraction": 0.85,
        "class_balance_gate_strata": ["all"],
    }


def test_class_balance_applies_only_to_all_samples() -> None:
    record = _stable_record(actor_better_fraction=0.06)

    all_checks = confirmation_gate_checks(
        record,
        thresholds=_thresholds(),
        stratum="all",
    )
    high_checks = confirmation_gate_checks(
        record,
        thresholds=_thresholds(),
        stratum="high_magnitude",
    )

    assert all_checks["class_balance"] is False
    assert "class_balance" not in high_checks
    assert all(high_checks.values())


def test_formal_decision_requires_all_six_rows() -> None:
    records = [
        {"policy_seed": seed, "stratum": stratum, "gate": "PASS"}
        for seed in (9301, 9302, 9303)
        for stratum in ("all", "high_magnitude")
    ]

    passed = independent_confirmation_decision(
        records,
        policy_seeds=[9301, 9302, 9303],
        strata=["all", "high_magnitude"],
        quick=False,
    )
    records[-1]["gate"] = "FAIL"
    failed = independent_confirmation_decision(
        records,
        policy_seeds=[9301, 9302, 9303],
        strata=["all", "high_magnitude"],
        quick=False,
    )

    assert passed["status"] == "PHYSICAL_H16_TARGET_INDEPENDENTLY_CONFIRMED"
    assert passed["supervised_probe_design_authorized"] is True
    assert passed["full_rl_authorized"] is False
    assert failed["status"] == "PHYSICAL_H16_TARGET_NOT_CONFIRMED"
    assert failed["supervised_probe_design_authorized"] is False


def test_quick_mode_cannot_authorize_scientific_decision() -> None:
    result = independent_confirmation_decision(
        [{"policy_seed": 9301, "stratum": "all", "gate": "PASS"}],
        policy_seeds=[9301],
        strata=["all", "high_magnitude"],
        quick=True,
    )

    assert result["status"] == "QUICK_SMOKE_ONLY"
    assert result["all_rows_pass"] is False
    assert result["supervised_probe_design_authorized"] is False


def test_effective_settings_freeze_h16_and_new_seed_namespace() -> None:
    experiment = _load_yaml(CONFIG)
    settings = _effective_settings(experiment, quick=False)

    assert settings["fixed_horizon"] == 16
    assert settings["terminal_horizon"] == 32
    assert settings["class_balance_gate_strata"] == ["all"]
    assert [
        item["base_seed"] for item in settings["confirmation_split"]["conditions"]
    ] == [3830000, 3840000, 3850000]
    assert settings["magnitude_thresholds"][9301] == 0.00852528514343565


def test_formal_preflight_locks_independent_confirmation(tmp_path: Path) -> None:
    experiment = _load_yaml(CONFIG)
    settings = _effective_settings(experiment, quick=False)
    settings["output_directory"] = str(tmp_path / "a5_r1_preflight")
    with patch("torch.cuda.is_available", return_value=True):
        result, _, checkpoints = preflight_s4_r3_physical_target_confirmation(
            CONFIG,
            experiment,
            settings,
            quick=False,
        )

    assert result["status"] == "READY_FOR_USER_CONFIRMATION"
    assert result["expected_branch_rollouts"] == 324
    assert result["expected_samples"] == 5184
    assert result["expected_action_pairs"] == 2592
    assert result["expected_metric_rows"] == 6
    assert result["seed_namespace"]["upstream_overlap"] == 0
    assert result["gradient_updates"] == 0
    assert result["horizon_reselection"] is False
    assert len(checkpoints) == 3

