from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import torch

from src.rl.s4_r3_physical_target_stability import (
    _effective_settings,
    PhysicalTargetPairDataset,
    horizon_stability_record,
    load_physical_target_pairs,
    preflight_s4_r3_physical_target_stability,
    select_stable_horizon,
)
from src.rl.s4_training import _load_yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "s4_r3_physical_target_stability_v1.yaml"
)


def test_formal_preflight_locks_data_and_safety(tmp_path: Path) -> None:
    experiment = _load_yaml(CONFIG)
    settings = _effective_settings(experiment, quick=False)
    settings["output_directory"] = str(tmp_path / "a5_preflight_contract")
    with patch("torch.cuda.is_available", return_value=True):
        result = preflight_s4_r3_physical_target_stability(
            CONFIG, experiment, settings, quick=False
        )

    assert result["status"] == "READY_FOR_USER_DIAGNOSTIC"
    assert result["planned_metric_rows"] == 300
    assert result["development_pairs"] == 5184
    assert result["validation_pairs"] == 2592
    assert result["gradient_updates"] == 0
    assert result["model_training"] is False
    assert result["new_episode_generation"] is False
    assert result["mechanism_audit_access"] is False


def test_pair_loader_exposes_all_physical_horizons() -> None:
    experiment = _load_yaml(CONFIG)
    dataset = load_physical_target_pairs(
        experiment["data"]["development"][9301],
        maximum_horizon=32,
        state_size=210,
        action_size=11,
    )

    assert dataset.reward_deltas.shape == (1728, 32)
    assert dataset.power_deltas.shape == (1728, 32)
    assert dataset.target_deltas.shape == (1728, 32)
    assert len(dataset.rows) == 1728


def test_stable_physical_curves_pass_declared_gate() -> None:
    signs = torch.tensor([-1.0] * 6 + [1.0] * 6).double()
    horizons = torch.arange(1, 33, dtype=torch.float64)
    rewards = signs[:, None] * horizons[None, :]
    dataset = PhysicalTargetPairDataset(
        reward_deltas=rewards,
        power_deltas=1.1 * rewards,
        target_deltas=0.9 * rewards,
        rows=[{"episode_seed": index} for index in range(12)],
    )
    record = horizon_stability_record(
        dataset,
        policy_seed=1,
        split="development",
        horizon=8,
        stratum="all",
        magnitude_threshold=8.0,
        terminal_horizon=32,
        local_offsets=[1, 2, 4],
        thresholds={
            "minimum_sign_agreement": 0.80,
            "minimum_balanced_accuracy": 0.65,
            "minimum_actor_better_fraction": 0.15,
            "maximum_actor_better_fraction": 0.85,
        },
        device=torch.device("cpu"),
    )

    assert record["gate"] == "PASS"
    assert record["reward_power_sign_agreement"] == 1.0
    assert record["reward_local_min_sign_agreement"] == 1.0
    assert record["target_reward_sign_agreement"] == 1.0


def _selection_rows(*, validation_short_passes: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in ("development", "validation"):
        for horizon in (8, 9):
            for seed in (1, 2):
                for stratum in ("all", "high_magnitude"):
                    passed = horizon == 9
                    if split == "validation" and horizon == 9:
                        passed = validation_short_passes
                    rows.append(
                        {
                            "split": split,
                            "horizon": horizon,
                            "policy_seed": seed,
                            "stratum": stratum,
                            "gate": "PASS" if passed else "FAIL",
                        }
                    )
    return rows


def test_selection_uses_shortest_common_development_candidate() -> None:
    result = select_stable_horizon(
        _selection_rows(validation_short_passes=True),
        settings={
            "policy_seeds": [1, 2],
            "strata": ["all", "high_magnitude"],
            "candidate_horizons": [8, 9],
        },
        quick=False,
    )

    assert result["selected_horizon"] == 9
    assert result["validation_gate"] == "PASS"
    assert result["status"] == "PHYSICAL_TARGET_HORIZON_STABLE"
    assert result["full_rl_authorized"] is False


def test_validation_failure_does_not_trigger_horizon_reselection() -> None:
    rows = _selection_rows(validation_short_passes=False)
    for seed in (1, 2):
        for stratum in ("all", "high_magnitude"):
            rows.append(
                {
                    "split": "development",
                    "horizon": 10,
                    "policy_seed": seed,
                    "stratum": stratum,
                    "gate": "PASS",
                }
            )
            rows.append(
                {
                    "split": "validation",
                    "horizon": 10,
                    "policy_seed": seed,
                    "stratum": stratum,
                    "gate": "PASS",
                }
            )
    result = select_stable_horizon(
        rows,
        settings={
            "policy_seeds": [1, 2],
            "strata": ["all", "high_magnitude"],
            "candidate_horizons": [8, 9, 10],
        },
        quick=False,
    )

    assert result["development_candidates"] == [9, 10]
    assert result["selected_horizon"] == 9
    assert result["validation_gate"] == "FAIL"
    assert result["status"] == "DEVELOPMENT_SELECTED_HORIZON_NOT_VALIDATED"


def test_quick_mode_never_authorizes_rl() -> None:
    result = select_stable_horizon(
        [],
        settings={"policy_seeds": [1], "strata": [], "candidate_horizons": []},
        quick=True,
    )

    assert result["status"] == "QUICK_SMOKE_ONLY"
    assert result["full_rl_authorized"] is False
    assert result["s4d3_authorized"] is False
