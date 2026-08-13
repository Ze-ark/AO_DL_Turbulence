from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from src.rl.s4_closed_loop_shift import (
    ShiftScenario,
    _effective_settings,
    _oracle_normalized_action,
    _summarize_shift_scenarios,
    _zero_alignment,
    preflight_s4_closed_loop_shift,
)
from src.rl.s4_oracle_bound import SCIENCE_METRICS
from src.rl.s4_training import _load_yaml, _project_path


CONFIG = Path("configs/experiments/s4_closed_loop_shift_v1.yaml")


def _science(value: float, *, episodes: int = 16) -> dict[str, torch.Tensor]:
    return {
        metric: torch.full((episodes,), value if metric != "phase_rmse" else 0.7)
        for metric in SCIENCE_METRICS
    }


def test_oracle_action_uses_prior_and_pending_baseline_without_mutation() -> None:
    state = torch.zeros(2, 210)
    prior = torch.linspace(-0.1, 0.1, 21).repeat(2, 1)
    baseline = torch.zeros_like(prior)
    state[:, -42:-21] = prior
    state[:, -21:] = baseline
    target = prior.clone()
    target[:, 10:] += 0.025
    result = _oracle_normalized_action(
        state,
        target,
        residual_component_limit_rad=0.05,
        residual_l2_budget_rad=10**0.5 * 0.05,
    )
    assert result.shape == (2, 11)
    assert torch.allclose(result, torch.full_like(result, 0.5), atol=1e-6)
    assert torch.allclose(state[:, -42:-21], prior)


def test_shift_summary_detects_late_on_policy_growth() -> None:
    baseline = _science(0.5)
    candidate = _science(0.55)
    candidate["phase_rmse"] = torch.full((16,), 0.6)
    baseline["phase_rmse"] = torch.full((16,), 0.7)
    candidate["violation_fraction"] = torch.full((16,), 0.01)
    baseline["violation_fraction"] = torch.zeros(16)
    early = {metric: torch.full((16,), 0.01) for metric in (
        "raw_oracle_mse",
        "applied_oracle_mse",
        "raw_oracle_cosine",
        "state_abs_z_mean",
        "state_abs_z_gt3_fraction",
        "prior_added_request_rms_rad",
        "post_added_request_rms_rad",
        "requested_added_residual_rms_rad",
        "realized_added_residual_rms_rad",
        "residual_projection_fraction",
        "final_projection_fraction",
        "violation_fraction",
    )}
    late = {key: value.clone() for key, value in early.items()}
    late["raw_oracle_mse"] = torch.full((16,), 0.02)
    late["state_abs_z_gt3_fraction"] = torch.full((16,), 0.04)
    late["post_added_request_rms_rad"] = torch.full((16,), 0.02)
    scenario = ShiftScenario(
        predictor_id="mlp_seed_1",
        scale=1.0,
        profile_id="nominal",
        condition_id="test",
        base_seed=10,
        candidate=candidate,
        baseline=baseline,
        temporal={"early": early, "late": late},
    )
    result = _summarize_shift_scenarios(
        [scenario],
        gate={
            "proposed_min_relative_power_gain": 0.02,
            "min_power_delta_ci95_low": 0.0,
            "min_strehl_delta_ci95_low": 0.0,
            "max_phase_rmse_delta_ci95_high": 0.0,
            "max_violation_fraction": 0.05,
            "require_all_profiles_pass": True,
        },
        time_bins=[
            {"id": "early", "start": 0, "stop": 1},
            {"id": "late", "start": 1, "stop": 2},
        ],
        thresholds={
            "on_policy_mse_over_offline_reference_min": 2.0,
            "on_policy_mse_ratio_late_over_early_min": 1.5,
            "state_ood_fraction_increase_min": 0.02,
            "cumulative_added_request_ratio_late_over_early_min": 1.25,
        },
        offline_reference={"mse": 0.005, "mean_cosine_similarity": 0.7},
    )
    assert result["closed_loop_gate"] == "PASS"
    assert result["shift_signals"]["status"] == "SUPPORTED"
    assert result["shift_signals"]["teacher_distribution_gap"]


def test_zero_alignment_checks_every_science_metric() -> None:
    science = _science(0.5, episodes=2)
    scenario = ShiftScenario(
        predictor_id="ridge",
        scale=0.0,
        profile_id="nominal",
        condition_id="test",
        base_seed=1,
        candidate=science,
        baseline={key: value.clone() for key, value in science.items()},
        temporal={},
    )
    assert _zero_alignment([scenario], tolerance=1e-7)["status"] == "PASS"


def test_formal_preflight_locks_frozen_inventory_and_record_counts(tmp_path: Path) -> None:
    experiment_path = _project_path(CONFIG)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=False)
    settings["output_directory"] = str(tmp_path / "formal")
    result = preflight_s4_closed_loop_shift(
        experiment_path,
        experiment,
        settings,
        quick=False,
    )
    assert result["status"] == "READY_FOR_USER_FROZEN_CHECKPOINT_DIAGNOSTIC"
    assert result["planned_records"] == {
        "scenario_records": 360,
        "episode_records": 5760,
        "temporal_records": 1800,
        "temporal_episode_records": 28800,
    }
    assert result["optimizer_updates"] == 0
    assert result["checkpoint_writes"] == 0
    assert not result["future_truth_in_predictor_inputs"]
    assert not result["real_slm_actions"]
    assert result["experiment_config_sha256"]


def test_preflight_rejects_future_truth_in_predictor_observation(
    tmp_path: Path,
) -> None:
    experiment_path = _project_path(CONFIG)
    experiment = _load_yaml(experiment_path)
    broken = deepcopy(experiment)
    broken["policy_observation"]["include_future_truth"] = True
    settings = _effective_settings(broken, quick=False)
    settings["output_directory"] = str(tmp_path / "formal")
    with pytest.raises(RuntimeError, match="observation contract"):
        preflight_s4_closed_loop_shift(
            experiment_path,
            broken,
            settings,
            quick=False,
        )


def test_preflight_rejects_training_authorization(tmp_path: Path) -> None:
    experiment_path = _project_path(CONFIG)
    experiment = _load_yaml(experiment_path)
    broken = deepcopy(experiment)
    broken["metadata"]["allow_training"] = True
    settings = _effective_settings(broken, quick=False)
    settings["output_directory"] = str(tmp_path / "formal")
    with pytest.raises(RuntimeError, match="allow_training"):
        preflight_s4_closed_loop_shift(
            experiment_path,
            broken,
            settings,
            quick=False,
        )


def test_quick_bins_cover_the_complete_quick_episode() -> None:
    experiment = _load_yaml(_project_path(CONFIG))
    settings = _effective_settings(experiment, quick=True)
    assert settings["time_bins"][0]["start"] == 0
    assert settings["time_bins"][-1]["stop"] == settings["steps_per_episode"]
