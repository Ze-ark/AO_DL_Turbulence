"""S4-A复杂动态闭环条件与门槛测试。"""

from dataclasses import replace

import pytest

from src.simulation.config import S1EnvConfig
from src.simulation.robust_control import (
    RobustnessCondition,
    closed_loop_robustness_gate,
    sealed_closed_loop_gate,
)


def test_robustness_condition_applies_dynamic_and_delay_parameters():
    base = S1EnvConfig(
        grid_size=32,
        turbulence_grid_multiplier=2,
        batch_size=2,
        episode_length=5,
        num_modes=2,
        wind_speed_mps=0.1,
    )
    condition = RobustnessCondition(
        identifier="combined",
        regime="combined",
        base_seed=10,
        wind_speed_mps=0.2,
        wind_direction_deg=40,
        frozen_flow_rho=0.995,
        wind_speed_modulation_fraction=0.1,
        wind_direction_modulation_deg=10,
        wind_modulation_period_frames=8,
        observation_noise_std_rad=0.02,
        slm_delay_frames=2,
    )

    configured = condition.environment_config(base)

    assert configured.wind_speed_mps == 0.2
    assert configured.frozen_flow_rho == 0.995
    assert configured.wind_modulation_period_frames == 8
    assert configured.slm_delay_frames == 2


def test_closed_loop_gate_requires_improvement_and_safe_every_condition():
    summary = {
        "controller": "ridge_predictor",
        "violation_fraction": {"mean": 0.01},
        "paired_delta_vs_no_correction": {
            "power_in_bucket": {"ci95_low": 0.02},
            "strehl": {"ci95_low": 0.03},
            "phase_rmse": {"ci95_high": -0.04},
        },
    }
    records = [
        {
            "controller": "ridge_predictor",
            "condition_id": "a",
            "power_delta_vs_no_correction": 0.02,
            "violation_fraction": 0.01,
        },
        {
            "controller": "ridge_predictor",
            "condition_id": "b",
            "power_delta_vs_no_correction": 0.01,
            "violation_fraction": 0.02,
        },
    ]

    gate = closed_loop_robustness_gate(
        summary,
        records,
        min_power_delta_ci95_low=0,
        min_strehl_delta_ci95_low=0,
        max_phase_rmse_delta_ci95_high=0,
        max_violation_fraction=0.05,
        require_every_condition_power_positive=True,
    )

    assert gate["validation_gate"] == "PASS"
    assert gate["every_condition_safe"] is True


def test_closed_loop_gate_fails_when_one_condition_loses_power():
    summary = {
        "controller": "ridge_predictor",
        "violation_fraction": {"mean": 0.01},
        "paired_delta_vs_no_correction": {
            "power_in_bucket": {"ci95_low": 0.02},
            "strehl": {"ci95_low": 0.03},
            "phase_rmse": {"ci95_high": -0.04},
        },
    }
    records = [
        {
            "controller": "ridge_predictor",
            "condition_id": "a",
            "power_delta_vs_no_correction": -0.001,
            "violation_fraction": 0.01,
        }
    ]

    gate = closed_loop_robustness_gate(
        summary,
        records,
        min_power_delta_ci95_low=0,
        min_strehl_delta_ci95_low=0,
        max_phase_rmse_delta_ci95_high=0,
        max_violation_fraction=0.05,
        require_every_condition_power_positive=True,
    )

    assert gate["validation_gate"] == "FAIL"


def test_sealed_gate_requires_practical_relative_power_gain():
    candidate = {
        "controller": "leaky_integrator",
        "power_in_bucket": {"mean": 0.64},
        "violation_fraction": {"mean": 0.01},
        "paired_delta_vs_no_correction": {
            "power_in_bucket": {"ci95_low": 0.051},
            "strehl": {"ci95_low": 0.02},
            "phase_rmse": {"ci95_high": -0.03},
        },
    }
    reference = {"power_in_bucket": {"mean": 0.60}}
    records = [{
        "controller": "leaky_integrator",
        "power_delta_vs_no_correction": 0.04,
        "violation_fraction": 0.01,
    }]

    gate = sealed_closed_loop_gate(
        candidate,
        reference,
        records,
        min_power_delta_ci95_low=0.05,
        min_mean_relative_power_gain=0.10,
        min_strehl_delta_ci95_low=0,
        max_phase_rmse_delta_ci95_high=0,
        max_violation_fraction=0.05,
        require_every_condition_power_positive=True,
    )

    assert gate["validation_gate"] == "FAIL"
    assert gate["relative_power_gain_pass"] is False


def test_sealed_gate_passes_when_practical_and_safety_checks_pass():
    candidate = {
        "controller": "leaky_integrator",
        "power_in_bucket": {"mean": 0.68},
        "violation_fraction": {"mean": 0.01},
        "paired_delta_vs_no_correction": {
            "power_in_bucket": {"ci95_low": 0.051},
            "strehl": {"ci95_low": 0.02},
            "phase_rmse": {"ci95_high": -0.03},
        },
    }
    reference = {"power_in_bucket": {"mean": 0.60}}
    records = [{
        "controller": "leaky_integrator",
        "power_delta_vs_no_correction": 0.08,
        "violation_fraction": 0.01,
    }]

    gate = sealed_closed_loop_gate(
        candidate,
        reference,
        records,
        min_power_delta_ci95_low=0.05,
        min_mean_relative_power_gain=0.10,
        min_strehl_delta_ci95_low=0,
        max_phase_rmse_delta_ci95_high=0,
        max_violation_fraction=0.05,
        require_every_condition_power_positive=True,
    )

    assert gate["validation_gate"] == "PASS"
    assert gate["relative_power_gain_pass"] is True
