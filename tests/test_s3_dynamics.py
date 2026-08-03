"""S3多帧动力学门槛的确定性小尺寸测试。"""

from dataclasses import replace

import pytest
import torch

from src.simulation.config import S1EnvConfig
from src.simulation.temporal_dynamics import (
    DynamicsCondition,
    GRUModalDynamics,
    condition_gate,
    constant_velocity_forecast,
    fit_ridge_autoregression,
    generate_temporal_dynamics_data,
    modal_normalization,
    paired_episode_comparison,
    ridge_autoregressive_forecast,
)


def _config(**overrides) -> S1EnvConfig:
    base = S1EnvConfig(
        grid_size=32,
        turbulence_grid_multiplier=2,
        batch_size=2,
        episode_length=6,
        num_modes=2,
        slm_delay_frames=0,
        slm_quantization_levels=0,
        slm_max_delta_rad=10,
        slm_phase_min_rad=-10,
        slm_phase_max_rad=10,
        wind_speed_mps=0,
    )
    return replace(base, **overrides)


def test_temporal_data_preserves_complete_episode_and_condition_metadata():
    progress: list[tuple[int, int]] = []
    conditions = [
        DynamicsCondition("first", 0.0, 0.0, 70),
        DynamicsCondition("second", 0.0, 45.0, 80),
    ]
    data = generate_temporal_dynamics_data(
        _config(),
        "cpu",
        conditions,
        sequence_length=3,
        frames_per_episode=5,
        random_action_std_rad=0.1,
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )

    assert data.histories.shape == (12, 3, 2)
    assert data.targets.shape == (12, 2)
    assert data.episode_seed.tolist() == [70, 70, 70, 71, 71, 71, 80, 80, 80, 81, 81, 81]
    assert data.condition_index.tolist() == [0] * 6 + [1] * 6
    assert data.target_step.tolist() == [3, 4, 5] * 4
    assert progress[-1] == (10, 10)
    assert torch.allclose(data.histories[:, -1], data.targets, atol=1e-5)


def test_training_normalization_has_one_value_per_mode():
    condition = [DynamicsCondition("still", 0.0, 0.0, 90)]
    data = generate_temporal_dynamics_data(
        _config(),
        "cpu",
        condition,
        sequence_length=2,
        frames_per_episode=4,
        random_action_std_rad=0,
    )

    mean, scale = modal_normalization(data)

    assert mean.shape == (2,)
    assert scale.shape == (2,)
    assert torch.all(scale > 0)


def test_gru_model_returns_one_modal_vector_per_sequence():
    model = GRUModalDynamics(num_modes=3, hidden_size=8, num_layers=2, dropout=0.1)

    output = model(torch.randn(4, 5, 3))

    assert output.shape == (4, 3)


def test_constant_velocity_forecast_uses_last_two_frames():
    history = torch.tensor([[[0.0, 1.0], [1.0, 2.0], [3.0, 4.0]]])

    prediction = constant_velocity_forecast(history, velocity_gain=1.0)

    assert torch.equal(prediction, torch.tensor([[5.0, 6.0]]))


def test_ridge_baseline_uses_full_history_and_recovers_linear_mapping():
    generator = torch.Generator().manual_seed(5)
    histories = torch.randn(40, 3, 2, generator=generator)
    targets = 0.25 * histories[:, 0] - 0.5 * histories[:, 1] + 1.5 * histories[:, 2]
    mean = histories.reshape(-1, 2).mean(dim=0)
    scale = histories.reshape(-1, 2).std(dim=0)

    weight, bias = fit_ridge_autoregression(
        histories,
        targets,
        mean,
        scale,
        ridge_alpha=1e-8,
    )
    prediction = ridge_autoregressive_forecast(
        histories,
        mean,
        scale,
        weight,
        bias,
    )

    assert torch.allclose(prediction, targets, atol=1e-5)


def test_episode_skill_gate_passes_for_consistent_improvement():
    target = torch.zeros(4, 2)
    gru_prediction = torch.zeros_like(target)
    linear_prediction = torch.ones_like(target)
    comparison = paired_episode_comparison(
        gru_prediction,
        linear_prediction,
        target,
        torch.ones(2),
        episode_seed=torch.tensor([10, 10, 20, 20]),
        condition_index=torch.tensor([0, 0, 1, 1]),
    )

    gate = condition_gate(
        comparison,
        min_mean_skill_score=0.05,
        min_ci95_low=0,
        require_every_condition_positive=True,
    )

    assert comparison["skill_score"]["mean"] == pytest.approx(1)
    assert gate["validation_gate"] == "PASS"
    assert gate["every_condition_positive"] is True
