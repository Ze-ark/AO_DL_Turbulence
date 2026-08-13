from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.rl.residual_control import ResidualTrackingController
from src.rl.residual_sac import ResidualSacAgent, SacConfig, TransitionReplayBuffer
from src.rl.s4_training import _residual_reward, run_s4_residual_sac
from src.simulation.controllers import TrackingLeakyIntegratorController


ROOT = Path(__file__).resolve().parents[1]


def _observation(batch: int = 3, modes: int = 10) -> torch.Tensor:
    generator = torch.Generator().manual_seed(123)
    residual = 0.2 * torch.randn(batch, modes, generator=generator)
    applied = 0.1 * torch.randn(batch, modes, generator=generator)
    metrics = torch.tensor([[0.4, 0.6]]).expand(batch, -1)
    return torch.cat((residual, applied, metrics), dim=-1)


def _residual_controller() -> ResidualTrackingController:
    return ResidualTrackingController(
        num_modes=10,
        modal_limit_rad=3.0,
        history_frames=4,
        residual_action_limit_rad=0.05,
        final_action_step_limit_rad=0.15,
        gain=0.25,
        leak=0.10,
        tracking_gain=0.50,
    )


def test_policy_state_has_expected_size_and_omits_oracle_metrics() -> None:
    observation = _observation()
    changed_metrics = observation.clone()
    changed_metrics[:, -2:] = torch.tensor([99.0, -99.0])

    first = _residual_controller().reset(observation)
    second = _residual_controller().reset(changed_metrics)

    assert first.shape == (3, 100)
    assert torch.equal(first, second)


def test_zero_residual_exactly_recovers_frozen_tracking_controller() -> None:
    observation = _observation()
    residual = _residual_controller()
    state = residual.reset(observation)
    assert state.shape[1] == residual.state_size

    baseline = TrackingLeakyIntegratorController(
        num_modes=10,
        modal_limit_rad=3.0,
        gain=0.25,
        leak=0.10,
        tracking_gain=0.50,
        max_request_step_rad=0.15,
    )
    baseline.reset(3, torch.device("cpu"), torch.float32)
    expected = baseline.action(observation)
    actual = residual.compose_action(torch.zeros(3, 10)).final_delta_rad
    assert torch.equal(actual, expected)
    assert torch.equal(residual.requested_modal, baseline.requested_modal)

    next_observation = observation.clone()
    next_observation[:, :10] *= 0.8
    residual.advance_observation(next_observation)
    expected_next = baseline.action(next_observation)
    actual_next = residual.compose_action(torch.zeros(3, 10)).final_delta_rad
    assert torch.equal(actual_next, expected_next)


def test_residual_action_cannot_expand_final_step_limit() -> None:
    controller = _residual_controller()
    controller.reset(_observation())
    action = controller.compose_action(torch.full((3, 10), 5.0))

    assert torch.all(action.normalized_request <= 1)
    assert torch.all(action.requested_residual_rad <= 0.05)
    assert torch.all(action.final_delta_rad.abs() <= 0.15 + 1e-7)
    assert torch.all(controller.requested_modal.abs() <= 3.0)


def test_replay_buffer_and_sac_update_are_finite_on_cpu_unit_test() -> None:
    config = SacConfig(state_size=12, action_size=2, hidden_size=32)
    agent = ResidualSacAgent(config, torch.device("cpu"))
    replay = TransitionReplayBuffer(128, 12, 2, seed=7)
    generator = torch.Generator().manual_seed(9)
    states = torch.randn(64, 12, generator=generator)
    actions = torch.tanh(torch.randn(64, 2, generator=generator))
    rewards = torch.randn(64, generator=generator)
    next_states = torch.randn(64, 12, generator=generator)
    dones = torch.zeros(64, dtype=torch.bool)
    replay.add_batch(states, actions, rewards, next_states, dones)

    metrics = agent.update(replay.sample(32, torch.device("cpu")))
    assert set(metrics) == {
        "critic_loss",
        "actor_loss",
        "alpha_loss",
        "alpha",
        "mean_q",
    }
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    deterministic = agent.act(states[:4], deterministic=True)
    assert deterministic.shape == (4, 2)
    assert torch.all(deterministic.abs() <= 1)


def test_reward_uses_measured_power_and_penalizes_residual_and_violation() -> None:
    reward_config = {
        "measured_power_weight": 1.0,
        "residual_action_weight": 0.01,
        "violation_weight": 0.5,
    }
    power = torch.tensor([0.7, 0.7])
    zero = _residual_reward(
        measured_power=power,
        normalized_residual=torch.zeros(2, 10),
        violation=torch.zeros(2),
        reward_config=reward_config,
    )
    penalized = _residual_reward(
        measured_power=power,
        normalized_residual=torch.ones(2, 10),
        violation=torch.tensor([0.1, 0.1]),
        reward_config=reward_config,
    )
    assert torch.allclose(zero, power)
    assert torch.all(penalized < zero)


def test_s4d2_preflight_verifies_d1_and_seed_isolation_without_cuda() -> None:
    quick_output = ROOT / "outputs" / "s4_residual_sac_v1_quick"
    if quick_output.exists():
        pytest.skip("quick output is intentionally preserved after a prior smoke run")
    result = run_s4_residual_sac(
        ROOT / "configs" / "experiments" / "s4_residual_sac_v1.yaml",
        quick=True,
        preflight_only=True,
    )
    assert result["status"] == "READY_FOR_QUICK_SMOKE"
    assert result["state_size"] == 100
    assert result["upstream_s4d1"]["gate"] == "PASS"
    assert result["upstream_s4d1"]["d1_data_allowed_in_training"] is False
    assert result["seed_isolation_verified"] is True
    assert result["sealed_s4d3_access"] is False
