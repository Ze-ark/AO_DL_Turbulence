"""S2 传统模态控制器和配对评估的确定性小尺寸测试。"""

from dataclasses import replace

import h5py
import pytest
import torch

from src.simulation.config import S1EnvConfig
from src.simulation.controllers import (
    DirectProjectionController,
    LeakyIntegratorController,
    LinearPredictiveController,
    NoCorrectionController,
    RidgePredictiveController,
    ResUNetModalController,
    TrackingLeakyIntegratorController,
)
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.evaluation import (
    export_rollout_h5,
    paired_delta,
    run_controller_rollout,
    summarize_rollouts,
)
from src.simulation.resunet_training import (
    generate_wavefront_supervised_data,
    modal_prediction_metrics,
    resunet_phase_modal_loss,
)


def _config(**overrides) -> S1EnvConfig:
    base = S1EnvConfig(
        grid_size=32,
        turbulence_grid_multiplier=1,
        batch_size=2,
        episode_length=4,
        num_modes=2,
        slm_delay_frames=0,
        slm_quantization_levels=0,
        slm_max_delta_rad=10,
        slm_phase_min_rad=-10,
        slm_phase_max_rad=10,
        wind_speed_mps=0.1,
    )
    return replace(base, **overrides)


def _observation(residual: torch.Tensor, applied: torch.Tensor) -> torch.Tensor:
    batch = residual.shape[0]
    return torch.cat((residual, applied, torch.ones(batch, 2)), dim=1)


def test_no_correction_keeps_zero_requested_command():
    controller = NoCorrectionController(num_modes=2, modal_limit_rad=3)
    controller.reset(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)

    action = controller.action(_observation(torch.tensor([[0.5, -0.4]]), torch.zeros(1, 2)))

    assert torch.equal(action, torch.zeros_like(action))


def test_direct_projection_requests_negative_inferred_turbulence():
    controller = DirectProjectionController(num_modes=2, modal_limit_rad=3)
    controller.reset(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
    residual = torch.tensor([[0.7, -0.2]])
    applied = torch.tensor([[0.1, 0.3]])

    action = controller.action(_observation(residual, applied))

    assert torch.allclose(action, applied - residual)


def test_leaky_integrator_updates_incremental_request():
    controller = LeakyIntegratorController(
        num_modes=1,
        modal_limit_rad=3,
        gain=0.5,
        leak=0.1,
    )
    controller.reset(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
    observation = _observation(torch.tensor([[1.0]]), torch.zeros(1, 1))

    first = controller.action(observation)
    second = controller.action(observation)

    assert first.item() == pytest.approx(-0.5)
    assert second.item() == pytest.approx(-0.45)
    assert controller.requested_modal.item() == pytest.approx(-0.95)


def test_tracking_integrator_uses_actual_action_and_limits_request_step():
    controller = TrackingLeakyIntegratorController(
        num_modes=1,
        modal_limit_rad=3,
        gain=0.4,
        leak=0.1,
        tracking_gain=0.5,
        max_request_step_rad=0.1,
    )
    controller.reset(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
    controller.requested_modal.fill_(-1.0)
    observation = _observation(
        residual=torch.tensor([[0.5]]),
        applied=torch.tensor([[-0.2]]),
    )

    action = controller.action(observation)

    assert action.item() == pytest.approx(0.1)
    assert controller.requested_modal.item() == pytest.approx(-0.9)


def test_tracking_integrator_reduces_to_leaky_when_tracking_is_disabled():
    leaky = LeakyIntegratorController(1, 3, gain=0.3, leak=0.1)
    tracking = TrackingLeakyIntegratorController(
        1,
        3,
        gain=0.3,
        leak=0.1,
        tracking_gain=0,
    )
    observation = _observation(torch.tensor([[0.7]]), torch.tensor([[0.2]]))
    for controller in (leaky, tracking):
        controller.reset(1, torch.device("cpu"), torch.float32)

    assert torch.equal(leaky.action(observation), tracking.action(observation))


def test_linear_predictor_extrapolates_disturbance_for_delay():
    controller = LinearPredictiveController(
        num_modes=1,
        modal_limit_rad=10,
        prediction_horizon=2,
        velocity_gain=1,
    )
    controller.reset(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)

    first = controller.action(_observation(torch.tensor([[1.0]]), torch.zeros(1, 1)))
    second = controller.action(_observation(torch.tensor([[1.5]]), torch.zeros(1, 1)))

    assert first.item() == pytest.approx(-1.0)
    assert second.item() == pytest.approx(-1.5)
    assert controller.requested_modal.item() == pytest.approx(-2.5)


def test_ridge_predictor_uses_fixed_history_and_two_frame_target():
    controller = RidgePredictiveController(
        num_modes=1,
        modal_limit_rad=10,
        sequence_length=2,
        prediction_horizon=2,
        normalization_mean=torch.zeros(1),
        normalization_scale=torch.ones(1),
        weight=torch.tensor([[0.0], [1.0]]),
        bias=torch.zeros(1),
    )
    controller.reset(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)

    first = controller.action(_observation(torch.tensor([[1.0]]), torch.zeros(1, 1)))
    second = controller.action(_observation(torch.tensor([[2.0]]), torch.zeros(1, 1)))

    assert first.item() == pytest.approx(-1.0)
    assert second.item() == pytest.approx(-1.0)
    assert controller.requested_modal.item() == pytest.approx(-2.0)


def test_resunet_controller_projects_predicted_residual_to_same_modal_action_space():
    config = _config(batch_size=1, num_modes=1, wind_speed_mps=0)
    environment = AdaptiveOpticsEnv(config, "cpu")

    class PhaseChannelModel(torch.nn.Module):
        def forward(self, value):
            return value[:, 1:2]

    controller = ResUNetModalController(
        config.num_modes,
        config.modal_limit_rad,
        PhaseChannelModel(),
        environment.basis,
        environment.pupil,
    )
    turbulence = 0.6 * environment.basis[0].unsqueeze(0)
    observation, _ = environment.reset(seed=1, turbulence_phase=turbulence)
    controller.reset(1, torch.device("cpu"), torch.float32)

    action = controller.action(observation, environment.ideal_wavefront_observation())

    assert action.item() == pytest.approx(-0.6, abs=1e-5)


def test_oracle_upper_bound_removes_controllable_modal_component():
    config = _config(batch_size=1, num_modes=1, wind_speed_mps=0)
    environment = AdaptiveOpticsEnv(config, "cpu")
    turbulence = 0.8 * environment.basis[0].unsqueeze(0)
    environment.reset(seed=5, turbulence_phase=turbulence)

    upper = environment.oracle_modal_upper_bound()

    assert upper["strehl"].item() == pytest.approx(1.0, abs=1e-5)
    assert upper["phase_rmse"].item() < 1e-5


def test_rollout_is_reproducible_and_exports_dynamic_contract(tmp_path):
    config = _config()
    first = run_controller_rollout(
        config,
        "cpu",
        NoCorrectionController(config.num_modes, config.modal_limit_rad),
        seed=40,
        include_oracle_upper_bound=True,
    )
    repeated = run_controller_rollout(
        config,
        "cpu",
        NoCorrectionController(config.num_modes, config.modal_limit_rad),
        seed=40,
    )

    assert torch.equal(first.observations, repeated.observations)
    assert first.observations.shape == (2, 5, config.observation_size)
    assert first.requested_modal.shape == (2, 4, config.num_modes)
    assert summarize_rollouts([first])["episodes"] == 2
    assert paired_delta([first], [repeated], "strehl")["mean"] == pytest.approx(0)

    output = tmp_path / "trajectory.h5"
    export_rollout_h5(first, output, config)
    with h5py.File(output, "r") as handle:
        assert handle.attrs["stage"] == "S2"
        assert handle.attrs["observation_kind"] == "oracle_modal_features"
        assert handle["observation/features"].shape == (2, 5, config.observation_size)
        assert handle["observation/true_features"].shape == (
            2,
            5,
            config.observation_size,
        )
        assert handle["action/requested_modal"].shape == (2, 4, config.num_modes)
        assert handle["meta/episode_seed"][:].tolist() == [40, 41]


def test_rollout_modal_noise_is_reproducible_and_does_not_change_true_metrics():
    config = _config()
    noisy = run_controller_rollout(
        config,
        "cpu",
        NoCorrectionController(config.num_modes, config.modal_limit_rad),
        seed=44,
        observation_noise_std_rad=0.1,
    )
    repeated = run_controller_rollout(
        config,
        "cpu",
        NoCorrectionController(config.num_modes, config.modal_limit_rad),
        seed=44,
        observation_noise_std_rad=0.1,
    )
    clean = run_controller_rollout(
        config,
        "cpu",
        NoCorrectionController(config.num_modes, config.modal_limit_rad),
        seed=44,
    )

    assert torch.equal(noisy.observations, repeated.observations)
    assert torch.equal(noisy.true_observations, clean.true_observations)
    assert torch.equal(noisy.strehl, clean.strehl)
    assert not torch.equal(
        noisy.observations[:, :, : config.num_modes],
        noisy.true_observations[:, :, : config.num_modes],
    )
    assert torch.equal(
        noisy.observations[:, :, config.num_modes :],
        noisy.true_observations[:, :, config.num_modes :],
    )


def test_dynamic_resunet_data_preserves_complete_episode_metadata():
    config = _config(batch_size=2, episode_length=3, num_modes=1)
    progress_updates: list[tuple[int, int]] = []
    data = generate_wavefront_supervised_data(
        config,
        "cpu",
        base_seeds=[70],
        frames_per_episode=3,
        random_action_std_rad=0,
        progress_callback=lambda completed, total: progress_updates.append((completed, total)),
    )

    assert data.inputs.shape == (6, 2, 32, 32)
    assert data.target_phase.shape == (6, 1, 32, 32)
    assert data.episode_seed.tolist() == [70, 71, 70, 71, 70, 71]
    assert data.step_id.tolist() == [0, 0, 1, 1, 2, 2]
    assert progress_updates == [(1, 3), (2, 3), (3, 3)]


def test_phase_modal_loss_and_metrics_are_ideal_for_exact_prediction():
    config = _config(batch_size=2, num_modes=1)
    environment = AdaptiveOpticsEnv(config, "cpu")
    target = torch.stack((0.4 * environment.basis[0], -0.2 * environment.basis[0]))[:, None]

    loss = resunet_phase_modal_loss(
        target,
        target,
        environment.basis,
        environment.pupil,
        modal_weight=1,
    )
    metrics = modal_prediction_metrics(
        target,
        target,
        environment.basis,
        environment.pupil,
    )

    assert loss["total"].item() == pytest.approx(0)
    assert torch.allclose(metrics["modal_cosine"], torch.ones(2))
    assert torch.allclose(metrics["relative_modal_rmse"], torch.zeros(2))
