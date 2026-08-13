"""S1 动态环境的 reset、step、动作和 GPU 测试。"""

from dataclasses import replace

import pytest
import torch

from src.simulation.config import S1EnvConfig
from src.simulation.env import AdaptiveOpticsEnv


def _test_config(**overrides):
    base = S1EnvConfig(
        grid_size=32,
        turbulence_grid_multiplier=1,
        batch_size=2,
        episode_length=3,
        num_modes=4,
        slm_delay_frames=0,
        slm_quantization_levels=0,
        slm_max_delta_rad=10,
        slm_phase_min_rad=-10,
        slm_phase_max_rad=10,
        wind_speed_mps=0.2,
    )
    return replace(base, **overrides)


def test_reset_is_deterministic_and_observation_has_declared_shape():
    config = _test_config()
    environment = AdaptiveOpticsEnv(config, "cpu")

    first, _ = environment.reset(seed=17)
    repeated, _ = environment.reset(seed=17)

    assert torch.equal(first, repeated)
    assert first.shape == (config.batch_size, config.observation_size)
    assert environment.episode_seeds.tolist() == [17, 18]


def test_step_advances_time_and_terminates_at_episode_length():
    config = _test_config(episode_length=2)
    environment = AdaptiveOpticsEnv(config, "cpu")
    environment.reset(seed=3)
    action = torch.zeros(config.batch_size, config.num_modes)

    _, _, first_terminal, _, _ = environment.step(action)
    _, _, second_terminal, _, _ = environment.step(action)

    assert not first_terminal.any()
    assert second_terminal.all()
    assert environment.step_count == 2


def test_known_modal_conjugate_action_improves_strehl():
    config = _test_config(batch_size=1, num_modes=1, wind_speed_mps=0, reward_action_weight=0)
    environment = AdaptiveOpticsEnv(config, "cpu")
    turbulence = 0.8 * environment.basis[0].unsqueeze(0)
    _, before = environment.reset(seed=1, turbulence_phase=turbulence)

    _, _, _, _, info = environment.step(torch.tensor([[-0.8]]))

    assert info["reward_strehl"].item() > before["strehl"].item()
    assert info["reward_strehl"].item() == pytest.approx(1.0, abs=1e-5)


def test_zero_turbulence_and_zero_action_keep_ideal_focus():
    config = _test_config(batch_size=1, wind_speed_mps=0)
    environment = AdaptiveOpticsEnv(config, "cpu")
    zero_phase = torch.zeros(1, config.grid_size, config.grid_size)
    _, before = environment.reset(seed=4, turbulence_phase=zero_phase)

    _, _, _, _, after = environment.step(torch.zeros(1, config.num_modes))

    assert before["strehl"].item() == pytest.approx(1.0, abs=1e-6)
    assert after["strehl"].item() == pytest.approx(1.0, abs=1e-6)


def test_ideal_wavefront_observation_contains_pupil_intensity_and_wrapped_residual_phase():
    config = _test_config(batch_size=1, num_modes=1, wind_speed_mps=0)
    environment = AdaptiveOpticsEnv(config, "cpu")
    turbulence = 4.2 * environment.basis[0].unsqueeze(0)
    environment.reset(seed=1, turbulence_phase=turbulence)

    wavefront = environment.ideal_wavefront_observation()

    assert wavefront.shape == (1, 2, config.grid_size, config.grid_size)
    assert torch.equal(wavefront[0, 0].bool(), environment.pupil)
    assert wavefront[:, 1].abs().max().item() <= torch.pi
    assert torch.equal(wavefront[0, 1][~environment.pupil], torch.zeros_like(wavefront[0, 1][~environment.pupil]))


def test_action_delay_is_visible_in_applied_modal_history():
    config = _test_config(batch_size=1, num_modes=1, slm_delay_frames=2, wind_speed_mps=0)
    environment = AdaptiveOpticsEnv(config, "cpu")
    environment.reset(seed=1, turbulence_phase=torch.zeros(1, 32, 32))
    command = torch.tensor([[0.5]])

    _, _, _, _, first = environment.step(command)
    _, _, _, _, second = environment.step(torch.zeros_like(command))
    _, _, _, _, third = environment.step(torch.zeros_like(command))

    assert torch.allclose(first["applied_modal"], torch.zeros_like(command), atol=1e-6)
    assert torch.allclose(second["applied_modal"], torch.zeros_like(command), atol=1e-6)
    assert torch.allclose(third["applied_modal"], command, atol=1e-5)


def test_focal_fft_conserves_energy():
    config = _test_config()
    environment = AdaptiveOpticsEnv(config, "cpu")

    _, metrics = environment.reset(seed=4)

    assert torch.max(metrics["relative_energy_error"]).item() < 2e-6


def test_config_rejects_a_screen_that_repeats_inside_one_episode():
    config = _test_config(
        turbulence_grid_multiplier=1,
        wind_speed_mps=100,
        episode_length=200,
    )

    with pytest.raises(ValueError, match="wrap completely"):
        config.validate()


def test_config_requires_a_period_when_time_varying_wind_is_enabled():
    config = _test_config(wind_speed_modulation_fraction=0.1)

    with pytest.raises(ValueError, match="period_frames"):
        config.validate()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the S1 GPU smoke test")
def test_s1_environment_runs_on_cuda_without_cpu_fallback():
    config = _test_config(batch_size=4)
    environment = AdaptiveOpticsEnv(config, "cuda")

    observation, _ = environment.reset(seed=5)
    next_observation, reward, _, _, _ = environment.step(
        torch.zeros(config.batch_size, config.num_modes, device="cuda")
    )

    assert observation.is_cuda
    assert next_observation.is_cuda
    assert reward.is_cuda
