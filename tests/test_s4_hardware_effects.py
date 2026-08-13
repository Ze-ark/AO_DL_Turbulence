"""S4-C硬件误差模块的确定性单元测试。"""

from dataclasses import replace

import h5py
import pytest
import torch

from src.simulation.config import S1EnvConfig
from src.simulation.controllers import NoCorrectionController
from src.simulation.evaluation import export_rollout_h5, run_controller_rollout
from src.simulation.hardware_effects import (
    HardwareAwareSlmModel,
    HardwareEffectsConfig,
    HardwareProfile,
    apply_registration_error,
)
from src.simulation.slm import TorchSlmModel


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


def test_nominal_hardware_model_matches_original_slm_exactly():
    original = TorchSlmModel(-3.14, 3.14, 0.35, 256, 2)
    stressed = HardwareAwareSlmModel(
        -3.14,
        3.14,
        0.35,
        256,
        2,
        HardwareEffectsConfig(),
    )
    shape = (2, 8, 8)
    original.reset(shape, torch.device("cpu"), torch.float32)
    stressed.reset(shape, torch.device("cpu"), torch.float32)
    generator = torch.Generator().manual_seed(8)

    for _ in range(5):
        request = torch.randn(shape, generator=generator)
        expected, _ = original.step(request)
        actual, _ = stressed.step(request)
        assert torch.equal(actual, expected)


def test_phase_scale_and_settling_fraction_change_applied_phase():
    model = HardwareAwareSlmModel(
        -10,
        10,
        10,
        0,
        0,
        HardwareEffectsConfig(phase_scale=0.5, settling_fraction=0.5),
    )
    model.reset((1, 4, 4), torch.device("cpu"), torch.float32)

    applied, diagnostics = model.step(torch.ones(1, 4, 4))

    assert torch.allclose(applied, torch.full_like(applied, 0.25))
    assert diagnostics["settling_limited_fraction"].item() == pytest.approx(1)


def test_registration_shift_uses_zero_padding_instead_of_wraparound():
    phase = torch.zeros(1, 5, 5)
    phase[0, 2, 1] = 1

    shifted = apply_registration_error(
        phase,
        shift_x_pixels=1,
        shift_y_pixels=0,
        rotation_deg=0,
    )

    assert shifted[0, 2, 2].item() == pytest.approx(1, abs=1e-6)
    assert shifted[0, 2, 0].item() == pytest.approx(0, abs=1e-6)


def test_hardware_profile_applies_slm_parameters_and_validates_ranges():
    profile = HardwareProfile.from_mapping(
        {
            "id": "combined",
            "label": "组合",
            "severity": "moderate",
            "required_for_gate": True,
            "slm_delay_frames": 3,
            "slm_quantization_levels": 64,
            "slm_max_delta_rad": 0.2,
            "observation_noise_std_rad": 0.05,
            "phase_scale": 0.85,
            "settling_fraction": 0.5,
            "shift_x_pixels": 1,
            "rotation_deg": 0.5,
            "power_noise_relative_std": 0.03,
        }
    )

    configured = profile.environment_config(_config())

    assert configured.slm_delay_frames == 3
    assert configured.slm_quantization_levels == 64
    assert profile.effects_config().phase_scale == pytest.approx(0.85)
    with pytest.raises(ValueError, match="settling_fraction"):
        HardwareEffectsConfig(settling_fraction=0).validate()


def test_rollout_records_hardware_actions_measurement_and_timing(tmp_path):
    config = _config()
    effects = HardwareEffectsConfig(
        phase_scale=0.9,
        settling_fraction=0.75,
        power_noise_relative_std=0.03,
    )
    first = run_controller_rollout(
        config,
        "cpu",
        NoCorrectionController(config.num_modes, config.modal_limit_rad),
        seed=50,
        hardware_effects=effects,
    )
    repeated = run_controller_rollout(
        config,
        "cpu",
        NoCorrectionController(config.num_modes, config.modal_limit_rad),
        seed=50,
        hardware_effects=effects,
    )

    assert torch.equal(first.measured_power_in_bucket, repeated.measured_power_in_bucket)
    assert not torch.equal(first.measured_power_in_bucket, first.power_in_bucket)
    assert first.delayed_modal.shape == (2, 4, config.num_modes)
    assert first.registered_modal.shape == (2, 4, config.num_modes)
    assert first.action_latency_ms.shape == (4,)

    output = tmp_path / "hardware_trajectory.h5"
    export_rollout_h5(first, output, config, metadata={"stage": "S4-C0"})
    with h5py.File(output, "r") as handle:
        assert handle.attrs["stage"] == "S4-C0"
        assert handle["action/delayed_modal"].shape == (2, 4, config.num_modes)
        assert handle["action/registered_modal"].shape == (2, 4, config.num_modes)
        assert handle["metric/measured_power_in_bucket"].shape == (2, 4)
        assert handle["timing/action_latency_ms"].shape == (4,)
