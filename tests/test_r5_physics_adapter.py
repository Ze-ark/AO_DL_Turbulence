"""小型确定性CPU数学测试；不生成正式性能结果。"""
from dataclasses import replace

import pytest
import torch

from src.rl.r5_physics_adapter import DifferentiableSlm, FixedRegistration, quantize_st
from src.simulation.config import S1EnvConfig
from src.simulation.hardware_effects import HardwareAwareSlmModel, HardwareEffectsConfig, apply_registration_error


@pytest.mark.parametrize("levels", [0, 1, 64, 256])
def test_quantizer_exact_forward_and_surrogate_gradient(levels):
    x = torch.linspace(-3.14, 3.14, 35, dtype=torch.float64, requires_grad=True)
    y = quantize_st(x, -3.14, 3.14, levels)
    step = 6.28 / max(1, levels - 1)
    expected = x if levels <= 1 else -3.14 + torch.round((x - -3.14) / step) * step
    assert torch.equal(y, expected)
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


@pytest.mark.parametrize("shift,rotation", [(0., 0.), (.7, 3.), (-1.1, -5.), (30., 0.)])
def test_registration_adjoint_matches_original_autograd(shift, rotation):
    x = torch.linspace(-.3, .4, 128, dtype=torch.float64).reshape(2, 8, 8).requires_grad_()
    e = HardwareEffectsConfig(shift_x_pixels=shift, shift_y_pixels=-shift/2, rotation_deg=rotation)
    adapter = FixedRegistration(x, e)
    actual = adapter(x)
    expected = apply_registration_error(x, shift_x_pixels=shift, shift_y_pixels=-shift/2, rotation_deg=rotation)
    assert torch.equal(actual, expected)
    probe = torch.cos(x.detach() * 10)
    g1 = torch.autograd.grad((actual * probe).sum(), x)[0]
    g2 = torch.autograd.grad((expected * probe).sum(), x)[0]
    torch.testing.assert_close(g1, g2, atol=2e-14, rtol=1e-13)


@pytest.mark.parametrize("delay", [0, 2, 3])
@pytest.mark.parametrize("settling", [.5, 1.])
def test_state_chain_hard_forward_parity_and_not_inplace(delay, settling):
    cfg = S1EnvConfig(grid_size=8, slm_delay_frames=delay)
    effects = HardwareEffectsConfig(settling_fraction=settling, shift_x_pixels=.3, rotation_deg=2.)
    adapter = DifferentiableSlm(cfg, effects)
    original = HardwareAwareSlmModel(cfg.slm_phase_min_rad, cfg.slm_phase_max_rad,
                                    cfg.slm_max_delta_rad, 256, delay, effects)
    for model in (adapter, original):
        model.reset((1, 8, 8), torch.device("cpu"), torch.float32)
    initial = adapter.state
    for step in range(10):
        request = torch.full((1, 8, 8), [0., .4, 4., -4., .05][step % 5])
        p1, d1 = adapter.step(request)
        p2, d2 = original.step(request)
        assert torch.equal(p1, p2)
        for key in d1:
            assert torch.equal(d1[key], d2[key])
    assert not initial.phase.any() and not initial.queue.any()


@pytest.mark.parametrize("delay", [0, 2, 3])
@pytest.mark.parametrize("settling", [.5, 1.])
def test_delayed_gradient_onset(delay, settling):
    cfg = S1EnvConfig(grid_size=8, slm_delay_frames=delay, slm_quantization_levels=0)
    model = DifferentiableSlm(cfg, HardwareEffectsConfig(settling_fraction=settling))
    model.reset((1, 2, 2), torch.device("cpu"), torch.float64)
    amplitude = torch.tensor(.1, dtype=torch.float64, requires_grad=True)
    values = []
    for t in range(5):
        request = amplitude.expand(1, 2, 2) if t == 0 else amplitude.expand(1, 2, 2)*0
        result, _ = model.step(request)
        values.append(result.mean())
    assert all(values[t] == 0 for t in range(delay))
    g = torch.autograd.grad(values[delay], amplitude)[0]
    assert float(g) == pytest.approx(settling)


def test_saturation_blocks_gradient_and_slew_is_not_removed():
    cfg = S1EnvConfig(grid_size=8, slm_delay_frames=0, slm_quantization_levels=0)
    model = DifferentiableSlm(cfg, HardwareEffectsConfig())
    model.reset((1, 2, 2), torch.device("cpu"), torch.float64)
    x = torch.full((1, 2, 2), 20., dtype=torch.float64, requires_grad=True)
    phase, info = model.step(x)
    phase.sum().backward()
    assert not x.grad.any()
    assert info["saturated_fraction"].item() == 1
    assert info["slew_limited_fraction"].item() == 1
    assert phase.max().item() == pytest.approx(cfg.slm_max_delta_rad)


def test_smooth_chain_finite_difference():
    cfg = S1EnvConfig(grid_size=8, slm_delay_frames=2, slm_quantization_levels=0)
    effects = HardwareEffectsConfig(settling_fraction=.5, rotation_deg=3., shift_x_pixels=.3)
    pattern = torch.sin(torch.arange(64, dtype=torch.float64)).reshape(1, 8, 8)*.1
    def loss(x):
        model = DifferentiableSlm(cfg, effects)
        model.reset(pattern.shape, pattern.device, pattern.dtype)
        values = []
        for t in range(8):
            phase, _ = model.step(pattern*x*(t+1)/8)
            values.append(torch.cos(phase+.2).mean())
        return torch.stack(values).mean()
    x = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    derivative = torch.autograd.grad(loss(x), x)[0]
    difference = (loss(x.detach()+1e-5)-loss(x.detach()-1e-5))/(2e-5)
    torch.testing.assert_close(derivative, difference, atol=1e-9, rtol=1e-6)


def test_no_reset_and_nonfinite_rejected():
    model = DifferentiableSlm(S1EnvConfig(grid_size=8), HardwareEffectsConfig())
    with pytest.raises(RuntimeError): model.step(torch.zeros(1, 8, 8))
    model.reset((1, 8, 8), torch.device("cpu"), torch.float32)
    with pytest.raises(ValueError): model.step(torch.full((1, 8, 8), float("nan")))
