"""小尺寸确定性CPU单元测试；不生成性能结论。"""
from dataclasses import replace
import pytest
import torch
from src.rl.r5_physics_adapter import BatchedDifferentiableSlm, DifferentiableSlm
from src.simulation.config import S1EnvConfig
from src.simulation.hardware_effects import HardwareProfile


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_mixed_profiles_match_forward_and_gradients(dtype):
    profiles = [HardwareProfile(str(i), str(i), 'test', False,
                slm_delay_frames=d, slm_quantization_levels=q, slm_max_delta_rad=.1+i*.05,
                settling_fraction=.5 if i % 2 else 1., phase_scale=.9 if i % 2 else 1.,
                shift_x_pixels=.7 if i % 2 else 0., rotation_deg=2. if i % 2 else 0.)
                for i, (d, q) in enumerate([(0, 0), (1, 64), (2, 128), (3, 256)])]
    cfg = S1EnvConfig(grid_size=8)
    batched = BatchedDifferentiableSlm(cfg, [p.as_record() for p in profiles], torch.device('cpu'))
    batched.reset((4, 8, 8), torch.device('cpu'), dtype)
    singles = [DifferentiableSlm(p.environment_config(cfg), p.effects_config()) for p in profiles]
    for one in singles:
        one.reset((1, 8, 8), torch.device('cpu'), dtype)
    pattern = torch.sin(torch.arange(4*8*8, dtype=dtype)).reshape(4, 8, 8)
    x = torch.tensor(.2, dtype=dtype, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    values, expected = [], []
    for t in range(9):
        a, ai = batched.step(pattern*x*(t+1)/9)
        bs = [m.step(pattern[i:i+1]*y*(t+1)/9) for i, m in enumerate(singles)]
        b = torch.cat([v[0] for v in bs])
        assert torch.equal(a, b)
        for key in ai:
            assert torch.equal(ai[key], torch.cat([v[1][key] for v in bs]))
        values.append(a.cos().sum())
        expected.append(b.cos().sum())
    ga = torch.autograd.grad(sum(values), x)[0]
    gb = torch.autograd.grad(sum(expected), y)[0]
    assert ga.abs() > 0
    torch.testing.assert_close(ga, gb, atol=1e-6 if dtype == torch.float32 else 1e-13, rtol=1e-6)


def test_invalid_request_rejected():
    p = HardwareProfile('a', 'a', 'test', False)
    m = BatchedDifferentiableSlm(S1EnvConfig(grid_size=8), [p.as_record()], torch.device('cpu'))
    with pytest.raises(RuntimeError):
        m.step(torch.zeros(1, 8, 8))
    with pytest.raises(ValueError):
        m.reset((2, 8, 8), torch.device('cpu'), torch.float32)
    m.reset((1, 8, 8), torch.device('cpu'), torch.float32)
    with pytest.raises(ValueError):
        m.step(torch.full((1, 8, 8), float('nan')))
