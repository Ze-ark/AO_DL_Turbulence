"""S1 泰勒冻结流的时间推进测试。"""

import torch

from src.simulation.turbulence import (
    advance_taylor_frozen_flow,
    phase_structure_function,
    periodic_fourier_shift,
    von_karman_phase_screens,
)


def test_integer_fourier_shift_matches_periodic_roll():
    phase = torch.arange(64, dtype=torch.float64).reshape(1, 8, 8)

    shifted = periodic_fourier_shift(phase, shift_x_pixels=2, shift_y_pixels=-1)

    expected = torch.roll(phase, shifts=(-1, 2), dims=(-2, -1))
    assert torch.allclose(shifted, expected, atol=1e-12, rtol=0)


def test_frozen_flow_is_reversible_for_fractional_translation():
    coordinate = torch.arange(32, dtype=torch.float64)
    y, x = torch.meshgrid(coordinate, coordinate, indexing="ij")
    phase = (torch.sin(2 * torch.pi * x / 32) + 0.4 * torch.cos(4 * torch.pi * y / 32))[None]

    advanced = advance_taylor_frozen_flow(phase, 0.37, -0.23, sample_pitch_m=1.0)
    restored = advance_taylor_frozen_flow(advanced, -0.37, 0.23, sample_pitch_m=1.0)

    assert torch.allclose(restored, phase - phase.mean(), atol=2e-12, rtol=0)


def test_zero_rho_returns_piston_removed_innovation():
    phase = torch.randn(2, 16, 16)
    innovation = torch.randn(2, 16, 16)

    advanced = advance_taylor_frozen_flow(
        phase,
        0.2,
        0.3,
        sample_pitch_m=1.0,
        rho=0,
        innovation=innovation,
    )

    assert torch.allclose(advanced, innovation - innovation.mean(dim=(-2, -1), keepdim=True))


def test_von_karman_screen_seed_is_reproducible():
    generator = torch.Generator().manual_seed(9)
    first = von_karman_phase_screens(2, 32, 0.4, 0.12, 10, 0.01, generator, torch.device("cpu"))
    generator.manual_seed(9)

    repeated = von_karman_phase_screens(2, 32, 0.4, 0.12, 10, 0.01, generator, torch.device("cpu"))

    assert torch.equal(first, repeated)
    assert torch.allclose(first.mean(dim=(-2, -1)), torch.zeros(2), atol=1e-6)


def test_von_karman_screens_follow_inertial_range_structure_function():
    generator = torch.Generator().manual_seed(21)
    r0 = 0.08
    screens = von_karman_phase_screens(
        24,
        128,
        1.0,
        r0,
        100.0,
        1e-3,
        generator,
        torch.device("cpu"),
        torch.float64,
    )

    separation, measured = phase_structure_function(screens, 1 / 128, 12)
    fit_slice = slice(1, 8)
    design = torch.stack((torch.log(separation[fit_slice]), torch.ones(7)), dim=1)
    slope = torch.linalg.lstsq(
        design,
        torch.log(measured[fit_slice]),
    ).solution[0]
    expected = 6.88 * (separation / r0) ** (5 / 3)
    amplitude_ratio = torch.median(measured[fit_slice] / expected[fit_slice])

    assert 1.15 < slope.item() < 2.15
    assert 0.35 < amplitude_ratio.item() < 2.8
