"""低阶实 Zernike 模态的离散正交基。"""

from __future__ import annotations

import torch


MODE_NAMES = (
    "水平倾斜",
    "垂直倾斜",
    "离焦",
    "零度像散",
    "四十五度像散",
    "水平彗差",
    "垂直彗差",
    "水平三叶",
    "垂直三叶",
    "球差",
)


def make_low_order_zernike_basis(
    grid_size: int,
    pupil_radius_fraction: float,
    num_modes: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """构造去掉 piston 的前十个低阶实模态，并在离散瞳面内正交化。"""
    if not 1 <= num_modes <= len(MODE_NAMES):
        raise ValueError(f"num_modes must be between 1 and {len(MODE_NAMES)}")
    coordinate = torch.linspace(-1, 1, grid_size, device=device, dtype=dtype)
    Y, X = torch.meshgrid(coordinate, coordinate, indexing="ij")
    normalized_radius = torch.sqrt(X.square() + Y.square()) / (2 * pupil_radius_fraction)
    x = X / (2 * pupil_radius_fraction)
    y = Y / (2 * pupil_radius_fraction)
    r2 = x.square() + y.square()
    pupil = normalized_radius <= 1
    candidates = (
        x,
        y,
        2 * r2 - 1,
        x.square() - y.square(),
        2 * x * y,
        (3 * r2 - 2) * x,
        (3 * r2 - 2) * y,
        x * (x.square() - 3 * y.square()),
        y * (3 * x.square() - y.square()),
        6 * r2.square() - 6 * r2 + 1,
    )
    orthonormal: list[torch.Tensor] = []
    mask = pupil.to(dtype)
    for candidate in candidates[:num_modes]:
        mode = candidate * mask
        for previous in orthonormal:
            mode = mode - (mode * previous).sum() / previous.square().sum() * previous
        mode = mode / torch.sqrt(mode[pupil].square().mean())
        orthonormal.append(mode)
    return torch.stack(orthonormal), pupil


def project_phase_to_modes(
    phase: torch.Tensor,
    basis: torch.Tensor,
    pupil: torch.Tensor,
) -> torch.Tensor:
    """把相位投影为离散正交模态系数。"""
    masked_phase = phase * pupil.to(phase.dtype)
    numerator = torch.einsum("...hw,mhw->...m", masked_phase, basis)
    denominator = basis.square().sum(dim=(-2, -1))
    return numerator / denominator


def synthesize_phase(coefficients: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """由模态系数合成二维相位。"""
    if coefficients.shape[-1] != basis.shape[0]:
        raise ValueError("coefficient count does not match basis")
    return torch.einsum("...m,mhw->...hw", coefficients, basis)
