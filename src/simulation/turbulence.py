"""Von Karman 初始相位屏与泰勒冻结流时间推进。"""

from __future__ import annotations

import math

import torch


def periodic_fourier_shift(
    field: torch.Tensor,
    shift_x_pixels: float | torch.Tensor,
    shift_y_pixels: float | torch.Tensor,
) -> torch.Tensor:
    """利用傅里叶移位定理做周期边界的亚像素平移。"""
    if field.ndim < 2 or field.shape[-1] != field.shape[-2]:
        raise ValueError("field must end in equal square spatial dimensions")
    height, width = field.shape[-2:]
    real_dtype = field.real.dtype
    fx = torch.fft.fftfreq(width, device=field.device, dtype=real_dtype)
    fy = torch.fft.fftfreq(height, device=field.device, dtype=real_dtype)
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    shift_x = torch.as_tensor(shift_x_pixels, device=field.device, dtype=real_dtype)
    shift_y = torch.as_tensor(shift_y_pixels, device=field.device, dtype=real_dtype)
    spatial_batch_dims = field.ndim - 2
    target_shape = tuple(shift_x.shape) + (1, 1)
    shift_x = shift_x.reshape(target_shape)
    shift_y = shift_y.reshape(tuple(shift_y.shape) + (1, 1))
    if shift_x.ndim > field.ndim or shift_y.ndim > field.ndim:
        raise ValueError("shift tensors have more batch dimensions than field")
    shift_x = shift_x.reshape((1,) * (spatial_batch_dims - shift_x.ndim + 2) + shift_x.shape)
    shift_y = shift_y.reshape((1,) * (spatial_batch_dims - shift_y.ndim + 2) + shift_y.shape)
    phase_ramp = torch.exp(-2j * math.pi * (FX * shift_x + FY * shift_y))
    shifted = torch.fft.ifft2(torch.fft.fft2(field) * phase_ramp)
    return shifted.real if not field.is_complex() else shifted


def advance_taylor_frozen_flow(
    phase: torch.Tensor,
    shift_x_m: float,
    shift_y_m: float,
    sample_pitch_m: float,
    rho: float = 1.0,
    innovation: torch.Tensor | None = None,
) -> torch.Tensor:
    """推进一帧冻结流；rho<1 时加入独立的“沸腾”相位。"""
    if sample_pitch_m <= 0:
        raise ValueError("sample_pitch_m must be positive")
    if not 0 <= rho <= 1:
        raise ValueError("rho must be between 0 and 1")
    translated = periodic_fourier_shift(
        phase,
        shift_x_pixels=shift_x_m / sample_pitch_m,
        shift_y_pixels=shift_y_m / sample_pitch_m,
    )
    if rho < 1 and innovation is None:
        raise ValueError("innovation is required when rho < 1")
    if innovation is None:
        advanced = translated
    else:
        if innovation.shape != phase.shape:
            raise ValueError("innovation must have the same shape as phase")
        advanced = rho * translated + math.sqrt(1 - rho**2) * innovation
    return advanced - advanced.mean(dim=(-2, -1), keepdim=True)


def von_karman_phase_screens(
    batch_size: int,
    grid_size: int,
    screen_size_m: float,
    r0_m: float,
    outer_scale_m: float,
    inner_scale_m: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """生成一批周期 Von Karman 相位屏，单位为弧度。"""
    if batch_size <= 0 or grid_size <= 0:
        raise ValueError("batch_size and grid_size must be positive")
    delta_f = 1.0 / screen_size_m
    frequency = (torch.arange(grid_size, device=device, dtype=dtype) - grid_size / 2) * delta_f
    FY, FX = torch.meshgrid(frequency, frequency, indexing="ij")
    radial_frequency = torch.sqrt(FX.square() + FY.square())
    inner_frequency = 5.92 / (2 * math.pi * inner_scale_m)
    outer_frequency = 1.0 / outer_scale_m
    spectrum = (
        0.023
        * r0_m ** (-5.0 / 3.0)
        * torch.exp(-(radial_frequency / inner_frequency).square())
        / (radial_frequency.square() + outer_frequency**2) ** (11.0 / 6.0)
    )
    spectrum[grid_size // 2, grid_size // 2] = 0
    real = torch.randn(
        batch_size, grid_size, grid_size, generator=generator, device=device, dtype=dtype
    )
    imag = torch.randn(
        batch_size, grid_size, grid_size, generator=generator, device=device, dtype=dtype
    )
    coefficients = torch.complex(real, imag) * torch.sqrt(spectrum) * delta_f
    phase = torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(coefficients, dim=(-2, -1))),
        dim=(-2, -1),
    ).real * grid_size**2
    return phase - phase.mean(dim=(-2, -1), keepdim=True)


def wind_displacement(speed_mps: float, direction_deg: float, dt_s: float) -> tuple[float, float]:
    """把风速、方向和帧间隔换算为横纵位移。"""
    angle = math.radians(direction_deg)
    return speed_mps * dt_s * math.cos(angle), speed_mps * dt_s * math.sin(angle)


def phase_structure_function(
    phase: torch.Tensor,
    sample_pitch_m: float,
    max_lag: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """估计一批相位屏的水平/垂直二阶结构函数。"""
    if max_lag <= 0 or max_lag >= min(phase.shape[-2:]):
        raise ValueError("max_lag must be positive and smaller than the spatial grid")
    values = []
    for lag in range(1, max_lag + 1):
        horizontal = phase[..., :, lag:] - phase[..., :, :-lag]
        vertical = phase[..., lag:, :] - phase[..., :-lag, :]
        values.append(0.5 * (horizontal.square().mean() + vertical.square().mean()))
    separation = torch.arange(1, max_lag + 1, device=phase.device, dtype=phase.dtype) * sample_pitch_m
    return separation, torch.stack(values)
