"""S1 瞳面相位与焦面质量指标。"""

from __future__ import annotations

import torch


def focal_plane_metrics(
    residual_phase: torch.Tensor,
    pupil: torch.Tensor,
    bucket_radius_pixels: float,
) -> dict[str, torch.Tensor]:
    """计算 Strehl、桶内功率、去 piston 相位 RMSE 和能量误差。"""
    if residual_phase.shape[-2:] != pupil.shape:
        raise ValueError("residual_phase and pupil spatial shapes must match")
    pupil_float = pupil.to(residual_phase.dtype)
    field = pupil_float * torch.exp(1j * residual_phase)
    focal_field = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )
    intensity = focal_field.abs().square()
    ideal_field = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(pupil_float, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )
    ideal_intensity = ideal_field.abs().square()
    total = intensity.sum(dim=(-2, -1))
    ideal_total = ideal_intensity.sum()
    normalized_peak = intensity.amax(dim=(-2, -1)) / total
    ideal_peak = ideal_intensity.max() / ideal_total

    grid_size = pupil.shape[0]
    pixel = torch.arange(grid_size, device=pupil.device, dtype=residual_phase.dtype) - grid_size // 2
    PY, PX = torch.meshgrid(pixel, pixel, indexing="ij")
    bucket = torch.sqrt(PX.square() + PY.square()) <= bucket_radius_pixels

    wrapped = torch.atan2(torch.sin(residual_phase), torch.cos(residual_phase))
    phasor = torch.where(pupil, torch.exp(1j * wrapped), torch.zeros_like(field))
    piston = torch.angle(phasor.sum(dim=(-2, -1), keepdim=True))
    piston_removed = torch.atan2(
        torch.sin(wrapped - piston),
        torch.cos(wrapped - piston),
    )
    pupil_pixels = pupil_float.sum()
    phase_rmse = torch.sqrt((piston_removed.square() * pupil_float).sum(dim=(-2, -1)) / pupil_pixels)
    input_energy = field.abs().square().sum(dim=(-2, -1))

    return {
        "strehl": normalized_peak / ideal_peak,
        "power_in_bucket": intensity[..., bucket].sum(dim=-1) / total,
        "phase_rmse": phase_rmse,
        "input_energy": input_energy,
        "focal_energy": total,
        "relative_energy_error": (total - input_energy).abs() / input_energy,
    }
