"""复光场重建、相位补偿及训练损失函数。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def complex_field_from_intensity_phase(
    intensity: torch.Tensor,
    phase: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """根据光强和相位计算复光场的实部与虚部。"""
    # 光场振幅等于光强的平方根；截断负值可避免数值误差产生 NaN。
    amplitude = torch.sqrt(torch.clamp(intensity, min=0.0))
    return amplitude * torch.cos(phase), amplitude * torch.sin(phase)


def compensate_field(
    input_intensity: torch.Tensor,
    input_phase: torch.Tensor,
    phi_corr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从输入相位中减去预测校正量，得到补偿后的复光场。"""
    compensated_phase = input_phase - phi_corr
    return complex_field_from_intensity_phase(input_intensity, compensated_phase)


def phase_smoothness_loss(phi_corr: torch.Tensor) -> torch.Tensor:
    """计算相位图在水平和垂直方向的一阶总变差。"""
    dx = torch.abs(phi_corr[..., :, 1:] - phi_corr[..., :, :-1]).mean()
    dy = torch.abs(phi_corr[..., 1:, :] - phi_corr[..., :-1, :]).mean()
    return dx + dy


def compensation_loss(
    phi_corr: torch.Tensor,
    input_intensity: torch.Tensor,
    input_phase: torch.Tensor,
    target_intensity: torch.Tensor,
    target_phase: torch.Tensor,
    intensity_weight: float = 0.2,
    smooth_weight: float = 0.01,
) -> dict[str, torch.Tensor]:
    """组合复光场、光强和相位平滑三项损失。"""
    # 分别将补偿结果与干净目标转换到复数域进行监督。
    comp_real, comp_imag = compensate_field(input_intensity, input_phase, phi_corr)
    target_real, target_imag = complex_field_from_intensity_phase(target_intensity, target_phase)

    # 复光场误差约束幅度与相位，光强项和光滑项提供辅助正则化。
    complex_loss = F.l1_loss(comp_real, target_real) + F.l1_loss(comp_imag, target_imag)
    comp_intensity = comp_real.square() + comp_imag.square()
    intensity_loss = F.l1_loss(comp_intensity, target_intensity)
    smooth_loss = phase_smoothness_loss(phi_corr)
    total = complex_loss + intensity_weight * intensity_loss + smooth_weight * smooth_loss

    return {
        "total": total,
        "complex": complex_loss,
        "intensity": intensity_loss,
        "smooth": smooth_loss,
    }
