"""S4-C纯仿真的可配置SLM与测量误差。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any

import torch
from torch.nn import functional as functional

from src.simulation.config import S1EnvConfig
from src.simulation.slm import TorchSlmModel


@dataclass(frozen=True)
class HardwareEffectsConfig:
    """尚未用H1实测确认的硬件误差；默认值严格等于无额外误差。"""

    phase_scale: float = 1.0
    settling_fraction: float = 1.0
    shift_x_pixels: float = 0.0
    shift_y_pixels: float = 0.0
    rotation_deg: float = 0.0
    power_noise_relative_std: float = 0.0

    def validate(self) -> None:
        values = asdict(self)
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError("hardware-effect values must be finite")
        if self.phase_scale <= 0:
            raise ValueError("phase_scale must be positive")
        if not 0 < self.settling_fraction <= 1:
            raise ValueError("settling_fraction must be in (0, 1]")
        if abs(self.rotation_deg) > 180:
            raise ValueError("rotation_deg must be within [-180, 180]")
        if self.power_noise_relative_std < 0:
            raise ValueError("power_noise_relative_std must be non-negative")

    @property
    def is_nominal(self) -> bool:
        return self == HardwareEffectsConfig()


@dataclass(frozen=True)
class HardwareProfile:
    """一个S4-C硬件压力档位，不包含湍流物理条件。"""

    identifier: str
    label: str
    severity: str
    required_for_gate: bool
    slm_delay_frames: int = 2
    slm_quantization_levels: int = 256
    slm_max_delta_rad: float = 0.35
    observation_noise_std_rad: float = 0.02
    phase_scale: float = 1.0
    settling_fraction: float = 1.0
    shift_x_pixels: float = 0.0
    shift_y_pixels: float = 0.0
    rotation_deg: float = 0.0
    power_noise_relative_std: float = 0.0

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "HardwareProfile":
        profile = cls(
            identifier=str(values["id"]),
            label=str(values.get("label", values["id"])),
            severity=str(values.get("severity", "single")),
            required_for_gate=bool(values.get("required_for_gate", True)),
            slm_delay_frames=int(values.get("slm_delay_frames", 2)),
            slm_quantization_levels=int(values.get("slm_quantization_levels", 256)),
            slm_max_delta_rad=float(values.get("slm_max_delta_rad", 0.35)),
            observation_noise_std_rad=float(values.get("observation_noise_std_rad", 0.02)),
            phase_scale=float(values.get("phase_scale", 1.0)),
            settling_fraction=float(values.get("settling_fraction", 1.0)),
            shift_x_pixels=float(values.get("shift_x_pixels", 0.0)),
            shift_y_pixels=float(values.get("shift_y_pixels", 0.0)),
            rotation_deg=float(values.get("rotation_deg", 0.0)),
            power_noise_relative_std=float(
                values.get("power_noise_relative_std", 0.0)
            ),
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("hardware profile id must not be empty")
        if self.slm_delay_frames < 0:
            raise ValueError("slm_delay_frames must be non-negative")
        if self.slm_quantization_levels < 0:
            raise ValueError("slm_quantization_levels must be non-negative")
        if self.slm_max_delta_rad < 0:
            raise ValueError("slm_max_delta_rad must be non-negative")
        if self.observation_noise_std_rad < 0:
            raise ValueError("observation_noise_std_rad must be non-negative")
        self.effects_config().validate()

    def environment_config(self, base: S1EnvConfig) -> S1EnvConfig:
        configured = replace(
            base,
            slm_delay_frames=self.slm_delay_frames,
            slm_quantization_levels=self.slm_quantization_levels,
            slm_max_delta_rad=self.slm_max_delta_rad,
        )
        configured.validate()
        return configured

    def effects_config(self) -> HardwareEffectsConfig:
        return HardwareEffectsConfig(
            phase_scale=self.phase_scale,
            settling_fraction=self.settling_fraction,
            shift_x_pixels=self.shift_x_pixels,
            shift_y_pixels=self.shift_y_pixels,
            rotation_deg=self.rotation_deg,
            power_noise_relative_std=self.power_noise_relative_std,
        )

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


class HardwareAwareSlmModel(TorchSlmModel):
    """在原SLM约束链上加入LUT比例、稳定时间和配准误差。"""

    def __init__(
        self,
        phase_min: float,
        phase_max: float,
        max_delta: float,
        quantization_levels: int,
        delay_frames: int,
        effects: HardwareEffectsConfig,
    ) -> None:
        super().__init__(
            phase_min,
            phase_max,
            max_delta,
            quantization_levels,
            delay_frames,
        )
        effects.validate()
        self.effects = effects

    def step(
        self, requested_phase: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.current_phase is None or self.command_queue is None:
            raise RuntimeError("reset must be called before step")
        if requested_phase.shape != self.current_phase.shape:
            raise ValueError("requested_phase shape does not match SLM state")

        scaled = requested_phase * self.effects.phase_scale
        clipped = scaled.clamp(self.phase_min, self.phase_max)
        saturated = clipped.ne(scaled)
        quantized = self._quantize(clipped)
        delayed = self._apply_delay(quantized)
        registered = apply_registration_error(
            delayed,
            shift_x_pixels=self.effects.shift_x_pixels,
            shift_y_pixels=self.effects.shift_y_pixels,
            rotation_deg=self.effects.rotation_deg,
        )
        requested_delta = registered - self.current_phase
        limited_delta = requested_delta.clamp(-self.max_delta, self.max_delta)
        actual_delta = limited_delta * self.effects.settling_fraction
        self.current_phase = self.current_phase + actual_delta

        nonzero_delta = limited_delta.ne(0)
        settling_limited = nonzero_delta & actual_delta.ne(limited_delta)
        diagnostics = {
            "saturated_fraction": saturated.float().mean(dim=(-2, -1)),
            "slew_limited_fraction": limited_delta.ne(requested_delta)
            .float()
            .mean(dim=(-2, -1)),
            "settling_limited_fraction": settling_limited.float().mean(dim=(-2, -1)),
            "delayed_command": delayed,
            "registered_command": registered,
        }
        return self.current_phase, diagnostics


def apply_registration_error(
    phase: torch.Tensor,
    *,
    shift_x_pixels: float,
    shift_y_pixels: float,
    rotation_deg: float,
) -> torch.Tensor:
    """把SLM图案相对光瞳平移和旋转，边界外使用零相位。"""
    if phase.ndim != 3:
        raise ValueError("phase must have shape [batch, height, width]")
    if shift_x_pixels == 0 and shift_y_pixels == 0 and rotation_deg == 0:
        return phase
    _, height, width = phase.shape
    radians = math.radians(rotation_deg)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    transform = phase.new_tensor(
        [
            [cosine, -sine, -2 * shift_x_pixels / width],
            [sine, cosine, -2 * shift_y_pixels / height],
        ]
    ).unsqueeze(0)
    transform = transform.expand(phase.shape[0], -1, -1)
    grid = functional.affine_grid(
        transform,
        size=(phase.shape[0], 1, height, width),
        align_corners=False,
    )
    return functional.grid_sample(
        phase.unsqueeze(1),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(1)


def noisy_power_measurement(
    true_power: torch.Tensor,
    relative_std: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """生成非负的乘性功率计读数；科学指标仍保留无噪真值。"""
    if relative_std < 0:
        raise ValueError("relative_std must be non-negative")
    if relative_std == 0:
        return true_power
    noise = torch.randn(
        true_power.shape,
        generator=generator,
        device=true_power.device,
        dtype=true_power.dtype,
    )
    return (true_power * (1 + relative_std * noise)).clamp_min(0)
