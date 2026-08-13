"""S1 动态环境的配置契约。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class S1EnvConfig:
    """泰勒冻结流动态环境的第一版参数。"""

    grid_size: int = 64
    turbulence_grid_multiplier: int = 4
    screen_size_m: float = 0.4
    wavelength_m: float = 1550e-9
    pupil_radius_fraction: float = 0.4
    r0_m: float = 0.12
    outer_scale_m: float = 10.0
    inner_scale_m: float = 0.01
    dt_s: float = 0.01
    episode_length: int = 200
    wind_speed_mps: float = 0.5
    wind_direction_deg: float = 30.0
    frozen_flow_rho: float = 1.0
    wind_speed_modulation_fraction: float = 0.0
    wind_direction_modulation_deg: float = 0.0
    wind_modulation_period_frames: int = 0
    wind_modulation_phase_deg: float = 0.0
    num_modes: int = 10
    modal_limit_rad: float = 3.0
    slm_phase_min_rad: float = -3.141592653589793
    slm_phase_max_rad: float = 3.141592653589793
    slm_max_delta_rad: float = 0.35
    slm_quantization_levels: int = 256
    slm_delay_frames: int = 1
    bucket_radius_pixels: float = 2.0
    reward_phase_weight: float = 0.05
    reward_action_weight: float = 0.01
    reward_violation_weight: float = 0.5
    batch_size: int = 32
    seed: int = 42

    @property
    def sample_pitch_m(self) -> float:
        return self.screen_size_m / self.grid_size

    @property
    def observation_size(self) -> int:
        return 2 * self.num_modes + 2

    @property
    def turbulence_grid_size(self) -> int:
        return self.grid_size * self.turbulence_grid_multiplier

    def validate(self) -> None:
        """尽早拒绝会破坏数值或物理含义的参数。"""
        if self.grid_size < 8 or self.grid_size % 2:
            raise ValueError("grid_size must be an even integer >= 8")
        if not isinstance(self.turbulence_grid_multiplier, int) or self.turbulence_grid_multiplier < 1:
            raise ValueError("turbulence_grid_multiplier must be a positive integer")
        if not 0 < self.pupil_radius_fraction <= 0.5:
            raise ValueError("pupil_radius_fraction must be in (0, 0.5]")
        if min(
            self.screen_size_m,
            self.wavelength_m,
            self.r0_m,
            self.outer_scale_m,
            self.inner_scale_m,
        ) <= 0:
            raise ValueError("optical lengths and turbulence scales must be positive")
        if self.dt_s <= 0 or self.episode_length <= 0 or self.wind_speed_mps < 0:
            raise ValueError("dt_s and episode_length must be positive")
        if not 0 <= self.frozen_flow_rho <= 1:
            raise ValueError("frozen_flow_rho must be between 0 and 1")
        if not 0 <= self.wind_speed_modulation_fraction <= 1:
            raise ValueError("wind_speed_modulation_fraction must be between 0 and 1")
        if self.wind_direction_modulation_deg < 0:
            raise ValueError("wind_direction_modulation_deg must be non-negative")
        modulation_enabled = (
            self.wind_speed_modulation_fraction > 0
            or self.wind_direction_modulation_deg > 0
        )
        if modulation_enabled and self.wind_modulation_period_frames < 2:
            raise ValueError(
                "wind_modulation_period_frames must be at least 2 when modulation is enabled"
            )
        if not modulation_enabled and self.wind_modulation_period_frames < 0:
            raise ValueError("wind_modulation_period_frames must be non-negative")
        if not 1 <= self.num_modes <= 256:
            raise ValueError("the simulation action size must be between 1 and 256")
        if self.slm_phase_max_rad <= self.slm_phase_min_rad:
            raise ValueError("SLM maximum phase must exceed minimum phase")
        if (
            self.slm_delay_frames < 0
            or self.slm_quantization_levels < 0
            or self.slm_max_delta_rad < 0
        ):
            raise ValueError("SLM delay, quantization, and phase slew limits must be non-negative")
        if self.bucket_radius_pixels < 0 or min(
            self.reward_phase_weight,
            self.reward_action_weight,
            self.reward_violation_weight,
        ) < 0:
            raise ValueError("metric radius and reward penalty weights must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        maximum_wind_speed = self.wind_speed_mps * (
            1 + self.wind_speed_modulation_fraction
        )
        episode_displacement_pixels = (
            maximum_wind_speed * self.dt_s * self.episode_length / self.sample_pitch_m
        )
        if episode_displacement_pixels >= self.turbulence_grid_size:
            raise ValueError(
                "the frozen-flow screen would wrap completely within one episode; "
                "increase turbulence_grid_multiplier or shorten the episode"
            )

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "S1EnvConfig":
        """从 YAML 的 environment 小节构造配置。"""
        config = cls(**values)
        config.validate()
        return config


def load_s1_config(path: str | Path) -> tuple[S1EnvConfig, str]:
    """读取 S1 YAML，并返回环境参数和显式设备名称。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = S1EnvConfig.from_mapping(raw["environment"])
    device = str(raw.get("runtime", {}).get("device", "cuda"))
    return config, device
