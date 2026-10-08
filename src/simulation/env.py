"""不依赖 Gym 的 S1 GPU 动态闭环环境。"""

from __future__ import annotations

import math
import torch

from src.simulation.config import S1EnvConfig
from src.simulation.hardware_effects import (
    HardwareAwareSlmModel,
    HardwareEffectsConfig,
    noisy_power_measurement,
)
from src.simulation.modes import (
    make_low_order_zernike_basis,
    make_pupil_mask,
    project_phase_to_modes,
    synthesize_phase,
)
from src.simulation.optics import focal_plane_metrics
from src.simulation.slm import TorchSlmModel
from src.simulation.turbulence import (
    advance_taylor_frozen_flow,
    modulated_wind_parameters,
    von_karman_phase_screens,
    wind_displacement,
)


class AdaptiveOpticsEnv:
    """泰勒冻结流、低维动作和焦面奖励组成的批量环境。"""

    def __init__(
        self,
        config: S1EnvConfig,
        device: torch.device | str,
        hardware_effects: HardwareEffectsConfig | None = None,
        basis_override: torch.Tensor | None = None,
        condition_batch: list[dict] | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.device = torch.device(device)
        self.dtype = torch.float32
        if basis_override is None:
            self.basis, self.pupil = make_low_order_zernike_basis(
                config.grid_size,
                config.pupil_radius_fraction,
                config.num_modes,
                self.device,
                self.dtype,
            )
        else:
            expected = (config.num_modes, config.grid_size, config.grid_size)
            if basis_override.shape != expected:
                raise ValueError(f"basis_override must have shape {expected}")
            self.basis = basis_override.to(self.device, self.dtype).clone()
            self.pupil = make_pupil_mask(
                config.grid_size,
                config.pupil_radius_fraction,
                self.device,
                self.dtype,
            )
            if not torch.isfinite(self.basis).all():
                raise ValueError("basis_override must contain only finite values")
            outside = self.basis[:, ~self.pupil]
            if outside.numel() and float(outside.abs().max()) > 1e-6:
                raise ValueError("basis_override must be zero outside the pupil")
        self.hardware_effects = hardware_effects or HardwareEffectsConfig()
        self.hardware_effects.validate()
        slm_arguments = (
            config.slm_phase_min_rad,
            config.slm_phase_max_rad,
            config.slm_max_delta_rad,
            config.slm_quantization_levels,
            config.slm_delay_frames,
        )
        if self.hardware_effects.is_nominal:
            self.slm = TorchSlmModel(*slm_arguments)
        else:
            self.slm = HardwareAwareSlmModel(
                *slm_arguments,
                effects=self.hardware_effects,
            )
        self.generators: list[torch.Generator] = []
        self.episode_seeds: torch.Tensor | None = None
        self.turbulence_phase: torch.Tensor | None = None
        self.requested_modal: torch.Tensor | None = None
        self.measurement_generator: torch.Generator | None = None
        self.step_count = 0
        ideal_field = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(self.pupil, dim=(-2, -1)), norm="ortho"),
            dim=(-2, -1),
        )
        self._ideal_intensity = ideal_field.abs().square()
        pixel = torch.arange(config.grid_size, device=self.device, dtype=self.dtype) - config.grid_size // 2
        py, px = torch.meshgrid(pixel, pixel, indexing="ij")
        self._bucket_mask = torch.sqrt(px.square() + py.square()) <= config.bucket_radius_pixels
        if condition_batch is not None:
            if len(condition_batch) != config.batch_size:
                raise ValueError("condition_batch length must equal config.batch_size")
            self.condition_batch = tuple(dict(item) for item in condition_batch)
        else:
            self.condition_batch = None

    def reset(
        self,
        seed: int | None = None,
        turbulence_phase: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """开始一组独立回合，并返回首个观测。"""
        actual_seed = self.config.seed if seed is None else seed
        seeds = [actual_seed + index for index in range(self.config.batch_size)]
        self.generators = [
            torch.Generator(device=self.device).manual_seed(one_seed)
            for one_seed in seeds
        ]
        self.episode_seeds = torch.tensor(seeds, device=self.device, dtype=torch.int64)
        self.measurement_generator = torch.Generator(device=self.device).manual_seed(
            actual_seed + 60_000_000
        )
        if turbulence_phase is None:
            self.turbulence_phase = self._new_phase_screens()
        else:
            expected_shape = (
                self.config.batch_size,
                self.config.turbulence_grid_size,
                self.config.turbulence_grid_size,
            )
            if turbulence_phase.shape != expected_shape:
                raise ValueError(f"turbulence_phase must have shape {expected_shape}")
            self.turbulence_phase = turbulence_phase.to(self.device, self.dtype).clone()
            self.turbulence_phase -= self.turbulence_phase.mean(dim=(-2, -1), keepdim=True)
        self.requested_modal = torch.zeros(
            self.config.batch_size,
            self.config.num_modes,
            device=self.device,
            dtype=self.dtype,
        )
        self.slm.reset(
            (self.config.batch_size, self.config.grid_size, self.config.grid_size),
            self.device,
            self.dtype,
        )
        self.step_count = 0
        return self._observation_and_info()

    def step(
        self,
        action_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """执行动作、计算当前奖励、推进湍流并返回下一观测。"""
        self._require_reset()
        expected = (self.config.batch_size, self.config.num_modes)
        if action_delta.shape != expected:
            raise ValueError(f"action_delta must have shape {expected}")
        action_delta = action_delta.to(self.device, self.dtype)
        unconstrained_modal = self.requested_modal + action_delta
        self.requested_modal = unconstrained_modal.clamp(
            -self.config.modal_limit_rad,
            self.config.modal_limit_rad,
        )
        modal_violation = self.requested_modal.ne(unconstrained_modal).float().mean(dim=-1)
        requested_phase = synthesize_phase(self.requested_modal, self.basis)
        applied_phase, slm_info = self.slm.step(requested_phase)
        current_metrics = self._metrics(applied_phase)
        if self.measurement_generator is None:
            raise RuntimeError("measurement generator is unavailable")
        measured_power = noisy_power_measurement(
            current_metrics["power_in_bucket"],
            self.hardware_effects.power_noise_relative_std,
            self.measurement_generator,
        )
        action_cost = action_delta.square().mean(dim=-1)
        violation = torch.maximum(
            modal_violation,
            torch.maximum(slm_info["saturated_fraction"], slm_info["slew_limited_fraction"]),
        )
        reward = (
            current_metrics["strehl"]
            - self.config.reward_phase_weight * current_metrics["phase_rmse"]
            - self.config.reward_action_weight * action_cost
            - self.config.reward_violation_weight * violation
        )

        self._advance_turbulence()
        self.step_count += 1
        observation, next_info = self._observation_and_info()
        terminated = torch.full(
            (self.config.batch_size,),
            self.step_count >= self.config.episode_length,
            device=self.device,
            dtype=torch.bool,
        )
        truncated = torch.zeros_like(terminated)
        info = {
            **next_info,
            "reward_strehl": current_metrics["strehl"],
            "reward_power_in_bucket": current_metrics["power_in_bucket"],
            "measured_power_in_bucket": measured_power,
            "reward_phase_rmse": current_metrics["phase_rmse"],
            "action_cost": action_cost,
            "violation_fraction": violation,
            "requested_modal": self.requested_modal.clone(),
            "applied_modal": project_phase_to_modes(applied_phase, self.basis, self.pupil),
            "delayed_modal": project_phase_to_modes(
                slm_info["delayed_command"], self.basis, self.pupil
            ),
            "registered_modal": project_phase_to_modes(
                slm_info.get("registered_command", slm_info["delayed_command"]),
                self.basis,
                self.pupil,
            ),
            "saturated_fraction": slm_info["saturated_fraction"],
            "slew_limited_fraction": slm_info["slew_limited_fraction"],
            "settling_limited_fraction": slm_info.get(
                "settling_limited_fraction",
                torch.zeros_like(slm_info["slew_limited_fraction"]),
            ),
        }
        return observation, reward, terminated, truncated, info

    def oracle_modal_upper_bound(self) -> dict[str, torch.Tensor]:
        """返回忽略 SLM 约束和时延的瞬时模态子空间理想上限。

        该接口会读取仿真真值，只允许用于诊断和论文上限，不得作为可部署
        控制器的在线输入。
        """
        self._require_reset()
        turbulence = self._current_turbulence_window()
        coefficients = project_phase_to_modes(turbulence, self.basis, self.pupil)
        ideal_correction = synthesize_phase(-coefficients, self.basis)
        return focal_plane_metrics(
            turbulence + ideal_correction,
            self.pupil,
            self.config.bucket_radius_pixels,
            ideal_intensity=self._ideal_intensity,
            bucket_mask=self._bucket_mask,
        )

    def oracle_disturbance_modal(self) -> torch.Tensor:
        """直接返回当前湍流的模态真值，仅用于不可部署的仿真诊断。

        不能用“残余模态减已施加模态”代替本接口：高维动作较大时，两个
        float32 数值相减会产生足以触发严格真值对齐门槛的消差误差。
        """
        self._require_reset()
        return project_phase_to_modes(
            self._current_turbulence_window(),
            self.basis,
            self.pupil,
        )

    def ideal_wavefront_observation(self) -> torch.Tensor:
        """返回无噪声瞳面强度与包裹相位，作为S2统一观测源。

        输出形状为 ``[batch, 2, height, width]``。第一通道是归一化
        强度，第二通道是瞳面内包裹到 ``[-pi, pi]`` 的残余相位。
        它是纯仿真理想传感器，不等同于传播后的接收面全息复场。
        """
        self._require_reset()
        if self.slm.current_phase is None:
            raise RuntimeError("SLM state is unavailable")
        residual_phase = self._current_turbulence_window() + self.slm.current_phase
        wrapped_phase = torch.atan2(torch.sin(residual_phase), torch.cos(residual_phase))
        pupil = self.pupil.to(self.dtype).expand(self.config.batch_size, -1, -1)
        wrapped_phase = wrapped_phase * pupil
        return torch.stack((pupil, wrapped_phase), dim=1)

    def _new_phase_screens(self) -> torch.Tensor:
        if not self.generators:
            raise RuntimeError("reset must initialize episode generators")
        return torch.cat(
            [
                von_karman_phase_screens(
                    1,
                    self.config.turbulence_grid_size,
                    self.config.screen_size_m * self.config.turbulence_grid_multiplier,
                    self.config.r0_m,
                    self.config.outer_scale_m,
                    self.config.inner_scale_m,
                    generator,
                    self.device,
                    self.dtype,
                )
                for generator in self.generators
            ],
            dim=0,
        )

    def _advance_turbulence(self) -> None:
        self._require_reset()
        if self.condition_batch is not None:
            values = self.condition_batch
            base_speed = torch.tensor([float(v["wind_speed_mps"]) for v in values], device=self.device, dtype=self.dtype)
            base_direction = torch.tensor([float(v["wind_direction_deg"]) for v in values], device=self.device, dtype=self.dtype)
            speed_fraction = torch.tensor([float(v.get("wind_speed_modulation_fraction", 0.0)) for v in values], device=self.device, dtype=self.dtype)
            direction_mod = torch.tensor([float(v.get("wind_direction_modulation_deg", 0.0)) for v in values], device=self.device, dtype=self.dtype)
            periods = torch.tensor([int(v.get("wind_modulation_period_frames", 0)) for v in values], device=self.device, dtype=self.dtype)
            phases = torch.tensor([float(v.get("wind_modulation_phase_deg", 0.0)) for v in values], device=self.device, dtype=self.dtype)
            enabled = (speed_fraction > 0) | (direction_mod > 0)
            angle = 2 * math.pi * self.step_count / periods.clamp_min(1) + torch.deg2rad(phases)
            wind_speed_mps = torch.where(enabled, base_speed * (1 + speed_fraction * torch.sin(angle)), base_speed)
            wind_direction_deg = torch.where(enabled, base_direction + direction_mod * torch.cos(angle), base_direction)
            radians = torch.deg2rad(wind_direction_deg)
            shift_x = wind_speed_mps * self.config.dt_s * torch.cos(radians) / self.config.sample_pitch_m
            shift_y = wind_speed_mps * self.config.dt_s * torch.sin(radians) / self.config.sample_pitch_m
            rho = torch.tensor([float(v.get("frozen_flow_rho", 1.0)) for v in values], device=self.device, dtype=self.dtype)
            innovation = self._new_phase_screens() if bool((rho < 1).any()) else None
            self.turbulence_phase = advance_taylor_frozen_flow(self.turbulence_phase, shift_x, shift_y, 1.0, rho, innovation)
            return
        wind_speed_mps, wind_direction_deg = modulated_wind_parameters(
            self.config.wind_speed_mps,
            self.config.wind_direction_deg,
            self.step_count,
            speed_modulation_fraction=self.config.wind_speed_modulation_fraction,
            direction_modulation_deg=self.config.wind_direction_modulation_deg,
            period_frames=self.config.wind_modulation_period_frames,
            phase_deg=self.config.wind_modulation_phase_deg,
        )
        shift_x, shift_y = wind_displacement(
            wind_speed_mps,
            wind_direction_deg,
            self.config.dt_s,
        )
        innovation = None
        if self.config.frozen_flow_rho < 1:
            innovation = self._new_phase_screens()
        self.turbulence_phase = advance_taylor_frozen_flow(
            self.turbulence_phase,
            shift_x,
            shift_y,
            self.config.sample_pitch_m,
            self.config.frozen_flow_rho,
            innovation,
        )

    def _metrics(self, applied_phase: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        self._require_reset()
        actual_phase = self.slm.current_phase if applied_phase is None else applied_phase
        if actual_phase is None:
            raise RuntimeError("SLM state is unavailable")
        residual_phase = self._current_turbulence_window() + actual_phase
        return focal_plane_metrics(
            residual_phase,
            self.pupil,
            self.config.bucket_radius_pixels,
            ideal_intensity=self._ideal_intensity,
            bucket_mask=self._bucket_mask,
        )

    def _observation_and_info(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._require_reset()
        metrics = self._metrics()
        if self.slm.current_phase is None:
            raise RuntimeError("SLM state is unavailable")
        residual_phase = self._current_turbulence_window() + self.slm.current_phase
        residual_modal = project_phase_to_modes(residual_phase, self.basis, self.pupil)
        applied_modal = project_phase_to_modes(self.slm.current_phase, self.basis, self.pupil)
        observation = torch.cat(
            (
                residual_modal,
                applied_modal,
                metrics["strehl"].unsqueeze(-1),
                metrics["power_in_bucket"].unsqueeze(-1),
            ),
            dim=-1,
        )
        return observation, metrics

    def _current_turbulence_window(self) -> torch.Tensor:
        """从更大的移动相位屏中央截取控制器实际看到的窗口。"""
        self._require_reset()
        start = (self.config.turbulence_grid_size - self.config.grid_size) // 2
        stop = start + self.config.grid_size
        return self.turbulence_phase[..., start:stop, start:stop]

    def _require_reset(self) -> None:
        if self.turbulence_phase is None or self.requested_modal is None:
            raise RuntimeError("reset must be called before using the environment")
