"""S2 低维模态传统控制器。

这些控制器只读取公开观测，不直接访问仿真器内部湍流相位。
当前 S1 观测中的残余模态属于 oracle（真值）观测，因此这些结果是
“控制器本身”的纯仿真基线，不代表真实全息感知性能。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from src.simulation.modes import project_phase_to_modes


class ModalController(ABC):
    """把当前观测转换为 SLM 请求模态增量的统一接口。"""

    name: str

    def __init__(self, num_modes: int, modal_limit_rad: float) -> None:
        if num_modes <= 0:
            raise ValueError("num_modes must be positive")
        if modal_limit_rad <= 0:
            raise ValueError("modal_limit_rad must be positive")
        self.num_modes = num_modes
        self.modal_limit_rad = modal_limit_rad
        self.requested_modal: torch.Tensor | None = None

    def reset(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        """在新的一批完整回合开始前清空控制器状态。"""
        self.requested_modal = torch.zeros(
            batch_size,
            self.num_modes,
            device=device,
            dtype=dtype,
        )

    @abstractmethod
    def action(
        self,
        observation: torch.Tensor,
        wavefront_observation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """返回相对于上一请求命令的模态增量。"""

    def _split_observation(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected = 2 * self.num_modes + 2
        if observation.ndim != 2 or observation.shape[1] != expected:
            raise ValueError(f"observation must have shape [batch, {expected}]")
        residual = observation[:, : self.num_modes]
        applied = observation[:, self.num_modes : 2 * self.num_modes]
        return residual, applied

    def _require_reset(self, observation: torch.Tensor) -> torch.Tensor:
        if self.requested_modal is None:
            raise RuntimeError("controller.reset must be called before action")
        if self.requested_modal.shape[0] != observation.shape[0]:
            raise ValueError("observation batch size changed after controller.reset")
        return self.requested_modal

    def _move_to_target(self, target: torch.Tensor) -> torch.Tensor:
        requested = self._require_reset(target)
        limited_target = target.clamp(-self.modal_limit_rad, self.modal_limit_rad)
        delta = limited_target - requested
        self.requested_modal = limited_target
        return delta


class NoCorrectionController(ModalController):
    """始终保持零相位命令，用作最低参照。"""

    name = "no_correction"

    def action(
        self,
        observation: torch.Tensor,
        wavefront_observation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._split_observation(observation)
        requested = self._require_reset(observation)
        return self._move_to_target(torch.zeros_like(requested))


class DirectProjectionController(ModalController):
    """由当前残余和已加载动作重建扰动，并直接请求其共轭。"""

    name = "direct_projection"

    def action(
        self,
        observation: torch.Tensor,
        wavefront_observation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual, applied = self._split_observation(observation)
        # residual = turbulence + applied，因此 -turbulence = applied - residual。
        target = applied - residual
        return self._move_to_target(target)


class LeakyIntegratorController(ModalController):
    """带泄漏项的积分器：保留旧命令，同时积分当前残余。"""

    name = "leaky_integrator"

    def __init__(
        self,
        num_modes: int,
        modal_limit_rad: float,
        gain: float,
        leak: float,
    ) -> None:
        super().__init__(num_modes, modal_limit_rad)
        if gain < 0:
            raise ValueError("gain must be non-negative")
        if not 0 <= leak <= 1:
            raise ValueError("leak must be between 0 and 1")
        self.gain = gain
        self.leak = leak

    def action(
        self,
        observation: torch.Tensor,
        wavefront_observation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual, _ = self._split_observation(observation)
        requested = self._require_reset(observation)
        target = (1 - self.leak) * requested - self.gain * residual
        return self._move_to_target(target)


class LinearPredictiveController(ModalController):
    """用相邻两帧模态差分外推扰动，再请求预测扰动的共轭。"""

    name = "linear_predictor"

    def __init__(
        self,
        num_modes: int,
        modal_limit_rad: float,
        prediction_horizon: int,
        velocity_gain: float = 1.0,
    ) -> None:
        super().__init__(num_modes, modal_limit_rad)
        if prediction_horizon < 0:
            raise ValueError("prediction_horizon must be non-negative")
        if velocity_gain < 0:
            raise ValueError("velocity_gain must be non-negative")
        self.prediction_horizon = prediction_horizon
        self.velocity_gain = velocity_gain
        self.previous_disturbance: torch.Tensor | None = None

    def reset(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        super().reset(batch_size, device, dtype)
        self.previous_disturbance = None

    def action(
        self,
        observation: torch.Tensor,
        wavefront_observation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual, applied = self._split_observation(observation)
        disturbance = residual - applied
        if self.previous_disturbance is None:
            velocity = torch.zeros_like(disturbance)
        else:
            velocity = disturbance - self.previous_disturbance
        forecast = disturbance + self.velocity_gain * self.prediction_horizon * velocity
        self.previous_disturbance = disturbance.clone()
        return self._move_to_target(-forecast)


class ResUNetModalController(ModalController):
    """把冻结ResUNet的相位预测投影到与传统控制器相同的模态动作。"""

    name = "resunet_static"

    def __init__(
        self,
        num_modes: int,
        modal_limit_rad: float,
        model: nn.Module,
        basis: torch.Tensor,
        pupil: torch.Tensor,
        controller_name: str = "resunet_static",
    ) -> None:
        super().__init__(num_modes, modal_limit_rad)
        if basis.shape[0] != num_modes:
            raise ValueError("basis mode count does not match num_modes")
        if basis.shape[-2:] != pupil.shape:
            raise ValueError("basis and pupil spatial shapes must match")
        self.model = model.eval()
        self.basis = basis
        self.pupil = pupil
        self.name = controller_name
        self.predicted_modal: torch.Tensor | None = None
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def action(
        self,
        observation: torch.Tensor,
        wavefront_observation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, applied = self._split_observation(observation)
        self._require_reset(observation)
        if wavefront_observation is None:
            raise ValueError("ResUNet controller requires a wavefront observation")
        if wavefront_observation.ndim != 4 or wavefront_observation.shape[1] != 2:
            raise ValueError("wavefront observation must have shape [batch, 2, height, width]")
        predicted_phase = self.model(wavefront_observation)
        if predicted_phase.ndim != 4 or predicted_phase.shape[1] != 1:
            raise ValueError("ResUNet output must have shape [batch, 1, height, width]")
        self.predicted_modal = project_phase_to_modes(
            predicted_phase[:, 0],
            self.basis,
            self.pupil,
        )
        # 模型预测的是需要从残余场中减去的相位；SLM相位采用相反符号。
        target = applied - self.predicted_modal
        return self._move_to_target(target)


def make_controller(
    kind: str,
    num_modes: int,
    modal_limit_rad: float,
    **parameters: float | int,
) -> ModalController:
    """由稳定字符串名称创建控制器，供 YAML 和评估入口复用。"""
    normalized = kind.strip().lower()
    if normalized == "no_correction":
        return NoCorrectionController(num_modes, modal_limit_rad)
    if normalized == "direct_projection":
        return DirectProjectionController(num_modes, modal_limit_rad)
    if normalized == "leaky_integrator":
        return LeakyIntegratorController(
            num_modes,
            modal_limit_rad,
            gain=float(parameters["gain"]),
            leak=float(parameters["leak"]),
        )
    if normalized == "linear_predictor":
        return LinearPredictiveController(
            num_modes,
            modal_limit_rad,
            prediction_horizon=int(parameters["prediction_horizon"]),
            velocity_gain=float(parameters.get("velocity_gain", 1.0)),
        )
    raise ValueError(f"unknown controller kind: {kind}")
