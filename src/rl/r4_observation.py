"""R4白名单观测与带动作时间索引的转移；不接受任意info字典。"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.rl.r4_control import (
    CausalActuatorEstimate, NominalCalibration, R4Limits, SafeRequest,
    finite_tensor, project_request,
)

FEATURE_FIELDS = {
    "measured_residual_rad": (0, 21), "issued_modal_rad": (21, 42),
    "estimated_applied_modal_rad": (42, 63), "previous_correction_normalized": (63, 74),
    "arrived_power": (74, 75), "observation_step": (75, 76),
    "last_command_step": (76, 77), "power_action_step": (77, 78), "power_valid": (78, 79),
}
FEATURE_SIZE = 79
SCHEMA = "r4_causal_transition_v1"


def simulation_residual_proxy(observation: torch.Tensor, *, generator: torch.Generator,
                              noise_std_rad: float) -> torch.Tensor:
    """只取残余21维并加显式种子噪声；这不是实际全息重建数据。"""
    if observation.ndim != 2 or observation.shape[1] != 44:
        raise ValueError("expected 21-mode simulator observation")
    if not 0 <= noise_std_rad < float("inf"):
        raise ValueError("invalid observation noise")
    residual = observation[:, :21].clone()
    finite_tensor(residual, residual.shape, "residual proxy")
    return residual + noise_std_rad * torch.randn(
        residual.shape, generator=generator, device=residual.device, dtype=residual.dtype)


@dataclass(frozen=True)
class PowerMeasurement:
    value: torch.Tensor
    action_step: int
    arrival_observation_step: int


@dataclass(frozen=True)
class HistoryView:
    features: torch.Tensor
    valid: torch.Tensor
    episode_id: str
    observation_step: int


@dataclass(frozen=True)
class CausalTransition:
    history: HistoryView
    action: SafeRequest
    next_history: HistoryView
    next_residual: torch.Tensor
    action_power: torch.Tensor
    action_power_valid: torch.Tensor
    action_step: int
    next_observation_step: int


class R4Interface:
    """同步观测步接口；功率允许迟到，迟到值不能冒充当前动作标签。"""

    def __init__(self, *, limits: R4Limits = R4Limits(), history_frames: int = 8,
                 calibration: NominalCalibration = NominalCalibration()):
        limits.validate()
        if type(history_frames) is not int or history_frames < 1:
            raise ValueError("invalid history length")
        self.limits = limits
        self.history_frames = history_frames
        self.estimator = CausalActuatorEstimate(calibration)
        self._features: torch.Tensor | None = None
        self._pending: tuple[HistoryView, SafeRequest] | None = None

    def reset(self, residual: torch.Tensor, *, episode_id: str) -> HistoryView:
        if residual.ndim != 2 or not episode_id:
            raise ValueError("reset requires residual and episode identity")
        finite_tensor(residual, (residual.shape[0], 21), "reset residual")
        self.episode_id, self.step = episode_id, 0
        self._requested = torch.zeros_like(residual)
        self._correction = residual.new_zeros((len(residual), 11))
        self._power = residual.new_zeros(len(residual))
        self._power_step = -1
        self._features = residual.new_zeros((len(residual), self.history_frames, FEATURE_SIZE))
        self._valid = torch.zeros((len(residual), self.history_frames), device=residual.device, dtype=torch.bool)
        self._pending = None
        self.estimator.reset(residual)
        self._append(residual)
        return self.snapshot()

    @property
    def requested(self) -> torch.Tensor:
        self._require_reset()
        return self._requested.clone()

    def _require_reset(self) -> None:
        if self._features is None:
            raise RuntimeError("interface requires reset")

    def snapshot(self) -> HistoryView:
        self._require_reset()
        return HistoryView(self._features.clone(), self._valid.clone(), self.episode_id, self.step)

    def _append(self, residual: torch.Tensor) -> None:
        scalar = lambda value: residual.new_full((len(residual), 1), value)
        frame = torch.cat((residual, self._requested, self.estimator.current, self._correction,
                           self._power[:, None], scalar(self.step), scalar(self.step - 1),
                           scalar(self._power_step), scalar(self._power_step >= 0)), dim=-1)
        self._features = torch.cat((self._features[:, 1:], frame[:, None]), dim=1)
        self._valid = torch.cat((self._valid[:, 1:], torch.ones_like(self._valid[:, :1])), dim=1)

    def issue(self, baseline_delta: torch.Tensor, correction: torch.Tensor, *, step: int) -> SafeRequest:
        self._require_reset()
        if step != self.step or self._pending is not None:
            raise RuntimeError("one command per observation, in order")
        action = project_request(self._requested, baseline_delta, correction, self.limits)
        # 对外返回副本，防止调用端原地改写破坏内部命令与历史一致性。
        internal = SafeRequest(*(getattr(action, name).clone() for name in action.__dataclass_fields__))
        self._pending = (self.snapshot(), internal)
        self._requested = internal.requested_modal_rad.clone()
        self._correction = internal.normalized_correction.clone()
        self.estimator.submit(self._requested, step)
        return action

    def observe_next(self, residual: torch.Tensor, *, step: int,
                     power: PowerMeasurement | None = None) -> CausalTransition:
        self._require_reset()
        if self._pending is None or step != self.step + 1:
            raise RuntimeError("next observation must follow exactly one command")
        finite_tensor(residual, self._requested.shape, "next residual")
        if residual.device != self._requested.device or residual.dtype != self._requested.dtype:
            raise ValueError("observation device/dtype mismatch")
        target_power = residual.new_zeros(len(residual))
        target_valid = torch.zeros(len(residual), device=residual.device, dtype=torch.bool)
        if power is not None:
            finite_tensor(power.value, (len(residual),), "power")
            if power.value.device != residual.device or power.value.dtype != residual.dtype:
                raise ValueError("power device/dtype mismatch")
            if (type(power.action_step) is not int or power.action_step < 0
                    or power.action_step >= step or power.arrival_observation_step != step
                    or power.action_step <= self._power_step or bool((power.value < 0).any())):
                raise ValueError("future, duplicate, negative or mis-timestamped power")
            self._power = power.value.clone()
            self._power_step = power.action_step
            if power.action_step == self.step:
                target_power = power.value.clone()
                target_valid.fill_(True)
        before, action = self._pending
        action_step = self.step
        self.step = step
        self._append(residual.clone())
        self._pending = None
        return CausalTransition(before, action, self.snapshot(), residual.clone(), target_power,
                                target_valid, action_step, step)
