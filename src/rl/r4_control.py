"""R4共有请求投影与因果名义执行器；不读取仿真器实际状态。"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import torch

from src.rl.residual_control import project_box_and_l2, scale_to_box_and_l2


@dataclass(frozen=True)
class R4Limits:
    modes: int = 21
    anchor_modes: int = 10
    correction_rad: float = 0.0125
    residual_rad: float = 0.05
    step_rad: float = 0.15
    modal_rad: float = 3.0

    def validate(self) -> None:
        if self.modes != 21 or self.anchor_modes != 10:
            raise ValueError("R4 requires the frozen 21=10+11 representation")
        if not all(math.isfinite(x) for x in
                   (self.correction_rad, self.residual_rad, self.step_rad, self.modal_rad)):
            raise ValueError("non-finite R4 limits")
        if not 0 < self.correction_rad <= self.residual_rad <= self.step_rad <= self.modal_rad:
            raise ValueError("invalid R4 action limits")


@dataclass(frozen=True)
class SafeRequest:
    raw_correction: torch.Tensor
    normalized_correction: torch.Tensor
    requested_residual_rad: torch.Tensor
    requested_delta_rad: torch.Tensor
    requested_modal_rad: torch.Tensor


def finite_tensor(value: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if value.shape != shape or not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"invalid {name}: expected finite floating tensor {shape}")


def project_request(prior: torch.Tensor, baseline_delta: torch.Tensor,
                    correction: torch.Tensor, limits: R4Limits) -> SafeRequest:
    """先约束高阶残差，再约束全动作和累计请求；保留内部可微区域。"""
    limits.validate()
    if prior.ndim != 2:
        raise ValueError("prior must be batch by modes")
    b = prior.shape[0]
    finite_tensor(prior, (b, limits.modes), "prior")
    finite_tensor(baseline_delta, prior.shape, "baseline")
    finite_tensor(correction, (b, limits.modes - limits.anchor_modes), "correction")
    if any(v.device != prior.device or v.dtype != prior.dtype for v in (baseline_delta, correction)):
        raise ValueError("R4 action device/dtype mismatch")
    root = math.sqrt(limits.anchor_modes)
    if bool((prior.abs() > limits.modal_rad + 1e-6).any()) or bool(
            (prior.norm(dim=-1) > root * limits.modal_rad + 1e-6).any()):
        raise ValueError("infeasible prior cannot be repaired by a bounded step")
    normalized = correction.clamp(-1, 1)
    high, _ = project_box_and_l2(
        baseline_delta[:, limits.anchor_modes:] + limits.correction_rad * normalized,
        component_limit=limits.residual_rad, l2_limit=root * limits.residual_rad)
    combined = torch.cat((baseline_delta[:, :limits.anchor_modes], high), dim=-1)
    combined = scale_to_box_and_l2(combined, component_limit=limits.step_rad,
                                    l2_limit=root * limits.step_rad)
    target, _ = project_box_and_l2(prior + combined, component_limit=limits.modal_rad,
                                    l2_limit=root * limits.modal_rad)
    delta = scale_to_box_and_l2(target - prior, component_limit=limits.step_rad,
                                l2_limit=root * limits.step_rad)
    return SafeRequest(correction.clone(), normalized, high, delta, prior + delta)


@dataclass(frozen=True)
class NominalCalibration:
    delay_frames: int = 2
    settling_fraction: float = 0.5
    modal_slew_rad: float = 0.15
    provenance: str = "simulation_nominal_assumption_not_hardware_calibrated"

    def validate(self) -> None:
        if (type(self.delay_frames) is not int or self.delay_frames < 0
                or not math.isfinite(self.settling_fraction)
                or not 0 < self.settling_fraction <= 1
                or not math.isfinite(self.modal_slew_rad) or self.modal_slew_rad <= 0
                or not self.provenance):
            raise ValueError("invalid nominal calibration")


class CausalActuatorEstimate:
    """模态空间的名义近似，不宣称等于逐像素SLM实际加载相位。"""

    def __init__(self, calibration: NominalCalibration):
        calibration.validate()
        self.calibration = calibration
        self._current: torch.Tensor | None = None
        self._queue: deque[torch.Tensor] = deque()
        self.next_command_step = 0

    def reset(self, reference: torch.Tensor) -> None:
        self._current = torch.zeros_like(reference)
        self._queue = deque(torch.zeros_like(reference) for _ in range(self.calibration.delay_frames))
        self.next_command_step = 0

    @property
    def current(self) -> torch.Tensor:
        if self._current is None:
            raise RuntimeError("estimator requires reset")
        return self._current.clone()

    def submit(self, requested: torch.Tensor, step: int) -> torch.Tensor:
        if self._current is None or step != self.next_command_step:
            raise RuntimeError("out-of-order actuator command")
        finite_tensor(requested, self._current.shape, "actuator request")
        if requested.device != self._current.device or requested.dtype != self._current.dtype:
            raise ValueError("actuator request device/dtype mismatch")
        if self.calibration.delay_frames:
            self._queue.append(requested.clone())
            delayed = self._queue.popleft()
        else:
            delayed = requested
        delta = (delayed - self._current).clamp(
            -self.calibration.modal_slew_rad, self.calibration.modal_slew_rad)
        self._current = self._current + self.calibration.settling_fraction * delta
        self.next_command_step += 1
        return self.current
