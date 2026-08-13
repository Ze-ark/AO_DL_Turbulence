"""冻结传统控制器与小幅RL残差动作的安全组合。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import torch

from src.simulation.controllers import TrackingLeakyIntegratorController


@dataclass(frozen=True)
class ResidualAction:
    """区分策略请求、传统动作和最终实际发送给环境的模态增量。"""

    normalized_request: torch.Tensor
    requested_residual_rad: torch.Tensor
    baseline_delta_rad: torch.Tensor
    final_delta_rad: torch.Tensor
    realized_residual_rad: torch.Tensor


class ResidualTrackingController:
    """在冻结跟踪积分器上叠加受限残差，并同步总请求状态。

    策略状态只包含控制器可获得的残余模态、实际加载模态和动作历史。
    S1观测末尾的仿真真值Strehl与桶内功率不会送入策略。
    """

    def __init__(
        self,
        *,
        num_modes: int,
        modal_limit_rad: float,
        history_frames: int,
        residual_action_limit_rad: float,
        final_action_step_limit_rad: float,
        gain: float,
        leak: float,
        tracking_gain: float,
    ) -> None:
        if history_frames <= 0:
            raise ValueError("history_frames must be positive")
        if not 0 < residual_action_limit_rad <= final_action_step_limit_rad:
            raise ValueError(
                "residual action limit must be positive and no larger than the final step limit"
            )
        self.num_modes = num_modes
        self.modal_limit_rad = modal_limit_rad
        self.history_frames = history_frames
        self.residual_action_limit_rad = residual_action_limit_rad
        self.final_action_step_limit_rad = final_action_step_limit_rad
        self.baseline = TrackingLeakyIntegratorController(
            num_modes=num_modes,
            modal_limit_rad=modal_limit_rad,
            gain=gain,
            leak=leak,
            tracking_gain=tracking_gain,
            max_request_step_rad=final_action_step_limit_rad,
        )
        self._history: deque[torch.Tensor] = deque(maxlen=history_frames)
        self._pending_prior_requested: torch.Tensor | None = None
        self._pending_baseline_delta: torch.Tensor | None = None

    @property
    def state_size(self) -> int:
        sensor_features = 2 * self.num_modes
        return self.history_frames * sensor_features + 2 * self.num_modes

    @property
    def requested_modal(self) -> torch.Tensor:
        if self.baseline.requested_modal is None:
            raise RuntimeError("reset must be called before requested_modal")
        return self.baseline.requested_modal

    def reset(self, observation: torch.Tensor) -> torch.Tensor:
        self._validate_observation(observation)
        self.baseline.reset(observation.shape[0], observation.device, observation.dtype)
        sensor = self._sensor_features(observation)
        self._history.clear()
        for _ in range(self.history_frames):
            self._history.append(sensor.clone())
        self._clear_pending()
        return self._prepare_state(observation)

    def advance_observation(self, observation: torch.Tensor) -> torch.Tensor:
        """记录一次新观测并准备下一次决策；每个环境时间步只能调用一次。"""
        self._validate_observation(observation)
        if self._pending_baseline_delta is not None:
            raise RuntimeError("compose_action must finish the pending decision first")
        self._history.append(self._sensor_features(observation).clone())
        return self._prepare_state(observation)

    def compose_action(self, normalized_residual: torch.Tensor) -> ResidualAction:
        """组合并投影动作，保证RL不会扩大传统控制器的单步动作范围。"""
        prior = self._pending_prior_requested
        baseline_delta = self._pending_baseline_delta
        if prior is None or baseline_delta is None:
            raise RuntimeError("reset or advance_observation must prepare a decision first")
        expected = (prior.shape[0], self.num_modes)
        if normalized_residual.shape != expected:
            raise ValueError(f"normalized residual must have shape {expected}")
        normalized = normalized_residual.to(prior.device, prior.dtype).clamp(-1, 1)
        residual_rad = normalized * self.residual_action_limit_rad
        combined = (baseline_delta + residual_rad).clamp(
            -self.final_action_step_limit_rad,
            self.final_action_step_limit_rad,
        )
        target = (prior + combined).clamp(-self.modal_limit_rad, self.modal_limit_rad)
        final_delta = target - prior
        realized_residual = final_delta - baseline_delta

        # 冻结的是控制律参数，不是内部请求状态。下一帧必须从RL实际请求后的
        # 总命令继续计算，否则控制器内部状态会与环境分叉。
        self.baseline.requested_modal = target
        self._clear_pending()
        return ResidualAction(
            normalized_request=normalized,
            requested_residual_rad=residual_rad,
            baseline_delta_rad=baseline_delta,
            final_delta_rad=final_delta,
            realized_residual_rad=realized_residual,
        )

    def _prepare_state(self, observation: torch.Tensor) -> torch.Tensor:
        if len(self._history) != self.history_frames:
            raise RuntimeError("observation history is incomplete")
        prior = self.requested_modal.clone()
        baseline_delta = self.baseline.action(observation)
        history = torch.stack(tuple(self._history), dim=1).flatten(start_dim=1)
        state = torch.cat((history, prior, baseline_delta), dim=-1)
        if state.shape[1] != self.state_size:
            raise RuntimeError("constructed residual policy state has the wrong size")
        self._pending_prior_requested = prior
        self._pending_baseline_delta = baseline_delta
        return state

    def _sensor_features(self, observation: torch.Tensor) -> torch.Tensor:
        # 明确丢弃末尾两个仿真真值质量指标，避免正式策略偷看模拟器答案。
        return observation[:, : 2 * self.num_modes]

    def _validate_observation(self, observation: torch.Tensor) -> None:
        expected = 2 * self.num_modes + 2
        if observation.ndim != 2 or observation.shape[1] != expected:
            raise ValueError(f"observation must have shape [batch, {expected}]")

    def _clear_pending(self) -> None:
        self._pending_prior_requested = None
        self._pending_baseline_delta = None


class AnchoredResidualTrackingController:
    """保持低维传统基线不变，并让RL在更高维动作基上学习残差。

    前 ``anchor_modes`` 个坐标由冻结跟踪积分器提供基线动作；其余坐标的
    基线恒为零。全部残差坐标共享由锚点维数定义的总L2预算，避免动作维数
    增加时可用总相位能量随之增加。
    """

    def __init__(
        self,
        *,
        num_modes: int,
        anchor_modes: int,
        modal_limit_rad: float,
        history_frames: int,
        residual_action_limit_rad: float,
        final_action_step_limit_rad: float,
        gain: float,
        leak: float,
        tracking_gain: float,
    ) -> None:
        if history_frames <= 0:
            raise ValueError("history_frames must be positive")
        if not 0 < anchor_modes <= num_modes:
            raise ValueError("anchor_modes must be in [1, num_modes]")
        if not 0 < residual_action_limit_rad <= final_action_step_limit_rad:
            raise ValueError(
                "residual action limit must be positive and no larger than the final step limit"
            )
        self.num_modes = num_modes
        self.anchor_modes = anchor_modes
        self.modal_limit_rad = modal_limit_rad
        self.history_frames = history_frames
        self.residual_action_limit_rad = residual_action_limit_rad
        self.final_action_step_limit_rad = final_action_step_limit_rad
        self.residual_l2_budget_rad = math.sqrt(anchor_modes) * residual_action_limit_rad
        self.final_l2_budget_rad = math.sqrt(anchor_modes) * final_action_step_limit_rad
        self.request_l2_budget_rad = math.sqrt(anchor_modes) * modal_limit_rad
        self.baseline = TrackingLeakyIntegratorController(
            num_modes=anchor_modes,
            modal_limit_rad=modal_limit_rad,
            gain=gain,
            leak=leak,
            tracking_gain=tracking_gain,
            max_request_step_rad=final_action_step_limit_rad,
        )
        self._requested_modal: torch.Tensor | None = None
        self._history: deque[torch.Tensor] = deque(maxlen=history_frames)
        self._pending_prior_requested: torch.Tensor | None = None
        self._pending_baseline_delta: torch.Tensor | None = None

    @property
    def state_size(self) -> int:
        sensor_features = 2 * self.num_modes
        return self.history_frames * sensor_features + 2 * self.num_modes

    @property
    def requested_modal(self) -> torch.Tensor:
        if self._requested_modal is None:
            raise RuntimeError("reset must be called before requested_modal")
        return self._requested_modal

    def reset(self, observation: torch.Tensor) -> torch.Tensor:
        self._validate_observation(observation)
        batch_size = observation.shape[0]
        self.baseline.reset(batch_size, observation.device, observation.dtype)
        self._requested_modal = torch.zeros(
            batch_size,
            self.num_modes,
            device=observation.device,
            dtype=observation.dtype,
        )
        sensor = self._sensor_features(observation)
        self._history.clear()
        for _ in range(self.history_frames):
            self._history.append(sensor.clone())
        self._clear_pending()
        return self._prepare_state(observation)

    def advance_observation(self, observation: torch.Tensor) -> torch.Tensor:
        self._validate_observation(observation)
        if self._pending_baseline_delta is not None:
            raise RuntimeError("compose_action must finish the pending decision first")
        self._history.append(self._sensor_features(observation).clone())
        return self._prepare_state(observation)

    def compose_action(self, normalized_residual: torch.Tensor) -> ResidualAction:
        prior = self._pending_prior_requested
        baseline_delta = self._pending_baseline_delta
        if prior is None or baseline_delta is None:
            raise RuntimeError("reset or advance_observation must prepare a decision first")
        expected = (prior.shape[0], self.num_modes)
        if normalized_residual.shape != expected:
            raise ValueError(f"normalized residual must have shape {expected}")

        normalized = normalized_residual.to(prior.device, prior.dtype).clamp(-1, 1)
        residual_rad, _ = project_box_and_l2(
            normalized * self.residual_action_limit_rad,
            component_limit=self.residual_action_limit_rad,
            l2_limit=self.residual_l2_budget_rad,
        )
        combined, _ = project_box_and_l2(
            baseline_delta + residual_rad,
            component_limit=self.final_action_step_limit_rad,
            l2_limit=self.final_l2_budget_rad,
        )
        projected_target, _ = project_box_and_l2(
            prior + combined,
            component_limit=self.modal_limit_rad,
            l2_limit=self.request_l2_budget_rad,
        )
        # 请求集合是凸集。把“prior -> projected_target”的线段按统一比例缩短，
        # 可同时保持累计请求可行和单步盒/L2约束；逐坐标再次裁剪会破坏该性质。
        final_delta = scale_to_box_and_l2(
            projected_target - prior,
            component_limit=self.final_action_step_limit_rad,
            l2_limit=self.final_l2_budget_rad,
        )
        target = prior + final_delta
        realized_residual = final_delta - baseline_delta

        self._requested_modal = target
        self.baseline.requested_modal = target[:, : self.anchor_modes].clone()
        self._clear_pending()
        return ResidualAction(
            normalized_request=normalized,
            requested_residual_rad=residual_rad,
            baseline_delta_rad=baseline_delta,
            final_delta_rad=final_delta,
            realized_residual_rad=realized_residual,
        )

    def _prepare_state(self, observation: torch.Tensor) -> torch.Tensor:
        if len(self._history) != self.history_frames:
            raise RuntimeError("observation history is incomplete")
        prior = self.requested_modal.clone()
        baseline_delta = torch.zeros_like(prior)
        baseline_delta[:, : self.anchor_modes] = self.baseline.action(
            self._anchor_observation(observation)
        )
        history = torch.stack(tuple(self._history), dim=1).flatten(start_dim=1)
        state = torch.cat((history, prior, baseline_delta), dim=-1)
        if state.shape[1] != self.state_size:
            raise RuntimeError("constructed residual policy state has the wrong size")
        self._pending_prior_requested = prior
        self._pending_baseline_delta = baseline_delta
        return state

    def _anchor_observation(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                observation[:, : self.anchor_modes],
                observation[:, self.num_modes : self.num_modes + self.anchor_modes],
                observation[:, -2:],
            ),
            dim=-1,
        )

    def _sensor_features(self, observation: torch.Tensor) -> torch.Tensor:
        return observation[:, : 2 * self.num_modes]

    def _validate_observation(self, observation: torch.Tensor) -> None:
        expected = 2 * self.num_modes + 2
        if observation.ndim != 2 or observation.shape[1] != expected:
            raise ValueError(f"observation must have shape [batch, {expected}]")

    def _clear_pending(self) -> None:
        self._pending_prior_requested = None
        self._pending_baseline_delta = None


def project_box_and_l2(
    values: torch.Tensor,
    *,
    component_limit: float,
    l2_limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """先限制每个坐标，再限制每个样本的总L2范数。"""
    if component_limit <= 0 or l2_limit <= 0:
        raise ValueError("action limits must be positive")
    safe_component_limit = component_limit * (1 - 1e-6)
    safe_l2_limit = l2_limit * (1 - 1e-6)
    clipped = values.clamp(-safe_component_limit, safe_component_limit)
    norm = torch.linalg.vector_norm(clipped, dim=-1, keepdim=True)
    scale = (safe_l2_limit / norm.clamp_min(1e-12)).clamp(max=1.0)
    projected = clipped * scale
    return projected, scale.squeeze(-1).lt(1 - 1e-7).to(values.dtype)


def scale_to_box_and_l2(
    values: torch.Tensor,
    *,
    component_limit: float,
    l2_limit: float,
) -> torch.Tensor:
    """只做统一比例缩放，以保持从可行旧请求到可行新请求的凸组合。"""
    if component_limit <= 0 or l2_limit <= 0:
        raise ValueError("action limits must be positive")
    safe_component_limit = component_limit * (1 - 1e-6)
    safe_l2_limit = l2_limit * (1 - 1e-6)
    max_abs = values.abs().amax(dim=-1, keepdim=True)
    norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    component_scale = safe_component_limit / max_abs.clamp_min(1e-12)
    l2_scale = safe_l2_limit / norm.clamp_min(1e-12)
    scale = torch.minimum(component_scale, l2_scale).clamp(max=1.0)
    return values * scale
