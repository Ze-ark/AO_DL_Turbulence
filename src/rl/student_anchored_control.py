"""R3冻结学生基座与小幅SAC修正的安全动作合成。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.rl.residual_control import AnchoredResidualTrackingController, ResidualAction
from src.rl.s4_high_order_learnability import HighOrderImitationPolicy


@dataclass(frozen=True)
class FrozenStudentPolicy:
    """冻结的学生网络及其训练时状态归一化。"""

    identifier: str
    model: HighOrderImitationPolicy
    state_mean: torch.Tensor
    state_scale: torch.Tensor

    @torch.no_grad()
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        normalized = (state - self.state_mean) / self.state_scale
        return self.model(normalized).clamp(-1, 1)


@dataclass(frozen=True)
class StudentAnchoredAction:
    """把学生基座、RL修正和安全投影后的动作分别保留下来。"""

    student_normalized: torch.Tensor
    correction_normalized: torch.Tensor
    intended_added_normalized: torch.Tensor
    composed: ResidualAction
    correction_projection_fraction: torch.Tensor
    final_projection_fraction: torch.Tensor


class StudentAnchoredResidualController:
    """传统十模态基线 + 冻结学生十一模态 + SAC十一模态小修正。"""

    def __init__(
        self,
        controller: AnchoredResidualTrackingController,
        student: FrozenStudentPolicy,
        *,
        student_scale: float,
        correction_component_limit_rad: float,
        projection_tolerance_rad: float = 1e-7,
    ) -> None:
        if student_scale <= 0 or student_scale > 1:
            raise ValueError("student_scale must be in (0, 1]")
        if correction_component_limit_rad <= 0:
            raise ValueError("correction component limit must be positive")
        if correction_component_limit_rad > controller.residual_action_limit_rad:
            raise ValueError("correction limit cannot exceed residual action limit")
        if projection_tolerance_rad < 0:
            raise ValueError("projection tolerance must be non-negative")
        self.controller = controller
        self.student = student
        self.student_scale = float(student_scale)
        self.correction_scale = (
            float(correction_component_limit_rad)
            / controller.residual_action_limit_rad
        )
        self.projection_tolerance_rad = float(projection_tolerance_rad)
        self.added_modes = controller.num_modes - controller.anchor_modes
        if self.added_modes <= 0:
            raise ValueError("student-anchored control requires added modes")

    @property
    def state_size(self) -> int:
        return self.controller.state_size

    @property
    def requested_modal(self) -> torch.Tensor:
        return self.controller.requested_modal

    def reset(self, observation: torch.Tensor) -> torch.Tensor:
        return self.controller.reset(observation)

    def advance_observation(self, observation: torch.Tensor) -> torch.Tensor:
        return self.controller.advance_observation(observation)

    @torch.no_grad()
    def student_action(self, state: torch.Tensor) -> torch.Tensor:
        action = self.student.predict(state)
        expected = (state.shape[0], self.added_modes)
        if action.shape != expected:
            raise RuntimeError(f"student action must have shape {expected}")
        return action

    def compose_action(
        self,
        state: torch.Tensor,
        correction_normalized: torch.Tensor,
    ) -> StudentAnchoredAction:
        expected = (state.shape[0], self.added_modes)
        if correction_normalized.shape != expected:
            raise ValueError(f"correction action must have shape {expected}")
        correction = correction_normalized.to(state.device, state.dtype).clamp(-1, 1)
        student = self.student_action(state)
        intended_added = self.student_scale * student + self.correction_scale * correction
        full = torch.zeros(
            state.shape[0],
            self.controller.num_modes,
            device=state.device,
            dtype=state.dtype,
        )
        full[:, self.controller.anchor_modes :] = intended_added
        composed = self.controller.compose_action(full)
        requested_added = composed.requested_residual_rad[
            :, self.controller.anchor_modes :
        ]
        intended_added_rad = (
            intended_added * self.controller.residual_action_limit_rad
        )
        correction_projection_fraction = (
            (requested_added - intended_added_rad).abs()
            > self.projection_tolerance_rad
        ).float().mean(dim=-1)
        realized_added = composed.realized_residual_rad[
            :, self.controller.anchor_modes :
        ]
        final_projection_fraction = (
            (realized_added - requested_added).abs() > self.projection_tolerance_rad
        ).float().mean(dim=-1)
        return StudentAnchoredAction(
            student_normalized=student,
            correction_normalized=correction,
            intended_added_normalized=intended_added,
            composed=composed,
            correction_projection_fraction=correction_projection_fraction,
            final_projection_fraction=final_projection_fraction,
        )


def load_frozen_student(
    checkpoint_path: str | Path,
    *,
    identifier: str,
    state_size: int,
    hidden_size: int,
    output_size: int,
    device: torch.device,
) -> FrozenStudentPolicy:
    """加载并冻结聚合学生检查点，拒绝缺失或错形状的归一化。"""
    payload: dict[str, Any] = torch.load(
        Path(checkpoint_path), map_location=device, weights_only=False
    )
    model = HighOrderImitationPolicy(state_size, hidden_size, output_size).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    state_mean = torch.as_tensor(payload["state_mean"], device=device).float()
    state_scale = torch.as_tensor(payload["state_scale"], device=device).float()
    if state_mean.shape != (state_size,) or state_scale.shape != (state_size,):
        raise RuntimeError("student normalization has the wrong shape")
    if not bool(torch.isfinite(state_mean).all()) or not bool(
        torch.isfinite(state_scale).all()
    ):
        raise RuntimeError("student normalization contains non-finite values")
    if bool((state_scale <= 0).any()):
        raise RuntimeError("student normalization scale must be positive")
    return FrozenStudentPolicy(
        identifier=identifier,
        model=model,
        state_mean=state_mean,
        state_scale=state_scale,
    )


def initialize_actor_from_student(
    actor: torch.nn.Module,
    student: FrozenStudentPolicy,
) -> None:
    """只迁移学生的两层特征骨干；SAC均值头保持精确零初始化。"""
    student_layers = student.model.network
    actor_backbone = getattr(actor, "backbone")
    actor_mean_head = getattr(actor, "mean_head")
    with torch.no_grad():
        actor_backbone[0].weight.copy_(student_layers[0].weight)
        actor_backbone[0].bias.copy_(student_layers[0].bias)
        actor_backbone[2].weight.copy_(student_layers[2].weight)
        actor_backbone[2].bias.copy_(student_layers[2].bias)
        actor_mean_head.weight.zero_()
        actor_mean_head.bias.zero_()
