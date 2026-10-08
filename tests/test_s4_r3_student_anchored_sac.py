from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.rl.residual_control import AnchoredResidualTrackingController
from src.rl.residual_sac import ResidualSacAgent, SacConfig
from src.rl.s4_high_order_learnability import HighOrderImitationPolicy
from src.rl.s4_r3_student_anchored_sac import run_s4_r3_student_anchored_sac
from src.rl.student_anchored_control import (
    FrozenStudentPolicy,
    StudentAnchoredResidualController,
    initialize_actor_from_student,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "s4_student_anchored_sac_r3_v1.yaml"


def _observation(batch: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1234)
    residual = 0.2 * torch.randn(batch, 21, generator=generator)
    applied = 0.1 * torch.randn(batch, 21, generator=generator)
    metrics = torch.tensor([[0.4, 0.6]]).expand(batch, -1)
    return torch.cat((residual, applied, metrics), dim=-1)


def _student(*, output_bias: float = 0.0) -> FrozenStudentPolicy:
    model = HighOrderImitationPolicy(210, 256, 11)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.network[4].bias.fill_(output_bias)
    model.eval()
    return FrozenStudentPolicy(
        identifier="synthetic_student",
        model=model,
        state_mean=torch.zeros(210),
        state_scale=torch.ones(210),
    )


def _controller(student: FrozenStudentPolicy) -> StudentAnchoredResidualController:
    base = AnchoredResidualTrackingController(
        num_modes=21,
        anchor_modes=10,
        modal_limit_rad=3.0,
        history_frames=4,
        residual_action_limit_rad=0.05,
        final_action_step_limit_rad=0.15,
        gain=0.25,
        leak=0.10,
        tracking_gain=0.50,
    )
    return StudentAnchoredResidualController(
        base,
        student,
        student_scale=0.50,
        correction_component_limit_rad=0.0125,
    )


def test_zero_correction_preserves_student_anchor_and_ten_mode_baseline() -> None:
    controller = _controller(_student(output_bias=0.2))
    state = controller.reset(_observation())
    action = controller.compose_action(state, torch.zeros(3, 11))

    expected_student = torch.full((3, 11), torch.tanh(torch.tensor(0.2)) * 0.5)
    assert torch.allclose(action.intended_added_normalized, expected_student)
    assert torch.count_nonzero(action.correction_normalized) == 0
    assert torch.allclose(
        action.composed.requested_residual_rad[:, 10:],
        expected_student * 0.05,
        atol=1e-7,
    )
    assert torch.all(action.composed.final_delta_rad.abs() <= 0.15 + 1e-7)


def test_correction_limit_is_exactly_one_quarter_of_shared_residual_limit() -> None:
    controller = _controller(_student())
    state = controller.reset(_observation())
    action = controller.compose_action(state, torch.ones(3, 11))

    assert torch.allclose(
        action.intended_added_normalized,
        torch.full((3, 11), 0.25),
    )
    assert torch.allclose(
        action.composed.requested_residual_rad[:, 10:],
        torch.full((3, 11), 0.0125),
    )


def test_student_backbone_transfer_keeps_deterministic_correction_zero() -> None:
    torch.manual_seed(12)
    student = _student()
    with torch.no_grad():
        student.model.network[0].weight.normal_()
        student.model.network[2].weight.normal_()
    agent = ResidualSacAgent(SacConfig(210, 11, hidden_size=256), torch.device("cpu"))
    initialize_actor_from_student(agent.actor, student)

    assert torch.equal(agent.actor.backbone[0].weight, student.model.network[0].weight)
    assert torch.equal(agent.actor.backbone[2].weight, student.model.network[2].weight)
    assert torch.count_nonzero(agent.act(torch.randn(4, 210), deterministic=True)) == 0


def test_r3_preflight_locks_upstream_student_action_and_six_runs() -> None:
    result = run_s4_r3_student_anchored_sac(
        CONFIG, quick=False, preflight_only=True
    )

    assert result["status"] == "READY_FOR_USER_TRAINING"
    assert result["upstream_status"] == "SINGLE_ROUND_STUDENT_AGGREGATION_PASS"
    assert result["state_size"] == 210
    assert result["action_size"] == 11
    assert result["student_frozen"] is True
    assert result["deterministic_initial_correction_zero"] is True
    assert result["total_runs"] == 6
    assert result["total_transitions"] == 2_995_200
    assert result["sealed_s4d3_access"] is False


def test_student_controller_rejects_correction_larger_than_shared_limit() -> None:
    base = AnchoredResidualTrackingController(
        num_modes=21,
        anchor_modes=10,
        modal_limit_rad=3.0,
        history_frames=4,
        residual_action_limit_rad=0.05,
        final_action_step_limit_rad=0.15,
        gain=0.25,
        leak=0.10,
        tracking_gain=0.50,
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        StudentAnchoredResidualController(
            base,
            _student(),
            student_scale=0.5,
            correction_component_limit_rad=0.051,
        )
