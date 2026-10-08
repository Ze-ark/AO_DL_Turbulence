"""R4白名单传统候选与MPC请求端回退；无环境真值输入。"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
from src.rl.r4_control import NominalCalibration, R4Limits, project_request
from src.rl.r4_dynamics import advance_history
from src.rl.r4_trajectory import anchor_delta


@dataclass(frozen=True)
class BaselineSpec:
    kind: str
    gain: float
    leak: float
    tracking_gain: float = .5
    prediction_steps: int = 2

    def validate(self) -> None:
        if self.kind not in ('integrator', 'tracking', 'ridge'):
            raise ValueError('unknown baseline')
        if not all(math.isfinite(x) for x in (self.gain, self.leak, self.tracking_gain)):
            raise ValueError('nonfinite baseline parameters')
        if not (0 < self.gain <= 1 and 0 <= self.leak <= 1 and 0 <= self.tracking_gain <= 1):
            raise ValueError('invalid baseline parameters')
        if self.prediction_steps != 2:
            raise ValueError('two-step nominal predictor is frozen')


def validate_history(h: torch.Tensor, v: torch.Tensor) -> None:
    b = len(h)
    if (not b or h.shape != (b, 8, 79) or v.shape != (b, 8) or v.dtype != torch.bool
            or v.device != h.device or not h.is_floating_point() or not bool(torch.isfinite(h).all())
            or not bool(v[:, -1].all()) or bool((v[:, :-1] & ~v[:, 1:]).any())):
        raise ValueError('invalid causal history: stop, do not fabricate fallback observations')
    if bool((h.masked_select(~v[:, :, None].expand_as(h)) != 0).any()):
        raise ValueError('invalid history padding')
    # 同时校验请求状态可行；不读取实际SLM状态。
    project_request(h[:, -1, 21:42], torch.zeros_like(h[:, -1, :21]), h.new_zeros(b, 11), R4Limits())


@torch.no_grad()
def baseline_delta(spec: BaselineSpec, h: torch.Tensor, v: torch.Tensor,
                   linear: list[nn.Module], calibration: NominalCalibration) -> tuple[torch.Tensor, int]:
    spec.validate(); validate_history(h, v)
    params = dict(gain=spec.gain, leak=spec.leak,
                  tracking_gain=0. if spec.kind == 'integrator' else spec.tracking_gain)
    ordinary = anchor_delta(h[:, -1], params)
    if spec.kind != 'ridge': return ordinary, 0
    if len(linear) != 3 or any(m.training or any(p.requires_grad for p in m.parameters()) for m in linear):
        raise ValueError('ridge predictor requires three frozen linear models')
    predictions = []
    for model in linear:
        history, valid = h.clone(), v.clone()
        for _ in range(spec.prediction_steps):
            request = project_request(history[:, -1, 21:42], anchor_delta(history[:, -1], params),
                                      h.new_zeros(len(h), 11), R4Limits())
            pred = model(history, valid, request.requested_delta_rad)
            if pred.shape != (len(h), 22) or not bool(torch.isfinite(pred).all()):
                raise RuntimeError('nonfinite ridge prediction; stop')
            history, valid = advance_history(history, valid, request.requested_delta_rad,
                                              request.normalized_correction, pred, calibration)
        predictions.append(pred[:, :21])
    predicted_residual = torch.stack(predictions).mean(0)
    return ordinary + spec.gain*(h[:, -1, :21]-predicted_residual), len(h)*3*spec.prediction_steps


def range_eligible(h: torch.Tensor, v: torch.Tensor, bounds: dict) -> torch.Tensor:
    """训练观测范围启发式；不是硬件安全保证或分布内证明。"""
    validate_history(h, v)
    residual_max = torch.as_tensor(bounds['residual_abs_max'], device=h.device, dtype=h.dtype)
    if residual_max.shape != (21,) or not bool(torch.isfinite(residual_max).all()) or not bool((residual_max > 0).all()):
        raise ValueError('invalid residual calibration')
    limit = float(bounds['power_max'])
    if not math.isfinite(limit) or limit <= 0: raise ValueError('invalid power calibration')
    ok_residual = ((h[:, :, :21].abs() <= residual_max).all(-1) | ~v).all(-1)
    measured = v & h[:, :, 78].bool()
    ok_power = (((h[:, :, 74] >= 0) & (h[:, :, 74] <= limit)) | ~measured).all(-1)
    return ok_residual & ok_power


def guarded_correction(h: torch.Tensor, v: torch.Tensor, proposal: torch.Tensor,
                       bounds: dict) -> tuple[torch.Tensor, list[str]]:
    eligible = range_eligible(h, v, bounds)
    if proposal.shape != (len(h), 11) or proposal.device != h.device or proposal.dtype != h.dtype:
        raise ValueError('invalid proposal contract')
    finite = torch.isfinite(proposal).all(-1)
    inside = (proposal.abs() <= 1).all(-1)
    accept = eligible & finite & inside
    reasons = ['accepted' if bool(accept[i]) else ('outside_observation_range' if not bool(eligible[i])
        else 'nonfinite_proposal' if not bool(finite[i]) else 'proposal_out_of_bounds') for i in range(len(h))]
    return torch.where(accept[:, None], proposal, torch.zeros_like(proposal)), reasons
