"""无环境引用的因果交叉熵规划器；不训练、不执行硬件动作。"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn

from src.rl.r4_control import R4Limits, NominalCalibration, project_request
from src.rl.r4_dynamics import advance_history
from src.rl.r4_trajectory import anchor_delta


@dataclass(frozen=True)
class SearchConfig:
    horizon: int = 8
    population: int = 128
    iterations: int = 4
    elites: int = 16
    initial_std: float = .5
    min_std: float = .05
    smoothing: float = .1
    discount: float = .99
    action_cost: float = .01
    disagreement: float = 1.

    def validate(self) -> None:
        if any(type(x) is not int or x < 1 for x in (self.horizon, self.population, self.iterations, self.elites)):
            raise ValueError('positive integer search dimensions required')
        if not 1 <= self.elites <= self.population or self.population < 2:
            raise ValueError('invalid elite count')
        if not all(math.isfinite(x) for x in (self.initial_std, self.min_std, self.smoothing, self.discount, self.action_cost, self.disagreement)):
            raise ValueError('nonfinite search config')
        if not (0 < self.min_std <= self.initial_std <= 1 and 0 <= self.smoothing < 1
                and 0 < self.discount <= 1 and self.action_cost >= 0 and self.disagreement >= 0):
            raise ValueError('invalid search config')


@torch.no_grad()
def sequence_scores(models: list[nn.Module], history: torch.Tensor, valid: torch.Tensor,
                    sequences: torch.Tensor, anchor: dict, calibration: NominalCalibration,
                    config: SearchConfig) -> torch.Tensor:
    """输入[B,N,H,11]，输出[B,N]；每个模型分别续推自己的观测。"""
    b, n, horizon, modes = sequences.shape
    if (horizon, modes) != (config.horizon, 11) or not torch.isfinite(sequences).all() or sequences.abs().max() > 1:
        raise ValueError('invalid candidate sequences')
    returns = []
    for model in models:
        h = history[:, None].expand(-1, n, -1, -1).reshape(b*n, 8, 79).clone()
        v = valid[:, None].expand(-1, n, -1).reshape(b*n, 8).clone()
        total = history.new_zeros(b*n)
        for step in range(horizon):
            u = sequences[:, :, step].reshape(b*n, 11)
            request = project_request(h[:, -1, 21:42], anchor_delta(h[:, -1], anchor), u, R4Limits())
            pred = model(h, v, request.requested_delta_rad)
            if pred.shape != (b*n, 22) or not torch.isfinite(pred).all():
                raise RuntimeError('invalid MPC prediction; stop without execution')
            total += config.discount**step * (pred[:, 21]-config.action_cost*u.square().mean(-1))
            h, v = advance_history(h, v, request.requested_delta_rad, request.normalized_correction, pred, calibration)
        returns.append(total.reshape(b, n))
    values = torch.stack(returns)
    score = values.mean(0)-config.disagreement*values.std(0, correction=0)
    if not torch.isfinite(score).all():
        raise RuntimeError('nonfinite MPC score')
    return score


@torch.no_grad()
def plan(models: list[nn.Module], history: torch.Tensor, valid: torch.Tensor, anchor: dict,
         calibration: NominalCalibration, config: SearchConfig, *, seed: int) -> dict:
    """无热启动/持久状态；相同输入及种子产生相同决策。只返回第一步请求。"""
    config.validate(); calibration.validate()
    if not models or any(m.training or any(p.requires_grad for p in m.parameters()) for m in models):
        raise ValueError('MPC requires frozen eval models')
    b = len(history)
    if (not b or history.shape != (b, 8, 79) or valid.shape != (b, 8)
            or valid.dtype != torch.bool or valid.device != history.device
            or not bool(valid[:, -1].all()) or not bool(torch.isfinite(history).all())):
        raise ValueError('invalid causal history')
    generator = torch.Generator(device=history.device).manual_seed(seed)
    mean = history.new_zeros(b, config.horizon, 11)
    std = torch.full_like(mean, config.initial_std)
    best_sequence = mean.clone(); best_score = history.new_full((b,), -torch.inf)
    trace = []; zero_score = None
    for iteration in range(config.iterations):
        noise = torch.randn((b, config.population, config.horizon, 11), generator=generator,
                            device=history.device, dtype=history.dtype)
        candidates = (mean[:, None]+std[:, None]*noise).clamp(-1, 1)
        candidates[:, 0] = 0
        scores = sequence_scores(models, history, valid, candidates, anchor, calibration, config)
        if zero_score is None: zero_score = scores[:, 0].clone()
        order = torch.argsort(scores, dim=1, descending=True, stable=True)
        elite = candidates.gather(1, order[:, :config.elites, None, None].expand(-1, -1, config.horizon, 11))
        value = scores.gather(1, order[:, :1]).squeeze(1)
        improved = value > best_score
        best_sequence = torch.where(improved[:, None, None], elite[:, 0], best_sequence)
        best_score = torch.maximum(best_score, value)
        mean = config.smoothing*mean+(1-config.smoothing)*elite.mean(1)
        std = (config.smoothing*std+(1-config.smoothing)*elite.std(1, correction=0)).clamp_min(config.min_std)
        trace.append(best_score.clone())
    request = project_request(history[:, -1, 21:42], anchor_delta(history[:, -1], anchor), best_sequence[:, 0], R4Limits())
    return dict(correction=best_sequence[:, 0], sequence=best_sequence, score=best_score,
                zero_score=zero_score, iteration_best=torch.stack(trace),
                requested_delta=request.requested_delta_rad, requested_modal=request.requested_modal_rad,
                model_forward_samples=b*config.population*config.iterations*config.horizon*len(models))
