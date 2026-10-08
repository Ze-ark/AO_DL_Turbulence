"""动作条件ARX及冻结ARX+GRU残差；只监督预测，不训练控制策略。"""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from src.rl.r4_control import NominalCalibration
from src.rl.r4_trajectory import EpisodeStore


def causal_features(history: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """绝对步索引转为相对当前时刻；不依赖未来或隐藏档位。"""
    x = history.clone()
    now = history[:, -1:, 75]
    x[:, :, 75:78] = (history[:, :, 75:78] - now[:, :, None]) / 8
    x = x * valid[:, :, None]
    return torch.cat((x, valid[:, :, None].to(x.dtype)), dim=-1)


def design_vector(history: torch.Tensor, valid: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
    return torch.cat((causal_features(history, valid).flatten(1), command), dim=-1)


def advance_history(history: torch.Tensor, valid: torch.Tensor, command: torch.Tensor,
                    correction: torch.Tensor, prediction: torch.Tensor,
                    calibration: NominalCalibration) -> tuple[torch.Tensor, torch.Tensor]:
    """只以预测值推进观测；名义命令队列从已有历史恢复，保留梯度。"""
    calibration.validate()
    if calibration.delay_frames >= history.shape[1]:
        raise ValueError("nominal delay exceeds available command history")
    last = history[:, -1]
    requested = last[:, 21:42] + command
    if calibration.delay_frames:
        i = -calibration.delay_frames
        delayed = history[:, i, 21:42] * valid[:, i, None]
    else:
        delayed = requested
    estimate = last[:, 42:63] + calibration.settling_fraction * (
        delayed - last[:, 42:63]).clamp(-calibration.modal_slew_rad, calibration.modal_slew_rad)
    frame = torch.cat((prediction[:, :21], requested, estimate, correction,
                       prediction[:, 21:22], last[:, 75:76]+1, last[:, 75:76],
                       last[:, 75:76], torch.ones_like(last[:, 75:76])), dim=-1)
    return (torch.cat((history[:, 1:], frame[:, None]), dim=1),
            torch.cat((valid[:, 1:], torch.ones_like(valid[:, :1])), dim=1))


class ARXDynamics(nn.Module):
    def __init__(self, x_mean: torch.Tensor, x_scale: torch.Tensor,
                 y_mean: torch.Tensor, y_scale: torch.Tensor):
        super().__init__()
        for name, value in dict(x_mean=x_mean, x_scale=x_scale, y_mean=y_mean, y_scale=y_scale).items():
            self.register_buffer(name, value.detach().clone())
        self.register_buffer("weight", x_mean.new_zeros(len(x_mean)+1, 22))

    def features(self, history: torch.Tensor, valid: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        return (design_vector(history, valid, command)-self.x_mean)/self.x_scale

    def forward(self, history: torch.Tensor, valid: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        x = self.features(history, valid, command)
        y = torch.cat((x, torch.ones_like(x[:, :1])), dim=-1) @ self.weight
        target = y*self.y_scale + self.y_mean
        return torch.cat((target[:, :21]+history[:, -1, :21], target[:, 21:]), dim=-1)


class ResidualGRUDynamics(nn.Module):
    def __init__(self, linear: ARXDynamics, hidden_size: int = 64):
        super().__init__()
        self.linear = deepcopy(linear).requires_grad_(False)
        self.gru = nn.GRU(80, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size+21, 22)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, history: torch.Tensor, valid: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        x = self.linear.features(history, valid, command)
        frames = x[:, :640].reshape(-1, 8, 80) * valid[:, :, None]
        # 补零步仍可能触发GRU偏置；逐帧掩码冻结无效步的隐藏状态。
        hidden = frames.new_zeros(1, len(frames), self.gru.hidden_size)
        for t in range(8):
            _, candidate = self.gru(frames[:, t:t+1], hidden)
            hidden = torch.where(valid[:, t][None, :, None], candidate, hidden)
        residual = self.head(torch.cat((hidden[0], x[:, 640:]), dim=-1)) * self.linear.y_scale
        return self.linear(history, valid, command) + residual


def rollout(model: nn.Module, history: torch.Tensor, valid: torch.Tensor,
            commands: torch.Tensor, corrections: torch.Tensor,
            calibration: NominalCalibration) -> torch.Tensor:
    predictions = []
    for k in range(commands.shape[1]):
        pred = model(history, valid, commands[:, k])
        predictions.append(pred)
        history, valid = advance_history(history, valid, commands[:, k], corrections[:, k], pred, calibration)
    return torch.stack(predictions, dim=1)


def normalized_errors(prediction: torch.Tensor, batch: dict, scale: torch.Tensor,
                      horizons: list[int]) -> torch.Tensor:
    target = torch.cat((batch["target_residual"], batch["target_power"][:, :, None]), dim=-1)
    indices = torch.tensor(horizons, device=prediction.device)-1
    sq = ((prediction[:, indices]-target[:, indices])/scale).square()
    return torch.stack((sq[:, :, :21].mean(-1), sq[:, :, 21]), dim=-1)


@torch.no_grad()
def training_statistics(store: EpisodeStore, cfg: dict, device: torch.device,
                        on_chunk=None) -> ARXDynamics:
    """只遍历训练数据计算尺度；不接受开发数据作为参数。"""
    sx = torch.zeros(661, device=device, dtype=torch.float64); xx = sx.clone()
    sy = torch.zeros(22, device=device, dtype=torch.float64); yy = sy.clone()
    # 标签尺度采用下一观测的尺度，残余变化均值单独计算。
    st = sy.clone(); tt = sy.clone(); n = 0
    for batch in all_transitions(store, torch.arange(store.count), cfg["ridge_chunk"], device):
        x = design_vector(batch["history"], batch["valid"], batch["commands"][:, 0]).double()
        target = torch.cat((batch["target_residual"][:, 0], batch["target_power"][:, :1]), dim=-1).double()
        y = target.clone(); y[:, :21] -= batch["history"][:, -1, :21].double()
        sx += x.sum(0); xx += x.square().sum(0)
        sy += y.sum(0); yy += y.square().sum(0)
        st += target.sum(0); tt += target.square().sum(0); n += len(x)
        if on_chunk: on_chunk()
    xmean = sx/n
    xscale = (xx/n-xmean.square()).clamp_min(0).sqrt().clamp_min(cfg["input_scale_floor"])
    target_scale = (tt/n-(st/n).square()).clamp_min(0).sqrt()
    target_scale[:21].clamp_(min=cfg["residual_scale_floor"])
    target_scale[21:].clamp_(min=cfg["power_scale_floor"])
    return ARXDynamics(xmean.float(), xscale.float(), (sy/n).float(), target_scale.float())


def all_transitions(store: EpisodeStore, pool: torch.Tensor, chunk: int, device: torch.device):
    for start in range(0, len(pool)*store.steps, chunk):
        i = torch.arange(start, min(start+chunk, len(pool)*store.steps))
        yield store.windows(pool[i//store.steps], i % store.steps, 1, device)


@torch.no_grad()
def ridge_sufficient_statistics(model: ARXDynamics, store: EpisodeStore, pool: torch.Tensor,
                                chunk: int, device: torch.device, on_chunk=None) -> tuple[torch.Tensor, torch.Tensor, int]:
    gram = torch.zeros(662, 662, device=device, dtype=torch.float64)
    cross = torch.zeros(662, 22, device=device, dtype=torch.float64)
    n = 0
    for batch in all_transitions(store, pool, chunk, device):
        x = model.features(batch["history"], batch["valid"], batch["commands"][:, 0]).double()
        x = torch.cat((x, torch.ones_like(x[:, :1])), dim=-1)
        y = torch.cat((batch["target_residual"][:, 0]-batch["history"][:, -1, :21],
                       batch["target_power"][:, :1]), dim=-1)
        y = ((y-model.y_mean)/model.y_scale).double()
        gram += x.T @ x; cross += x.T @ y; n += len(x)
        if on_chunk: on_chunk()
    return gram, cross, n


@torch.no_grad()
def solve_ridge(model: ARXDynamics, stats: tuple, alpha: float) -> ARXDynamics:
    gram, cross, n = stats
    penalty = torch.eye(len(gram), device=gram.device, dtype=gram.dtype)*alpha
    penalty[-1, -1] = 0
    result = deepcopy(model)
    result.weight.copy_(torch.linalg.solve(gram/n+penalty, cross/n).float())
    if not bool(torch.isfinite(result.weight).all()):
        raise RuntimeError("non-finite ridge solution")
    return result
