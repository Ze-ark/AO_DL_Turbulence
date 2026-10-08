"""R5硬前向、代理量化梯度与固定配准伴随；不执行训练或硬件动作。"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from src.rl.r4_baselines import validate_history
from src.simulation.config import S1EnvConfig
from src.simulation.hardware_effects import HardwareEffectsConfig, HardwareProfile


class _Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, phase: torch.Tensor, low: float, high: float, levels: int):
        step = (high - low) / (levels - 1)
        return low + torch.round((phase - low) / step) * step

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        # 仅为代理梯度，绝不是阶梯函数的数学导数。
        return grad, None, None, None


def quantize_st(phase: torch.Tensor, low: float, high: float, levels: int) -> torch.Tensor:
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError("invalid phase range")
    if type(levels) is not int or levels < 0:
        raise ValueError("invalid quantization levels")
    return phase if levels <= 1 else _Quantize.apply(phase, low, high, levels)


class _Registration(torch.autograd.Function):
    @staticmethod
    def forward(ctx, phase: torch.Tensor, grid: torch.Tensor,
                indices: torch.Tensor, weights: torch.Tensor):
        ctx.save_for_backward(indices, weights)
        ctx.shape = phase.shape
        return F.grid_sample(phase[:, None], grid.expand(len(phase), -1, -1, -1),
                             mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0]

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        indices, weights = ctx.saved_tensors
        # 每个输入像素固定顺序收集其贡献，避免CUDA scatter原子加法。
        value = (grad.reshape(len(grad), -1)[:, indices] * weights).sum(-1)
        return value.reshape(ctx.shape), None, None, None


class FixedRegistration:
    """固定标定参数的双线性配准；只对输入相位求导，不学习真实配准。"""

    def __init__(self, reference: torch.Tensor, effects: HardwareEffectsConfig):
        effects.validate()
        _, height, width = reference.shape
        radians = math.radians(effects.rotation_deg)
        cosine, sine = math.cos(radians), math.sin(radians)
        transform = reference.new_tensor([
            [cosine, -sine, -2 * effects.shift_x_pixels / width],
            [sine, cosine, -2 * effects.shift_y_pixels / height],
        ])[None]
        self.grid = F.affine_grid(transform, (1, 1, height, width), align_corners=False)
        x = ((self.grid[0, :, :, 0] + 1) * width - 1) / 2
        y = ((self.grid[0, :, :, 1] + 1) * height - 1) / 2
        x0, y0 = x.floor(), y.floor()
        input_ids, output_ids, contributions = [], [], []
        output = torch.arange(height * width, device=reference.device).reshape(height, width)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            xi, yi = x0 + dx, y0 + dy
            valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
            weight = (1 - (x - xi).abs()) * (1 - (y - yi).abs())
            input_ids.append((yi.long() * width + xi.long())[valid])
            output_ids.append(output[valid])
            contributions.append(weight[valid])
        inputs = torch.cat(input_ids)
        order = torch.argsort(inputs, stable=True)
        inputs = inputs[order]
        outputs = torch.cat(output_ids)[order]
        values = torch.cat(contributions)[order]
        counts = torch.bincount(inputs, minlength=height * width)
        starts = counts.cumsum(0) - counts
        ranks = torch.arange(len(inputs), device=reference.device) - torch.repeat_interleave(starts, counts)
        columns = max(1, int(counts.max()))
        self.indices = torch.zeros(height * width, columns, dtype=torch.long, device=reference.device)
        self.weights = reference.new_zeros(height * width, columns)
        self.indices[inputs, ranks] = outputs
        self.weights[inputs, ranks] = values

    def __call__(self, phase: torch.Tensor) -> torch.Tensor:
        return _Registration.apply(phase, self.grid, self.indices, self.weights)


@dataclass(frozen=True)
class SlmState:
    phase: torch.Tensor
    queue: torch.Tensor


def slm_transition(state: SlmState, request: torch.Tensor, config: S1EnvConfig,
                   effects: HardwareEffectsConfig,
                   registration: FixedRegistration | None) -> tuple[SlmState, dict[str, torch.Tensor]]:
    """函数式SLM转移；旧state不原地修改，跨帧梯度不断开。"""
    if (request.shape != state.phase.shape or request.device != state.phase.device
            or request.dtype != state.phase.dtype or not bool(torch.isfinite(request).all())):
        raise ValueError("invalid SLM request")
    scaled = request * effects.phase_scale
    clipped = scaled.clamp(config.slm_phase_min_rad, config.slm_phase_max_rad)
    quantized = quantize_st(clipped, config.slm_phase_min_rad, config.slm_phase_max_rad,
                           config.slm_quantization_levels)
    if config.slm_delay_frames:
        delayed = state.queue[0]
        queue = torch.cat((state.queue[1:], quantized[None]))
    else:
        delayed, queue = quantized, state.queue
    registered = delayed if registration is None else registration(delayed)
    delta = registered - state.phase
    limited = delta.clamp(-config.slm_max_delta_rad, config.slm_max_delta_rad)
    actual = limited * effects.settling_fraction
    result = SlmState(state.phase + actual, queue)
    return result, {
        "saturated_fraction": clipped.ne(scaled).float().mean((-2, -1)),
        "slew_limited_fraction": limited.ne(delta).float().mean((-2, -1)),
        "settling_limited_fraction": (limited.ne(0) & actual.ne(limited)).float().mean((-2, -1)),
        "delayed_command": delayed,
        "registered_command": registered,
    }


class DifferentiableSlm:
    """仅供仿真注入的兼容封装；reset前不分配设备张量。"""

    def __init__(self, config: S1EnvConfig, effects: HardwareEffectsConfig):
        config.validate()
        effects.validate()
        self.config, self.effects = config, effects
        self.state: SlmState | None = None
        self.registration: FixedRegistration | None = None

    @property
    def current_phase(self) -> torch.Tensor | None:
        return None if self.state is None else self.state.phase

    @property
    def command_queue(self) -> torch.Tensor | None:
        return None if self.state is None else self.state.queue

    def reset(self, shape: tuple[int, int, int], device: torch.device, dtype: torch.dtype) -> None:
        phase = torch.zeros(shape, device=device, dtype=dtype)
        self.state = SlmState(phase, phase.new_zeros((self.config.slm_delay_frames,) + shape))
        e = self.effects
        self.registration = (FixedRegistration(phase, e)
                             if e.shift_x_pixels or e.shift_y_pixels or e.rotation_deg else None)

    def step(self, request: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.state is None:
            raise RuntimeError("SLM requires reset")
        self.state, info = slm_transition(self.state, request, self.config, self.effects, self.registration)
        return self.state.phase, info


class BatchedDifferentiableSlm:
    """批量硬件档位SLM；每个batch样本保留独立延迟、量化和配准参数。"""

    def __init__(self, config: S1EnvConfig, profiles: list[dict], device: torch.device):
        config.validate()
        if not profiles:
            raise ValueError("profiles must not be empty")
        self.config = config
        self.device = device
        self.profiles = [HardwareProfile.from_mapping(dict(p, id=p.get('identifier', p.get('id', str(i)))))
                         for i, p in enumerate(profiles)]
        self.phase_scale = torch.tensor([p["phase_scale"] for p in profiles], device=device)
        self.settling = torch.tensor([p["settling_fraction"] for p in profiles], device=device)
        self.levels = torch.tensor([p["slm_quantization_levels"] for p in profiles], device=device, dtype=torch.float32)
        self.delays = torch.tensor([p["slm_delay_frames"] for p in profiles], device=device, dtype=torch.long)
        self.shift_x = torch.tensor([p["shift_x_pixels"] for p in profiles], device=device)
        self.shift_y = torch.tensor([p["shift_y_pixels"] for p in profiles], device=device)
        self.rotation = torch.tensor([p["rotation_deg"] for p in profiles], device=device)
        self.state: SlmState | None = None
        self.grids: torch.Tensor | None = None

    @property
    def current_phase(self):
        return None if self.state is None else self.state.phase

    def reset(self, shape, device, dtype):
        batch, height, width = shape
        if batch != len(self.profiles):
            raise ValueError('one hardware profile per sample required')
        max_delay = int(self.delays.max().item())
        phase = torch.zeros(shape, device=device, dtype=dtype)
        self.state = SlmState(phase, phase.new_zeros((max_delay, batch, height, width)))
        # 相同参数连续分组；复用旧模型的硬前向和确定性配准伴随。
        self.groups = []
        start = 0
        while start < batch:
            profile = self.profiles[start]
            stop = start + 1
            while stop < batch and self.profiles[stop] == profile:
                stop += 1
            effects = profile.effects_config()
            registration = (FixedRegistration(phase[start:stop], effects)
                            if effects.shift_x_pixels or effects.shift_y_pixels or effects.rotation_deg else None)
            self.groups.append((slice(start, stop), profile, registration))
            start = stop
        self._delay_indices = (max_delay - self.delays).clamp(0, max(0, max_delay - 1))
        self._batch_indices = torch.arange(batch, device=device)
        self._zero_delay = (self.delays == 0)[:, None, None]
        self._settling = phase.new_tensor([p.settling_fraction for p in self.profiles])[:, None, None]
        self._slew = phase.new_tensor([p.slm_max_delta_rad for p in self.profiles])[:, None, None]

    def step(self, request):
        if self.state is None:
            raise RuntimeError("SLM requires reset")
        if (request.shape != self.state.phase.shape or request.device != self.state.phase.device
                or request.dtype != self.state.phase.dtype or not bool(torch.isfinite(request).all())):
            raise ValueError('invalid batched SLM request')
        b = request.shape[0]
        scaled = torch.cat([request[part] * p.phase_scale for part, p, _ in self.groups])
        clipped = scaled.clamp(self.config.slm_phase_min_rad, self.config.slm_phase_max_rad)
        quantized = torch.cat([quantize_st(clipped[part], self.config.slm_phase_min_rad,
                              self.config.slm_phase_max_rad, p.slm_quantization_levels)
                              for part, p, _ in self.groups])
        max_delay = self.state.queue.shape[0]
        if max_delay:
            delayed = torch.where(self._zero_delay, quantized,
                                  self.state.queue[self._delay_indices, self._batch_indices])
            queue = torch.cat((self.state.queue[1:], quantized.unsqueeze(0)), dim=0)
        else:
            delayed, queue = quantized, self.state.queue
        registered = torch.cat([delayed[part] if reg is None else reg(delayed[part])
                                for part, _, reg in self.groups])
        delta = registered - self.state.phase
        limited = delta.clamp(-self._slew, self._slew)
        actual = limited * self._settling
        phase = self.state.phase + actual
        self.state = SlmState(phase, queue)
        return phase, {
            "saturated_fraction": clipped.ne(scaled).float().mean((-2, -1)),
            "slew_limited_fraction": limited.ne(delta).float().mean((-2, -1)),
            "settling_limited_fraction": (limited.ne(0) & actual.ne(limited)).float().mean((-2, -1)),
            "delayed_command": delayed,
            "registered_command": registered,
        }


def causal_policy_features(history: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """只接受白名单张量，不接受env/info；用伪开环模态替换原残余。"""
    validate_history(history, valid)
    return torch.cat((history[..., :21] - history[..., 42:63], history[..., 21:]), -1)


def measured_objective(power: torch.Tensor, correction: torch.Tensor,
                       previous: torch.Tensor, action_weight: float, smooth_weight: float) -> torch.Tensor:
    if (correction.shape != previous.shape or correction.shape != (len(power), 11)
            or action_weight < 0 or smooth_weight < 0):
        raise ValueError("invalid objective contract")
    return (power - action_weight * correction.square().mean(-1)
            - smooth_weight * (correction - previous).square().mean(-1))
