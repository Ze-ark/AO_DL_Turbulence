"""GPU 版相位型 SLM 约束和帧延迟模型。"""

from __future__ import annotations

import torch


class TorchSlmModel:
    """把请求相位转为实际加载相位，不隐藏饱和和延迟。"""

    def __init__(
        self,
        phase_min: float,
        phase_max: float,
        max_delta: float,
        quantization_levels: int,
        delay_frames: int,
    ) -> None:
        if phase_max <= phase_min:
            raise ValueError("phase_max must exceed phase_min")
        if max_delta < 0 or quantization_levels < 0 or delay_frames < 0:
            raise ValueError("SLM limits must be non-negative")
        self.phase_min = phase_min
        self.phase_max = phase_max
        self.max_delta = max_delta
        self.quantization_levels = quantization_levels
        self.delay_frames = delay_frames
        self.current_phase: torch.Tensor | None = None
        self.command_queue: torch.Tensor | None = None

    def reset(self, shape: tuple[int, int, int], device: torch.device, dtype: torch.dtype) -> None:
        """清空液晶状态和延迟队列。"""
        self.current_phase = torch.zeros(shape, device=device, dtype=dtype)
        self.command_queue = torch.zeros(
            (self.delay_frames,) + shape,
            device=device,
            dtype=dtype,
        )

    def step(self, requested_phase: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """执行一次相位范围、量化、延迟和单步变化限制。"""
        if self.current_phase is None or self.command_queue is None:
            raise RuntimeError("reset must be called before step")
        if requested_phase.shape != self.current_phase.shape:
            raise ValueError("requested_phase shape does not match SLM state")
        clipped = requested_phase.clamp(self.phase_min, self.phase_max)
        saturated = clipped.ne(requested_phase)
        quantized = self._quantize(clipped)
        delayed = self._apply_delay(quantized)
        requested_delta = delayed - self.current_phase
        limited_delta = requested_delta.clamp(-self.max_delta, self.max_delta)
        self.current_phase = self.current_phase + limited_delta
        diagnostics = {
            "saturated_fraction": saturated.float().mean(dim=(-2, -1)),
            "slew_limited_fraction": limited_delta.ne(requested_delta).float().mean(dim=(-2, -1)),
            "delayed_command": delayed,
        }
        return self.current_phase, diagnostics

    def _quantize(self, phase: torch.Tensor) -> torch.Tensor:
        if self.quantization_levels <= 1:
            return phase
        step = (self.phase_max - self.phase_min) / (self.quantization_levels - 1)
        return self.phase_min + torch.round((phase - self.phase_min) / step) * step

    def _apply_delay(self, command: torch.Tensor) -> torch.Tensor:
        if self.delay_frames == 0:
            return command
        if self.command_queue is None:
            raise RuntimeError("SLM queue has not been initialized")
        delayed = self.command_queue[0]
        self.command_queue = torch.cat((self.command_queue[1:], command.unsqueeze(0)), dim=0)
        return delayed
