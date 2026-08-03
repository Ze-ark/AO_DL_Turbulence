"""在 IDE 终端中显示统一的训练进度、损失、剩余时间和显存占用。"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sized
from typing import Any, TypeVar

import torch
from tqdm.auto import tqdm


T = TypeVar("T")


def progress_bar(
    iterable: Iterable[T],
    *,
    description: str,
    unit: str,
    leave: bool = True,
) -> tqdm[T]:
    """创建适合 PyCharm、VS Code 和普通终端的动态进度条。"""
    total = len(iterable) if isinstance(iterable, Sized) else None
    return tqdm(
        iterable,
        total=total,
        desc=description,
        unit=unit,
        dynamic_ncols=True,
        leave=leave,
        mininterval=0.2,
    )


def update_progress(
    bar: tqdm[Any],
    *,
    device: torch.device,
    metrics: dict[str, float],
) -> None:
    """刷新当前平均指标；CUDA 可用时同时显示已分配和已保留显存。"""
    postfix = {name: f"{value:.5f}" for name, value in metrics.items()}
    memory = gpu_memory_status(device)
    if memory:
        postfix["显存"] = memory
    bar.set_postfix(postfix, refresh=False)


def gpu_memory_status(device: torch.device) -> str:
    """返回紧凑的 CUDA 显存状态；CPU 测试返回空字符串。"""
    if device.type != "cuda" or not torch.cuda.is_available():
        return ""
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    return f"{allocated:.2f}/{reserved:.2f}GB"


def progress_message(message: str) -> None:
    """在不破坏活动进度条的情况下打印阶段或检查点信息。"""
    tqdm.write(message)


def counted_progress(
    *,
    total: int,
    description: str,
    unit: str,
) -> tqdm[None]:
    """为通过回调更新的任务创建计数型进度条。"""
    return tqdm(
        total=total,
        desc=description,
        unit=unit,
        dynamic_ncols=True,
        leave=True,
        mininterval=0.2,
    )


def advance_to(bar: tqdm[Any], completed: int) -> None:
    """把回调式进度条推进到绝对计数，避免重复累计。"""
    delta = completed - int(bar.n)
    if delta > 0:
        bar.update(delta)


def iter_progress(bar: tqdm[T]) -> Iterator[T]:
    """为类型检查器提供显式迭代入口。"""
    yield from bar
