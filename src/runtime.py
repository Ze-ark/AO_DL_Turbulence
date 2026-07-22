"""运行设备选择与环境检查工具。"""

from __future__ import annotations

import torch


def resolve_device(requested: str | None) -> torch.device:
    """解析配置中的设备名称，并在 CUDA 不可用时尽早报错。"""
    requested = (requested or "cpu").strip().lower()
    # 配置文件允许使用更直观的 gpu 别名。
    if requested == "gpu":
        requested = "cuda"

    # 当前项目的训练流程仅支持 GPU，避免误用 CPU 导致训练时间过长。
    if requested == "cpu":
        raise RuntimeError("This project is configured for GPU-only training. Set runtime.device to cuda or gpu.")

    # 在构造 torch.device 前检查当前 PyTorch 是否具备 CUDA 支持。
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU training was requested, but this PyTorch installation does not have CUDA enabled. "
            "Install a CUDA-enabled PyTorch build or run on a machine with supported NVIDIA GPU drivers."
        )

    return torch.device(requested)
