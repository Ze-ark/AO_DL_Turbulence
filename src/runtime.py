from __future__ import annotations

import torch


def resolve_device(requested: str | None) -> torch.device:
    """Resolve a configured GPU device and fail fast when CUDA is unavailable."""
    requested = (requested or "cpu").strip().lower()
    if requested == "gpu":
        requested = "cuda"

    if requested == "cpu":
        raise RuntimeError("This project is configured for GPU-only training. Set runtime.device to cuda or gpu.")

    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU training was requested, but this PyTorch installation does not have CUDA enabled. "
            "Install a CUDA-enabled PyTorch build or run on a machine with supported NVIDIA GPU drivers."
        )

    return torch.device(requested)
