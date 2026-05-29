import torch

from src.runtime import resolve_device


import pytest


def test_resolve_device_rejects_cuda_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="GPU training was requested"):
        resolve_device("cuda")



def test_resolve_device_accepts_gpu_alias(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    device = resolve_device("gpu")

    assert device.type == "cuda"
