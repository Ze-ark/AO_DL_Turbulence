"""运行设备解析逻辑的单元测试。"""

import torch

from src.runtime import resolve_device


import pytest


def test_resolve_device_rejects_cuda_when_unavailable(monkeypatch):
    """验证 CUDA 不可用时请求 GPU 会立即报错。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="GPU training was requested"):
        resolve_device("cuda")



def test_resolve_device_accepts_gpu_alias(monkeypatch):
    """验证 gpu 别名可正确解析为 CUDA 设备。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    device = resolve_device("gpu")

    assert device.type == "cuda"
