"""相位补偿网络结构与输出有效性测试。"""

import torch

from src.models.resunet_phase import ResUNetPhase


def test_resunet_phase_preserves_spatial_shape():
    """验证网络输出保持空间尺寸并且不包含 NaN 或 Inf。"""
    model = ResUNetPhase(in_channels=2, base_channels=8)
    x = torch.randn(2, 2, 32, 32)

    y = model(x)

    assert y.shape == (2, 1, 32, 32)
    assert torch.isfinite(y).all()
