import torch

from src.models.resunet_phase import ResUNetPhase


def test_resunet_phase_preserves_spatial_shape():
    model = ResUNetPhase(in_channels=2, base_channels=8)
    x = torch.randn(2, 2, 32, 32)

    y = model(x)

    assert y.shape == (2, 1, 32, 32)
    assert torch.isfinite(y).all()
