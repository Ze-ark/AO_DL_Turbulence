from pathlib import Path
import torch
from scripts.confirm_s4_r5_policy import _interval

def test_confirmation_interval_uses_family_weather_units():
    x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.float64)
    out = _interval(x, 1, 100)
    assert out["mean"] == 3.5
    assert len(out["family_means"]) == 3
