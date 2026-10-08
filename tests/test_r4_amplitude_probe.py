"""仅CPU小型确定性单元测试；不生成实验性能结果。"""
import pytest
import torch
from src.rl.r4_amplitude_probe import budget, scaled_action, require_close, summarize, predict
from src.rl.r4_control import NominalCalibration


def test_budget():
    assert budget(False) == dict(physical_transitions=177408, model_forward_samples=221184)
    assert budget(True) == dict(physical_transitions=1664, model_forward_samples=3072)


@pytest.mark.parametrize('multiplier', [0., .5, 1., 1.5])
def test_scaled_action(multiplier):
    x = torch.linspace(-1, 1, 11)[None]
    before = x.clone()
    actual = scaled_action(x, multiplier)
    assert torch.equal(actual, (x*multiplier).clamp(-1, 1))
    assert torch.equal(x, before)
    assert actual.abs().max() <= 1


@pytest.mark.parametrize('x,m', [(torch.zeros(2, 10), 1.), (torch.full((2, 11), float('nan')), 1.),
                                 (torch.full((2, 11), 1.01), .5), (torch.zeros(2, 11), 2.)])
def test_reject_bad_actions(x, m):
    with pytest.raises(ValueError):
        scaled_action(x, m)


def test_alignment_is_fail_closed():
    require_close(torch.zeros(2), torch.zeros(2), 'match')
    for x in (torch.ones(2), torch.zeros(3), torch.full((2,), float('nan'))):
        with pytest.raises(RuntimeError):
            require_close(x, torch.zeros(2), 'mismatch')


def test_paired_weather_summary():
    x = torch.zeros(3, 6, 32, 4, 4, 4)
    x[..., 0] = torch.tensor([0., 1., 2., 3.])[None, None, None, None, :]
    result = summarize(x, seed=1, repeats=100)
    assert result['independent_weather'] == 96
    for row, delta in zip(result['comparisons'], (1., -1., 2.)):
        assert row['mean_metric_delta'][0] == delta
        assert row['power_ci_bonferroni'] == [delta, delta]
    assert result == summarize(x, seed=1, repeats=100)
    with pytest.raises(ValueError):
        summarize(x[:, :, :31], seed=1, repeats=100)


def test_prediction_does_not_receive_future_truth_or_mutate_history():
    class Model:
        def __call__(self, h, v, action):
            return h.new_zeros(len(h), 22)
    h = torch.zeros(2, 8, 79); v = torch.ones(2, 8, dtype=torch.bool)
    old = h.clone()
    result = predict([Model()]*3, h, v, torch.zeros(2, 11),
                     dict(gain=.15, leak=.1, tracking_gain=0.), NominalCalibration(), 8)
    assert result.shape == (2, 3, 8)
    assert torch.equal(h, old)
    assert torch.isfinite(result).all()
