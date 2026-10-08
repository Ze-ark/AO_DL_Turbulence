"""CPU小型确定性统计测试，不用于实验排名。"""
import pytest
import torch
from src.rl.r4_amplitude_closed_loop import summarize


def fixture():
    baseline = torch.ones(3, 6, 32, 6, dtype=torch.float64)*.5
    baseline[..., 2] = 0
    means = baseline[None].repeat(2, 1, 1, 1, 1)
    means[0, ..., 0] += .002
    means[1, ..., 0] += .01
    return means, baseline


def test_complete_pair_and_baseline_gates():
    x, b = fixture()
    s = summarize(x, b, seed=1, repeats=100)
    assert s['all_development_gates_pass']
    assert s['relative_power_gain_vs_integrator'] == pytest.approx(.02)
    assert s['mean_metric_deltas'][0][0] == pytest.approx(.008)
    assert s == summarize(x, b, seed=1, repeats=100)
    assert not s['rl_authorized']


@pytest.mark.parametrize('metric,delta', [(0, -.02), (1, -.01), (2, .002)])
def test_any_bad_outcome_blocks(metric, delta):
    x, b = fixture()
    x[1, ..., metric] += delta
    assert not summarize(x, b, seed=1, repeats=100)['all_development_gates_pass']


def test_small_improvement_does_not_lower_old_gate():
    x, b = fixture(); x[1, ..., 0] = .503
    s = summarize(x, b, seed=1, repeats=100)
    assert s['gates']['increment_ci_positive']
    assert not s['gates']['power_vs_integrator_at_least_one_percent']


def test_missing_and_nonfinite_rejected():
    x, b = fixture()
    with pytest.raises(ValueError): summarize(x[:, :, :, :31], b, seed=1, repeats=100)
    x[0, 0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError): summarize(x, b, seed=1, repeats=100)
