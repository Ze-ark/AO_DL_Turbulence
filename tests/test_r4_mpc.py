"""确定性小尺寸CPU测试；不代表控制性能。"""
import pytest
import torch
from src.rl.r4_control import NominalCalibration
from src.rl.r4_mpc import SearchConfig, plan, sequence_scores
from src.rl.r4_observation import R4Interface

ANCHOR = dict(gain=.25, leak=.1, tracking_gain=.5)

class Toy(torch.nn.Module):
    def __init__(self, flat=False):
        super().__init__(); self.flat = flat
    def forward(self, h, v, command):
        power = command[:, 10:11] if not self.flat else torch.zeros_like(command[:, :1])
        return torch.cat((h[:, -1, :21]*.9, power), -1)

def setup():
    snap = R4Interface().reset(torch.zeros(1, 21), episode_id='unit')
    return snap.features, snap.valid

def test_search_repeatability_cost_and_nonmutation():
    h, v = setup(); before = h.clone(); model = Toy().eval()
    c = SearchConfig(horizon=3, population=8, iterations=2, elites=2)
    a = plan([model]*3, h, v, ANCHOR, NominalCalibration(), c, seed=11)
    b = plan([model]*3, h, v, ANCHOR, NominalCalibration(), c, seed=11)
    assert all(torch.equal(a[k], b[k]) for k in a if torch.is_tensor(a[k]))
    assert torch.equal(h, before) and a['model_forward_samples'] == 144
    assert a['score'] >= a['zero_score'] and a['requested_delta'].abs().max() <= .15
    assert a['correction'].abs().max() <= 1
    assert bool((a['iteration_best'][1:] >= a['iteration_best'][:-1]).all())

def test_flat_score_tie_selects_zero_and_new_call_has_no_warm_state():
    h, v = setup(); c = SearchConfig(horizon=2, population=4, iterations=2, elites=2, action_cost=0)
    for seed in (1, 2, 1):
        result = plan([Toy(True).eval()], h, v, ANCHOR, NominalCalibration(), c, seed=seed)
        assert not result['sequence'].any() and not result['requested_delta'].any()

def test_future_actions_can_change_score():
    h, v = setup(); c = SearchConfig(horizon=2, action_cost=0)
    u = torch.zeros(1, 2, 2, 11); u[:, 1, 1, 0] = 1
    score = sequence_scores([Toy().eval()], h, v, u, ANCHOR, NominalCalibration(), c)
    assert score[0, 1] > score[0, 0]

@pytest.mark.parametrize('fault', ['nan', 'mask', 'prior', 'training'])
def test_invalid_inputs_stop(fault):
    h, v = setup(); model = Toy().eval()
    if fault == 'nan': h[:, -1, 0] = float('nan')
    elif fault == 'mask': v[:, -1] = False
    elif fault == 'prior': h[:, -1, 21] = 4
    else: model.train()
    with pytest.raises(ValueError):
        plan([model], h, v, ANCHOR, NominalCalibration(), SearchConfig(population=4, elites=2), seed=1)

@pytest.mark.parametrize('kwargs', [dict(elites=129), dict(min_std=0), dict(smoothing=1), dict(discount=float('nan'))])
def test_bad_config(kwargs):
    with pytest.raises(ValueError): SearchConfig(**kwargs).validate()

def test_nonfinite_prediction_stops():
    class Bad(Toy):
        def forward(self, h, v, command): return super().forward(h, v, command)*float('nan')
    h, v = setup()
    with pytest.raises(RuntimeError): plan([Bad().eval()], h, v, ANCHOR, NominalCalibration(), SearchConfig(population=4, elites=2), seed=1)
