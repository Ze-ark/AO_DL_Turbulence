"""小型确定性CPU测试，不评价物理性能。"""
from pathlib import Path
import pytest
import torch
import yaml
from src.rl.r4_baselines import BaselineSpec, baseline_delta, guarded_correction, range_eligible, validate_history
from src.rl.r4_control import NominalCalibration, R4Limits, project_request
from src.rl.r4_observation import R4Interface

def setup():
    s = R4Interface().reset(torch.full((2, 21), .1), episode_id='unit')
    return s.features, s.valid, dict(residual_abs_max=[1.]*21, power_max=1.)

class Toy(torch.nn.Module):
    def forward(self, h, v, command):
        return torch.cat((h[:, -1, :21]*.5, torch.ones(len(h), 1)), -1)

def test_integrator_and_tracking_formula():
    h, v, _ = setup(); h[:, -1, 21:42] = .2; h[:, -1, 42:63] = .1
    a, n = baseline_delta(BaselineSpec('integrator', .25, .1), h, v, [], NominalCalibration())
    b, m = baseline_delta(BaselineSpec('tracking', .25, .1), h, v, [], NominalCalibration())
    torch.testing.assert_close(a, torch.full_like(a, -.045))
    torch.testing.assert_close(b, a-.05)
    assert n == m == 0

def test_ridge_two_step_prediction_and_no_mutation():
    h, v, _ = setup(); before = h.clone()
    delta, calls = baseline_delta(BaselineSpec('ridge', .25, .1), h, v, [Toy().eval()]*3, NominalCalibration())
    torch.testing.assert_close(delta, torch.full_like(delta, -.25*.025))
    assert calls == 12 and torch.equal(h, before)

@pytest.mark.parametrize('fault,reason', [('nan', 'nonfinite_proposal'), ('large', 'proposal_out_of_bounds'), ('range', 'outside_observation_range')])
def test_per_episode_fallback_preserves_other_episode(fault, reason):
    h, v, bounds = setup(); proposal = torch.full((2, 11), .2)
    if fault == 'nan': proposal[0, 0] = float('nan')
    elif fault == 'large': proposal[0, 0] = 2
    else: h[0, -1, 0] = 2
    correction, reasons = guarded_correction(h, v, proposal, bounds)
    assert reasons == [reason, 'accepted']
    assert not correction[0].any() and torch.equal(correction[1], proposal[1])
    base, _ = baseline_delta(BaselineSpec('tracking', .25, .1), h, v, [], NominalCalibration())
    actual = project_request(h[:, -1, 21:42], base, correction, R4Limits())
    zero = project_request(h[:, -1, 21:42], base, torch.zeros_like(correction), R4Limits())
    assert torch.equal(actual.requested_delta_rad[0], zero.requested_delta_rad[0])

@pytest.mark.parametrize('fault', ['nan', 'padding', 'missing', 'prior'])
def test_unreliable_current_state_stops(fault):
    h, v, _ = setup()
    if fault == 'nan': h[0, -1, 0] = float('nan')
    elif fault == 'padding': h[0, 0, 0] = 1
    elif fault == 'missing': v[0, -1] = False
    else: h[0, -1, 21] = 4
    with pytest.raises(ValueError): validate_history(h, v)

def test_missing_power_not_misread_as_measurement():
    h, v, bounds = setup(); h[:, -1, 74] = 2
    assert range_eligible(h, v, bounds).all()
    h[:, -1, 78] = 1
    assert not range_eligible(h, v, bounds).any()

def test_grid_and_budget():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/experiments/s4_r4_baseline_safety_v1.yaml').read_text(encoding='utf-8'))
    grid = cfg['candidate_grid']
    candidates = [BaselineSpec(k,g,l) for k in grid['kinds'] for g in grid['gains'] for l in grid['leaks']]
    assert len(candidates) == len(set(candidates)) == grid['count'] == 18
    for c in candidates: c.validate()
    assert 2*2*12*4 == cfg['physical_transitions']
    assert 2*2*12*128*4*8*3 == cfg['max_mpc_forward_samples']

def test_bad_model_is_not_silently_used():
    h,v,_=setup()
    with pytest.raises(ValueError): baseline_delta(BaselineSpec('ridge',.25,.1), h,v,[Toy()]*3,NominalCalibration())
