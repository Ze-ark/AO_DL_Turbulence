"""小型确定性CPU单元测试，不产生科学性能结果。"""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from src.rl.r4_action_response import model_branch
from src.rl.r4_control import NominalCalibration
from src.rl.r4_delta_learning import PairStore, accuracy_gain, paired_losses, pulse_rollout
from src.rl.r4_delta_experiment import settings
from src.rl.r4_dynamics import ARXDynamics, ResidualGRUDynamics
from src.rl.r4_observation import R4Interface

CPU = torch.device('cpu')
ANCHOR = dict(gain=.25, leak=.1, tracking_gain=.5)


def record(split='train'):
    corrections = torch.zeros(2, 11); corrections[:, 0] = torch.tensor([-1., 1.])
    return dict(split=split, source='simulation_residual_proxy_not_holography',
        history=torch.zeros(2, 8, 79), valid=torch.ones(2, 8, dtype=torch.bool),
        corrections=corrections, powers=torch.ones(2, 2, 8), zero_power=torch.zeros(2, 8),
        weather=torch.tensor([11, 12]), family=0, profile='unit', probe=0,
        actual_future_commands=torch.full((2, 8, 21), float('nan')))


def test_rollout_matches_original_causal_model_branch_and_preserves_history():
    linear = ARXDynamics(torch.zeros(661), torch.ones(661), torch.zeros(22), torch.ones(22))
    linear.weight[640:661, :21] = torch.eye(21)*.1
    linear.weight[650, 21] = .2
    snap = R4Interface().reset(torch.ones(2, 21)*.1, episode_id='unit')
    before = snap.features.clone()
    correction = torch.zeros(2, 11); correction[:, 0] = 1
    actual = pulse_rollout(linear, snap.features, snap.valid, correction, ANCHOR, NominalCalibration())
    expected = model_branch(linear, snap.features, snap.valid, ANCHOR, NominalCalibration(), 0, 1., 8)
    torch.testing.assert_close(actual, expected['power'], rtol=0, atol=0)
    assert torch.equal(before, snap.features)


def test_delta_gradients_reach_gru_and_head_but_not_frozen_linear():
    torch.manual_seed(100)
    linear = ARXDynamics(torch.zeros(661), torch.ones(661), torch.zeros(22), torch.ones(22))
    model = ResidualGRUDynamics(linear, hidden_size=4)
    torch.nn.init.normal_(model.head.weight, std=.01)
    frozen = deepcopy(model.linear.state_dict())
    snap = R4Interface().reset(torch.ones(2, 21)*.1, episode_id='unit')
    correction = torch.zeros(2, 11); correction[:, 0] = 1
    p = pulse_rollout(model, snap.features, snap.valid, correction, ANCHOR, NominalCalibration())
    z = pulse_rollout(model, snap.features, snap.valid, torch.zeros_like(correction), ANCHOR, NominalCalibration())
    _, loss = paired_losses(p, z, torch.ones_like(p)*.1, torch.zeros_like(z), torch.tensor(1.), torch.tensor(.1))
    loss.backward()
    assert sum(float(x.grad.abs().sum()) for x in model.gru.parameters()) > 0
    assert sum(float(x.grad.abs().sum()) for x in model.head.parameters()) > 0
    assert all(torch.equal(v, frozen[k]) for k, v in model.linear.state_dict().items())
    assert all(p.grad is None for p in model.linear.parameters())


def test_pair_store_whitelist_scale_and_identical_sampling():
    store = PairStore([record()], 'train')
    assert store.count == 4
    assert 'actual_future_commands' not in store.records[0]
    assert float(store.delta_scale(CPU, 1e-5)) == 1
    pool = store.pool(42)
    a = store.sample(pool, 8, torch.Generator().manual_seed(73), CPU)
    b = store.sample(store.pool(42), 8, torch.Generator().manual_seed(73), CPU)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert set(a) == {'history', 'valid', 'correction', 'power', 'zero_power'}


@pytest.mark.parametrize('operation', ['scale', 'pool', 'sample'])
def test_development_never_enters_training(operation):
    store = PairStore([record('development')], 'development')
    with pytest.raises(ValueError):
        if operation == 'scale': store.delta_scale(CPU, 1e-5)
        elif operation == 'pool': store.pool(1)
        else: store.sample(torch.arange(4), 1, torch.Generator(), CPU)


@pytest.mark.parametrize('fault', ['split', 'source', 'shape', 'nan'])
def test_invalid_pair_rejected(fault):
    r = record()
    if fault == 'split': r['split'] = 'development'
    elif fault == 'source': r['source'] = 'true_holography'
    elif fault == 'shape': r['history'] = torch.zeros(2, 7, 79)
    else: r['powers'][0, 0, 0] = float('nan')
    with pytest.raises(ValueError): PairStore([r], 'train')


def test_analytic_loss_and_zero_scale_rejection():
    a, z = torch.ones(2, 8), torch.zeros(2, 8)
    absolute, delta = paired_losses(a, z, z, z, torch.tensor(2.), torch.tensor(.5))
    assert float(absolute) == .125 and float(delta) == 4
    assert paired_losses(a, a, z, z, torch.tensor(1.), torch.tensor(1.))[1] == 0
    with pytest.raises(ValueError): paired_losses(a, z, z, z, torch.tensor(0.), torch.tensor(1.))


def test_weather_paired_accuracy_gain_and_replication():
    actual = torch.tensor([1., -1.]*6)
    weather = torch.arange(6).repeat_interleave(2)
    family = weather // 2
    def result(repeats):
        return accuracy_gain(actual.repeat(repeats), actual.repeat(repeats), -actual.repeat(repeats),
            torch.zeros_like(actual).repeat(repeats), weather.repeat(repeats), family.repeat(repeats), 12, 100)
    assert result(1) == result(2)
    assert result(1)['difference'] == 1 and result(1)['ci95'] == [1., 1.]
    family[1] = 2
    with pytest.raises(ValueError): accuracy_gain(actual, actual, -actual, actual*0, weather, family, 12, 100)


def test_no_identifiable_signal_is_not_a_pass():
    z = torch.zeros(4)
    r = accuracy_gain(z, z, z, z, torch.arange(4), torch.zeros(4), 1, 10)
    assert r['status'] == 'INSUFFICIENT_SIGNAL' and r['ci95'] is None


@pytest.mark.parametrize('quick,expected', [(False, (410112, 38016, 9216000)), (True, (504, 48, 576))])
def test_declared_budget(quick, expected):
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root/'configs/experiments/s4_r4_delta_supervision_v1.yaml').read_text(encoding='utf-8'))
    c = settings(cfg, quick)
    n = 3*c['per_family']*len(c['profiles'])
    values = (n*(max(c['probes'])+len(c['probes'])*(1+2*len(c['modes']))*8),
              n*len(c['probes'])*2*len(c['modes']),
              6*c['updates']*8*(c['trajectory_batch']+2*c['pair_batch']))
    assert values == expected
