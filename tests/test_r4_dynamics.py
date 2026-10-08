"""小尺寸确定性CPU单元测试；不作为模型性能证据。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from src.rl.r4_control import NominalCalibration
from src.rl.r4_dynamics import (ARXDynamics, ResidualGRUDynamics, advance_history, causal_features,
    normalized_errors, ridge_sufficient_statistics, rollout, solve_ridge, training_statistics)
from src.rl.r4_dynamics_experiment import settings, weather_seeds, preflight, evaluate
from src.rl.r4_observation import R4Interface, PowerMeasurement
from src.rl.r4_trajectory import EpisodeStore, anchor_delta, collect_episode_batch
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.simulation.config import S1EnvConfig
from src.simulation.hardware_effects import HardwareProfile

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return yaml.safe_load((ROOT/'configs/experiments/s4_r4_dynamics_v1.yaml').read_text(encoding='utf-8'))


def fixture_store() -> EpisodeStore:
    frames = torch.zeros(4, 13, 79)
    frames[:, :, :21] = torch.arange(13)[None, :, None]*.01
    frames[:, :, 74] = .5
    frames[:, :, 75] = torch.arange(13)
    frames[:, :, 76:78] = (torch.arange(13)-1)[None, :, None]
    frames[:, 1:, 78] = 1
    return EpisodeStore(dict(frames=frames, commands=torch.zeros(4, 12, 21),
        corrections=torch.zeros(4, 12, 11), powers=torch.ones(4, 12)*.5,
        weather_seeds=torch.tensor([10, 10, 20, 20])))


@pytest.mark.parametrize('delay,settling', [(0,1.), (2,.5), (3,.5)])
def test_model_history_matches_causal_interface(delay, settling):
    cal=NominalCalibration(delay, settling)
    a=R4Interface(calibration=cal); z=torch.zeros(2,21)
    h=a.reset(z,episode_id='unit')
    for t in range(10):
        action=a.issue(z,torch.ones(2,11)*.2,step=t)
        pred=torch.cat((torch.ones_like(z)*(t+1),torch.ones(2,1)*.7),dim=-1)
        next_h,next_v=advance_history(h.features,h.valid,action.requested_delta_rad,action.normalized_correction,pred,cal)
        tr=a.observe_next(pred[:,:21],step=t+1,power=PowerMeasurement(pred[:,21],t,t+1))
        torch.testing.assert_close(next_h,tr.next_history.features,rtol=0,atol=0)
        assert torch.equal(next_v,tr.next_history.valid)
        h=tr.next_history


def test_windows_never_cross_boundaries_and_preserve_last_observation():
    store=fixture_store()
    b=store.windows(torch.tensor([0,3]),torch.tensor([0,4]),8,torch.device('cpu'))
    assert b['valid'].sum(1).tolist()==[1,5]
    assert not b['history'][0,:-1].any()
    torch.testing.assert_close(b['target_residual'][1,-1],store.data['frames'][3,-1,:21])
    with pytest.raises(ValueError):store.windows(torch.tensor([0]),torch.tensor([5]),8,torch.device('cpu'))
    with pytest.raises(ValueError):store.windows(torch.tensor([-1]),torch.tensor([0]),8,torch.device('cpu'))


def test_whitelist_ignores_poisoned_audit_and_anchor_uses_estimate_only():
    store=fixture_store(); data=dict(store.data, audit_applied_modal=torch.full((4,13,21),float('nan')))
    assert 'audit_applied_modal' not in EpisodeStore(data).data
    frame=store.data['frames'][:,0].clone(); frame[:,42:63]=.2
    delta=anchor_delta(frame,dict(gain=.25,leak=.1,tracking_gain=.5))
    torch.testing.assert_close(delta,torch.ones_like(delta)*.1)


def test_weather_bootstrap_keeps_all_branches():
    s=fixture_store(); pool=s.bootstrap_pool(13)
    counts=torch.bincount(pool,minlength=4)
    assert counts[0]==counts[1] and counts[2]==counts[3] and len(pool)==4


def test_relative_timestamp_features_do_not_leak_absolute_episode_time():
    b=fixture_store().windows(torch.tensor([0]),torch.tensor([7]),1,torch.device('cpu'))
    other=b['history'].clone();other[:,:,75:78]+=100
    torch.testing.assert_close(causal_features(b['history'],b['valid']),causal_features(other,b['valid']))


def unit_linear():
    m=ARXDynamics(torch.zeros(661),torch.ones(661),torch.zeros(22),torch.ones(22))
    m.weight[640:661,:21]=torch.eye(21)*.2
    m.weight[-1,21]=.5
    return m


def test_gru_starts_exactly_at_linear_and_freezes_linear_parameters():
    torch.manual_seed(1); linear=unit_linear(); model=ResidualGRUDynamics(linear,8)
    b=fixture_store().windows(torch.tensor([0]),torch.tensor([0]),1,torch.device('cpu'))
    x=(b['history'],b['valid'],b['commands'][:,0])
    torch.testing.assert_close(model(*x),linear(*x),rtol=0,atol=0)
    assert all(not p.requires_grad for p in model.linear.parameters())
    model(*x).sum().backward()
    assert model.head.weight.grad is not None


def test_rollout_action_gradients_and_future_action_causality():
    b=fixture_store().windows(torch.tensor([0]),torch.tensor([0]),8,torch.device('cpu'))
    commands=b['commands'].clone().requires_grad_()
    pred=rollout(unit_linear(),b['history'],b['valid'],commands,b['corrections'],NominalCalibration())
    pred[:,-1,:21].sum().backward()
    assert commands.grad[:,0].abs().sum()>0
    changed=commands.detach().clone();changed[:,-1]=1
    later=rollout(unit_linear(),b['history'],b['valid'],changed,b['corrections'],NominalCalibration())
    torch.testing.assert_close(pred[:,:-1],later[:,:-1],rtol=0,atol=0)
    assert not torch.equal(pred[:,-1],later[:,-1])


def test_loss_modal_and_power_equal_weight_and_requested_horizons():
    b=fixture_store().windows(torch.tensor([0]),torch.tensor([0]),8,torch.device('cpu'))
    pred=torch.cat((b['target_residual'],b['target_power'][:,:,None]),dim=-1)
    pred[:,:,21]+=2
    err=normalized_errors(pred,b,torch.ones(22),[1,2,4,8])
    assert err.shape==(1,4,2) and err.mean()==2


def test_ridge_train_only_statistics_and_small_fit(cfg):
    s=fixture_store(); device=torch.device('cpu')
    model=training_statistics(s,cfg['model'],device)
    assert model.y_scale.min()>0
    stats=ridge_sufficient_statistics(model,s,torch.arange(s.count),16,device)
    fitted=solve_ridge(model,stats,.01)
    b=s.windows(torch.tensor([0]),torch.tensor([3]),1,device)
    pred=fitted(b['history'],b['valid'],b['commands'][:,0])
    torch.testing.assert_close(pred[:,:21],b['target_residual'][:,0],atol=1e-5,rtol=0)
    assert torch.isfinite(fitted.weight).all()


def test_formal_quick_seeds_and_budgets(cfg):
    groups=[]
    for quick in (False,True):
        c=settings(cfg,quick)
        for split in ('train','development'):
            seeds={x for f in range(3) for x in weather_seeds(c,split,f)}
            assert not any(seeds & g for g in groups)
            assert all(x//10000==376 for x in seeds)
            groups.append(seeds)
    assert list(map(len,groups))==[384,96,6,6]
    assert len(groups[0])*6*2*200==921600
    assert len(groups[1])*6*2*200==230400


def test_cpu_formal_config_rejected_before_outputs(cfg,tmp_path):
    cfg['runtime']['device']='cpu';p=tmp_path/'bad.yaml';p.write_text(yaml.safe_dump(cfg),encoding='utf-8')
    with pytest.raises(ValueError,match='CUDA'):preflight(p,False)


def test_persistence_missing_initial_power_uses_training_mean_not_padding(cfg):
    s=fixture_store();s.data['frames'][:,0,74]=0
    c=settings(cfg,True);c['training']['evaluation_starts']=[0]
    result=evaluate(None,s,c,torch.device('cpu'),torch.ones(22),initial_power_mean=.5)
    assert all(row[1]==0 for row in result['per_horizon'])
    with pytest.raises(ValueError,match='training-only'):
        evaluate(None,s,c,torch.device('cpu'),torch.ones(22))


def test_small_collection_final_frame_and_corruption_rejected(cfg,tmp_path):
    config=S1EnvConfig(grid_size=16,batch_size=2,num_modes=21,episode_length=12)
    basis,_,_=build_action_basis(config,ActionRepresentation('unit','zernike',21),torch.device('cpu'))
    data,audit=collect_episode_batch(config,HardwareProfile('unit','unit','unit',True),basis,23,
                                    'bounded_excitation',cfg,lambda:None)
    assert data['frames'].shape==(2,13,79)
    assert 'applied_modal' not in data and 'applied_modal' in audit
    path=tmp_path/'one.pt';torch.save(data,path)
    s=EpisodeStore.from_files([path]);assert s.count==2
    data['powers'][0,0]+=1;torch.save(data,path)
    with pytest.raises(ValueError,match='alignment'):EpisodeStore.from_files([path])
