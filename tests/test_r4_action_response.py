"""小尺寸确定性CPU测试，不生成正式动作排名。"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from src.rl.r4_action_response import (direction_statistics, model_branch, noise_threshold, physical_branch, pulse)
from src.rl.r4_control import NominalCalibration
from src.rl.r4_dynamics import ARXDynamics
from src.rl.r4_observation import R4Interface
from src.rl.r4_response_experiment import budget,seeds_for,preflight
from src.rl.s4_representation_capacity import ActionRepresentation,build_action_basis
from src.simulation.config import S1EnvConfig
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.hardware_effects import HardwareProfile

ROOT=Path(__file__).resolve().parents[1]


def test_budget_and_explicit_training_development_reuse():
    c=yaml.safe_load((ROOT/'configs/experiments/s4_r4_action_response_v1.yaml').read_text())
    p=yaml.safe_load((ROOT/'configs/experiments/s4_r4_dynamics_v1.yaml').read_text(encoding='utf-8'))
    b=budget(c,False)
    assert b==dict(calibration_weather=48,development_weather=96,calibration_transitions=59904,
        development_transitions=410112,action_pairs=38016,model_forward_samples=1907712)
    groups=[]
    for q in (False,True):
        for phase in ('calibration','development'):
            group={x for f in range(3) for x in seeds_for(c,p,phase,f,q)}
            assert not any(group&old for old in groups);groups.append(group)
    assert list(map(len,groups))==[48,96,6,6]


def test_pulse_one_step_and_invalid_mode():
    assert pulse(2,10,1.,0,torch.device('cpu'))[:,10].eq(1).all()
    assert not pulse(2,10,1.,1,torch.device('cpu')).any()
    with pytest.raises(ValueError):pulse(2,11,1.,0,torch.device('cpu'))


def test_pair_branches_do_not_mutate_original_and_repeat_only_changes_measurement():
    cfg=S1EnvConfig(grid_size=16,num_modes=21,batch_size=2,episode_length=16)
    basis,_,_=build_action_basis(cfg,ActionRepresentation('unit','zernike',21),torch.device('cpu'))
    profile=HardwareProfile('unit','unit','unit',True,power_noise_relative_std=.01)
    env=AdaptiveOpticsEnv(profile.environment_config(cfg),'cpu',profile.effects_config(),basis_override=basis)
    obs,_=env.reset(seed=29);a=R4Interface();a.reset(obs[:,:21],episode_id='unit')
    sensor=torch.Generator().manual_seed(59);before=deepcopy(env);history=a.snapshot().features.clone()
    anchor=dict(gain=.25,leak=.1,tracking_gain=.5)
    zero=physical_branch(env,a,sensor,profile,anchor,0,0.,8,lambda:None)
    same=physical_branch(env,a,sensor,profile,anchor,0,0.,8,lambda:None)
    repeat=physical_branch(env,a,sensor,profile,anchor,0,0.,8,lambda:None,measurement_seed=77)
    assert all(torch.equal(zero[k],same[k]) for k in zero)
    assert torch.equal(zero['audit_power'],repeat['audit_power'])
    assert torch.equal(zero['requested_delta'],repeat['requested_delta'])
    assert not torch.equal(zero['power'],repeat['power'])
    assert env.step_count==before.step_count==0 and a.step==0
    assert torch.equal(env.turbulence_phase,before.turbulence_phase)
    assert torch.equal(history,a.snapshot().features)


def test_model_uses_its_own_causal_continuation_and_projection():
    a=R4Interface();h=a.reset(torch.ones(2,21)*.1,episode_id='unit')
    linear=ARXDynamics(torch.zeros(661),torch.ones(661),torch.zeros(22),torch.ones(22))
    linear.weight[-1,21]=.6
    linear.weight[640:661,:21]=torch.eye(21)
    anchor=dict(gain=.25,leak=.1,tracking_gain=.5)
    result=model_branch(linear,h.features,h.valid,anchor,NominalCalibration(),3,1.,8)
    assert result['power'].shape==(2,8) and result['requested_delta'].abs().max()<=.15
    assert not torch.equal(result['requested_delta'][:,0],result['requested_delta'][:,1])
    assert torch.equal(h.features,a.snapshot().features)


def test_threshold_is_from_repeated_difference_std_with_floor():
    assert noise_threshold(torch.zeros(9),3.,1e-7)==1e-7
    assert noise_threshold(torch.tensor([-1.,0.,1.]),3.,1e-7)==3.
    with pytest.raises(ValueError):noise_threshold(torch.tensor([1.]),3.,1e-7)


@pytest.mark.parametrize('kind,expected',[('correct',1.),('reverse',0.),('ties',0.)])
def test_weather_cluster_accuracy_and_ties(kind,expected):
    weather=torch.arange(60).repeat_interleave(2);family=weather//20
    actual=torch.tensor([-1.,1.]).repeat(60)
    prediction=actual if kind=='correct' else (-actual if kind=='reverse' else torch.zeros_like(actual))
    r=direction_statistics(actual,prediction,torch.ones_like(actual)*.1,weather,family,seed=3,replicates=100)
    assert r['balanced_accuracy']==expected and r['ci95']==[expected,expected]
    assert r['status']==('ACTION_DIRECTION_PASS' if kind=='correct' else 'ACTION_DIRECTION_FAIL')
    assert r['identifiable_weather']==60


def test_signal_insufficient_and_invalid_vectors_fail_closed():
    x=torch.ones(6);w=torch.arange(6);f=torch.zeros(6,dtype=torch.long)
    r=direction_statistics(x,x,x*2,w,f,seed=3,replicates=10)
    assert r['status']=='INSUFFICIENT_SIGNAL' and r['balanced_accuracy'] is None
    with pytest.raises(ValueError):direction_statistics(x*float('nan'),x,x,w,f,seed=3,replicates=10)


def test_cpu_config_rejected_before_output(tmp_path):
    p=tmp_path/'bad.yaml';p.write_text(yaml.safe_dump(dict(stage='S4-D2-R4-1B1',runtime=dict(device='cpu'))))
    with pytest.raises(ValueError,match='CUDA'):preflight(p,False)
