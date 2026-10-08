"""小型CPU契约测试，不生成性能结果。"""
import pytest
import torch
from src.rl.r4_selected_smoke import correction_for, validate_config
from src.rl.r4_selected_anchor import selected_request
from src.rl.r4_baselines import BaselineSpec
from src.rl.r4_control import NominalCalibration
from src.rl.r4_observation import R4Interface
from src.rl.s4_training import _load_yaml, _project_path


def test_frozen_budget():
    cfg = _load_yaml(_project_path('configs/experiments/s4_r4_selected_smoke_v1.yaml'))
    validate_config(cfg)
    assert cfg['physical_transitions'] == 3*3*2*12
    assert cfg['max_mpc_forward_samples'] == 3*2*2*12*128*4*8*3
    cfg['steps'] = 200
    with pytest.raises(ValueError): validate_config(cfg)


@pytest.mark.parametrize('controller', ['linear_mpc', 'gru_mpc'])
def test_range_fallback_has_no_model_calls_and_keeps_integrator(controller):
    snap = R4Interface().reset(torch.ones(2,21), episode_id='unit')
    h,v = snap.features,snap.valid
    spec = BaselineSpec('integrator', .15, .1)
    u,reasons,score,valid,n = correction_for(controller, [], h,v,spec,NominalCalibration(),
        dict(residual_abs_max=[.1]*21,power_max=1.), 1)
    assert n == 0 and not valid.any() and not u.any()
    assert reasons == ['outside_observation_range']*2
    assert torch.isnan(score).all()
    assert selected_request(spec,h,v,u).requested_delta_rad.abs().max()>0


def test_invalid_state_stops_before_fallback():
    snap = R4Interface().reset(torch.ones(2,21), episode_id='unit')
    snap.features[0,-1,0] = float('nan')
    with pytest.raises(ValueError):
        correction_for('gru_mpc', [], snap.features,snap.valid,BaselineSpec('integrator',.15,.1),
            NominalCalibration(),dict(residual_abs_max=[1.]*21,power_max=1.),1)
