"""小型确定性CPU选择规则检查；不产生正式性能结果。"""
from pathlib import Path
import pytest
import torch
import yaml
from src.rl.r4_baseline_selection import candidates, select_candidate
from src.rl.r4_baselines import BaselineSpec

def inputs():
    means = torch.zeros(18, 3, 2, 2, 6, dtype=torch.float64)
    means[..., 0] = .4; means[..., 2] = .02
    return means, torch.zeros(18, dtype=torch.bool)

def test_safety_excludes_highest_power_and_selects_next():
    m, bad = inputs()
    m[0, ..., 0] = .9; m[0, ..., 2] = .03
    m[1, ..., 0] = .7
    s = select_candidate(m, bad)
    assert s['selected_index'] == 1 and s['eligible'][0] is False
    assert s['status'] == 'BASELINE_SELECTED_REQUIRES_AUDIT'

def test_ties_use_fixed_order_not_reference_favoritism():
    m, bad = inputs()
    assert select_candidate(m, bad)['selected_index'] == 0
    bad[0] = True
    assert select_candidate(m, bad)['selected_index'] == 1

def test_all_weather_and_profiles_are_averaged_not_selected():
    m, bad = inputs(); m[0, 0, 0, 0, 0] = 1.0
    expected = .4+.6/12
    s = select_candidate(m,bad)
    assert s['candidate_means'][0][0] == pytest.approx(expected)
    assert s['selected_index'] == 0

def test_threshold_boundary_and_invalid_reference():
    m,bad=inputs();m[...,2]=0;m[1,...,2]=.001;m[1,...,0]=.8
    assert select_candidate(m,bad)['selected_index'] == 1
    m[1,...,2]=.00100001
    assert select_candidate(m,bad)['selected_index'] == 0
    bad[9]=True
    assert select_candidate(m,bad)['status'] == 'REFERENCE_INVALID'

@pytest.mark.parametrize('fault', ['nan', 'missing', 'empty', 'mask', 'threshold'])
def test_invalid_results_rejected(fault):
    m,bad=inputs();limit=.001
    if fault=='nan': m[0,0,0,0,0]=float('nan')
    elif fault=='missing': m=m[:17]
    elif fault=='empty': m=m[:,:,:0]
    elif fault=='mask': bad=bad.float()
    else: limit=.002
    with pytest.raises(ValueError): select_candidate(m,bad,violation_increase=limit)

def test_frozen_grid_reference_and_budgets():
    root=Path(__file__).resolve().parents[1]
    read=lambda name:yaml.safe_load((root/'configs/experiments'/name).read_text(encoding='utf-8'))
    c=read('s4_r4_baseline_selection_v1.yaml'); grid=read('s4_r4_baseline_safety_v1.yaml')['candidate_grid']
    specs=candidates(grid)
    assert specs[c['reference_index']]==BaselineSpec('tracking',.25,.1)
    assert len(specs)*3*c['per_family']*6*c['steps']==c['physical_transitions']==2073600
    assert sum(s.kind=='ridge' for s in specs)*3*c['per_family']*6*c['steps']*6==c['ridge_forward_samples']==4147200
    q=c['quick']
    assert 18*3*q['per_family']*len(q['profiles'])*q['steps']==2592
    assert 6*3*q['per_family']*len(q['profiles'])*q['steps']*6==5184
