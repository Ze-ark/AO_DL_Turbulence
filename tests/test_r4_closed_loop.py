"""小型确定性CPU统计单元测试；不评价真实实验性能。"""
import pytest
import torch
from src.rl.r4_closed_loop import summarize

def fixture():
    x=torch.ones(3,3,6,32,6,dtype=torch.float64)*.5
    x[:,:,:,:,2]=0
    return x

def test_positive_paired_effect_passes_and_is_repeatable():
    x=fixture(); x[2,:,:,:,0]+=.02
    a=summarize(x,seed=12,repeats=100)
    assert a==summarize(x,seed=12,repeats=100)
    assert all(a['gates'].values()) and a['relative_power_gain']==pytest.approx(.04)

@pytest.mark.parametrize('metric,delta',[(0,0.),(1,-.01),(2,.002)])
def test_each_gate_can_fail(metric,delta):
    x=fixture(); x[2,:,:,:,0]+=.02
    x[2,:,:,:,metric]=x[0,:,:,:,metric]+delta
    assert summarize(x,seed=1,repeats=100)['status'].startswith('DEVELOPMENT_FAIL')

def test_no_cherry_picking_profiles():
    x=fixture(); x[2,:,0,:,0]+=.1; x[2,:,1:,:,0]-=.1
    assert summarize(x,seed=1,repeats=100)['relative_power_gain']<0

def test_incomplete_or_nonfinite_stop():
    x=fixture()
    with pytest.raises(ValueError): summarize(x[:,:,:,:31],seed=1,repeats=10)
    x[0,0,0,0,0]=float('nan')
    with pytest.raises(ValueError): summarize(x,seed=1,repeats=10)
