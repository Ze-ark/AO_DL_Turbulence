"""CPU微型确定性恢复完整性测试，不生成性能结果。"""
from copy import deepcopy
import pytest
import torch
from src.rl.r4_amplitude_recovery import validate_batch, inventory, METRICS


def fixture():
    meta=dict(controller='gru_mpc',family=0,profile='nominal',seed=1)
    rows=[]; frames=torch.zeros(1,3,79)
    for step in range(2):
        r={k:torch.tensor([.5]) for k in (*METRICS,'measured_power')}
        r.update({k:torch.zeros(1,21) for k in ('requested_delta','requested_modal','applied_modal')})
        r.update(pre_scale_correction=torch.zeros(1,11),correction=torch.zeros(1,11),
            clipped_components=torch.zeros(1,11,dtype=torch.bool),terminated=torch.tensor([step==1]),
            predicted_valid=torch.zeros(1,dtype=torch.bool),unscaled_plan_score=torch.tensor([float('nan')]),
            zero_score=torch.tensor([float('nan')]),reasons=['outside_observation_range'],model_forward_samples=0)
        frames[:,step+1,74]=.5; rows.append(r)
    data=dict(**meta,step=1,seeds=[1],source='simulation_residual_proxy_not_holography',frames=frames,rows=rows)
    return data,meta


def test_valid_complete_batch():
    d,m=fixture(); result=validate_batch(d,m,batch=1,steps=2,reference=deepcopy(d))
    assert result['metrics']==[[.5]*6]
    assert result['model_forward_samples']==0
    assert result['fallback_counts']=={'outside_observation_range':2}


@pytest.mark.parametrize('case',range(8))
def test_bad_batch_rejected(case):
    d,m=fixture()
    if case==0: d['rows'].pop()
    if case==1: d['seeds']=[2]
    if case==2: d['rows'][-1]['terminated'].fill_(False)
    if case==3: d['rows'][0]['reward_power_in_bucket'].fill_(float('nan'))
    if case==4: d['rows'][0]['correction'].fill_(.1)
    if case==5: d['rows'][0]['requested_modal'].fill_(.1)
    if case==6: d['rows'][0]['model_forward_samples']=42
    if case==7: d['rows'][0]['clipped_components'].fill_(True)
    with pytest.raises((ValueError,RuntimeError)):
        validate_batch(d,m,batch=1,steps=2)


def test_inventory_has_no_duplicate_weather_slots():
    parent={'families':[{'id':f} for f in ('a','b','c')],'profile_ids':[str(i) for i in range(6)]}
    cfg={'controllers':['gru_mpc','gru_mpc_x15'],'development_starts':[3762048,3762304,3762560]}
    assert len(inventory(parent,cfg,False))==72
    assert len(inventory(parent,cfg,True))==2
