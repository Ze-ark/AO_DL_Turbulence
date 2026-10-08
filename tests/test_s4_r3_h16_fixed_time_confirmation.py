"""A9小型确定性CPU单元测试；不产生正式性能结果。"""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

from src.rl.s4_r3_h16_fixed_time_contract import (
    ARMS, BOUNDARY, budget, effective_settings, load_contract, namespace_check, preflight, _seed_blocks,
)
from src.rl.s4_r3_h16_fixed_time_training import (
    FixedTimeProbe, episode_groups, fit_probe, loss_terms, metrics, predictions,
    require_frozen, seal_training, selection_key, validate_data, validate_inventory,
)
from src.rl.s4_r3_h16_fixed_time_analysis import error_changes, interpret, primary_comparisons, _load_model
from src.rl.s4_r3_h16_reward_pairwise_probe import RewardPairDataset
from src.rl.s4_training import _file_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/s4_r3_h16_fixed_time_confirmation_v1.yaml"


def settings():
    return load_contract(CONFIG, quick=True)[1]


def pairs(offset=0):
    g = torch.Generator().manual_seed(31+offset)
    rows = [dict(condition_id=f"c_{i//8}", episode_seed=offset+i//4,
                 profile_id=f"p{i%2}", probe_step=(i%4)//2) for i in range(24)]
    y = torch.tensor([.1, -.2]*12)
    return RewardPairDataset(torch.randn(24, 221, generator=g), y, y.clone(), rows)


def test_design_budget_and_namespace():
    s = settings(); d = s["design"]
    formal = effective_settings(d, quick=False)
    b = budget(formal)
    assert (b["planned_fits"], b["inventory_records"], b["endpoint_records"], b["gradient_records"]) == (36,324,1152,82944)
    assert budget(s)["gradient_records"] == 108
    report = namespace_check(formal, s, {398,395,390})
    assert len(report) == 8 and max(r["maximum"] for r in report) == 3997801
    with pytest.raises(RuntimeError, match="namespace"):
        namespace_check(formal, s, {399})
    overlap = deepcopy(s)
    overlap["splits"]["selection"] = overlap["splits"]["training"]
    with pytest.raises(RuntimeError, match="overlap"):
        namespace_check(formal, overlap, set())
    assert _seed_blocks({"nested": [{"base_seed": 3900127}, {"test_base_seeds": [3861000,3871000]}]}) == {390,386,387}


def test_quick_run_tag_only_changes_output_directory():
    original = settings()
    _, tagged = load_contract(CONFIG, quick=True, quick_run_tag='retry1')
    assert tagged.pop('quick_run_tag') == 'retry1'
    assert tagged.pop('output_directory') == original.pop('output_directory') + '_retry1'
    assert tagged == original
    for label in ('../escape', 'x/y', 'C:\\x', '', 'a'*33):
        with pytest.raises(ValueError, match='quick run tag'):
            load_contract(CONFIG, quick=True, quick_run_tag=label)
    with pytest.raises(ValueError, match='quick run tag'):
        load_contract(CONFIG, quick=False, quick_run_tag='retry1')


def test_existing_quick_directory_still_cannot_be_overwritten(tmp_path):
    cfg, s = load_contract(CONFIG, quick=True, quick_run_tag='retry1')
    output = tmp_path / 's4_r3_h16_fixed_time_confirmation_v1_quick_retry1'
    output.mkdir()
    with patch('src.rl.s4_r3_h16_fixed_time_contract.resolve_device', return_value=torch.device('cuda')), \
         patch('src.rl.s4_r3_h16_fixed_time_contract.safe_path', side_effect=lambda value: tmp_path if value == 'outputs' else output):
        with pytest.raises(FileExistsError, match='preserve it'):
            preflight(CONFIG, cfg, s)


def test_same_condition_templates_really_match():
    s = settings()
    strip = lambda c: {k:v for k,v in c.items() if k not in ('id','base_seed')}
    for a,b,c in zip(s['splits']['training']['conditions'],s['splits']['selection']['conditions'],s['splits']['confirmation_id']['conditions']):
        assert strip(a) == strip(b) == strip(c)
    assert strip(s['splits']['training']['conditions'][0]) != strip(s['splits']['confirmation_shift']['conditions'][0])


@pytest.mark.parametrize('change', ['cpu','rl','design_only','budget_override','bad_hash'])
def test_runtime_contract_fails_closed(tmp_path, change):
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    if change=='cpu': cfg['runtime']['device']='cpu'
    elif change=='rl': cfg['boundary']['full_rl_trained']=True
    elif change=='design_only': cfg=settings()['design']
    elif change=='budget_override': cfg['maximum_updates']=1
    else: cfg['design_sha256']='0'*64
    p=tmp_path/'bad.yaml';p.write_text(yaml.safe_dump(cfg),encoding='utf-8')
    with pytest.raises(RuntimeError): load_contract(p,quick=True)


def test_cpu_unavailable_preflight_does_not_write():
    cfg,s=load_contract(CONFIG,quick=True)
    with patch('torch.cuda.is_available',return_value=False), pytest.raises(RuntimeError,match='CUDA'):
        preflight(CONFIG,cfg,s)


def test_initialization_and_single_task_gradient_semantics():
    torch.manual_seed(7)
    initial=FixedTimeProbe(221,8,'shared').state_dict()
    models={a:FixedTimeProbe(221,8,a) for a in ARMS}
    for model in models.values(): model.load_shared_initial(initial)
    x=pairs().features; target=pairs().reward_delta; weight=torch.tensor(2.)
    ref=models['shared'](x)[0]
    for model in models.values():
        assert all(torch.equal(t,ref) for t in model(x))
    for a in ARMS:
        total,reg,cls=loss_terms(models[a],x,target,weight,huber_delta=1.)
        assert (reg is None)==(a=='classification_only')
        assert (cls is None)==(a=='regression_only')
        expected = (reg if reg is not None else 0)+(0.25*cls if cls is not None else 0)
        assert torch.equal(total,expected)
        total.backward()
        assert models[a].network[0].weight.grad is not None


def test_classification_score_never_becomes_reward():
    d=pairs(); model=FixedTimeProbe(221,8,'classification_only')
    norms=dict(feature_mean=torch.zeros(221),feature_scale=torch.ones(221),target_scale=torch.tensor(999.))
    v,score=predictions(model,d,norms,torch.device('cpu'))
    assert v is None
    m=metrics(v,score,d,0.)
    assert m['value_mae'] is None and m['value_sign_ba'] is None
    perfect=metrics(d.reward_delta,d.reward_delta,d,0.)
    assert perfect['value_mae']==0 and perfect['balanced_accuracy']==1
    part=RewardPairDataset(d.features[:1],d.reward_delta[:1],d.power_delta[:1],d.rows[:1])
    assert metrics(None,d.reward_delta[:1],part,0.)['balanced_accuracy'] is None
    with pytest.raises(RuntimeError,match='non-finite'): metrics(None,score*float('nan'),d,0.)


def test_duplicate_and_partial_episode_detection():
    d=pairs(); bad=RewardPairDataset(d.features,d.reward_delta,d.power_delta,[d.rows[0]]+d.rows[:-1])
    with pytest.raises(RuntimeError,match='duplicate'):episode_groups(bad)
    split=dict(conditions=[dict(id='c_0',base_seed=0),dict(id='c_1',base_seed=2),dict(id='c_2',base_seed=4)],
               episodes_per_condition=2,profile_ids=['p0','p1'],probe_steps=[0,1])
    validate_data(d,split,9301,collection_record=dict(policy_seed=9301))
    assert all('policy_seed' not in row for row in d.rows)
    for record in ({}, dict(policy_seed=9302)):
        with pytest.raises(RuntimeError,match='policy identity'):
            validate_data(d,split,9301,collection_record=record)
    bad=RewardPairDataset(d.features[:-1],d.reward_delta[:-1],d.power_delta[:-1],d.rows[:-1])
    with pytest.raises(RuntimeError,match='incomplete'):
        validate_data(bad,split,9301,collection_record=dict(policy_seed=9301))


@pytest.fixture
def trained(tmp_path):
    s=settings();s['model']['hidden_sizes']=[8,8]
    torch.manual_seed(31);initial=FixedTimeProbe(221,8,'shared').state_dict()
    fits=[fit_probe(training=pairs(100),selection=pairs(200),s=s,policy_seed=9301,replicate=0,task=a,
                    initial=initial,directory=tmp_path/a,device=torch.device('cpu')) for a in ARMS]
    return tmp_path,s,fits,initial


def test_fixed_schedule_paired_batches_and_progress(trained):
    out,s,fits,initial=trained
    inv=validate_inventory(fits,s)
    assert len(inv)==16 and len({f['batch_order_sha256'] for f in fits})==1
    for fit in fits:
        zero=next(r for r in fit['inventory'] if r['role']=='fixed' and r['update']==0)
        ck=torch.load(ROOT/zero['path'],weights_only=False)
        assert all(torch.equal(ck['probe'][k],v) for k,v in initial.items())
        assert torch.equal(ck['normalization']['feature_mean'],pairs(100).features.mean(0))
        assert ck['independent_confirmation_used_for_selection'] is False
        log=[json.loads(x) for x in (out/fit['arm']/'progress.jsonl').read_text(encoding='utf-8').splitlines()]
        assert [x['update'] for x in log]==[4,8]
        assert all('estimated_remaining_seconds' in x and 'cuda_allocated_gb' in x for x in log)
    assert set(inspect.signature(fit_probe).parameters).isdisjoint({'confirmation','confirmation_id','test'})


def test_freeze_rejects_incomplete_or_changed_models(trained):
    out,s,fits,_=trained
    with pytest.raises(RuntimeError,match='incomplete fits'):seal_training(fits[:-1],s,out)
    bad=deepcopy(fits);bad[0]['inventory']=bad[0]['inventory'][1:]
    with pytest.raises(RuntimeError,match='schedule'):validate_inventory(bad,s)
    marker=seal_training(fits,s,out)
    assert len(require_frozen(out,marker,s))==16
    with pytest.raises(FileExistsError):seal_training(fits,s,out)
    rec=fits[0]['inventory'][0]
    p=ROOT/rec['path'];p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(RuntimeError,match='hash changed'):require_frozen(out,marker,s)


def test_missing_freeze_and_wrong_checkpoint_identity(trained):
    out,s,fits,_=trained
    with pytest.raises(RuntimeError,match='freeze'):require_frozen(out,dict(path=str(out/'TRAINING_FROZEN.json'),sha256='0'*64),s)
    rec=deepcopy(fits[0]['inventory'][0]); rec['update']=999
    with pytest.raises(RuntimeError,match='identity'):_load_model(rec,torch.device('cpu'))


def statistic_fixture():
    s=settings();s['policy_seeds']=[9301,9302,9303];s['replicate_indices']=[0,1,2];s['quick']=False
    e=np.ones((3,3,3,2,6));e[:,:,:,1,:]=[1] # override below
    e[:,:,0,1,:]=1.3;e[:,:,1,1,:]=1.1;e[:,:,2,1,:]=.9
    train=np.ones_like(e);train[:,:,:,1,:]=.8
    keys=[(f'f{i//2}',i) for i in range(6)]
    return s,e,train,keys


def test_known_relative_changes_and_familywise_pairing():
    s,e,tr,keys=statistic_fixture()
    rows,decision=primary_comparisons(e,tr,keys,s)
    assert len(rows)==9 and decision['status']=='FIXED_TIME_GENERALIZATION_DEGRADATION_SUPPORTED'
    for row in rows:
        expected={'shared_mae_deterioration':.3,'split_attenuation':.2,'regression_only_attenuation':.4}[row['metric']]
        assert row['estimate']==pytest.approx(expected)
        assert row['familywise_ci_low']==pytest.approx(expected)
        assert row['episodes']==6 and row['family_size']==9
    # Same per-episode ratio for all arms -> paired attenuation identically zero despite heterogeneity.
    e[:,:,:,0,:]=np.arange(1,7);e[:,:,:,1,:]=e[:,:,:,0,:]*1.5
    rows,_=primary_comparisons(e,tr,keys,s)
    assert all(abs(r['familywise_ci_high'])<1e-12 for r in rows if r['metric']!='shared_mae_deterioration')


def test_averages_individual_changes_not_ensemble_or_ratio_of_means():
    s,e,tr,keys=statistic_fixture()
    e[:,0,0,0,:]=1;e[:,0,0,1,:]=2
    e[:,1,0,0,:]=10;e[:,1,0,1,:]=10
    e[:,2,0,0,:]=100;e[:,2,0,1,:]=100
    rows,_=primary_comparisons(e,tr,keys,s)
    assert rows[0]['estimate']==pytest.approx(1/3)


def test_no_invented_result_for_zero_denominator_or_bad_pairing():
    s,e,tr,keys=statistic_fixture();e[:,:,0,0,:]=0
    rows,decision=primary_comparisons(e,tr,keys,s)
    assert rows[0]['estimate'] is None and 'NOT_CONFIRMED' in decision['status']
    with pytest.raises(ValueError,match='duplicate'):primary_comparisons(e,tr,[keys[0]]*6,s)
    with pytest.raises(ValueError):error_changes(np.array([[-1.,2.]]),floor=1e-12)


def test_attenuation_requires_supported_primary_and_training_improvement():
    s,e,tr,keys=statistic_fixture();tr[:,:,:,1,:]=1.1
    rows,decision=primary_comparisons(e,tr,keys,s)
    assert all(not x for x in decision['attenuation_supported'].values())
    assert not decision['full_rl_authorized']
    s['quick']=True;assert interpret(rows,s)['status']=='QUICK_SMOKE_ONLY'
    with pytest.raises(RuntimeError,match='missing'):interpret(rows[:-1],s)


def test_selection_rules_and_tie_behavior():
    a=dict(balanced_accuracy=.7,value_mae=.2);b=dict(balanced_accuracy=.7,value_mae=.1)
    assert selection_key('shared',b)>selection_key('shared',a)
    assert selection_key('classification_only',a)==selection_key('classification_only',b)
    assert selection_key('regression_only',b)>selection_key('regression_only',a)


def test_import_and_cli_help_do_not_start_work():
    code="import torch; from unittest.mock import patch; from pathlib import Path\nwith patch('torch.cuda.init',side_effect=AssertionError('GPU')), patch.object(Path,'mkdir',side_effect=AssertionError('write')):\n import src.rl.s4_r3_h16_fixed_time_confirmation\n"
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True,capture_output=True)
    result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/train_s4_r3_h16_fixed_time_confirmation.py'),'--help'],cwd=ROOT,check=True,capture_output=True,text=True,encoding='utf-8')
    assert '--preflight-only' in result.stdout and '--quick' in result.stdout
