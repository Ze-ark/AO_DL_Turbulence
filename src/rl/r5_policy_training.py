"""R5-2物理梯度残差策略训练：只训练11维小修正，正式运行由用户启动。"""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime, timezone
import json, os, time, traceback
from pathlib import Path
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from src.rl.r4_baselines import BaselineSpec, baseline_delta
from src.rl.r4_trajectory import anchor_delta
from src.rl.r4_control import NominalCalibration
from src.rl.r4_dynamics_experiment import Progress, safe_git_record
from src.rl.r4_interface_smoke import write_json
from src.rl.r4_observation import PowerMeasurement, R4Interface, simulation_residual_proxy
from src.rl.r5_physics_adapter import causal_policy_features, measured_objective
from src.rl.r5_batched_environment import R5BatchedEnvironment
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _profiles, _project_path
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.robust_control import RobustnessCondition

class ResidualGRUPolicy(nn.Module):
    def __init__(self, hidden_size=64, output_size=11):
        super().__init__(); self.gru=nn.GRU(79,hidden_size,batch_first=True); self.head=nn.Sequential(nn.Linear(hidden_size,64),nn.Tanh(),nn.Linear(64,output_size)); nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
    def forward(self, history, valid):
        x=causal_policy_features(history,valid)
        x=x*valid[...,None].to(x.dtype)
        out,h=self.gru(x)
        return torch.tanh(self.head(h[-1]))

def _preflight(path, quick):
    cfg=_load_yaml(_project_path(path));
    if cfg['stage']!='S4-D2-R5-2' or cfg['runtime']!={'device':'cuda','formal_owner':'user_ide','automatic_retry':False}: raise ValueError('R5-2 frozen CUDA config required')
    if cfg['boundary']!={'confirmation_access':False,'real_slm_actions':False,'s4d3_access':False,'automatic_retry':False,'policy_leakage_verified':False}: raise ValueError('R5-2 boundary changed')
    r0=_project_path(cfg['r5_0_summary']); r1=_project_path(cfg['r5_1_summary']);
    if _file_sha256(r0)!=cfg['r5_0_summary_sha256'] or json.loads(r0.read_text())['status']!='R5_0_CHECK_PASS': raise RuntimeError('R5-0 prerequisite changed')
    if _file_sha256(r1)!=cfg['r5_1_summary_sha256'] or json.loads(r1.read_text())['status']!='BASELINE_SELECTED_REQUIRES_READ_ONLY_AUDIT': raise RuntimeError('R5-1 prerequisite changed')
    sel=json.loads((_project_path(cfg['r5_1_selection'])).read_text());
    if sel['selected_index']!=0 or sel['candidate_means'][0][0] <= max(sel['candidate_means'][i][0] for i in range(1,9)): raise RuntimeError('integrator base is not frozen strongest candidate')
    n=cfg['quick']['episodes_per_family'] if quick else cfg['data']['episodes_per_family']; steps=cfg['quick']['episode_length'] if quick else cfg['data']['episode_length']; updates=cfg['quick']['updates_per_initialization'] if quick else cfg['training']['updates_per_initialization']; inits=cfg['quick']['initializations'] if quick else cfg['training']['initializations']; batch=cfg['quick']['logical_batch_episodes'] if quick else cfg['data']['logical_batch_episodes']
    if batch < 1 or n < 1: raise ValueError('episode batch contract')
    out=_project_path(cfg['quick_directory'] if quick else cfg['output_directory']);
    if out.exists(): raise FileExistsError(f'preserve existing R5-2 output: {out}')
    device=resolve_device('cuda'); return cfg,dict(quick=quick,episodes=n,steps=steps,updates=updates,inits=inits,batch=batch,device=str(device),training_updates=0,confirmation_access=False,real_slm_actions=False,s4d3_access=False,policy_leakage_verified=False)

def _rollout_batch(policy,cfg,s,device,seed,families,profiles,basis,base,cal):
    # 湍流条件与硬件档位合并为一个异构 CUDA batch；策略看不到档位标签。
    first = RobustnessCondition.from_mapping(dict(families[0], base_seed=seed))
    env_cfg = replace(first.environment_config(base), batch_size=len(families) * len(profiles), episode_length=s['steps'])
    env=R5BatchedEnvironment(env_cfg,device,basis,families,profiles,cfg['data']['sensor_seed_offset'])
    raw,_=env.reset(seed=seed); proxy=env.proxy; interface=R4Interface(calibration=cal); interface.reset(proxy(raw),episode_id=f'r5-{seed}'); batch=len(families)*len(profiles); prev=torch.zeros((batch,11),device=device); scores=[]
    timings = {'policy_forward': 0.0, 'env_step': 0.0, 'interface': 0.0, 'objective': 0.0}
    for t in range(s['steps']):
        tick = time.perf_counter()
        view=interface.snapshot()
        timings['interface'] += time.perf_counter() - tick
        tick = time.perf_counter()
        if cfg['training'].get('gradient_checkpointing', False) and torch.is_grad_enabled():
            corr=checkpoint(lambda h,v: policy(h,v), view.features, view.valid, use_reentrant=False)
        else:
            corr=policy(view.features,view.valid)
        timings['policy_forward'] += time.perf_counter() - tick
        base_delta=anchor_delta(view.features[:,-1], {'gain':.15,'leak':.10,'tracking_gain':.50})
        tick = time.perf_counter()
        action=interface.issue(base_delta,corr,step=t); raw,_,term,trunc,info=env.step(action.requested_delta_rad)
        timings['env_step'] += time.perf_counter() - tick
        tick = time.perf_counter()
        interface.observe_next(proxy(raw),step=t+1,power=PowerMeasurement(info['measured_power_in_bucket'],t,t+1)); scores.append(measured_objective(info['measured_power_in_bucket'],action.normalized_correction,prev,cfg['objective']['action_weight'],cfg['objective']['smooth_weight']).mean()); prev=action.normalized_correction
        timings['objective'] += time.perf_counter() - tick
        if bool(trunc.any()) or bool(term.all()) != (t==s['steps']-1): raise RuntimeError('R5-2 incomplete episode')
    return torch.stack(scores).mean(), timings

def run(path='configs/experiments/s4_r5_policy_training_v1.yaml',quick=False,preflight_only=False):
    cfg,s=_preflight(path,quick)
    if preflight_only:return s
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8'); perf=cfg.get('performance', {}); torch.use_deterministic_algorithms(bool(perf.get('deterministic', True))); torch.backends.cudnn.enabled=bool(perf.get('cudnn_enabled', True)); torch.backends.cudnn.benchmark=bool(perf.get('cudnn_benchmark', False)); torch.backends.cuda.matmul.allow_tf32=bool(perf.get('allow_tf32', False))
    out=_project_path(cfg['quick_directory'] if quick else cfg['output_directory']); out.mkdir(parents=True,exist_ok=False); write_json(out/'preflight.json',s); write_json(out/'config.json',cfg); write_json(out/'runtime.json',dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),git=safe_git_record())); (out/'checkpoints').mkdir()
    base,_=load_s1_config(_project_path(cfg['environment_config'])); base=replace(base,num_modes=21,batch_size=1,episode_length=s['steps']); device=resolve_device('cuda'); basis,_,_=build_action_basis(base,ActionRepresentation('r5_zernike21','zernike',21),device); profiles=_profiles(cfg,cfg['profile_ids']); cal=NominalCalibration(); progress=Progress(out,device); total=s['inits']*s['updates']; progress.phase('R5-2物理梯度策略训练',total); results=[]; timing_log=(out/'timing.jsonl').open('w',encoding='utf-8')
    try:
        for init in range(s['inits']):
            torch.manual_seed(5299001+init); torch.cuda.manual_seed_all(5299001+init); policy=ResidualGRUPolicy(cfg['policy']['hidden_size'],cfg['policy']['output_size']).to(device); opt=torch.optim.Adam(policy.parameters(),lr=cfg['training']['learning_rate']); running=0.
            for update in range(1,s['updates']+1):
                opt.zero_grad(set_to_none=True); losses=[]; timing_totals={'policy_forward':0.0,'env_step':0.0,'interface':0.0,'objective':0.0}
                # 每次更新固定覆盖 3 类湍流×6 档硬件误差=18 条完整回合，
                # 让一个更新同时看到所有预注册条件；模型不读取档位标识。
                seed = cfg['data']['train_seed_base'] + ((update - 1) % cfg['data']['episodes_per_family'])
                rollout_loss, rollout_timing = _rollout_batch(policy,cfg,s,device,seed,cfg['families'],profiles,basis,base,cal); losses.append(rollout_loss)
                for key,value in rollout_timing.items(): timing_totals[key] += value
                loss=-torch.stack(losses).mean(); backward_start=time.perf_counter(); loss.backward(); backward_seconds=time.perf_counter()-backward_start; torch.nn.utils.clip_grad_norm_(policy.parameters(),cfg['training']['gradient_norm_limit'],error_if_nonfinite=True); opt.step(); running+=float(loss.detach()); timing_log.write(json.dumps({'initialization':init,'update':update,**timing_totals,'backward':backward_seconds},ensure_ascii=False)+'\n'); timing_log.flush(); progress.tick({'初始化':init+1,'初始化总数':s['inits'],'当前批次':update,'平均损失':running/update,'策略前向秒':timing_totals['policy_forward'],'环境步进秒':timing_totals['env_step'],'反向秒':backward_seconds});
                if update%cfg['training']['checkpoint_interval_updates']==0 or update==s['updates']: torch.save({'state_dict':policy.state_dict(),'optimizer':opt.state_dict(),'init':init,'update':update,'config':cfg},out/'checkpoints'/f'policy_{init}_{update:05d}.pt')
            results.append({'initialization':init,'updates':s['updates'],'final_loss':running/s['updates']})
        result={'status':'R5_2_QUICK_SMOKE_COMPLETE' if quick else 'R5_2_TRAINING_COMPLETE_REQUIRES_AUDIT','results':results,'training_updates':s['inits']*s['updates'],'confirmation_access':False,'real_slm_actions':False,'s4d3_access':False,'policy_leakage_verified':False,'material_passport':{'origin_skill':'academic-research-suite / experiment-agent','origin_mode':'run','verification_status':'UNVERIFIED','version_label':'r5_2_policy_training_v1'},'next_action':'停止等待只读审计，不自动进入独立确认'}; write_json(out/'summary.json',result); write_json(out/'SUCCESS.json',{'summary_sha256':_file_sha256(out/'summary.json')}); return result
    except Exception: write_json(out/'failure.json',{'traceback':traceback.format_exc(),'automatic_retry':False}); raise
    finally: timing_log.close(); progress.close()
