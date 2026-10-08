"""R4-1B1冻结模型动作方向诊断；失败或信号不足均停止，不执行MPC/RL。"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback

import torch

from src.rl.r4_action_response import (direction_statistics, model_branch, noise_threshold,
    physical_branch, physical_step, pulse)
from src.rl.r4_control import NominalCalibration, R4Limits
from src.rl.r4_dynamics import ARXDynamics, ResidualGRUDynamics
from src.rl.r4_dynamics_experiment import Progress, safe_git_record, verify_hashes
from src.rl.r4_interface_smoke import seed_blocks, write_json
from src.rl.r4_observation import R4Interface, simulation_residual_proxy
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _profiles, _project_path, _relative
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.robust_control import RobustnessCondition

SOURCES="configs/experiments/s4_r4_action_response_v1_sources.json"


def seeds_for(cfg: dict, parent: dict, phase: str, family: int, quick: bool) -> list[int]:
    if quick:
        q=cfg['quick'];start=q['namespace_seed']+q['family_offsets'][family]
        if phase=='development':start+=q['development_offset']
        return list(range(start,start+q['per_family']))
    d=parent['data'];start=d['namespace_seed']+d['family_offsets'][family]
    if phase=='development':start+=d['development_offset']
    n=cfg['calibration_train_per_family'] if phase=='calibration' else cfg['development_per_family']
    return list(range(start,start+n))


def budget(cfg: dict, quick: bool) -> dict:
    probes=cfg['quick']['probe_steps'] if quick else cfg['probe_steps']
    modes=cfg['quick']['high_mode_indices'] if quick else cfg['high_mode_indices']
    ntrain=3*(cfg['quick']['per_family'] if quick else cfg['calibration_train_per_family'])
    ndev=3*(cfg['quick']['per_family'] if quick else cfg['development_per_family'])
    branches=1+2*len(modes);h=cfg['horizon']
    return dict(calibration_weather=ntrain,development_weather=ndev,
        calibration_transitions=ntrain*6*(max(probes)+len(probes)*2*h),
        development_transitions=ndev*6*(max(probes)+len(probes)*branches*h),
        action_pairs=ndev*6*len(probes)*2*len(modes),
        model_forward_samples=ndev*6*len(probes)*branches*h*6)


def preflight(config_path: str|Path, quick: bool) -> tuple[dict,dict,dict]:
    path=_project_path(config_path);cfg=_load_yaml(path)
    if cfg['stage']!='S4-D2-R4-1B1' or cfg['runtime']!=dict(device='cuda',formal_owner='user_ide',automatic_retry=False):
        raise ValueError('requires user-owned CUDA diagnostic')
    if cfg['boundary']!=dict(model_updates=0,rl_updates=0,mpc_evaluation=False,confirmation_access=False,real_slm_actions=False,s4d3_access=False):
        raise ValueError('R4 response boundary changed')
    manifest=json.loads(_project_path(SOURCES).read_text(encoding='utf-8'));verify_hashes(manifest)
    if _relative(path) not in manifest:raise ValueError('unfrozen response config')
    parent=_load_yaml(_project_path(cfg['upstream_config']))
    up=_project_path(cfg['upstream_directory'])
    if _file_sha256(up/'summary.json')!=cfg['upstream_summary_sha256']:raise RuntimeError('upstream summary changed')
    summary=json.loads((up/'summary.json').read_text(encoding='utf-8'))
    success=json.loads((up/'SUCCESS.json').read_text(encoding='utf-8'))
    if success['summary_sha256']!=cfg['upstream_summary_sha256'] or summary['status']!='R4_1A_SUPERVISED_TRAINING_COMPLETE_REQUIRES_AUDIT':
        raise RuntimeError('R4-1A not completed')
    artifacts=json.loads((up/'artifact_manifest.json').read_text(encoding='utf-8'))
    if _file_sha256(up/'artifact_manifest.json')!=summary['artifact_manifest_sha256']:raise RuntimeError('upstream manifest changed')
    verify_hashes(artifacts)
    upstream_sources=json.loads((up/'preflight.json').read_text(encoding='utf-8'))['frozen_files'];verify_hashes(upstream_sources)
    # 只决定是否具备动作诊断前提；不得把点误差门槛写成R4-1整体通过。
    for m in range(3):
        ck=torch.load(up/'checkpoints'/f'gru_{m}_best.pt',map_location='cpu',weights_only=True)
        gru=ck['metrics']['per_horizon'][-1];lin=summary['members'][m]['linear']['per_horizon'][-1];hold=summary['persistence']['per_horizon'][-1]
        if not all(g<=.95*l and g<=.95*p for g,l,p in zip(gru,lin,hold)):
            raise RuntimeError('nonlinear prediction prerequisite not met; review design fallback')
    output=_project_path(cfg['quick_directory' if quick else 'output_directory']).resolve()
    expected=_project_path('outputs/s4_r4_action_response_v1'+('_quick' if quick else '')).resolve()
    if output!=expected or output.exists():raise FileExistsError(f'preserve response output: {output}')
    own={_project_path(cfg[k]).resolve() for k in ('quick_directory','output_directory')}
    historical=set();scanned=0
    files=list(_project_path('configs').rglob('*.yaml'))
    for name in ('preflight.json','effective_config.json','data_manifest.json'):files+=list(_project_path('outputs').glob('*/'+name))
    for p in files:
        if p.resolve()==path.resolve() or p.parent.resolve() in own:continue
        obj=_load_yaml(p) if p.suffix=='.yaml' else json.loads(p.read_text(encoding='utf-8'))
        historical|=seed_blocks(obj);scanned+=1
    if cfg['quick']['namespace_seed']//10000 in historical:raise RuntimeError('quick seed block collision')
    split_sets=[]
    for mode in (False,True):
        for phase in ('calibration','development'):
            seeds={s for f in range(3) for s in seeds_for(cfg,parent,phase,f,mode)}
            if any(s<0 or s>=4000000 or s//10000==378 for s in seeds):raise ValueError('reserved seeds')
            if any(seeds&old for old in split_sets):raise ValueError('response split overlap')
            split_sets.append(seeds)
    device=resolve_device('cuda')
    manifest.update(upstream_sources);manifest.update(artifacts)
    for p in (path,_project_path(SOURCES),up/'summary.json',up/'SUCCESS.json',up/'artifact_manifest.json'):
        manifest[_relative(p)]=_file_sha256(p)
    return cfg,parent,dict(status='READY_FOR_DIAGNOSTIC_SMOKE' if quick else 'READY_FOR_USER_IDE',
        quick=quick,device=str(device),historical_files_scanned=scanned,budget=budget(cfg,quick),
        prediction_prerequisite='3_OF_3_H8_POINT_ERROR_PASS',frozen_files=manifest,**cfg['boundary'])


def load_models(up: Path, device: torch.device) -> list:
    result=[]
    for m in range(3):
        state=torch.load(up/'checkpoints'/f'linear_{m}.pt',map_location=device,weights_only=True)['state_dict']
        linear=ARXDynamics(state['x_mean'],state['x_scale'],state['y_mean'],state['y_scale']);linear.load_state_dict(state)
        gru=ResidualGRUDynamics(linear,64).to(device)
        gru.load_state_dict(torch.load(up/'checkpoints'/f'gru_{m}_best.pt',map_location=device,weights_only=True)['state_dict'])
        result.extend([linear.eval().requires_grad_(False),gru.eval().requires_grad_(False)])
    return result


@torch.no_grad()
def execute(cfg: dict,parent: dict,report: dict,output: Path) -> dict:
    device=resolve_device('cuda');quick=report['quick'];torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled=False;torch.backends.cuda.matmul.allow_tf32=False
    probes=cfg['quick']['probe_steps'] if quick else cfg['probe_steps']
    modes=cfg['quick']['high_mode_indices'] if quick else cfg['high_mode_indices']
    batch_size=cfg['quick']['per_family'] if quick else cfg['batch_episodes']
    horizon=cfg['horizon'];base,_=load_s1_config(_project_path(parent['environment_config']))
    base=replace(base,num_modes=21,batch_size=batch_size)
    basis,_,_=build_action_basis(base,ActionRepresentation('r4_zernike21','zernike',21),device)
    profiles=_profiles(parent,parent['profile_ids']);cal=NominalCalibration(**parent['nominal_calibration'])
    models=load_models(_project_path(cfg['upstream_directory']),device)
    progress=Progress(output,device);(output/'branches').mkdir()
    differences={p.identifier:[] for p in profiles};thresholds={};rows=[];branch_manifest=[]
    model_calls=0;physical_transitions=0
    def tick():
        nonlocal physical_transitions
        physical_transitions+=batch_size;progress.tick()
    def save(name,data):
        path=output/'branches'/name
        torch.save(data,path);branch_manifest.append(dict(file=_relative(path),sha256=_file_sha256(path)))
    try:
        for phase in ('calibration','development'):
            progress.phase(f'{phase} 动作响应诊断',report['budget'][phase+'_transitions']//batch_size)
            for f,family in enumerate(parent['families']):
                all_seeds=seeds_for(cfg,parent,phase,f,quick)
                for profile in profiles:
                    for start in range(0,len(all_seeds),batch_size):
                        seeds=all_seeds[start:start+batch_size]
                        if len(seeds)!=batch_size:raise ValueError('partial batch is not frozen')
                        condition=RobustnessCondition.from_mapping(dict(family,base_seed=seeds[0]))
                        env=AdaptiveOpticsEnv(profile.environment_config(condition.environment_config(base)),device,
                            profile.effects_config(),basis_override=basis)
                        raw,_=env.reset(seed=seeds[0])
                        sensor=torch.Generator(device=device).manual_seed(seeds[0]+parent['data']['sensor_seed_offset'])
                        interface=R4Interface(calibration=cal)
                        interface.reset(simulation_residual_proxy(raw,generator=sensor,noise_std_rad=profile.observation_noise_std_rad),episode_id=str(seeds[0]))
                        for t in range(max(probes)+1):
                            if t in probes:
                                snapshot=interface.snapshot()
                                zero=physical_branch(env,interface,sensor,profile,parent['collector_anchor'],0,0.,horizon,tick)
                                prefix=f'{phase}_{family["id"]}_{profile.identifier}_{seeds[0]}_{t}'
                                if phase=='calibration':
                                    repeat=physical_branch(env,interface,sensor,profile,parent['collector_anchor'],0,0.,horizon,tick,
                                        measurement_seed=seeds[0]+cfg['noise']['independent_measurement_seed_offset']+t)
                                    if not torch.equal(zero['requested_delta'],repeat['requested_delta']) or not torch.equal(zero['audit_power'],repeat['audit_power']):
                                        raise RuntimeError('repeat measurement changed physical trajectory')
                                    differences[profile.identifier].append((repeat['power'].mean(1)-zero['power'].mean(1)).cpu())
                                    save(prefix+'.pt',dict(seeds=seeds,zero={k:v.cpu() for k,v in zero.items()},repeat={k:v.cpu() for k,v in repeat.items()}))
                                else:
                                    predicted_zero=[model_branch(model,snapshot.features,snapshot.valid,parent['collector_anchor'],cal,0,0.,horizon) for model in models]
                                    model_calls+=len(models)*batch_size*horizon
                                    save(prefix+'_zero.pt',dict(seeds=seeds,history=snapshot.features.cpu(),valid=snapshot.valid.cpu(),
                                        actual={k:v.cpu() for k,v in zero.items()},models=[{k:v.cpu() for k,v in r.items()} for r in predicted_zero]))
                                    for mode in modes:
                                        for sign in (-1.,1.):
                                            actual=physical_branch(env,interface,sensor,profile,parent['collector_anchor'],mode,sign,horizon,tick)
                                            predicted=[model_branch(model,snapshot.features,snapshot.valid,parent['collector_anchor'],cal,mode,sign,horizon) for model in models]
                                            model_calls+=len(models)*batch_size*horizon
                                            actual_delta=actual['power'].mean(1)-zero['power'].mean(1)
                                            pred_delta=torch.stack([r['power'].mean(1)-z['power'].mean(1) for r,z in zip(predicted,predicted_zero)],dim=1)
                                            save(prefix+f'_{mode}_{int(sign)}.pt',dict(seeds=seeds,actual={k:v.cpu() for k,v in actual.items()},
                                                models=[{k:v.cpu() for k,v in r.items()} for r in predicted]))
                                            for i,seed in enumerate(seeds):
                                                row=dict(weather_seed=seed,family=f,profile=profile.identifier,probe=t,mode=mode,sign=sign,
                                                    actual_delta=float(actual_delta[i]),predicted_deltas=pred_delta[i].tolist(),threshold=thresholds[profile.identifier])
                                                rows.append(row)
                                                with (output/'pairs.jsonl').open('a',encoding='utf-8') as file:file.write(json.dumps(row)+'\n')
                            if t<max(probes):
                                physical_step(env,interface,sensor,profile,pulse(batch_size,0,0.,0,device),parent['collector_anchor']);tick()
            if phase=='calibration':
                thresholds={p:noise_threshold(torch.cat(v),cfg['noise']['threshold_std_multiplier'],cfg['noise']['numerical_floor']) for p,v in differences.items()}
                write_json(output/'noise_calibration.json',dict(thresholds=thresholds,source='training_weather_only',
                    repeated_differences={p:torch.cat(v).tolist() for p,v in differences.items()}))
                calibration_sha=_file_sha256(output/'noise_calibration.json')
        if physical_transitions!=sum(report['budget'][x] for x in ('calibration_transitions','development_transitions')) or model_calls!=report['budget']['model_forward_samples']:
            raise RuntimeError('response interaction budget mismatch')
        if _file_sha256(output/'noise_calibration.json')!=calibration_sha:raise RuntimeError('calibration changed after development')
        tensors={k:torch.tensor([r[k] for r in rows],device=device) for k in ('weather_seed','family','actual_delta','threshold')}
        pred=torch.tensor([r['predicted_deltas'] for r in rows],device=device)
        specs={'primary_gru_ensemble':pred[:,[1,3,5]].mean(1),'linear_ensemble_diagnostic':pred[:,[0,2,4]].mean(1)}
        specs.update({f'gru_member_{i}_diagnostic':pred[:,2*i+1] for i in range(3)})
        statistics={}
        for name,value in specs.items():
            statistics[name]=direction_statistics(tensors['actual_delta'],value,tensors['threshold'],tensors['weather_seed'],tensors['family'],
                seed=cfg['statistics']['bootstrap_seed'],replicates=cfg['quick']['bootstrap_replicates'] if quick else cfg['statistics']['bootstrap_replicates'],
                min_weather=cfg['statistics']['min_identifiable_weather'],accuracy_min=cfg['statistics']['balanced_accuracy_min'],lower_exclusive=cfg['statistics']['ci95_lower_exclusive'])
        write_json(output/'branch_manifest.json',branch_manifest)
        verify_hashes(report['frozen_files'])
        for record in branch_manifest:verify_hashes({record['file']:record['sha256']})
        artifacts={_relative(p):_file_sha256(p) for p in output.rglob('*') if p.is_file()}
        write_json(output/'artifact_manifest.json',artifacts)
        return dict(material_passport=dict(origin_skill='academic-research-suite / experiment-agent',origin_mode='run',
            origin_date=datetime.now(timezone.utc).isoformat(),verification_status='UNVERIFIED',version_label='r4_1b1_response_v1'),
            status='QUICK_SMOKE_ONLY' if quick else statistics['primary_gru_ensemble']['status'],
            statistics={} if quick else statistics,quick_statistics_diagnostic_only=statistics if quick else {},
            budget=report['budget'],actual_physical_transitions=physical_transitions,model_forward_samples=model_calls,
            source='simulation_residual_proxy_not_holography',calibration_sha256=calibration_sha,
            artifact_manifest_sha256=_file_sha256(output/'artifact_manifest.json'),**cfg['boundary'],
            r4_1_gate='NOT_EVALUATED_MPC_PENDING',next_action='停止等待只读验收；不自动训练、规划或访问独立确认。')
    finally:progress.close()


def run(config_path: str|Path,*,quick: bool=False,preflight_only: bool=False) -> dict:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    cfg,parent,report=preflight(config_path,quick)
    if preflight_only:return {k:v for k,v in report.items() if k!='frozen_files'}
    output=_project_path(cfg['quick_directory' if quick else 'output_directory']);output.mkdir(parents=True,exist_ok=False)
    write_json(output/'preflight.json',report);write_json(output/'effective_config.json',cfg)
    write_json(output/'runtime.json',dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),git=safe_git_record()))
    try:
        result=execute(cfg,parent,report,output);write_json(output/'summary.json',result)
        write_json(output/'SUCCESS.json',dict(status=result['status'],summary_sha256=_file_sha256(output/'summary.json')))
        return result
    except Exception as exc:
        write_json(output/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),automatic_retry=False));raise
