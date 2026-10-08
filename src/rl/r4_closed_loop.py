"""冻结三组完整闭环；正式执行必须由用户显式启动。"""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import time
import traceback
import torch
from src.rl.r4_baseline_selection import METRICS, read
from src.rl.r4_baselines import baseline_delta, guarded_correction, range_eligible
from src.rl.r4_control import NominalCalibration
from src.rl.r4_dynamics_experiment import Progress, verify_hashes, safe_git_record
from src.rl.r4_interface_smoke import write_json
from src.rl.r4_mpc import SearchConfig
from src.rl.r4_observation import R4Interface, PowerMeasurement, simulation_residual_proxy
from src.rl.r4_response_experiment import load_models
from src.rl.r4_selected_anchor import load_selected, selected_plan, selected_request
from src.rl.r4_trajectory import EpisodeStore
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _project_path, _relative, _load_yaml, _file_sha256, _profiles
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.robust_control import RobustnessCondition

SOURCES = 'configs/experiments/s4_r4_closed_loop_v1_sources.json'


def summarize(means: torch.Tensor, *, seed: int, repeats: int = 20000) -> dict:
    """[控制器,类型,档位,天气,指标]；先平均档位，按完整天气分层配对。"""
    if means.shape != (3, 3, 6, 32, 6) or not bool(torch.isfinite(means).all()):
        raise ValueError('complete finite formal episodes required')
    values = means.double().mean(2)
    average = values.mean((1, 2)); difference = values[2]-values[0]
    generator = torch.Generator(device=means.device).manual_seed(seed)
    boot = means.new_zeros(repeats, dtype=torch.float64)
    for family in range(3):
        indices = torch.randint(32, (repeats, 32), device=means.device, generator=generator)
        boot += difference[family, :, 0][indices].mean(1)/3
    ci = torch.quantile(boot, means.new_tensor([.025, .975], dtype=torch.float64))
    if float(average[0, 0]) <= 0: raise ValueError('nonpositive reference power')
    relative = float(average[2, 0]/average[0, 0]-1)
    gates = dict(power_at_least_one_percent=relative >= .01, paired_ci_positive=float(ci[0]) > 0,
        strehl_not_lower=float(average[2, 1]-average[0, 1]) >= 0,
        violation_increase_at_most_point001=float(average[2, 2]-average[0, 2]) <= .001)
    return dict(status='DEVELOPMENT_PASS_REQUIRES_AUDIT' if all(gates.values()) else 'DEVELOPMENT_FAIL_REQUIRES_AUDIT',
        gates=gates, controller_means=average.tolist(), relative_power_gain=relative,
        paired_power_difference_ci95=ci.tolist(), family_delta=difference.mean(1).tolist(),
        profile_delta=(means[2]-means[0]).double().mean((0, 2)).tolist(),
        bootstrap_repeats=repeats, bootstrap_seed=seed, independent_confirmation=False)


def model_ensembles(device: torch.device) -> dict:
    models = load_models(_project_path('outputs/s4_r4_dynamics_v1'), device)
    result = dict(integrator=[], linear_mpc=models[::2], gru_mpc=models[1::2])
    for i, model in enumerate(result['gru_mpc']):
        path = _project_path(f'outputs/s4_r4_delta_supervision_v1/checkpoints/absolute_plus_delta_{i}_02000.pt')
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True)['state_dict'])
        model.eval().requires_grad_(False)
    return result


def preflight(path: str | Path, mode: str) -> tuple:
    if mode not in ('formal', 'quick', 'batch'): raise ValueError('unknown run mode')
    cfg = _load_yaml(_project_path(path)); own = read(_project_path(SOURCES)); verify_hashes(own)
    if _relative(_project_path(path)) not in own: raise ValueError('unfrozen configuration')
    spec, frozen = load_selected()
    up = _project_path(cfg['upstream']); s = read(up/'summary.json')
    if (_file_sha256(up/'summary.json') != cfg['upstream_sha256']
            or read(up/'SUCCESS.json')['summary_sha256'] != cfg['upstream_sha256']
            or s['status'] != 'SELECTED_ANCHOR_SHORT_CLOSED_LOOP_COMPLETE'
            or _file_sha256(up/'artifact_manifest.json') != s['artifact_manifest_sha256']):
        raise RuntimeError('short closed-loop prerequisite changed')
    frozen.update(read(up/'preflight.json')['frozen_files']); frozen.update(read(up/'artifact_manifest.json'))
    frozen.update(own)
    for p in (up/'summary.json', up/'SUCCESS.json', up/'artifact_manifest.json', _project_path(SOURCES)):
        frozen[_relative(p)] = _file_sha256(p)
    if mode == 'formal':
        # 正式入口必须引用本版本通过的规模检查与快速完整回合，不自动代跑准备。
        for suffix, expected in (('_batch', 'BATCH_CHECK_PASS'), ('_quick', 'QUICK_COMPLETE_NO_RANKING')):
            p = _project_path(cfg['output_directory']+suffix)
            summary = read(p/'summary.json')
            if summary['status'] != expected or read(p/'SUCCESS.json')['summary_sha256'] != _file_sha256(p/'summary.json'):
                raise RuntimeError('preparation prerequisite incomplete')
            if _file_sha256(p/'artifact_manifest.json') != summary['artifact_manifest_sha256']:
                raise RuntimeError('preparation artifacts changed')
            prior = read(p/'preflight.json')['frozen_files']
            if prior.get(SOURCES) != _file_sha256(_project_path(SOURCES)):
                raise RuntimeError('preparation uses different source version')
            frozen.update(prior); frozen.update(read(p/'artifact_manifest.json'))
            for name in ('summary.json', 'SUCCESS.json', 'artifact_manifest.json'):
                frozen[_relative(p/name)] = _file_sha256(p/name)
    verify_hashes(frozen)
    parent = _load_yaml(_project_path(cfg['parent']))
    if cfg['development_starts'] != [parent['data']['namespace_seed']+parent['data']['development_offset']+o for o in parent['data']['family_offsets']]:
        raise ValueError('weather split changed')
    if cfg['physical_transitions'] != 3*3*6*32*200 or cfg['max_model_forward_samples'] != 2*3*6*32*200*12288:
        raise ValueError('formal budget mismatch')
    suffix = '' if mode == 'formal' else '_'+mode
    output = _project_path(cfg['output_directory']+suffix)
    if output.exists(): raise FileExistsError(f'preserve output: {output}')
    if shutil.disk_usage(_project_path('.')).free < (4096 if mode == 'formal' else 128)*1024**2:
        raise RuntimeError('insufficient output disk space')
    device = resolve_device('cuda')
    report = dict(mode=mode, device=str(device), frozen_files=frozen, status='READY_FOR_USER_IDE' if mode=='formal' else 'READY_FOR_DIAGNOSTIC',
        physical_transitions=345600 if mode=='formal' else 72 if mode=='quick' else 0,
        max_model_forward_samples=2831155200 if mode=='formal' else 589824 if mode=='quick' else 786432,
        training_updates=0, real_slm_actions=False, confirmation_access=False)
    return cfg, parent, spec, output, report


@torch.no_grad()
def batch_check(models, spec, cal, output, device) -> dict:
    records = read(_project_path('outputs/s4_r4_dynamics_v1/data_manifest.json'))['records']
    source = next(r['file'] for r in records if r['split']=='train')
    batch = EpisodeStore.from_files([_project_path(source)]).windows(torch.arange(16), torch.full((16,),80), 8, device)
    h,v = batch['history'],batch['valid']; rows=[]; timings=[]
    for name in ('linear_mpc','gru_mpc'):
        previous=None
        for repeat in range(2):
            print(f'B=16原规模检查：{name}，{repeat+1}/2', flush=True)
            torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device); start=time.perf_counter()
            result=selected_plan(models[name],h,v,spec,cal,SearchConfig(),seed=3769600)
            torch.cuda.synchronize(device)
            timings.append(dict(controller=name,repeat=repeat,seconds=time.perf_counter()-start,
                peak_allocated_gb=torch.cuda.max_memory_allocated(device)/1024**3,
                peak_reserved_gb=torch.cuda.max_memory_reserved(device)/1024**3))
            if previous is not None and not all(torch.equal(result[k],previous[k]) for k in result if torch.is_tensor(result[k])):
                raise RuntimeError('batch decision repeatability mismatch')
            previous=result; rows.append({k:v.cpu() if torch.is_tensor(v) else v for k,v in result.items()})
    torch.save(rows,output/'decisions.pt'); write_json(output/'timing.json',timings)
    return dict(status='BATCH_CHECK_PASS',model_forward_samples=sum(r['model_forward_samples'] for r in rows),
                physical_transitions=0,source=source,timing=timings,ranking={})


@torch.no_grad()
def execute(cfg,parent,spec,output,report,models,cal,device) -> dict:
    quick=report['mode']=='quick'; n,b,t=(2,2,12) if quick else (32,16,200)
    families=parent['families'][:1] if quick else parent['families']
    profiles=_profiles(parent,parent['profile_ids'][:1] if quick else parent['profile_ids'])
    starts=[3760000] if quick else cfg['development_starts']
    base,_=load_s1_config(_project_path(parent['environment_config']))
    base=replace(base,num_modes=21,batch_size=b,episode_length=t)
    basis,_,_=build_action_basis(base,ActionRepresentation('r4_zernike21','zernike',21),device)
    bounds=read(_project_path('outputs/s4_r4_baseline_safety_v1/calibration.json'))
    means=torch.zeros(3,len(families),len(profiles),n,6,device=device,dtype=torch.float64)
    progress=Progress(output,device); progress.phase('三组完整闭环' if not quick else '入口快速验证（不排名）',report['physical_transitions']//b)
    physical=calls=0; active={}; manifest=[]; reasons_total={}; planned_total=0
    (output/'trajectories').mkdir()
    try:
        for ci,name in enumerate(cfg['controllers']):
            for fi,family in enumerate(families):
                for pi,profile in enumerate(profiles):
                    for offset in range(0,n,b):
                        seed=starts[fi]+offset; active=dict(controller=name,family=fi,profile=profile.identifier,seed=seed,step=0)
                        condition=RobustnessCondition.from_mapping(dict(family,base_seed=seed))
                        env=AdaptiveOpticsEnv(profile.environment_config(condition.environment_config(base)),device,profile.effects_config(),basis_override=basis)
                        raw,_=env.reset(seed=seed); sensor=torch.Generator(device=device).manual_seed(seed+parent['data']['sensor_seed_offset'])
                        proxy=lambda x:simulation_residual_proxy(x,generator=sensor,noise_std_rad=profile.observation_noise_std_rad)
                        interface=R4Interface(calibration=cal); interface.reset(proxy(raw),episode_id=str(seed))
                        frames=[interface.snapshot().features[:,-1].cpu()]; rows=[]
                        for step in range(t):
                            active['step']=step; snap=interface.snapshot(); h,v=snap.features,snap.valid
                            u=h.new_zeros(b,11); predicted=h.new_full((b,),float('nan')); zero=predicted.clone()
                            valid=torch.zeros(b,device=device,dtype=torch.bool); reasons=['baseline']*b; call_count=0
                            torch.cuda.synchronize(device); begin=time.perf_counter()
                            if name!='integrator':
                                eligible=range_eligible(h,v,bounds)
                                # 固定完整批量搜索：不同回退掩码不会重排其余天气的随机数。
                                if bool(eligible.any()):
                                    # 无效范围行用首个合格状态填充，保留随机数行位置；丢弃填充行输出。
                                    first=int(torch.nonzero(eligible)[0,0])
                                    plan_h=torch.where(eligible[:,None,None],h,h[first:first+1])
                                    plan_v=torch.where(eligible[:,None],v,v[first:first+1])
                                    result=selected_plan(models[name],plan_h,plan_v,spec,cal,SearchConfig(),
                                        seed=cfg['search_seed']+fi*100000+pi*10000+offset*200+step)
                                    u[eligible]=result['correction'][eligible]; predicted[eligible]=result['score'][eligible]
                                    zero[eligible]=result['zero_score'][eligible]; valid=eligible.clone()
                                    call_count=result['model_forward_samples']
                                u,reasons=guarded_correction(h,v,u,bounds)
                            calls+=call_count; planned_total+=int(valid.sum())
                            expected=selected_request(spec,h,v,u); delta,_=baseline_delta(spec,h,v,[],cal)
                            action=interface.issue(delta,u,step=step)
                            if not torch.equal(expected.requested_delta_rad,action.requested_delta_rad): raise RuntimeError('request mismatch')
                            torch.cuda.synchronize(device); latency=time.perf_counter()-begin
                            raw,_,terminated,truncated,info=env.step(action.requested_delta_rad); physical+=b
                            interface.observe_next(proxy(raw),step=step+1,power=PowerMeasurement(info['measured_power_in_bucket'],step,step+1))
                            if bool(truncated.any()) or not bool((terminated==(step==t-1)).all()): raise RuntimeError('incomplete episode')
                            if not torch.allclose(action.requested_modal_rad,info['requested_modal'],atol=1e-6,rtol=0): raise RuntimeError('environment action mismatch')
                            audit={k:info[k].cpu() for k in (*METRICS,'applied_modal')}
                            if any(not bool(torch.isfinite(x).all()) for x in audit.values()): raise RuntimeError('nonfinite physical metrics')
                            for reason in reasons: reasons_total[reason]=reasons_total.get(reason,0)+1
                            rows.append(dict(**audit,requested_delta=action.requested_delta_rad.cpu(),requested_modal=action.requested_modal_rad.cpu(),
                                correction=u.cpu(),predicted_score=predicted.cpu(),zero_score=zero.cpu(),predicted_valid=valid.cpu(),
                                reasons=reasons,model_forward_samples=call_count,latency_seconds=latency,terminated=terminated.cpu(),
                                measured_power=info['measured_power_in_bucket'].cpu()))
                            frames.append(interface.snapshot().features[:,-1].cpu())
                            progress.tick({'控制器':ci+1,'回合步':step+1,'物理转移':physical,'模型前向':calls})
                        ep=torch.stack([torch.stack([r[k] for r in rows],1).to(device).double().mean(1) for k in METRICS],-1)
                        means[ci,fi,pi,offset:offset+b]=ep
                        path=output/'trajectories'/f'{name}_{family["id"]}_{profile.identifier}_{seed}.pt'
                        torch.save(dict(**active,split='training_diagnostic' if quick else 'development',source='simulation_residual_proxy_not_holography',
                            seeds=list(range(seed,seed+b)),frames=torch.stack(frames,1),rows=rows),path)
                        manifest.append(dict(file=_relative(path),sha256=_file_sha256(path),**active))
                        with (output/'episode_metrics.jsonl').open('a',encoding='utf-8') as stream:
                            for j,vals in enumerate(ep.cpu().tolist()):
                                stream.write(json.dumps(dict(controller=name,family=fi,profile=profile.identifier,seed=seed+j,metrics=vals))+'\n')
        if physical!=report['physical_transitions'] or calls>report['max_model_forward_samples']: raise RuntimeError('budget mismatch')
        progress.close(); torch.save(dict(means=means.cpu(),metric_order=METRICS),output/'metrics.pt')
        write_json(output/'trajectory_manifest.json',manifest)
        analysis={} if quick else summarize(means,seed=cfg['bootstrap_seed'],repeats=cfg['bootstrap_repeats'])
        return dict(status='QUICK_COMPLETE_NO_RANKING' if quick else analysis['status'],analysis=analysis,physical_transitions=physical,
            model_forward_samples=calls,planned_actions=planned_total,fallback_counts=reasons_total,completed_batches=len(manifest))
    except Exception:
        write_json(output/'interrupted_context.json',dict(**active,physical_transitions=physical,model_forward_samples=calls))
        raise
    finally: progress.close()


def run(path: str | Path, *, mode: str, preflight_only: bool=False) -> dict:
    cfg,parent,spec,output,report=preflight(path,mode)
    if preflight_only: return {k:v for k,v in report.items() if k!='frozen_files'}
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if os.environ['CUBLAS_WORKSPACE_CONFIG'] not in (':4096:8',':16:8'): raise ValueError('unsupported CUDA setting')
    device=resolve_device('cuda'); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled=False; torch.backends.cuda.matmul.allow_tf32=False
    output.mkdir(parents=True,exist_ok=False); write_json(output/'preflight.json',report); write_json(output/'config.json',cfg)
    write_json(output/'runtime.json',dict(gpu=torch.cuda.get_device_name(device),torch=str(torch.__version__),git=safe_git_record()))
    try:
        models=model_ensembles(device); cal=NominalCalibration(**parent['nominal_calibration'])
        result=batch_check(models,spec,cal,output,device) if mode=='batch' else execute(cfg,parent,spec,output,report,models,cal,device)
        verify_hashes(report['frozen_files'])
        artifacts={_relative(p):_file_sha256(p) for p in output.rglob('*') if p.is_file()}
        write_json(output/'artifact_manifest.json',artifacts)
        result.update(training_updates=0,real_slm_actions=False,confirmation_access=False,automatic_retry=False,
            artifact_manifest_sha256=_file_sha256(output/'artifact_manifest.json'),
            material_passport=dict(origin_skill='academic-research-suite',origin_mode='run',origin_date=datetime.now(timezone.utc).isoformat(),
                verification_status='REQUIRES_AUDIT',version_label='r4_closed_loop_v1'),next_action='停止，等待只读审计，不自动训练。')
        write_json(output/'summary.json',result); write_json(output/'SUCCESS.json',dict(summary_sha256=_file_sha256(output/'summary.json')))
        return result
    except Exception:
        write_json(output/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False)); raise
