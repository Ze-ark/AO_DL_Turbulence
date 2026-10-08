"""持续受限放大两组完整闭环；正式执行必须由用户显式启动。"""
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

from src.rl.r4_closed_loop import model_ensembles
from src.rl.r4_amplitude_probe import scaled_action, require_close

SOURCES = 'configs/experiments/s4_r4_amplitude_closed_loop_v1_sources.json'


def summarize(means: torch.Tensor, baseline: torch.Tensor, *, seed: int, repeats: int=20000) -> dict:
    if means.shape != (2,3,6,32,6) or baseline.shape != (3,6,32,6):
        raise ValueError('incomplete paired episodes')
    if not bool(torch.isfinite(means).all()) or not bool(torch.isfinite(baseline).all()):
        raise ValueError('nonfinite paired metrics')
    all_values=torch.cat((baseline[None],means),0).double()
    values=all_values.mean(2); averages=values.mean((1,2))
    differences=torch.stack((values[2]-values[1],values[2]-values[0]))
    generator=torch.Generator(device=means.device).manual_seed(seed)
    boot=means.new_zeros((repeats,2),dtype=torch.float64)
    for f in range(3):
        draws=torch.randint(32,(repeats,32),device=means.device,generator=generator)
        for k in range(2): boot[:,k]+=differences[k,f,:,0][draws].mean(1)/3
    ci=torch.quantile(boot,means.new_tensor([.0125,.9875],dtype=torch.float64),dim=0).T
    if not bool((averages[:2,0]>0).all()): raise ValueError('nonpositive reference power')
    delta=differences.mean((1,2)); relative=averages[2,0]/averages[:2,0]-1
    gates=dict(increment_ci_positive=float(ci[0,0])>0,
        strehl_not_lower_than_original=float(delta[0,1])>=0,
        violation_increase_vs_original_at_most_point001=float(delta[0,2])<=.001,
        power_vs_integrator_at_least_one_percent=float(relative[0])>=.01,
        integrator_comparison_ci_positive=float(ci[1,0])>0,
        strehl_not_lower_than_integrator=float(delta[1,1])>=0,
        violation_increase_vs_integrator_at_most_point001=float(delta[1,2])<=.001)
    return dict(status='AMPLITUDE_CLOSED_LOOP_COMPLETE_REQUIRES_AUDIT', gates=gates,
        all_development_gates_pass=all(gates.values()), controller_order=['cached_integrator','original','scaled_1p5'],
        controller_means=averages.tolist(), comparisons=['scaled_minus_original','scaled_minus_integrator'],
        mean_metric_deltas=delta.tolist(), power_ci97p5_bonferroni=ci.tolist(),
        relative_power_gain_vs_integrator=float(relative[0]),relative_power_gain_vs_original=float(relative[1]),
        family_delta=differences.mean(2).tolist(), profile_delta=torch.stack((all_values[2]-all_values[1],all_values[2]-all_values[0])).mean((1,3)).tolist(),
        bootstrap_seed=seed,bootstrap_repeats=repeats,independent_confirmation=False,rl_authorized=False)


def preflight(path: str | Path, mode: str) -> tuple:
    if mode not in ('formal','quick'): raise ValueError('unknown mode')
    cfg=_load_yaml(_project_path(path)); own=read(_project_path(SOURCES)); verify_hashes(own)
    if _relative(_project_path(path)) not in own: raise ValueError('unfrozen config')
    spec,frozen=load_selected()
    up=_project_path(cfg['upstream']); summary=read(up/'summary.json')
    if (_file_sha256(up/'summary.json')!=cfg['upstream_sha256'] or
        read(up/'SUCCESS.json')['summary_sha256']!=cfg['upstream_sha256'] or
        _file_sha256(up/'artifact_manifest.json')!=summary['artifact_manifest_sha256'] or
        summary['status']!='AMPLITUDE_DIAGNOSTIC_COMPLETE_REQUIRES_AUDIT'):
        raise RuntimeError('amplitude diagnostic prerequisite changed')
    frozen.update(read(up/'preflight.json')['frozen_files']); frozen.update(read(up/'artifact_manifest.json')); frozen.update(own)
    for p in (up/'summary.json',up/'SUCCESS.json',up/'artifact_manifest.json',_project_path(SOURCES)):
        frozen[_relative(p)]=_file_sha256(p)
    if mode=='formal':
        q=_project_path(cfg['output_directory']+'_quick'); s=read(q/'summary.json')
        if (s['status']!='QUICK_COMPLETE_NO_RANKING' or
            read(q/'SUCCESS.json')['summary_sha256']!=_file_sha256(q/'summary.json') or
            _file_sha256(q/'artifact_manifest.json')!=s['artifact_manifest_sha256'] or
            read(q/'preflight.json')['frozen_files'].get(SOURCES)!=_file_sha256(_project_path(SOURCES))):
            raise RuntimeError('same-version quick prerequisite missing')
        frozen.update(read(q/'artifact_manifest.json'))
        for name in ('summary.json','SUCCESS.json','artifact_manifest.json'): frozen[_relative(q/name)]=_file_sha256(q/name)
    verify_hashes(frozen)
    parent=_load_yaml(_project_path(cfg['parent']))
    if (cfg['controllers']!=['gru_mpc','gru_mpc_x15'] or cfg['development_starts']!=[3762048,3762304,3762560]
        or cfg['physical_transitions']!=230400 or cfg['max_model_forward_samples']!=2831155200
        or cfg['search_seed']!=3769600 or cfg['training_updates']!=0 or cfg['automatic_retry'] is not False):
        raise ValueError('frozen experiment boundary changed')
    output=_project_path(cfg['output_directory']+('_quick' if mode=='quick' else ''))
    if output.exists(): raise FileExistsError(f'preserve existing output: {output}')
    if shutil.disk_usage(_project_path('.')).free<4096*1024**2: raise RuntimeError('need 4 GiB output space')
    device=resolve_device('cuda')
    report=dict(mode=mode,device=str(device),frozen_files=frozen,status='READY_FOR_USER_IDE' if mode=='formal' else 'READY_FOR_DIAGNOSTIC',
        physical_transitions=230400 if mode=='formal' else 48,max_model_forward_samples=2831155200 if mode=='formal' else 589824,
        training_updates=0,real_slm_actions=False,confirmation_access=False)
    return cfg,parent,spec,output,report


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
    means=torch.zeros(2,len(families),len(profiles),n,6,device=device,dtype=torch.float64)
    progress=Progress(output,device); progress.phase('原力度与受限1.5倍闭环' if not quick else '入口快速验证（不排名）',report['physical_transitions']//b)
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
                        reference=None
                        if not quick and name=='gru_mpc':
                            reference=torch.load(_project_path('outputs/s4_r4_closed_loop_v1/trajectories')/f'gru_mpc_{family["id"]}_{profile.identifier}_{seed}.pt',map_location=device,weights_only=True)
                            require_close(interface.snapshot().features[:,-1],reference['frames'][:,0],'original reset')
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
                                    result=selected_plan(models['gru_mpc'],plan_h,plan_v,spec,cal,SearchConfig(),
                                        seed=cfg['search_seed']+fi*100000+pi*10000+offset*200+step)
                                    u[eligible]=result['correction'][eligible]; predicted[eligible]=result['score'][eligible]
                                    zero[eligible]=result['zero_score'][eligible]; valid=eligible.clone()
                                    call_count=result['model_forward_samples']
                                u,reasons=guarded_correction(h,v,u,bounds)
                            before_scale=u.clone()
                            multiplier=1.5 if name=='gru_mpc_x15' else 1.
                            clipped=(before_scale*multiplier).abs()>1
                            u=scaled_action(before_scale,multiplier)
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
                            if reference is not None:
                                require_close(interface.snapshot().features[:,-1],reference['frames'][:,step+1],'original next frame')
                                require_close(u,reference['rows'][step]['correction'],'original correction')
                                for key in (*METRICS,'applied_modal'):
                                    require_close(info[key],reference['rows'][step][key],'original '+key)
                            for reason in reasons: reasons_total[reason]=reasons_total.get(reason,0)+1
                            rows.append(dict(**audit,requested_delta=action.requested_delta_rad.cpu(),requested_modal=action.requested_modal_rad.cpu(),
                                correction=u.cpu(),pre_scale_correction=before_scale.cpu(),clipped_components=clipped.cpu(),unscaled_plan_score=predicted.cpu(),zero_score=zero.cpu(),predicted_valid=valid.cpu(),
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
        analysis={} if quick else summarize(means,torch.load(_project_path('outputs/s4_r4_closed_loop_v1/metrics.pt'),map_location=device,weights_only=True)['means'][0],seed=cfg['bootstrap_seed'],repeats=cfg['bootstrap_repeats'])
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
        result=execute(cfg,parent,spec,output,report,models,cal,device)
        verify_hashes(report['frozen_files'])
        artifacts={_relative(p):_file_sha256(p) for p in output.rglob('*') if p.is_file()}
        write_json(output/'artifact_manifest.json',artifacts)
        result.update(training_updates=0,real_slm_actions=False,confirmation_access=False,automatic_retry=False,
            artifact_manifest_sha256=_file_sha256(output/'artifact_manifest.json'),
            material_passport=dict(origin_skill='academic-research-suite',origin_mode='run',origin_date=datetime.now(timezone.utc).isoformat(),
                verification_status='REQUIRES_AUDIT',version_label='r4_amplitude_closed_loop_v1'),next_action='停止，等待只读审计，不自动训练。')
        write_json(output/'summary.json',result); write_json(output/'SUCCESS.json',dict(summary_sha256=_file_sha256(output/'summary.json')))
        return result
    except Exception:
        write_json(output/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False)); raise
