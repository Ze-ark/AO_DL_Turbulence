"""按完整批次恢复；旧输出只读，新输出独立保存。"""
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

from src.rl.r4_amplitude_closed_loop import summarize
from collections import Counter

SOURCES = 'configs/experiments/s4_r4_amplitude_recovery_v1_sources.json'


def validate_batch(data: dict, expected: dict, *, batch: int, steps: int,
                   reference: dict | None = None) -> dict:
    """纯校验，不选择表现好的回合；损坏或不完整一律停止。"""
    for key in ('controller','family','profile','seed'):
        if data[key]!=expected[key]: raise ValueError('batch identity mismatch: '+key)
    if data['step']!=steps-1 or data['seeds']!=list(range(expected['seed'],expected['seed']+batch)):
        raise ValueError('incomplete batch identity')
    if data['source']!='simulation_residual_proxy_not_holography':
        raise ValueError('wrong observation source')
    frames=data['frames']; rows=data['rows']
    if frames.shape!=(batch,steps+1,79) or len(rows)!=steps or not bool(torch.isfinite(frames).all()):
        raise ValueError('incomplete/nonfinite batch')
    reasons=Counter(); calls=planned=0
    for step,row in enumerate(rows):
        for k in (*METRICS,'measured_power'):
            if row[k].shape!=(batch,) or not bool(torch.isfinite(row[k]).all()):
                raise ValueError('invalid metric: '+k)
        for k in ('requested_delta','requested_modal','applied_modal'):
            if row[k].shape!=(batch,21) or not bool(torch.isfinite(row[k]).all()):
                raise ValueError('invalid action: '+k)
        u=row['pre_scale_correction']; m=1.5 if expected['controller']=='gru_mpc_x15' else 1.
        if not torch.equal(row['correction'],scaled_action(u,m)):
            raise ValueError('scaled action mismatch')
        if row['clipped_components'].dtype!=torch.bool or not torch.equal(row['clipped_components'],(u*m).abs()>1):
            raise ValueError('clip mask mismatch')
        if row['terminated'].shape!=(batch,) or row['terminated'].dtype!=torch.bool or not bool((row['terminated']==(step==steps-1)).all()):
            raise ValueError('incomplete termination')
        require_close(row['requested_modal'],frames[:,step+1,21:42],'saved request')
        require_close(row['requested_delta'],frames[:,step+1,21:42]-frames[:,step,21:42],'saved delta')
        require_close(row['correction'],frames[:,step+1,63:74],'saved correction')
        require_close(row['measured_power'],frames[:,step+1,74],'saved power')
        if bool((row['requested_delta'].abs()>.150001).any()): raise ValueError('step bound')
        valid=row['predicted_valid']
        if valid.shape!=(batch,) or valid.dtype!=torch.bool: raise ValueError('prediction mask')
        for k in ('unscaled_plan_score','zero_score'):
            if row[k].shape!=(batch,) or not bool(torch.isfinite(row[k][valid]).all()) or not bool(torch.isnan(row[k][~valid]).all()):
                raise ValueError('invalid score mask')
        if len(row['reasons'])!=batch: raise ValueError('missing fallback reasons')
        if any(x not in ('accepted','outside_observation_range','nonfinite_proposal','proposal_out_of_bounds') for x in row['reasons']):
            raise ValueError('unknown fallback')
        for j,reason in enumerate(row['reasons']):
            if reason!='accepted' and bool(row['correction'][j].any()): raise ValueError('nonzero fallback')
        count=row['model_forward_samples']
        if count not in (0,batch*12288) or (bool(valid.any())!=(count>0)): raise ValueError('forward accounting mismatch')
        if reference is not None:
            for k in (*METRICS,'correction','requested_delta','requested_modal','applied_modal','measured_power'):
                require_close(row[k],reference['rows'][step][k],'reference '+k)
        calls+=count; planned+=int(valid.sum()); reasons.update(row['reasons'])
    if reference is not None: require_close(frames,reference['frames'],'reference frames')
    ep=torch.stack([torch.stack([r[k] for r in rows],1).double().mean(1) for k in METRICS],-1)
    return dict(metrics=ep.tolist(),model_forward_samples=calls,planned_actions=planned,
                fallback_counts=dict(reasons))


def inventory(parent: dict, cfg: dict, quick: bool) -> dict:
    batch,steps,n=(2,12,2) if quick else (16,200,32)
    starts=[3760000] if quick else cfg['development_starts']
    result={}
    for ci,name in enumerate(cfg['controllers']):
        for fi,family in enumerate(parent['families'][:1] if quick else parent['families']):
            for pi,profile in enumerate(parent['profile_ids'][:1] if quick else parent['profile_ids']):
                for offset in range(0,n,batch):
                    seed=starts[fi]+offset
                    filename=f'{name}_{family["id"]}_{profile}_{seed}.pt'
                    result[filename]=dict(controller=name,family=fi,profile=profile,seed=seed,
                        controller_index=ci,profile_index=pi,offset=offset,step=steps-1)
    return result


def preflight(path: str | Path, mode: str) -> tuple:
    if mode not in ('quick','formal'): raise ValueError('unknown recovery mode')
    own=read(_project_path(SOURCES)); verify_hashes(own)
    if _relative(_project_path(path)) not in own: raise ValueError('unfrozen recovery config')
    recovery=_load_yaml(_project_path(path)); quick=mode=='quick'
    source=_project_path(recovery['quick_source_directory' if quick else 'source_directory'])
    snapshot=read(_project_path(recovery['source_snapshot'])); verify_hashes(snapshot)
    # 固定目录清单，防止恢复期间混入新文件或遗漏已保存批次。
    expected_files={p for p in snapshot if p.startswith(_relative(source)+'/')}
    actual_files={_relative(p) for p in source.rglob('*') if p.is_file()}
    if expected_files!=actual_files: raise RuntimeError('source directory inventory changed')
    prior=read(source/'preflight.json'); verify_hashes(prior['frozen_files'])
    cfg=_load_yaml(_project_path(recovery['experiment_config']))
    if read(source/'config.json')!=cfg: raise ValueError('source experiment configuration changed')
    if prior['mode']!=mode: raise ValueError('source mode mismatch')
    if not quick and ((source/'SUCCESS.json').exists() or (source/'summary.json').exists()):
        raise ValueError('completed source must not be resumed')
    spec,frozen=load_selected(); frozen.update(prior['frozen_files']); frozen.update(snapshot); frozen.update(own)
    frozen[SOURCES]=_file_sha256(_project_path(SOURCES))
    frozen[_relative(_project_path(recovery['source_snapshot']))]=_file_sha256(_project_path(recovery['source_snapshot']))
    device=resolve_device('cuda')
    runtime=read(source/'runtime.json')
    if runtime['torch']!=str(torch.__version__) or runtime['gpu']!=torch.cuda.get_device_name(device):
        raise RuntimeError('recovery runtime changed; requires audit')
    parent=_load_yaml(_project_path(cfg['parent']))
    expected=inventory(parent,cfg,quick); available=list((source/'trajectories').glob('*.pt'))
    if any(p.name not in expected for p in available): raise ValueError('unknown saved trajectory')
    if len(available)!=(2 if quick else recovery['expected_reused_batches']):
        raise ValueError('saved batch count changed')
    b,t=(2,12) if quick else (16,200)
    verified={}
    # 快速恢复故意仅复用原力度组，重算放大组，用旧完整快速结果检查恢复等价性。
    for file in available:
        meta=expected[file.name]; data=torch.load(file,map_location=device,weights_only=True)
        if data['split']!=('training_diagnostic' if quick else 'development'): raise ValueError('split changed')
        reference=None
        if meta['controller']=='gru_mpc':
            ref=_project_path('outputs/s4_r4_closed_loop_v1'+('_quick' if quick else ''))/'trajectories'/file.name
            reference=torch.load(ref,map_location=device,weights_only=True)
        audit=validate_batch(data,meta,batch=b,steps=t,reference=reference)
        if quick and meta['controller']!='gru_mpc': continue
        verified[file.name]=dict(file=_relative(file),sha256=_file_sha256(file),**meta,**audit)
    # 对照已落盘逐天气汇总，避免接入错误批次或重复天气。
    episode_rows=[json.loads(line) for line in (source/'episode_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    row_index={}
    for row in episode_rows:
        key=(row['controller'],row['family'],row['profile'],row['seed'])
        if key in row_index: raise ValueError('duplicate episode row')
        row_index[key]=row['metrics']
    if len(row_index)!=len(available)*b: raise ValueError('incomplete saved episode index')
    for record in verified.values():
        for j,metrics in enumerate(record['metrics']):
            saved=row_index[(record['controller'],record['family'],record['profile'],record['seed']+j)]
            if metrics!=saved: raise ValueError('episode mean disagrees with trajectory')
    last=json.loads((source/'progress.jsonl').read_text(encoding='utf-8').splitlines()[-1])
    spent=last['物理转移']; old_calls=last['模型前向']
    reused=len(verified)*b*t; reused_calls=sum(v['model_forward_samples'] for v in verified.values())
    if spent<reused or old_calls<reused_calls: raise ValueError('source progress accounting')
    if not quick:
        q=_project_path(recovery['output_directory']+'_quick'); qs=read(q/'summary.json')
        if (qs['status']!='QUICK_COMPLETE_NO_RANKING' or not qs['quick_recovery_matches_source']
            or read(q/'SUCCESS.json')['summary_sha256']!=_file_sha256(q/'summary.json')
            or _file_sha256(q/'artifact_manifest.json')!=qs['artifact_manifest_sha256']
            or read(q/'preflight.json')['frozen_files'].get(SOURCES)!=_file_sha256(_project_path(SOURCES))):
            raise RuntimeError('same-version recovery smoke required')
        frozen.update(read(q/'artifact_manifest.json'))
        for name in ('summary.json','SUCCESS.json','artifact_manifest.json'): frozen[_relative(q/name)]=_file_sha256(q/name)
    verify_hashes(frozen)
    output=_project_path(recovery['output_directory']+('_quick' if quick else ''))
    if output.exists(): raise FileExistsError(f'preserve recovery output: {output}')
    if shutil.disk_usage(_project_path('.')).free<4096*1024**2: raise RuntimeError('need 4 GiB free')
    missing=len(expected)-len(verified)
    report=dict(mode=mode,status='READY_FOR_RECOVERY_SMOKE' if quick else 'READY_FOR_USER_IDE',
        frozen_files=frozen,reused=verified,reused_batches=len(verified),missing_batches=missing,
        physical_transitions=missing*b*t,max_model_forward_samples=missing*b*t*12288,
        reused_physical_transitions=reused,reused_model_forward_samples=reused_calls,
        prior_recorded_physical_transitions=spent,prior_recorded_model_forward_samples=old_calls,
        discarded_recorded_physical_transitions=spent-reused,source_directory=_relative(source),
        device=str(device),training_updates=0,real_slm_actions=False,confirmation_access=False)
    return cfg,parent,spec,output,report


def append_episode_rows(output: Path, meta: dict, metrics: list) -> None:
    with (output/'episode_metrics.jsonl').open('a',encoding='utf-8') as stream:
        for j,vals in enumerate(metrics):
            stream.write(json.dumps(dict(controller=meta['controller'],family=meta['family'],
                profile=meta['profile'],seed=meta['seed']+j,metrics=vals))+'\n')


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
                        filename=f'{name}_{family["id"]}_{profile.identifier}_{seed}.pt'
                        if filename in report['reused']:
                            record=report['reused'][filename]
                            means[ci,fi,pi,offset:offset+b]=torch.tensor(record['metrics'],device=device,dtype=torch.float64)
                            manifest.append(dict(file=record['file'],sha256=record['sha256'],reused=True,**dict(active,step=t-1)))
                            append_episode_rows(output,active,record['metrics'])
                            planned_total+=record['planned_actions']
                            for reason,count in record['fallback_counts'].items():
                                reasons_total[reason]=reasons_total.get(reason,0)+count
                            print(f"已核验复用：{filename}",flush=True)
                            continue
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
                        manifest.append(dict(file=_relative(path),sha256=_file_sha256(path),reused=False,**active))
                        with (output/'completed_batch_receipts.jsonl').open('a',encoding='utf-8') as stream:
                            stream.write(json.dumps(manifest[-1])+'\n')
                        with (output/'episode_metrics.jsonl').open('a',encoding='utf-8') as stream:
                            for j,vals in enumerate(ep.cpu().tolist()):
                                stream.write(json.dumps(dict(controller=name,family=fi,profile=profile.identifier,seed=seed+j,metrics=vals))+'\n')
        if physical!=report['physical_transitions'] or calls>report['max_model_forward_samples']: raise RuntimeError('budget mismatch')
        progress.close(); torch.save(dict(means=means.cpu(),metric_order=METRICS),output/'metrics.pt')
        write_json(output/'trajectory_manifest.json',manifest)
        quick_matches=False
        if quick:
            source=_project_path(report['source_directory'])
            original_means=torch.load(source/'metrics.pt',map_location=device,weights_only=True)['means']
            if not torch.equal(means,original_means): raise RuntimeError('recovery smoke aggregate mismatch')
            for record in manifest:
                if record['reused']: continue
                fresh=torch.load(_project_path(record['file']),map_location=device,weights_only=True)
                prior=torch.load(source/'trajectories'/Path(record['file']).name,map_location=device,weights_only=True)
                if not torch.equal(fresh['frames'],prior['frames']): raise RuntimeError('recovery smoke trajectory mismatch')
                for a,z in zip(fresh['rows'],prior['rows']):
                    for key in (*METRICS,'correction','requested_delta','requested_modal','applied_modal'):
                        if not torch.equal(a[key],z[key]): raise RuntimeError('recovery smoke action mismatch')
            quick_matches=True
        analysis={} if quick else summarize(means,torch.load(_project_path('outputs/s4_r4_closed_loop_v1/metrics.pt'),map_location=device,weights_only=True)['means'][0],seed=cfg['bootstrap_seed'],repeats=cfg['bootstrap_repeats'])
        return dict(status='QUICK_COMPLETE_NO_RANKING' if quick else analysis['status'],analysis=analysis,physical_transitions=physical,
            quick_recovery_matches_source=quick_matches,reused_batches=report['reused_batches'],new_batches=report['missing_batches'],
            logical_physical_transitions=physical+report['reused_physical_transitions'],
            logical_model_forward_samples=calls+report['reused_model_forward_samples'],
            cumulative_recorded_physical_transitions=physical+report['prior_recorded_physical_transitions'],
            cumulative_recorded_model_forward_samples=calls+report['prior_recorded_model_forward_samples'],
            discarded_recorded_physical_transitions=report['discarded_recorded_physical_transitions'],
            model_forward_samples=calls,planned_actions=planned_total,fallback_counts=reasons_total,completed_batches=len(manifest))
    except Exception:
        write_json(output/'interrupted_context.json',dict(**active,physical_transitions=physical,model_forward_samples=calls))
        raise
    finally: progress.close()


def run(path: str | Path, *, mode: str, preflight_only: bool=False) -> dict:
    cfg,parent,spec,output,report=preflight(path,mode)
    if preflight_only: return {k:v for k,v in report.items() if k not in ('frozen_files','reused')}
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
                verification_status='REQUIRES_AUDIT',version_label='r4_amplitude_recovery_v1'),next_action='停止，等待只读审计，不自动训练。')
        write_json(output/'summary.json',result); write_json(output/'SUCCESS.json',dict(summary_sha256=_file_sha256(output/'summary.json')))
        return result
    except Exception:
        write_json(output/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False)); raise
