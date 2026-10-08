"""O2-D3：四档人工读出噪声、冻结控制器的完整回合开发对照。

复用封存 D2 因果回合和 C 完整天气统计，扩展到全部噪声档的配对变化。
不训练、不访问旧确认轨迹、不重新拟合归一化器、不修改观测保护。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback
from typing import Any

import torch

from observation_bridge import noise_closed_loop as technical
from src.runtime import resolve_device

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/experiments/observation_bridge_o2_d3_noise_development_v1.yaml"
prior, short, optics = technical.prior, technical.short, technical.optics
D2_PINS = {
    "observation_bridge/noise_closed_loop.py": "a82e6b415d7b16182ddd72ee128f72b9956ed45d3efabaf6c6b8618ce3cb28a7",
    "scripts/verify_observation_bridge_o2_noise_closed_loop.py": "17364f515584a0b1c776a1338a705eed4065e40fe367fce68ab8044d44fb8bd8",
    "tests/test_observation_bridge_o2_noise_closed_loop.py": "8fbf9ed1cc8631bf5e02b9edcc19e31c934b67ea17c8032eb77c7c6bdc3f2597",
    technical.CONFIG: "5001088a562298de82ae15272c36a7e5d2971af0d694d8250bc65c2673d83377",
}
D2_OUTPUTS = {
    "outputs/observation_bridge_o2_d2_noise_closed_loop_v1":
        ("e55bb14be5e636ce68f17570312066c4e6b6246933d6a7db2551d8be977ad17e", 425, False),
    "outputs/observation_bridge_o2_d2_noise_closed_loop_v1_quick":
        ("313e57cd14f51cd168c44f7259a8570a9072412e1258371e75b8a0ecf3a809af", 61, True),
}


def read_config(path: str | Path = CONFIG) -> dict[str, Any]:
    return prior.read_config(path)


def validate_config(cfg: dict[str, Any]) -> None:
    expected = dict(
        schema="observation_bridge_o2_d3_noise_development_v1",
        scope="synthetic_read_noise_development_not_confirmation", device="cuda",
        optics_config="configs/experiments/observation_bridge_o2_optics_v1.yaml",
        parent="configs/experiments/s4_r5_policy_training_v1.yaml",
        data=dict(weather_seed_base=8700000, weather_count=8, weather_seed_stride=10, episode_length=200,
                  camera_ids=["noiseless", "small_read_noise", "medium_read_noise", "large_read_noise"]),
        quick=dict(weather_seed_base=8760000, weather_count=1, weather_seed_stride=10, episode_length=28,
                   camera_ids=["noiseless", "large_read_noise"]),
        family_ids=["frozen", "boiling", "varying"],
        camera_conditions=[dict(id="noiseless", read_noise_std=0.0), dict(id="small_read_noise", read_noise_std=.001),
                           dict(id="medium_read_noise", read_noise_std=.3), dict(id="large_read_noise", read_noise_std=1.0)],
        camera_seed_offset=170000000,
        camera_noise_distribution="additive_gaussian_then_clip_negative_intensity_to_zero",
        camera_noise_units="arbitrary_synthetic_intensity", camera_draw_including_noiseless=True,
        batch_size=3, controller_branches=13, policy_initializations=[0, 1, 2],
        scorer_seeds=[7564000, 7564001, 7564002], selector_start_step=25,
        integrator=dict(gain=.15, leak=.1, tracking_gain=.5), policy_scale=1.75, candidate_epsilon=.1,
        thresholds=dict(ideal_modal_error_max_rad=.001, noisy_modal_rmse_mean_rad=.01,
                        noisy_modal_rmse_p95_rad=.02, replay_max_absolute_error=1e-6,
                        invalid_observations=0, failed_or_truncated_episodes=0),
        statistics=dict(cluster="complete_weather", stratification="fixed_scorer_fold", bootstrap_seed=8790000,
                        bootstrap_repeats=5000, interval=.95,
                        interval_scope="pointwise_descriptive_not_simultaneous_not_confirmation"), gain_threshold=None,
        boundary=dict(training_updates=0, independent_confirmation=False, old_confirmation_trajectory_access=False,
                      real_data_access=False, real_slm_actions=False, historical_gate_reclassification=False,
                      truth_fallback=False, automatic_retry=False))
    if (not isinstance(cfg, dict) or set(cfg) != set(expected) | {"output_directory", "quick_directory"}
            or any(json.dumps(cfg.get(k), sort_keys=True, allow_nan=False) != json.dumps(v, sort_keys=True)
                   for k, v in expected.items())):
        raise ValueError("O2-D3 fixed development contract changed")


def budget(spec: dict[str, Any]) -> dict[str, int]:
    return technical.budget(spec)


def verify_prerequisites() -> dict[str, Any]:
    for name, digest in D2_PINS.items():
        if optics.file_sha256(ROOT / name) != digest:
            raise RuntimeError(f"O2-D3 frozen D2 source changed: {name}")
    result = technical.verify_prerequisites()
    dcfg = technical.read_config()
    technical.validate_config(dcfg)
    for relative, (digest, count, quick) in D2_OUTPUTS.items():
        directory = ROOT / relative
        success, summary = short.read_json(directory / "SUCCESS.json"), short.read_json(directory / "summary.json")
        names = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
                 if p.is_file() and p.name != "SUCCESS.json"}
        status = "O2_D2_TECHNICAL_SMOKE_ONLY" if quick else "O2_D2_COMPLETE_REQUIRES_READ_ONLY_AUDIT"
        if (len(names) != count or set(success["artifact_sha256"]) != names
                or success["summary_sha256"] != digest or optics.file_sha256(directory / "summary.json") != digest
                or success["status"] != status or summary["status"] != status
                or (directory / "failure.json").exists() or (directory / "partial").exists()):
            raise RuntimeError("O2-D3 completed D2 seal changed")
        for name, expected in success["artifact_sha256"].items():
            target = (directory / name).resolve()
            if not target.is_relative_to(directory.resolve()) or optics.file_sha256(target) != expected:
                raise RuntimeError("O2-D3 completed D2 artifact changed")
        if (short.read_json(directory / "source_manifest.json") != technical.source_manifest(technical.CONFIG)
                or short.read_json(directory / "effective_config.json") != dcfg):
            raise RuntimeError("O2-D3 D2 executed source identity changed")
        expected_budget = technical.budget(dcfg["quick" if quick else "data"])
        if (summary["completed_episodes"] != expected_budget["complete_episodes"]
                or summary["completed_physical_transitions"] != expected_budget["physical_transitions"]
                or summary["invalid_observations"] != 0 or summary["failed_or_truncated_episodes"] != 0
                or summary["replay_max_absolute_error"] != 0 or not summary["observation_quality"]["targets_met"]):
            raise RuntimeError("O2-D3 D2 technical prerequisite not passed")
    result.update(D2_artifacts_checked=486, D2_output_summary_sha256={p:v[0] for p,v in D2_OUTPUTS.items()},
                  frozen_D2_source_sha256=D2_PINS)
    return result


def stream_manifest(cfg: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    streams = technical.stream_manifest(cfg, quick=quick)
    current = set().union(*(set(streams[k]) for k in
                           ("weather_bases", "turbulence", "power", "unused_proxy_sensor", "camera")))
    dcfg = technical.read_config()
    d2 = [technical.stream_manifest(dcfg, quick=q) for q in (False, True)]
    bootstrap = cfg["statistics"]["bootstrap_seed"]
    reserved = set(range(8770000, 8780000))
    reserved |= {n + offset for n in range(8770000, 8780000) for offset in (50000000, 60000000, 170000000)}
    if (current & reserved or bootstrap in current or bootstrap in reserved
            or any((current | {bootstrap}) & short._integers(m) for m in d2)):
        raise RuntimeError("O2-D3 D2/bootstrap/unit stream collision")
    # 同一封存清单检查也用于统计种子；仅生成种子清单，不执行旧实验。
    probe = {**cfg, "data": {**cfg["data"], "weather_seed_base": bootstrap, "weather_count": 1}}
    technical.stream_manifest(probe, quick=False)
    # 统计种子也与更早的已声明流及另一 D3 模式互斥。
    old = prior._streams(read_config(prior.CONFIG), read_config(prior.CONFIG)["data"])
    other = prior._streams(cfg, cfg["data" if quick else "quick"])
    static_cfg = technical.static.read_config()
    older = [old, prior._streams(read_config(prior.CONFIG), read_config(prior.CONFIG)["quick"]), other]
    older.extend(technical.static.stream_manifest(static_cfg, quick=q) for q in (False, True))
    if any(bootstrap in short._integers(m) for m in older):
        raise RuntimeError("O2-D3 statistical seed collides with declared history")
    streams.update(historical_manifests_checked=19, technical_unit_namespace=8770000,
                   bootstrap_seed=bootstrap, bootstrap_draws_used=not quick,
                   disjointness_scope="declared_G2_C1_B_C_D1_D2_other_D3_mode_and_reserved_units")
    return streams


def source_manifest(path: str | Path) -> dict[str, Any]:
    target = Path(path) if Path(path).is_absolute() else ROOT / path
    names = ["observation_bridge/noise_development.py", "scripts/evaluate_observation_bridge_o2_noise_development.py",
             "tests/test_observation_bridge_o2_noise_development.py"]
    return dict(new_source_sha256={n: optics.file_sha256(ROOT / n) for n in names},
                config_sha256=optics.file_sha256(target), frozen_D2_source=technical.source_manifest(technical.CONFIG))


def preflight(path: str | Path = CONFIG, *, quick: bool = False) -> tuple:
    cfg = read_config(path); validate_config(cfg)
    spec = cfg["quick" if quick else "data"]
    output = (ROOT / cfg["quick_directory" if quick else "output_directory"]).resolve()
    if output == (ROOT / "outputs").resolve() or not output.is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("O2-D3 output must be a new child of outputs")
    if output.exists():
        raise FileExistsError(f"preserve existing O2-D3 output: {output}")
    frozen, streams = verify_prerequisites(), stream_manifest(cfg, quick=quick)
    short.configure_runtime()
    device = resolve_device(cfg["device"])
    if device.type != "cuda":
        raise RuntimeError("O2-D3 requires CUDA; no CPU fallback")
    parent = read_config(cfg["parent"])
    policies, scorers, models = short.load_assets(device, parent)
    if any(w in m.get("train_weather", []) + m.get("held_out_weather", [])
           for m in models for w in streams["weather_bases"]):
        raise RuntimeError("O2-D3 weather used by frozen scorer")
    report = dict(status="O2_D3_READY_FOR_TECHNICAL_SMOKE" if quick else "O2_D3_READY_FOR_USER_IDE", quick=quick,
                  **cfg["boundary"], **budget(spec), episode_length=spec["episode_length"], weather_count=spec["weather_count"],
                  camera_ids=spec["camera_ids"], device=str(device), loaded_policies=len(policies), loaded_scorers=len(scorers),
                  output_directory=str(output), frozen_sources=frozen, stream_manifest=streams, gain_threshold=None,
                  scientific_gain_analysis=not quick, preflight_model_forward_calls=0, preflight_environment_transitions=0,
                  synthetic_nominal_profile=short.nominal_profile(parent).as_record(),
                  fold_routing="weather_index_modulo_4_fixed_before_observation")
    return cfg, spec, output, device, parent, policies, scorers, models, report


def active_config(cfg: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    return {**cfg, "camera_conditions":[c for c in cfg["camera_conditions"] if c["id"] in spec["camera_ids"]]}


@torch.no_grad()
def episode_rows(trace: dict, audit: dict, record: dict, cfg: dict, spec: dict, filename: str) -> list[dict]:
    values = dict(power=audit["action_power"], strehl=audit["action_strehl"], phase_rmse=audit["action_phase_rmse"],
        violation=audit["violation"], requested_applied_gap_rad=(audit["requested_modal"]-audit["applied_modal"]).abs().mean(-1),
        requested_step_abs_rad=trace["requested_delta"].abs().mean(-1), requested_modal_abs_rad=trace["requested_modal"].abs().mean(-1),
        observation_latency_ms=audit["observation_latency_ms"], decision_latency_ms=audit["decision_latency_ms"],
        camera_negative_clip_fraction=audit["camera_negative_clip_fraction"], modal_rmse_rad=audit["modal_rmse_rad"],
        representation_fit_rmse_rad=audit["fit_rmse_rad"])
    if set(values) != set(prior.METRICS) or any(v.device.type != "cuda" or not bool(torch.isfinite(v).all()) for v in values.values()):
        raise ValueError("O2-D3 episode metrics must be finite CUDA values")
    means = {k:v.double().mean(0) for k,v in values.items()}
    rows = []
    for fi,family in enumerate(cfg["family_ids"]):
        rows.append(dict(weather_seed=record["weather_seed"], weather_index=record["weather_index"], family=family, family_index=fi,
            camera_condition=record["camera_condition"], read_noise_std=record["read_noise_std"], controller=record["controller"],
            member=record["member"], scorer_seed=record["scorer_seed"], scorer_fold=record["scorer_fold"],
            episode_length=spec["episode_length"], turbulence_stream_seed=record["weather_seed"]+fi,
            power_stream_seed=record["weather_seed"]+60000000, camera_stream_seed=record["weather_seed"]+fi+cfg["camera_seed_offset"],
            camera_frames=record["camera_frames_per_family"], failed=False, truncated=False, trajectory_file=filename,
            selected_nonoriginal_fraction=float(trace["choice"][:,fi].ne(0).float().mean()),
            **{k:float(v[fi]) for k,v in means.items()}))
    return rows


def validate_rows(rows: list[dict], cfg: dict, spec: dict) -> dict:
    mapping = prior.validate_rows(rows, active_config(cfg,spec), spec)
    stds={c["id"]:c["read_noise_std"] for c in cfg["camera_conditions"]}
    for r in rows:
        expected=f"weather_{r['weather_index']:02d}_{r['camera_condition']}_{r['controller']}.pt"
        if r["read_noise_std"]!=stds[r["camera_condition"]] or r["trajectory_file"]!=expected:
            raise ValueError("O2-D3 camera/trajectory provenance mismatch")
        if any(r[k]<0 for k in prior.METRICS) or any(r[k]>1+1e-6 for k in ("power","strehl")):
            raise ValueError("O2-D3 metrics outside physical domain")
    return mapping


def _power_clusters(mapping: dict, cfg: dict, spec: dict, device: torch.device) -> dict:
    branches=prior.controller_specs(cfg)
    groups=dict(integrator=[branches[0]], original=[b for b in branches if b["member"] is not None and b["scorer_seed"] is None],
                current=[b for b in branches if b["scorer_seed"] is not None])
    weather=[spec["weather_seed_base"]+i*spec["weather_seed_stride"] for i in range(spec["weather_count"])]
    return {c["id"]:{name:torch.tensor([[[mapping[(c["id"],w,b["controller"],f)]["power"] for f in cfg["family_ids"]]
                    for b in members] for w in weather],device=device,dtype=torch.float64).mean((1,2))
                    for name,members in groups.items()} for c in cfg["camera_conditions"]}


@torch.no_grad()
def summarize(rows: list[dict], cfg: dict, spec: dict, *, device: torch.device, quick: bool) -> dict:
    mapping=validate_rows(rows,cfg,spec)
    if quick:
        return {}
    if device.type!='cuda':
        raise ValueError("O2-D3 scientific development statistics require CUDA")
    result=prior.summarize(rows,cfg,spec,device=device,quick=False)
    # C 的同名噪声对照仅指 0.001；不把它冒称所有噪声档的效应。
    result.pop("camera_noise_effect")
    result.pop("current_increment_change_noisy_minus_noiseless")
    power=_power_clusters(mapping,cfg,spec,device)
    draws=prior.stratified_draws(8,repeats=cfg["statistics"]["bootstrap_repeats"],seed=cfg["statistics"]["bootstrap_seed"],device=device)
    zero=power['noiseless'];effects={}
    for camera in cfg['camera_conditions'][1:]:
        current=power[camera['id']]
        comparisons={}
        for method in ('integrator','original','current'):
            difference=current[method]-zero[method]
            comparisons[method]=dict(noisy_minus_noiseless_power=float(difference.mean()),
                                    descriptive_ci95=prior.paired_interval(difference,draws),
                                    per_weather_power_difference=difference.tolist())
        increment=(current['current']-current['integrator'])-(zero['current']-zero['integrator'])
        noisy_gain=(current['current'].mean()-current['integrator'].mean())/current['integrator'].mean()
        zero_gain=(zero['current'].mean()-zero['integrator'].mean())/zero['integrator'].mean()
        noisy_boot=(current['current'][draws].mean(1)-current['integrator'][draws].mean(1))/current['integrator'][draws].mean(1)
        zero_boot=(zero['current'][draws].mean(1)-zero['integrator'][draws].mean(1))/zero['integrator'][draws].mean(1)
        change=noisy_boot-zero_boot
        if not bool(torch.isfinite(change).all()):
            raise ValueError('O2-D3 nonfinite noise effect')
        effects[camera['id']]=dict(read_noise_std=camera['read_noise_std'],method_power_changes=comparisons,
            current_increment_change_noisy_minus_noiseless=dict(mean_power_difference=float(increment.mean()),
                descriptive_ci95=prior.paired_interval(increment,draws)),
            current_relative_gain_change_fraction=float(noisy_gain-zero_gain),
            current_relative_gain_change_descriptive_ci95_fraction=[float(v) for v in torch.quantile(change,change.new_tensor([.025,.975]))])
    result.update(camera_noise_effects_vs_noiseless=effects,
                  interval_scope='pointwise_descriptive_conditional_on_frozen_models_and_fixed_fold_routing_not_simultaneous',
                  statistical_backend='cuda',quality_and_safety_not_implied_by_power_gain=True,
                  camera_noise_units=cfg['camera_noise_units'],real_camera_noise_calibrated=False)
    return result


@torch.no_grad()
def run(path: str | Path = CONFIG, *, quick: bool = False, preflight_only: bool = False) -> dict[str, Any]:
    cfg,spec,output,device,parent,policies,scorers,models,report=preflight(path,quick=quick)
    if preflight_only:return report
    output.mkdir(parents=True,exist_ok=False)
    context=dict(physical_transitions=0,completed_episode_batches=0,incomplete_batch_size=0)
    started=time.perf_counter()
    try:
        for name in ('trajectories','audit'):(output/name).mkdir()
        for name,value in (('effective_config.json',cfg),('preflight.json',report),('model_manifest.json',models),
                           ('stream_manifest.json',report['stream_manifest']),('source_manifest.json',source_manifest(path))):
            optics.write_json(output/name,value)
        sensor,bridge=optics.make_components(read_config(cfg['optics_config']),device)
        rows,records,manifest=[],[],[]
        rmse:dict[str,list[torch.Tensor]]={}
        ideal_max=replay_max=0.;prefixes=policy_calls=scorer_calls=progress_count=0
        rng_states:dict[int,list[str]]={}
        def progress(row:dict)->None:
            nonlocal progress_count
            progress_count+=1
            elapsed=time.perf_counter()-started;count,total=row['physical_transitions'],report['physical_transitions']
            row.update(total_physical_transitions=total,elapsed_seconds=elapsed,eta_seconds=elapsed/count*(total-count),
                       transitions_per_second=count/max(elapsed,1e-9),cuda_allocated_gb=torch.cuda.memory_allocated(device)/2**30)
            with (output/'progress.jsonl').open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            if row['observation_step']%20==0 or row['observation_step']==spec['episode_length']:
                print(f"O2-D3 {'技术冒烟' if quick else '开发对照'} {count}/{total} | 天气{row['weather_seed']} "
                      f"{row['camera_condition']} {row['controller']} {row['observation_step']}/{spec['episode_length']}帧 | "
                      f"速度={row['transitions_per_second']:.1f}转移/s | 剩余={row['eta_seconds']:.1f}s | "
                      f"显存={row['cuda_allocated_gb']:.2f}GB",flush=True)
        for wi,seed in enumerate(report['stream_manifest']['weather_bases']):
            for camera in active_config(cfg,spec)['camera_conditions']:
                originals={}
                for branch in prior.controller_specs(cfg):
                    policy=policies.get(branch['member'])
                    selector=None if branch['scorer_seed'] is None else scorers[wi%4,branch['scorer_seed']]
                    trace,audit,record=technical.rollout(cfg,spec,parent,branch,seed=seed,weather_index=wi,camera=camera,
                        sensor=sensor,bridge=bridge,policy=policy,selector=selector,progress=progress,
                        context=context,partial_directory=output/'partial')
                    context.update(phase='save_and_replay',incomplete_batch_size=0)
                    name=f"weather_{wi:02d}_{camera['id']}_{branch['controller']}.pt"
                    for directory,values in (('trajectories',trace),('audit',audit)):
                        technical.save_tensors(output/directory/name,values)
                    visible=torch.load(output/'trajectories'/name,map_location=device,weights_only=True)
                    record['replay']=short.replay_visible(visible,{**cfg,'episode_length':spec['episode_length']},policy,selector)
                    replay_max=max(replay_max,record['replay']['max_absolute_error'])
                    if selector is None and branch['member'] is not None:originals[branch['member']]=trace
                    if selector is not None:
                        short.require_prefix(originals[branch['member']],trace,cfg['selector_start_step'])
                        record['paired_prefix_exact']=True;prefixes+=1
                    final_rng=record['camera_final_rng_sha256']
                    if seed in rng_states and rng_states[seed]!=final_rng:
                        raise RuntimeError('O2-D3 exogenous camera draw count differs across branches/levels')
                    rng_states[seed]=final_rng
                    rmse.setdefault(camera['id'],[]).append(audit['modal_rmse_rad'].detach().clone())
                    if camera['read_noise_std']==0:ideal_max=max(ideal_max,record['modal_error_max_rad'])
                    part=episode_rows(trace,audit,record,cfg,spec,name)
                    rows.extend(part);records.append(record)
                    manifest.append(dict(file=name,**branch,weather_seed=seed,camera_condition=camera['id'],
                        visible_sha256=optics.file_sha256(output/'trajectories'/name),audit_sha256=optics.file_sha256(output/'audit'/name)))
                    for filename,values in (('records.jsonl',part),('batch_records.jsonl',[record])):
                        with (output/filename).open('a',encoding='utf-8') as handle:
                            for value in values:handle.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
                    context['completed_episode_batches']+=1
                    policy_calls+=record['policy_forward_calls'];scorer_calls+=record['scorer_forward_calls']
        technical.validate_records(records,cfg,spec);validate_rows(rows,cfg,spec)
        if (context['physical_transitions']!=report['physical_transitions'] or progress_count!=report['batched_steps']
                or len(rows)!=report['complete_episodes'] or policy_calls!=report['policy_forward_calls']
                or scorer_calls!=report['scorer_forward_calls'] or prefixes!=report['paired_prefix_checks']):
            raise RuntimeError('O2-D3 complete execution budget mismatch')
        context.update(phase='statistics',incomplete_batch_size=0)
        quality=technical.observation_quality(rmse,ideal_max,cfg)
        analysis=summarize(rows,cfg,spec,device=device,quick=quick)
        verify_prerequisites()
        if short.read_json(output/'source_manifest.json')!=source_manifest(path):
            raise RuntimeError('O2-D3 source/config changed during execution')
        git=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',check=False)
        result=dict(report,status='O2_D3_TECHNICAL_SMOKE_ONLY' if quick else 'O2_D3_DEVELOPMENT_COMPLETE_REQUIRES_READ_ONLY_AUDIT',
            completed_episodes=len(rows),completed_episode_batches=len(records),completed_physical_transitions=context['physical_transitions'],
            invalid_observations=0,failed_or_truncated_episodes=0,completed_paired_prefix_checks=prefixes,
            camera_rng_pair_checks=len(records),replay_max_absolute_error=replay_max,
            completed_policy_forward_calls=policy_calls,completed_scorer_forward_calls=scorer_calls,
            replay_policy_forward_calls=policy_calls,replay_scorer_forward_calls=scorer_calls,replay_environment_transitions=0,
            observation_quality=quality,analysis=analysis,raw_camera_frames_saved=0,inverse_crime_limitation=True,
            real_accuracy_verified=False,real_camera_noise_calibrated=False,realtime_verified=False,
            elapsed_seconds=time.perf_counter()-started,
            runtime=dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(device),deterministic=True,allow_tf32=False,
                cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],git_head=git.stdout.strip() if git.returncode==0 else None),
            latency_scope='batch-3 CUDA image measurement; decision excludes safety projection; excludes audit/IO/exposure; not real end-to-end latency',
            next_action='Read-only audit; no automatic rerun, tuning, training, confirmation or hardware actions.')
        optics.write_json(output/'trajectory_manifest.json',manifest);optics.write_json(output/'summary.json',result)
        artifacts={p.relative_to(output).as_posix():optics.file_sha256(p) for p in output.rglob('*') if p.is_file()}
        optics.write_json(output/'SUCCESS.json',dict(status=result['status'],summary_sha256=artifacts['summary.json'],artifact_sha256=artifacts))
        return result
    except BaseException as exc:
        optics.write_json(output/'failure.json',dict(status='O2_D3_STOPPED_NO_AUTOMATIC_RETRY',exception=type(exc).__name__,message=str(exc),
            rejection_type=technical.static.rejection_type(exc) if isinstance(exc,ValueError) else None,
            traceback=traceback.format_exc(),last_context=context,**cfg['boundary']))
        raise
