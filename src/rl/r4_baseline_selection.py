"""同天气完整回合筛选18个传统候选；不训练、不运行MPC。"""
from __future__ import annotations
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import torch
from src.rl.r4_baselines import BaselineSpec, baseline_delta
from src.rl.r4_control import NominalCalibration
from src.rl.r4_dynamics import ARXDynamics
from src.rl.r4_dynamics_experiment import Progress, verify_hashes, safe_git_record
from src.rl.r4_interface_smoke import write_json
from src.rl.r4_observation import R4Interface, PowerMeasurement, simulation_residual_proxy
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _profiles, _project_path, _relative
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.robust_control import RobustnessCondition

SOURCES = 'configs/experiments/s4_r4_baseline_selection_v1_sources.json'
METRICS = ('reward_power_in_bucket', 'reward_strehl', 'violation_fraction',
           'reward_phase_rmse', 'saturated_fraction', 'slew_limited_fraction')


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def candidates(grid: dict) -> list[BaselineSpec]:
    result = [BaselineSpec(k, g, l, grid['tracking_gain'], grid['prediction_steps'])
              for k in grid['kinds'] for g in grid['gains'] for l in grid['leaks']]
    for spec in result: spec.validate()
    if len(result) != 18 or len(set(result)) != 18:
        raise ValueError('18 unique frozen candidates required')
    return result


def select_candidate(means: torch.Tensor, invalid: torch.Tensor, *, reference: int = 9,
                     violation_increase: float = .001) -> dict:
    """输入[候选,天气类型,档位,天气,指标]；等权完整天气，不挑帧。"""
    if means.ndim != 5 or means.shape[0] != 18 or means.shape[1] != 3 or means.shape[-1] != len(METRICS) or any(n == 0 for n in means.shape):
        raise ValueError('incomplete candidate metric tensor')
    if invalid.shape != (18,) or invalid.dtype != torch.bool or not 0 <= reference < 18:
        raise ValueError('invalid candidate validity mask')
    if not bool(torch.isfinite(means).all()) or violation_increase != .001:
        raise ValueError('nonfinite results or changed selection threshold')
    # 先平均档位与完整天气，再平均天气类型；所有类型同权。
    average = means.double().mean(2).mean(2).mean(1)
    if bool(invalid[reference]):
        return dict(status='REFERENCE_INVALID', selected_index=None)
    increase = average[:, 2]-average[reference, 2]
    eligible = ~invalid & (increase <= violation_increase)
    power = torch.where(eligible, average[:, 0], torch.full_like(average[:, 0], -torch.inf))
    winner = int(torch.argmax(power)) if bool(eligible.any()) else None
    return dict(status='BASELINE_SELECTED_REQUIRES_AUDIT' if winner is not None else 'NO_ELIGIBLE_BASELINE',
                selected_index=winner, reference_index=reference, eligible=eligible.tolist(),
                violation_increase=increase.tolist(), candidate_means=average.tolist(),
                interpretation='development_selection_not_independent_confirmation', tie_break='first_config_index')


def preflight(config_path: str | Path, quick: bool) -> tuple[dict, dict, dict]:
    path = _project_path(config_path); cfg = _load_yaml(path)
    own = read(_project_path(SOURCES)); verify_hashes(own)
    if _relative(path) not in own or cfg['runtime'] != dict(device='cuda', formal_owner='user_ide', automatic_retry=False):
        raise ValueError('frozen user-owned CUDA configuration required')
    if cfg['stage'] != 'S4-D2-R4-1B2-BASELINE-SELECTION': raise ValueError('wrong stage')
    up = _project_path(cfg['upstream']); summary = read(up/'summary.json')
    if _file_sha256(up/'summary.json') != cfg['upstream_sha256'] or read(up/'SUCCESS.json')['summary_sha256'] != cfg['upstream_sha256']:
        raise RuntimeError('upstream completion hash changed')
    if summary['status'] != 'DIAGNOSTIC_CLOSED_LOOP_COMPLETE' or _file_sha256(up/'artifact_manifest.json') != summary['artifact_manifest_sha256']:
        raise RuntimeError('upstream incomplete')
    frozen = read(up/'preflight.json')['frozen_files']; frozen.update(read(up/'artifact_manifest.json')); frozen.update(own)
    for p in (up/'summary.json', up/'SUCCESS.json', up/'artifact_manifest.json', _project_path(SOURCES)):
        frozen[_relative(p)] = _file_sha256(p)
    if cfg['parent'] not in frozen or cfg['candidate_config'] not in frozen: raise ValueError('unfrozen parent')
    verify_hashes(frozen)
    parent = _load_yaml(_project_path(cfg['parent']))
    specs = candidates(_load_yaml(_project_path(cfg['candidate_config']))['candidate_grid'])
    if specs[cfg['reference_index']] != BaselineSpec('tracking', .25, .1): raise ValueError('wrong reference')
    pd = parent['data']; starts = cfg['development_family_starts']
    if starts != [pd['namespace_seed']+pd['development_offset']+x for x in pd['family_offsets']]:
        raise ValueError('development seeds differ from frozen source')
    n = cfg['quick']['per_family'] if quick else cfg['per_family']
    b = cfg['quick']['batch_size'] if quick else cfg['batch_size']
    t = cfg['quick']['steps'] if quick else cfg['steps']
    profiles = cfg['quick']['profiles'] if quick else parent['profile_ids']
    selected = {s+i for s in starts for i in range(n)}
    train = {pd['namespace_seed']+off+i for off in pd['family_offsets'] for i in range(pd['train_per_family'])}
    if selected & train or max(selected) >= 4000000 or n % b:
        raise ValueError('split or batch contract violation')
    physical = 18*3*n*len(profiles)*t; calls = 6*3*n*len(profiles)*t*3*2
    if not quick and (physical != cfg['physical_transitions'] or calls != cfg['ridge_forward_samples']):
        raise ValueError('formal budget changed')
    output = _project_path(cfg['quick_directory' if quick else 'output_directory'])
    if output.resolve() != _project_path('outputs/s4_r4_baseline_selection_v1'+('_quick' if quick else '')).resolve() or output.exists():
        raise FileExistsError(f'preserve selection output: {output}')
    if shutil.disk_usage(_project_path('.')).free < (128 if quick else 4096)*1024**2:
        raise RuntimeError('insufficient free disk for preserved trajectories')
    device = resolve_device('cuda')
    settings = dict(parent=parent, specs=specs, starts=starts, per_family=n, batch=b, steps=t, profiles=profiles)
    return cfg, settings, dict(status='READY_FOR_QUICK_SMOKE' if quick else 'READY_FOR_USER_IDE',
        quick=quick, device=str(device), frozen_files=frozen, physical_transitions=physical,
        ridge_forward_samples=calls, candidates=18, weather=len(selected), training_updates=0,
        mpc_calls=0, confirmation_access=False, s4d3_access=False, real_slm_actions=False)


@torch.no_grad()
def execute(cfg: dict, settings: dict, report: dict, output: Path) -> dict:
    c = settings; parent = c['parent']; device = resolve_device('cuda')
    base, _ = load_s1_config(_project_path(parent['environment_config']))
    base = replace(base, num_modes=21, batch_size=c['batch'], episode_length=c['steps'])
    basis, _, _ = build_action_basis(base, ActionRepresentation('r4_zernike21', 'zernike', 21), device)
    cal = NominalCalibration(**parent['nominal_calibration']); linear = []
    for i in range(3):
        state = torch.load(_project_path(f'outputs/s4_r4_dynamics_v1/checkpoints/linear_{i}.pt'), map_location=device, weights_only=True)['state_dict']
        model = ARXDynamics(state['x_mean'], state['x_scale'], state['y_mean'], state['y_scale']).to(device)
        model.load_state_dict(state); linear.append(model.eval().requires_grad_(False))
    means = torch.zeros(18, 3, len(c['profiles']), c['per_family'], len(METRICS), device=device, dtype=torch.float64)
    invalid = torch.zeros(18, dtype=torch.bool, device=device)
    profiles = _profiles(parent, c['profiles'])
    progress = Progress(output, device); manifest = []; physical = model_calls = 0; active = {}
    write_json(output/'candidates.json', [asdict(s) for s in c['specs']])
    (output/'trajectories').mkdir()
    try:
        progress.phase('18组传统候选筛选' if not report['quick'] else '18候选快速冒烟（不选优）', report['physical_transitions']//c['batch'])
        for ci, spec in enumerate(c['specs']):
            for fi, family in enumerate(parent['families']):
                for pi, profile in enumerate(profiles):
                    for offset in range(0, c['per_family'], c['batch']):
                        seed = c['starts'][fi]+offset
                        active = dict(candidate=ci, family=fi, profile=profile.identifier, seed=seed, step=0)
                        condition = RobustnessCondition.from_mapping(dict(family, base_seed=seed))
                        env = AdaptiveOpticsEnv(profile.environment_config(condition.environment_config(base)), device,
                            profile.effects_config(), basis_override=basis)
                        raw, _ = env.reset(seed=seed)
                        sensor = torch.Generator(device=device).manual_seed(seed+parent['data']['sensor_seed_offset'])
                        proxy = lambda x: simulation_residual_proxy(x, generator=sensor, noise_std_rad=profile.observation_noise_std_rad)
                        interface = R4Interface(calibration=cal); interface.reset(proxy(raw), episode_id=str(seed))
                        frames = [interface.snapshot().features[:, -1].cpu()]
                        records = {k:[] for k in (*METRICS, 'measured_power', 'requested_delta', 'requested_modal', 'applied_modal', 'terminated')}
                        latencies = []
                        for step in range(c['steps']):
                            active['step'] = step; snap = interface.snapshot()
                            torch.cuda.synchronize(device); start = time.perf_counter()
                            delta, calls = baseline_delta(spec, snap.features, snap.valid, linear, cal)
                            action = interface.issue(delta, delta.new_zeros(c['batch'], 11), step=step)
                            torch.cuda.synchronize(device); latencies.append(time.perf_counter()-start)
                            model_calls += calls
                            if bool((action.requested_delta_rad.abs() > .150001).any()) or bool((action.requested_delta_rad.norm(dim=-1) > .15*(10**.5)+1e-6).any()):
                                invalid[ci] = True; raise RuntimeError('projected request out of bounds')
                            raw, _, terminated, truncated, info = env.step(action.requested_delta_rad)
                            physical += c['batch']
                            interface.observe_next(proxy(raw), step=step+1,
                                power=PowerMeasurement(info['measured_power_in_bucket'], step, step+1))
                            if bool(truncated.any()) or not bool((terminated == (step == c['steps']-1)).all()):
                                raise RuntimeError('incomplete or prematurely terminated episode')
                            if not torch.allclose(action.requested_modal_rad, info['requested_modal'], atol=1e-6, rtol=0):
                                raise RuntimeError('request alignment mismatch')
                            values = {k:info[k] for k in METRICS}
                            values.update(measured_power=info['measured_power_in_bucket'], requested_delta=action.requested_delta_rad,
                                          requested_modal=action.requested_modal_rad, applied_modal=info['applied_modal'], terminated=terminated)
                            if any(not bool(torch.isfinite(v).all()) for v in values.values()):
                                invalid[ci] = True; raise RuntimeError('nonfinite trajectory; no selection')
                            for k, v in values.items(): records[k].append(v.cpu())
                            frames.append(interface.snapshot().features[:, -1].cpu())
                            progress.tick({'候选': ci+1, '候选总数': 18, '回合步': step+1, '物理转移': physical})
                        tensors = {k:torch.stack(v, dim=1) for k,v in records.items()}
                        episode_means = torch.stack([tensors[k].to(device).double().mean(1) for k in METRICS], -1)
                        means[ci, fi, pi, offset:offset+c['batch']] = episode_means
                        file = output/'trajectories'/f'c{ci:02d}_{family["id"]}_{profile.identifier}_{seed}.pt'
                        torch.save(dict(schema='r4_baseline_selection_episode_v1', source='simulation_residual_proxy_not_holography',
                            split='development', candidate=ci, spec=asdict(spec), family=fi, profile=profile.identifier,
                            seeds=list(range(seed, seed+c['batch'])), frames=torch.stack(frames, 1), latency_seconds=latencies, **tensors), file)
                        manifest.append(dict(file=_relative(file), sha256=_file_sha256(file), **active))
                        with (output/'episode_metrics.jsonl').open('a', encoding='utf-8') as stream:
                            for j, values in enumerate(episode_means.cpu().tolist()):
                                stream.write(json.dumps(dict(candidate=ci, family=fi, profile=profile.identifier,
                                    weather_seed=seed+j, metrics=dict(zip(METRICS, values))))+'\n')
        if physical != report['physical_transitions'] or model_calls != report['ridge_forward_samples']:
            raise RuntimeError('completed run differs from frozen budget')
        verify_hashes(report['frozen_files']); progress.close()
        write_json(output/'trajectory_manifest.json', manifest)
        torch.save(dict(means=means.cpu(), invalid=invalid.cpu(), metric_order=METRICS), output/'candidate_metrics.pt')
        selection = {} if report['quick'] else select_candidate(means, invalid, reference=cfg['reference_index'], violation_increase=cfg['violation_increase_max'])
        if selection.get('selected_index') is not None:
            selected = selection['selected_index']
            write_json(output/'selected_baseline.json', dict(index=selected, spec=asdict(c['specs'][selected]),
                selection=selection, linear_checkpoint_hashes={f'linear_{i}':_file_sha256(_project_path(f'outputs/s4_r4_dynamics_v1/checkpoints/linear_{i}.pt')) for i in range(3)},
                status='REQUIRES_READ_ONLY_AUDIT', deployable_hardware_verified=False))
        artifacts = {_relative(p):_file_sha256(p) for p in output.rglob('*') if p.is_file()}
        write_json(output/'artifact_manifest.json', artifacts)
        return dict(status='QUICK_SMOKE_ONLY' if report['quick'] else selection['status'], selection=selection,
            physical_transitions=physical, ridge_forward_samples=model_calls, completed_trajectory_batches=len(manifest),
            material_passport=dict(origin_skill='academic-research-suite / experiment-agent', origin_mode='run',
                origin_date=datetime.now(timezone.utc).isoformat(), verification_status='UNVERIFIED', version_label='r4_baseline_selection_v1'),
            training_updates=0, mpc_calls=0, confirmation_access=False, real_slm_actions=False, s4d3_access=False,
            artifact_manifest_sha256=_file_sha256(output/'artifact_manifest.json'), next_action='停止等待用户通知后的只读审计，不自动进入MPC或RL。')
    except Exception:
        write_json(output/'interrupted_context.json', dict(**active, physical_transitions=physical, ridge_forward_samples=model_calls))
        raise
    finally:
        progress.close()


def run(config_path: str | Path, *, quick: bool = False, preflight_only: bool = False) -> dict:
    cfg, settings, report = preflight(config_path, quick)
    if preflight_only: return {k:v for k,v in report.items() if k != 'frozen_files'}
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if os.environ['CUBLAS_WORKSPACE_CONFIG'] not in (':4096:8', ':16:8'): raise ValueError('unsupported deterministic CUDA setting')
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.enabled = False; torch.backends.cuda.matmul.allow_tf32 = False
    output = _project_path(cfg['quick_directory' if quick else 'output_directory']); output.mkdir(parents=True, exist_ok=False)
    write_json(output/'preflight.json', report); write_json(output/'config.json', cfg)
    write_json(output/'runtime.json', dict(torch=str(torch.__version__), cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), git=safe_git_record()))
    try:
        result = execute(cfg, settings, report, output); write_json(output/'summary.json', result)
        write_json(output/'SUCCESS.json', dict(summary_sha256=_file_sha256(output/'summary.json')))
        return result
    except Exception:
        write_json(output/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise
