"""传统候选、训练范围校准和192次转移的小型物理闭环检查。"""
from __future__ import annotations
from dataclasses import replace
import json
import os
from pathlib import Path
import time
import traceback
import torch
from src.rl.r4_baselines import BaselineSpec, baseline_delta, guarded_correction, range_eligible
from src.rl.r4_control import NominalCalibration
from src.rl.r4_dynamics_experiment import Progress, verify_hashes, safe_git_record
from src.rl.r4_interface_smoke import write_json
from src.rl.r4_mpc import SearchConfig, plan
from src.rl.r4_observation import R4Interface, PowerMeasurement, simulation_residual_proxy
from src.rl.r4_response_experiment import load_models
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _profiles, _project_path, _relative
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.robust_control import RobustnessCondition


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def preflight(config_path: str | Path) -> tuple[dict, dict]:
    cfg = _load_yaml(_project_path(config_path))
    if (cfg['stage'] != 'S4-D2-R4-1B2-BASELINE-SAFETY' or cfg['device'] != 'cuda'
            or not cfg['diagnostic_only'] or cfg['select_baseline'] or cfg['formal_comparison_authorized']
            or cfg['training_updates'] != 0): raise ValueError('diagnostic-only CUDA scope required')
    expected = dict(family_index=0, base_seed=3760000, batch_size=2, profiles=['nominal', 'delay_3'],
        steps=12, controllers=['integrator', 'tracking', 'ridge', 'gru_mpc'], gain=.25, leak=.1, search_seed=3769400)
    if cfg['smoke'] != expected or cfg['calibration'] != dict(source='original_training_frames_only', max_multiplier=1.25, residual_floor=.02, power_floor=.001):
        raise ValueError('smoke or calibration budget changed')
    if (cfg['physical_transitions'], cfg['max_mpc_forward_samples'], cfg['ridge_forward_samples']) != (192, 589824, 288):
        raise ValueError('invalid budget')
    if cfg['candidate_grid'] != dict(kinds=['integrator', 'tracking', 'ridge'], gains=[.15, .25, .35],
            leaks=[.05, .1], tracking_gain=.5, prediction_steps=2, count=18):
        raise ValueError('candidate grid changed')
    up = _project_path(cfg['upstream']); s = read(up/'summary.json')
    digest = '5a22c0c6a395fa0a7022f5c99f17381d298681b7fd94de23a395bf67a02d1818'
    if cfg['upstream_sha256'] != digest or _file_sha256(up/'summary.json') != digest or read(up/'SUCCESS.json')['summary_sha256'] != digest:
        raise RuntimeError('timing source changed')
    if s['status'] != 'DIAGNOSTIC_TIMING_COMPLETE' or _file_sha256(up/'artifact_manifest.json') != s['artifact_manifest_sha256']:
        raise RuntimeError('timing incomplete')
    frozen = read(up/'preflight.json')['frozen_files']; frozen.update(read(up/'artifact_manifest.json'))
    for p in (up/'summary.json', up/'SUCCESS.json', up/'artifact_manifest.json', _project_path(config_path),
              _project_path('src/rl/r4_baselines.py'), _project_path('src/rl/r4_baseline_safety.py'),
              _project_path('scripts/check_s4_r4_baseline_safety.py'), _project_path('tests/test_r4_baselines.py')):
        frozen[_relative(p)] = _file_sha256(p)
    if cfg['parent'] not in frozen: raise ValueError('parent not frozen')
    verify_hashes(frozen)
    output = _project_path(cfg['output_directory'])
    if output.resolve() != _project_path('outputs/s4_r4_baseline_safety_v1').resolve() or output.exists():
        raise FileExistsError(f'preserve baseline-safety output: {output}')
    device = resolve_device(cfg['device'])
    return cfg, dict(status='READY_FOR_DIAGNOSTIC_SMOKE', device=str(device), frozen_files=frozen,
        physical_transitions=192, max_mpc_forward_samples=589824, ridge_forward_samples=288, training_updates=0)


@torch.no_grad()
def calibrate(parent: dict, cfg: dict, output: Path, device: torch.device, progress: Progress) -> dict:
    old = _project_path('outputs/s4_r4_dynamics_v1')
    records = [r for r in read(old/'data_manifest.json')['records'] if r['split'] == 'train']
    progress.phase('读取原训练范围（不训练）', len(records))
    residual = torch.zeros(21, device=device); power = torch.zeros((), device=device)
    weather = set(); count = 0
    for r in records:
        data = torch.load(_project_path(r['file']), map_location='cpu', weights_only=True)
        # 拆分保存在冻结data_manifest中，不在原轨迹张量文件内。
        if data['schema'] != 'r4_complete_episode_v1' or data['source'] != 'simulation_residual_proxy_not_holography':
            raise ValueError('calibration source violation')
        frames = data['frames'].to(device)
        if not bool(torch.isfinite(frames).all()): raise RuntimeError('nonfinite training frames')
        residual = torch.maximum(residual, frames[:, :, :21].abs().amax((0, 1)))
        arrived = frames[:, :, 78].bool()
        if bool(arrived.any()): power = torch.maximum(power, frames[:, :, 74][arrived].max())
        weather.update(data['weather_seeds'].tolist()); count += frames.shape[0]*frames.shape[1]
        progress.tick()
    c = cfg['calibration']
    bounds = dict(source=c['source'], source_files=[r['file'] for r in records], training_weather=sorted(weather),
        frames=count, residual_abs_max=(residual.clamp_min(c['residual_floor'])*c['max_multiplier']).cpu().tolist(),
        power_max=float(power.clamp_min(c['power_floor'])*c['max_multiplier']), hardware_guarantee=False)
    if len(weather) != 384: raise RuntimeError('incomplete training calibration')
    write_json(output/'calibration.json', bounds)
    return bounds


@torch.no_grad()
def execute(cfg: dict, report: dict, output: Path, device: torch.device) -> dict:
    parent = _load_yaml(_project_path(cfg['parent'])); sm = cfg['smoke']
    progress = Progress(output, device)
    try:
        bounds = calibrate(parent, cfg, output, device, progress)
        models = load_models(_project_path('outputs/s4_r4_dynamics_v1'), device)
        linear, gru = models[::2], models[1::2]
        for i, model in enumerate(gru):
            ck = _project_path(f'outputs/s4_r4_delta_supervision_v1/checkpoints/absolute_plus_delta_{i}_02000.pt')
            model.load_state_dict(torch.load(ck, map_location=device, weights_only=True)['state_dict'])
        base, _ = load_s1_config(_project_path(parent['environment_config']))
        base = replace(base, num_modes=21, batch_size=sm['batch_size'], episode_length=200)
        basis, _, _ = build_action_basis(base, ActionRepresentation('r4_zernike21', 'zernike', 21), device)
        cal = NominalCalibration(**parent['nominal_calibration'])
        physical = mpc_calls = ridge_calls = 0; fallback_counts = {}; timing = []
        progress.phase('小型物理闭环（不排名）', 96)
        (output/'trajectories').mkdir()
        for profile in _profiles(parent, sm['profiles']):
            for controller in sm['controllers']:
                seed = sm['base_seed']
                condition = RobustnessCondition.from_mapping(dict(parent['families'][0], base_seed=seed))
                env = AdaptiveOpticsEnv(profile.environment_config(condition.environment_config(base)), device,
                    profile.effects_config(), basis_override=basis)
                raw, _ = env.reset(seed=seed)
                sensor = torch.Generator(device=device).manual_seed(seed+parent['data']['sensor_seed_offset'])
                interface = R4Interface(calibration=cal)
                proxy = lambda obs: simulation_residual_proxy(obs, generator=sensor, noise_std_rad=profile.observation_noise_std_rad)
                interface.reset(proxy(raw), episode_id=f'{profile.identifier}/{controller}/{seed}')
                rows = []
                for step in range(sm['steps']):
                    snapshot = interface.snapshot(); h, v = snapshot.features, snapshot.valid
                    spec = BaselineSpec('tracking' if controller == 'gru_mpc' else controller, sm['gain'], sm['leak'])
                    base_delta, calls = baseline_delta(spec, h, v, linear, cal); ridge_calls += calls
                    u = h.new_zeros(len(h), 11); reasons = ['baseline']*len(h)
                    predicted = h.new_full((len(h),), float('nan')); predicted_valid = torch.zeros(len(h), device=device, dtype=torch.bool)
                    if controller == 'gru_mpc':
                        eligible = range_eligible(h, v, bounds)
                        if bool(eligible.any()):
                            torch.cuda.synchronize(device); begin = time.perf_counter()
                            result = plan(gru, h[eligible], v[eligible], parent['collector_anchor'], cal,
                                          SearchConfig(), seed=sm['search_seed']+step)
                            torch.cuda.synchronize(device)
                            timing.append(dict(batch=int(eligible.sum()), seconds=time.perf_counter()-begin))
                            mpc_calls += result['model_forward_samples']; u[eligible] = result['correction']
                            predicted[eligible] = result['score']; predicted_valid[eligible] = True
                        u, reasons = guarded_correction(h, v, u, bounds)
                    for reason in reasons: fallback_counts[reason] = fallback_counts.get(reason, 0)+1
                    action = interface.issue(base_delta, u, step=step)
                    raw, _, terminated, truncated, info = env.step(action.requested_delta_rad)
                    interface.observe_next(proxy(raw), step=step+1,
                        power=PowerMeasurement(info['measured_power_in_bucket'], step, step+1))
                    if bool(terminated.any()) or bool(truncated.any()): raise RuntimeError('unexpected short episode termination')
                    if not torch.allclose(action.requested_modal_rad, info['requested_modal'], atol=1e-6, rtol=0):
                        raise RuntimeError('request alignment mismatch')
                    if bool((action.requested_delta_rad.abs() > .150001).any()): raise RuntimeError('projected request violation')
                    audit = {k:info[k].cpu() for k in ('reward_power_in_bucket', 'reward_strehl', 'violation_fraction', 'applied_modal', 'reward_phase_rmse')}
                    if any(not bool(torch.isfinite(x).all()) for x in audit.values()): raise RuntimeError('nonfinite physical audit')
                    rows.append(dict(history=h.cpu(), valid=v.cpu(), correction=u.cpu(), reasons=reasons,
                        requested_delta=action.requested_delta_rad.cpu(), requested_modal=action.requested_modal_rad.cpu(),
                        predicted_score=predicted.cpu(), predicted_valid=predicted_valid.cpu(), **audit))
                    physical += len(h); progress.tick({'物理转移': physical})
                torch.save(dict(profile=profile.identifier, controller=controller, seeds=[seed, seed+1],
                    source='simulation_diagnostic_only', rows=rows), output/'trajectories'/f'{profile.identifier}_{controller}.pt')
        if physical != 192 or ridge_calls != 288 or mpc_calls > 589824: raise RuntimeError('smoke budget mismatch')
        verify_hashes(report['frozen_files']); progress.close()
        write_json(output/'timing.json', timing)
        artifacts = {_relative(p):_file_sha256(p) for p in output.rglob('*') if p.is_file()}
        write_json(output/'artifact_manifest.json', artifacts)
        return dict(status='DIAGNOSTIC_CLOSED_LOOP_COMPLETE', physical_transitions=physical,
            mpc_forward_samples=mpc_calls, ridge_forward_samples=ridge_calls, training_updates=0,
            fallback_counts=fallback_counts, baseline_selected=False, formal_comparison_authorized=False,
            real_slm_actions=False, ranking={}, artifact_manifest_sha256=_file_sha256(output/'artifact_manifest.json'))
    finally:
        progress.close()


def run(config_path: str | Path, *, preflight_only: bool = False) -> dict:
    cfg, report = preflight(config_path)
    if preflight_only: return {k:v for k,v in report.items() if k != 'frozen_files'}
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if os.environ['CUBLAS_WORKSPACE_CONFIG'] not in (':4096:8', ':16:8'): raise ValueError('unsupported CUBLAS setting')
    device = resolve_device('cuda'); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False; torch.backends.cuda.matmul.allow_tf32 = False
    output = _project_path(cfg['output_directory']); output.mkdir(parents=True, exist_ok=False)
    write_json(output/'preflight.json', report); write_json(output/'config.json', cfg)
    write_json(output/'runtime.json', dict(gpu=torch.cuda.get_device_name(device), torch=str(torch.__version__), git=safe_git_record()))
    try:
        result = execute(cfg, report, output, device); write_json(output/'summary.json', result)
        write_json(output/'SUCCESS.json', dict(summary_sha256=_file_sha256(output/'summary.json')))
        return result
    except Exception:
        write_json(output/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise
