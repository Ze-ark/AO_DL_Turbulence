"""共同积分器的三组短闭环诊断；没有正式比较或训练入口。"""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import os
import time
import traceback
import torch
from src.rl.r4_baseline_safety import read
from src.rl.r4_baselines import baseline_delta, guarded_correction, range_eligible
from src.rl.r4_baseline_selection import METRICS
from src.rl.r4_control import NominalCalibration
from src.rl.r4_dynamics_experiment import Progress, verify_hashes, safe_git_record
from src.rl.r4_interface_smoke import write_json
from src.rl.r4_mpc import SearchConfig
from src.rl.r4_observation import R4Interface, PowerMeasurement, simulation_residual_proxy
from src.rl.r4_response_experiment import load_models
from src.rl.r4_selected_anchor import load_selected, selected_plan, selected_request
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _project_path, _relative, _load_yaml, _file_sha256
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.robust_control import RobustnessCondition
from src.rl.s4_training import _profiles


def validate_config(c: dict) -> None:
    expected = dict(stage='S4-D2-R4-1B2-SELECTED-SMOKE', device='cuda', diagnostic_only=True,
        output_directory='outputs/s4_r4_selected_smoke_v1',
        parent='configs/experiments/s4_r4_dynamics_v1.yaml',
        calibration='outputs/s4_r4_baseline_safety_v1/calibration.json',
        profiles=['nominal', 'delay_3', 'settling_050'], controllers=['integrator', 'linear_mpc', 'gru_mpc'],
        base_seed=3760000, search_seed=3769500, batch_size=2, steps=12,
        physical_transitions=216, max_mpc_forward_samples=1769472, training_updates=0,
        formal_comparison_authorized=False)
    if c != expected: raise ValueError('diagnostic scope or budget changed')


@torch.no_grad()
def correction_for(controller, models, h, v, spec, cal, bounds, seed):
    """范围外不调用模型；有效状态下退回共同积分器，模型内部错误直接停止。"""
    if controller not in ('integrator', 'linear_mpc', 'gru_mpc'):
        raise ValueError('unknown controller')
    u = h.new_zeros(len(h), 11)
    score = h.new_full((len(h),), float('nan'))
    score_valid = torch.zeros(len(h), device=h.device, dtype=torch.bool)
    calls = 0; eligible = range_eligible(h, v, bounds)
    if controller == 'integrator':
        return u, ['baseline']*len(h), score, score_valid, calls
    if bool(eligible.any()):
        result = selected_plan(models, h[eligible], v[eligible], spec, cal, SearchConfig(), seed=seed)
        u[eligible] = result['correction']; score[eligible] = result['score']
        score_valid[eligible] = True; calls = result['model_forward_samples']
    u, reasons = guarded_correction(h, v, u, bounds)
    return u, reasons, score, score_valid, calls


def run(config_path: str | Path, *, preflight_only: bool = False) -> dict:
    cfg = _load_yaml(_project_path(config_path)); validate_config(cfg)
    spec, frozen = load_selected()
    for name in (str(config_path), 'src/rl/r4_selected_anchor.py', 'src/rl/r4_selected_smoke.py',
                 'scripts/check_s4_r4_selected_smoke.py', 'tests/test_r4_selected_anchor.py',
                 'tests/test_r4_selected_smoke.py'):
        path = _project_path(name); frozen[_relative(path)] = _file_sha256(path)
    verify_hashes(frozen)
    if cfg['calibration'] not in frozen: raise ValueError('unfrozen safety calibration')
    output = _project_path(cfg['output_directory'])
    if output.exists(): raise FileExistsError(f'preserve diagnostic output: {output}')
    device = resolve_device('cuda')
    report = dict(status='READY_FOR_SHORT_DIAGNOSTIC', frozen_files=frozen,
                  physical_transitions=216, max_mpc_forward_samples=1769472, training_updates=0)
    if preflight_only: return {k:v for k,v in report.items() if k != 'frozen_files'}
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if os.environ['CUBLAS_WORKSPACE_CONFIG'] not in (':4096:8', ':16:8'): raise ValueError('unsupported CUDA setting')
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'preflight.json', report); write_json(output/'config.json', cfg)
    write_json(output/'runtime.json', dict(gpu=torch.cuda.get_device_name(device), torch=str(torch.__version__), git=safe_git_record()))
    progress = Progress(output, device)
    physical = calls = 0; active = {}
    try:
        parent = _load_yaml(_project_path(cfg['parent'])); bounds = read(_project_path(cfg['calibration']))
        models = load_models(_project_path('outputs/s4_r4_dynamics_v1'), device)
        ensembles = {'linear_mpc': models[::2], 'gru_mpc': models[1::2], 'integrator': []}
        for i, model in enumerate(ensembles['gru_mpc']):
            ck = _project_path(f'outputs/s4_r4_delta_supervision_v1/checkpoints/absolute_plus_delta_{i}_02000.pt')
            model.load_state_dict(torch.load(ck, map_location=device, weights_only=True)['state_dict'])
            model.eval().requires_grad_(False)
        base, _ = load_s1_config(_project_path(parent['environment_config']))
        base = replace(base, num_modes=21, batch_size=2, episode_length=200)
        basis, _, _ = build_action_basis(base, ActionRepresentation('r4_zernike21', 'zernike', 21), device)
        cal = NominalCalibration(**parent['nominal_calibration'])
        timing = []; reasons_count = {}; (output/'trajectories').mkdir()
        progress.phase('共同积分器三组短闭环（不排名）', 108)
        for profile in _profiles(parent, cfg['profiles']):
            for controller in cfg['controllers']:
                active = dict(profile=profile.identifier, controller=controller, step=0)
                seed = cfg['base_seed']
                condition = RobustnessCondition.from_mapping(dict(parent['families'][0], base_seed=seed))
                env = AdaptiveOpticsEnv(profile.environment_config(condition.environment_config(base)), device,
                                       profile.effects_config(), basis_override=basis)
                raw, _ = env.reset(seed=seed)
                sensor = torch.Generator(device=device).manual_seed(seed+parent['data']['sensor_seed_offset'])
                proxy = lambda x: simulation_residual_proxy(x, generator=sensor, noise_std_rad=profile.observation_noise_std_rad)
                interface = R4Interface(calibration=cal); interface.reset(proxy(raw), episode_id=str(seed))
                rows = []
                for step in range(cfg['steps']):
                    active['step'] = step; snap = interface.snapshot(); h, v = snap.features, snap.valid
                    torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device); start = time.perf_counter()
                    u, reasons, score, score_valid, n = correction_for(controller, ensembles[controller], h, v, spec, cal,
                                                                     bounds, cfg['search_seed']+step)
                    calls += n
                    expected = selected_request(spec, h, v, u)
                    delta, _ = baseline_delta(spec, h, v, [], cal)
                    action = interface.issue(delta, u, step=step)
                    if not torch.equal(expected.requested_delta_rad, action.requested_delta_rad):
                        raise RuntimeError('interface and planner request mismatch')
                    torch.cuda.synchronize(device)
                    timing.append(dict(controller=controller, profile=profile.identifier, step=step, batch=2,
                        planned_batch=int(score_valid.sum()), seconds=time.perf_counter()-start,
                        peak_allocated_gb=torch.cuda.max_memory_allocated(device)/1024**3))
                    raw, _, terminated, truncated, info = env.step(action.requested_delta_rad); physical += 2
                    interface.observe_next(proxy(raw), step=step+1, power=PowerMeasurement(info['measured_power_in_bucket'], step, step+1))
                    if bool(terminated.any()) or bool(truncated.any()): raise RuntimeError('unexpected short-prefix termination')
                    if not torch.allclose(action.requested_modal_rad, info['requested_modal'], atol=1e-6, rtol=0):
                        raise RuntimeError('environment request mismatch')
                    audit = {k:info[k].cpu() for k in (*METRICS, 'applied_modal')}
                    if any(not bool(torch.isfinite(x).all()) for x in audit.values()): raise RuntimeError('nonfinite physical metrics')
                    for reason in reasons: reasons_count[reason] = reasons_count.get(reason, 0)+1
                    rows.append(dict(history=h.cpu(), valid=v.cpu(), next_frame=interface.snapshot().features[:, -1].cpu(),
                        correction=u.cpu(), reasons=reasons, predicted_score=score.cpu(), predicted_valid=score_valid.cpu(),
                        requested_delta=action.requested_delta_rad.cpu(), requested_modal=action.requested_modal_rad.cpu(),
                        measured_power=info['measured_power_in_bucket'].cpu(), **audit))
                    progress.tick({'物理转移': physical, '模型前向': calls})
                torch.save(dict(source='simulation_diagnostic_only', split='training_weather_diagnostic',
                    profile=profile.identifier, controller=controller, seeds=[seed, seed+1], rows=rows),
                    output/'trajectories'/f'{profile.identifier}_{controller}.pt')
        if physical != 216 or calls > cfg['max_mpc_forward_samples']: raise RuntimeError('budget mismatch')
        verify_hashes(frozen); progress.close(); write_json(output/'timing.json', timing)
        artifacts = {_relative(p):_file_sha256(p) for p in output.rglob('*') if p.is_file()}
        write_json(output/'artifact_manifest.json', artifacts)
        result = dict(status='SELECTED_ANCHOR_SHORT_CLOSED_LOOP_COMPLETE', physical_transitions=physical,
            mpc_forward_samples=calls, training_updates=0, fallback_counts=reasons_count, ranking={},
            formal_comparison_authorized=False, real_slm_actions=False,
            material_passport=dict(origin_skill='academic-research-suite', origin_mode='run', verification_status='DIAGNOSTIC_ONLY',
                origin_date=datetime.now(timezone.utc).isoformat(), version_label='r4_selected_smoke_v1'),
            artifact_manifest_sha256=_file_sha256(output/'artifact_manifest.json'))
        write_json(output/'summary.json', result)
        write_json(output/'SUCCESS.json', dict(summary_sha256=_file_sha256(output/'summary.json')))
        return result
    except Exception:
        write_json(output/'failure.json', dict(traceback=traceback.format_exc(), context=active,
                   physical_transitions=physical, mpc_forward_samples=calls, automatic_retry=False))
        raise
    finally: progress.close()
