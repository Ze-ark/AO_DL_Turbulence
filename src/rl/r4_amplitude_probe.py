"""冻结R1轨迹同状态分支诊断；仅首步改变力度，后续零修正积分器。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import traceback
import torch

from src.rl.r4_action_response import physical_step
from src.rl.r4_baseline_selection import read
from src.rl.r4_closed_loop import model_ensembles
from src.rl.r4_control import NominalCalibration, R4Limits, project_request
from src.rl.r4_dynamics import advance_history
from src.rl.r4_dynamics_experiment import Progress, verify_hashes, safe_git_record
from src.rl.r4_interface_smoke import write_json
from src.rl.r4_observation import R4Interface, simulation_residual_proxy
from src.rl.r4_selected_anchor import load_selected, anchor_parameters
from src.rl.r4_trajectory import anchor_delta
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _project_path, _relative, _file_sha256, _load_yaml, _profiles
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.robust_control import RobustnessCondition

SOURCES = 'configs/experiments/s4_r4_amplitude_probe_v1_sources.json'
METRICS = ('audit_power', 'audit_strehl', 'audit_violation', 'audit_phase_rmse')


def scaled_action(action: torch.Tensor, multiplier: float) -> torch.Tensor:
    if (action.ndim != 2 or action.shape[1] != 11 or
            not bool(torch.isfinite(action).all()) or bool((action.abs() > 1).any())
            or multiplier not in (0., .5, 1., 1.5)):
        raise ValueError('invalid frozen amplitude probe')
    return (action * multiplier).clamp(-1, 1)


def budget(quick: bool) -> dict[str, int]:
    batches, probes, prefix = (1, 2, 40) if quick else (36, 4, 180)
    return dict(physical_transitions=batches*16*(prefix+probes*4*8),
                model_forward_samples=batches*16*probes*4*8*3)


def require_close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if (actual.shape != expected.shape or not bool(torch.isfinite(actual).all())
            or not torch.allclose(actual, expected, atol=1e-6, rtol=0)):
        raise RuntimeError(f'amplitude replay alignment failed: {label}')


@torch.no_grad()
def predict(models: list, h: torch.Tensor, v: torch.Tensor, u: torch.Tensor,
            anchor: dict, cal: NominalCalibration, horizon: int) -> torch.Tensor:
    predictions = []
    for model in models:
        mh, mv = h.clone(), v.clone()
        powers = []
        for k in range(horizon):
            action = project_request(mh[:, -1, 21:42], anchor_delta(mh[:, -1], anchor),
                                     u if k == 0 else torch.zeros_like(u), R4Limits())
            out = model(mh, mv, action.requested_delta_rad)
            if not bool(torch.isfinite(out).all()):
                raise RuntimeError('nonfinite branch prediction')
            powers.append(out[:, 21])
            mh, mv = advance_history(mh, mv, action.requested_delta_rad,
                                     action.normalized_correction, out, cal)
        predictions.append(torch.stack(powers, 1))
    return torch.stack(predictions, 1)  # [天气,模型成员,未来步]


def summarize(values: torch.Tensor, *, seed: int, repeats: int) -> dict:
    # [类型,档位,天气,探测时刻,力度,指标]；不把档位/帧作为独立样本。
    if values.shape != (3, 6, 32, 4, 4, 4) or not bool(torch.isfinite(values).all()):
        raise ValueError('requires all predeclared weather and probes')
    collapsed = values.double().mean((1, 3))  # [类型,天气,力度,指标]
    generator = torch.Generator(device=values.device).manual_seed(seed)
    pairs = ((3, 2), (1, 2), (2, 0))
    boot = values.new_zeros((repeats, 3), dtype=torch.float64)
    for f in range(3):
        draws = torch.randint(32, (repeats, 32), device=values.device, generator=generator)
        for j, (a, b) in enumerate(pairs):
            boot[:, j] += (collapsed[f, :, a, 0]-collapsed[f, :, b, 0])[draws].mean(1)/3
    # 三项预声明对比：Bonferroni 98.333%逐项区间，整体名义覆盖至少95%。
    quantiles = values.new_tensor([.05/6, 1-.05/6], dtype=torch.float64)
    ci = torch.quantile(boot, quantiles, dim=0).T.tolist()
    comparisons = []
    for j, (a, b) in enumerate(pairs):
        comparisons.append(dict(amplitude_indices=[a, b], mean_metric_delta=(collapsed[:, :, a]-collapsed[:, :, b]).mean((0, 1)).tolist(),
                                power_ci_bonferroni=ci[j]))
    return dict(comparisons=comparisons, metric_order=METRICS, independent_weather=96,
                family_means=collapsed.mean(1).tolist(), profile_means=values.double().mean((0, 2, 3)).tolist(),
                bootstrap_seed=seed, bootstrap_repeats=repeats, ranking_or_rl_gate=False)


def preflight(path: str | Path, quick: bool) -> tuple:
    own = read(_project_path(SOURCES)); verify_hashes(own)
    if _relative(_project_path(path)) not in own:
        raise ValueError('configuration not frozen')
    cfg = _load_yaml(_project_path(path)); up = _project_path(cfg['upstream'])
    summary = read(up/'summary.json')
    if (_file_sha256(up/'summary.json') != cfg['upstream_sha256'] or
            read(up/'SUCCESS.json')['summary_sha256'] != cfg['upstream_sha256'] or
            _file_sha256(up/'artifact_manifest.json') != summary['artifact_manifest_sha256']):
        raise RuntimeError('formal closed-loop provenance changed')
    spec, frozen = load_selected()
    frozen.update(read(up/'preflight.json')['frozen_files'])
    frozen.update(read(up/'artifact_manifest.json')); frozen.update(own)
    for p in (up/'summary.json', up/'SUCCESS.json', up/'artifact_manifest.json', _project_path(SOURCES)):
        frozen[_relative(p)] = _file_sha256(p)
    if not quick:
        q = _project_path(cfg['output_directory']+'_quick')
        qs = read(q/'summary.json')
        if (qs['status'] != 'QUICK_COMPLETE_NO_CONCLUSION' or
                read(q/'SUCCESS.json')['summary_sha256'] != _file_sha256(q/'summary.json') or
                _file_sha256(q/'artifact_manifest.json') != qs['artifact_manifest_sha256'] or
                read(q/'preflight.json')['frozen_files'].get(SOURCES) != _file_sha256(_project_path(SOURCES))):
            raise RuntimeError('same-version quick prerequisite missing')
        frozen.update(read(q/'artifact_manifest.json'))
        for name in ('summary.json', 'SUCCESS.json', 'artifact_manifest.json'):
            frozen[_relative(q/name)] = _file_sha256(q/name)
    verify_hashes(frozen)
    if (cfg['probe_steps'] != [8, 40, 100, 180] or cfg['multipliers'] != [0., .5, 1., 1.5]
            or cfg['horizon'] != 8 or any(cfg[k] != v for k, v in budget(False).items())):
        raise ValueError('predeclared design or budget changed')
    output = _project_path(cfg['output_directory']+('_quick' if quick else ''))
    if output.exists():
        raise FileExistsError(f'preserve existing output: {output}')
    device = resolve_device('cuda')
    records = [r for r in read(up/'trajectory_manifest.json') if r['controller'] == 'gru_mpc']
    parent = _load_yaml(_project_path(cfg['parent']))
    starts = [3762048, 3762304, 3762560]
    expected = {(f, p, starts[f]+o) for f in range(3) for p in parent['profile_ids'] for o in (0, 16)}
    if len(records) != 36 or {(r['family'], r['profile'], r['seed']) for r in records} != expected:
        raise RuntimeError('incomplete or duplicate development records')
    if quick:
        records = [next(r for r in records if (r['family'], r['profile'], r['seed']) == (0, 'nominal', starts[0]))]
    report = dict(status='READY_FOR_DIAGNOSTIC' if quick else 'READY_FOR_USER_IDE',
                  frozen_files=frozen, quick=quick, device=str(device), **budget(quick))
    return cfg, parent, anchor_parameters(spec), records, output, report


@torch.no_grad()
def execute(cfg: dict, parent: dict, anchor: dict, records: list, output: Path, report: dict) -> dict:
    device = resolve_device('cuda'); cal = NominalCalibration(**parent['nominal_calibration'])
    models = model_ensembles(device)['gru_mpc']
    base, _ = load_s1_config(_project_path(parent['environment_config']))
    base = replace(base, num_modes=21, batch_size=16, episode_length=200)
    basis, _, _ = build_action_basis(base, ActionRepresentation('r4_zernike21', 'zernike', 21), device)
    profiles = {p.identifier: p for p in _profiles(parent, parent['profile_ids'])}
    probes = cfg['probe_steps'][:2] if report['quick'] else cfg['probe_steps']
    values = torch.zeros(3, 6, 32, 4, 4, 4, dtype=torch.float64, device=device)
    progress = Progress(output, device); progress.phase('同状态动作力度诊断', report['physical_transitions']//16)
    physical = calls = 0; manifest = []; active = {}
    (output/'branches').mkdir()
    def tick() -> None:
        nonlocal physical
        physical += 16
        progress.tick(dict(物理转移=physical, 模型前向=calls))
    try:
        for record in records:
            active = dict(record)
            saved = torch.load(_project_path(record['file']), map_location=device, weights_only=True)
            f, seed = record['family'], record['seed']; profile = profiles[record['profile']]
            condition = RobustnessCondition.from_mapping(dict(parent['families'][f], base_seed=seed))
            env = AdaptiveOpticsEnv(profile.environment_config(condition.environment_config(base)), device,
                                    profile.effects_config(), basis_override=basis)
            raw, _ = env.reset(seed=seed)
            sensor = torch.Generator(device=device).manual_seed(seed+parent['data']['sensor_seed_offset'])
            interface = R4Interface(calibration=cal)
            interface.reset(simulation_residual_proxy(raw, generator=sensor, noise_std_rad=profile.observation_noise_std_rad), episode_id=str(seed))
            require_close(interface.snapshot().features[:, -1], saved['frames'][:, 0], 'reset')
            results = []
            for step in range(max(probes)+1):
                active['replay_step'] = step
                require_close(interface.snapshot().features[:, -1], saved['frames'][:, step], f'frame {step}')
                original = saved['rows'][step]['correction']
                if step in probes:
                    probe_result = []; snap = interface.snapshot()
                    for multiplier in cfg['multipliers']:
                        u = scaled_action(original, multiplier)
                        branch_env, branch_interface, branch_sensor = deepcopy((env, interface, sensor))
                        rows = []
                        for k in range(cfg['horizon']):
                            row = physical_step(branch_env, branch_interface, branch_sensor, profile,
                                                u if k == 0 else torch.zeros_like(u), anchor)
                            if multiplier == 1. and k == 0:
                                require_close(row['audit_power'], saved['rows'][step]['reward_power_in_bucket'], 'original first power')
                                require_close(row['applied_modal'], saved['rows'][step]['applied_modal'], 'original first actuator')
                            rows.append(row); tick()
                        tensors = {name: torch.stack([r[name] for r in rows], 1) for name in rows[0]}
                        if any(not bool(torch.isfinite(t).all()) for t in tensors.values()):
                            raise RuntimeError('nonfinite physical branch')
                        prediction = predict(models, snap.features, snap.valid, u, anchor, cal, cfg['horizon'])
                        calls += 16*3*cfg['horizon']
                        probe_result.append(dict(multiplier=multiplier, correction=u.cpu(),
                            clipped_components=((original*multiplier).abs()>1).cpu(),
                            predicted_power=prediction.cpu(), **{k:v.cpu() for k,v in tensors.items()}))
                    pi = parent['profile_ids'].index(profile.identifier)
                    offset = seed-[3762048, 3762304, 3762560][f]; qi = probes.index(step)
                    for a, result in enumerate(probe_result):
                        values[f, pi, offset:offset+16, qi, a] = torch.stack([result[k].double().mean(1) for k in METRICS], -1).to(device)
                    results.append(dict(step=step, branches=probe_result))
                    # 分支必须没有改变主轨迹（含传感器随机流，后续逐步重放也检验）。
                    require_close(interface.snapshot().features, snap.features, 'branch isolation')
                if step < max(probes):
                    row = physical_step(env, interface, sensor, profile, original, anchor); tick()
                    require_close(row['audit_power'], saved['rows'][step]['reward_power_in_bucket'], f'power {step}')
                    require_close(row['requested_modal'], saved['rows'][step]['requested_modal'], f'request {step}')
                    require_close(row['applied_modal'], saved['rows'][step]['applied_modal'], f'actuator {step}')
            target = output/'branches'/Path(record['file']).name
            torch.save(dict(source=record, seeds=saved['seeds'], probes=results), target)
            manifest.append(dict(file=_relative(target), sha256=_file_sha256(target)))
        if dict(physical_transitions=physical, model_forward_samples=calls) != budget(report['quick']):
            raise RuntimeError('diagnostic budget mismatch')
        write_json(output/'branch_manifest.json', manifest)
        if not report['quick']:
            torch.save(dict(values=values.cpu(), metric_order=METRICS), output/'metrics.pt')
        analysis = {} if report['quick'] else summarize(values, seed=cfg['bootstrap_seed'], repeats=cfg['bootstrap_repeats'])
        return dict(status='QUICK_COMPLETE_NO_CONCLUSION' if report['quick'] else 'AMPLITUDE_DIAGNOSTIC_COMPLETE_REQUIRES_AUDIT',
                    **budget(report['quick']), completed_batches=len(records), analysis=analysis)
    except Exception:
        write_json(output/'interrupted_context.json', dict(active=active, physical_transitions=physical, model_forward_samples=calls))
        raise
    finally:
        progress.close()


def run(path: str | Path, *, quick: bool = False, preflight_only: bool = False) -> dict:
    cfg, parent, anchor, records, output, report = preflight(path, quick)
    if preflight_only:
        return {k:v for k,v in report.items() if k != 'frozen_files'}
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if os.environ['CUBLAS_WORKSPACE_CONFIG'] not in (':4096:8', ':16:8'):
        raise ValueError('unsupported CUDA determinism setting')
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'preflight.json', report); write_json(output/'config.json', cfg)
    write_json(output/'runtime.json', dict(torch=str(torch.__version__), gpu=torch.cuda.get_device_name(), git=safe_git_record()))
    try:
        result = execute(cfg, parent, anchor, records, output, report)
        verify_hashes(report['frozen_files'])
        write_json(output/'artifact_manifest.json', {_relative(p):_file_sha256(p) for p in output.rglob('*') if p.is_file()})
        result.update(training_updates=0, confirmation_access=False, real_slm_actions=False, automatic_retry=False,
            artifact_manifest_sha256=_file_sha256(output/'artifact_manifest.json'),
            material_passport=dict(origin_skill='academic-research-suite', origin_mode='run',
                origin_date=datetime.now(timezone.utc).isoformat(), verification_status='REQUIRES_AUDIT', version_label='r4_amplitude_probe_v1'),
            next_action='停止，等待只读审计；不自动调参或训练。')
        write_json(output/'summary.json', result)
        write_json(output/'SUCCESS.json', dict(summary_sha256=_file_sha256(output/'summary.json')))
        return result
    except Exception:
        write_json(output/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise
