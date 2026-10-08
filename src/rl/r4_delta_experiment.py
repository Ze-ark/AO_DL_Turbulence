"""已批准的一次R4差值监督对照；原结果不改，正式执行由用户拥有。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback

import torch

from src.rl.r4_action_response import direction_statistics, physical_branch, physical_step, pulse
from src.rl.r4_control import NominalCalibration
from src.rl.r4_delta_learning import PairStore, accuracy_gain, paired_losses, pulse_rollout
from src.rl.r4_dynamics import normalized_errors, rollout
from src.rl.r4_dynamics_experiment import Progress, evaluate, safe_git_record, verify_hashes
from src.rl.r4_interface_smoke import write_json
from src.rl.r4_observation import R4Interface, simulation_residual_proxy
from src.rl.r4_response_experiment import load_models
from src.rl.r4_trajectory import EpisodeStore
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _profiles, _project_path, _relative
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.robust_control import RobustnessCondition

SOURCES = 'configs/experiments/s4_r4_delta_supervision_v1_sources.json'


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def settings(cfg: dict, quick: bool) -> dict:
    d = _load_yaml(_project_path(cfg['design']))
    parent = _load_yaml(_project_path(d['parent_config']))
    q = cfg['quick']
    return dict(design=d, parent=parent, quick=quick,
        per_family=q['per_family'] if quick else d['data']['per_family'],
        batch=q['per_family'] if quick else parent['data']['batch_episodes'],
        profiles=q['profiles'] if quick else parent['profile_ids'],
        probes=q['probe_steps'] if quick else d['data']['probe_steps'],
        modes=q['modes'] if quick else list(range(d['data']['high_modes'])),
        updates=q['updates'] if quick else d['training']['updates_per_member_per_arm'],
        trajectory_batch=q['trajectory_batch'] if quick else d['training']['trajectory_batch'],
        pair_batch=q['pair_batch'] if quick else d['training']['pair_batch'],
        interval=q['evaluation_interval'] if quick else d['training']['evaluation_interval'],
        replicates=q['bootstrap_replicates'] if quick else d['evaluation']['bootstrap_replicates'])


def preflight(config_path: str | Path, quick: bool) -> tuple[dict, dict, dict]:
    path = _project_path(config_path); cfg = _load_yaml(path)
    if cfg['stage'] != 'S4-D2-R4-1B1-R1' or cfg['runtime'] != dict(device='cuda', formal_owner='user_ide', automatic_retry=False):
        raise ValueError('requires user-owned CUDA training')
    own = read(_project_path(SOURCES)); verify_hashes(own)
    if _relative(path) not in own or cfg['budget_approval']['date'] != '2026-09-09':
        raise ValueError('unfrozen or unapproved runtime config')
    c = settings(cfg, quick); d = c['design']; up = _project_path(cfg['upstream'])
    if _file_sha256(up/'summary.json') != cfg['upstream_summary_sha256']:
        raise RuntimeError('D1 summary changed')
    summary = read(up/'summary.json')
    if summary['status'] != 'ATTRIBUTION_COMPLETE_REQUIRES_AUDIT' or read(up/'SUCCESS.json')['summary_sha256'] != cfg['upstream_summary_sha256']:
        raise RuntimeError('D1 completion missing')
    frozen = dict(own); frozen.update(read(up/'preflight.json')['frozen_files'])
    if _file_sha256(up/'artifact_manifest.json') != summary['artifact_manifest_sha256']:
        raise RuntimeError('D1 manifest mismatch')
    frozen.update(read(up/'artifact_manifest.json'))
    for p in (path, _project_path(SOURCES), up/'summary.json', up/'SUCCESS.json', up/'artifact_manifest.json'):
        frozen[_relative(p)] = _file_sha256(p)
    dev = _project_path(cfg['quick_development'] if quick else d['data']['development_source'])
    if quick:
        if _file_sha256(dev/'summary.json') != cfg['quick_development_summary_sha256']:
            raise RuntimeError('quick development changed')
        devsummary = read(dev/'summary.json')
        if _file_sha256(dev/'artifact_manifest.json') != devsummary['artifact_manifest_sha256']:
            raise RuntimeError('quick development manifest changed')
        frozen.update(read(dev/'artifact_manifest.json'))
        for p in (dev/'summary.json', dev/'SUCCESS.json', dev/'artifact_manifest.json'):
            frozen[_relative(p)] = _file_sha256(p)
    verify_hashes(frozen)
    train_weather = {s+i for s in d['data']['training_family_starts'] for i in range(c['per_family'])}
    pd = c['parent']['data']
    original = {pd['namespace_seed']+off+i for off in pd['family_offsets'] for i in range(pd['train_per_family'])}
    development = {r['weather_seed'] for r in map(json.loads, (dev/'pairs.jsonl').read_text(encoding='utf-8').splitlines())}
    if not train_weather <= original or train_weather & development:
        raise RuntimeError('training/development split violation')
    output = _project_path(cfg['quick_directory' if quick else 'output_directory']).resolve()
    if output != _project_path('outputs/s4_r4_delta_supervision_v1'+('_quick' if quick else '')).resolve() or output.exists():
        raise FileExistsError(f'preserve delta-supervision output: {output}')
    n = len(train_weather)*len(c['profiles']); choices = 1+2*len(c['modes'])
    transitions = n*(max(c['probes'])+len(c['probes'])*choices*8)
    pairs = n*len(c['probes'])*2*len(c['modes'])
    training_calls = 6*c['updates']*8*(c['trajectory_batch']+2*c['pair_batch'])
    if not quick and (transitions != 410112 or pairs != 38016 or training_calls != 9216000):
        raise RuntimeError('approved budget changed')
    device = resolve_device('cuda')
    return cfg, c, dict(status='READY_FOR_QUICK_CHECK' if quick else 'READY_FOR_USER_IDE', device=str(device),
        quick=quick, frozen_files=frozen, development_directory=_relative(dev),
        training_weather=sorted(train_weather), new_physical_transitions=transitions,
        training_pairs=pairs, training_updates=6*c['updates'], training_model_forward_samples=training_calls,
        rl_updates=0, mpc_evaluation=False, confirmation_access=False, real_slm_actions=False, s4d3_access=False)


@torch.no_grad()
def collect_pairs(c: dict, output: Path, device: torch.device, progress: Progress) -> tuple[PairStore, int]:
    parent = c['parent']; d = c['design']; records, manifest = [], []
    base, _ = load_s1_config(_project_path(parent['environment_config']))
    base = replace(base, num_modes=21, batch_size=c['batch'])
    basis, _, _ = build_action_basis(base, ActionRepresentation('r4_zernike21', 'zernike', 21), device)
    cal = NominalCalibration(**parent['nominal_calibration'])
    transitions = 0
    (output/'training_pairs').mkdir(); (output/'audit').mkdir()
    def tick() -> None:
        nonlocal transitions
        transitions += c['batch']; progress.tick()
    for f, family in enumerate(parent['families']):
        for profile in _profiles(parent, c['profiles']):
            for offset in range(0, c['per_family'], c['batch']):
                seed = d['data']['training_family_starts'][f]+offset
                condition = RobustnessCondition.from_mapping(dict(family, base_seed=seed))
                env = AdaptiveOpticsEnv(profile.environment_config(condition.environment_config(base)), device,
                                       profile.effects_config(), basis_override=basis)
                raw, _ = env.reset(seed=seed)
                sensor = torch.Generator(device=device).manual_seed(seed+parent['data']['sensor_seed_offset'])
                interface = R4Interface(calibration=cal)
                interface.reset(simulation_residual_proxy(raw, generator=sensor,
                    noise_std_rad=profile.observation_noise_std_rad), episode_id=str(seed))
                for step in range(max(c['probes'])+1):
                    if step in c['probes']:
                        snap = interface.snapshot()
                        zero = physical_branch(env, interface, sensor, profile, parent['collector_anchor'], 0, 0., 8, tick)
                        powers, corrections, audits = [], [], {'zero': {k:v.cpu() for k,v in zero.items()}}
                        for mode in c['modes']:
                            for sign in (-1., 1.):
                                branch = physical_branch(env, interface, sensor, profile, parent['collector_anchor'], mode, sign, 8, tick)
                                powers.append(branch['power'].cpu())
                                corrections.append(pulse(1, mode, sign, 0, device)[0].cpu())
                                audits[f'{mode}_{int(sign)}'] = {k:v.cpu() for k,v in branch.items()}
                        r = dict(split='train', source='simulation_residual_proxy_not_holography',
                            history=snap.features.cpu(), valid=snap.valid.cpu(), zero_power=zero['power'].cpu(),
                            powers=torch.stack(powers), corrections=torch.stack(corrections),
                            weather=torch.arange(seed, seed+c['batch']), family=f, profile=profile.identifier, probe=step)
                        name = f'{family["id"]}_{profile.identifier}_{seed}_{step}.pt'
                        target = output/'training_pairs'/name; torch.save(r, target)
                        torch.save(audits, output/'audit'/name)
                        manifest.append(dict(file=_relative(target), sha256=_file_sha256(target),
                            audit_file=_relative(output/'audit'/name), audit_sha256=_file_sha256(output/'audit'/name)))
                        records.append(r)
                    if step < max(c['probes']):
                        physical_step(env, interface, sensor, profile, pulse(c['batch'], 0, 0., 0, device), parent['collector_anchor']); tick()
    write_json(output/'training_pair_manifest.json', dict(schema='r4_delta_pairs_v1', split='train', records=manifest))
    return PairStore(records, 'train'), transitions


def development_pairs(directory: Path, parent: dict, quick: bool) -> PairStore:
    records = []
    for path in sorted((directory/'branches').glob('development_*_zero.pt')):
        prefix = path.stem[:-5]
        f = next(i for i, family in enumerate(parent['families']) if prefix.startswith('development_'+family['id']+'_'))
        profile, _, step = prefix[len('development_'+parent['families'][f]['id']+'_'):].rsplit('_', 2)
        if quick and profile != 'nominal':
            continue
        zero = torch.load(path, map_location='cpu', weights_only=True)
        powers, corrections = [], []
        for mode in ([0, 10] if quick else range(11)):
            for sign in (-1, 1):
                b = torch.load(path.with_name(prefix+f'_{mode}_{sign}.pt'), map_location='cpu', weights_only=True)
                powers.append(b['actual']['power'])
                corrections.append(pulse(1, mode, float(sign), 0, torch.device('cpu'))[0])
        records.append(dict(split='development', source='simulation_residual_proxy_not_holography',
            history=zero['history'], valid=zero['valid'], zero_power=zero['actual']['power'],
            powers=torch.stack(powers), corrections=torch.stack(corrections),
            weather=torch.tensor(zero['seeds']), family=f, profile=profile, probe=int(step)))
    result = PairStore(records, 'development')
    if result.count != (24 if quick else 38016):
        raise RuntimeError('incomplete development pairs')
    return result


@torch.no_grad()
def evaluate_pairs(model, store: PairStore, parent: dict, device: torch.device, progress: Progress) -> tuple[torch.Tensor, int]:
    model.eval(); predictions = []; calls = 0
    cal = NominalCalibration(**parent['nominal_calibration'])
    for r in store.records:
        history, valid = r['history'].to(device), r['valid'].to(device)
        b = len(history)
        zero = pulse_rollout(model, history, valid, torch.zeros(b, 11, device=device), parent['collector_anchor'], cal)
        calls += b*8; progress.tick()
        for correction in r['corrections']:
            power = pulse_rollout(model, history, valid, correction.to(device).expand(b, -1), parent['collector_anchor'], cal)
            predictions.append((power.mean(1)-zero.mean(1)).cpu())
            calls += b*8; progress.tick()
    return torch.cat(predictions), calls


def execute(cfg: dict, c: dict, report: dict, output: Path) -> dict:
    device = resolve_device('cuda'); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False; torch.backends.cuda.matmul.allow_tf32 = False
    parent, d = c['parent'], c['design']; up = _project_path(d['models']['source'])
    progress, validation_progress = Progress(output, device), Progress(output, device)
    try:
        progress.phase('配对标签采集（训练天气）', report['new_physical_transitions']//c['batch'])
        pairs, physical_count = collect_pairs(c, output, device, progress)
        if physical_count != report['new_physical_transitions'] or pairs.count != report['training_pairs']:
            raise RuntimeError('collection budget mismatch')
        devpairs = development_pairs(_project_path(report['development_directory']), parent, c['quick'])
        if set(pairs.data['weather'].tolist()) & set(devpairs.data['weather'].tolist()):
            raise RuntimeError('pair weather leakage')
        scale = pairs.delta_scale(device, d['loss']['delta_scale_floor'])
        write_json(output/'delta_scale.json', dict(source='all_training_pairs_only', rms_floor=d['loss']['delta_scale_floor'], value=float(scale)))
        data_manifest = read(up/'data_manifest.json')
        stores = {}
        for split in ('train', 'development'):
            paths = [_project_path(r['file']) for r in data_manifest['records'] if r['split'] == split]
            stores[split] = EpisodeStore.from_files(paths[:1] if c['quick'] else paths)
        if set(stores['train'].data['weather_seeds'].tolist()) & set(stores['development'].data['weather_seeds'].tolist()):
            raise RuntimeError('trajectory weather leakage')
        write_json(output/'development_pair_index.json', devpairs.metadata)
        (output/'checkpoints').mkdir(); (output/'development').mkdir()
        parents = load_models(up, device)[1::2]
        parent_summary = read(up/'summary.json'); final = {}; final_metrics = {}; sampling_states = {}
        training_calls = direction_calls = trajectory_eval_calls = update_count = 0
        cal = NominalCalibration(**parent['nominal_calibration'])
        evalcfg = deepcopy(parent)
        if c['quick']:
            evalcfg['training']['evaluation_starts'] = [0, 4]
        for arm, weight in zip(d['models']['arms'], d['loss']['delta_weight_by_arm']):
            final[arm] = []; final_metrics[arm] = []
            for member, seed in enumerate(parent['model']['member_seeds']):
                model = deepcopy(parents[member])
                model.gru.requires_grad_(True); model.head.requires_grad_(True)
                model.linear.requires_grad_(False)
                optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=d['training']['learning_rate'])
                trajectory_pool = stores['train'].bootstrap_pool(seed)
                pair_pool = pairs.pool(seed)
                tg = torch.Generator().manual_seed(seed+100); pg = torch.Generator().manual_seed(seed+200)
                torch.save(dict(state_dict=model.state_dict(), seed=seed), output/'checkpoints'/f'{arm}_{member}_initial.pt')
                progress.phase(f'{arm} 模型{member+1}/3 总更新{6*c["updates"]}批', c['updates'])
                running = 0.
                for update in range(1, c['updates']+1):
                    model.train(); optimizer.zero_grad(set_to_none=True)
                    batch = stores['train'].sample(trajectory_pool, c['trajectory_batch'], 8, tg, device)
                    pair = pairs.sample(pair_pool, c['pair_batch'], pg, device)
                    prediction = rollout(model, batch['history'], batch['valid'], batch['commands'], batch['corrections'], cal)
                    trajectory_loss = normalized_errors(prediction, batch, model.linear.y_scale, parent['model']['horizons']).mean()
                    pp = pulse_rollout(model, pair['history'], pair['valid'], pair['correction'], parent['collector_anchor'], cal)
                    zp = pulse_rollout(model, pair['history'], pair['valid'], torch.zeros_like(pair['correction']), parent['collector_anchor'], cal)
                    absolute_loss, delta_loss = paired_losses(pp, zp, pair['power'], pair['zero_power'], model.linear.y_scale[21], scale)
                    loss = trajectory_loss+absolute_loss+weight*delta_loss
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError('non-finite supervised loss')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], d['training']['gradient_clip'], error_if_nonfinite=True)
                    optimizer.step(); update_count += 1
                    training_calls += 8*(c['trajectory_batch']+2*c['pair_batch'])
                    value = float(loss.detach()); running += value
                    row = dict(arm=arm, member=member, update=update, loss=value, trajectory=float(trajectory_loss.detach()),
                               absolute=float(absolute_loss.detach()), delta=float(delta_loss.detach()), gradient_norm=float(norm))
                    with (output/'loss_history.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(row)+'\n')
                    progress.tick({'平均损失': running/update, '当前损失': value, '总更新已完成': update_count})
                    if update % c['interval'] == 0:
                        checkpoint = dict(state_dict=model.state_dict(), seed=seed, update=update, arm=arm,
                            trajectory_generator=tg.get_state(), pair_generator=pg.get_state(), delta_scale=float(scale))
                        torch.save(checkpoint, output/'checkpoints'/f'{arm}_{member}_{update:05d}.pt')
                        validation_progress.phase(f'{arm} 模型{member+1} 第{update}批动作评价', sum(1+len(r['corrections']) for r in devpairs.records))
                        delta, calls = evaluate_pairs(model, devpairs, parent, device, validation_progress)
                        direction_calls += calls
                        # 仅读取开发数据的评价：不参与梯度、归一化、采样或早停。
                        metrics = evaluate(model, stores['development'], evalcfg, device, model.linear.y_scale)
                        trajectory_eval_calls += metrics['windows']*8
                        torch.save(dict(predicted_delta=delta, metrics=metrics, arm=arm, member=member, update=update),
                                   output/'development'/f'{arm}_{member}_{update:05d}.pt')
                        with (output/'development_history.jsonl').open('a', encoding='utf-8') as stream:
                            stream.write(json.dumps(dict(arm=arm, member=member, update=update, **metrics))+'\n')
                        if update == c['updates']:
                            final[arm].append(delta); final_metrics[arm].append(metrics)
                # 证明两组批序列没有因为评价或运行顺序改变。
                states = (tg.get_state(), pg.get_state())
                if member in sampling_states and not all(torch.equal(a, b) for a, b in zip(states, sampling_states[member])):
                    raise RuntimeError('arm sampling sequences diverged')
                sampling_states[member] = states
                if any(not torch.equal(v, parents[member].linear.state_dict()[k]) for k, v in model.linear.state_dict().items()):
                    raise RuntimeError('frozen linear normalization changed')
        if training_calls != report['training_model_forward_samples'] or update_count != report['training_updates']:
            raise RuntimeError('training budget mismatch')
        if not c['quick'] and direction_calls != 7630848:
            raise RuntimeError('development direction budget mismatch')
        actual = devpairs.data['power'].to(device).mean(1)-devpairs.data['zero_power'].to(device).mean(1)
        thresholds = read(_project_path(report['development_directory'])/'noise_calibration.json')['thresholds']
        threshold = torch.tensor([thresholds[r['profile']] for r in devpairs.metadata], device=device)
        weather = devpairs.data['weather'].to(device)
        family = torch.tensor([r['family'] for r in devpairs.metadata], device=device)
        stats = {}; preds = {}
        for arm in d['models']['arms']:
            preds[arm] = torch.stack(final[arm]).to(device).mean(0)
            stats[arm] = direction_statistics(actual, preds[arm], threshold, weather, family,
                seed=cfg['bootstrap_seed'], replicates=c['replicates'])
        gain = accuracy_gain(actual, preds['absolute_plus_delta'], preds['absolute_only'], threshold,
                             weather, family, cfg['bootstrap_seed'], c['replicates'])
        prediction_pass = all(all(g <= .95*l and g <= .95*p for g, l, p in zip(
            final_metrics['absolute_plus_delta'][m]['per_horizon'][-1],
            parent_summary['members'][m]['linear']['per_horizon'][-1], parent_summary['persistence']['per_horizon'][-1])) for m in range(3))
        passed = (stats['absolute_plus_delta']['status'] == 'ACTION_DIRECTION_PASS' and prediction_pass
                  and gain.get('ci95') is not None and gain['ci95'][0] > 0 and gain.get('invalid_replicates') == 0)
        torch.save(dict(actual_delta=actual.cpu(), thresholds=threshold.cpu(),
                        predictions={k:v.cpu() for k,v in preds.items()}), output/'final_pairs.pt')
        verify_hashes(report['frozen_files'])
        for r in read(output/'training_pair_manifest.json')['records']:
            verify_hashes({r['file']:r['sha256'], r['audit_file']:r['audit_sha256']})
        progress.close(); validation_progress.close()
        artifacts = {_relative(p):_file_sha256(p) for p in output.rglob('*') if p.is_file()}
        write_json(output/'artifact_manifest.json', artifacts)
        analysis = dict(direction=stats, paired_gain=gain, prediction_gate_all_three=prediction_pass, final_prediction_metrics=final_metrics)
        return dict(material_passport=dict(origin_skill='academic-research-suite / experiment-agent', origin_mode='run',
            origin_date=datetime.now(timezone.utc).isoformat(), verification_status='UNVERIFIED', version_label='r4_delta_v1'),
            status='QUICK_SMOKE_ONLY' if c['quick'] else ('DELTA_SUPERVISION_DEVELOPMENT_PASS' if passed else 'DELTA_SUPERVISION_DEVELOPMENT_FAIL'),
            analysis={} if c['quick'] else analysis, quick_diagnostic_only=analysis if c['quick'] else {},
            new_physical_transitions=physical_count, training_pairs=pairs.count, training_updates=update_count,
            model_forward_samples=dict(training=training_calls, development_direction=direction_calls, development_prediction=trajectory_eval_calls),
            identical_sampling_verified=True, frozen_linear_verified=True, primary_checkpoint='final_only',
            artifact_manifest_sha256=_file_sha256(output/'artifact_manifest.json'), original_b1_status='ACTION_DIRECTION_FAIL',
            rl_updates=0, mpc_evaluation=False, confirmation_access=False, real_slm_actions=False, s4d3_access=False,
            next_action='停止等待只读验收；不自动重训、增加预算或放行MPC/RL。')
    finally:
        progress.close(); validation_progress.close()


def run(config_path: str | Path, *, quick: bool = False, preflight_only: bool = False) -> dict:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if os.environ['CUBLAS_WORKSPACE_CONFIG'] not in (':4096:8', ':16:8'):
        raise ValueError('deterministic CUDA requires a supported CUBLAS_WORKSPACE_CONFIG')
    cfg, c, report = preflight(config_path, quick)
    if preflight_only:
        return {k:v for k,v in report.items() if k not in ('frozen_files', 'training_weather')}
    output = _project_path(cfg['quick_directory' if quick else 'output_directory'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'preflight.json', report); write_json(output/'effective_config.json', dict(runtime=cfg, settings=c))
    write_json(output/'runtime.json', dict(torch=str(torch.__version__), cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), git=safe_git_record()))
    try:
        result = execute(cfg, c, report, output); write_json(output/'summary.json', result)
        write_json(output/'SUCCESS.json', dict(status=result['status'], summary_sha256=_file_sha256(output/'summary.json')))
        return result
    except Exception as exc:
        write_json(output/'failure.json', dict(error=str(exc), traceback=traceback.format_exc(), automatic_retry=False))
        raise
