"""仅重放已保存的R4分支，定位历史与命令续推敏感性；不构造环境或训练。"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback

import torch
from torch import nn

from src.rl.r4_action_response import model_branch, pulse
from src.rl.r4_control import NominalCalibration, R4Limits, project_request
from src.rl.r4_dynamics import advance_history
from src.rl.r4_dynamics_experiment import Progress, safe_git_record, verify_hashes
from src.rl.r4_interface_smoke import write_json
from src.rl.r4_response_experiment import load_models
from src.rl.r4_trajectory import anchor_delta
from src.rl.s4_training import _file_sha256, _load_yaml, _project_path, _relative
from src.runtime import resolve_device

SOURCES = 'configs/experiments/s4_r4_response_attribution_v1_sources.json'
ARMS = ('full_self', 'current_self', 'full_recorded_commands_privileged')
BOUNDARY = dict(new_physical_transitions=0, model_updates=0, rl_updates=0,
                mpc_evaluation=False, confirmation_access=False,
                real_slm_actions=False, s4d3_access=False)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def initial_history_view(history: torch.Tensor, valid: torch.Tensor,
                         step: int) -> tuple[torch.Tensor, torch.Tensor]:
    """只遮蔽初始旧帧，预测产生的新帧正常积累；不修改控制器的名义队列。"""
    x, mask = history.clone(), valid.clone()
    cutoff = max(0, history.shape[1] - 1 - step)
    x[:, :cutoff] = 0
    mask[:, :cutoff] = False
    return x, mask


@torch.no_grad()
def ablated_branch(model: nn.Module, history: torch.Tensor, valid: torch.Tensor,
                   anchor: dict, calibration: NominalCalibration, mode: int,
                   sign: float, horizon: int, arm: str,
                   recorded_commands: torch.Tensor | None = None) -> dict:
    if arm not in ARMS:
        raise ValueError('unknown diagnostic arm')
    privileged = arm == ARMS[2]
    if privileged != (recorded_commands is not None):
        raise ValueError('recorded future commands require the explicit privileged arm')
    if privileged and (recorded_commands.shape != (len(history), horizon, 21)
                       or not bool(torch.isfinite(recorded_commands).all())):
        raise ValueError('invalid recorded command sequence')
    if arm == ARMS[0]:
        return model_branch(model, history, valid, anchor, calibration, mode, sign, horizon)
    history, valid = history.clone(), valid.clone()
    powers, commands = [], []
    for k in range(horizon):
        correction = pulse(len(history), mode, sign, k, history.device)
        if privileged:
            command = recorded_commands[:, k]
        else:
            command = project_request(history[:, -1, 21:42],
                anchor_delta(history[:, -1], anchor), correction, R4Limits()).requested_delta_rad
        x, mask = initial_history_view(history, valid, k) if arm == ARMS[1] else (history, valid)
        prediction = model(x, mask, command)
        if not bool(torch.isfinite(prediction).all()):
            raise RuntimeError('non-finite attribution prediction')
        powers.append(prediction[:, 21]); commands.append(command)
        # 全历史专供名义执行器队列使用，避免把历史消融变成延迟模型改动。
        history, valid = advance_history(history, valid, command, correction, prediction, calibration)
    return dict(power=torch.stack(powers, 1), requested_delta=torch.stack(commands, 1))


def response_metrics(actual: torch.Tensor, prediction: torch.Tensor,
                     threshold: torch.Tensor) -> dict:
    if actual.shape != prediction.shape or actual.shape != threshold.shape:
        raise ValueError('unaligned response arrays')
    if any(not bool(torch.isfinite(x).all()) for x in (actual, prediction, threshold)):
        raise ValueError('non-finite response arrays')
    mask = actual.abs() > threshold
    pos, neg = (actual > 0) & mask, (actual < 0) & mask
    ba = (.5*((prediction[pos] > 0).double().mean() +
              (prediction[neg] < 0).double().mean())) if pos.any() and neg.any() else None
    return dict(total=len(actual), identifiable=int(mask.sum()), positive=int(pos.sum()),
                negative=int(neg.sum()), balanced_accuracy=float(ba) if ba is not None else None,
                response_rmse_all=float((prediction.double()-actual.double()).square().mean().sqrt())
                    if len(actual) else None,
                actual_response_rms_all=float(actual.double().square().mean().sqrt()) if len(actual) else None)


def paired_direction_intervals(actual: torch.Tensor, predictions: torch.Tensor,
                               threshold: torch.Tensor, weather: torch.Tensor,
                               family: torch.Tensor, seed: int, replicates: int) -> dict:
    """同一批天气抽样用于三臂；两个配对差异用97.5%区间，不触发放行。"""
    if predictions.shape != (len(actual), 3) or not (actual.shape == threshold.shape == weather.shape == family.shape):
        raise ValueError('unaligned paired arrays')
    if any(not bool(torch.isfinite(x).all()) for x in (actual, predictions, threshold)):
        raise ValueError('non-finite paired arrays')
    if replicates < 1:
        raise ValueError('positive bootstrap count required')
    unique, inverse = weather.unique(sorted=True, return_inverse=True)
    pos, neg = (actual > threshold), (actual < -threshold)
    counts = torch.zeros(len(unique), 8, device=actual.device, dtype=torch.float64)
    raw = torch.cat((pos[:, None], neg[:, None], (predictions > 0) & pos[:, None],
                     (predictions < 0) & neg[:, None]), 1).double()
    counts.index_add_(0, inverse, raw)
    total = counts.sum(0)
    base = dict(weather=len(unique), identifiable_weather=int((counts[:, :2].sum(1) > 0).sum()),
                replicates=replicates, interval_level_per_comparison=.975,
                comparisons=2, gate_effect='NONE_DEVELOPMENT_DIAGNOSTIC_ONLY')
    if not len(unique) or not bool((total[:2] > 0).all()):
        return dict(base, status='INSUFFICIENT_SIGNAL', contrasts={})
    generator = torch.Generator(device=actual.device).manual_seed(seed)
    sampled = torch.zeros(replicates, 8, device=actual.device, dtype=torch.float64)
    for f in family.unique(sorted=True):
        members = weather[family == f].unique(sorted=True)
        if any(bool((family[weather == w] != f).any()) for w in members):
            raise ValueError('weather assigned to multiple families')
        indices = torch.searchsorted(unique, members)
        draws = torch.randint(len(indices), (replicates, len(indices)),
                              device=actual.device, generator=generator)
        sampled += counts[indices[draws]].sum(1)
    def ba(c: torch.Tensor) -> torch.Tensor:
        return .5*(c[..., 2:5]/c[..., 0:1] + c[..., 5:8]/c[..., 1:2])
    valid = (sampled[:, :2] > 0).all(1)
    point = ba(total)
    boots = ba(sampled[valid])
    contrasts = {}
    for name, left, right in [('history_full_minus_current', 0, 1),
                              ('recorded_commands_minus_self', 2, 0)]:
        ci = torch.quantile(boots[:, left]-boots[:, right],
            torch.tensor([.0125, .9875], device=actual.device, dtype=torch.float64)).tolist() if valid.any() else None
        contrasts[name] = dict(balanced_accuracy_difference=float(point[left]-point[right]),
                               ci97_5=ci)
    return dict(base, status='DESCRIPTIVE_ONLY', invalid_bootstrap_replicates=int((~valid).sum()), contrasts=contrasts)


def preflight(config_path: str | Path, quick: bool) -> tuple[dict, dict]:
    path = _project_path(config_path); cfg = _load_yaml(path)
    if cfg['stage'] != 'S4-D2-R4-1B1-D1' or cfg['runtime'] != dict(
            device='cuda', formal_owner='user_ide', automatic_retry=False):
        raise ValueError('requires user-owned CUDA attribution diagnostic')
    if cfg['boundary'] != BOUNDARY or tuple(cfg['arms']) != ARMS or cfg['horizon'] != 8:
        raise ValueError('attribution boundary changed')
    own = read_json(_project_path(SOURCES)); verify_hashes(own)
    if _relative(path) not in own:
        raise ValueError('config must be frozen')
    up = _project_path(cfg['input_directory'])
    summary = read_json(up/'summary.json')
    if _file_sha256(up/'summary.json') != cfg['input_summary_sha256'] or summary['status'] != 'ACTION_DIRECTION_FAIL':
        raise RuntimeError('expected frozen B1 negative result')
    if read_json(up/'SUCCESS.json')['summary_sha256'] != cfg['input_summary_sha256']:
        raise RuntimeError('B1 completion marker mismatch')
    if _file_sha256(up/'artifact_manifest.json') != summary['artifact_manifest_sha256']:
        raise RuntimeError('B1 artifact manifest mismatch')
    frozen = dict(own)
    frozen.update(read_json(up/'preflight.json')['frozen_files'])
    frozen.update(read_json(up/'artifact_manifest.json'))
    for p in (path, _project_path(SOURCES), up/'summary.json', up/'SUCCESS.json', up/'artifact_manifest.json'):
        frozen[_relative(p)] = _file_sha256(p)
    verify_hashes(frozen)
    if _relative(_project_path(cfg['parent_config'])) not in frozen:
        raise ValueError('parent config not in frozen inputs')
    model_dir = _project_path(cfg['model_directory'])
    if any(_relative(model_dir/'checkpoints'/f'gru_{i}_best.pt') not in frozen for i in range(3)):
        raise ValueError('unfrozen model checkpoint')
    output = _project_path(cfg['quick_directory' if quick else 'output_directory']).resolve()
    expected = _project_path('outputs/s4_r4_response_attribution_v1'+('_quick' if quick else '')).resolve()
    if output != expected or output.exists():
        raise FileExistsError(f'preserve attribution output: {output}')
    zeros = sorted(p for p in frozen if p.startswith(_relative(up/'branches')+'/development_') and p.endswith('_zero.pt'))
    if quick:
        q = cfg['quick']; prefix = f'development_{q["family"]}_{q["profile"]}_{q["base_seed"]}_'
        zeros = [p for p in zeros if Path(p).name.startswith(prefix)]
    if len(zeros) != (3 if quick else 108):
        raise RuntimeError('incomplete frozen probe inventory')
    modes = cfg['quick']['modes'] if quick else list(range(11))
    device = resolve_device('cuda')
    return cfg, dict(status='READY_FOR_SMALL_CUDA_CHECK' if quick else 'READY_FOR_USER_IDE',
        quick=quick, device=str(device), zero_files=zeros, modes=modes, frozen_files=frozen,
        action_pairs=len(zeros)*16*len(modes)*2,
        model_forward_samples=len(zeros)*(1+2*len(modes))*16*8*3*3, **BOUNDARY)


@torch.no_grad()
def execute(cfg: dict, report: dict, output: Path) -> dict:
    device = resolve_device('cuda')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    parent = _load_yaml(_project_path(cfg['parent_config']))
    calibration = NominalCalibration(**parent['nominal_calibration'])
    models = load_models(_project_path(cfg['model_directory']), device)[1::2]
    up = _project_path(cfg['input_directory'])
    source_rows = [json.loads(line) for line in (up/'pairs.jsonl').read_text(encoding='utf-8').splitlines()]
    key = lambda r: (r['weather_seed'], r['profile'], r['probe'], r['mode'], int(r['sign']))
    lookup = {key(r): r for r in source_rows}
    if len(lookup) != len(source_rows):
        raise RuntimeError('duplicate upstream pairs')
    progress = Progress(output, device)
    progress.phase('固定分支三臂重放（不训练）', len(report['zero_files'])*(1+2*len(report['modes'])))
    (output/'predictions').mkdir()
    rows, max_errors = [], dict(baseline_power=0., baseline_commands=0., original_delta=0.)
    model_samples = 0
    def load(p: Path) -> dict:
        return torch.load(p, map_location=device, weights_only=True)
    try:
        for filename in report['zero_files']:
            path = _project_path(filename); zero = load(path); prefix = path.stem[:-5]
            family = next(i for i, f in enumerate(parent['families']) if prefix.startswith('development_'+f['id']+'_'))
            rest = prefix[len('development_'+parent['families'][family]['id']+'_'):]
            profile, _, probe_text = rest.rsplit('_', 2); probe = int(probe_text)
            seeds = zero['seeds']; history, valid = zero['history'], zero['valid']
            if len(seeds) != 16:
                raise RuntimeError('upstream batch size changed')
            saved = dict(seeds=seeds, profile=profile, probe=probe, family=family,
                         valid_initial=valid.sum(1).cpu(), source_zero=filename, branches={})
            zero_predictions = None
            choices = [(0, 0)] + [(m, s) for m in report['modes'] for s in (-1, 1)]
            for mode, sign in choices:
                original = zero if sign == 0 else load(path.with_name(prefix+f'_{mode}_{sign}.pt'))
                predictions = []
                command_rmse = []
                for arm in ARMS:
                    member_results = [ablated_branch(model, history, valid, parent['collector_anchor'],
                        calibration, mode, float(sign), 8, arm,
                        original['actual']['requested_delta'] if arm == ARMS[2] else None) for model in models]
                    powers = torch.stack([r['power'] for r in member_results])
                    commands = torch.stack([r['requested_delta'] for r in member_results])
                    predictions.append(powers)
                    command_rmse.append((commands-original['actual']['requested_delta'][None]).double().square().mean((0, 2, 3)).sqrt())
                    if arm == ARMS[0]:
                        old_p = torch.stack([original['models'][i]['power'] for i in (1, 3, 5)])
                        old_c = torch.stack([original['models'][i]['requested_delta'] for i in (1, 3, 5)])
                        max_errors['baseline_power'] = max(max_errors['baseline_power'], float((powers-old_p).abs().max()))
                        max_errors['baseline_commands'] = max(max_errors['baseline_commands'], float((commands-old_c).abs().max()))
                        if not torch.equal(powers, old_p) or not torch.equal(commands, old_c):
                            raise RuntimeError('baseline replay differs from frozen B1; preserve output, do not retry')
                    model_samples += len(seeds)*8*3
                powers = torch.stack(predictions)  # arm, member, batch, horizon
                if bool((valid.sum(1) == 1).all()) and not torch.equal(powers[0], powers[1]):
                    raise RuntimeError('single-frame negative control differs')
                saved['branches'][f'{mode}_{sign}'] = dict(power=powers.cpu(), command_rmse=torch.stack(command_rmse).cpu())
                if sign == 0:
                    zero_predictions = powers
                else:
                    actual_delta = original['actual']['power'].mean(1)-zero['actual']['power'].mean(1)
                    member_delta = powers.mean(-1)-zero_predictions.mean(-1)
                    ensemble_delta = member_delta.mean(1)
                    for i, seed in enumerate(seeds):
                        r = dict(lookup[(seed, profile, probe, mode, sign)])
                        err = abs(float(actual_delta[i])-r['actual_delta'])
                        max_errors['original_delta'] = max(max_errors['original_delta'], err)
                        if err != 0:
                            raise RuntimeError('raw branch no longer matches upstream label')
                        r.pop('predicted_deltas')
                        r.update(initial_valid=int(valid[i].sum()), predictions=ensemble_delta[:, i].tolist(),
                                 member_predictions=member_delta[:, :, i].tolist())
                        rows.append(r)
                        with (output/'pairs.jsonl').open('a', encoding='utf-8') as stream:
                            stream.write(json.dumps(r, ensure_ascii=False)+'\n')
                progress.tick()
            torch.save(saved, output/'predictions'/(prefix+'.pt'))
        if len(rows) != report['action_pairs'] or model_samples != report['model_forward_samples']:
            raise RuntimeError('attribution budget mismatch')
        t = {k: torch.tensor([r[k] for r in rows], device=device) for k in
             ('actual_delta', 'threshold', 'weather_seed', 'family', 'initial_valid', 'probe')}
        predictions = torch.tensor([r['predictions'] for r in rows], device=device)
        late = t['initial_valid'] == 8
        stat = cfg['statistics']
        paired = paired_direction_intervals(t['actual_delta'][late], predictions[late], t['threshold'][late],
            t['weather_seed'][late], t['family'][late], stat['bootstrap_seed'],
            cfg['quick']['bootstrap_replicates'] if report['quick'] else stat['bootstrap_replicates'])
        def group_metrics(mask: torch.Tensor) -> dict:
            return {arm: response_metrics(t['actual_delta'][mask], predictions[mask, i], t['threshold'][mask])
                    for i, arm in enumerate(ARMS)}
        groups = dict(all=group_metrics(torch.ones(len(rows), device=device, dtype=torch.bool)),
                      full_initial_history=group_metrics(late))
        for field in ('probe', 'family', 'profile'):
            groups[field] = {str(value): group_metrics(torch.tensor([r[field] == value for r in rows], device=device))
                             for value in sorted({r[field] for r in rows})}
        verify_hashes(report['frozen_files'])
        progress.close()
        artifacts = {_relative(p): _file_sha256(p) for p in output.rglob('*') if p.is_file()}
        write_json(output/'artifact_manifest.json', artifacts)
        analysis = dict(paired_full_history_states=paired, descriptive_groups=groups)
        return dict(material_passport=dict(origin_skill='academic-research-suite / experiment-agent', origin_mode='run',
            origin_date=datetime.now(timezone.utc).isoformat(), verification_status='UNVERIFIED', version_label='r4_b1_d1_v1'),
            status='QUICK_SMOKE_ONLY' if report['quick'] else 'ATTRIBUTION_COMPLETE_REQUIRES_AUDIT',
            analysis={} if report['quick'] else analysis,
            quick_diagnostic_only=analysis if report['quick'] else {},
            pairs=len(rows), model_forward_samples=model_samples, replay_max_abs_errors=max_errors,
            artifact_manifest_sha256=_file_sha256(output/'artifact_manifest.json'), **BOUNDARY,
            original_b1_status='ACTION_DIRECTION_FAIL', gate_effect='NONE',
            next_action='停止等待只读验收；不自动改模型、扩样、训练或放行MPC。')
    finally:
        progress.close()


def run(config_path: str | Path, *, quick: bool = False, preflight_only: bool = False) -> dict:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    cfg, report = preflight(config_path, quick)
    if preflight_only:
        return {k: v for k, v in report.items() if k not in ('frozen_files', 'zero_files')}
    output = _project_path(cfg['quick_directory' if quick else 'output_directory'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'preflight.json', report); write_json(output/'effective_config.json', cfg)
    write_json(output/'runtime.json', dict(torch=str(torch.__version__), cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), git=safe_git_record()))
    try:
        result = execute(cfg, report, output)
        write_json(output/'summary.json', result)
        write_json(output/'SUCCESS.json', dict(status=result['status'], summary_sha256=_file_sha256(output/'summary.json')))
        return result
    except Exception as exc:
        write_json(output/'failure.json', dict(error=str(exc), traceback=traceback.format_exc(), automatic_retry=False))
        raise
