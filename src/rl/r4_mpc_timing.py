"""固定12次原规模决策，仅软件计时，不形成控制性能结论。"""
from __future__ import annotations
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import traceback
import torch

from src.rl.r4_control import NominalCalibration
from src.rl.r4_dynamics_experiment import Progress, verify_hashes, safe_git_record
from src.rl.r4_interface_smoke import write_json
from src.rl.r4_mpc import SearchConfig, plan
from src.rl.r4_response_experiment import load_models
from src.rl.r4_trajectory import EpisodeStore
from src.rl.s4_training import _project_path, _relative, _load_yaml, _file_sha256
from src.runtime import resolve_device


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def run(config_path: str | Path, *, preflight_only: bool = False) -> dict:
    cfg = _load_yaml(_project_path(config_path)); up = _project_path(cfg['upstream'])
    expected = dict(stage='S4-D2-R4-1B2-TIMING', device='cuda', diagnostic_only=True,
        starts=[0, 80, 160], repeats=2, batch_size=1, seed=3769300, physical_transitions=0,
        training_updates=0, formal_comparison_authorized=False)
    if any(cfg[k] != v for k, v in expected.items()) or cfg['search'] != asdict(SearchConfig()):
        raise ValueError('timing-only frozen scope changed')
    if cfg['summary_sha256'] != '0743feb4a454717bea1fcec407c2746047766ffa68b798cef4bf294050e85dc5':
        raise ValueError('unapproved model source')
    if _file_sha256(up/'summary.json') != cfg['summary_sha256'] or read(up/'SUCCESS.json')['summary_sha256'] != cfg['summary_sha256']:
        raise RuntimeError('upstream hash mismatch')
    summary = read(up/'summary.json')
    if summary['status'] != 'DELTA_SUPERVISION_DEVELOPMENT_PASS' or _file_sha256(up/'artifact_manifest.json') != summary['artifact_manifest_sha256']:
        raise RuntimeError('upstream incomplete')
    frozen = read(up/'preflight.json')['frozen_files']; frozen.update(read(up/'artifact_manifest.json'))
    for path in (up/'summary.json', up/'SUCCESS.json', up/'artifact_manifest.json',
                 _project_path(config_path), _project_path('src/rl/r4_mpc.py'),
                 _project_path('src/rl/r4_mpc_timing.py'), _project_path('scripts/check_s4_r4_mpc_timing.py'),
                 _project_path('tests/test_r4_mpc.py')):
        frozen[_relative(path)] = _file_sha256(path)
    if cfg['parent'] not in frozen:
        raise ValueError('parent not frozen upstream')
    verify_hashes(frozen)
    output = _project_path(cfg['output_directory'])
    if output.resolve() != _project_path('outputs/s4_r4_mpc_timing_v1').resolve() or output.exists():
        raise FileExistsError(f'preserve timing output: {output}')
    device = resolve_device('cuda')
    report = dict(status='READY_FOR_DIAGNOSTIC_TIMING', decisions=12, model_forward_samples=147456,
                  physical_transitions=0, training_updates=0, device=str(device), frozen_files=frozen)
    if preflight_only: return {k:v for k,v in report.items() if k != 'frozen_files'}
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if os.environ['CUBLAS_WORKSPACE_CONFIG'] not in (':4096:8', ':16:8'):
        raise ValueError('unsupported deterministic CUDA configuration')
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'preflight.json', report); write_json(output/'config.json', cfg)
    progress = Progress(output, device)
    try:
        parent = _load_yaml(_project_path(cfg['parent'])); old = _project_path('outputs/s4_r4_dynamics_v1')
        models = load_models(old, device); linear = models[::2]; gru = models[1::2]
        for i, model in enumerate(gru):
            state = torch.load(up/'checkpoints'/f'absolute_plus_delta_{i}_02000.pt', map_location=device, weights_only=True)
            model.load_state_dict(state['state_dict']); model.eval().requires_grad_(False)
        manifest = read(old/'data_manifest.json')
        source = next(r['file'] for r in manifest['records'] if r['split'] == 'train')
        store = EpisodeStore.from_files([_project_path(source)])
        config = SearchConfig(**cfg['search']); cal = NominalCalibration(**parent['nominal_calibration'])
        progress.phase('原规模MPC计时（无物理动作）', 12)
        calls = 0; timings = {}; results = []
        for name, ensemble in [('linear', linear), ('gru', gru)]:
            timings[name] = []
            for start in cfg['starts']:
                batch = store.windows(torch.tensor([0]), torch.tensor([start]), 8, device)
                h, valid = batch['history'], batch['valid']; before = h.clone(); previous = None
                for repeat in range(cfg['repeats']):
                    torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
                    begin = time.perf_counter()
                    result = plan(ensemble, h, valid, parent['collector_anchor'], cal, config, seed=cfg['seed']+start)
                    torch.cuda.synchronize(device); seconds = time.perf_counter()-begin
                    calls += result['model_forward_samples']
                    assert torch.equal(h, before)
                    assert bool((result['score'] >= result['zero_score']).all())
                    if previous is not None:
                        assert all(torch.equal(result[k], previous[k]) for k in result if torch.is_tensor(result[k]))
                    previous = result
                    timings[name].append(seconds)
                    row = dict(model=name, start=start, repeat=repeat, seconds=seconds,
                        peak_allocated_gb=torch.cuda.max_memory_allocated(device)/1024**3,
                        peak_reserved_gb=torch.cuda.max_memory_reserved(device)/1024**3)
                    with (output/'timing.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(row)+'\n')
                    results.append(dict(**row, **{k:v.cpu() if torch.is_tensor(v) else v for k,v in result.items()}))
                    progress.tick({'本次秒数': seconds})
        if calls != report['model_forward_samples']: raise RuntimeError('timing budget mismatch')
        verify_hashes(frozen); progress.close()
        torch.save(results, output/'decisions.pt')
        stats = {}
        for name, times in timings.items():
            values = torch.tensor(times[1:], device=device, dtype=torch.float64)
            stats[name] = dict(first_call_seconds=times[0], warm_samples=5,
                median_seconds=float(values.median()), p95_seconds=float(torch.quantile(values, .95)),
                serial_decisions_per_second=1/float(values.mean()),
                serial_115200_decisions_hours_extrapolation=float(values.mean())*115200/3600)
        write_json(output/'runtime.json', dict(torch=str(torch.__version__), cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(device), git=safe_git_record()))
        artifacts = {_relative(p):_file_sha256(p) for p in output.rglob('*') if p.is_file()}
        write_json(output/'artifact_manifest.json', artifacts)
        result = dict(status='DIAGNOSTIC_TIMING_COMPLETE', scientific_status='NO_CONTROL_PERFORMANCE_CLAIM',
            statistics=stats, source_training_file=source, decisions=12, model_forward_samples=calls,
            physical_transitions=0, training_updates=0, reproducible_decisions=True,
            batch_size=1, batched_throughput_measured=False, closed_loop_tested=False,
            safety_calibrated=False, formal_comparison_authorized=False,
            material_passport=dict(origin_skill='academic-research-suite', origin_mode='run',
                verification_status='UNVERIFIED', version_label='r4_mpc_timing_v1'),
            artifact_manifest_sha256=_file_sha256(output/'artifact_manifest.json'))
        write_json(output/'summary.json', result)
        write_json(output/'SUCCESS.json', dict(summary_sha256=_file_sha256(output/'summary.json')))
        return result
    except Exception:
        write_json(output/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise
    finally:
        progress.close()
