"""只读CUDA接入检查：重放保存的请求，两个模型决策，不推进物理环境。"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.rl.r4_selected_anchor import load_selected, selected_request, selected_plan
from src.rl.r4_mpc import SearchConfig
from src.rl.r4_control import NominalCalibration
from src.rl.r4_response_experiment import load_models
from src.rl.r4_dynamics_experiment import verify_hashes
from src.rl.s4_training import _project_path, _load_yaml
from src.runtime import resolve_device


@torch.no_grad()
def main() -> None:
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if os.environ['CUBLAS_WORKSPACE_CONFIG'] not in (':4096:8', ':16:8'):
        raise ValueError('unsupported deterministic setting')
    device = resolve_device('cuda')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    spec, frozen = load_selected()
    directory = _project_path('outputs/s4_r4_baseline_selection_v1')
    records = json.loads((directory/'trajectory_manifest.json').read_text(encoding='utf-8'))
    count = 0; sample = None
    for row in records:
        if row['candidate'] != 1: continue
        data = torch.load(_project_path(row['file']), map_location=device, weights_only=True)
        frames = data['frames']; b, t = data['requested_delta'].shape[:2]
        indices = torch.arange(t, device=device)[:, None]-torch.arange(7, -1, -1, device=device)
        valid = (indices >= 0)[None].expand(b, -1, -1).reshape(b*t, 8)
        history = frames[:, indices.clamp_min(0)].reshape(b*t, 8, 79).clone()
        history.masked_fill_(~valid[:, :, None], 0)
        request = selected_request(spec, history, valid, frames.new_zeros(b*t, 11))
        if not torch.equal(request.requested_delta_rad, data['requested_delta'].reshape(b*t, 21)):
            raise RuntimeError('saved integrator request replay mismatch')
        if not torch.equal(request.requested_modal_rad, data['requested_modal'].reshape(b*t, 21)):
            raise RuntimeError('saved cumulative request replay mismatch')
        if sample is None: sample = (history[80:81].clone(), valid[80:81].clone())
        count += b*t
    if count != 115200 or sample is None: raise RuntimeError('incomplete selected baseline records')
    print('115200个已存动作重放一致；开始2次原规模模型内决策检查。', flush=True)
    models = load_models(_project_path('outputs/s4_r4_dynamics_v1'), device)[1::2]
    for i, model in enumerate(models):
        path = _project_path(f'outputs/s4_r4_delta_supervision_v1/checkpoints/absolute_plus_delta_{i}_02000.pt')
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True)['state_dict'])
        model.eval().requires_grad_(False)
    parent = _load_yaml(_project_path('configs/experiments/s4_r4_dynamics_v1.yaml'))
    cal = NominalCalibration(**parent['nominal_calibration'])
    h, v = sample
    first = selected_plan(models, h, v, spec, cal, SearchConfig(), seed=3769300)
    second = selected_plan(models, h, v, spec, cal, SearchConfig(), seed=3769300)
    if not all(torch.equal(first[k], second[k]) for k in first if torch.is_tensor(first[k])):
        raise RuntimeError('decisions not reproducible')
    verify_hashes(frozen)
    print(json.dumps(dict(status='ANCHOR_INTEGRATION_DIAGNOSTIC_PASS', replayed_requests=count,
        model_decisions=2, model_forward_samples=first['model_forward_samples']+second['model_forward_samples'],
        physical_transitions=0, training_updates=0, real_slm_actions=False,
        formal_comparison_run=False, frozen_files=len(frozen)), indent=2))


if __name__ == '__main__': main()
