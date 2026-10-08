"""冻结选优结果到旧规划器的显式适配；不修改旧实验。"""
from __future__ import annotations

import json
from pathlib import Path
import torch
from torch import nn
from src.rl.r4_baselines import BaselineSpec, validate_history
from src.rl.r4_control import NominalCalibration, R4Limits, project_request
from src.rl.r4_dynamics_experiment import verify_hashes
from src.rl.r4_mpc import SearchConfig, plan
from src.rl.r4_trajectory import anchor_delta
from src.rl.s4_training import _file_sha256, _project_path, _relative

SELECTION_SHA256 = 'ee0b2f30b2e2507872ebc3d854bcb849a390c63c4db2f997d48ac73d08857b69'


def anchor_parameters(spec: BaselineSpec) -> dict[str, float]:
    """仅适配已验收赢家；通用字段不等于实际启用的控制项。"""
    spec.validate()
    if spec != BaselineSpec('integrator', .15, .1):
        raise ValueError('only the audited integrator is supported')
    return dict(gain=spec.gain, leak=spec.leak, tracking_gain=0.)


def load_selected() -> tuple[BaselineSpec, dict[str, str]]:
    directory = _project_path('outputs/s4_r4_baseline_selection_v1')
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    summary = read(directory/'summary.json')
    if (_file_sha256(directory/'summary.json') != SELECTION_SHA256
            or read(directory/'SUCCESS.json')['summary_sha256'] != SELECTION_SHA256):
        raise RuntimeError('selection completion changed')
    if _file_sha256(directory/'artifact_manifest.json') != summary['artifact_manifest_sha256']:
        raise RuntimeError('selection artifact manifest changed')
    frozen = read(directory/'preflight.json')['frozen_files']
    frozen.update(read(directory/'artifact_manifest.json'))
    for name in ('summary.json', 'SUCCESS.json', 'artifact_manifest.json'):
        frozen[_relative(directory/name)] = _file_sha256(directory/name)
    verify_hashes(frozen)
    selected = read(directory/'selected_baseline.json')
    if selected['index'] != 1 or selected['selection'] != summary['selection']:
        raise ValueError('selected baseline mismatch')
    spec = BaselineSpec(**selected['spec'])
    anchor_parameters(spec)
    return spec, frozen


@torch.no_grad()
def selected_request(spec: BaselineSpec, history: torch.Tensor, valid: torch.Tensor,
                     correction: torch.Tensor):
    validate_history(history, valid)
    delta = anchor_delta(history[:, -1], anchor_parameters(spec))
    return project_request(history[:, -1, 21:42], delta, correction, R4Limits())


@torch.no_grad()
def selected_plan(models: list[nn.Module], history: torch.Tensor, valid: torch.Tensor,
                  spec: BaselineSpec, calibration: NominalCalibration,
                  config: SearchConfig, *, seed: int) -> dict:
    validate_history(history, valid)
    result = plan(models, history, valid, anchor_parameters(spec), calibration, config, seed=seed)
    request = selected_request(spec, history, valid, result['correction'])
    if (not torch.equal(request.requested_delta_rad, result['requested_delta'])
            or not torch.equal(request.requested_modal_rad, result['requested_modal'])):
        raise RuntimeError('model and execution anchor mismatch')
    return result
