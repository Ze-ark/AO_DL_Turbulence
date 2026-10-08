"""小尺寸确定性CPU测试；不是正式算法结果。"""
from __future__ import annotations

import pytest
import torch
from torch import nn
import yaml

from src.rl.r4_control import NominalCalibration
from src.rl.r4_observation import R4Interface
from src.rl.r4_response_attribution import (
    ARMS, ablated_branch, initial_history_view, paired_direction_intervals,
    preflight, response_metrics,
)

ANCHOR = dict(gain=.25, leak=.1, tracking_gain=.5)


class Recorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, history, valid, command):
        self.seen.append((history.clone(), valid.clone(), command.clone()))
        result = command.new_zeros(len(command), 22)
        result[:, 21] = .6 + command[:, 0]
        return result


def snapshot():
    interface = R4Interface()
    s = interface.reset(torch.ones(2, 21)*.1, episode_id='unit')
    return s.features, s.valid


def test_initial_mask_preserves_input_and_accumulates_future_frames():
    h = torch.ones(2, 8, 79); valid = torch.ones(2, 8, dtype=torch.bool)
    for step in range(8):
        view, mask = initial_history_view(h, valid, step)
        assert mask.sum(1).eq(step+1).all()
        assert view[:, :7-step].eq(0).all()
        assert view[:, -1].eq(1).all()
    assert h.eq(1).all() and valid.all()


def test_single_frame_negative_control_exact():
    h, v = snapshot()
    full = ablated_branch(Recorder(), h, v, ANCHOR, NominalCalibration(), 0, 1., 8, ARMS[0])
    current = ablated_branch(Recorder(), h, v, ANCHOR, NominalCalibration(), 0, 1., 8, ARMS[1])
    assert all(torch.equal(full[k], current[k]) for k in full)


def test_history_ablation_does_not_erase_nominal_command_queue():
    h, _ = snapshot(); h = h[:, -1:].repeat(1, 8, 1)
    h[:, :, 21:42] = torch.arange(8)[None, :, None]*.01
    h[:, :, 75] = torch.arange(8)
    v = torch.ones(2, 8, dtype=torch.bool)
    models = [Recorder(), Recorder()]
    full = ablated_branch(models[0], h, v, ANCHOR, NominalCalibration(), 0, 0., 8, ARMS[0])
    current = ablated_branch(models[1], h, v, ANCHOR, NominalCalibration(), 0, 0., 8, ARMS[1])
    assert torch.equal(full['requested_delta'], current['requested_delta'])
    for full_seen, masked_seen in zip(models[0].seen, models[1].seen):
        assert torch.equal(full_seen[0][:, -1, 42:63], masked_seen[0][:, -1, 42:63])
    assert models[1].seen[0][1].sum().item() == 2
    assert models[0].seen[0][1].sum().item() == 16


def test_recorded_commands_explicit_privilege_and_no_future_observation():
    h, v = snapshot(); commands = torch.zeros(2, 8, 21)
    commands[:, :, 0] = torch.arange(8)*.001
    model = Recorder()
    result = ablated_branch(model, h, v, ANCHOR, NominalCalibration(), 0, 0., 8, ARMS[2], commands)
    assert torch.equal(result['requested_delta'], commands)
    assert torch.equal(result['power'], .6+commands[:, :, 0])
    with pytest.raises(ValueError, match='privileged'):
        ablated_branch(model, h, v, ANCHOR, NominalCalibration(), 0, 0., 8, ARMS[0], commands)
    with pytest.raises(ValueError, match='privileged'):
        ablated_branch(model, h, v, ANCHOR, NominalCalibration(), 0, 0., 8, ARMS[2])
    with pytest.raises(ValueError, match='invalid recorded'):
        ablated_branch(model, h, v, ANCHOR, NominalCalibration(), 0, 0., 8, ARMS[2], commands[:, :1])


def test_metrics_keep_noise_filtered_rows_in_all_pair_error():
    a = torch.tensor([1., -1., .01]); p = torch.tensor([1., 0., 3.])
    r = response_metrics(a, p, torch.full_like(a, .1))
    assert r['total'] == 3 and r['identifiable'] == 2 and r['balanced_accuracy'] == .5
    assert r['response_rmse_all'] == pytest.approx(float((a.double()-p.double()).square().mean().sqrt()))
    assert response_metrics(a, p, torch.ones_like(a)*2)['balanced_accuracy'] is None
    with pytest.raises(ValueError, match='non-finite'):
        response_metrics(a, p*float('nan'), a)


def paired_inputs():
    a = torch.tensor([-1., 1.]).repeat(6)
    predictions = torch.stack((a, -a, a), 1)
    weather = torch.arange(6).repeat_interleave(2)
    return a, predictions, a.abs()*.1, weather, weather//2


def test_paired_ci_direction_and_same_weather_repetition():
    args = paired_inputs()
    result = paired_direction_intervals(*args, seed=19, replicates=100)
    assert result['contrasts']['history_full_minus_current'] == dict(balanced_accuracy_difference=1., ci97_5=[1., 1.])
    assert result['contrasts']['recorded_commands_minus_self'] == dict(balanced_accuracy_difference=0., ci97_5=[0., 0.])
    repeated = tuple(x.repeat_interleave(3, dim=0) for x in args)
    assert paired_direction_intervals(*repeated, seed=19, replicates=100) == result
    assert result['weather'] == 6 and result['gate_effect'] == 'NONE_DEVELOPMENT_DIAGNOSTIC_ONLY'


def test_paired_signal_empty_nonfinite_and_mixed_family_fail_closed():
    a, p, t, w, f = paired_inputs()
    assert paired_direction_intervals(a, p, t*20, w, f, 1, 10)['status'] == 'INSUFFICIENT_SIGNAL'
    assert paired_direction_intervals(a[:0], p[:0], t[:0], w[:0], f[:0], 1, 10)['status'] == 'INSUFFICIENT_SIGNAL'
    with pytest.raises(ValueError, match='non-finite'):
        paired_direction_intervals(a, p*float('nan'), t, w, f, 1, 10)
    f[0] = 2
    with pytest.raises(ValueError, match='multiple families'):
        paired_direction_intervals(a, p, t, w, f, 1, 10)


def test_preflight_cpu_rejected_without_output(tmp_path):
    path = tmp_path/'bad.yaml'
    path.write_text(yaml.safe_dump(dict(stage='S4-D2-R4-1B1-D1', runtime=dict(device='cpu'))))
    with pytest.raises(ValueError, match='CUDA'):
        preflight(path, False)
