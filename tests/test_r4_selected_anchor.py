"""小尺寸确定性CPU单元测试，不生成性能结果。"""
import pytest
import torch
from src.rl.r4_baselines import BaselineSpec, baseline_delta
from src.rl.r4_control import NominalCalibration, R4Limits, project_request
from src.rl.r4_dynamics import advance_history
from src.rl.r4_mpc import SearchConfig, sequence_scores
from src.rl.r4_observation import R4Interface
from src.rl.r4_selected_anchor import anchor_parameters, selected_request, selected_plan

SPEC = BaselineSpec('integrator', .15, .1)


def state():
    snap = R4Interface().reset(torch.ones(2, 21)*.2, episode_id='unit')
    h, v = snap.features, snap.valid
    h[:, -1, 21:42] = .03
    h[:, -1, 42:63] = -.07  # 必须与请求不同，否则误启跟踪项也可能蒙混过关。
    return h, v


def test_zero_correction_matches_winner_and_ignores_tracking():
    h, v = state()
    d, calls = baseline_delta(SPEC, h, v, [], NominalCalibration())
    expected = project_request(h[:, -1, 21:42], d, torch.zeros(2, 11), R4Limits())
    actual = selected_request(SPEC, h, v, torch.zeros(2, 11))
    assert calls == 0 and anchor_parameters(SPEC)['tracking_gain'] == 0
    assert torch.equal(expected.requested_delta_rad, actual.requested_delta_rad)
    changed = h.clone(); changed[:, -1, 42:63] = .8
    assert torch.equal(selected_request(SPEC, changed, v, torch.zeros(2, 11)).requested_delta_rad,
                       actual.requested_delta_rad)


class CheckingModel(torch.nn.Module):
    def forward(self, h, v, command):
        # 零修正候选的每个预测时刻均必须使用积分器，不只检查第一步。
        d, _ = baseline_delta(SPEC, h, v, [], NominalCalibration())
        expected = project_request(h[:, -1, 21:42], d, h.new_zeros(len(h), 11), R4Limits())
        assert torch.equal(command, expected.requested_delta_rad)
        return torch.cat((h[:, -1, :21]*.8, h.new_ones(len(h), 1)*.5), -1)


def test_every_internal_prediction_uses_same_anchor():
    h, v = state(); c = SearchConfig(horizon=4)
    scores = sequence_scores([CheckingModel().eval()], h, v, torch.zeros(2, 3, 4, 11),
                             anchor_parameters(SPEC), NominalCalibration(), c)
    assert torch.isfinite(scores).all()


def test_actual_first_request_and_repeatability():
    class Model(torch.nn.Module):
        def forward(self, h, v, command):
            return torch.cat((h[:, -1, :21]*.9, command[:, 10:11]), -1)
    h, v = state(); before = h.clone(); models = [Model().eval()]
    c = SearchConfig(horizon=2, population=4, iterations=2, elites=2)
    a = selected_plan(models, h, v, SPEC, NominalCalibration(), c, seed=42)
    b = selected_plan(models, h, v, SPEC, NominalCalibration(), c, seed=42)
    assert torch.equal(a['requested_delta'], b['requested_delta']) and torch.equal(h, before)


@pytest.mark.parametrize('spec', [BaselineSpec('tracking', .15, .1), BaselineSpec('ridge', .15, .1),
                                 BaselineSpec('integrator', .25, .1)])
def test_other_baselines_not_silently_approximated(spec):
    with pytest.raises(ValueError): anchor_parameters(spec)
