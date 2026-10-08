"""R5白名单/梯度链的小型CPU测试。"""
import inspect
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from src.rl.r4_observation import R4Interface, PowerMeasurement, simulation_residual_proxy
from src.rl.r5_physics_adapter import causal_policy_features, measured_objective
from src.rl.r5_contract_check import validate_config, direction_summary, used_seed_blocks


def test_pseudo_open_loop_sign_and_gradient():
    h = torch.zeros(1, 8, 79, dtype=torch.float64)
    h[:, -1, :21] = .7
    h[:, -1, 42:63] = .2
    h.requires_grad_()
    valid = torch.zeros(1, 8, dtype=torch.bool); valid[:, -1] = True
    transformed = causal_policy_features(h, valid)
    torch.testing.assert_close(transformed[:, -1, :21], torch.full((1, 21), .5, dtype=h.dtype))
    transformed[..., :21].sum().backward()
    assert h.grad[..., :21].eq(1).all()
    assert h.grad[..., 42:63].eq(-1).all()


def test_privileged_fields_cannot_change_legal_features():
    observation = torch.ones(1, 44)
    contaminated = observation.clone(); contaminated[:, 21:] = float("nan")
    features = []
    for raw in (observation, contaminated):
        residual = simulation_residual_proxy(raw, generator=torch.Generator().manual_seed(31), noise_std_rad=.02)
        view = R4Interface().reset(residual, episode_id="unit")
        features.append(causal_policy_features(view.features, view.valid))
    assert torch.equal(*features)
    assert list(inspect.signature(causal_policy_features).parameters) == ["history", "valid"]


def test_interface_retains_gradient_across_feedback():
    adapter = R4Interface()
    adapter.reset(torch.zeros(1, 21), episode_id="unit")
    correction = torch.ones(1, 11, requires_grad=True)*.2
    action = adapter.issue(torch.zeros(1, 21), correction, step=0)
    residual = action.requested_modal_rad*.5
    transition = adapter.observe_next(residual, step=1, power=PowerMeasurement(residual.square().sum(-1), 0, 1))
    features = causal_policy_features(transition.next_history.features, transition.next_history.valid)
    grad = torch.autograd.grad(features[:, -1, :21].sum(), correction)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_padding_and_missing_latest_observation_rejected():
    h = torch.zeros(1, 8, 79); valid = torch.zeros(1, 8, dtype=torch.bool)
    with pytest.raises(ValueError): causal_policy_features(h, valid)
    valid[:, -1] = True; h[:, 0] = 1
    with pytest.raises(ValueError): causal_policy_features(h, valid)


def test_objective_uses_only_arrived_measurement_and_action():
    c = torch.ones(1, 11)*.2
    value = measured_objective(torch.tensor([.7]), c, c*0, .01, .001)
    assert value.item() == pytest.approx(.7-.01*.04-.001*.04)


def test_configuration_rejects_cpu_design_and_changed_budget():
    root = Path(__file__).resolve().parents[1]
    source = root/'configs/experiments/s4_r5_physics_check_v1.yaml'
    for field in ('cpu','design','budget'):
        cfg = yaml.safe_load(source.read_text(encoding='utf-8'))
        validate_config(cfg)
        if field == 'cpu': cfg['runtime']['device'] = 'cpu'
        if field == 'design': cfg['design_only'] = True
        if field == 'budget': cfg['formal']['steps'] = 20
        with pytest.raises(ValueError): validate_config(cfg)


def test_ties_and_missing_positive_direction_do_not_pass():
    row = dict(objective=.5,violation_fraction=0.,saturated_fraction=0.,slew_limited_fraction=0.)
    records = [dict(seed=i,profile='nominal',branches={'0.0':row,'.25':row,'0.25':row,'-0.25':row}) for i in range(12)]
    result = direction_summary(records,9,dict(primary_scale=.25,safety_increase_maximum=.001))
    assert not result['passed'] and result['ties']==12


def test_import_has_no_side_effects_and_help_exists():
    root = Path(__file__).resolve().parents[1]
    code = "from pathlib import Path\nfrom unittest.mock import patch\nimport torch\nwith patch.object(Path,'mkdir',side_effect=AssertionError('write')), patch.object(torch.cuda,'init',side_effect=AssertionError('gpu')):\n import src.rl.r5_physics_adapter\n import src.rl.r5_contract_check\n"
    subprocess.run([sys.executable,'-B','-c',code],cwd=root,check=True,capture_output=True)
    result = subprocess.run([sys.executable,'-X','utf8','-B','scripts/check_s4_r5_physics.py','--help'],
                            cwd=root,check=True,capture_output=True,text=True,encoding='utf-8')
    assert '--quick' in result.stdout and '--preflight-only' in result.stdout


def test_quick_metadata_does_not_consume_reserved_formal_seeds():
    declared = dict(formal={'seed_base':5200000},quick={'seed_base':5280000})
    assert used_seed_blocks(declared,own_effective_config=True)==set()
    assert used_seed_blocks(declared)=={520,528}
    executed = dict(seed_audit='conservative_blocks_and_known_offsets',
                    streams={'weather':[5280000],'sensor':[55280000],'power':[65280000]})
    assert used_seed_blocks(executed)=={528,5528,6528}
    assert 520 not in used_seed_blocks(executed)
