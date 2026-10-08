"""R4小型确定性CPU契约测试，不生成正式性能结果。"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from src.rl.r4_control import CausalActuatorEstimate, NominalCalibration, R4Limits, project_request
from src.rl.r4_observation import PowerMeasurement, R4Interface, simulation_residual_proxy
from src.rl.r4_interface_smoke import preflight, seed_blocks
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.simulation.config import S1EnvConfig
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.hardware_effects import HardwareAwareSlmModel, HardwareEffectsConfig


def test_forbidden_simulator_fields_never_reach_proxy():
    observation = torch.arange(88, dtype=torch.float32).reshape(2, 44)
    poisoned = observation.clone()
    poisoned[:, 21:] = float("nan")
    a = simulation_residual_proxy(observation, generator=torch.Generator().manual_seed(7), noise_std_rad=.02)
    b = simulation_residual_proxy(poisoned, generator=torch.Generator().manual_seed(7), noise_std_rad=.02)
    assert torch.equal(a, b)
    clean = R4Interface().reset(a, episode_id="one")
    tainted = R4Interface().reset(b, episode_id="one")
    assert torch.equal(clean.features, tainted.features)


def test_history_is_causal_and_reset_clears_all_state():
    adapter = R4Interface()
    obs = torch.ones(2, 21)
    first = adapter.reset(obs, episode_id="one")
    assert first.valid.sum().item() == 2 and not first.features[:, :-1].any()
    assert not first.features[:, -1, 78].any()
    first.features.fill_(999)
    assert not (adapter.snapshot().features == 999).any()
    before = adapter.snapshot()
    action = adapter.issue(torch.zeros_like(obs), torch.ones(2, 11), step=0)
    action.requested_modal_rad.fill_(999)
    assert float(adapter.requested.max()) < .05
    tr = adapter.observe_next(obs * 500, step=1, power=PowerMeasurement(torch.ones(2), 0, 1))
    assert torch.equal(tr.history.features, before.features)
    assert tr.next_history.features[:, -1, :21].eq(500).all()
    reset = adapter.reset(obs, episode_id="two")
    assert reset.observation_step == 0 and reset.valid.sum().item() == 2
    assert not adapter.requested.any() and not adapter.estimator.current.any()
    assert not reset.features[:, -1, 74].any()


def test_power_label_and_late_measurement_are_not_shifted():
    adapter = R4Interface()
    obs = torch.zeros(1, 21)
    adapter.reset(obs, episode_id="one")
    adapter.issue(obs, torch.zeros(1, 11), step=0)
    tr = adapter.observe_next(obs, step=1)
    assert not tr.action_power_valid.any()
    adapter.issue(obs, torch.zeros(1, 11), step=1)
    late = adapter.observe_next(obs, step=2, power=PowerMeasurement(torch.tensor([.7]), 0, 2))
    assert not late.action_power_valid.any()
    assert late.next_history.features[0, -1, 77] == 0
    assert late.next_history.features[0, -1, 74] == pytest.approx(.7)
    adapter.issue(obs, torch.zeros(1, 11), step=2)
    current = adapter.observe_next(obs, step=3, power=PowerMeasurement(torch.tensor([.8]), 2, 3))
    assert current.action_power_valid.all() and current.action_power.item() == pytest.approx(.8)


@pytest.mark.parametrize("kind", ["future", "wrong_arrival", "negative", "nan"])
def test_bad_power_rejected_before_observation_state_advances(kind):
    a = R4Interface(); z = torch.zeros(1, 21)
    a.reset(z, episode_id="one"); a.issue(z, torch.zeros(1, 11), step=0)
    p = PowerMeasurement(torch.tensor([.5]), 0, 1)
    if kind == "future": p = replace(p, action_step=1)
    if kind == "wrong_arrival": p = replace(p, arrival_observation_step=2)
    if kind == "negative": p = replace(p, value=torch.tensor([-.5]))
    if kind == "nan": p = replace(p, value=torch.tensor([float("nan")]))
    with pytest.raises(ValueError): a.observe_next(z, step=1, power=p)
    assert a.snapshot().observation_step == 0


def test_out_of_order_commands_and_observations_fail():
    a = R4Interface(); z = torch.zeros(1, 21); c = torch.zeros(1, 11)
    with pytest.raises(RuntimeError): a.snapshot()
    a.reset(z, episode_id="one")
    with pytest.raises(RuntimeError): a.observe_next(z, step=1)
    with pytest.raises(RuntimeError): a.issue(z, c, step=1)
    a.issue(z, c, step=0)
    with pytest.raises(RuntimeError): a.issue(z, c, step=0)
    with pytest.raises(RuntimeError): a.observe_next(z, step=2)


@pytest.mark.parametrize("delay,settling", [(0, 1.), (3, 1.), (3, .5)])
def test_nominal_estimate_matches_simple_calibration_but_not_hidden_state(delay, settling):
    est = CausalActuatorEstimate(NominalCalibration(delay, settling, 1.0, "unit_test_calibration"))
    est.reset(torch.zeros(1, 1))
    slm = HardwareAwareSlmModel(-2, 2, 1, 0, delay, HardwareEffectsConfig(settling_fraction=settling))
    slm.reset((1, 1, 1), torch.device("cpu"), torch.float32)
    for t in range(8):
        command = torch.tensor([[.1 if t == 1 else 0.]])
        predicted = est.submit(command, t)
        actual, _ = slm.step(command.reshape(1, 1, 1))
        torch.testing.assert_close(predicted, actual.reshape(1, 1), rtol=0, atol=0)
    with pytest.raises(RuntimeError): est.submit(command, 0)
    est.reset(command)
    assert not est.current.any() and est.next_command_step == 0


def test_projection_respects_step_total_and_residual_budgets_under_stress():
    generator = torch.Generator().manual_seed(23)
    limits = R4Limits(); prior = torch.zeros(20, 21)
    for _ in range(40):
        base = torch.randn(20, 21, generator=generator) * 10
        correction = torch.randn(20, 11, generator=generator) * 20
        r = project_request(prior, base, correction, limits)
        assert r.normalized_correction.abs().max() <= 1
        assert r.requested_residual_rad.abs().max() <= .05
        assert r.requested_residual_rad.norm(dim=-1).max() <= math.sqrt(10)*.05
        assert r.requested_delta_rad.abs().max() <= .15
        assert r.requested_delta_rad.norm(dim=-1).max() <= math.sqrt(10)*.15 + 1e-6
        assert r.requested_modal_rad.abs().max() <= 3
        assert r.requested_modal_rad.norm(dim=-1).max() <= math.sqrt(10)*3 + 1e-6
        prior = r.requested_modal_rad
    with pytest.raises(ValueError): project_request(prior*float("nan"), base, correction, limits)
    with pytest.raises(ValueError): project_request(prior+100, base, correction, limits)


def test_interior_projection_keeps_correct_action_gradient_and_zero_correction():
    prior = torch.zeros(1, 21, dtype=torch.float64)
    correction = torch.full((1, 11), .2, dtype=torch.float64, requires_grad=True)
    r = project_request(prior, prior, correction, R4Limits())
    r.requested_delta_rad.sum().backward()
    torch.testing.assert_close(correction.grad, torch.full_like(correction, .0125), rtol=0, atol=1e-12)
    base = prior.clone(); base[:, 0] = .03
    zero = project_request(prior, base, torch.zeros_like(correction), R4Limits())
    torch.testing.assert_close(zero.requested_delta_rad, base, rtol=0, atol=1e-12)


def test_actual_env_transition_preserves_action_power_and_next_time():
    cfg = S1EnvConfig(grid_size=16, batch_size=1, num_modes=21, episode_length=2,
                      slm_quantization_levels=0, slm_delay_frames=0)
    basis, _, _ = build_action_basis(cfg, ActionRepresentation("unit", "zernike", 21), torch.device("cpu"))
    env = AdaptiveOpticsEnv(cfg, "cpu", basis_override=basis)
    raw, _ = env.reset(seed=17)
    a = R4Interface(); a.reset(raw[:, :21], episode_id="unit")
    action = a.issue(torch.zeros(1, 21), torch.ones(1, 11), step=0)
    nxt, _, _, _, info = env.step(action.requested_delta_rad)
    tr = a.observe_next(nxt[:, :21], step=1, power=PowerMeasurement(info['measured_power_in_bucket'], 0, 1))
    torch.testing.assert_close(tr.action_power, info['reward_power_in_bucket'], rtol=0, atol=0)
    assert tr.action_step == 0 and tr.next_observation_step == 1
    assert not torch.equal(info['reward_power_in_bucket'], nxt[:, -1])


def test_namespace_includes_all_seed_keys_and_cpu_config_rejected(tmp_path):
    assert seed_blocks({'training_seed_base': 3770000, 'nested': [{'episode_seeds': [3760001, 3760002]}]}) == {376,377}
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root/'configs/experiments/s4_r4_interface_smoke_v1.yaml').read_text(encoding='utf-8'))
    cfg['runtime']['device']='cpu'
    path = tmp_path/'bad.yaml'; path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    with pytest.raises(ValueError, match='CUDA'): preflight(path)


def test_import_does_not_allocate_cuda_or_write_and_cli_has_preflight():
    root = Path(__file__).resolve().parents[1]
    code = "from pathlib import Path\nfrom unittest.mock import patch\nimport torch\nwith patch.object(Path,'mkdir',side_effect=AssertionError('write')), patch.object(torch.cuda,'init',side_effect=AssertionError('gpu')):\n import src.rl.r4_interface_smoke\n"
    subprocess.run([sys.executable,'-B','-c',code],cwd=root,check=True,capture_output=True)
    help_result = subprocess.run([sys.executable,'-B','scripts/run_s4_r4_interface_smoke.py','--help'],
                                cwd=root,check=True,capture_output=True,text=True,encoding='utf-8')
    assert '--preflight-only' in help_result.stdout
