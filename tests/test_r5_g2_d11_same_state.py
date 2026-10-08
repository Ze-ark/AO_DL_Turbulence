"""G2-D11 CPU 确定性单元测试：状态分叉、合法动作和延迟配对；无正式性能结论。"""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from scripts import diagnose_s4_r5_g2_d11_same_state as d11
from src.rl.r4_observation import R4Interface
from src.rl.s4_training import _load_yaml, _project_path


@pytest.mark.parametrize("field,value", [("deployment_scale", 2.), ("candidate_epsilon", .2),
                                        ("source_state", "oracle"), ("runtime", {"device": "cpu"}),
                                        ("boundary", {"confirmation_access": True})])
def test_contract_rejects_scope_changes(field, value):
    cfg = _load_yaml(_project_path(d11.CONFIG))
    d11._contract(cfg)
    cfg[field] = value
    with pytest.raises(ValueError, match="合同"):
        d11._contract(cfg)


def test_fresh_weather_and_exact_execution_budget():
    d11.verify_streams()
    formal, quick = d11.stream_manifest(False), d11.stream_manifest(True)
    assert formal["weather_bases"] == [7_540_000 + 10 * i for i in range(8)]
    assert max(formal["turbulence"]) < 7_600_000
    assert set(formal["turbulence"]).isdisjoint(quick["turbulence"])
    cfg = _load_yaml(_project_path(d11.CONFIG))
    assert d11.budget(cfg["data"])["physical_transitions"] == 950400
    assert d11.budget(cfg["data"])["records"] == 64800
    assert d11.budget(cfg["data"])["paired_source_states"] == 2592
    assert d11.budget(cfg["quick"])["physical_transitions"] == 7776


def test_candidates_do_not_expand_limits_or_use_truth():
    command = torch.tensor([[1.4, -.8, -1.2, 0., .1, .2, .3, .4, .5, .6, .7]])
    original = command.clone()
    candidates = d11.candidate_commands(command, .1)
    assert list(candidates) == list(d11.CANDIDATES)
    assert len(candidates) == 25
    assert torch.equal(command, original)
    assert torch.equal(candidates["original"].clamp(-1, 1), candidates["equivalent_clamp"])
    assert candidates["clipped_inward"][0, :3].tolist() == pytest.approx([.9, -.8, -.9])
    for name, value in candidates.items():
        if name != "original":
            assert bool((value.abs() <= 1).all())
        if name.startswith("coordinate"):
            assert int(value.ne(command.clamp(-1, 1)).sum()) <= 1
    with pytest.raises(ValueError):
        d11.candidate_commands(command[:, :10], .1)


def fake_state():
    interface = R4Interface()
    interface.reset(torch.zeros(1, 21), episode_id="unit")
    env = SimpleNamespace(turbulence_phase=torch.zeros(1, 4, 4), requested_modal=torch.zeros(1, 21),
                          slm=SimpleNamespace(state=SimpleNamespace(phase=torch.zeros(1, 4, 4), queue=torch.zeros(2, 1, 4, 4))),
                          generators=[torch.Generator().manual_seed(1)], sensor_generators=[torch.Generator().manual_seed(2)],
                          power_generators=[torch.Generator().manual_seed(3)], measurement_generator=torch.Generator().manual_seed(4), step_count=0)
    return env, interface


def test_fork_copies_queues_interfaces_and_rng_without_mutating_source():
    source = fake_state()
    original_features = source[1]._features.clone()
    branch = d11.fork_state(source)
    assert len(d11.require_same_state(source, branch)) == 64
    branch[0].slm.state.queue.add_(1)
    assert not bool(source[0].slm.state.queue.any())
    torch.randn(3, generator=branch[0].generators[0])
    assert not torch.equal(source[0].generators[0].get_state(), branch[0].generators[0].get_state())
    branch[1]._features.add_(1)
    assert torch.equal(source[1]._features, original_features)
    branch = d11.fork_state(source)
    branch[1].estimator._queue[0].add_(1)
    with pytest.raises(RuntimeError, match="队列"):
        d11.require_same_state(source, branch)


class DelayedEnv:
    def __init__(self):
        self.config = SimpleNamespace(episode_length=20)
        self.deltas = []
    def step(self, delta):
        self.deltas.append(delta.clone())
        power = torch.tensor([.7 + .01 * len(self.deltas)])
        info = {name: power.clone() if "power" in name or "strehl" in name else torch.zeros(1)
                for name in d11.TRACE_METRICS.values()}
        info["applied_modal"] = torch.zeros(1, 21)
        return None, None, torch.tensor([False]), torch.tensor([False]), info


def test_branch_issues_one_action_and_holds_request():
    interface = R4Interface()
    interface.reset(torch.zeros(1, 21), episode_id="unit")
    env = DelayedEnv()
    result = d11.probe_branch((env, interface), torch.ones(1, 11), step=0, hold_steps=5,
                             progress=SimpleNamespace(tick=lambda metrics: None))
    assert len(env.deltas) == 5
    assert bool(env.deltas[0].any())
    assert all(not bool(v.any()) for v in env.deltas[1:])
    assert result["power"].shape == (1, 5)
    assert result["requested_delta"].shape == (1, 21)


def synthetic(monkeypatch):
    cfg = _load_yaml(_project_path(d11.CONFIG))
    cfg["data"].update(weather_count=1, initializations=1, probe_steps=[5], hold_steps=5)
    cfg["statistics"]["bootstrap_repeats"] = 20
    monkeypatch.setattr(d11, "stream_manifest", lambda quick: {"weather_bases": [7_540_000]})
    rows = []
    original_command = torch.tensor([[1.4] * 6 + [.2] * 5], dtype=torch.float64)
    commands = d11.candidate_commands(original_command, cfg["candidate_epsilon"])
    for condition in d11.CONDITIONS:
        for family in d11.FAMILIES:
            for slot, profile in enumerate(d11.PROFILES):
                for name in d11.CANDIDATES:
                    effect = .001 if name == "clipped_inward" else .002 if name.startswith("coordinate") else 0.
                    power = [.7, .7] + [.7 + effect] * 3
                    row = {key: [0.] * 5 for key in d11.TRACE_METRICS}
                    row.update(hardware_condition=condition, weather_seed=7_540_000, probe_step=5, member=0,
                               family=family, slot=slot, profile=f"nominal_for_{profile}" if condition == "nominal_clone" else profile,
                               candidate=name, source_state_sha256="0" * 64, slm_delay_frames=2, hold_steps=5,
                               deployment_scale=1.75, hold_rule="one_action_then_zero_request_increment",
                               original_clipped_fraction=6/11, power=power, post_arrival_power=.7 + effect,
                               violation_max=0., saturation_max=0., slew_max=0.,
                               command=commands[name][0].tolist(), normalized=commands[name][0].clamp(-1, 1).tolist(),
                               requested_delta=[0.] * 21, requested_modal=[0.] * 21)
                    rows.append(row)
    return rows, cfg


def test_summary_separates_fixed_intervention_from_truth_selected_upper_bound(monkeypatch):
    rows, cfg = synthetic(monkeypatch)
    result = d11.summarize(rows, cfg, device=torch.device("cpu"))
    assert len(result["group_table"]) == 36
    for cell in result["cells"].values():
        assert cell["weather_clusters"] == 1
        assert cell["fixed_clipped_inward_power_delta"] == pytest.approx(.001)
        assert cell["fixed_clipped_inward_descriptive_ci95"] == pytest.approx([.001, .001])
        assert cell["oracle_candidate_gap"] == pytest.approx(.002)
    assert result["prior_d9_gate_reclassified"] is False
    assert "not_deployable" in result["oracle_scope"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "hash", "profile", "nan", "delay", "unsafe", "equivalent", "before_arrival"])
def test_summary_fail_closed(monkeypatch, mutation):
    rows, cfg = synthetic(monkeypatch)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif mutation == "hash":
        rows[2]["source_state_sha256"] = "1" * 64
    elif mutation == "profile":
        rows[2]["profile"] = "wrong"
    elif mutation == "nan":
        rows[2]["power"][3] = float("nan")
    elif mutation == "delay":
        rows[2]["slm_delay_frames"] = 9
    elif mutation == "unsafe":
        rows[2]["requested_delta"][0] = 1.
    elif mutation == "equivalent":
        rows[1]["strehl"][4] = .001
    else:
        rows[2]["power"][0] = .9
    with pytest.raises(RuntimeError, match="G2-D11"):
        d11.summarize(rows, cfg, device=torch.device("cpu"))


def test_safety_filtered_oracle_excludes_unsafe_candidate(monkeypatch):
    rows, cfg = synthetic(monkeypatch)
    for r in rows:
        if r["candidate"].startswith("coordinate"):
            r["violation_max"] = .002
            r["violation"] = [.0, .0, .002, .002, .002]
    result = d11.summarize(rows, cfg, device=torch.device("cpu"))
    for cell in result["cells"].values():
        assert cell["oracle_candidate_gap"] == pytest.approx(.002)
        assert cell["safety_filtered_oracle_gap"] == pytest.approx(.001)


def test_changed_helper_fails_before_device_and_output(monkeypatch):
    monkeypatch.setattr(d11, "_file_sha256", lambda path: "0" * 64)
    with pytest.raises(RuntimeError, match="冻结工具"):
        d11.preflight()
