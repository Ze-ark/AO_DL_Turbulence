"""G2-D12-A CPU 小型确定性测试；不产生正式性能结果。"""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from scripts import prepare_s4_r5_g2_d12_causal_dataset as d12
from src.rl.r4_control import R4Limits, project_request


def test_exact_budgets_and_frozen_configuration():
    cfg = d12.source._load_yaml(d12.source._project_path(d12.CONFIG))
    d12.contract(cfg)
    parent = d12.source._load_yaml(d12.source._project_path(d12.source.CONFIG))
    assert d12.budget(parent["data"]) == {"samples": 2592, "complete_source_episodes": 864,
        "batched_environment_steps": 9600, "physical_transitions": 172800,
        "policy_forward_calls": 9600, "policy_forward_sample_steps": 172800, "new_candidate_probes": 0}
    assert d12.budget(parent["quick"])["physical_transitions"] == 576
    assert d12.budget(parent["quick"])["samples"] == 36


@pytest.mark.parametrize("field,value", [("input_fields", ["history", "profile"]),
    ("split", {"unit": "random_frame"}), ("runtime", {"device": "cpu"}),
    ("boundary", {"training_updates": 1}), ("source_config", "other.yaml")])
def test_contract_rejects_leakage_or_scope_changes(field, value):
    cfg = d12.source._load_yaml(d12.source._project_path(d12.CONFIG))
    cfg[field] = value
    with pytest.raises(ValueError, match="合同"):
        d12.contract(cfg)


def history_fixture(step=5):
    history = torch.zeros(2, 8, 79)
    times = torch.arange(step - 7, step + 1)
    valid = (times >= 0).expand(2, -1).clone()
    history[:, valid[0], 75] = times[valid[0]].float()
    history[:, valid[0], 76] = times[valid[0]].float() - 1
    history[:, valid[0], 77] = times[valid[0]].float() - 1
    history[:, valid[0], 78] = (times[valid[0]] > 0).float()
    return history, valid, torch.zeros(2, 11)


@pytest.mark.parametrize("step", [5, 25, 75, 150])
def test_only_causal_inputs_are_copied(step):
    history, valid, command = history_fixture(step)
    result = d12.causal_inputs(history, valid, command, step=step)
    assert set(result) == {"history", "valid", "command"}
    history.add_(10); command.add_(1)
    assert result["history"][0, -1, 75] == step
    assert not result["command"].any()


@pytest.mark.parametrize("mutation", ["future_observation", "future_power", "command_time", "valid", "nan", "shape"])
def test_causal_guard_fails_closed(mutation):
    history, valid, command = history_fixture()
    if mutation == "future_observation": history[0, -1, 75] = 6
    elif mutation == "future_power": history[0, -1, 77] = 5
    elif mutation == "command_time": history[0, -1, 76] = 5
    elif mutation == "valid": valid[0, 0] = True
    elif mutation == "nan": command[0, 0] = float("nan")
    else: command = command[:, :10]
    with pytest.raises((RuntimeError, ValueError)):
        d12.causal_inputs(history, valid, command, step=5)


def target_fixture():
    command = torch.zeros(1, 11)
    projected = project_request(torch.zeros(1, 21), torch.zeros(1, 21), command, R4Limits())
    action = type(projected)(*(getattr(projected, k)[0] for k in projected.__dataclass_fields__))
    variants = {}
    for name in d12.source.CANDIDATES:
        gain = .002 if name == "clipped_inward" else 0.
        row = {"profile": "unit", "source_state_sha256": "a" * 64, "slm_delay_frames": 2,
               "hold_steps": 5, "power": [.7, .7] + [.7 + gain] * 3,
               "measured_power": [.68, .68] + [.68 + gain] * 3,
               "post_arrival_power": .7 + gain}
        for key in ("violation", "saturation", "slew"):
            row[key] = [0.] * 5; row[f"{key}_max"] = 0.
        for key, value in (("command", command[0]), ("normalized", action.normalized_correction),
                ("requested_delta", action.requested_delta_rad), ("requested_modal", action.requested_modal_rad)):
            row[key] = value.tolist()
        variants[name] = row
    kwargs = dict(command=command[0], action=action, state_hash="a" * 64,
                  profile=SimpleNamespace(identifier="unit", slm_delay_frames=2), device=torch.device("cpu"), tolerance=1e-7)
    return variants, kwargs


def test_labels_are_separate_preserve_immediate_safety_and_delay_window():
    variants, kwargs = target_fixture()
    variants["clipped_inward"]["violation"][0] = .002
    variants["clipped_inward"]["violation_max"] = .002
    result = d12.aligned_targets(variants, **kwargs)
    assert result["power_delta"].shape == (25,)
    assert float(result["power_delta"][2]) == pytest.approx(.002)
    assert result["safety_maxima"][2, 0] == .002
    assert not result["safety_pass"][2]
    assert "history" not in result


@pytest.mark.parametrize("mutation", ["hash", "profile", "delay", "command", "request", "mean", "max", "nan", "equivalent", "range"])
def test_label_alignment_fails_closed(mutation):
    variants, kwargs = target_fixture()
    row = variants["clipped_inward"]
    if mutation == "hash": row["source_state_sha256"] = "b" * 64
    elif mutation == "profile": row["profile"] = "wrong"
    elif mutation == "delay": row["slm_delay_frames"] = 1
    elif mutation == "command": variants["original"]["command"][0] = .1
    elif mutation == "request": variants["original"]["requested_delta"][0] = .1
    elif mutation == "mean": row["post_arrival_power"] += .1
    elif mutation == "max": row["violation_max"] = .1
    elif mutation == "nan": row["power"][0] = float("nan")
    elif mutation == "equivalent": variants["equivalent_clamp"]["measured_power"][4] += .1
    else: row["violation"][0] = 1.2; row["violation_max"] = 1.2
    with pytest.raises(RuntimeError, match="G2-D12-A"):
        d12.aligned_targets(variants, **kwargs)


def test_source_integrity_failure_precedes_cuda_and_output(monkeypatch):
    monkeypatch.setattr(d12.source, "_file_sha256", lambda path: "0" * 64)
    monkeypatch.setattr(d12.source, "resolve_device", lambda name: pytest.fail("must fail before CUDA"))
    with pytest.raises(RuntimeError, match="冻结来源"):
        d12.preflight()


def test_source_grid_rejects_missing_data(tmp_path):
    (tmp_path / "records.jsonl").write_text("", encoding="utf-8")
    upstream = d12.source._load_yaml(d12.source._project_path(d12.source.CONFIG))
    with pytest.raises(RuntimeError, match="完整网格"):
        d12.load_groups(tmp_path, upstream["quick"], quick=True)
