"""G2-D5 冻结梯度诊断的小型确定性 CPU 测试；不产生科学结果。"""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from scripts import diagnose_s4_r5_g2_d5_gradient as d5
from src.rl.s4_training import _load_yaml, _project_path


def test_contract_and_streams_are_isolated() -> None:
    cfg = _load_yaml(_project_path(d5.CONFIG))
    d5._contract(cfg)
    changed = deepcopy(cfg)
    changed["boundary"]["confirmation_access"] = True
    with pytest.raises(ValueError, match="合同"):
        d5._contract(changed)

    formal, quick = d5.stream_manifest(False), d5.stream_manifest(True)
    assert formal["weather_bases"] == [7_320_000, 7_320_010]
    assert quick["weather_bases"] == [7_330_000]
    historical = (
        d5.d4.stream_manifest(False), d5.d4.stream_manifest(True),
        d5.d4.d3.stream_manifest(False), d5.d4.d3.stream_manifest(True),
        d5.d4.d3.train.stream_manifest(quick=False),
        d5.d4.d3.train.stream_manifest(quick=True),
        d5.d4.d3.train.g2.stream_manifest(False),
        d5.d4.d3.train.g2.stream_manifest(True),
    )
    for name in ("turbulence", "sensor", "power"):
        assert len(set(formal[name])) == len(formal[name])
        assert len(set(quick[name])) == len(quick[name])
        assert set(formal[name]).isdisjoint(quick[name])
        old = set().union(*(set(stream[name]) for stream in historical))
        assert set(formal[name]).isdisjoint(old)
        assert set(quick[name]).isdisjoint(old)


def test_gradient_helper_distinguishes_connected_zero_and_unconnected() -> None:
    correction = torch.tensor([[0.2, -0.3]], dtype=torch.float64, requires_grad=True)
    connected = correction.square().sum(dim=-1)
    gradient, graph_connected = d5._gradient(connected, correction)
    assert graph_connected
    torch.testing.assert_close(gradient, 2 * correction)
    gradient, graph_connected = d5._gradient(torch.tensor([0.7], dtype=torch.float64), correction)
    assert not graph_connected
    torch.testing.assert_close(gradient, torch.zeros_like(correction))
    gradient, graph_connected = d5._gradient((correction * 0).sum(dim=-1), correction)
    assert graph_connected  # 数值零与计算图断开不是一回事。
    torch.testing.assert_close(gradient, torch.zeros_like(correction))
    unrelated = torch.tensor([0.8], dtype=torch.float64, requires_grad=True)
    gradient, graph_connected = d5._gradient(unrelated, correction)
    assert not graph_connected
    torch.testing.assert_close(gradient, torch.zeros_like(correction))


def _gradient_probe() -> tuple[dict, dict, dict]:
    batch, horizon = len(d5.d4.d3.FAMILIES), 4
    correction = torch.full((batch, 11), 0.2, dtype=torch.float64)
    physical = torch.full((batch, horizon), 0.7, dtype=torch.float64)
    physical_gradient = torch.zeros(batch, horizon, 11, dtype=torch.float64)
    physical_gradient[:, 2:, 0] = 0.1
    gradient = {
        "raw": correction / 1.75,
        "correction": correction,
        "normalized": correction,
        "previous": torch.zeros_like(correction),
        "physical": physical,
        "measured_first": physical[:, 0],
        "objective_first": physical[:, 0] - 0.001,
        "action_penalty": torch.full((batch,), 0.0004),
        "smooth_penalty": torch.full((batch,), 0.00004),
        "physical_gradient": physical_gradient,
        "physical_gradient_graph_connected_batch_by_lag": [False, False, True, True],
        "measured_first_gradient": torch.zeros_like(correction),
        "measured_first_gradient_connected": False,
        "action_penalty_gradient": torch.full_like(correction, 0.001),
        "action_penalty_gradient_connected": True,
        "smooth_penalty_gradient": torch.full_like(correction, 0.0001),
        "smooth_penalty_gradient_connected": True,
        "objective_first_gradient": torch.full_like(correction, -0.0011),
        "objective_first_gradient_connected": True,
        "requested_delta": torch.zeros(batch, 21, dtype=torch.float64),
    }
    minus = {"physical": physical.clone(), "objective_first": physical[:, 0].clone()}
    plus = {"physical": physical.clone(), "objective_first": physical[:, 0].clone()}
    plus["physical"][:, 2:] += 0.01
    return gradient, minus, plus


def _probe_rows(gradient: dict, minus: dict, plus: dict) -> list[dict]:
    profile = SimpleNamespace(slm_delay_frames=2, identifier="delay_2")
    return d5._rows(
        gradient, minus, plus, condition="nominal_clone", seed=7_320_000,
        step=0, member=0, arm=d5.d4.ARMS[0], profiles=[profile],
        state_hash="same", epsilon=0.05, tolerance=1e-9,
    )


def test_rows_require_zero_response_before_delay() -> None:
    gradient, minus, plus = _gradient_probe()
    rows = _probe_rows(gradient, minus, plus)
    assert len(rows) == len(d5.d4.d3.FAMILIES)
    assert rows[0]["radial_grad_physical_power_by_lag"][:2] == [0.0, 0.0]
    assert rows[0]["radial_comparable_by_lag"] == [False, False, True, True]
    assert rows[0]["radial_direction_agrees_by_lag"] == [None, None, True, True]

    leaked_gradient = deepcopy(gradient)
    leaked_gradient["physical_gradient"][0, 0, 0] = 0.01
    with pytest.raises(RuntimeError, match="延迟到达前已有动作"):
        _probe_rows(leaked_gradient, minus, plus)

    leaked_power = deepcopy(plus)
    leaked_power["physical"][0, 1] += 0.01
    with pytest.raises(RuntimeError, match="延迟到达前硬前向"):
        _probe_rows(gradient, minus, leaked_power)


def _synthetic_rows() -> tuple[list[dict], dict]:
    cfg = {
        "quick": {"seed_base": 7_330_000, "weather_count": 1,
                  "probe_steps": [0], "initializations": 1, "hold_steps": 5},
        "gradient_zero_tolerance": 1e-9,
        "hard_forward_alpha_epsilon": 0.05,
    }
    rows = []
    for condition in d5.d4.d3.CONDITIONS:
        for family in d5.d4.d3.FAMILIES:
            for slot in range(len(d5.d4.d3.PROFILES)):
                for arm in d5.d4.ARMS:
                    rows.append({
                        "hardware_condition": condition,
                        "weather_seed": 7_330_000, "probe_step": 0,
                        "member": 0, "family": family, "slot": slot,
                        "controller": arm, "source_state_sha256": "same",
                        "correction_clipped_any": False, "correction_zero": False,
                        "radial_comparable_by_lag": [False, False, True, True, True],
                        "radial_direction_agrees_by_lag": [None, None, True, True, True],
                        "radial_grad_physical_power_by_lag": [0, 0, 0.1, 0.1, 0.1],
                        "hard_central_radial_power_by_lag": [0, 0, 0.1, 0.1, 0.1],
                        "radial_grad_action_penalty": 0.01,
                        "radial_grad_smooth_penalty": 0.001,
                        "radial_grad_training_objective_first": -0.011,
                    })
    return rows, cfg


def test_summary_requires_complete_same_state_pairs() -> None:
    rows, cfg = _synthetic_rows()
    result = d5.summarize(rows, cfg, quick=True)
    assert result["status"] == "DEVELOPMENT_GRADIENT_MECHANISM_NO_GATE"
    assert result["cells"]["nominal_clone"]["records"] == 36
    assert result["cells"]["hardware_shift"]["strata"]["all"]["per_lag"][2][
        "direction_agreement_fraction"] == 1

    with pytest.raises(RuntimeError, match="数量或索引不完整"):
        d5.summarize(rows[:-1], cfg, quick=True)
    wrong_seed = deepcopy(rows)
    wrong_seed[0]["weather_seed"] += 1
    with pytest.raises(RuntimeError, match="数量或索引不完整"):
        d5.summarize(wrong_seed, cfg, quick=True)
    duplicate = deepcopy(rows)
    duplicate[1]["controller"] = duplicate[0]["controller"]
    with pytest.raises(RuntimeError, match="控制器记录重复"):
        d5.summarize(duplicate, cfg, quick=True)
    different_state = deepcopy(rows)
    different_state[1]["source_state_sha256"] = "different"
    with pytest.raises(RuntimeError, match="哈希不一致"):
        d5.summarize(different_state, cfg, quick=True)
