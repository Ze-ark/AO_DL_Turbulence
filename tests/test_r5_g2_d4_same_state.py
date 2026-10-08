"""G2-D4 冻结合同、同状态检查、目标分解和配对汇总的 CPU 单元测试。"""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from scripts import diagnose_s4_r5_g2_d4_same_state as d4
from src.rl.r4_observation import R4Interface
from src.rl.s4_training import _load_yaml, _project_path


def test_contract_and_stream_isolation() -> None:
    cfg = _load_yaml(_project_path(d4.CONFIG))
    d4._contract(cfg)
    changed = deepcopy(cfg)
    changed["boundary"]["confirmation_access"] = True
    with pytest.raises(ValueError, match="合同"):
        d4._contract(changed)
    formal, quick = d4.stream_manifest(False), d4.stream_manifest(True)
    assert formal["weather_bases"] == [7_300_000 + 10 * i for i in range(4)]
    for name in ("turbulence", "sensor", "power"):
        assert len(set(formal[name])) == len(formal[name])
        assert set(formal[name]).isdisjoint(quick[name])
        assert set(formal[name]).isdisjoint(d4.d3.stream_manifest(False)[name])


def test_preflight_checks_lineage_without_creating_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(d4, "resolve_device", lambda requested: torch.device("cpu"))
    output = _project_path("outputs/s4_r5_g2_d4_same_state_v1")
    if output.exists():
        with pytest.raises(FileExistsError, match="保留已有"):
            d4.preflight()
    else:
        _, _, _, report, _, device = d4.preflight()
        assert report["paired_source_states"] == 1728
        assert report["records"] == 3456
        assert report["physical_transitions"] == 233280
        assert report["confirmation_access"] is False
        assert device.type == "cpu"  # 仅单元测试替身，真实入口必须解析 CUDA。


def _fake_state(seed: int = 23):
    interface = R4Interface()
    interface.reset(torch.zeros(1, 21), episode_id="deterministic")
    generator = lambda offset: torch.Generator().manual_seed(seed + offset)
    env = SimpleNamespace(
        turbulence_phase=torch.zeros(1, 4, 4), requested_modal=torch.zeros(1, 21),
        slm=SimpleNamespace(state=SimpleNamespace(
            phase=torch.zeros(1, 4, 4), queue=torch.zeros(2, 1, 4, 4))),
        generators=[generator(1)], sensor_generators=[generator(2)],
        power_generators=[generator(3)], measurement_generator=generator(4),
    )
    return env, interface


def test_same_state_checks_hidden_physics_and_random_streams() -> None:
    a, b = _fake_state(), _fake_state()
    assert len(d4._same_state(a, b)) == 64
    b[0].turbulence_phase[0, 0, 0] = 1
    with pytest.raises(RuntimeError, match="物理状态"):
        d4._same_state(a, b)
    c = _fake_state(24)
    with pytest.raises(RuntimeError, match="随机流"):
        d4._same_state(a, c)


class _ConstantPolicy(torch.nn.Module):
    def forward(self, history: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return history.new_full((len(history), 11), .8)


class _ProbeEnv:
    def __init__(self) -> None:
        self.config = SimpleNamespace(episode_length=10)
        self.deltas: list[torch.Tensor] = []

    def step(self, delta: torch.Tensor):
        self.deltas.append(delta.clone())
        index = len(self.deltas)
        power = torch.tensor([.7 + .01 * index])
        info = {
            "reward_power_in_bucket": power,
            "measured_power_in_bucket": torch.tensor([.7]),
            "applied_modal": torch.zeros(1, 21),
            "violation_fraction": torch.zeros(1),
            "saturated_fraction": torch.zeros(1),
            "slew_limited_fraction": torch.zeros(1),
        }
        return None, None, torch.tensor([False]), torch.tensor([False]), info


def test_branch_records_raw_clamped_objective_and_hold_only() -> None:
    interface = R4Interface()
    interface.reset(torch.zeros(1, 21), episode_id="test")
    env = _ProbeEnv()
    progress = SimpleNamespace(tick=lambda: None)
    result = d4._branch((env, interface), _ConstantPolicy(), scale=1.75, step=0,
                        hold_steps=3, action_weight=.01, smooth_weight=.001,
                        progress=progress)
    assert len(env.deltas) == 3
    assert torch.count_nonzero(env.deltas[0]) > 0
    assert torch.count_nonzero(env.deltas[1]) == 0
    assert torch.count_nonzero(env.deltas[2]) == 0
    assert result["raw_abs"].item() == pytest.approx(.8)
    assert result["clipped_fraction"].item() == pytest.approx(1)
    assert result["normalized_abs"].item() == pytest.approx(1)
    assert result["training_objective_first"].item() == pytest.approx(.689)
    assert result["physical_power_last"].item() == pytest.approx(.73)


def _synthetic_rows() -> tuple[list[dict], dict]:
    cfg = {"data": {"seed_base": 7_300_000, "weather_count": 1, "probe_steps": [0]}}
    rows = []
    for condition in d4.d3.CONDITIONS:
        for member in range(3):
            for family in d4.d3.FAMILIES:
                for slot in range(6):
                    for arm in d4.ARMS:
                        matched = arm == d4.ARMS[1]
                        row = {metric: 0.0 for metric in d4.METRICS}
                        row.update({"hardware_condition": condition, "weather_seed": 7_300_000,
                                    "probe_step": 0, "member": member, "family": family,
                                    "slot": slot, "controller": arm,
                                    "source_state_sha256": "same",
                                    "raw_abs": .3 if matched else .5,
                                    "physical_power_last": .69 if matched else .70,
                                    "training_objective_first": .71 if matched else .70})
                        rows.append(row)
    return rows, cfg


def test_summary_requires_complete_same_state_pairs() -> None:
    rows, cfg = _synthetic_rows()
    result = d4.summarize(rows, cfg)
    assert result["status"] == "DEVELOPMENT_MECHANISM_DIAGNOSTIC_NO_GATE"
    assert result["cells"]["nominal_clone"]["paired_states"] == 54
    assert result["cells"]["hardware_shift"]["matched_raw_output_smaller_fraction"] == 1
    assert result["cells"]["nominal_clone"]["first_objective_vs_delayed_power_opposite_fraction"] == 1
    with pytest.raises(RuntimeError, match="数量"):
        d4.summarize(rows[:-1], cfg)
    bad = deepcopy(rows)
    bad[1]["source_state_sha256"] = "different"
    with pytest.raises(RuntimeError, match="哈希"):
        d4.summarize(bad, cfg)
    wrong_seed = deepcopy(rows)
    wrong_seed[0]["weather_seed"] += 1
    with pytest.raises(RuntimeError, match="配对缺失"):
        d4.summarize(wrong_seed, cfg)
