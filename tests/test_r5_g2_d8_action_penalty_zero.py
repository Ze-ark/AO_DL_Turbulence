"""G2-D8 单因素训练的确定性合同测试；不启动正式训练。"""
from __future__ import annotations

import pytest
import torch

from scripts import train_s4_r5_g2_d8_action_penalty_zero as d8
from src.rl.r5_physics_adapter import measured_objective
from src.rl.s4_training import _load_yaml, _project_path


def test_only_training_objective_action_weight_changes() -> None:
    cfg = _load_yaml(_project_path(d8.CONFIG))
    d2_cfg = _load_yaml(_project_path(d8.d2.CONFIG))
    d8._contract(cfg)
    assert cfg["data"] == d2_cfg["data"]
    assert cfg["training"] == d2_cfg["training"]
    assert cfg["objective"] == {**d2_cfg["objective"], "action_weight": 0.0}
    assert cfg["new_arm"]["training_scale"] == cfg["comparator"]["training_scale"] == 1.75
    assert cfg["new_arm"]["deployment_scale"] == cfg["comparator"]["deployment_scale"] == 1.75
    assert cfg["boundary"]["confirmation_access"] is False
    assert cfg["boundary"]["development_evaluation_access"] is False
    assert cfg["boundary"]["real_slm_actions"] is False


def test_contract_rejects_a_second_factor() -> None:
    cfg = _load_yaml(_project_path(d8.CONFIG))
    cfg["objective"]["smooth_weight"] = 0.0
    with pytest.raises(ValueError, match="单因素"):
        d8._contract(cfg)


@pytest.mark.parametrize("update", [1, 2, 3, 4, 511, 512])
def test_formal_schedule_is_exactly_d2(update: int) -> None:
    assert d8.schedule(update, quick=False) == d8.d2.schedule(update, quick=False)


def test_quick_weather_is_separate_from_training_and_reserved_development() -> None:
    formal = d8.stream_manifest(quick=False)
    quick = d8.stream_manifest(quick=True)
    assert formal == d8.d2.stream_manifest(quick=False)
    assert [d8.schedule(index, quick=True) for index in (1, 2)] == [
        ("nominal_clone", d8.QUICK_BASE), ("hardware_shift", d8.QUICK_BASE)]
    assert not set(quick["turbulence"]) & set(formal["turbulence"])
    assert not set(quick["sensor"]) & set(formal["sensor"])
    assert not set(quick["power"]) & set(formal["power"])
    assert d8.QUICK_BASE > d8.DEVELOPMENT_BASE + 10 * 31
    assert d8.CONFIRMATION_BASE > d8.QUICK_BASE + 10_000


def test_budget_is_one_arm_three_initializations() -> None:
    cfg = _load_yaml(_project_path(d8.CONFIG))
    assert cfg["training"]["initializations"] * cfg["training"]["updates_per_arm_initialization"] == 1536
    assert 1536 * 18 * cfg["data"]["episode_length"] == 5_529_600
    assert cfg["quick"]["initializations"] * cfg["quick"]["updates_per_arm_initialization"] == 2


def test_action_weight_is_the_only_objective_difference_on_cpu() -> None:
    measured = torch.tensor([0.7, 0.6], dtype=torch.float64)
    correction = torch.zeros((2, 11), dtype=torch.float64)
    correction[0, :2] = torch.tensor([0.2, -0.3])
    correction[1, :2] = torch.tensor([0.4, 0.1])
    previous = torch.zeros_like(correction)
    old = measured_objective(measured, correction, previous, 0.01, 0.001)
    new = measured_objective(measured, correction, previous, 0.0, 0.001)
    expected = 0.01 * correction.square().mean(dim=-1)
    assert torch.allclose(new - old, expected)


def test_old_comparator_evidence_is_read_only_and_complete() -> None:
    cfg = _load_yaml(_project_path(d8.CONFIG))
    _, _, _, source_hashes, comparator_hashes = d8._verify_d2(cfg)
    assert set(source_hashes) == {0, 1, 2}
    assert set(comparator_hashes) == {0, 1, 2}
    assert all(len(digest) == 64 for digest in source_hashes.values())
    assert all(len(digest) == 64 for digest in comparator_hashes.values())
