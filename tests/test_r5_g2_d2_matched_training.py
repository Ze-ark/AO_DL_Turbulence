"""G2-D2 等预算微调合同与随机流的确定性单元测试。"""
from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from scripts import train_s4_r5_g2_d2_matched as g2d2
from src.rl.s4_training import _load_yaml, _project_path


def test_contract_freezes_equal_budget_and_no_confirmation() -> None:
    config = _load_yaml(_project_path(g2d2.CONFIG))
    g2d2._contract(config)
    for field, changed in (
        ("arms", [{"id": "matched_only", "training_scale": 1.75, "deployment_scale": 1.75}]),
        ("training", dict(config["training"], updates_per_arm_initialization=1024)),
        ("data", dict(config["data"], train_seed_base=6_800_000)),
        ("objective", dict(config["objective"], action_weight=0.0)),
        ("boundary", dict(config["boundary"], confirmation_access=True)),
        ("g2_d1_output_hashes", {}),
    ):
        altered = deepcopy(config)
        altered[field] = changed
        with pytest.raises(ValueError, match="合同"):
            g2d2._contract(altered)


def test_schedule_pairs_conditions_with_same_weather_and_disjoint_streams() -> None:
    formal = g2d2.stream_manifest(quick=False)
    smoke = g2d2.stream_manifest(quick=True)
    assert len(formal["weather_bases"]) == 256
    assert formal["weather_bases"] == [7_000_000 + 3 * i for i in range(256)]
    assert len(formal["turbulence"]) == len(set(formal["turbulence"])) == 256 * 18
    assert len(formal["sensor"]) == len(set(formal["sensor"])) == 256 * 6
    assert len(formal["power"]) == len(set(formal["power"])) == 256 * 6
    for name in ("turbulence", "sensor", "power"):
        assert set(formal[name]).isdisjoint(smoke[name])
        assert set(formal[name]).isdisjoint(g2d2.g2.stream_manifest(False)[name])
        assert set(formal[name]).isdisjoint(g2d2.g2.stream_manifest(True)[name])
    for index in range(256):
        assert g2d2.schedule(2 * index + 1) == ("nominal_clone", 7_000_000 + 3 * index)
        assert g2d2.schedule(2 * index + 2) == ("hardware_shift", 7_000_000 + 3 * index)
    with pytest.raises(ValueError, match="更新序号"):
        g2d2.schedule(513)


def test_training_scale_only_changes_action_and_retains_gradients() -> None:
    class ToyPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.raw = torch.nn.Parameter(torch.full((1, 11), 0.5))

        def forward(self, history: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            return self.raw.expand(history.shape[0], -1)

    base = ToyPolicy()
    history = torch.zeros(2, 8, 79)
    valid = torch.ones(2, 8, dtype=torch.bool)
    unmatched = g2d2.TrainingScale(base, 1.0)(history, valid)
    matched = g2d2.TrainingScale(base, 1.75)(history, valid)
    assert torch.equal(unmatched, base.raw.expand(2, -1))
    assert torch.equal(matched, unmatched * 1.75)
    matched.sum().backward()
    assert torch.equal(base.raw.grad, torch.full_like(base.raw, 3.5))
    with pytest.raises(ValueError, match="倍率"):
        g2d2.TrainingScale(base, 2.0)


def test_preflight_is_read_only_and_counts_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(g2d2, "resolve_device", lambda requested: torch.device("cpu"))
    for quick, updates, transitions in ((False, 3072, 11_059_200), (True, 4, 1152)):
        output = _project_path("outputs/s4_r5_g2_d2_matched_training_v1_quick_r1" if quick
                               else "outputs/s4_r5_g2_d2_matched_training_v1")
        if output.exists():
            with pytest.raises(FileExistsError, match="保留已有"):
                g2d2.preflight(quick=quick)
            continue
        _, _, _, report, _, device = g2d2.preflight(quick=quick)
        assert report["total_updates"] == updates
        assert report["physical_transitions"] == transitions
        assert report["confirmation_access"] is False
        assert report["real_slm_actions"] is False
        assert max(report["maximum_displacement_pixels"]) < report["turbulence_grid_pixels"]
        assert device.type == "cpu"  # 仅测试替身；真实入口必须解析 CUDA。
