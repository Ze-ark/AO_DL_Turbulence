"""G1-D3 合同、冻结来源和共同标称桥接的确定性单元测试。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import json

import pytest
import torch

from scripts import diagnose_s4_r5_g1_d2_factorial as d2
from scripts import diagnose_s4_r5_g1_d3_nominal_bridge as bridge
from src.rl.s4_training import _load_yaml, _project_path
from src.simulation.hardware_effects import HardwareProfile


def test_contract_freezes_source_weather_and_development_only_boundary() -> None:
    cfg = _load_yaml(_project_path(bridge.CONFIG))
    bridge._contract(cfg)
    for key, changed in (
        ("g1_d2_output_hashes", {}),
        ("data", dict(cfg["data"], seed_base=6_800_000)),
        ("selected_scale", 2.0),
        ("boundary", dict(cfg["boundary"], confirmation_access=True)),
    ):
        altered = deepcopy(cfg)
        altered[key] = changed
        with pytest.raises(ValueError, match="合同"):
            bridge._contract(altered)


def test_nominal_clones_share_physics_and_have_unique_slots() -> None:
    _, _, parent = bridge._verify_lineage(_load_yaml(_project_path(bridge.CONFIG)))
    clones = bridge._nominal_profiles(parent)
    assert len({item.identifier for item in clones}) == 6
    physical = [item.name for item in fields(HardwareProfile)
                if item.name not in {"identifier", "label"}]
    for other in clones[1:]:
        assert all(getattr(other, name) == getattr(clones[0], name) for name in physical)
    assert [item.identifier for item in clones] == [f"nominal_for_{item}" for item in d2.SHIFT_PROFILES]


def test_quick_streams_are_disjoint_but_formal_streams_equal_d2() -> None:
    formal, quick = bridge._manifest(False), bridge._manifest(True)
    assert formal == d2.stream_manifest(False)
    assert quick["weather_bases"] == [6_720_000]
    for name in ("turbulence", "sensor", "power"):
        assert set(formal[name]).isdisjoint(quick[name])


def test_preflight_reports_cuda_route_and_refuses_existing_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridge, "resolve_device", lambda _: torch.device("cpu"))
    for quick in (False, True):
        output = _project_path("outputs/s4_r5_g1_d3_nominal_bridge_v1_quick" if quick
                               else "outputs/s4_r5_g1_d3_nominal_bridge_v1")
        if output.exists():
            with pytest.raises(FileExistsError, match="保留已有"):
                bridge.preflight(quick=quick)
            continue
        _, _, _, report, _, device = bridge.preflight(quick=quick)
        assert report["status"] == ("READY_FOR_DIAGNOSTIC" if quick else "READY_FOR_USER_IDE")
        assert report["physical_transitions"] == (2_304 if quick else 691_200)
        assert max(report["maximum_displacement_pixels"]) < report["turbulence_grid_pixels"]
        assert report["confirmation_access"] is False
        assert device.type == "cpu"  # 测试替身；正式入口仍经 resolve_device 强制 CUDA。


def test_bridge_identity_holds_for_every_element() -> None:
    gain = torch.tensor([[.2, .3, .1], [.5, .4, .2]],
                        dtype=torch.float64).reshape(2, 3, 1, 1, 1)
    effects = bridge._effect_tensors(gain)
    assert effects["old_minus_nominal"].flatten().tolist() == pytest.approx([.1, .3])
    assert effects["new_minus_nominal"].flatten().tolist() == pytest.approx([.2, .2])
    assert effects["new_minus_old"].flatten().tolist() == pytest.approx([.1, -.1])


def test_synthetic_nominal_from_old_arm_produces_zero_old_nominal_effect() -> None:
    cfg = _load_yaml(_project_path(bridge.CONFIG))
    with (_project_path(cfg["g1_d2_output"]) / "records.jsonl").open(encoding="utf-8") as handle:
        historical = [json.loads(line) for line in handle]
    nominal = []
    for row in historical:
        if row["hardware_condition"] == "original":
            clone = dict(row)
            clone["hardware_condition"] = "nominal_bridge"
            clone["profile"] = f"nominal_for_{d2.SHIFT_PROFILES[row['slot']]}"
            nominal.append(clone)
    result = bridge.summarize(nominal, cfg, torch.device("cpu"))
    assert result["bridge_identity_checked_per_weather_family_slot"] is True
    assert result["independent_confirmation"] is False
    assert result["contrasts"]["old_minus_nominal"]["overall_absolute_advantage_change"] == pytest.approx(0)
    assert (result["contrasts"]["new_minus_nominal"]["overall_absolute_advantage_change"]
            == pytest.approx(result["contrasts"]["new_minus_old"]["overall_absolute_advantage_change"]))
