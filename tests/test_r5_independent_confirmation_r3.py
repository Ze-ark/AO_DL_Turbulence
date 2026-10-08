"""R5-R3 冻结确认合同测试；仅合成数据，不打开正式天气。"""
from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from scripts.confirm_s4_r5_policy_r3 import (
    CONFIG, FAMILIES, FORMAL_BASE, PROFILES, QUICK_BASE, SCALE,
    _contract, preflight, stream_manifest, summarize,
)
from src.rl.s4_training import _load_yaml, _project_path


def test_streams_fresh_and_separate() -> None:
    formal, quick = stream_manifest(False), stream_manifest(True)
    assert len(formal["turbulence"]) == len(set(formal["turbulence"])) == 1152
    assert len(formal["sensor"]) == len(set(formal["sensor"])) == 384
    assert len(formal["power"]) == len(set(formal["power"])) == 384
    assert len(quick["turbulence"]) == 2
    assert set(formal["turbulence"]).isdisjoint(quick["turbulence"])
    assert min(formal["turbulence"]) == FORMAL_BASE
    assert min(quick["turbulence"]) == QUICK_BASE
    assert max(formal["turbulence"]) < QUICK_BASE


def test_contract_cannot_change_scale_threshold_or_weather() -> None:
    cfg = _load_yaml(_project_path(CONFIG))
    _contract(cfg)
    for path, value in (("selected_scale", 2.0), ("thresholds", {"relative_power_gain": .009,
                                                                  "maximum_safety_increase": .001}),
                        ("data", {"seed_base": FORMAL_BASE + 100, "seed_stride": 10,
                                  "weather_count": 64, "episode_length": 200})):
        changed = deepcopy(cfg)
        changed[path] = value
        with pytest.raises(ValueError, match="冻结确认合同"):
            _contract(changed)


def test_preflight_frozen_d2_selection_and_preserved_formal_output() -> None:
    formal_output = _project_path("outputs/s4_r5_independent_confirmation_r3_v1")
    if formal_output.exists():
        with pytest.raises(FileExistsError, match="保留已有确认输出"):
            preflight(CONFIG)
    else:
        _, report, _, device = preflight(CONFIG)
        assert str(device) == "cuda"
        assert report["status"] == "READY_FOR_USER_IDE"
        assert report["development_selection"]["selected_scale"] == SCALE
        assert report["development_selection"]["development_only"]
        assert report["physical_transitions"] == 921600


def _rows(gain: float = .01) -> list[dict]:
    rows = []
    for controller in ("integrator",) + tuple(f"policy_{i}_scale_{SCALE}" for i in range(3)):
        for family in FAMILIES:
            for index in range(64):
                for profile in PROFILES:
                    policy = controller != "integrator"
                    rows.append({"controller": controller, "family": family, "profile": profile,
                                 "weather_seed": FORMAL_BASE + 10 * index,
                                 "scale": SCALE if policy else 0.0,
                                 "power": .5 + (gain if policy else 0),
                                 "strehl": .4 + (.01 if policy else 0),
                                 "phase_rmse": .8 - (.01 if policy else 0),
                                 "violation": .001, "saturation": .001,
                                 "slew_limited": .001,
                                 "normalized_correction_clipped_fraction": .2 if policy else 0.0})
    return rows


def test_paired_summary_and_original_gate() -> None:
    cfg = _load_yaml(_project_path(CONFIG))
    result = summarize(_rows(), cfg, torch.device("cpu"))
    assert result["relative_power_gain"] == pytest.approx(.02)
    assert result["absolute_power_gain_ci95"][0] > 0
    assert result["positive_family_profile_cells"] == 18
    assert result["normalized_correction_clipped_fraction"] == pytest.approx(.2)
    assert result["preliminary_all_gates"]
    assert not summarize(_rows(.004), cfg, torch.device("cpu"))["preliminary_gates"]["mean_relative_power_at_least_1pct"]


def test_missing_duplicate_or_wrong_scale_cannot_be_ranked() -> None:
    cfg = _load_yaml(_project_path(CONFIG))
    rows = _rows()
    with pytest.raises(RuntimeError, match="缺失、重复"):
        summarize(rows[:-1], cfg, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="缺失、重复"):
        summarize(rows + [rows[0]], cfg, torch.device("cpu"))
    wrong = deepcopy(rows)
    wrong[-1]["scale"] = 1.5
    with pytest.raises(RuntimeError, match="倍率错误"):
        summarize(wrong, cfg, torch.device("cpu"))
