"""R5-D2 仅CPU确定性合同测试；不生成正式性能结果。"""
from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from scripts.diagnose_s4_r5_margin_d2 import (
    CorrectionClampTelemetry,
    FAMILIES,
    PROFILES,
    SCALES,
    _contract,
    _stream_manifest,
    summarize_development,
)
from src.rl.s4_training import _load_yaml, _project_path


CONFIG = "configs/experiments/s4_r5_margin_development_d2_v1.yaml"


class FixedPolicy(nn.Module):
    def forward(self, history: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        value = history.new_tensor([0.8, -0.6, 0.3] + [0.0] * 8)
        return value.expand(len(history), -1)


def test_clamp_meter_preserves_policy_and_counts_preclip_values() -> None:
    policy = FixedPolicy()
    meter = CorrectionClampTelemetry(policy, 2.0)
    history = torch.zeros(2, 8, 79)
    valid = torch.ones(2, 8, dtype=torch.bool)
    for _ in range(2):
        assert torch.equal(meter(history, valid), policy(history, valid))
    assert meter.rates(2, 2) == pytest.approx([2 / 11, 2 / 11])
    with pytest.raises(RuntimeError, match="incomplete"):
        meter.rates(3, 2)


def test_new_weather_streams_unique_and_separate_from_prior_r5() -> None:
    formal = _stream_manifest(False, 3, 6)
    quick = _stream_manifest(True, 1, 2)
    assert len(formal["turbulence"]) == len(set(formal["turbulence"])) == 576
    assert len(quick["turbulence"]) == len(set(quick["turbulence"])) == 2
    assert min(formal["turbulence"]) == 6200000
    assert max(formal["turbulence"]) == 6205312
    assert set(formal["turbulence"]).isdisjoint(quick["turbulence"])
    assert min(formal["turbulence"]) > 5800000
    assert len(set(formal["sensor"])) == len(formal["sensor"]) == 192
    assert len(set(formal["power"])) == len(formal["power"]) == 192


def test_contract_rejects_scale_or_selection_drift() -> None:
    cfg = _load_yaml(_project_path(CONFIG))
    _contract(cfg)
    changed = deepcopy(cfg)
    changed["scales"][1] = 1.6
    with pytest.raises(ValueError, match="frozen development"):
        _contract(changed)
    changed = deepcopy(cfg)
    changed["selection"]["maximum_safety_increase"] = 0.01
    with pytest.raises(ValueError, match="frozen development"):
        _contract(changed)


def _synthetic_rows(quick: bool) -> list[dict]:
    cfg = _load_yaml(_project_path(CONFIG))
    spec = cfg["quick"] if quick else cfg["data"]
    families = FAMILIES[:1] if quick else FAMILIES
    profiles = ("nominal", "combined_moderate") if quick else PROFILES
    records = []
    for weather in range(spec["weather_count"]):
        seed = spec["seed_base"] + 10 * weather
        for family in families:
            for profile in profiles:
                base = {"controller": "integrator", "scale": 0.0,
                        "weather_seed": seed, "family": family, "profile": profile,
                        "power": 0.7, "strehl": 0.5, "phase_rmse": 0.9,
                        "violation": 0.01, "saturation": 0.001,
                        "slew_limited": 0.001,
                        "normalized_correction_clipped_fraction": 0.0}
                records.append(base)
                for member in range(3):
                    for scale, gain in zip(SCALES, (0.007, 0.008, 0.009, 0.010), strict=True):
                        row = dict(base)
                        row.update(controller=f"policy_{member}_scale_{scale}", scale=scale,
                                   power=base["power"] + gain,
                                   strehl=base["strehl"] + 0.005,
                                   phase_rmse=base["phase_rmse"] - 0.005,
                                   violation=base["violation"] + (0.002 if scale == 2.0 else 0),
                                   normalized_correction_clipped_fraction=0.05)
                        records.append(row)
    return records


def test_predeclared_selection_excludes_unsafe_best_and_uses_one_global_scale() -> None:
    cfg = _load_yaml(_project_path(CONFIG))
    result = summarize_development(_synthetic_rows(False), cfg, quick=False)
    assert len(result["candidate_comparisons"]) == 4
    assert result["selected_scale"] == 1.75
    assert result["development_go_for_new_confirmation_design"] is True
    assert result["candidate_comparisons"][-1]["eligible"] is False


def test_smoke_never_recommends_confirmation_and_duplicate_rows_fail() -> None:
    cfg = _load_yaml(_project_path(CONFIG))
    rows = _synthetic_rows(True)
    result = summarize_development(rows, cfg, quick=True)
    assert result["status"] == "QUICK_SMOKE_NO_CONCLUSION"
    assert result["selected_scale"] is None
    assert "candidate_comparisons" not in result
    assert result["development_go_for_new_confirmation_design"] is False
    with pytest.raises(RuntimeError, match="incomplete or duplicated"):
        summarize_development(rows + [rows[0]], cfg, quick=True)
