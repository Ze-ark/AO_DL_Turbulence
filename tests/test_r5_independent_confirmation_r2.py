"""R5-R2 合同测试：只用合成数值，不生成确认天气。"""
from __future__ import annotations

import torch

from src.rl.r5_independent_confirmation_r2 import (
    development_choice,
    source_bundle_sha256,
    summarize,
)
from src.rl.r5_margin_development import effective_stream_seeds
from src.rl.s4_training import _load_yaml, _project_path


FAMILIES = ("frozen", "boiling", "varying")
PROFILES = ("nominal", "delay_3", "settling_050", "registration_moderate",
            "registration_severe", "combined_moderate")


def test_source_and_confirmation_weather_contract() -> None:
    cfg = _load_yaml(_project_path("configs/experiments/s4_r5_independent_confirmation_r2.yaml"))
    assert source_bundle_sha256() == cfg["source_bundle_sha256"]
    streams = effective_stream_seeds(5700000, 32, 10, 6, 3)
    assert len(streams) == len(set(streams)) == 576
    assert min(streams) == 5700000
    assert max(streams) == 5705312


def test_development_selects_one_global_scale() -> None:
    rows = []
    for family in FAMILIES:
        for profile in PROFILES:
            for index in range(32):
                seed = 5240000 + index * 10
                rows.append({"controller": "integrator", "family": family,
                             "profile": profile, "weather_seed": seed,
                             "power": 0.6, "strehl": 0.5, "phase_rmse": 0.8,
                             "violation": 0.001, "saturation": 0.001,
                             "slew_limited": 0.001})
                for member in range(3):
                    for scale in (0.75, 1.0, 1.25):
                        rows.append({"controller": f"policy_{member}_scale_{scale}",
                                     "family": family, "profile": profile,
                                     "weather_seed": seed, "power": 0.6 + 0.005 * scale,
                                     "strehl": 0.51, "phase_rmse": 0.79,
                                     "violation": 0.001, "saturation": 0.001,
                                     "slew_limited": 0.001})
    assert development_choice(rows)["selected_scale"] == 1.25
    rows.pop()
    try:
        development_choice(rows)
    except RuntimeError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("missing development row was accepted")


def test_confirmation_summary_uses_paired_weather_and_original_gates() -> None:
    rows = []
    for controller in ("integrator", "policy_0_scale_1.25", "policy_1_scale_1.25",
                       "policy_2_scale_1.25"):
        for family in FAMILIES:
            for index in range(32):
                for profile in PROFILES:
                    policy = controller != "integrator"
                    rows.append({"controller": controller, "family": family,
                                 "profile": profile, "weather_seed": 5700000 + 10 * index,
                                 "power": 0.5 + (0.01 if policy else 0),
                                 "strehl": 0.4 + (0.01 if policy else 0),
                                 "phase_rmse": 0.8 - (0.01 if policy else 0),
                                 "violation": 0.001, "saturation": 0.001,
                                 "slew_limited": 0.001})
    cfg = {"statistics": {"bootstrap_seed": 17, "bootstrap_repeats": 20000},
           "thresholds": {"relative_power_gain": 0.01,
                          "maximum_safety_increase": 0.001}}
    result = summarize(rows, cfg, torch.device("cpu"))
    assert abs(result["relative_power_gain"] - 0.02) < 1e-12
    assert result["positive_family_profile_cells"] == 18
    assert result["preliminary_all_gates"]
    assert result["absolute_power_gain_ci95"][0] > 0
