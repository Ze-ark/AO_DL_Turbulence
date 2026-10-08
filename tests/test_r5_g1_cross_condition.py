"""R5-G1 仅用合成记录测试冻结条件与配对统计。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from scripts import run_s4_r5_g1_cross_condition as g1
from scripts.confirm_s4_r5_policy_r3 import stream_manifest as r3_stream_manifest
from src.rl.s4_training import _load_yaml, _project_path


def test_seed_streams_are_unique_and_do_not_reuse_r3() -> None:
    manifests = [g1.stream_manifest(arm, quick) for arm in ("wind", "hardware")
                 for quick in (False, True)]
    combined = [seed for manifest in manifests for seed in manifest["turbulence"]]
    assert len(set(combined)) == len(combined) == 2 * (1152 + 2)
    assert set(combined).isdisjoint(r3_stream_manifest(False)["turbulence"])
    for manifest in manifests:
        assert len(set(manifest["sensor"])) == len(manifest["sensor"])
        assert len(set(manifest["power"])) == len(manifest["power"])


def test_frozen_design_refuses_weather_scale_or_gate_changes() -> None:
    cfg = _load_yaml(_project_path(g1.CONFIG))
    g1._contract(cfg)
    for key, value in (("wind_speed_multiplier", 1.30),
                       ("selected_scale", 1.5),
                       ("thresholds", {"relative_power_gain": .009,
                                       "maximum_safety_increase": .001}),
                       ("data", dict(cfg["data"], wind_seed_base=6_300_000))):
        changed = deepcopy(cfg)
        changed[key] = value
        with pytest.raises(ValueError, match="冻结"):
            g1._contract(changed)


def test_both_arm_preflights_check_frozen_sources_without_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # 正式目录在 G1 完成后必须保留；仅把本测试的输出解析到临时目录。
    original_project_path = g1._project_path
    output_names = set(_load_yaml(original_project_path(g1.CONFIG))["outputs"].values())

    def isolated_output_path(value: str | Path) -> Path:
        if str(value) in output_names:
            return tmp_path / Path(value).name
        return original_project_path(value)

    monkeypatch.setattr(g1, "_project_path", isolated_output_path)
    monkeypatch.setattr(g1, "resolve_device", lambda _: torch.device("cpu"))
    for arm in ("wind", "hardware"):
        _, _, report, output, device = g1.preflight(arm=arm)
        assert report["status"] == "READY_FOR_USER_IDE"
        assert report["physical_transitions"] == 921600
        assert report["selected_scale"] == 1.75
        assert not output.exists()
        assert device.type == "cpu"  # 仅测试中替代设备；正式预检强制 CUDA。
        assert max(report["maximum_displacement_pixels"]) < report["turbulence_grid_pixels"]
    _, _, wind, _, _ = g1.preflight(arm="wind")
    assert wind["maximum_displacement_pixels"] == pytest.approx([158.4, 193.6, 242.88])
    _, _, hardware, _, _ = g1.preflight(arm="hardware")
    assert hardware["profile_ids"] == list(g1.HARDWARE_PROFILES)


def _rows(arm: str, gain: float = .01) -> list[dict]:
    cfg = _load_yaml(_project_path(g1.CONFIG))
    names = tuple(cfg["hardware_shift_profiles"]) if arm == "hardware" else g1.PROFILES
    base = g1.ARM_SEEDS[arm][0]
    rows = []
    for controller in ("integrator",) + tuple(f"policy_{member}_scale_1.75" for member in range(3)):
        for family_index, family in enumerate(g1.FAMILIES):
            for weather in range(64):
                for profile_index, profile in enumerate(names):
                    policy = controller != "integrator"
                    rows.append({"arm": arm, "controller": controller,
                                 "family": family, "profile": profile,
                                 "weather_seed": base + 10 * weather,
                                 "turbulence_stream_seed": base + 10 * weather + 1000 * profile_index + family_index,
                                 "scale": 1.75 if policy else 0.0,
                                 "power": .5 + (gain if policy else 0),
                                 "strehl": .4 + (.01 if policy else 0),
                                 "phase_rmse": .8 - (.01 if policy else 0),
                                 "violation": .001, "saturation": .001,
                                 "slew_limited": .001,
                                 "normalized_correction_clipped_fraction": .2 if policy else 0.0,
                                 "requested_applied_gap_abs": .03,
                                 "policy_forward_seconds_per_step": .003})
    return rows


@pytest.mark.parametrize("arm", ["wind", "hardware"])
def test_paired_summary_and_one_percent_gate(arm: str) -> None:
    cfg = _load_yaml(_project_path(g1.CONFIG))
    result = g1.summarize(_rows(arm), cfg, arm=arm, device=torch.device("cpu"))
    assert result["relative_power_gain"] == pytest.approx(.02)
    assert result["positive_family_profile_cells"] == 18
    assert result["absolute_power_gain_ci95"][0] > 0
    assert result["preliminary_all_gates"]
    assert result["normalized_correction_clipped_fraction"] == pytest.approx(.2)
    weak = g1.summarize(_rows(arm, .004), cfg, arm=arm, device=torch.device("cpu"))
    assert not weak["preliminary_gates"]["mean_relative_power_at_least_1pct"]


def test_missing_duplicate_or_misaligned_records_rejected() -> None:
    cfg = _load_yaml(_project_path(g1.CONFIG))
    rows = _rows("hardware")
    with pytest.raises(RuntimeError, match="缺失、重复"):
        g1.summarize(rows[:-1], cfg, arm="hardware", device=torch.device("cpu"))
    with pytest.raises(RuntimeError, match="缺失、重复"):
        g1.summarize(rows + [rows[0]], cfg, arm="hardware", device=torch.device("cpu"))
    changed = deepcopy(rows)
    changed[-1]["turbulence_stream_seed"] += 1
    with pytest.raises(RuntimeError, match="错位种子"):
        g1.summarize(changed, cfg, arm="hardware", device=torch.device("cpu"))
