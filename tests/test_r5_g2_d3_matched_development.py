"""G2-D3 配对闭环评价合同、随机流与天气聚类统计的确定性测试。"""
from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from scripts import diagnose_s4_r5_g2_d3_matched_development as d3
from src.rl.s4_training import _load_yaml, _project_path


def test_contract_rejects_changed_gate_or_lineage() -> None:
    cfg = _load_yaml(_project_path(d3.CONFIG))
    d3._contract(cfg)
    for field, replacement in (
        ("deployment_scale", 1.0),
        ("data", dict(cfg["data"], weather_count=24)),
        ("thresholds", dict(cfg["thresholds"], minimum_relative_gain=0.0)),
        ("boundary", dict(cfg["boundary"], confirmation_access=True)),
        ("training_output_hashes", {}),
    ):
        changed = deepcopy(cfg)
        changed[field] = replacement
        with pytest.raises(ValueError, match="合同"):
            d3._contract(changed)


def test_streams_are_paired_unique_and_disjoint() -> None:
    formal, smoke = d3.stream_manifest(False), d3.stream_manifest(True)
    assert formal["weather_bases"] == [7_100_000 + 10 * i for i in range(32)]
    assert len(set(formal["turbulence"])) == 32 * 18
    assert len(set(formal["sensor"])) == len(set(formal["power"])) == 32 * 6
    for name in ("turbulence", "sensor", "power"):
        assert set(formal[name]).isdisjoint(smoke[name])
        assert set(formal[name]).isdisjoint(d3.train.stream_manifest(quick=False)[name])
        assert set(formal[name]).isdisjoint(d3.train.stream_manifest(quick=True)[name])
    assert max(formal["turbulence"]) < 7_200_000


def _synthetic_rows(*, matched_power: dict[str, float] | None = None) -> list[dict]:
    power_by_condition = matched_power or {name: .709 for name in d3.CONDITIONS}
    rows = []
    for condition in d3.CONDITIONS:
        for controller in d3.CONTROLLERS:
            baseline = controller == "integrator"
            old = controller.startswith("original_")
            matched = controller.startswith("train_scale_1_75_")
            power = (.7 if baseline else .706 if old else power_by_condition[condition]
                     if matched else .707)
            for family_index, family in enumerate(d3.FAMILIES):
                for seed in d3.stream_manifest(False)["weather_bases"]:
                    for slot, profile in enumerate(d3.PROFILES):
                        row = {name: 0.0 for name in d3.METRICS}
                        row.update({"hardware_condition": condition, "controller": controller,
                                    "family": family, "slot": slot, "weather_seed": seed,
                                    "profile": f"nominal_for_{profile}" if condition == "nominal_clone"
                                    else profile,
                                    "scale": 0.0 if baseline else d3.SCALE,
                                    "turbulence_stream_seed": seed + 1000 * slot + family_index,
                                    "power": power,
                                    "strehl": .7 if baseline else .71 if not matched else .72,
                                    "phase_rmse": .3 if baseline else .29 if not matched else .28})
                        rows.append(row)
    return rows


def test_summarize_uses_complete_weather_pairs_and_both_conditions() -> None:
    cfg = _load_yaml(_project_path(d3.CONFIG))
    rows = _synthetic_rows()
    result = d3.summarize(rows, cfg, device=torch.device("cpu"))
    assert result["continue_criteria"]["all"] is True
    assert result["paired_unit"].startswith("same_complete_weather")
    for name in d3.CONDITIONS:
        cell = result["cells"][name]
        assert cell["matched_minus_unmatched_absolute_power"] == pytest.approx(.002)
        assert cell["matched_minus_unmatched_paired_weather_ci97_5"][0] > 0
        assert cell["matched_relative_gain"] > .0105
    altered = _synthetic_rows(matched_power={"nominal_clone": .709, "hardware_shift": .706})
    result = d3.summarize(altered, cfg, device=torch.device("cpu"))
    assert result["continue_criteria"]["nominal_clone"]["all"] is True
    assert result["continue_criteria"]["hardware_shift"]["all"] is False
    assert result["continue_criteria"]["all"] is False
    with pytest.raises(RuntimeError, match="缺失"):
        d3.summarize(rows[:-1], cfg, device=torch.device("cpu"))


def test_preflight_is_read_only_and_counts_complete_episodes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(d3, "resolve_device", lambda requested: torch.device("cpu"))
    for quick, episodes, transitions in ((False, 11_520, 2_304_000), (True, 360, 5_760)):
        output = _project_path("outputs/s4_r5_g2_d3_matched_development_v1_quick" if quick
                               else "outputs/s4_r5_g2_d3_matched_development_v1")
        if output.exists():
            with pytest.raises(FileExistsError, match="保留已有"):
                d3.preflight(quick=quick)
            continue
        _, _, _, report, _, device = d3.preflight(quick=quick)
        assert report["complete_episodes"] == episodes
        assert report["physical_transitions"] == transitions
        assert report["confirmation_access"] is False
        assert report["training_updates"] == 0
        assert report["real_slm_actions"] is False
        assert max(report["maximum_displacement_pixels"]) < report["turbulence_grid_pixels"]
        assert device.type == "cpu"  # 仅单元测试替身；真实入口必须解析 CUDA。
