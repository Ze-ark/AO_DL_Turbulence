"""G2-D9 天气配对、冻结证据和评价门槛的 CPU 小型确定性测试。"""
from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from scripts import diagnose_s4_r5_g2_d9_action_penalty_development as d9
from src.rl.s4_training import _load_yaml, _project_path


@pytest.mark.parametrize("field,replacement", [
    ("deployment_scale", 2.0),
    ("thresholds", {"minimum_relative_gain": .01, "maximum_safety_increase": .001}),
    ("data", {"seed_base": 7100000, "seed_stride": 10, "weather_count": 32, "episode_length": 200}),
    ("statistics", {"bootstrap_seed": 7533456, "bootstrap_repeats": 5000, "primary_interval_per_condition": .95}),
    ("controller_ids", ["integrator", "action_penalty_0_0"]),
    ("d8_final_checkpoints", {}),
    ("boundary", {"confirmation_access": True}),
])
def test_contract_rejects_changed_gate_or_comparison(field: str, replacement: object) -> None:
    cfg = _load_yaml(_project_path(d9.CONFIG))
    d9._contract(cfg)
    altered = deepcopy(cfg)
    altered[field] = replacement
    with pytest.raises(ValueError, match="合同"):
        d9._contract(altered)


def test_fresh_streams_follow_reserved_weather_and_no_collisions() -> None:
    d9._verify_stream_separation()
    formal, smoke = d9.stream_manifest(False), d9.stream_manifest(True)
    assert formal["weather_bases"] == [d9.d8.DEVELOPMENT_BASE + 10 * i for i in range(32)]
    assert smoke["weather_bases"] == [7520000]
    for name, width in (("turbulence", 18), ("sensor", 6), ("power", 6)):
        assert len(formal[name]) == len(set(formal[name])) == 32 * width
        assert set(formal[name]).isdisjoint(smoke[name])
        assert set(formal[name]).isdisjoint(d9.d8.stream_manifest(quick=True)[name])
        assert set(formal[name]).isdisjoint(d9.d8.stream_manifest(quick=False)[name])
    assert max(formal["turbulence"]) < d9.d8.CONFIRMATION_BASE
    assert len(d9.CONDITIONS) * len(d9.CONTROLLERS) * 32 * 18 == 8064


def _rows() -> list[dict]:
    result = []
    for condition in d9.CONDITIONS:
        for index, controller in enumerate(d9.CONTROLLERS):
            baseline, new = index == 0, index >= 4
            for family_index, family in enumerate(d9.FAMILIES):
                for seed in d9.stream_manifest(False)["weather_bases"]:
                    for slot, profile in enumerate(d9.PROFILES):
                        row = {name: 0.0 for name in d9.METRICS}
                        row.update({
                            "hardware_condition": condition, "controller": controller,
                            "family": family, "weather_seed": seed, "slot": slot,
                            "profile": f"nominal_for_{profile}" if condition == "nominal_clone" else profile,
                            "episode_length": 200, "scale": 0.0 if baseline else 1.75,
                            "turbulence_stream_seed": seed + 1000 * slot + family_index,
                            "power": .7 if baseline else .709 if new else .707,
                            "strehl": .7 if baseline else .72 if new else .71,
                            "phase_rmse": .3 if baseline else .28 if new else .29,
                        })
                        result.append(row)
    return result


def _summarize(rows: list[dict]) -> dict:
    return d9.summarize(rows, _load_yaml(_project_path(d9.CONFIG)), device=torch.device("cpu"))


def test_statistics_require_both_hardware_conditions_and_weather_pairs() -> None:
    rows = _rows()
    result = _summarize(rows)
    assert result["continue_criteria"]["all"] is True
    assert result["independent_weather_count_per_condition"] == 32
    assert result["independent_confirmation"] is False
    for name in d9.CONDITIONS:
        cell = result["cells"][name]
        assert cell["new_minus_old_absolute_power"] == pytest.approx(.002)
        assert cell["new_minus_old_paired_weather_ci97_5"] == pytest.approx([.002, .002])
        assert cell["new_relative_gain"] == pytest.approx(.009 / .7)
    for row in rows:
        if row["hardware_condition"] == "hardware_shift" and row["controller"].startswith(d9.d8.ARM):
            row["power"] = .706
    result = _summarize(rows)
    assert result["continue_criteria"]["nominal_clone"]["all"] is True
    assert result["continue_criteria"]["hardware_shift"]["all"] is False
    assert result["continue_criteria"]["all"] is False


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "profile", "scale", "seed", "nan", "length", "safety_range"])
def test_incomplete_or_misaligned_records_fail_closed(mutation: str) -> None:
    rows = _rows()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = dict(rows[0])
    else:
        field, value = {
            "profile": ("profile", "nominal"), "scale": ("scale", 1.0),
            "seed": ("turbulence_stream_seed", -1), "nan": ("power", float("nan")),
            "length": ("episode_length", 16), "safety_range": ("violation", 1.1),
        }[mutation]
        rows[-1][field] = value
    with pytest.raises(RuntimeError, match="配对|错位"):
        _summarize(rows)


@pytest.mark.parametrize("group", ["member", "family"])
def test_positive_aggregate_cannot_hide_negative_member_or_family(group: str) -> None:
    rows = _rows()
    for row in rows:
        if row["controller"].startswith(d9.d8.ARM):
            negative = (row["controller"] == f"{d9.d8.ARM}_0" if group == "member"
                        else row["family"] == d9.FAMILIES[0])
            row["power"] = .705 if negative else .712
    result = _summarize(rows)
    for name in d9.CONDITIONS:
        criteria = result["continue_criteria"][name]
        assert criteria["new_minus_old_power_positive"] is True
        assert criteria[f"each_{group}_effect_positive"] is False
        assert criteria["all"] is False


def test_cluster_bootstrap_does_not_count_correlated_slots_as_weather() -> None:
    rows = _rows()
    halfway = d9.FORMAL_BASE + 160
    for row in rows:
        if row["controller"].startswith(d9.d8.ARM):
            row["power"] = .737 if row["weather_seed"] < halfway else .681
    result = _summarize(rows)
    for name in d9.CONDITIONS:
        cell = result["cells"][name]
        assert cell["new_minus_old_absolute_power"] > 0
        assert cell["new_minus_old_paired_weather_ci97_5"][0] < 0
        assert cell["continue_criteria"]["all"] is False


@pytest.mark.parametrize("metric,value", [("strehl", .705), ("phase_rmse", .295),
                                          ("violation", .002), ("saturation", .002),
                                          ("slew_limited", .002)])
def test_optical_and_safety_gates_compare_against_old_and_integrator(metric: str, value: float) -> None:
    rows = _rows()
    for row in rows:
        if row["controller"].startswith(d9.d8.ARM):
            row[metric] = value
    result = _summarize(rows)
    assert result["continue_criteria"]["all"] is False


def test_positive_power_below_development_margin_still_fails() -> None:
    rows = _rows()
    for row in rows:
        if row["controller"].startswith(d9.d8.ARM):
            row["power"] = .7072
    result = _summarize(rows)
    for name in d9.CONDITIONS:
        criteria = result["continue_criteria"][name]
        assert criteria["new_minus_old_power_positive"] is True
        assert criteria["new_relative_gain_at_least_1_05_percent"] is False
        assert criteria["all"] is False


def test_changed_training_entry_is_rejected_before_loading_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    original = d9._file_sha256
    monkeypatch.setattr(d9, "_file_sha256", lambda path: "0" * 64 if path == _project_path(d9.d8.CONFIG) else original(path))
    with pytest.raises(RuntimeError, match="冻结配置"):
        d9._verify_training(_load_yaml(_project_path(d9.CONFIG)))
