"""G1-D2 只用合成记录测试四格合同、完整天气配对和效应公式。"""
from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from scripts import diagnose_s4_r5_g1_d2_factorial as diagnostic
from scripts import diagnose_s4_r5_g1_hardware_mechanism as d1
from scripts import run_s4_r5_g1_cross_condition as g1
from src.rl.s4_training import _load_yaml, _project_path


def test_contract_freezes_factorial_and_development_boundary() -> None:
    cfg = _load_yaml(_project_path(diagnostic.CONFIG))
    diagnostic._contract(cfg)
    for key, changed in (
        ("selected_scale", 2.0),
        ("wind_speed_multiplier", 1.20),
        ("data", dict(cfg["data"], seed_base=6_600_000)),
        ("g1_d1_output_hashes", {}),
        ("boundary", dict(cfg["boundary"], confirmation_access=True)),
    ):
        altered = deepcopy(cfg)
        altered[key] = changed
        with pytest.raises(ValueError, match="合同"):
            diagnostic._contract(altered)


def test_all_three_streams_are_fresh_and_shared_across_cells() -> None:
    formal, quick = diagnostic.stream_manifest(False), diagnostic.stream_manifest(True)
    assert len(formal["weather_bases"]) == 24
    assert len(formal["turbulence"]) == len(set(formal["turbulence"])) == 432
    assert len(formal["sensor"]) == len(set(formal["sensor"])) == 144
    assert len(formal["power"]) == len(set(formal["power"])) == 144
    historical = [g1.r3_stream_manifest(False), g1.r3_stream_manifest(True)]
    historical.extend(g1.stream_manifest(arm, old_quick)
                      for arm in ("wind", "hardware") for old_quick in (False, True))
    historical.extend(d1.stream_manifest(old_quick) for old_quick in (False, True))
    for name in ("turbulence", "sensor", "power"):
        assert set(formal[name]).isdisjoint(quick[name])
        assert all(set(formal[name]).isdisjoint(item[name]) for item in historical)
        assert all(set(quick[name]).isdisjoint(item[name]) for item in historical)


def test_conditions_preserve_family_and_profile_slot_order() -> None:
    g1_cfg = _load_yaml(_project_path(g1.CONFIG))
    r3_cfg = _load_yaml(_project_path(g1_cfg["r3_config"]))
    parent = _load_yaml(_project_path(r3_cfg["parent"]))
    original, old_profiles = diagnostic._conditions(parent, "original", "original", 1.10)
    fast, new_profiles = diagnostic._conditions(parent, "faster_110", "shift", 1.10)
    assert [family["id"] for family in original] == [family["id"] for family in fast] == list(diagnostic.FAMILIES)
    assert [item.identifier for item in old_profiles] == list(diagnostic.ORIGINAL_PROFILES)
    assert [item.identifier for item in new_profiles] == list(diagnostic.SHIFT_PROFILES)
    for before, after in zip(original, fast, strict=True):
        assert after["wind_speed_mps"] == pytest.approx(before["wind_speed_mps"] * 1.10)


def test_preflight_checks_lineage_cuda_route_and_preserves_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostic, "resolve_device", lambda _: torch.device("cpu"))
    for quick in (False, True):
        output = _project_path("outputs/s4_r5_g1_d2_factorial_v1_quick_r1" if quick
                               else "outputs/s4_r5_g1_d2_factorial_v1")
        if output.exists():
            with pytest.raises(FileExistsError, match="保留已有"):
                diagnostic.preflight(quick=quick)
            continue
        _, _, _, report, _, device = diagnostic.preflight(quick=quick)
        assert report["status"] == ("READY_FOR_DIAGNOSTIC" if quick else "READY_FOR_USER_IDE")
        assert report["physical_transitions"] == (4_608 if quick else 1_382_400)
        assert report["shared_streams_between_all_four_cells"] is True
        assert report["confirmation_access"] is False
        assert max(report["maximum_displacement_pixels"]) < report["turbulence_grid_pixels"]
        assert device.type == "cpu"  # 仅单元测试替代；入口仍通过 resolve_device 强制 CUDA。


def _synthetic_rows() -> list[dict]:
    rows = []
    base = diagnostic.QUICK_BASE
    controllers = ("integrator",) + tuple(f"policy_{i}_scale_{diagnostic.SCALE}" for i in range(3))
    gains = {("original", "original"): .010,
             ("original", "shift"): .012,
             ("faster_110", "original"): .015,
             ("faster_110", "shift"): .021}
    metrics = {name: .001 for name in diagnostic.METRICS if name != "power"}
    metrics.update({"phase_rmse": .8, "strehl": .4,
                    "policy_forward_seconds_per_step": .003,
                    "policy_forward_p95_seconds": .004})
    for wind in diagnostic.WINDS:
        for hardware in diagnostic.HARDWARE:
            profiles = diagnostic.ORIGINAL_PROFILES if hardware == "original" else diagnostic.SHIFT_PROFILES
            for controller in controllers:
                policy = controller != "integrator"
                for family_index, family in enumerate(diagnostic.FAMILIES):
                    for slot, profile in enumerate(profiles):
                        row = {"wind_condition": wind, "hardware_condition": hardware,
                               "controller": controller, "family": family,
                               "slot": slot, "profile": profile, "weather_seed": base,
                               "turbulence_stream_seed": base + 1_000 * slot + family_index,
                               "scale": diagnostic.SCALE if policy else 0.0,
                               "power": .50 + (gains[(wind, hardware)] if policy else 0.0),
                               **metrics}
                        row["normalized_correction_clipped_fraction"] = .2 if policy else 0.0
                        rows.append(row)
    return rows


def test_factorial_main_effects_and_equivalent_interaction_formulas() -> None:
    cfg = _load_yaml(_project_path(diagnostic.CONFIG))
    result = diagnostic.summarize(_synthetic_rows(), cfg, quick=True, device=torch.device("cpu"))
    assert result["independent_confirmation"] is False
    cells = result["cells"]
    g00 = cells["original__original"]["policy_minus_integrator_power"]
    g01 = cells["original__shift"]["policy_minus_integrator_power"]
    g10 = cells["faster_110__original"]["policy_minus_integrator_power"]
    g11 = cells["faster_110__shift"]["policy_minus_integrator_power"]
    effects = result["factorial_effects"]
    assert effects["wind_main"]["absolute_policy_increment_effect"] == pytest.approx(.007)
    assert effects["hardware_main"]["absolute_policy_increment_effect"] == pytest.approx(.004)
    assert effects["interaction"]["absolute_policy_increment_effect"] == pytest.approx(.004)
    assert effects["interaction"]["absolute_policy_increment_effect"] == pytest.approx((g11 - g01) - (g10 - g00))
    assert effects["interaction"]["absolute_policy_increment_effect"] == pytest.approx((g11 - g10) - (g01 - g00))
    assert cells["original__original"]["relative_power_gain"] == pytest.approx(.02)
    assert cells["faster_110__shift"]["relative_power_gain"] == pytest.approx(.042)
    assert cells["original__original"]["other_deltas"]["strehl"] == pytest.approx(0.0)
    assert cells["original__original"]["other_deltas"]["phase_rmse"] == pytest.approx(0.0)
    assert cells["original__original"]["raw_metrics"]["strehl"] == {"integrator": pytest.approx(.4),
                                                                 "policy": pytest.approx(.4)}
    assert result["paired_unit"].endswith("bootstrap_complete_weather")


def test_bootstrap_resamples_one_weather_index_for_all_families() -> None:
    effect = torch.tensor([[0., 2.], [10., 12.], [20., 22.]], dtype=torch.float64)
    draws = torch.tensor([[0, 0], [1, 1], [0, 1]], dtype=torch.long)
    quantiles = torch.tensor([0., 1.], dtype=torch.float64)
    assert diagnostic._paired_ci(effect, draws, quantiles) == pytest.approx([10., 12.])


def test_summary_rejects_missing_duplicate_and_misaligned_pairs() -> None:
    cfg = _load_yaml(_project_path(diagnostic.CONFIG))
    rows = _synthetic_rows()
    with pytest.raises(RuntimeError, match="缺失、重复"):
        diagnostic.summarize(rows[:-1], cfg, quick=True, device=torch.device("cpu"))
    with pytest.raises(RuntimeError, match="缺失、重复"):
        diagnostic.summarize(rows + [rows[0]], cfg, quick=True, device=torch.device("cpu"))
    wrong = deepcopy(rows)
    wrong[-1]["turbulence_stream_seed"] += 1
    with pytest.raises(RuntimeError, match="错位"):
        diagnostic.summarize(wrong, cfg, quick=True, device=torch.device("cpu"))
