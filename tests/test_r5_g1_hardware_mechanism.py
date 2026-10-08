"""G1-D1 只用合成记录测试新天气、名义槽位配对及开发汇总。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields

import pytest
import torch

from scripts import diagnose_s4_r5_g1_hardware_mechanism as diagnostic
from scripts import run_s4_r5_g1_cross_condition as g1
from src.rl.s4_training import _load_yaml, _project_path
from src.simulation.hardware_effects import HardwareProfile


def test_contract_refuses_reusing_g1_or_changing_frozen_scale() -> None:
    cfg = _load_yaml(_project_path(diagnostic.CONFIG))
    diagnostic._contract(cfg)
    for key, changed in (
        ("selected_scale", 2.0),
        ("data", dict(cfg["data"], seed_base=6_500_000)),
        ("g1_entry_sha256", "0" * 64),
        ("g1_output_hashes", {}),
    ):
        altered = deepcopy(cfg)
        altered[key] = changed
        with pytest.raises(ValueError, match="合同"):
            diagnostic._contract(altered)


def test_streams_are_fresh_and_shared_between_conditions() -> None:
    formal, quick = diagnostic.stream_manifest(False), diagnostic.stream_manifest(True)
    assert len(formal["weather_bases"]) == 24
    assert len(formal["turbulence"]) == len(set(formal["turbulence"])) == 432
    assert len(formal["sensor"]) == len(set(formal["sensor"])) == 144
    assert len(formal["power"]) == len(set(formal["power"])) == 144
    assert len(quick["turbulence"]) == 18
    assert set(formal["turbulence"]).isdisjoint(quick["turbulence"])
    for arm in ("wind", "hardware"):
        for old_quick in (False, True):
            assert set(formal["turbulence"]).isdisjoint(g1.stream_manifest(arm, old_quick)["turbulence"])
            assert set(quick["turbulence"]).isdisjoint(g1.stream_manifest(arm, old_quick)["turbulence"])


def test_six_nominal_clones_have_unique_ids_but_identical_physics() -> None:
    g1_cfg = _load_yaml(_project_path(g1.CONFIG))
    r3_cfg = _load_yaml(_project_path(g1_cfg["r3_config"]))
    parent = _load_yaml(_project_path(r3_cfg["parent"]))
    clones, shifts = diagnostic._profile_pairs(parent)
    assert [item.identifier for item in shifts] == list(diagnostic.PROFILES)
    assert len({item.identifier for item in clones + shifts}) == 12
    physical_fields = [item.name for item in fields(HardwareProfile)
                       if item.name not in {"identifier", "label"}]
    for clone in clones[1:]:
        assert all(getattr(clone, name) == getattr(clones[0], name) for name in physical_fields)
    for clone, shift in zip(clones, shifts, strict=True):
        assert clone.identifier == f"nominal_for_{shift.identifier}"
        assert any(getattr(clone, name) != getattr(shift, name) for name in physical_fields)


def test_preflight_checks_lineage_and_preserves_existing_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostic, "resolve_device", lambda _: torch.device("cpu"))
    for quick in (False, True):
        output = _project_path("outputs/s4_r5_g1_hardware_mechanism_v1_quick"
                               if quick else "outputs/s4_r5_g1_hardware_mechanism_v1")
        if output.exists():
            with pytest.raises(FileExistsError, match="保留已有"):
                diagnostic.preflight(quick=quick)
            continue
        _, _, _, report, _, device = diagnostic.preflight(quick=quick)
        assert report["status"] == ("READY_FOR_DIAGNOSTIC" if quick else "READY_FOR_USER_IDE")
        assert report["physical_transitions"] == (2_304 if quick else 691_200)
        assert report["shared_streams_between_conditions"]
        assert report["confirmation_access"] is False
        assert max(report["maximum_displacement_pixels"]) < report["turbulence_grid_pixels"]
        assert device.type == "cpu"  # 仅单元测试替代；正式入口仍强制 CUDA。


def _synthetic_rows() -> list[dict]:
    rows = []
    base = diagnostic.QUICK_BASE
    controllers = ("integrator",) + tuple(f"policy_{index}_scale_{diagnostic.SCALE}"
                                          for index in range(3))
    metrics = {name: .001 for name in diagnostic.METRICS if name != "power"}
    metrics["phase_rmse"] = .8
    metrics["strehl"] = .4
    metrics["policy_forward_seconds_per_step"] = .003
    metrics["policy_forward_p95_seconds"] = .004
    for condition in diagnostic.CONDITIONS:
        for controller in controllers:
            policy = controller != "integrator"
            for family_index, family in enumerate(diagnostic.FAMILIES):
                for slot, target in enumerate(diagnostic.PROFILES):
                    shift = condition == "hardware_shift"
                    # 名义条件的策略增量 .01；误差条件的策略增量 .015。
                    power = (.49 if shift else .50) + ((.015 if shift else .01) if policy else 0)
                    row = {"condition": condition, "controller": controller,
                           "family": family, "slot": slot, "target_profile": target,
                           "profile": target if shift else f"nominal_for_{target}",
                           "weather_seed": base,
                           "turbulence_stream_seed": base + slot * 1_000 + family_index,
                           "scale": diagnostic.SCALE if policy else 0.0,
                           "power": power, **metrics}
                    row["normalized_correction_clipped_fraction"] = .2 if policy else 0.0
                    rows.append(row)
    return rows


def test_summary_paired_counterfactual_and_no_confirmation_gate() -> None:
    cfg = _load_yaml(_project_path(diagnostic.CONFIG))
    result = diagnostic.summarize(_synthetic_rows(), cfg, quick=True, device=torch.device("cpu"))
    assert result["independent_confirmation"] is False
    assert result["overall_policy_increment_shift_minus_nominal"] == pytest.approx(.005)
    for entry in result["profile_results"].values():
        assert entry["policy_increment_shift_minus_nominal"] == pytest.approx(.005)
        assert entry["policy_power_shift_minus_nominal"] == pytest.approx(-.005)
        assert entry["integrator_power_shift_minus_nominal"] == pytest.approx(-.01)
        assert entry["exploratory_paired_ci95"] == pytest.approx([.005, .005])
        assert entry["other_metrics"]["normalized_correction_clipped_fraction"]["shift_policy"] == pytest.approx(.2)


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
