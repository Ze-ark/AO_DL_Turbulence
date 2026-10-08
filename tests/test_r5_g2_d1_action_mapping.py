"""R5-G2-D1 动作映射、配对随机流与冻结边界的确定性测试。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
import json

import pytest
import torch

from scripts import diagnose_s4_r5_g2_d1_action_mapping as g2
from src.rl.r5_batched_environment import R5BatchedEnvironment
from src.rl.s4_training import _load_yaml, _project_path
from src.simulation.config import load_s1_config
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.hardware_effects import HardwareProfile


def test_contract_freezes_mapping_new_weather_and_no_confirmation() -> None:
    cfg = _load_yaml(_project_path(g2.CONFIG))
    g2._contract(cfg)
    for key, changed in (
        ("selected_scale", 2.0),
        ("smooth_epsilon", 1e-4),
        ("mapping_ids", ["smooth", "hard"]),
        ("g1_d3_output_hashes", {}),
        ("data", dict(cfg["data"], seed_base=6_700_000)),
        ("boundary", dict(cfg["boundary"], confirmation_access=True)),
    ):
        altered = deepcopy(cfg)
        altered[key] = changed
        with pytest.raises(ValueError, match="合同"):
            g2._contract(altered)


def test_hard_is_existing_mapping_and_smooth_is_bounded() -> None:
    raw = torch.tensor([[-.9, -.8, -.4, 0., .1, .2, .4, .6, .8, .9, 1.]],
                       dtype=torch.float64)
    hard = g2._map_action(raw, "hard")
    smooth = g2._map_action(raw, "smooth")
    assert torch.equal(hard, raw * g2.SCALE)
    assert bool((hard.abs() > 1).any())
    assert bool((smooth.abs() <= 1).all())
    assert float(smooth[0, 3]) == 0.0
    assert float(smooth[0, 0]) == pytest.approx(-float(smooth[0, 9]))
    assert float(smooth[0, 1]) == pytest.approx(-float(smooth[0, 8]))
    assert float(smooth[0, 2]) == pytest.approx(-float(smooth[0, 6]))
    assert float(smooth[0, 8]) == pytest.approx(float(torch.tanh(
        g2.SCALE * torch.atanh(raw[0, 8]))))
    with pytest.raises(ValueError, match="映射"):
        g2._map_action(raw, "unregistered")


def test_telemetry_tracks_pre_projection_clipping() -> None:
    class FixedPolicy(torch.nn.Module):
        def forward(self, history: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            return history[:, 0, :11]

    raw = torch.tensor([[.8] * 11, [.1] * 11], dtype=torch.float64)
    history = torch.zeros(2, 1, 79, dtype=torch.float64)
    history[:, 0, :11] = raw
    valid = torch.ones(2, 1, dtype=torch.bool)
    hard = g2.ActionMappingTelemetry(FixedPolicy(), "hard")
    smooth = g2.ActionMappingTelemetry(FixedPolicy(), "smooth")
    assert torch.equal(hard(history, valid), raw)
    assert bool(((smooth(history, valid) * g2.SCALE).abs() <= 1).all())
    hard_raw, hard_actual = hard.rates(1, 2)
    smooth_raw, smooth_actual = smooth.rates(1, 2)
    assert hard_raw == [1.0, 0.0]
    assert hard_actual == [1.0, 0.0]
    assert smooth_raw == [1.0, 0.0]
    assert smooth_actual == [0.0, 0.0]


def test_two_six_slot_batches_share_weather_streams() -> None:
    formal, quick = g2.stream_manifest(False), g2.stream_manifest(True)
    assert formal["weather_bases"] == [6_800_000 + 10 * i for i in range(24)]
    assert quick["weather_bases"] == [6_810_000]
    for name in ("turbulence", "sensor", "power"):
        assert set(formal[name]).isdisjoint(quick[name])
    for seed in formal["weather_bases"]:
        for slot in range(6):
            for family in range(3):
                expected = seed + 1000 * slot + family
                nominal_stream = expected
                shift_stream = expected
                assert nominal_stream == shift_stream
                assert nominal_stream in formal["turbulence"]


def test_nominal_clones_are_physically_identical_and_slot_aligned() -> None:
    cfg = _load_yaml(_project_path(g2.CONFIG))
    _, parent = g2._verify_lineage(cfg)
    nominal, shift = g2.d1._profile_pairs(parent)
    fields_to_compare = [item.name for item in fields(HardwareProfile)
                         if item.name not in {"identifier", "label"}]
    assert [item.identifier for item in shift] == list(g2.PROFILES)
    assert [item.identifier for item in nominal] == [f"nominal_for_{name}" for name in g2.PROFILES]
    assert all(all(getattr(clone, key) == getattr(nominal[0], key)
                   for key in fields_to_compare) for clone in nominal)


def test_actual_environment_seed_schedule_is_equal_per_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _load_yaml(_project_path(g2.CONFIG))
    _, parent = g2._verify_lineage(cfg)
    nominal, shift = g2.d1._profile_pairs(parent)
    base, _ = load_s1_config(_project_path(parent["environment_config"]))
    base = replace(base, num_modes=21, batch_size=1, episode_length=16)

    def fake_init(self, config, device, **kwargs):
        self.config, self.device = config, torch.device(device)

    monkeypatch.setattr(AdaptiveOpticsEnv, "__init__", fake_init)
    monkeypatch.setattr(AdaptiveOpticsEnv, "reset", lambda self, seed=None: (None, {}))
    monkeypatch.setattr(AdaptiveOpticsEnv, "_new_phase_screens", lambda self: torch.empty(0))

    seeds = []
    for profiles in (nominal, shift):
        env = R5BatchedEnvironment(base, torch.device("cpu"), torch.empty(0),
                                   parent["families"], profiles,
                                   parent["data"]["sensor_seed_offset"])
        env.reset(seed=g2.QUICK_BASE)
        env._new_phase_screens()
        seeds.append((tuple(g.initial_seed() for g in env.generators),
                      tuple(g.initial_seed() for g in env.sensor_generators),
                      tuple(g.initial_seed() for g in env.power_generators)))
    assert seeds[0] == seeds[1]
    turbulence, sensor, power = seeds[0]
    assert turbulence == tuple(g2.QUICK_BASE + slot * 1000 + family
                               for slot in range(6) for family in range(3))
    assert sensor == tuple(g2.QUICK_BASE + slot * 1000 + 50_000_000 for slot in range(6))
    assert power == tuple(g2.QUICK_BASE + slot * 1000 + 60_000_000 for slot in range(6))


def _synthetic_rows() -> list[dict]:
    rows = []
    seed = g2.QUICK_BASE
    controllers = ("integrator",) + tuple(f"policy_{member}_{mapping}"
                                         for mapping in g2.MAPPINGS for member in range(3))
    for condition in g2.CONDITIONS:
        for controller in controllers:
            mapping = "integrator" if controller == "integrator" else controller.split("_")[-1]
            for family_index, family in enumerate(g2.FAMILIES):
                for slot, profile in enumerate(g2.PROFILES):
                    row = {metric: 0.0 for metric in g2.METRICS}
                    row.update({
                        "hardware_condition": condition, "controller": controller,
                        "mapping": mapping, "family": family, "slot": slot,
                        "profile": f"nominal_for_{profile}" if condition == "nominal_clone" else profile,
                        "weather_seed": seed, "turbulence_stream_seed": seed + slot * 1000 + family_index,
                        "scale": 0.0 if mapping == "integrator" else g2.SCALE,
                        "power": .5 if mapping == "integrator" else (.51 if mapping == "hard" else .52),
                        "raw_hard_clipped_fraction": .0 if mapping == "integrator" else .2,
                        "normalized_correction_clipped_fraction": .2 if mapping == "hard" else .0,
                    })
                    rows.append(row)
    return rows


def test_synthetic_summary_uses_paired_complete_weather_and_rejects_mismatch() -> None:
    cfg = _load_yaml(_project_path(g2.CONFIG))
    rows = _synthetic_rows()
    result = g2.summarize(rows, cfg, quick=True, device=torch.device("cpu"))
    assert result["independent_confirmation"] is False
    assert result["overall_smooth_minus_hard_absolute_power"] == pytest.approx(.01)
    for condition in g2.CONDITIONS:
        cell = result["cells"][condition]
        assert cell["smooth_minus_hard_absolute_power"] == pytest.approx(.01)
        assert cell["hard_relative_gain"] == pytest.approx(.02)
        assert cell["smooth_relative_gain"] == pytest.approx(.04)
        assert cell["development_continue_criteria"]["all"] is True
        assert len(cell["slot_absolute_powers_and_gains"]) == 6
        for slot in cell["slot_absolute_powers_and_gains"]:
            assert slot["integrator_power"] == pytest.approx(.5)
            assert slot["hard_policy_power"] == pytest.approx(.51)
            assert slot["smooth_policy_power"] == pytest.approx(.52)
            assert slot["hard_policy_minus_integrator_power"] == pytest.approx(.01)
            assert slot["smooth_policy_minus_integrator_power"] == pytest.approx(.02)
            assert slot["hard_relative_gain"] == pytest.approx(.02)
            assert slot["smooth_relative_gain"] == pytest.approx(.04)
    assert result["development_continue_criteria"]["all"] is True
    with pytest.raises(RuntimeError, match="缺失"):
        g2.summarize(rows[:-1], cfg, quick=True, device=torch.device("cpu"))
    altered = deepcopy(rows)
    altered[-1]["turbulence_stream_seed"] += 1
    with pytest.raises(RuntimeError, match="错位"):
        g2.summarize(altered, cfg, quick=True, device=torch.device("cpu"))


def test_development_continue_criteria_fail_without_two_condition_improvement() -> None:
    cfg = _load_yaml(_project_path(g2.CONFIG))
    rows = _synthetic_rows()
    for row in rows:
        if row["hardware_condition"] == "hardware_shift" and row["mapping"] == "smooth":
            row["power"] = .509
            row["strehl"] = -.01
            row["phase_rmse"] = .01
            row["violation"] = .002
    result = g2.summarize(rows, cfg, quick=True, device=torch.device("cpu"))
    assert result["cells"]["nominal_clone"]["development_continue_criteria"]["all"] is True
    failed = result["cells"]["hardware_shift"]["development_continue_criteria"]
    assert failed["smooth_minus_hard_mean_positive"] is False
    assert failed["paired_weather_ci95_lower_positive"] is False
    assert failed["strehl_non_decrease"] is False
    assert failed["phase_rmse_non_increase"] is False
    assert failed["violation_increase_at_most_0_001"] is False
    assert failed["all"] is False
    assert result["development_continue_criteria"]["all"] is False


def test_post_creation_write_failure_preserves_original_error_and_partial_evidence(
        tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _load_yaml(_project_path(g2.CONFIG))
    output = tmp_path / "new-output"
    report = {"status": "READY_FOR_DIAGNOSTIC", "config_sha256": "synthetic",
              "entry_sha256": "synthetic", "physical_transitions": 0}
    monkeypatch.setattr(g2, "preflight", lambda *args, **kwargs:
                        (cfg, {}, {}, report, output, torch.device("cpu")))
    original_write = g2.write_json

    def fail_first(path, payload):
        if path.name == "preflight.json":
            raise OSError("injected preflight write failure")
        return original_write(path, payload)

    monkeypatch.setattr(g2, "write_json", fail_first)
    with pytest.raises(OSError, match="injected preflight write failure"):
        g2.run(quick=True)
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert "injected preflight write failure" in failure["traceback"]
    assert failure["automatic_retry"] is False


def test_mkdir_race_does_not_write_into_existing_output(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _load_yaml(_project_path(g2.CONFIG))
    output = tmp_path / "existing-output"
    output.mkdir()
    marker = output / "preserve.txt"
    marker.write_text("user data", encoding="utf-8")
    monkeypatch.setattr(g2, "preflight", lambda *args, **kwargs:
                        (cfg, {}, {}, {}, output, torch.device("cpu")))
    with pytest.raises(FileExistsError):
        g2.run(quick=True)
    assert marker.read_text(encoding="utf-8") == "user data"
    assert not (output / "failure.json").exists()


def test_preflight_counts_fail_closed_and_refuses_existing_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(g2, "resolve_device", lambda requested: torch.device("cpu"))
    for quick, count in ((False, 1_209_600), (True, 4_032)):
        output = _project_path("outputs/s4_r5_g2_d1_action_mapping_v1_quick_r2" if quick
                               else "outputs/s4_r5_g2_d1_action_mapping_v1")
        if output.exists():
            with pytest.raises(FileExistsError, match="保留已有"):
                g2.preflight(quick=quick)
            continue
        _, _, _, report, _, device = g2.preflight(quick=quick)
        assert report["physical_transitions"] == count
        assert report["shared_streams_between_conditions_and_controllers"] is True
        assert report["confirmation_access"] is False
        assert max(report["maximum_displacement_pixels"]) < report["turbulence_grid_pixels"]
        assert device.type == "cpu"  # 仅测试替身；真实入口会请求 CUDA。
