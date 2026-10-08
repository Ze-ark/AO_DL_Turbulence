"""G2-C1 小型确定性 CPU 人工张量测试，不产生科学实验性能。"""
from copy import deepcopy
import inspect
import json

import pytest
import torch

from scripts import confirm_s4_r5_g2_c1 as c


def config():
    return c.source._load_yaml(c.source._project_path(c.CONFIG))


def fixture_rows(*, quick=False, gain=.0101):
    cfg = config(); spec = cfg["quick" if quick else "data"]; rows = []
    for condition in c.source.CONDITIONS:
        for wi, seed in enumerate(c.stream_manifest(spec)["weather_bases"]):
            for branch in c.frozen.controller_specs(spec):
                for family in c.source.FAMILIES:
                    for slot in range(6):
                        selected = branch["scorer_seed"] is not None
                        row = {**branch, "hardware_condition": condition, "weather_seed": seed,
                            "family": family, "slot": slot,
                            "profile": f"nominal_for_{c.source.PROFILES[slot]}" if condition == "nominal_clone" else c.source.PROFILES[slot],
                            "episode_length": spec["episode_length"], "scorer_fold": wi % 4 if selected else None,
                            "turbulence_stream_seed": seed + 1000 * slot + c.source.FAMILIES.index(family),
                            "sensor_stream_seed": seed + 1000 * slot + 50_000_000,
                            "power_stream_seed": seed + 1000 * slot + 60_000_000,
                            "selector_calls": spec["episode_length"] - spec["selector_start_step"] if selected else 0,
                            "selected_nonoriginal_fraction": .5 if selected else 0.,
                            "decision_seconds_per_batch": .001, "decision_p95_seconds_per_batch": .002}
                        row.update({k: 0. for k in c.frozen.METRICS})
                        row["power"] = 1. + gain if selected else (1.009 if branch["member"] is not None else 1.)
                        row["measured_power"] = row["power"]
                        row["strehl"] = .5; row["phase_rmse"] = .2
                        rows.append(row)
    return rows


def statistics_cfg():
    cfg = config(); cfg["statistics"]["bootstrap_repeats"] = 24
    return cfg


def test_exact_contract_and_budget():
    cfg = config(); c.contract(cfg)
    assert c.frozen.budget(cfg["data"]) == {"episode_batches": 1664, "complete_episodes": 29952,
        "batched_environment_steps": 332800, "physical_transitions": 5990400,
        "policy_forward_calls": 307200, "scorer_forward_calls": 201600, "batch_size": 18}
    assert c.frozen.budget(cfg["quick"]) == {"episode_batches": 6, "complete_episodes": 108,
        "batched_environment_steps": 192, "physical_transitions": 3456,
        "policy_forward_calls": 128, "scorer_forward_calls": 14, "batch_size": 18}
    assert cfg["thresholds"]["minimum_relative_gain"] == .01
    assert not cfg["historical_development_reference"]["gate_for_this_confirmation"]


@pytest.mark.parametrize("key,value", [("runtime", {"device": "cpu"}), ("selector", {"checkpoint": "best"}),
    ("data", {"weather_count": 8}), ("quick", {"technical_only": False}),
    ("thresholds", {"minimum_relative_gain": .0105}), ("statistics", {"interval": .975}),
    ("boundary", {"training_updates": 1}), ("output_directory", "outputs/old")])
def test_contract_prevents_posthoc_tuning(key, value):
    cfg = config(); cfg[key] = value
    with pytest.raises(ValueError, match="合同"):
        c.contract(cfg)


def test_formal_and_technical_streams_disjoint_and_unique():
    cfg = config(); a, b = (c.stream_manifest(cfg[k]) for k in ("data", "quick"))
    c.require_disjoint_streams(a, b)
    assert len(a["weather_bases"]) == 64 and len(a["turbulence"]) == 64 * 18
    assert [list(a["scorer_fold_by_weather"].values()).count(i) for i in range(4)] == [16] * 4
    assert b["weather_bases"] == [7810000]


@pytest.mark.parametrize("key", ["turbulence", "sensor", "power"])
def test_stream_overlap_is_rejected(key):
    cfg = config(); a, b = (c.stream_manifest(cfg[k]) for k in ("data", "quick"))
    b[key][0] = a[key][0]
    with pytest.raises(ValueError, match="重叠"):
        c.require_disjoint_streams(a, b)
    b[key][0] = b[key][1]
    with pytest.raises(ValueError, match="内部重复"):
        c.require_disjoint_streams(a, b)


def splits():
    return [{"fold": i, "train_weather": [10 + i], "held_out_weather": [20 + i]} for i in range(4)]


def test_fold_route_balanced_and_independent_of_outcomes():
    spec = config()["data"]; weather = c.stream_manifest(spec)["weather_bases"]
    assert [c.assigned_fold(w, spec, splits()) for w in weather] == list(range(4)) * 16
    assert list(inspect.signature(c.assigned_fold).parameters) == ["seed", "spec", "splits"]


@pytest.mark.parametrize("which", ["train_weather", "held_out_weather"])
def test_any_fold_seen_weather_is_rejected(which):
    spec = config()["data"]; bad = splits(); bad[3][which].append(spec["seed_base"])
    with pytest.raises(ValueError, match="复用"):
        c.assigned_fold(spec["seed_base"], spec, bad)


def test_unknown_weather_and_duplicate_fold_rejected():
    spec = config()["data"]
    with pytest.raises(ValueError, match="身份"):
        c.assigned_fold(9999, spec, splits())
    bad = splits(); bad[3]["fold"] = 0
    with pytest.raises(ValueError, match="身份"):
        c.assigned_fold(spec["seed_base"], spec, bad)


def test_bootstrap_keeps_all_four_fold_strata_and_is_deterministic():
    cfg = statistics_cfg(); a = c.bootstrap_draws(cfg["data"], cfg["statistics"], torch.device("cpu"))
    b = c.bootstrap_draws(cfg["data"], cfg["statistics"], torch.device("cpu"))
    assert torch.equal(a, b) and a.shape == (24, 64)
    for fold in range(4):
        assert bool((a[:, fold * 16:(fold + 1) * 16] % 4 == fold).all())
    cfg["data"]["weather_count"] = 9
    with pytest.raises(ValueError, match="均衡"):
        c.bootstrap_draws(cfg["data"], cfg["statistics"], torch.device("cpu"))


def test_ratio_interval_pairs_denominator_and_rejects_nonpositive_base():
    numerator = torch.tensor([.01, .04], dtype=torch.float64)
    denominator = torch.tensor([1., 2.], dtype=torch.float64)
    draws = torch.tensor([[0, 0], [1, 1], [0, 1]])
    low, high = c.interval(numerator, draws, denominator)
    assert .01 <= low <= high <= .02
    with pytest.raises(ValueError, match="必须为正"):
        c.interval(numerator, draws, denominator * 0)
    with pytest.raises(ValueError, match="非有限"):
        c.interval(numerator * float("nan"), draws)


def test_between_one_and_one_point_zero_five_passes_original_target():
    result = c.summarize(fixture_rows(), statistics_cfg(), device=torch.device("cpu"))
    assert result["both_conditions_precomputed_pass"]
    assert not result["historical_gate_reclassification"]
    assert not result["historical_multiple_attempts_corrected"]
    assert len(result["group_table"]) == 36
    for cell in result["cells"].values():
        assert cell["current_relative_gain_vs_integrator"] == pytest.approx(.0101)
        assert .01 < cell["current_relative_gain_vs_integrator"] < .0105
        assert cell["weather_clusters"] == 64
        assert cell["current_minus_original_power"] == pytest.approx(.0011)
        assert len(cell["by_frozen_combination_power_advantage"]) == 9
    json.dumps(result, allow_nan=False)


def test_below_one_percent_or_one_failed_condition_cannot_be_averaged_away():
    rows = fixture_rows()
    for row in rows:
        if row["hardware_condition"] == "hardware_shift" and row["scorer_seed"] is not None:
            row["power"] = 1.0099
    result = c.summarize(rows, statistics_cfg(), device=torch.device("cpu"))
    assert result["cells"]["nominal_clone"]["precomputed_all_checks_pass"]
    assert not result["cells"]["hardware_shift"]["precomputed_all_checks_pass"]
    assert not result["both_conditions_precomputed_pass"]


def test_negative_family_cannot_be_hidden_by_positive_total():
    rows = fixture_rows()
    for row in rows:
        if row["scorer_seed"] is not None:
            row["power"] = .999 if row["family"] == "frozen" else 1.025
    result = c.summarize(rows, statistics_cfg(), device=torch.device("cpu"))
    for cell in result["cells"].values():
        assert cell["current_relative_gain_vs_integrator"] > .01
        assert not cell["precomputed_checks_require_audit"]["each_turbulence_family_positive"]
        assert not cell["precomputed_all_checks_pass"]


def test_ci_need_not_have_relative_lower_bound_above_one_percent():
    rows = fixture_rows(); weather = c.stream_manifest(config()["data"])["weather_bases"]
    for row in rows:
        if row["scorer_seed"] is not None:
            row["power"] = 1.006 if weather.index(row["weather_seed"]) < 32 else 1.0146
    result = c.summarize(rows, statistics_cfg(), device=torch.device("cpu"))
    for cell in result["cells"].values():
        assert cell["current_relative_gain_vs_integrator"] > .01
        assert cell["current_relative_gain_ci95"][0] < .01
        assert cell["precomputed_all_checks_pass"]


@pytest.mark.parametrize("metric,value,check", [("strehl", .49, "strehl_non_decrease_vs_integrator"),
    ("phase_rmse", .21, "phase_rmse_non_increase_vs_integrator"),
    ("violation", .0011, "mean_safety_increase_at_most_0_001_vs_integrator"),
    ("saturation", .0011, "mean_safety_increase_at_most_0_001_vs_integrator"),
    ("slew", .0011, "mean_safety_increase_at_most_0_001_vs_integrator")])
def test_quality_and_safety_remain_mandatory(metric, value, check):
    rows = fixture_rows()
    for row in rows:
        if row["scorer_seed"] is not None:
            row[metric] = value
    result = c.summarize(rows, statistics_cfg(), device=torch.device("cpu"))
    assert not result["both_conditions_precomputed_pass"]
    assert all(not cell["precomputed_checks_require_audit"][check] for cell in result["cells"].values())


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "profile", "member", "scorer_fold", "sensor_stream_seed",
    "power_stream_seed", "turbulence_stream_seed", "episode_length", "selector_calls", "nan", "fraction", "latency"])
def test_summary_rejects_misaligned_records(mutation):
    rows = fixture_rows(quick=True)
    if mutation == "missing": rows.pop()
    elif mutation == "duplicate": rows.append(deepcopy(rows[0]))
    elif mutation == "nan": rows[0]["power"] = float("nan")
    elif mutation == "fraction": rows[0]["selected_nonoriginal_fraction"] = 1.1
    elif mutation == "latency": rows[0]["decision_seconds_per_batch"] = float("inf")
    elif mutation == "profile": rows[0][mutation] = "wrong"
    else: rows[0][mutation] = 99
    with pytest.raises(ValueError):
        c.summarize(rows, config(), device=torch.device("cpu"), quick=True)


def test_quick_has_no_scientific_analysis_or_confirmation_flag():
    assert c.summarize(fixture_rows(quick=True), config(), device=torch.device("cpu"), quick=True) == {}


def test_no_cpu_fallback_and_no_assets_loaded_before_device_check(monkeypatch, tmp_path):
    original = c.source._project_path
    monkeypatch.setattr(c.source, "_project_path", lambda p: tmp_path / "missing" if str(p).startswith("outputs/s4_r5_g2_c1") else original(p))
    monkeypatch.setattr(c, "resolve_device", lambda _: (_ for _ in ()).throw(RuntimeError("CUDA unavailable")))
    monkeypatch.setattr(c.frozen, "load_assets", lambda *a, **k: pytest.fail("must not load assets"))
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        c.preflight()


@pytest.mark.parametrize("quick", [False, True])
def test_existing_output_preserved_before_gpu_access(monkeypatch, tmp_path, quick):
    original = c.source._project_path
    monkeypatch.setattr(c.source, "_project_path", lambda p: tmp_path if str(p).startswith("outputs/s4_r5_g2_c1") else original(p))
    monkeypatch.setattr(c, "resolve_device", lambda _: pytest.fail("no GPU access"))
    with pytest.raises(FileExistsError, match="保留"):
        c.preflight(quick=quick)


def test_preflight_generates_no_output_or_forward_calls(monkeypatch, tmp_path):
    original = c.source._project_path
    monkeypatch.setattr(c.source, "_project_path", lambda p: tmp_path / "new" if str(p).startswith("outputs/s4_r5_g2_c1") else original(p))
    monkeypatch.setattr(c, "resolve_device", lambda _: torch.device("cuda"))
    monkeypatch.setattr(c, "verify_streams", lambda cfg: None)
    monkeypatch.setattr(c, "verify_development", lambda: {"completed_episodes": 3744})
    monkeypatch.setattr(c.frozen, "load_assets", lambda *a, **k: ({}, {}, {}, splits(), []))
    monkeypatch.setattr(c.frozen.local, "read_json", lambda p: [])
    report = c.run(preflight_only=True)
    assert report["preflight_model_forward_calls"] == report["preflight_environment_transitions"] == 0
    assert not report["confirmation_access"]
    assert not (tmp_path / "new").exists()


def test_hash_constants_well_formed_and_all_d13_evidence_locked():
    assert set(c.D13_HASHES) == {"summary", "records", "progress", "model_manifest", "trajectory_manifest", "stream_manifest", "config"}
    for digest in (*c.D13_HASHES.values(), c.D13_ENTRY, c.D13_CONFIG):
        assert len(digest) == 64 and int(digest, 16) >= 0


def test_rollout_and_causal_selector_are_reused_without_mutation():
    body = inspect.getsource(c.run)
    assert "frozen.rollout(" in body and "frozen.require_prefix(" in body
    assert "frozen.select_command =" not in body
    assert list(inspect.signature(c.frozen.select_command).parameters) == ["history", "valid", "original", "model", "normalizer"]
