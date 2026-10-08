"""G2-D13 小型确定性 CPU 人工张量单元测试；不生成科学实验结果。"""
from copy import deepcopy
import inspect
import json

import pytest
import torch

from scripts import evaluate_s4_r5_g2_d13_closed_loop as d
from src.rl.r4_control import R4Limits, project_request


def config():
    return d.source._load_yaml(d.source._project_path(d.CONFIG))


def inputs(n=2):
    history = torch.zeros(n, 8, 79)
    history[..., 75] = torch.arange(8)
    history[..., 76] = torch.arange(8) - 1
    history[..., 77] = torch.arange(8) - 1
    history[..., 78] = 1
    valid = torch.ones(n, 8, dtype=torch.bool)
    command = torch.ones(n, 11) * 1.4
    norm = {"mean": torch.zeros(76), "std": torch.ones(76),
            "command_mean": torch.zeros(11), "command_std": torch.ones(11)}
    return history, valid, command, norm


class Prediction:
    def __init__(self, index=0, score=-1.):
        self.index, self.score = index, score

    def __call__(self, x, valid, command):
        assert x.shape[-1] == 76 and command.shape[-1] == 11
        result = x.new_full((len(x), 23), -1.)
        result[:, self.index] = self.score
        return result


def rows_fixture(quick=False):
    cfg = config(); spec = cfg["quick" if quick else "data"]; rows = []
    weather = d.source.stream_manifest(quick)["weather_bases"]
    for condition in d.source.CONDITIONS:
        for wi, seed in enumerate(weather):
            for branch in d.controller_specs(spec):
                for family in d.source.FAMILIES:
                    for slot in range(6):
                        row = {**branch, "hardware_condition": condition, "weather_seed": seed, "family": family,
                            "slot": slot, "profile": f"nominal_for_{d.source.PROFILES[slot]}" if condition == "nominal_clone" else d.source.PROFILES[slot],
                            "episode_length": spec["episode_length"], "turbulence_stream_seed": seed + 1000 * slot + d.source.FAMILIES.index(family),
                            "sensor_stream_seed": seed + 1000 * slot + 50_000_000, "power_stream_seed": seed + 1000 * slot + 60_000_000,
                            "scorer_fold": None if branch["scorer_seed"] is None else (-1 if quick else wi // 2),
                            "selector_calls": 0 if branch["scorer_seed"] is None else spec["episode_length"] - spec["selector_start_step"],
                            "selected_nonoriginal_fraction": 0.5 if branch["scorer_seed"] is not None else 0.}
                        row.update({key: 0. for key in d.METRICS})
                        row["power"] = 1.02 if branch["scorer_seed"] is not None else 1. if branch["member"] is None else 1.01
                        row["measured_power"] = row["power"] - .001
                        row["strehl"] = .5; row["phase_rmse"] = .1
                        rows.append(row)
    return rows


def test_contract_and_exact_budgets():
    cfg = config(); d.contract(cfg)
    assert d.budget(cfg["data"]) == {"episode_batches": 208, "complete_episodes": 3744,
        "batched_environment_steps": 41600, "physical_transitions": 748800,
        "policy_forward_calls": 38400, "scorer_forward_calls": 25200, "batch_size": 18}
    assert d.budget(cfg["quick"])["physical_transitions"] == 1728
    assert len(d.controller_specs(cfg["data"])) == 13
    assert cfg["boundary"]["training_updates"] == 0


@pytest.mark.parametrize("key,value", [("runtime", {"device": "cpu"}), ("purpose", "confirmation"),
    ("boundary", {"real_slm_actions": True}), ("selector", {"kind": "history"}),
    ("data", {"episode_length": 12}), ("quick", {"technical_only": False}),
    ("reference_thresholds", {"minimum_relative_gain": .01}), ("statistics", {"best_weather": True})])
def test_contract_cannot_tune_or_expand_scope(key, value):
    cfg = config(); cfg[key] = value
    with pytest.raises(ValueError, match="合同"):
        d.contract(cfg)


def test_hash_constants_are_valid_hex():
    for digest in (*d.LOCAL_HASHES.values(), d.LOCAL_ENTRY_SHA256, d.LOCAL_CONFIG_SHA256):
        assert len(digest) == 64 and int(digest, 16) >= 0


def test_route_requires_unique_unseen_weather():
    splits = [{"fold": 0, "train_weather": [20], "held_out_weather": [10]},
              {"fold": 1, "train_weather": [10], "held_out_weather": [20]}]
    assert d.route_fold(10, splits, quick=False)["fold"] == 0
    splits[0]["train_weather"].append(10)
    with pytest.raises(ValueError, match="唯一未见"):
        d.route_fold(10, splits, quick=False)


@pytest.mark.parametrize("splits", [[], [{"train_weather": [], "held_out_weather": [10]}] * 2])
def test_route_rejects_unknown_or_duplicate(splits):
    with pytest.raises(ValueError):
        d.route_fold(10, splits, quick=False)


def test_quick_is_technical_seen_weather():
    split = {"fold": -1, "train_weather": [10], "held_out_weather": []}
    assert d.route_fold(10, [split], quick=True) == split


@pytest.mark.parametrize("score", [-1., 0.])
def test_negative_or_tied_prediction_keeps_raw_original(score):
    history, valid, command, norm = inputs()
    selected, choice, _ = d.select_command(history, valid, command, Prediction(score=score), norm)
    assert torch.equal(selected, command) and not bool(choice.any())


def test_candidate_units_not_scaled_twice_and_projection_unchanged():
    history, valid, command, norm = inputs()
    selected, choice, _ = d.select_command(history, valid, command, Prediction(index=1, score=2.), norm)
    assert choice.tolist() == [3, 3]  # coordinate_00_minus：23输出的下标1对应25候选的下标3。
    assert torch.allclose(selected[:, 0], torch.tensor([.9, .9]))
    assert torch.equal(selected[:, 1:], torch.ones(2, 10))
    action = project_request(torch.zeros(2, 21), torch.zeros(2, 21), selected, R4Limits())
    assert torch.allclose(action.requested_delta_rad[:, 10], torch.tensor([.01125, .01125]))
    assert action.requested_delta_rad.abs().max() <= .15


def test_selection_has_only_causal_signature_and_no_mutation():
    assert list(inspect.signature(d.select_command).parameters) == ["history", "valid", "original", "model", "normalizer"]
    history, valid, command, norm = inputs(); saved = [x.clone() for x in (history, valid, command)]
    d.select_command(history, valid, command, Prediction(score=1.), norm)
    assert all(torch.equal(a, b) for a, b in zip(saved, (history, valid, command)))


def test_current_scorer_drops_absolute_clocks():
    history, valid, command, norm = inputs()
    torch.manual_seed(0); model = d.training.CandidateScorer("current").eval()
    a = d.select_command(history, valid, command, model, norm)
    history[..., 75:78] += 900
    b = d.select_command(history, valid, command, model, norm)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_feature_whitelist_rejects_info_dictionary():
    history, valid, command, _ = inputs()
    with pytest.raises(ValueError, match="只允许"):
        d.training.features({"history": history, "valid": valid, "command": command, "info": {"truth": 1}})


def test_prefix_exactness_and_divergence_after_activation():
    original = {"metrics_and_actions": {k: torch.zeros(16, 18) for k in (*d.METRICS,
        "history_frame", "history_valid", "original_command", "selected_command", "requested_delta", "requested_modal", "applied_modal")}}
    candidate = deepcopy(original); candidate["metrics_and_actions"]["power"][8:] = 1
    d.require_prefix(original, candidate, 8)
    candidate["metrics_and_actions"]["requested_modal"][7] = 1
    with pytest.raises(RuntimeError, match="启用前"):
        d.require_prefix(original, candidate, 8)


def test_trace_means_require_complete_finite_episodes():
    trace = {k: torch.arange(16.).expand(18, -1).T for k in d.METRICS}
    assert d.trace_means(trace, 16)["power"].tolist() == [7.5] * 18
    trace["power"] = trace["power"][:15]
    with pytest.raises(ValueError, match="长度"):
        d.trace_means(trace, 16)


def test_trace_means_reject_nan():
    trace = {k: torch.zeros(16, 18) for k in d.METRICS}; trace["strehl"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="非有限"):
        d.trace_means(trace, 16)


def test_ratio_bootstrap_resamples_denominator_together():
    numerator = torch.tensor([.01, .04], dtype=torch.float64)
    denominator = torch.tensor([1., 2.], dtype=torch.float64)
    draws = torch.tensor([[0, 0], [1, 1], [0, 1]])
    bounds = d.bootstrap_interval(numerator, draws, denominator)
    assert .01 <= bounds[0] <= bounds[1] <= .02
    with pytest.raises(ValueError, match="必须为正"):
        d.bootstrap_interval(numerator, draws, denominator * 0)


def test_complete_pairing_statistics_without_gate_promotion():
    cfg = config(); result = d.summarize(rows_fixture(), cfg, device=torch.device("cpu"))
    assert len(result["group_table"]) == 36
    for cell in result["cells"].values():
        assert cell["current_minus_original_power"] == pytest.approx(.01)
        assert cell["current_relative_gain_vs_integrator"] == pytest.approx(.02)
        assert cell["original_relative_gain_vs_integrator"] == pytest.approx(.01)
        assert cell["positive_weather_count"] == 8
        assert len(cell["by_scorer_seed_current_minus_original"]) == 3
    assert not result["prior_d9_gate_reclassified"] and not result["independent_confirmation"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "profile", "member", "scorer_fold", "sensor_stream_seed",
    "power_stream_seed", "turbulence_stream_seed", "episode_length", "selector_calls", "nan", "fraction"])
def test_summary_rejects_incomplete_or_misaligned_records(mutation):
    rows = rows_fixture(quick=True)
    if mutation == "missing": rows.pop()
    elif mutation == "duplicate": rows.append(deepcopy(rows[0]))
    elif mutation == "nan": rows[0]["power"] = float("nan")
    elif mutation == "fraction": rows[0]["selected_nonoriginal_fraction"] = 1.1
    elif mutation == "profile": rows[0][mutation] = "wrong"
    else: rows[0][mutation] = 99
    with pytest.raises(ValueError):
        d.summarize(rows, config(), device=torch.device("cpu"), quick=True)


def test_quick_omits_scientific_analysis():
    assert d.summarize(rows_fixture(quick=True), config(), device=torch.device("cpu"), quick=True) == {}


def test_cuda_only_preflight_fails_before_loading_assets(monkeypatch, tmp_path):
    original = d.source._project_path
    monkeypatch.setattr(d.source, "_project_path", lambda p: tmp_path / "missing" if str(p).startswith("outputs/s4_r5_g2_d13") else original(p))
    monkeypatch.setattr(d, "resolve_device", lambda name: (_ for _ in ()).throw(RuntimeError("CUDA unavailable")))
    monkeypatch.setattr(d, "load_assets", lambda *a, **kw: pytest.fail("must not load assets on CPU"))
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        d.preflight()


def test_existing_output_is_preserved_before_gpu_access(monkeypatch, tmp_path):
    original = d.source._project_path
    monkeypatch.setattr(d.source, "_project_path", lambda p: tmp_path if str(p).startswith("outputs/s4_r5_g2_d13") else original(p))
    monkeypatch.setattr(d, "resolve_device", lambda _: pytest.fail("no GPU access for existing output"))
    with pytest.raises(FileExistsError, match="保留"):
        d.preflight()
