"""G2-D12-C 小型确定性 CPU 单元测试；全用人工张量，不形成科学结果。"""
from copy import deepcopy
import inspect
import json

import pytest
import torch

from scripts import evaluate_s4_r5_g2_d12_held_out as c


def config():
    return c.training.pairing.source._load_yaml(c.training.pairing.source._project_path(c.CONFIG))


def split_fixture():
    return {"fold": 0, "train_indices": [2, 3], "held_out_indices": [0, 1],
            "train_weather": [20], "held_out_weather": [10], "technical_only": False}


def checkpoint_fixture():
    return {"kind": "history", "fold": 0, "seed": 7564000, "update": 1000,
        "held_out_evaluation": False, "train_weather": [20], "held_out_weather": [10],
        "target_scale": 10000., "feature_columns": list(c.training.FEATURE_COLUMNS),
        "candidate_order": list(c.training.pairing.source.CANDIDATES),
        "dataset_hashes": c.training.DATA_HASHES[False], "training_only_constant_candidate": 3,
        "normalizer": {"mean": torch.zeros(76), "std": torch.ones(76),
                       "command_mean": torch.zeros(11), "command_std": torch.ones(11)}}


def target_fixture(n=4):
    power = torch.zeros(n, 25, dtype=torch.float64)
    power[:, 2] = -.002; power[:, 3] = .003
    safety = torch.zeros(n, 25, 3, dtype=torch.float64); safety[:, 2, 0] = .002
    return {"power_delta": power, "measured_power_delta": power / 2,
        "safety_maxima": safety, "safety_pass": (safety <= safety[:, :1] + .001).all(-1)}


def rows_fixture():
    rows = []; source = c.training.pairing.source
    for condition in source.CONDITIONS:
        for wi, seed in enumerate(source.stream_manifest(False)["weather_bases"]):
            for step in (25, 75, 150):
                for member in range(3):
                    for family in source.FAMILIES:
                        for slot in range(6):
                            rows.append({"hardware_condition": condition, "weather_seed": seed,
                                "weather_fold": wi // 2, "probe_step": step, "member": member,
                                "family": family, "slot": slot, "sample_index": len(rows)})
    return rows


def test_frozen_contract_and_methods():
    c.contract(config())
    assert c.METHODS[4] == "truth_oracle_reference"
    assert config()["boundary"]["training_updates"] == 0
    assert config()["statistics"]["interval_scope"].startswith("descriptive")


@pytest.mark.parametrize("field,value", [("methods", ["oracle"]), ("runtime", {"device": "cpu"}),
    ("inference", {"future_safety_filter": True}), ("statistics", {"best_ci": True}),
    ("boundary", {"confirmation_access": True}), ("purpose", "independent_confirmation"),
    ("training_config", "changed.yaml"), ("quick", {"scientific_analysis": True})])
def test_contract_cannot_expand_scope(field, value):
    cfg = config(); cfg[field] = value
    with pytest.raises(ValueError, match="合同"):
        c.contract(cfg)


def test_held_out_route_cannot_mix_models_that_saw_weather():
    rows = [{"weather_fold": 0, "weather_seed": 10}] * 2 + [{"weather_fold": 1, "weather_seed": 20}] * 2
    assert c.evaluation_indices(split_fixture(), rows, quick=False) == [0, 1]
    rows[0] = {"weather_fold": 0, "weather_seed": 20}
    with pytest.raises(ValueError, match="已见天气"):
        c.evaluation_indices(split_fixture(), rows, quick=False)


@pytest.mark.parametrize("mutation", ["overlap", "duplicate", "empty", "wrong_fold"])
def test_evaluation_route_rejects_bad_fold(mutation):
    split = split_fixture(); rows = [{"weather_fold": 0, "weather_seed": 10}] * 4
    if mutation == "overlap": split["held_out_indices"] = [0, 2]
    elif mutation == "duplicate": split["held_out_indices"] = [0, 0]
    elif mutation == "empty": split["held_out_indices"] = []
    else: rows[0] = {"weather_fold": 1, "weather_seed": 10}
    with pytest.raises(ValueError):
        c.evaluation_indices(split, rows, quick=False)


def test_quick_is_explicitly_in_sample_technical():
    split = split_fixture()
    assert c.evaluation_indices(split, [], quick=True) == [2, 3]
    assert config()["quick"]["scientific_analysis"] is False


@pytest.mark.parametrize("mutation", ["fold", "seed", "kind", "update", "train_weather", "held_out_weather",
    "normalizer_shape", "normalizer_nan", "normalizer_zero", "candidate_order", "constant", "columns"])
def test_checkpoint_must_be_last_correct_fold_and_whitelist(mutation):
    ck = checkpoint_fixture()
    if mutation in ("fold", "seed", "update"): ck[mutation] = 2
    elif mutation == "kind": ck["kind"] = "current"
    elif mutation == "train_weather": ck[mutation] = [10, 20]
    elif mutation == "held_out_weather": ck[mutation] = [20]
    elif mutation == "normalizer_shape": ck["normalizer"]["mean"] = torch.zeros(79)
    elif mutation == "normalizer_nan": ck["normalizer"]["mean"][0] = float("nan")
    elif mutation == "normalizer_zero": ck["normalizer"]["std"][0] = 0
    elif mutation == "candidate_order": ck[mutation] = list(reversed(ck[mutation]))
    elif mutation == "constant": ck["training_only_constant_candidate"] = 25
    else: ck["feature_columns"].append(75)
    with pytest.raises(ValueError):
        c.validate_checkpoint(ck, split_fixture(), "history", 7564000, quick=False)


def test_valid_checkpoint_shape_and_training_fold():
    c.validate_checkpoint(checkpoint_fixture(), split_fixture(), "history", 7564000, quick=False)


def test_future_labels_can_only_change_oracle_not_model_or_constant():
    models = torch.tensor([[[2, 3], [4, 5]]] * 4)
    constant = torch.full((4,), 7)
    targets = target_fixture()
    before = c.method_choices(models, constant, targets["power_delta"])
    changed = targets["power_delta"].clone(); changed[:, 8] = 1
    after = c.method_choices(models, constant, changed)
    assert torch.equal(before[:, :4], after[:, :4])
    assert before[:, 4].eq(3).all() and after[:, 4].eq(8).all()


def test_unsafe_negative_model_choice_is_reported_not_filtered():
    targets = target_fixture(); models = torch.full((4, 2, 3), 2, dtype=torch.long)
    choices = c.method_choices(models, torch.full((4,), 3), targets["power_delta"])
    scores = c.score_choices(choices, targets, tolerance=.001)
    assert choices[:, 2:4].eq(2).all()
    assert scores["power_delta"][:, 2:4].eq(-.002).all()
    assert not scores["safety_pass"][:, 2:4].any()
    assert scores["oracle_regret"][:, 2:4].eq(.005).all()
    assert scores["safety_delta"][:, 2:4, :, 0].eq(.002).all()


def test_power_and_measured_power_not_conflated():
    targets = target_fixture(); models = torch.full((4, 2, 1), 3, dtype=torch.long)
    scores = c.score_choices(c.method_choices(models, torch.full((4,), 3), targets["power_delta"]), targets, tolerance=.001)
    assert scores["power_delta"][:, 2].eq(.003).all()
    assert scores["measured_power_delta"][:, 2].eq(.0015).all()
    assert not scores["power_delta"][:, 0].any()


@pytest.mark.parametrize("mutation", ["safety", "power_shape", "nan", "safety_range", "choices"])
def test_scoring_invalid_targets_or_choices_stop(mutation):
    targets = target_fixture(); models = torch.full((4, 2, 1), 3, dtype=torch.long)
    choices = c.method_choices(models, torch.full((4,), 3), targets["power_delta"])
    if mutation == "safety": targets["safety_pass"].fill_(True)
    elif mutation == "power_shape": targets["power_delta"] = targets["power_delta"][:, :24]
    elif mutation == "nan": targets["measured_power_delta"][0, 2] = float("nan")
    elif mutation == "safety_range": targets["safety_maxima"][0, 2, 0] = 2
    else: choices[0, 2, 0] = 25
    with pytest.raises(ValueError):
        c.score_choices(choices, targets, tolerance=.001)


def test_causal_forward_signature_and_outputs_do_not_take_targets():
    assert list(inspect.signature(c.forward_causal).parameters) == ["model", "x", "valid", "command", "normalizer"]
    model = c.training.CandidateScorer("current").eval()
    x = torch.zeros(2, 8, 76); mask = torch.ones(2, 8, dtype=torch.bool); cmd = torch.zeros(2, 11)
    norm = checkpoint_fixture()["normalizer"]
    assert torch.equal(c.forward_causal(model, x, mask, cmd, norm), model(x, mask, cmd))


def test_zero_weather_interval_finite_and_not_frame_bootstrap():
    result = c.descriptive_interval(torch.zeros(8, dtype=torch.float64), torch.zeros(5, 8, dtype=torch.long))
    assert result["descriptive_ci95"] == [0., 0.] and result["weather_count"] == 8
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="八个天气"):
        c.descriptive_interval(torch.zeros(1296), torch.zeros(5, 8, dtype=torch.long))


def test_summarize_keeps_all_seeds_groups_and_zero_oracle_null():
    rows = rows_fixture(); n = len(rows); targets = target_fixture(n)
    targets["power_delta"].zero_(); targets["measured_power_delta"].zero_()
    model_choices = torch.zeros(n, 2, 3, dtype=torch.long)
    evaluation = c.score_choices(c.method_choices(model_choices, torch.zeros(n, dtype=torch.long), targets["power_delta"]), targets, tolerance=.001)
    predictions = torch.zeros(n, 2, 3, 23)
    cfg = config(); cfg["statistics"]["weather_bootstrap_repeats"] = 5
    analysis, weather, groups = c.summarize(rows, evaluation, predictions, targets, cfg, [1, 2, 3])
    assert len(weather) == 16 and len(groups) == 108
    assert analysis["continuation_gate_evaluated"] is False
    for cell in analysis["cells"].values():
        assert cell["independent_weather_count"] == 8
        assert len(cell["by_training_seed"]) == 3
        assert cell["oracle_capture_fraction"]["current"] is None
        assert cell["paired_comparisons"]["history_vs_current"]["power_delta"]["descriptive_ci95"] == [0., 0.]
    json.dumps(analysis, allow_nan=False)


def test_existing_output_fails_without_overwrite(monkeypatch, tmp_path):
    cfg = config(); src = c.training.pairing.source; original_path = src._project_path
    monkeypatch.setattr(c, "verify_training", lambda quick: (tmp_path, {}, {}))
    monkeypatch.setattr(c.training, "verify_dataset", lambda quick: tmp_path)
    monkeypatch.setattr(src, "_project_path", lambda p: tmp_path if str(p) == cfg["output_directory"] else original_path(p))
    with pytest.raises(FileExistsError, match="禁止覆盖"):
        c.preflight()


def test_cuda_unavailable_cannot_fall_back(monkeypatch, tmp_path):
    cfg = config(); src = c.training.pairing.source; original_path = src._project_path
    monkeypatch.setattr(c, "verify_training", lambda quick: (tmp_path, {}, {}))
    monkeypatch.setattr(c.training, "verify_dataset", lambda quick: tmp_path)
    monkeypatch.setattr(src, "_project_path", lambda p: tmp_path / "not_created" if str(p) == cfg["output_directory"] else original_path(p))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="GPU training was requested"):
        c.preflight()
