"""R4-1C小型确定性CPU测试，不产生正式性能结果。"""
from copy import deepcopy
import pytest
import torch

from src.rl.r4_closed_loop_repair import (budget, mix_batches, prediction_gate,
    require_disjoint, settings, split_seeds, summarize_closed_loop, validate_config)
from src.rl.r4_trajectory import EpisodeStore
from src.rl.s4_training import _load_yaml, _project_path


def config():
    return _load_yaml(_project_path("configs/experiments/s4_r4_closed_loop_repair_v1.yaml"))


def test_exact_budget_includes_development_collection_and_pair_loss():
    cfg = config(); validate_config(cfg)
    b = budget(settings(cfg, False), "train")
    assert b == dict(physical_transitions=230400, max_planning_forward_samples=2831155200,
                    training_updates=12000, training_forward_samples=9216000,
                    prediction_forward_samples=1244160)
    assert budget(settings(cfg, False), "evaluate")["physical_transitions"] == 345600
    q = budget(settings(cfg, True), "train")
    assert q == dict(physical_transitions=48, max_planning_forward_samples=589824,
                    training_updates=12, training_forward_samples=768, prediction_forward_samples=3264)


@pytest.mark.parametrize("key,value", [("updates", 4000), ("pair_batch", 0), ("primary_checkpoint", "best"),
                                      ("mixed_new_fraction", .7), ("learning_rate", .001)])
def test_training_contract_rejects_unplanned_changes(key, value):
    cfg = config(); cfg["training"][key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_all_seed_splits_disjoint_and_confirmation_unused():
    cfg = config(); splits = split_seeds(cfg)
    assert len(set().union(*splits.values())) == 294
    cfg["data"]["development_starts"] = cfg["data"]["train_starts"]
    with pytest.raises(ValueError, match="overlap"):
        split_seeds(cfg)


def test_mixing_only_replaces_second_half_and_control_ignores_new():
    old = {"history": torch.arange(4.)[:, None], "target": torch.arange(4.)}
    new = {"history": torch.full((2, 1), 100.), "target": torch.full((2,), 100.)}
    saved = deepcopy(old)
    control = mix_batches(old, new, "old_data")
    mixed = mix_batches(old, new, "mixed_data")
    assert torch.equal(control["target"], saved["target"])
    assert mixed["target"].tolist() == [0., 1., 100., 100.]
    assert torch.equal(old["history"], saved["history"])
    new["target"][0] = -999
    assert torch.equal(mix_batches(old, new, "old_data")["target"], saved["target"])
    with pytest.raises(ValueError):
        mix_batches(old, {k: v[:1] for k, v in new.items()}, "mixed_data")


def metric_values(a=1., b=.9):
    return {"old_data": torch.full((3, 4, 2), a).tolist(),
            "mixed_data": torch.full((3, 4, 2), b).tolist()}


def test_prediction_heads_cannot_mask_each_other():
    m = metric_values(); assert prediction_gate(m)["passed"]
    m["mixed_data"][0][-1][1] = 2.
    assert not prediction_gate(m)["passed"]
    m = metric_values(0., 0.)
    assert not prediction_gate(m)["passed"]
    m = metric_values(); m["old_data"][0][0][0] = float("nan")
    with pytest.raises(ValueError):
        prediction_gate(m)


def store(seed):
    return EpisodeStore(dict(frames=torch.zeros(1, 9, 79), commands=torch.zeros(1, 8, 21),
                             corrections=torch.zeros(1, 8, 11), powers=torch.zeros(1, 8),
                             weather_seeds=torch.tensor([seed])))


def test_weather_and_paired_labels_cannot_leak():
    class Pairs:
        data = {"weather": torch.tensor([1])}
    stores = {k: store(s) for k, s in zip(("old_train", "old_development", "train", "development"), (1, 2, 3, 4))}
    require_disjoint(stores, Pairs())
    stores["development"] = store(3)
    with pytest.raises(ValueError, match="leakage"):
        require_disjoint(stores, Pairs())
    stores["development"] = store(4); Pairs.data = {"weather": torch.tensor([2])}
    with pytest.raises(ValueError, match="outside"):
        require_disjoint(stores, Pairs())


def physical_fixture():
    m = torch.full((3, 3, 6, 32, 6), .5, dtype=torch.float64)
    m[1, ..., 0] = .505; m[2, ..., 0] = .51
    return m


def test_complete_safe_improvement_is_development_only():
    m = physical_fixture()
    a = summarize_closed_loop(m, 5, 30)
    assert all(a["gates"].values())
    assert not a["rl_authorized"]
    assert a == summarize_closed_loop(m, 5, 30)


@pytest.mark.parametrize("case", ["small", "strehl", "phase", "violation", "saturation", "slew", "control", "family"])
def test_every_closed_loop_gate_can_block(case):
    m = physical_fixture()
    if case == "small": m[2, ..., 0] = .502
    elif case == "strehl": m[2, ..., 1] = .499
    elif case == "phase": m[2, ..., 3] = .501
    elif case == "violation": m[2, ..., 2] = .502
    elif case == "saturation": m[2, ..., 4] = .502
    elif case == "slew": m[2, ..., 5] = .502
    elif case == "control": m[1, ..., 0] = .52
    else: m[2, 0, ..., 0] = .499
    assert not all(summarize_closed_loop(m, 6, 30)["gates"].values())


def test_partial_and_nonfinite_closed_loop_is_not_success():
    m = physical_fixture()
    with pytest.raises(ValueError): summarize_closed_loop(m[:, :, :, :31], 0, 20)
    m[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError): summarize_closed_loop(m, 0, 20)
