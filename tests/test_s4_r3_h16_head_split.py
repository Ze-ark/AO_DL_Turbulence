"""A8小型确定性CPU测试；不产生正式训练或物理性能结论。"""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src.rl.s4_r3_h16_head_split import (
    ARMS, HeadSplitProbe, _cluster_sums, _effective_settings, _initial_probe_state,
    checkpoint_outputs, compare_heads, fit_head_probe, interpret_heads, measure_outputs,
    preflight_head_split, seal_heads, subgroup_outputs, validate_coverage, verify_control_parity,
)
from src.rl.s4_r3_h16_data_scaling import fit_scaling_probe
from src.rl.s4_r3_h16_reward_pairwise_probe import RewardPairDataset
from src.rl.s4_training import _file_sha256, _load_yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/s4_r3_h16_head_split_v1.yaml"


def test_cli_help_uses_utf8_and_does_not_start_training() -> None:
    result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/train_s4_r3_h16_head_split.py"), "--help"],
                            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
    assert "正式训练由用户在IDE启动" in result.stdout
    assert "只读核对" in result.stdout


def _settings() -> dict:
    settings = _effective_settings(_load_yaml(CONFIG), quick=True)
    settings.update(maximum_updates=4, validation_interval_updates=2, batch_size=8,
                    cluster_bootstrap_replicates=20)
    return settings


def _pairs(offset: int = 0) -> RewardPairDataset:
    generator = torch.Generator().manual_seed(100 + offset)
    features = torch.randn(48, 221, generator=generator)
    target = torch.where(torch.arange(48) % 2 == 0, .2, -.2)
    rows = [{"condition_id": f"c{(i // 4) % 3}", "episode_seed": offset + i // 4,
             "episode_index": i // 4, "profile_id": f"p{i % 2}", "probe_step": i % 4}
            for i in range(48)]
    return RewardPairDataset(features, target, target.clone(), rows)


def test_initial_outputs_and_parameter_budget() -> None:
    settings = _settings()
    initial = _initial_probe_state(settings=settings, policy_seed=9301, device=torch.device("cpu"))
    models = {arm: HeadSplitProbe(221, 256, arm) for arm in ARMS}
    for model in models.values():
        model.load_shared_initial(initial)
    sample = _pairs().features[:4]
    outputs = [models[arm](sample) for arm in ARMS]
    assert all(torch.equal(outputs[0][0], tensor) for output in outputs for tensor in output)
    assert sum(p.numel() for p in models["shared"].parameters()) == 122881
    assert sum(p.numel() for p in models["split"].parameters()) == 123138


def test_separate_output_gradients_do_not_cross_last_layer() -> None:
    model = HeadSplitProbe(221, 16, "split")
    value, score = model(_pairs().features)
    F.huber_loss(value, _pairs().reward_delta).backward()
    assert model.sign_head.weight.grad is None
    assert model.network[4].weight.grad is not None
    model.zero_grad(set_to_none=True)
    value, score = model(_pairs().features)
    F.binary_cross_entropy_with_logits(score, (_pairs().reward_delta > 0).float()).backward()
    assert model.network[4].weight.grad is None
    assert model.sign_head.weight.grad is not None
    assert model.network[0].weight.grad is not None


def test_scores_are_never_used_as_reward_magnitudes() -> None:
    data = _pairs()
    metrics = measure_outputs(data.reward_delta, -data.reward_delta * 1000, data, 0)
    assert metrics["value"]["mae"] == 0
    assert metrics["ranking"]["balanced_accuracy"] == 0
    assert metrics["value_sign_ranking"]["balanced_accuracy"] == 1
    assert metrics["head_sign_disagreement_fraction"] == 1
    assert "mae" not in metrics["ranking"]
    with pytest.raises(ValueError, match="shapes"):
        measure_outputs(data.reward_delta[:2], data.reward_delta, data, 0)
    with pytest.raises(RuntimeError, match="non-finite"):
        measure_outputs(data.reward_delta, data.reward_delta * float("nan"), data, 0)


def test_cluster_comparison_has_known_gain_and_preserves_repeats() -> None:
    data, settings = _pairs(), _settings()
    shared = (data.reward_delta + .1, data.reward_delta)
    split = (data.reward_delta + .05, data.reward_delta)
    result = compare_heads(shared, split, data, settings, 3)
    assert result["clusters"] == 12
    assert result["relative_mae_reduction"]["estimate"] == pytest.approx(.5, abs=1e-6)
    assert result["ranking_ba_delta"]["familywise_ci_low"] == 0
    repeated = RewardPairDataset(torch.cat([data.features] * 2), torch.cat([data.reward_delta] * 2),
                                 torch.cat([data.power_delta] * 2), data.rows * 2)
    duplicated = compare_heads(tuple(torch.cat([v] * 2) for v in shared),
                               tuple(torch.cat([v] * 2) for v in split), repeated, settings, 3)
    assert duplicated["clusters"] == 12
    assert duplicated["relative_mae_reduction"] == result["relative_mae_reduction"]
    bad_rows = deepcopy(data.rows)
    bad_rows[0]["condition_id"] = "different"
    with pytest.raises(RuntimeError, match="multiple conditions"):
        _cluster_sums(np.ones((48, 1)), bad_rows, replicates=20, seed=1)


def test_subgroups_preserve_startup_and_single_class() -> None:
    data = _pairs()
    groups = subgroup_outputs(data.reward_delta, data.reward_delta, data, 0, policy_seed=9301, arm="split")
    assert {g["group_id"] for g in groups if g["group_kind"] == "probe_step"} == {0, 1, 2, 3}
    assert any(not g["two_class"] for g in groups)
    with pytest.raises(RuntimeError, match="coverage"):
        validate_coverage(data, episodes=384, pairs=6912)
    duplicated = RewardPairDataset(data.features, data.reward_delta, data.power_delta, [data.rows[0]] * 48)
    with pytest.raises(RuntimeError, match="duplicated"):
        validate_coverage(duplicated, episodes=1, pairs=48)


def test_formal_preflight_is_read_only_and_matches_design(tmp_path: Path) -> None:
    experiment = _load_yaml(CONFIG)
    settings = _effective_settings(experiment, quick=False)
    settings["output_directory"] = str(tmp_path / "formal")
    with patch("torch.cuda.is_available", return_value=True):
        result, _, policies = preflight_head_split(CONFIG, experiment, settings, quick=False)
    assert result["planned_fits"] == 6
    assert result["maximum_updates_per_fit"] == 10000
    assert result["expected_test_branches"] == 324
    assert result["test_namespace"]["episodes"] == 96
    assert result["test_namespace"]["minimum"] == 3861000
    assert result["new_test_opened"] is False
    assert len(policies) == 3
    assert not (tmp_path / "formal").exists()
    assert all(spec["split"] in ("large", "validation") for data in result["data_specs"].values() for spec in data.values())
    (tmp_path / "formal").mkdir()
    with patch("torch.cuda.is_available", return_value=True), pytest.raises(FileExistsError):
        preflight_head_split(CONFIG, experiment, settings, quick=False)


@pytest.mark.parametrize("field", ["allow_full_rl_training", "allow_actor_updates", "allow_new_training_data", "allow_real_hardware_actions"])
def test_no_scope_expansion(field: str, tmp_path: Path) -> None:
    experiment = _load_yaml(CONFIG)
    experiment["metadata"][field] = True
    settings = _effective_settings(experiment, quick=True)
    settings["output_directory"] = str(tmp_path / "guard")
    with pytest.raises(RuntimeError, match="safety flag"):
        preflight_head_split(CONFIG, experiment, settings, quick=True)


@pytest.mark.parametrize("change", ["budget", "test", "selection", "cpu", "quick_overlap"])
def test_contract_guards(change: str, tmp_path: Path) -> None:
    experiment = _load_yaml(CONFIG)
    if change == "budget":
        experiment["design"]["maximum_updates"] = 20000
    elif change == "test":
        experiment["data"]["test_base_seeds"][0] = 4000000
    elif change == "selection":
        experiment["design"]["selection"] = "test_best"
    elif change == "quick_overlap":
        experiment["quick"]["test_seed_offset"] = 0
    else:
        experiment["runtime"]["device"] = "cpu"
    settings = _effective_settings(experiment, quick=True)
    settings["output_directory"] = str(tmp_path / "guard")
    with patch("torch.cuda.is_available", return_value=True), pytest.raises(RuntimeError):
        preflight_head_split(CONFIG, experiment, settings, quick=True)


def test_cpu_tiny_shared_fit_reproduces_a7_and_has_no_test_parameter(tmp_path: Path) -> None:
    assert not any("test" in name for name in inspect.signature(fit_head_probe).parameters)
    settings = _settings()
    initial = _initial_probe_state(settings=settings, policy_seed=9301, device=torch.device("cpu"))
    untouched = deepcopy(initial)
    old = fit_scaling_probe(development=_pairs(100), validation=_pairs(200), settings=settings,
                           policy_seed=9301, scale="large", objective={"id": "paired_delta_plus_balanced_sign", "balanced_sign_weight": .25},
                           initial_state=initial, output_directory=tmp_path / "old", device=torch.device("cpu"))
    new = fit_head_probe(training=_pairs(100), validation=_pairs(200), settings=settings,
                         policy_seed=9301, arm="shared", initial=initial, output=tmp_path / "new", device=torch.device("cpu"))
    assert verify_control_parity(new, old)["matched"]
    assert new["updates_completed"] == 4
    assert all(torch.equal(initial[k], untouched[k]) for k in initial)
    log = [json.loads(line) for line in (ROOT / new["progress_log"]).read_text().splitlines()]
    assert [r["update"] for r in log] == [2, 4]
    assert all("estimated_remaining_seconds" in r and "cuda_allocated_gb" in r for r in log)
    value, score, _ = checkpoint_outputs(new, _pairs(300), torch.device("cpu"))
    assert value.shape == score.shape == (48,)
    assert set(new["predictions"]) == {"training", "validation"}
    assert new["actor_updates"] == new["original_critic_updates"] == 0
    bad = deepcopy(old)
    bad["best_update"] += 1
    with pytest.raises(RuntimeError, match="did not reproduce"):
        verify_control_parity(new, bad)


def test_cpu_tiny_split_fit_saves_two_heads(tmp_path: Path) -> None:
    settings = _settings()
    initial = _initial_probe_state(settings=settings, policy_seed=9301, device=torch.device("cpu"))
    fit = fit_head_probe(training=_pairs(100), validation=_pairs(200), settings=settings,
                         policy_seed=9301, arm="split", initial=initial, output=tmp_path, device=torch.device("cpu"))
    checkpoint = torch.load(ROOT / fit["checkpoint"], weights_only=False)
    assert "sign_head.weight" in checkpoint["probe"]
    assert torch.equal(checkpoint["normalization"]["feature_mean"], _pairs(100).features.mean(dim=0))
    payload = torch.load(ROOT / fit["predictions"]["validation"]["path"], weights_only=False)
    assert set(payload) == {"reward_prediction", "ranking_logit", "reward_delta", "power_delta", "rows"}
    assert "q1" not in checkpoint and "actor" not in checkpoint


def test_seal_requires_all_fits_budget_and_control_parity(tmp_path: Path) -> None:
    settings = _settings()
    fits = []
    for arm in ARMS:
        path = tmp_path / f"{arm}.pt"
        torch.save({"probe": {}}, path)
        fits.append({"policy_seed": 9301, "arm": arm, "updates_completed": 4,
                     "checkpoint": str(path), "checkpoint_sha256": _file_sha256(path)})
    with pytest.raises(RuntimeError, match="all fits"):
        seal_heads(fits[:1], settings, tmp_path)
    partial = deepcopy(fits)
    partial[0]["updates_completed"] = 3
    with pytest.raises(RuntimeError, match="budget"):
        seal_heads(partial, settings, tmp_path)
    formal = {**settings, "quick": False}
    with pytest.raises(RuntimeError, match="parity"):
        seal_heads(fits, formal, tmp_path)
    marker = seal_heads(fits, settings, tmp_path)
    assert json.loads(marker.read_text())["new_test_opened"] is False
    with pytest.raises(FileExistsError):
        seal_heads(fits, settings, tmp_path)


def test_interpretation_uses_both_endpoints_and_never_authorizes_rl() -> None:
    settings = _settings()
    settings["quick"] = False
    metrics = {"ranking": {"balanced_accuracy": .7, "matthews_correlation": .3},
               "ranking_ci": {"low": .55}, "value": {"mae_better_than_constant": True}}
    fits = [{"policy_seed": 9301, "arm": arm, "validation": metrics, "independent_test": metrics,
             "test_subgroups": [{"group_kind": "profile_id" if i < 6 else "condition_id",
                                 "group_id": str(i), "two_class": True, "balanced_accuracy": .6} for i in range(9)]}
            for arm in ARMS]
    comparisons = [{"policy_seed": 9301, "relative_mae_reduction": {"estimate": .1, "familywise_ci_low": .01},
                    "ranking_ba_delta": {"familywise_ci_low": -.01}}]
    result = interpret_heads(fits, comparisons, settings)
    assert result["status"] == "CONSISTENT_HEAD_SPLIT_BENEFIT"
    assert result["scalar_q_gate_equivalent"] is False and result["full_rl_authorized"] is False
    assert result["all_split_probe_gates_pass"] is True
    missing = deepcopy(fits)
    missing[1]["test_subgroups"] = []
    assert interpret_heads(missing, comparisons, settings)["all_split_probe_gates_pass"] is False
    comparisons[0]["ranking_ba_delta"]["familywise_ci_low"] = -.03
    assert interpret_heads(fits, comparisons, settings)["status"] == "HEAD_SPLIT_BENEFIT_NOT_CONFIRMED"
    with pytest.raises(RuntimeError, match="incomplete"):
        interpret_heads(fits[:1], comparisons, settings)
