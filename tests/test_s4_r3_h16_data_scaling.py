"""A7的小型确定性CPU单元测试，不产生正式仿真性能结果。"""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.rl.s4_r3_h16_data_scaling import (
    _checkpoint_predictions, _effective_settings, _initial_probe_state,
    _select_old_episodes, _split_seeds, concatenate_pairs, fit_scaling_probe,
    paired_episode_bootstrap, preflight_s4_r3_h16_data_scaling, scaling_decision,
    seal_training, verify_dataset_separation, verify_new_seed_namespace,
)
from src.rl.s4_r3_h16_reward_pairwise_probe import RewardPairDataset
from src.rl.s4_training import _load_yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/s4_r3_h16_data_scaling_v1.yaml"


def _pairs(seed_offset: int = 0, episodes: int = 12) -> RewardPairDataset:
    generator = torch.Generator().manual_seed(100 + seed_offset)
    count = episodes * 4
    features = torch.randn(count, 221, generator=generator)
    reward = torch.where(torch.arange(count) % 2 == 0, 0.2, -0.2)
    rows = [{"condition_id": f"c{(i // 4) % 3}", "episode_seed": seed_offset + i // 4,
             "episode_index": i // 4, "profile_id": f"p{i % 2}", "probe_step": i % 4}
            for i in range(count)]
    data = RewardPairDataset(features, reward, reward.clone(), rows)
    data.validate(feature_size=221)
    return data


def _settings() -> dict:
    return _effective_settings(_load_yaml(CONFIG), quick=True)


def test_formal_contract_and_seed_spaces(tmp_path: Path) -> None:
    experiment = _load_yaml(CONFIG)
    settings = _effective_settings(experiment, quick=False)
    settings["output_directory"] = str(tmp_path / "preflight")
    with patch("torch.cuda.is_available", return_value=True):
        result, _, checkpoints = preflight_s4_r3_h16_data_scaling(CONFIG, experiment, settings, quick=False)
    assert result["planned_probe_fits"] == 12
    assert result["expected_branch_rollouts"] == 1296
    assert result["maximum_updates_per_fit"] == 10000
    assert result["new_test_opened"] is False
    assert [entry["episodes"] for entry in result["seed_namespace"]["splits"]] == [96, 96, 96, 96]
    assert len(checkpoints) == 3
    for checkpoint in checkpoints:
        assert result["frozen_input_hashes"][checkpoint["student_path"]] == checkpoint["student_sha256"]


@pytest.mark.parametrize("change", ["rl", "budget", "test_early", "cpu"])
def test_safety_changes_are_rejected(change: str, tmp_path: Path) -> None:
    experiment = _load_yaml(CONFIG)
    settings = _effective_settings(experiment, quick=False)
    settings["output_directory"] = str(tmp_path / "guard")
    if change == "rl":
        experiment["metadata"]["allow_actor_updates"] = True
    elif change == "budget":
        experiment["training"]["early_stopping"] = True
    elif change == "test_early":
        experiment["data"]["test_generation_after_all_checkpoints_frozen"] = False
    else:
        experiment["runtime"]["device"] = "cpu"
    with patch("torch.cuda.is_available", return_value=True), pytest.raises(RuntimeError):
        preflight_s4_r3_h16_data_scaling(CONFIG, experiment, settings, quick=False)


def test_new_seed_overlap_and_reserved_test_are_rejected() -> None:
    settings = _settings()
    protected = _split_seeds(settings["added_splits"][0])
    with pytest.raises(RuntimeError, match="overlap"):
        verify_new_seed_namespace(settings, protected=protected, reserved=4000000)
    settings["test_split"]["conditions"][0]["base_seed"] = 4000000
    with pytest.raises(RuntimeError, match="reserved"):
        verify_new_seed_namespace(settings, protected=set(), reserved=4000000)


def test_nested_training_and_complete_episode_subsampling() -> None:
    small, addition = _pairs(100), _pairs(1000)
    large = concatenate_pairs([small, addition])
    assert torch.equal(large.features[:len(small.rows)], small.features)
    assert len({row["episode_seed"] for row in large.rows}) == 24
    selected = _select_old_episodes(small, 2)
    assert len(selected.rows) == 6 * 4
    counts = {}
    for row in selected.rows:
        counts[row["episode_seed"]] = counts.get(row["episode_seed"], 0) + 1
    assert set(counts.values()) == {4}
    with pytest.raises(RuntimeError, match="duplicated"):
        concatenate_pairs([small, small])
    with pytest.raises(RuntimeError, match="leakage"):
        verify_dataset_separation({"train": large, "validation": addition})


def test_paired_cluster_bootstrap_keeps_shared_episode_repeats() -> None:
    data = _pairs()
    result = paired_episode_bootstrap(-data.reward_delta, data.reward_delta, data,
             replicates=100, seed=17, familywise_alpha=0.05, family_size=6)
    assert result["clusters"] == 12
    assert result["balanced_accuracy_gain"] == 1.0
    assert result["familywise_ci_low"] == 1.0
    equal = paired_episode_bootstrap(data.reward_delta, data.reward_delta, data,
             replicates=100, seed=17, familywise_alpha=0.05, family_size=6)
    assert equal["ci95_low"] == equal["ci95_high"] == 0.0
    # 复制同一回合的观测不能把簇数翻倍。
    repeated = RewardPairDataset(torch.cat([data.features] * 2), torch.cat([data.reward_delta] * 2),
                                 torch.cat([data.power_delta] * 2), data.rows * 2)
    repeated_result = paired_episode_bootstrap(-repeated.reward_delta, repeated.reward_delta, repeated,
             replicates=100, seed=17, familywise_alpha=0.05, family_size=6)
    assert repeated_result["clusters"] == 12


def test_cpu_unit_fit_has_no_test_access_and_runs_full_budget(tmp_path: Path) -> None:
    assert "independent_test" not in inspect.signature(fit_scaling_probe).parameters
    settings = _settings()
    settings.update(maximum_updates=4, validation_interval_updates=2, batch_size=8,
                    cluster_bootstrap_replicates=20)
    initial = _initial_probe_state(settings=settings, policy_seed=9301, device=torch.device("cpu"))
    initial_copy = deepcopy(initial)
    fit = fit_scaling_probe(development=_pairs(100), validation=_pairs(200),
                           settings=settings, policy_seed=9301, scale="small",
                           objective=settings["objectives"][1], initial_state=initial,
                           output_directory=tmp_path, device=torch.device("cpu"))
    assert fit["updates_completed"] == 4
    assert fit["sample_presentations"] == 32
    assert fit["independent_test_used_for_selection"] is False
    checkpoint = torch.load(ROOT / fit["checkpoint"], map_location="cpu", weights_only=False)
    assert set(checkpoint["normalization"]) == {"feature_mean", "feature_scale", "target_scale"}
    assert torch.equal(checkpoint["normalization"]["feature_mean"], _pairs(100).features.mean(dim=0))
    assert "actor" not in checkpoint and "q1" not in checkpoint
    assert all(torch.equal(initial[key], initial_copy[key]) for key in initial)
    logs = (ROOT / fit["progress_log"]).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["update"] for line in logs] == [2, 4]
    prediction, _ = _checkpoint_predictions(fit, _pairs(300), device=torch.device("cpu"))
    assert prediction.shape == (48,)
    with pytest.raises(RuntimeError, match="all fits"):
        seal_training([fit], settings=settings, output=tmp_path)


def test_data_benefit_needs_all_seeds_and_corrected_interval() -> None:
    settings = _effective_settings(_load_yaml(CONFIG), quick=False)
    comparisons = [{"policy_seed": seed, "objective": objective["id"],
                    "balanced_accuracy_gain": 0.03, "familywise_ci_low": 0.001}
                   for seed in settings["policy_seeds"] for objective in settings["objectives"]]
    result = scaling_decision(comparisons, [], settings=settings, quick=False)
    assert result["status"] == "CONSISTENT_BENEFIT_FROM_MORE_EPISODES"
    assert result["full_rl_authorized"] is False
    assert result["critic_design_authorized"] is False
    for row in comparisons:
        if row["policy_seed"] == 9302:
            row["familywise_ci_low"] = -0.001
    result = scaling_decision(comparisons, [], settings=settings, quick=False)
    assert result["status"] == "DATA_BENEFIT_NOT_CONFIRMED_AT_THIS_BUDGET"
    assert scaling_decision(comparisons, [], settings=settings, quick=True)["status"] == "QUICK_SMOKE_ONLY"


def test_seal_rejects_partial_budget_and_changed_checkpoint(tmp_path: Path) -> None:
    from src.rl.s4_training import _file_sha256

    settings = _settings()
    fits = []
    for scale in ("small", "large"):
        for objective in settings["objectives"]:
            path = tmp_path / f"{scale}_{objective['id']}.pt"
            torch.save({"probe": torch.zeros(1)}, path)
            fits.append({"policy_seed": 9301, "scale": scale, "objective": objective["id"],
                         "updates_completed": settings["maximum_updates"],
                         "checkpoint": str(path), "checkpoint_sha256": _file_sha256(path)})
    incomplete = deepcopy(fits)
    incomplete[0]["updates_completed"] -= 1
    with pytest.raises(RuntimeError, match="budget"):
        seal_training(incomplete, settings=settings, output=tmp_path)
    tampered = deepcopy(fits)
    tampered[0]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="hash mismatch"):
        seal_training(tampered, settings=settings, output=tmp_path)
    marker = seal_training(fits, settings=settings, output=tmp_path)
    assert json.loads(marker.read_text(encoding="utf-8"))["test_generated"] is False
    with pytest.raises(FileExistsError):
        seal_training(fits, settings=settings, output=tmp_path)
