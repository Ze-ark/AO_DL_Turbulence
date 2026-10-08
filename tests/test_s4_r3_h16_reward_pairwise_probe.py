from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.rl.s4_r3_h16_reward_pairwise_probe import (
    RewardPairDataset,
    _effective_settings,
    _initial_probe_state,
    _normalization_from_training,
    _verify_split_separation,
    evaluate_h16_prediction,
    fit_h16_reward_probe,
    interpret_h16_reward_fits,
    load_reward_pair_dataset,
    preflight_s4_r3_h16_reward_pairwise_probe,
)
from src.rl.s4_training import _load_yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "s4_r3_h16_reward_pairwise_probe_v1.yaml"
)


def _synthetic_pairs(samples: int = 96, seed_offset: int = 0) -> RewardPairDataset:
    generator = torch.Generator().manual_seed(42 + seed_offset)
    features = torch.randn(samples, 221, generator=generator)
    indices = torch.arange(samples)
    reward = torch.where(indices % 2 == 0, 0.2, -0.2).float()
    reward = reward + 0.01 * features[:, 0]
    power = reward + 0.005 * features[:, 1]
    rows = [
        {
            "profile_id": f"profile_{(index // 2) % 2}",
            "condition_id": f"condition_{(index // 4) % 3}",
            "probe_step": index % 3,
            "episode_index": index,
            "episode_seed": 100000 * seed_offset + 1000 + index,
        }
        for index in range(samples)
    ]
    dataset = RewardPairDataset(
        features=features,
        reward_delta=reward,
        power_delta=power,
        rows=rows,
    )
    dataset.validate(feature_size=221)
    return dataset


def _passing_metrics() -> dict[str, object]:
    return {
        "reward_ranking": {
            "balanced_accuracy": 0.70,
            "matthews_correlation": 0.40,
            "mae": 0.01,
            "constant_training_mean_mae": 0.02,
            "mae_better_than_constant": True,
            "balanced_accuracy_cluster_ci": {"low": 0.60, "high": 0.80},
        },
        "physical_power_alignment": {"status": "OK"},
        "high_magnitude_reward_ranking": {"status": "OK"},
    }


def test_formal_preflight_locks_three_splits_and_single_factor(
    tmp_path: Path,
) -> None:
    experiment = _load_yaml(CONFIG)
    settings = _effective_settings(experiment, quick=False)
    settings["output_directory"] = str(tmp_path / "a6_preflight_contract")
    with patch("torch.cuda.is_available", return_value=True):
        result = preflight_s4_r3_h16_reward_pairwise_probe(
            CONFIG,
            experiment,
            settings,
            quick=False,
        )

    assert result["status"] == "READY_FOR_USER_TRAINING"
    assert result["planned_probe_fits"] == 6
    assert result["development_pairs"] == 5184
    assert result["validation_pairs"] == 2592
    assert result["independent_test_pairs"] == 2592
    assert result["independent_test_used_for_model_selection"] is False
    assert result["original_critic_updates"] == 0
    assert result["actor_updates"] == 0
    assert result["full_rl_training"] is False


def test_pair_builder_uses_direct_h16_reward_and_power_deltas() -> None:
    experiment = _load_yaml(CONFIG)
    spec = experiment["data"]["independent_test"][9301]
    dataset = load_reward_pair_dataset(
        spec,
        target_source="empirical_reward_returns",
        power_source="empirical_power_returns",
        horizon_index=15,
        state_size=210,
        action_size=11,
    )

    assert dataset.features.shape == (864, 221)
    assert dataset.reward_delta.shape == (864,)
    assert dataset.power_delta.shape == (864,)
    assert 0.15 < float((dataset.reward_delta > 0).float().mean()) < 0.25
    assert float(
        ((dataset.reward_delta > 0) == (dataset.power_delta > 0)).float().mean()
    ) > 0.80


def test_three_way_episode_leakage_is_rejected() -> None:
    development = _synthetic_pairs(seed_offset=1)
    validation = _synthetic_pairs(seed_offset=2)
    independent_test = RewardPairDataset(
        features=development.features,
        reward_delta=development.reward_delta,
        power_delta=development.power_delta,
        rows=development.rows,
    )
    with pytest.raises(RuntimeError, match="leakage"):
        _verify_split_separation(
            {
                "development": development,
                "validation": validation,
                "independent_test": independent_test,
            }
        )


def test_evaluation_reports_reward_power_and_high_magnitude() -> None:
    dataset = _synthetic_pairs()
    result = evaluate_h16_prediction(
        dataset.reward_delta.clone(),
        dataset,
        training_reward_mean=float(dataset.reward_delta.mean()),
        magnitude_threshold=0.1,
        bootstrap_replicates=100,
        bootstrap_seed=123,
        confidence_level=0.95,
    )

    assert result["reward_ranking"]["balanced_accuracy"] == 1.0
    assert result["reward_ranking"]["balanced_accuracy_cluster_ci"]["low"] == 1.0
    assert result["physical_power_alignment"]["balanced_accuracy"] == 1.0
    assert result["high_magnitude_reward_ranking"]["balanced_accuracy"] == 1.0


def test_tiny_fit_writes_probe_only_checkpoint_and_keeps_test_out_of_selection(
    tmp_path: Path,
) -> None:
    experiment = _load_yaml(CONFIG)
    settings = _effective_settings(experiment, quick=True)
    settings.update(
        {
            "maximum_updates": 4,
            "validation_interval_updates": 2,
            "early_stopping_patience_updates": 4,
            "cluster_bootstrap_replicates": 20,
            "batch_size": 16,
            "high_magnitude_thresholds": {9301: 0.0},
        }
    )
    development = _synthetic_pairs(seed_offset=1)
    validation = _synthetic_pairs(seed_offset=2)
    independent_test = _synthetic_pairs(seed_offset=3)
    normalization = _normalization_from_training(development, settings)
    initial = _initial_probe_state(
        settings=settings,
        policy_seed=9301,
        device=torch.device("cpu"),
    )
    result, subgroup_rows = fit_h16_reward_probe(
        development=development,
        validation=validation,
        independent_test=independent_test,
        objective=settings["objectives"][1],
        settings=settings,
        policy_seed=9301,
        normalization=normalization,
        initial_state=initial,
        output_directory=tmp_path,
        device=torch.device("cpu"),
    )
    checkpoint_path = ROOT / result["checkpoint"]
    if not checkpoint_path.exists():
        checkpoint_path = Path(result["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert result["updates_completed"] == 4
    assert result["independent_test_used_for_selection"] is False
    assert checkpoint["independent_test_used_for_selection"] is False
    assert checkpoint["label_source"] == "empirical_reward_returns"
    assert "probe" in checkpoint
    assert "actor" not in checkpoint
    assert checkpoint["original_critic_updates"] == 0
    assert checkpoint["actor_updates"] == 0
    assert len(subgroup_rows) == 5


def test_passing_probe_only_authorizes_non_bootstrapped_critic_design() -> None:
    experiment = _load_yaml(CONFIG)
    settings = _effective_settings(experiment, quick=False)
    fits = []
    for seed in settings["policy_seeds"]:
        for objective in (
            "paired_delta_regression",
            "paired_delta_plus_balanced_sign",
        ):
            fits.append(
                {
                    "policy_seed": seed,
                    "objective": objective,
                    "validation": _passing_metrics(),
                    "independent_test": _passing_metrics(),
                    "independent_test_subgroup_minimum_balanced_accuracy": 0.60,
                    "independent_test_subgroups_all_two_class": True,
                }
            )
    result = interpret_h16_reward_fits(fits, settings=settings, quick=False)

    assert result["status"] == "STABLE_H16_LABEL_RESTORES_PAIRWISE_LEARNABILITY"
    assert result["critic_design_authorized"] is True
    assert result["authorization_scope"] == "non_bootstrapped_critic_design_only"
    assert result["full_rl_authorized"] is False
    assert result["s4d3_authorized"] is False
