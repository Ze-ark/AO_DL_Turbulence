from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import torch

from src.rl.s4_r3_pairwise_rank_learnability import (
    _effective_settings,
    _initial_probe_state,
    _normalization_from_training,
    PairDataset,
    classification_metrics,
    cluster_bootstrap_balanced_accuracy,
    fit_pairwise_probe,
    interpret_pairwise_fits,
    load_pair_dataset,
    preflight_s4_r3_pairwise_rank_learnability,
)
from src.rl.s4_training import _load_yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "s4_r3_pairwise_rank_learnability_v1.yaml"
)


def _synthetic_pairs(samples: int = 48) -> PairDataset:
    generator = torch.Generator().manual_seed(42)
    features = torch.randn(samples, 221, generator=generator)
    target = 0.2 * features[:, 0] - 0.15 * features[:, 210]
    rows = [
        {
            "profile_id": f"profile_{index % 2}",
            "condition_id": f"condition_{(index // 8) % 3}",
            "probe_step": index % 3,
            "episode_index": (index // 2) % 8,
            "episode_seed": 1000 + (index // 2) % 8,
        }
        for index in range(samples)
    ]
    dataset = PairDataset(features=features, target_delta=target, rows=rows)
    dataset.validate(feature_size=221)
    return dataset


def test_formal_preflight_locks_pair_counts_and_seals_audit(
    tmp_path: Path,
) -> None:
    experiment = _load_yaml(CONFIG)
    settings = _effective_settings(experiment, quick=False)
    # The real A3 directory is sealed after the formal run. Use a fresh path so
    # this contract test remains repeatable without touching formal evidence.
    settings["output_directory"] = str(tmp_path / "a3_preflight_contract")
    with patch("torch.cuda.is_available", return_value=True):
        result = preflight_s4_r3_pairwise_rank_learnability(
            CONFIG, experiment, settings, quick=False
        )

    assert result["status"] == "READY_FOR_USER_TRAINING"
    assert result["planned_probe_fits"] == 6
    assert result["development_pairs"] == 5184
    assert result["validation_pairs"] == 2592
    assert result["mechanism_audit_access"] is False
    assert result["actor_updates"] == 0
    assert result["s4d3_access"] is False


def test_pair_builder_uses_shared_state_and_direct_h32_delta() -> None:
    experiment = _load_yaml(CONFIG)
    spec = experiment["data"]["development"][9301]
    dataset = load_pair_dataset(
        spec,
        horizon_index=31,
        state_size=210,
        action_size=11,
    )

    assert dataset.features.shape == (1728, 221)
    assert dataset.target_delta.shape == (1728,)
    assert 0.25 < float((dataset.target_delta > 0).float().mean()) < 0.35


def test_balanced_accuracy_exposes_majority_classifier() -> None:
    target = torch.tensor([-1.0] * 7 + [1.0] * 3)
    prediction = torch.full_like(target, -1.0)
    metrics = classification_metrics(prediction, target)

    assert metrics["raw_accuracy"] == 0.7
    assert metrics["majority_class_accuracy"] == 0.7
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["matthews_correlation"] == 0.0


def test_cluster_bootstrap_uses_complete_trajectory_units() -> None:
    dataset = _synthetic_pairs()
    prediction = dataset.target_delta.clone()
    result = cluster_bootstrap_balanced_accuracy(
        prediction,
        dataset.target_delta,
        dataset.rows,
        replicates=100,
        seed=123,
        confidence_level=0.95,
    )

    assert result["unit"] == "condition_x_episode"
    assert result["clusters"] == 24
    assert result["low"] == 1.0
    assert result["high"] == 1.0


def test_tiny_pairwise_fit_writes_probe_only_checkpoint(tmp_path: Path) -> None:
    experiment = _load_yaml(CONFIG)
    settings = _effective_settings(experiment, quick=True)
    settings.update(
        {
            "maximum_updates": 4,
            "validation_interval_updates": 2,
            "early_stopping_patience_updates": 4,
            "cluster_bootstrap_replicates": 20,
            "batch_size": 16,
        }
    )
    dataset = _synthetic_pairs()
    normalization = _normalization_from_training(dataset, settings)
    initial = _initial_probe_state(
        settings=settings,
        policy_seed=9301,
        device=torch.device("cpu"),
    )
    result = fit_pairwise_probe(
        development=dataset,
        validation=dataset,
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
    assert "probe" in checkpoint
    assert "actor" not in checkpoint
    assert checkpoint["original_critic_updates"] == 0
    assert checkpoint["actor_updates"] == 0


def test_interpretation_never_authorizes_full_rl() -> None:
    experiment = _load_yaml(CONFIG)
    settings = _effective_settings(experiment, quick=False)
    fits = []
    for seed in settings["policy_seeds"]:
        for objective in ("paired_delta_regression", "paired_delta_plus_balanced_sign"):
            fits.append(
                {
                    "policy_seed": seed,
                    "objective": objective,
                    "balanced_accuracy": 0.7,
                    "matthews_correlation": 0.4,
                    "mae_better_than_constant": True,
                    "balanced_accuracy_cluster_ci": {"low": 0.6, "high": 0.8},
                }
            )
    result = interpret_pairwise_fits(fits, settings=settings, quick=False)

    assert result["status"] == "INDIRECT_ABSOLUTE_Q_DIFFERENCING_BOTTLENECK"
    assert result["full_rl_authorized"] is False
    assert result["s4d3_authorized"] is False
