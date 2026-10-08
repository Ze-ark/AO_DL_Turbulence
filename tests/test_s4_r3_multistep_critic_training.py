from __future__ import annotations

from pathlib import Path

import torch

from src.rl.s4_r3_multistep_critic_repair import CriticTargetSpec
from src.rl.s4_r3_multistep_critic_training import (
    ProbeDataset,
    _clustered_t_distribution,
    _effective_settings,
    _initial_critic_states,
    fit_frozen_critic_pair,
    preflight_s4_r3_multistep_critic_training,
)
from src.rl.s4_training import _load_yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "s4_r3_multistep_critic_repair_v1.yaml"
)


def _synthetic_dataset(samples: int = 8) -> ProbeDataset:
    generator = torch.Generator().manual_seed(123)
    states = torch.randn(samples, 210, generator=generator)
    actions = torch.zeros(samples, 11)
    rows = []
    for index in range(samples):
        candidate = "zero" if index % 2 == 0 else "actor"
        actions[index].fill_(0.0 if candidate == "zero" else 0.2)
        rows.append(
            {
                "split": "synthetic",
                "arm": "student_backbone",
                "policy_seed": 9301,
                "student_id": "synthetic",
                "profile_id": "nominal",
                "condition_id": f"condition_{index // 4}",
                "probe_step": 0,
                "episode_index": (index // 2) % 2,
                "episode_seed": 1000 + index // 2,
                "candidate": candidate,
            }
        )
    base = 0.1 * states[:, :1] + 0.2 * actions[:, :1]
    curve = torch.cat([base + 0.01 * horizon for horizon in range(32)], dim=1)
    dataset = ProbeDataset(
        states=states,
        actions=actions,
        n_step_targets=curve.double(),
        empirical_reward_returns=(curve - 0.05).double(),
        empirical_power_returns=(curve - 0.02).double(),
        original_q=(curve[:, 0] + 0.3).float(),
        rows=rows,
    )
    dataset.validate(max_horizon=32)
    return dataset


def test_r3_d2a_formal_preflight_locks_counts_and_safety() -> None:
    experiment = _load_yaml(CONFIG)
    settings = _effective_settings(experiment, quick=False)
    result, _, checkpoints = preflight_s4_r3_multistep_critic_training(
        CONFIG, experiment, settings, quick=False
    )

    assert result["status"] == "READY_FOR_USER_TRAINING"
    assert len(checkpoints) == 3
    assert result["planned_critic_fits"] == 12
    assert result["expected_branch_rollouts"] == 972
    assert result["expected_samples"] == 18_144
    assert result["expected_horizon_labels"] == 580_608
    assert result["maximum_updates_per_fit"] == 20_000
    assert result["actor_updates"] == 0
    assert result["alpha_updates"] == 0
    assert result["sealed_s4d3_access"] is False
    assert result["real_slm_actions"] is False


def test_r3_d2a_quick_preflight_is_small_but_keeps_all_targets() -> None:
    experiment = _load_yaml(CONFIG)
    settings = _effective_settings(experiment, quick=True)
    result, _, checkpoints = preflight_s4_r3_multistep_critic_training(
        CONFIG, experiment, settings, quick=True
    )

    assert result["status"] == "READY_FOR_QUICK_SMOKE"
    assert len(checkpoints) == 1
    assert result["planned_critic_fits"] == 4
    assert result["expected_branch_rollouts"] == 12
    assert result["expected_samples"] == 24
    assert result["expected_horizon_labels"] == 768
    assert result["maximum_updates_per_fit"] == 8


def test_clustered_student_t_uses_trajectory_clusters() -> None:
    values = torch.tensor(
        [value for cluster in range(24) for value in (-0.2 - cluster * 0.001,) * 3],
        dtype=torch.float64,
    )
    cluster_ids = [f"cluster_{cluster}" for cluster in range(24) for _ in range(3)]
    result = _clustered_t_distribution(values, cluster_ids)

    assert result["clusters"] == 24
    assert result["raw_observations"] == 72
    assert result["ci95_high"] < 0
    assert result["two_sided_p_value"] < 0.001


def test_tiny_critic_fit_writes_checkpoint_without_actor_state(tmp_path: Path) -> None:
    experiment = _load_yaml(CONFIG)
    settings = _effective_settings(experiment, quick=True)
    settings["maximum_updates"] = 4
    settings["validation_interval_updates"] = 2
    settings["log_interval_updates"] = 2
    settings["early_stopping_patience_updates"] = 4
    dataset = _synthetic_dataset()
    initial = _initial_critic_states(
        settings=settings, policy_seed=9301, device=torch.device("cpu")
    )

    result = fit_frozen_critic_pair(
        training=dataset,
        validation=dataset,
        spec=CriticTargetSpec("n_step_16", "n_step", 16),
        settings=settings,
        policy_seed=9301,
        initial_states=initial,
        output_directory=tmp_path,
        device=torch.device("cpu"),
    )

    checkpoint = torch.load(
        ROOT / result["checkpoint"] if not Path(result["checkpoint"]).is_absolute() else result["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    assert result["updates_completed"] == 4
    assert checkpoint["actor_updates"] == 0
    assert checkpoint["alpha_updates"] == 0
    assert "actor" not in checkpoint

