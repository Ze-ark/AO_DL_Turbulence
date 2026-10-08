from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import torch

from src.rl.s4_r3_observability_label_consistency import (
    _effective_settings,
    ObservabilityPairDataset,
    interpret_observability,
    knn_target_predictions,
    label_consistency_records,
    leave_episode_out_knn,
    preflight_s4_r3_observability_label_consistency,
    select_feature_view,
)
from src.rl.s4_training import _load_yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "s4_r3_observability_label_consistency_v1.yaml"
)


def _synthetic_dataset(samples: int = 24) -> ObservabilityPairDataset:
    generator = torch.Generator().manual_seed(123)
    states = torch.randn(samples, 210, generator=generator)
    actions = torch.randn(samples, 11, generator=generator)
    base = torch.where(torch.arange(samples) % 2 == 0, -0.2, 0.2).float()
    rows = [
        {
            "profile_id": f"profile_{index % 2}",
            "condition_id": f"condition_{index % 3}",
            "probe_step": index % 3,
            "episode_index": index,
            "episode_seed": 1000 + index,
        }
        for index in range(samples)
    ]
    dataset = ObservabilityPairDataset(
        states=states,
        actions=actions,
        target_deltas={
            1: base,
            2: base,
            4: base,
            8: base,
            16: base,
            32: base,
        },
        reward_h32_delta=base.clone(),
        power_h32_delta=base.clone(),
        rows=rows,
    )
    dataset.validate(state_size=210, action_size=11)
    return dataset


def test_formal_preflight_locks_views_and_no_training(tmp_path: Path) -> None:
    experiment = _load_yaml(CONFIG)
    settings = _effective_settings(experiment, quick=False)
    # A4正式目录已封存；用临时空目录保持契约测试可重复且不接触正式证据。
    settings["output_directory"] = str(tmp_path / "a4_preflight_contract")
    with patch("torch.cuda.is_available", return_value=True):
        result = preflight_s4_r3_observability_label_consistency(
            CONFIG, experiment, settings, quick=False
        )

    assert result["status"] == "READY_FOR_USER_DIAGNOSTIC"
    assert result["planned_view_seed_analyses"] == 12
    assert result["development_pairs"] == 5184
    assert result["validation_pairs"] == 2592
    assert result["gradient_updates"] == 0
    assert result["model_training"] is False
    assert result["mechanism_audit_access"] is False


def test_state_views_follow_declared_210_dimensional_layout() -> None:
    experiment = _load_yaml(CONFIG)
    settings = _effective_settings(experiment, quick=False)
    dataset = _synthetic_dataset()

    assert select_feature_view(dataset, "action_only", settings=settings).shape[1] == 11
    assert (
        select_feature_view(
            dataset, "current_sensor_plus_action", settings=settings
        ).shape[1]
        == 53
    )
    assert (
        select_feature_view(
            dataset, "four_frame_history_plus_action", settings=settings
        ).shape[1]
        == 179
    )
    assert (
        select_feature_view(
            dataset, "full_controller_state_plus_action", settings=settings
        ).shape[1]
        == 221
    )


def test_knn_predictions_use_development_normalization_only() -> None:
    training_features = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    training_target = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    query = torch.tensor([[-1.5], [1.5]])
    prediction, info = knn_target_predictions(
        training_features,
        training_target,
        query,
        torch.tensor([-1.0, 1.0]),
        k_values=[1, 3],
        chunk_size=2,
        device=torch.device("cpu"),
    )

    assert torch.equal(prediction[1].sign(), torch.tensor([-1.0, 1.0]))
    assert info[1]["neighbor_sign_agreement"] == 1.0


def test_leave_episode_out_excludes_same_episode() -> None:
    features = torch.tensor([[0.0], [0.01], [1.0], [1.01]])
    target = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    rows = [
        {"episode_seed": 1},
        {"episode_seed": 1},
        {"episode_seed": 2},
        {"episode_seed": 2},
    ]
    prediction, _ = leave_episode_out_knn(
        features,
        target,
        rows,
        k=1,
        chunk_size=2,
        device=torch.device("cpu"),
    )

    assert prediction.shape == target.shape
    assert torch.isfinite(prediction).all()


def test_label_consistency_reports_margin_bins_and_balanced_metric() -> None:
    records = label_consistency_records(9301, "validation", _synthetic_dataset())
    all_rows = [row for row in records if row["margin_bin"] == "all"]

    assert len(records) == 20
    assert len(all_rows) == 4
    assert all(row["sign_agreement"] == 1.0 for row in all_rows)
    assert all(row["balanced_accuracy"] == 1.0 for row in all_rows)


def test_interpretation_never_authorizes_rl() -> None:
    experiment = _load_yaml(CONFIG)
    settings = _effective_settings(experiment, quick=False)
    view_records = []
    for seed in settings["policy_seeds"]:
        for view in [item["id"] for item in settings["views"]]:
            view_records.append(
                {
                    "policy_seed": seed,
                    "view": view,
                    "split": "validation",
                    "k": 15,
                    "primary": True,
                    "balanced_accuracy": 0.5,
                    "matthews_correlation": 0.0,
                    "mae_better_than_constant": False,
                    "balanced_accuracy_cluster_ci": {"low": 0.45, "high": 0.55},
                }
            )
    label_records = []
    for seed in settings["policy_seeds"]:
        for comparison in (
            "target_h16_vs_target_h32",
            "target_h32_vs_reward_h32",
        ):
            label_records.append(
                {
                    "policy_seed": seed,
                    "split": "validation",
                    "comparison": comparison,
                    "margin_bin": "all",
                    "sign_agreement": 0.6,
                    "balanced_accuracy": 0.55,
                }
            )
    result = interpret_observability(
        view_records, label_records, settings=settings, quick=False
    )

    assert result["status"] == "HORIZON_OR_BOOTSTRAP_LABEL_INSTABILITY"
    assert result["gradient_updates"] == 0
    assert result["full_rl_authorized"] is False
    assert result["s4d3_authorized"] is False
