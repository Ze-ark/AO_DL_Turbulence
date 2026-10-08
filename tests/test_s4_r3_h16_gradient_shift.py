"""A8-D2小型确定性CPU测试；不产生正式科学结果。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

from src.rl.s4_r3_h16_gradient_shift import (
    FAMILIES,
    TARGET_FIELDS,
    _build_descriptors,
    _condition_family,
    _effective_settings,
    descriptor_drift,
    interpret_gradient_shift,
    matched_gap_interval,
    nearest_matches,
    preflight_gradient_shift,
    two_sample_interval,
)
from src.rl.s4_r3_h16_reward_pairwise_probe import RewardPairDataset
from src.rl.s4_training import _load_yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/s4_r3_h16_gradient_shift_v1.yaml"


def _stat_rows(offset: float, *, n: int = 4) -> list[dict]:
    rows = []
    for family_index, family in enumerate(FAMILIES):
        for i in range(n):
            rows.append({"condition_family": family, "episode_seed": family_index * 100 + i,
                         "gradient_cosine": offset + family_index * 0.01 + i * 0.001})
    return rows


def _dataset() -> RewardPairDataset:
    features, rewards, powers, rows = [], [], [], []
    generator = torch.Generator().manual_seed(123)
    for family_index, family in enumerate(FAMILIES):
        for episode in range(2):
            for pair in range(18):
                features.append(torch.randn(5, generator=generator) + family_index)
                rewards.append(0.2 if pair % 3 == 0 else -0.1)
                powers.append(0.1 if pair % 2 == 0 else -0.05)
                rows.append({"condition_id": f"tiny_{family}", "episode_seed": family_index * 10 + episode,
                             "episode_index": episode, "profile_id": f"p{pair % 6}",
                             "probe_step": [0, 80, 160][pair % 3]})
    return RewardPairDataset(torch.stack(features), torch.tensor(rewards), torch.tensor(powers), rows)


def test_cli_help_describes_read_only_scope() -> None:
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/diagnose_s4_r3_h16_gradient_shift.py"), "--help"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True,
    )
    assert "不训练" in result.stdout
    assert "只读核对" in result.stdout


def test_condition_family_is_strict() -> None:
    assert _condition_family("r3d2_val_boiling") == "boiling"
    assert _condition_family("r3d2_dev_combined") == "combined"
    with pytest.raises(RuntimeError, match="unknown"):
        _condition_family("other")


def test_episode_descriptors_use_complete_episodes() -> None:
    data = _dataset()
    mean, scale = data.features.double().mean(0), data.features.double().std(0).clamp_min(1e-12)
    selected = {(family, family_index * 10 + episode) for family_index, family in enumerate(FAMILIES)
                for episode in range(2)}
    rows, centroids = _build_descriptors(data, policy_seed=1, split="training",
                                         feature_mean=mean, feature_scale=scale,
                                         selected=selected, device=torch.device("cpu"))
    assert len(rows) == 6 and centroids.shape == (6, 5)
    assert all(row["pairs"] == 18 and row["positive_fraction"] == pytest.approx(1 / 3) for row in rows)
    assert set(TARGET_FIELDS).issubset(rows[0])


def test_two_sample_interval_reports_validation_minus_training() -> None:
    training, validation = _stat_rows(-0.4), _stat_rows(0.4)
    result = two_sample_interval(training, validation, "gradient_cosine", replicates=200,
                                 seed=7, device=torch.device("cpu"), familywise_alpha=0.05, family_size=24)
    assert result["estimate"] == pytest.approx(0.8)
    assert result["familywise_ci_low"] > 0.79
    assert result["paired"] is False


def test_descriptor_drift_and_nearest_matching_are_deterministic() -> None:
    training_rows = _stat_rows(0, n=4)
    validation_rows = _stat_rows(0, n=2)
    training = torch.tensor([[i, i % 3 + 0.1] for i in range(12)], dtype=torch.float64)
    validation = torch.tensor([[i + 0.5, i % 2 + 0.2] for i in range(6)], dtype=torch.float64)
    drift = descriptor_drift(training, validation, training_rows, validation_rows,
                             names=["a", "b"], replicates=200, seed=10,
                             device=torch.device("cpu"), familywise_alpha=0.05, family_size=24,
                             permutation_family_size=6)
    assert drift["dimensions"] == 2
    assert set(drift["descriptors"]) == {"a", "b"}
    assert 0 < drift["permutation_test"]["p_value"] <= 1
    assert drift["permutation_test"]["familywise_alpha"] == pytest.approx(0.05 / 6)
    mapping, quality = nearest_matches(training, validation, training_rows, validation_rows)
    assert len(mapping) == 6 and quality["validation_episodes"] == 6
    assert 0 < quality["unique_match_fraction"] <= 1


def test_matched_gap_uses_descriptor_pairs_not_physical_pairing() -> None:
    training, validation = _stat_rows(-0.2, n=4), _stat_rows(0.3, n=2)
    mapping = {}
    for family_index, family in enumerate(FAMILIES):
        for i in range(2):
            mapping[family, family_index * 100 + i] = family, family_index * 100 + i
    result = matched_gap_interval(training, validation, mapping, replicates=200, seed=11,
                                  device=torch.device("cpu"), familywise_alpha=0.05, family_size=24)
    assert result["estimate"] == pytest.approx(0.5)
    assert result["paired"] == "descriptor_matched_not_physical_pair"


def _interpretation_inputs() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    shifts = [{"policy_seed": seed, "arm": arm,
               "gradient_cosine_shift": {"familywise_ci_low": 0.3}}
              for seed in (9301, 9302, 9303) for arm in ("shared", "split")]
    target = [{"policy_seed": seed, "rms_standardized_drift": {"estimate": 0.3},
               "permutation_test": {"p_value": 0.001}}
              for seed in (9301, 9302, 9303)]
    feature = [{"policy_seed": seed, "rms_standardized_drift": {"estimate": 0.1},
                "permutation_test": {"p_value": 0.5}}
               for seed in (9301, 9302, 9303)]
    matches = [{"policy_seed": seed, "arm": arm, "method": method,
                "absolute_gap_closure": 0.6 if method == "target" else 0.1,
                "unique_match_fraction": 0.75, "maximum_reuse_count": 2}
               for seed in (9301, 9302, 9303) for arm in ("shared", "split")
               for method in ("target", "feature")]
    return shifts, target, feature, matches


def test_interpretation_requires_reversal_drift_closure_and_stability() -> None:
    shifts, target, feature, matches = _interpretation_inputs()
    result = interpret_gradient_shift(shifts, target, feature, matches,
                                      policy_seeds=[9301, 9302, 9303], meaningful_cosine_shift=0.2,
                                      meaningful_drift=0.25, minimum_supporting_seeds=2,
                                      minimum_gap_closure=0.5, minimum_unique_fraction=0.5,
                                      maximum_reuse=4, distribution_permutation_alpha=0.05 / 6,
                                      quick=False)
    assert result["status"] == "REVERSAL_ASSOCIATED_WITH_TARGET_DISTRIBUTION"
    assert result["target_supporting_seeds"] == [9301, 9302, 9303]
    assert result["causal_explanation_authorized"] is False
    matches[0]["maximum_reuse_count"] = 8
    matches[4]["maximum_reuse_count"] = 8
    result = interpret_gradient_shift(shifts, target, feature, matches,
                                      policy_seeds=[9301, 9302, 9303], meaningful_cosine_shift=0.2,
                                      meaningful_drift=0.25, minimum_supporting_seeds=2,
                                      minimum_gap_closure=0.5, minimum_unique_fraction=0.5,
                                      maximum_reuse=4, distribution_permutation_alpha=0.05 / 6,
                                      quick=False)
    assert result["target_distribution_associated_candidate"] is False


def test_formal_preflight_is_read_only_and_excludes_test_samples(tmp_path: Path) -> None:
    experiment = _load_yaml(CONFIG)
    settings = _effective_settings(experiment, quick=False)
    settings["output_directory"] = str(tmp_path / "formal")
    with patch("torch.cuda.is_available", return_value=True):
        result = preflight_gradient_shift(CONFIG, experiment, settings, quick=False)
    assert result["status"] == "READY_FOR_USER_DIAGNOSTIC"
    assert result["expected_descriptor_records"] == 1296
    assert result["expected_gradient_shift_comparisons"] == 6
    assert result["expected_matching_comparisons"] == 12
    assert result["independent_test_sample_paths"] == []
    assert result["checkpoint_loaded"] is False and result["gradient_recomputed"] is False
    assert not (tmp_path / "formal").exists()


@pytest.mark.parametrize("field", ["allow_checkpoint_loading", "allow_gradient_recomputation", "allow_training",
                                     "allow_optimizer_updates", "allow_full_rl_training", "allow_real_hardware_actions"])
def test_scope_expansion_is_rejected(field: str, tmp_path: Path) -> None:
    experiment = deepcopy(_load_yaml(CONFIG))
    experiment["metadata"][field] = True
    settings = _effective_settings(experiment, quick=True)
    settings["output_directory"] = str(tmp_path / "guard")
    with patch("torch.cuda.is_available", return_value=True), pytest.raises(RuntimeError, match="safety flag"):
        preflight_gradient_shift(CONFIG, experiment, settings, quick=True)
