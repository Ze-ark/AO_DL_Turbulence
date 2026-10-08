"""A8-D1小型确定性CPU测试；不产生正式诊断或性能结论。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

from src.rl.s4_r3_h16_gradient_conflict import (
    _effective_settings,
    _episode_gradient,
    _model_digest,
    gradient_relation,
    interpret_gradient_conflict,
    paired_interval,
    preflight_gradient_conflict,
    stratified_interval,
)
from src.rl.s4_r3_h16_head_split import HeadSplitProbe
from src.rl.s4_r3_h16_reward_pairwise_probe import RewardPairDataset
from src.rl.s4_training import _load_yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/s4_r3_h16_gradient_conflict_v1.yaml"


def _rows(value: float, *, shift: float = 0.0) -> list[dict]:
    return [{"condition_id": f"c{i // 4}", "episode_seed": i, "metric": value + shift + i * 0.01}
            for i in range(8)]


def _tiny_pairs() -> RewardPairDataset:
    generator = torch.Generator().manual_seed(77)
    features = torch.randn(18, 221, generator=generator)
    target = torch.where(torch.arange(18) % 3 == 0, torch.tensor(0.2), torch.tensor(-0.1))
    rows = [{"condition_id": "tiny", "episode_seed": 100, "episode_index": 0,
             "profile_id": f"p{i % 2}", "probe_step": [0, 80, 160][i % 3]} for i in range(18)]
    return RewardPairDataset(features, target, target.clone(), rows)


def test_cli_help_is_utf8_and_does_not_run_diagnostic() -> None:
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/diagnose_s4_r3_h16_gradient_conflict.py"), "--help"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True,
    )
    assert "不训练" in result.stdout
    assert "只读核对" in result.stdout


def test_gradient_relation_known_directions_and_weight() -> None:
    same = gradient_relation([torch.tensor([1.0, 0.0])], [torch.tensor([2.0, 0.0])], classification_weight=0.25)
    opposed = gradient_relation([torch.tensor([1.0, 0.0])], [torch.tensor([-2.0, 0.0])], classification_weight=0.25)
    orthogonal = gradient_relation([torch.tensor([1.0, 0.0])], [torch.tensor([0.0, 3.0])], classification_weight=0.25)
    assert same["gradient_cosine"] == pytest.approx(1)
    assert opposed["gradient_cosine"] == pytest.approx(-1)
    assert orthogonal["gradient_cosine"] == pytest.approx(0)
    assert same["weighted_classification_to_regression_norm_ratio"] == pytest.approx(0.5)
    with pytest.raises(RuntimeError, match="zero"):
        gradient_relation([torch.zeros(2)], [torch.ones(2)], classification_weight=0.25)


def test_episode_gradient_is_read_only_and_uses_shared_trunk() -> None:
    model = HeadSplitProbe(221, 16, "split")
    data = _tiny_pairs()
    checkpoint = {"normalization": {"feature_mean": data.features.mean(dim=0),
                                    "feature_scale": data.features.std(dim=0).clamp_min(1e-3),
                                    "target_scale": data.reward_delta.std().clamp_min(1e-3)}}
    before = _model_digest(model)
    metrics = _episode_gradient(model, data, list(range(18)), checkpoint,
                                positive_weight=torch.tensor(2.0), classification_weight=0.25,
                                huber_delta=1.0, device=torch.device("cpu"))
    assert -1 <= metrics["gradient_cosine"] <= 1
    assert set(["network_0_gradient_cosine", "network_2_gradient_cosine"]).issubset(metrics)
    assert _model_digest(model) == before
    assert all(parameter.grad is None for parameter in model.parameters())


def test_stratified_and_paired_intervals_use_episode_clusters() -> None:
    interval = stratified_interval(_rows(1.0), "metric", replicates=1000, seed=1,
                                   familywise_alpha=0.05, family_size=20)
    assert interval["clusters"] == 8
    assert interval["unit"] == "complete_episode_seed"
    paired = paired_interval(_rows(1.0), _rows(1.0, shift=0.2), "metric", replicates=1000, seed=2,
                             familywise_alpha=0.05, family_size=20)
    assert paired["estimate"] == pytest.approx(0.2)
    assert paired["familywise_ci_low"] == pytest.approx(0.2)
    duplicated = _rows(1.0)
    duplicated.append(dict(duplicated[0]))
    with pytest.raises(RuntimeError, match="duplicate"):
        stratified_interval(duplicated, "metric", replicates=100, seed=1,
                            familywise_alpha=0.05, family_size=20)


def _synthetic_summaries(conflict: bool) -> tuple[list[dict], list[dict], list[dict]]:
    summaries = []
    for seed in (9301, 9302, 9303):
        for arm in ("shared", "split"):
            for split in ("training", "validation"):
                summaries.append({"policy_seed": seed, "arm": arm, "split": split,
                                  "gradient_cosine": {"familywise_ci_high": -0.1 if conflict else 0.1},
                                  "conflict_fraction": {"familywise_ci_low": 0.6 if conflict else 0.4}})
    heads = [{"policy_seed": seed, "split": split,
              "split_minus_shared_cosine": {"familywise_ci_low": 0.06}}
             for seed in (9301, 9302, 9303) for split in ("training", "validation")]
    failures = [{"split": split, "failed_mean_minus_reference_cosine": {"familywise_ci_high": -0.06}}
                for split in ("training", "validation")]
    return summaries, heads, failures


def test_interpretation_requires_all_preregistered_conditions() -> None:
    summaries, heads, failures = _synthetic_summaries(True)
    result = interpret_gradient_conflict(summaries, heads, failures, reference_seed=9301,
                                         failure_seeds=[9302, 9303], meaningful_difference=0.05, quick=False)
    assert result["status"] == "SHARED_TRUNK_CONFLICT_AND_FAILURE_PATTERN_SUPPORTED"
    assert result["independent_trunk_experiment_design_authorized"] is True
    assert result["full_rl_authorized"] is False
    summaries, heads, failures = _synthetic_summaries(False)
    result = interpret_gradient_conflict(summaries, heads, failures, reference_seed=9301,
                                         failure_seeds=[9302, 9303], meaningful_difference=0.05, quick=False)
    assert result["status"] == "SHARED_TRUNK_CONFLICT_NOT_CONFIRMED"
    assert result["independent_trunk_experiment_design_authorized"] is False


def test_formal_preflight_is_read_only_and_excludes_independent_test(tmp_path: Path) -> None:
    experiment = _load_yaml(CONFIG)
    settings = _effective_settings(experiment, quick=False)
    settings["output_directory"] = str(tmp_path / "formal")
    with patch("torch.cuda.is_available", return_value=True):
        result = preflight_gradient_conflict(CONFIG, experiment, settings, quick=False)
    assert result["status"] == "READY_FOR_USER_DIAGNOSTIC"
    assert result["expected_episode_records"] == 2592
    assert result["selected_episodes_per_split"] == {"training": 384, "validation": 48}
    assert result["independent_test_sample_paths"] == []
    assert result["optimizer_updates"] == 0 and result["training_run"] is False
    assert not (tmp_path / "formal").exists()


@pytest.mark.parametrize("field", ["allow_training", "allow_optimizer_updates", "allow_full_rl_training",
                                     "allow_s4d3_access", "allow_real_hardware_actions"])
def test_scope_expansion_is_rejected(field: str, tmp_path: Path) -> None:
    experiment = _load_yaml(CONFIG)
    experiment["metadata"][field] = True
    settings = _effective_settings(experiment, quick=True)
    settings["output_directory"] = str(tmp_path / "guard")
    with patch("torch.cuda.is_available", return_value=True), pytest.raises(RuntimeError, match="safety flag"):
        preflight_gradient_conflict(CONFIG, experiment, settings, quick=True)


def test_cpu_runtime_is_forbidden(tmp_path: Path) -> None:
    experiment = deepcopy(_load_yaml(CONFIG))
    experiment["runtime"]["device"] = "cpu"
    settings = _effective_settings(experiment, quick=True)
    settings["output_directory"] = str(tmp_path / "guard")
    with patch("torch.cuda.is_available", return_value=True), pytest.raises(RuntimeError, match="CUDA"):
        preflight_gradient_conflict(CONFIG, experiment, settings, quick=True)
