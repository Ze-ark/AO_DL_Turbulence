"""候选重选的CPU确定性测试；不读取正式结果。"""
import torch

from src.rl.r4_candidate_reselection import (
    analyze_reselection, crossfit_margin_gate, stratified_fold_assignment,
    tune_robust_margin_gate,
)


def test_stratified_folds_are_balanced_and_deterministic():
    first = stratified_fold_assignment(seed=17)
    second = stratified_fold_assignment(seed=17)
    assert torch.equal(first, second)
    assert first.shape == (3, 32)
    for family in range(3):
        assert torch.equal(torch.bincount(first[family]), torch.full((4,), 8))


def test_robust_gate_can_prefer_zero_when_every_action_is_harmful():
    score = torch.zeros(3, 6, 32, 4, 5, dtype=torch.float64)
    score[..., 1] = 1
    actual = torch.zeros_like(score)
    actual[..., 1] = -1
    train = torch.ones(3, 32, dtype=torch.bool)
    result = tune_robust_margin_gate(score, actual, train)
    assert result["threshold"] > 1
    assert result["train_mean_gain"] == 0


def test_crossfit_uses_every_weather_once_as_test():
    score = torch.zeros(3, 6, 32, 4, 5, dtype=torch.float64)
    score[..., 1] = 1
    actual = score.clone()
    assignment = stratified_fold_assignment(seed=5)
    choices, records = crossfit_margin_gate(score, actual, assignment)
    assert choices.shape == (3, 6, 32, 4)
    assert bool((choices == 1).all())
    assert len(records) == 4
    assert all(record["train_weather"] == 72 for record in records)
    assert all(record["test_weather"] == 24 for record in records)


def test_full_analysis_is_deterministic_and_never_authorizes_rl():
    measured = torch.zeros(3, 6, 32, 4, 5, dtype=torch.float64)
    audit = measured.clone()
    measured[..., 1] = 1
    audit[..., 1] = 1
    members = measured[..., None].expand(3, 6, 32, 4, 5, 3).clone()
    metrics = torch.zeros(3, 6, 32, 4, 5, 4, dtype=torch.float64)
    metrics[..., 1, 0] = 1
    metrics[..., 1, 1] = 1
    metrics[..., 1, 3] = -1
    first, saved = analyze_reselection(
        members, measured, audit, metrics, disagreement=1, folds=4,
        split_seed=3, bootstrap_seed=4, bootstrap_repeats=20)
    second, _ = analyze_reselection(
        members, measured, audit, metrics, disagreement=1, folds=4,
        split_seed=3, bootstrap_seed=4, bootstrap_repeats=20)
    assert first == second
    assert first["rl_authorized"] is False
    assert first["independent_confirmation_access"] is False
    assert first["nominated_rules"] == ["conservative_safe_gate", "ensemble_mean_safe_gate"]
    assert set(saved["choices"]) == {
        "zero", "original_selected", "conservative_argmax", "ensemble_mean_argmax",
        "conservative_safe_gate", "ensemble_mean_safe_gate", "oracle_upper_bound"}

