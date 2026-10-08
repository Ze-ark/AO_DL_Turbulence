"""独立候选确认的CPU纯函数测试；不读取正式输出。"""
import torch

from src.rl.r4_candidate_confirmation import budget, frozen_gate_choices, summarize


def test_budget_matches_predeclared_formal_and_quick_counts():
    assert budget(False) == {
        "physical_transitions": 79488,
        "max_model_forward_samples": 640687104,
    }
    assert budget(True) == {
        "physical_transitions": 64,
        "max_model_forward_samples": 221424,
    }


def test_frozen_gate_uses_strict_threshold_and_eligibility():
    scores = torch.zeros(3, 5, 3, dtype=torch.float64)
    scores[0, 1] = .4
    scores[1, 2] = .3
    scores[2, 3] = .5
    choice, margin = frozen_gate_choices(
        scores, .3, torch.tensor([True, True, False]))
    assert choice.tolist() == [1, 0, 0]
    assert torch.allclose(margin, torch.tensor([.4, .3, .5], dtype=torch.float64))


def test_summary_passes_positive_safe_synthetic_result_and_never_authorizes_rl():
    objective = torch.zeros(3, 3, 6, 16, 4, dtype=torch.float64)
    measured = objective.clone()
    objective[2] = 1
    measured[2] = 1
    objective[1] = .5
    measured[1] = .5
    metrics = torch.zeros(3, 3, 6, 16, 4, 4, dtype=torch.float64)
    metrics[0, ..., 0] = 10
    metrics[1, ..., 0] = 10.5
    metrics[2, ..., 0] = 11
    metrics[2, ..., 1] = .1
    metrics[2, ..., 3] = -1
    choices = torch.ones(3, 6, 16, 4, dtype=torch.long)
    result = summarize(objective, measured, metrics, choices, seed=7, repeats=20)
    assert result["all_confirmation_gates_pass"] is True
    assert result["full_closed_loop_design_authorized"] is True
    assert result["full_closed_loop_performance_established"] is False
    assert result["rl_authorized"] is False


def test_summary_fails_when_objective_is_not_positive():
    objective = torch.zeros(3, 3, 6, 16, 4, dtype=torch.float64)
    metrics = torch.zeros(3, 3, 6, 16, 4, 4, dtype=torch.float64)
    metrics[..., 0] = 1
    choices = torch.zeros(3, 6, 16, 4, dtype=torch.long)
    result = summarize(objective, objective.clone(), metrics, choices, seed=8, repeats=20)
    assert result["all_confirmation_gates_pass"] is False
    assert result["full_closed_loop_design_authorized"] is False
