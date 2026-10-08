"""闭环候选排序诊断的CPU确定性单元测试；不产生性能结论。"""
import pytest
import torch

from src.rl.r4_closed_loop_ranking import (
    CANDIDATES, budget, candidate_sequences, discounted_objective,
    ranking_metrics, require_score_replay_close, summarize,
)
from src.rl.r4_mpc import SearchConfig


def test_budget_is_frozen():
    assert budget(False) == {"physical_transitions": 195840, "model_forward_samples": 28588032}
    assert budget(True) == {"physical_transitions": 1920, "model_forward_samples": 397056}


def test_candidate_panel_order_and_clipping():
    selected = torch.linspace(-1, 1, 2 * 8 * 11).reshape(2, 8, 11)
    before = selected.clone()
    panel = candidate_sequences(selected)
    assert panel.shape == (2, len(CANDIDATES), 8, 11)
    assert torch.equal(panel[:, 0], torch.zeros_like(selected))
    assert torch.equal(panel[:, 1], -selected)
    assert torch.equal(panel[:, 2], selected * .5)
    assert torch.equal(panel[:, 3], selected)
    assert torch.equal(panel[:, 4], (selected * 1.5).clamp(-1, 1))
    assert torch.equal(selected, before)


@pytest.mark.parametrize("bad", [torch.zeros(2, 7, 11), torch.zeros(2, 8, 10),
                                  torch.full((2, 8, 11), 1.01),
                                  torch.full((2, 8, 11), float("nan"))])
def test_candidate_panel_rejects_invalid_input(bad):
    with pytest.raises(ValueError):
        candidate_sequences(bad)


def test_discounted_objective_matches_manual_sum():
    config = SearchConfig(horizon=8, discount=1., action_cost=.25)
    sequences = torch.zeros(2, 5, 8, 11)
    sequences[:, 3] = 1
    power = torch.ones(2, 5, 8)
    result = discounted_objective(power, sequences, config)
    assert torch.equal(result[:, 0], torch.full((2,), 8.))
    assert torch.equal(result[:, 3], torch.full((2,), 6.))


def test_discounted_objective_rejects_device_mismatch_when_cuda_exists():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available for device-boundary test")
    with pytest.raises(ValueError):
        discounted_objective(torch.ones(1, 5, 8, device="cuda"),
                             torch.zeros(1, 5, 8, 11), SearchConfig())


def test_ranking_metrics_perfect_and_reverse():
    model = torch.tensor([[0., 1., 2., 3., 4.]])
    perfect = ranking_metrics(model, model)
    reverse = ranking_metrics(model, -model)
    assert perfect["pairwise_accuracy"].item() == 1
    assert perfect["top1_match"].item() == 1
    assert perfect["regret"].item() == 0
    assert reverse["pairwise_accuracy"].item() == 0
    assert reverse["top1_match"].item() == 0
    assert reverse["regret"].item() == 4


def test_score_replay_tolerance_accepts_observed_float32_batch_drift():
    expected = torch.tensor([5.6888346672058105])
    observed = torch.tensor([5.688836097717285])
    assert require_score_replay_close(observed, expected) == pytest.approx(1.430511474609375e-6)
    with pytest.raises(RuntimeError, match="score replay drift"):
        require_score_replay_close(expected + 1e-3, expected)


def test_summary_uses_weather_level_and_is_deterministic():
    base = torch.arange(5, dtype=torch.float64)
    measured = base.expand(3, 6, 32, 4, 5).clone()
    audit = measured.clone()
    members = measured[..., None].expand(3, 6, 32, 4, 5, 3).clone()
    result = summarize(members, measured, audit, disagreement=1., seed=11, repeats=100)
    assert result == summarize(members, measured, audit, disagreement=1., seed=11, repeats=100)
    assert result["independent_weather"] == 96
    assert result["state_comparisons"] == 2304
    assert result["statistics"]["conservative_vs_audit"]["pairwise_accuracy"] == {
        "mean": 1.0, "ci95": [1.0, 1.0]}
    assert result["rl_authorized"] is False
    with pytest.raises(ValueError):
        summarize(members[:, :, :31], measured[:, :, :31], audit[:, :, :31],
                  disagreement=1., seed=11, repeats=100)
