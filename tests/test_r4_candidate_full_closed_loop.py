"""冻结重选完整闭环的CPU统计与预算测试。"""
import pytest
import torch

from src.rl.r4_candidate_full_closed_loop import budget, summarize


def fixture() -> torch.Tensor:
    values = torch.ones(3, 3, 6, 32, 6, dtype=torch.float64) * .5
    values[..., 2] = 0
    values[..., 4] = 0
    values[..., 5] = 0
    return values


def counts(nonzero: bool = True) -> torch.Tensor:
    result = torch.zeros(5, dtype=torch.long)
    result[1 if nonzero else 0] = 3 * 6 * 32 * 200
    return result


def test_budgets_are_frozen():
    assert budget(True) == {
        "physical_transitions": 72,
        "max_model_forward_samples": 592704,
    }
    assert budget(False) == {
        "physical_transitions": 345600,
        "max_model_forward_samples": 2844979200,
    }


def test_positive_two_percent_safe_effect_passes_but_never_authorizes_rl():
    values = fixture()
    values[1, ..., 0] += .005
    values[2, ..., 0] += .01
    values[2, ..., 1] += .01
    values[2, ..., 3] -= .01
    first = summarize(values, counts(), seed=7, repeats=40)
    second = summarize(values, counts(), seed=7, repeats=40)
    assert first == second
    assert first["all_confirmation_gates_pass"] is True
    assert first["relative_power_gain"] == pytest.approx(.02)
    assert first["full_closed_loop_performance_established"] is True
    assert first["rl_authorized"] is False


@pytest.mark.parametrize("change", ["small_power", "strehl", "violation",
                                     "phase", "saturation", "slew", "zero_action"])
def test_each_scientific_or_safety_gate_can_fail(change: str):
    values = fixture()
    values[2, ..., 0] += .01
    values[2, ..., 1] += .01
    values[2, ..., 3] -= .01
    action_counts = counts()
    if change == "small_power":
        values[2, ..., 0] = values[0, ..., 0] + .004
    elif change == "strehl":
        values[2, ..., 1] = values[0, ..., 1] - .001
    elif change == "violation":
        values[2, ..., 2] = .002
    elif change == "phase":
        values[2, ..., 3] = values[0, ..., 3] + .001
    elif change == "saturation":
        values[2, ..., 4] = .002
    elif change == "slew":
        values[2, ..., 5] = .002
    else:
        action_counts = counts(False)
    assert summarize(values, action_counts, seed=8, repeats=40)[
        "all_confirmation_gates_pass"] is False


def test_one_negative_family_fails_robustness_gate():
    values = fixture()
    values[2, ..., 0] += .02
    values[2, 0, ..., 0] = values[0, 0, ..., 0] - .001
    result = summarize(values, counts(), seed=9, repeats=40)
    assert result["confirmation_gates"]["all_family_power_means_positive"] is False


def test_incomplete_or_nonfinite_inputs_stop():
    values = fixture()
    with pytest.raises(ValueError):
        summarize(values[..., :31, :], counts(), seed=1, repeats=10)
    values[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        summarize(values, counts(), seed=1, repeats=10)
