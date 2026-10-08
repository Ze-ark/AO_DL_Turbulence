"""并列感知汇总的CPU确定性测试；不读取正式结果。"""
import torch

from src.rl.r4_closed_loop_ranking_finalize import summarize_tie_aware, tie_aware_metrics


def test_all_tied_state_is_explicitly_undefined_for_ranking():
    score = torch.zeros(2, 5)
    result = tie_aware_metrics(score, score)
    assert torch.equal(result["pairwise_comparable"], torch.zeros(2))
    assert not bool(result["top1_defined"].any())
    assert not bool(result["regret_defined"].any())
    assert not bool(result["selected_zero_sign_defined"].any())
    assert torch.equal(result["selected_minus_zero"], torch.zeros(2))
    assert torch.equal(result["oracle_minus_zero"], torch.zeros(2))


def test_tie_aware_summary_reports_coverage_and_is_deterministic():
    base = torch.arange(5, dtype=torch.float64)
    measured = base.expand(3, 6, 32, 4, 5).clone()
    audit = measured.clone()
    members = measured[..., None].expand(3, 6, 32, 4, 5, 3).clone()
    # 28个状态全部并列，但每个天气仍有其他有效状态。
    flat_measured = measured.reshape(-1, 5); flat_audit = audit.reshape(-1, 5)
    flat_members = members.reshape(-1, 5, 3)
    flat_measured[:28] = 0; flat_audit[:28] = 0; flat_members[:28] = 0
    result = summarize_tie_aware(members, measured, audit,
        disagreement=1., seed=7, repeats=100)
    assert result == summarize_tie_aware(members, measured, audit,
        disagreement=1., seed=7, repeats=100)
    assert result["all_tied_uninformative_states"] == 28
    assert result["informative_pairwise_states"] == 2276
    coverage = result["statistics"]["conservative_vs_audit"]["pairwise_accuracy"]["coverage"]
    assert coverage == {"defined_states": 2276, "total_states": 2304,
                        "comparable_pairs": 22760, "total_pairs": 23040}
    assert result["statistics"]["conservative_vs_audit"]["pairwise_accuracy"]["mean"] == 1.
    assert result["rl_authorized"] is False
