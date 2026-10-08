"""A10小型确定性CPU单元测试；不产生正式性能结果。"""
from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch
import yaml

from src.rl.s4_r3_h16_reward_pairwise_probe import RewardPairDataset
from src.rl.s4_r3_h16_veto_gate_confirmation import (
    actor_normalized_action_bound_violations, comparison_rows, episode_decisions, gate_decision,
)
from src.rl.s4_r3_h16_veto_gate_contract import (
    BOUNDARY, budget, effective_settings, load_contract, safe_path, selected_models,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/s4_r3_h16_veto_gate_confirmation_v1.yaml"
DESIGN = ROOT / "configs/experiments/s4_r3_h16_veto_gate_confirmation_design_v1.yaml"


def settings(*, quick=True):
    return load_contract(CONFIG, quick=quick)[1]


def tiny_dataset() -> RewardPairDataset:
    rows = []
    for condition, base in (("a10_confirmation_id_frozen", 10),
                            ("a10_confirmation_id_boiling", 20),
                            ("a10_confirmation_id_combined", 30)):
        for episode in range(2):
            for profile in ("p0", "p1"):
                for probe in (0, 4):
                    rows.append(dict(condition_id=condition, episode_seed=base + episode,
                                     episode_index=episode, profile_id=profile, probe_step=probe))
    n = len(rows)
    reward = torch.tensor(([.2, -.1, .3, -.2] * 6), dtype=torch.float32)
    power = reward * .5
    return RewardPairDataset(torch.zeros(n, 221), reward, power, rows)


def test_contract_and_budget_are_frozen():
    formal, quick = settings(quick=False), settings(quick=True)
    assert budget(formal) == dict(dataset_files=6, unique_weather_episodes=192,
                                  policy_episode_instances=576, paired_action_samples=10368,
                                  collection_branches=648, frozen_models=9,
                                  episode_decision_records=576,
                                  primary_comparison_records=9, shifted_comparison_records=9)
    assert budget(quick) == dict(dataset_files=2, unique_weather_episodes=12,
                                 policy_episode_instances=12, paired_action_samples=48,
                                 collection_branches=48, frozen_models=3,
                                 episode_decision_records=12,
                                 primary_comparison_records=3, shifted_comparison_records=3)
    assert set().union(*(set(range(c["base_seed"], c["base_seed"] + split["episodes_per_condition"]))
                         for split in formal["splits"].values() for c in split["conditions"])).isdisjoint(
        set().union(*(set(range(c["base_seed"], c["base_seed"] + split["episodes_per_condition"]))
                      for split in quick["splits"].values() for c in split["conditions"])))
    assert BOUNDARY["supervised_updates"] == 0 and not BOUNDARY["closed_loop_rollout"]


@pytest.mark.parametrize("change", ["cpu", "boundary", "design_only", "extra", "hash"])
def test_runtime_contract_fails_closed(tmp_path, change):
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if change == "cpu":
        cfg["runtime"]["device"] = "cpu"
    elif change == "boundary":
        cfg["boundary"]["critic_updates"] = 1
    elif change == "design_only":
        cfg = yaml.safe_load(DESIGN.read_text(encoding="utf-8"))
    elif change == "extra":
        cfg["threshold_override"] = 1
    else:
        cfg["design_sha256"] = "0" * 64
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_contract(path, quick=True)


def test_safe_path_rejects_escape():
    with pytest.raises(ValueError):
        safe_path("../outside")
    with pytest.raises(ValueError):
        safe_path(ROOT)


def test_gate_uses_mean_original_reward_and_strict_zero_threshold():
    values = torch.tensor([[1., -2., 0.], [-.5, 1., 0.], [1., 2., 0.]])
    mean, release = gate_decision(values, threshold=0.)
    assert torch.equal(mean, torch.tensor([.5, 1/3, 0.], dtype=torch.float64))
    assert torch.equal(release, torch.tensor([True, True, False]))
    with pytest.raises(RuntimeError, match="non-finite"):
        gate_decision(values * float("nan"), threshold=0.)
    with pytest.raises(ValueError):
        gate_decision(values[0], threshold=0.)


def test_episode_outcomes_keep_complete_episode_unit():
    data = tiny_dataset()
    prediction = data.reward_delta.double()
    release = prediction > 0
    rows = episode_decisions(policy_seed=9301, split="confirmation_id", data=data,
                             mean_prediction=prediction, release=release,
                             actor_normalized_bound_violation=torch.zeros(
                                 len(prediction), dtype=torch.bool))
    assert len(rows) == 6 and len({(r["condition_id"], r["episode_seed"]) for r in rows}) == 6
    assert all(r["pairs"] == 4 and r["actor_release_fraction"] == .5 for r in rows)
    assert all(r["gate_reward_vs_zero"] > 0 and r["gate_power_vs_zero"] > 0 for r in rows)
    assert all(r["gate_reward_vs_always_actor"] > 0 for r in rows)
    assert all(r["gate_normalized_bound_violation_rate"] == 0 for r in rows)


def test_action_bound_audit_aligns_raw_actor_rows(tmp_path):
    data = tiny_dataset()
    raw_rows, states, actions = [], [], []
    for row in reversed(data.rows):
        for candidate in ("zero", "actor"):
            raw_rows.append(dict(**row, candidate=candidate))
            states.append(torch.zeros(210))
            action = torch.zeros(11)
            if candidate == "actor":
                action[0] = 1.01 if row["episode_seed"] == 10 else .8
            actions.append(action)
    path = tmp_path / "raw.pt"
    torch.save(dict(rows=raw_rows, states=torch.stack(states), actions=torch.stack(actions)), path)
    violations = actor_normalized_action_bound_violations(path, data)
    assert int(violations.sum()) == 4


def test_quick_run_tag_only_changes_output_directory():
    original = settings(quick=True)
    _, tagged = load_contract(CONFIG, quick=True, quick_run_tag="retry1")
    assert tagged.pop("quick_run_tag") == "retry1"
    assert tagged.pop("output_directory") == original.pop("output_directory") + "_retry1"
    assert tagged == original
    for label in ("../escape", "x/y", "C:\\x", "", "a" * 33):
        with pytest.raises(ValueError, match="quick run tag"):
            load_contract(CONFIG, quick=True, quick_run_tag=label)
    with pytest.raises(ValueError, match="quick run tag"):
        load_contract(CONFIG, quick=False, quick_run_tag="retry1")


def result_rows(value: float) -> list[dict]:
    rows = []
    for policy in (9301, 9302, 9303):
        for split in ("confirmation_id", "confirmation_shift"):
            for family, base in (("frozen", 10), ("boiling", 20), ("combined", 30)):
                for episode in range(2):
                    rows.append(dict(policy_seed=policy, split=split,
                        condition_id=f"a10_{split}_{family}", condition_family=family,
                        episode_seed=base + episode, gate_reward_vs_zero=value,
                        gate_power_vs_zero=value, gate_reward_vs_always_actor=value))
    return rows


def test_statistics_pass_and_fail_without_treating_models_as_episodes():
    s = settings(quick=False)
    for split in s["splits"].values():
        split["episodes_per_condition"] = 2
    s["statistics"]["bootstrap_replicates"] = 100
    positive = result_rows(.2)
    rows, decision = comparison_rows(positive, s, split="confirmation_id")
    assert len(rows) == 9 and decision["status"] == "VETO_GATE_INDEPENDENT_CONFIRMATION_PASS"
    assert all(r["estimate"] == pytest.approx(.2) and r["episodes"] == 6 for r in rows)
    negative = deepcopy(positive)
    for row in negative:
        if row["policy_seed"] in (9301, 9302):
            row["gate_power_vs_zero"] = -.1
    _, decision = comparison_rows(negative, s, split="confirmation_id")
    assert decision["status"] == "VETO_GATE_INDEPENDENT_CONFIRMATION_FAIL"
    duplicate = positive + [positive[0]]
    with pytest.raises(RuntimeError, match="duplicate"):
        comparison_rows(duplicate, s, split="confirmation_id")
    quick = settings(quick=True)
    for split in quick["splits"].values():
        split["episodes_per_condition"] = 2
    quick_rows = [r for r in positive if r["policy_seed"] == 9301]
    _, qdecision = comparison_rows(quick_rows, quick, split="confirmation_id")
    assert qdecision["status"] == "QUICK_SMOKE_ONLY"


def test_selected_models_are_exact_a9_regression_only_set():
    s = settings(quick=False)
    design = s["design"]
    summary = json.loads((ROOT / design["upstream"]["summary"]).read_text(encoding="utf-8"))
    manifest = {}
    models = selected_models(summary, s, manifest)
    assert set(models) == {9301, 9302, 9303}
    assert all(len(records) == 3 for records in models.values())
    assert all(r["arm"] == "regression_only" and r["role"] == "selected"
               for records in models.values() for r in records)
    bad = deepcopy(summary)
    bad["fits"] = [f for f in bad["fits"] if not (f["arm"] == "regression_only" and f["replicate"] == 2)]
    with pytest.raises(RuntimeError, match="incomplete"):
        selected_models(bad, s, {})


def test_no_training_api_or_side_effectful_import():
    import src.rl.s4_r3_h16_veto_gate_confirmation as module
    source = inspect.getsource(module)
    assert "optimizer" not in source and ".backward(" not in source
    code = "from pathlib import Path\nfrom unittest.mock import patch\nwith patch.object(Path,'mkdir',side_effect=AssertionError('write')):\n import src.rl.s4_r3_h16_veto_gate_confirmation\n"
    subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, check=True, capture_output=True)
    help_result = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/confirm_s4_r3_h16_veto_gate.py"), "--help"],
        cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8")
    assert all(flag in help_result.stdout for flag in
               ("--quick", "--quick-run-tag", "--preflight-only"))
