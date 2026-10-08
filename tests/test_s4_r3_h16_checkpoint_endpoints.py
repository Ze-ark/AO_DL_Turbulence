"""A8-D3小型确定性CPU单元测试；不生成正式性能或机制结论。"""

from __future__ import annotations

from copy import deepcopy
import importlib
import math
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

import src.rl.s4_r3_h16_checkpoint_endpoints as endpoint
from src.rl.s4_r3_h16_head_split import HeadSplitProbe, validate_coverage
from src.rl.s4_r3_h16_reward_pairwise_probe import RewardPairDataset
from src.rl.s4_training import _load_yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/s4_r3_h16_checkpoint_endpoints_v1.yaml"


def config() -> dict:
    return _load_yaml(CONFIG)


def rows(offset: int, shift: float = 0.0) -> list[dict]:
    return [{"condition_id": f"example_{family}", "episode_seed": offset + i * 4 + j,
             "gradient_cosine": -0.2 + 0.03 * j + shift}
            for i, family in enumerate(endpoint.FAMILIES) for j in range(4)]


def gap(bt: list[dict], lt: list[dict], bv: list[dict], lv: list[dict]) -> dict:
    return endpoint.paired_gap_interval(bt, lt, bv, lv, replicates=1000, seed=10, device=torch.device("cpu"))


def tiny_data() -> RewardPairDataset:
    generator = torch.Generator().manual_seed(77)
    target = torch.tensor([0.2, -0.1, -0.3] * 6)
    return RewardPairDataset(torch.randn(18, 221, generator=generator), target, target.clone(),
                             [{"condition_id": "tiny_frozen", "episode_seed": 1,
                               "profile_id": f"p{i // 3}", "probe_step": [0, 80, 160][i % 3]}
                              for i in range(18)])


def checkpoint(arm: str = "shared", *, hidden: int = 256) -> tuple[HeadSplitProbe, dict]:
    torch.manual_seed(44)
    model = HeadSplitProbe(221, hidden, arm)
    saved = {"policy_seed": 9301, "arm": arm, "update": endpoint.BEST_UPDATES[9301][arm],
             "best_update": endpoint.BEST_UPDATES[9301][arm], "config": {"feature_size": 221, "hidden_size": hidden},
             "target_horizon": 16, "label_source": "empirical_reward_returns", "training_reward_mean": 0.0,
             "independent_test_used_for_selection": False, "ranking_score_is_reward": False,
             "probe": deepcopy(model.state_dict()), "normalization": {"feature_mean": torch.zeros(221),
               "feature_scale": torch.ones(221), "target_scale": torch.tensor(0.2)},
             "original_critic_updates": 0, "actor_updates": 0, "alpha_updates": 0, "student_updates": 0}
    return model, saved


def history() -> list[dict]:
    result = []
    for update in range(100, 10001, 100):
        result.append({"policy_seed": 9301, "arm": "shared", "update": update, "total_updates": 10000,
                       "validation_balanced_accuracy": 0.7 if update == 2300 else 0.5,
                       "validation_value_mae": 0.01, "best_update": 100 if update < 2300 else 2300})
    return result


def comparisons() -> list[dict]:
    return [{"policy_seed": seed, "arm": arm, "delta_gap": {"familywise_ci_low": -0.1, "familywise_ci_high": 0.1}}
            for seed in endpoint.SEEDS for arm in endpoint.ARMS]


def test_config_and_quick_budget() -> None:
    experiment = config()
    endpoint._validate_contract(experiment)
    assert endpoint._effective_settings(experiment, quick=False)["policy_seeds"] == [9301, 9302, 9303]
    quick = endpoint._effective_settings(experiment, quick=True)
    assert quick["bootstrap_replicates"] == 100
    assert 1 * 2 * 2 * (3 * quick["episodes_per_condition"] * 2) == 48


@pytest.mark.parametrize("flag", ["allow_training", "allow_optimizer_updates", "allow_new_data_generation",
                                     "allow_independent_test_access", "allow_full_rl_training",
                                     "allow_s4d3_access", "allow_real_hardware_actions", "allow_secret_training"])
def test_safety_flags_cannot_expand(flag: str) -> None:
    experiment = config()
    experiment["metadata"][flag] = True
    with pytest.raises(RuntimeError, match="safety flag"):
        endpoint._validate_contract(experiment)


@pytest.mark.parametrize("section,key,value", [("design", "pairs_per_episode", 17),
    ("comparison", "family_size", 1), ("comparison", "meaningful_gap_change", 0.1),
    ("alignment", "atol", 0.01), ("runtime", "device", "cpu")])
def test_contract_and_tolerances_are_fixed(section: str, key: str, value: object) -> None:
    experiment = config()
    experiment[section][key] = value
    with pytest.raises(RuntimeError):
        endpoint._validate_contract(experiment)


def test_import_and_cli_help_do_not_execute() -> None:
    with patch("torch.load", side_effect=AssertionError("unexpected checkpoint read")), \
            patch("torch.cuda.is_available", side_effect=AssertionError("unexpected GPU query")):
        importlib.reload(endpoint)
    result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/diagnose_s4_r3_h16_checkpoint_endpoints.py"), "--help"],
                            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
    assert "不训练" in result.stdout and "只读核对" in result.stdout


def test_path_hash_and_output_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(endpoint, "_project_path", lambda p: tmp_path / p)
    safe = tmp_path / "safe.txt"
    safe.write_text("frozen", encoding="utf-8")
    digest = endpoint._file_sha256(safe)
    manifest: dict = {}
    assert endpoint._pin("safe.txt", digest, manifest, expected="safe.txt") == safe
    with pytest.raises(RuntimeError, match="hash mismatch"):
        endpoint._pin("safe.txt", "0" * 64, {})
    for name in ("../escape.pt", "outputs/independent_test_raw.pt", "s4d3/data.pt"):
        with pytest.raises(RuntimeError):
            endpoint._safe_input(name)
    with pytest.raises(RuntimeError, match="unexpected input"):
        endpoint._safe_input("safe.txt", expected="other.txt")
    out = endpoint._output_path("outputs/s4_r3_h16_checkpoint_endpoints_v1")
    out.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        endpoint._output_path("outputs/s4_r3_h16_checkpoint_endpoints_v1")
    with pytest.raises(RuntimeError):
        endpoint._output_path("outputs/s4_r3_h16_head_split_v1")


def test_no_cuda_fails_before_reading_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = config()
    monkeypatch.setattr(endpoint, "_output_path", lambda p: tmp_path / "no_output")
    with patch("torch.cuda.is_available", return_value=False), \
            patch.object(endpoint, "_pin", side_effect=AssertionError("input read after missing CUDA")), \
            pytest.raises(RuntimeError, match="CUDA"):
        endpoint.preflight_endpoints(CONFIG, experiment, endpoint._effective_settings(experiment, quick=False))
    assert not (tmp_path / "no_output").exists()


def test_history_selection_and_missing_rows() -> None:
    original = history()
    assert endpoint.validate_history(original, seed=9301, arm="shared")["best_update"] == 2300
    original[50]["best_update"] = 100
    with pytest.raises(RuntimeError, match="selection"):
        endpoint.validate_history(original, seed=9301, arm="shared")
    with pytest.raises(RuntimeError, match="missing/duplicated"):
        endpoint.validate_history(history()[1:], seed=9301, arm="shared")


@pytest.mark.parametrize("arm", ["shared", "split"])
def test_checkpoint_identity_shape_and_normalization(arm: str) -> None:
    _, saved = checkpoint(arm)
    endpoint.validate_checkpoint(saved, seed=9301, arm=arm, role="best")
    saved["update"] = 10000
    endpoint.validate_checkpoint(saved, seed=9301, arm=arm, role="last")
    with pytest.raises(RuntimeError, match="identity/update"):
        endpoint.validate_checkpoint(saved, seed=9301, arm=arm, role="best")
    saved["normalization"]["feature_scale"][0] = 0
    with pytest.raises(RuntimeError, match="normalization"):
        endpoint.validate_checkpoint(saved, seed=9301, arm=arm, role="last")
    _, saved = checkpoint(arm)
    saved["probe"]["network.0.bias"] = torch.zeros(10)
    with pytest.raises(RuntimeError, match="parameter shape"):
        endpoint.validate_checkpoint(saved, seed=9301, arm=arm, role="best")


def test_identical_weights_and_gradient_reads_never_update() -> None:
    data = tiny_data()
    model, saved = checkpoint("split", hidden=16)
    other = deepcopy(model)
    before = endpoint._model_digest(model)
    with patch("torch.optim.Adam", side_effect=AssertionError("optimizer forbidden")):
        first = endpoint.evaluate_endpoint(model, data, saved, positive_weight=torch.tensor(2.0), device=torch.device("cpu"))
        second = endpoint.evaluate_endpoint(other, data, saved, positive_weight=torch.tensor(2.0), device=torch.device("cpu"))
        assert first == second
        gradient = endpoint._episode_gradient(model, data, list(range(18)), saved, positive_weight=torch.tensor(2.0),
                                             classification_weight=0.25, huber_delta=1.0, device=torch.device("cpu"))
    assert first["tp"] + first["fn"] == 6
    assert first["fp"] + first["tn"] == 12
    assert math.isfinite(gradient["gradient_cosine"])
    assert endpoint._model_digest(model) == before
    assert all(p.grad is None for p in model.parameters())


def test_zero_change_and_known_paired_delta_gap() -> None:
    zero = gap(rows(0), rows(0), rows(100), rows(100))
    assert zero["estimate"] == 0 and zero["familywise_ci_low"] == 0 and zero["familywise_ci_high"] == 0
    result = gap(rows(0), rows(0, 0.1), rows(100), rows(100, 0.5))
    assert result["estimate"] == pytest.approx(0.4)
    assert result["familywise_ci_low"] == pytest.approx(0.4)
    assert result["checkpoint_paired"] and not result["training_validation_paired"]
    assert result["family_size"] == 6 and result["unit"] == "complete_episode_seed"


def test_resampling_reproducible_and_corrected_interval_is_wider() -> None:
    changed = rows(100, 0.3)
    for i, row in enumerate(changed):
        row["gradient_cosine"] += i * 0.02
    one = gap(rows(0), rows(0, 0.1), rows(100), changed)
    assert one == gap(rows(0), rows(0, 0.1), rows(100), changed)
    assert one["familywise_ci_low"] <= one["ci95_low"] <= one["ci95_high"] <= one["familywise_ci_high"]


def test_pairing_leakage_and_finite_guards() -> None:
    with pytest.raises(RuntimeError, match="keys"):
        gap(rows(0), rows(0)[:-1], rows(100), rows(100))
    duplicated = rows(0) + rows(0)[:1]
    with pytest.raises(RuntimeError, match="duplicate"):
        gap(duplicated, duplicated, rows(100), rows(100))
    with pytest.raises(RuntimeError, match="leakage"):
        gap(rows(0), rows(0), rows(0), rows(0))
    invalid = rows(0)
    invalid[0]["gradient_cosine"] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        gap(invalid, rows(0), rows(100), rows(100))


def test_interpretation_needs_two_same_direction_shared_seeds() -> None:
    sample = comparisons()
    assert not endpoint.interpret_endpoints(sample, quick=False)["checkpoint_sensitivity_supported"]
    for index in (0, 2):
        sample[index]["delta_gap"] = {"familywise_ci_low": 0.21, "familywise_ci_high": 0.5}
    result = endpoint.interpret_endpoints(sample, quick=False)
    assert result["status"] == "CHECKPOINT_SENSITIVE_GAP_CANDIDATE"
    assert not result["new_model_training_authorized"] and not result["overfitting_confirmed"]
    sample[2]["delta_gap"] = {"familywise_ci_low": -0.5, "familywise_ci_high": -0.21}
    assert not endpoint.interpret_endpoints(sample, quick=False)["checkpoint_sensitivity_supported"]
    sample[0]["delta_gap"] = {"familywise_ci_low": 0.2, "familywise_ci_high": 0.5}
    assert endpoint.interpret_endpoints(sample, quick=False)["positive_supporting_seeds"] == []
    assert endpoint.interpret_endpoints(sample, quick=True)["status"] == "QUICK_SMOKE_ONLY"
    with pytest.raises(RuntimeError, match="coverage"):
        endpoint.interpret_endpoints(sample[:-1], quick=False)


def test_alignment_checks_all_metrics_keys_and_conflict_sign() -> None:
    ref = {"policy_seed": 9301, "arm": "shared", "split": "training", "condition_id": "tiny_frozen",
           "episode_seed": 1, "pairs": 18, "gradient_cosine": 0.2, "gradient_conflict": 0.0,
           "regression_gradient_norm": 1.0}
    assert endpoint.check_gradient_alignment(dict(ref), ref) == 0
    actual = dict(ref, regression_gradient_norm=1.2)
    with pytest.raises(RuntimeError, match="ALIGNMENT_FAILED"):
        endpoint.check_gradient_alignment(actual, ref)
    actual = dict(ref, gradient_conflict=1)
    with pytest.raises(RuntimeError, match="conflict sign"):
        endpoint.check_gradient_alignment(actual, ref)


def test_data_coverage_rejects_duplicate_pairs() -> None:
    data = tiny_data()
    validate_coverage(data, episodes=1, pairs=18)
    data.rows[1].update(data.rows[0])
    with pytest.raises(RuntimeError, match="duplicated"):
        validate_coverage(data, episodes=1, pairs=18)


def test_metric_alignment_rejects_wrong_last_validation() -> None:
    spec = {"last_validation_balanced_accuracy": 0.6, "last_validation_value_mae": 0.01}
    endpoint._align_metrics({"balanced_accuracy": 0.6, "value_mae": 0.01}, role="last", split="validation", fit={}, inventory=spec)
    with pytest.raises(RuntimeError, match="ALIGNMENT_FAILED"):
        endpoint._align_metrics({"balanced_accuracy": 0.61, "value_mae": 0.01}, role="last", split="validation", fit={}, inventory=spec)
