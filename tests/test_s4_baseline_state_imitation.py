from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from src.rl.s4_baseline_state_imitation import (
    _effective_settings,
    _interpretation,
    _lf_canonical_sha256,
    _matches_exact_or_lf_canonical,
    _validate_deployment_settings,
    preflight_s4_baseline_state_imitation,
)
from src.rl.s4_high_order_learnability import _validate_seed_splits
from src.rl.s4_training import _load_yaml, _project_path


CONFIG = Path("configs/experiments/s4_baseline_state_imitation_v1.yaml")


def test_formal_preflight_locks_single_factor_design_and_record_counts(
    tmp_path: Path,
) -> None:
    experiment_path = _project_path(CONFIG)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=False)
    settings["output_directory"] = str(tmp_path / "formal")
    result = preflight_s4_baseline_state_imitation(
        experiment_path,
        experiment,
        settings,
        quick=False,
    )
    assert result["status"] == "READY_FOR_USER_BASELINE_STATE_SUPERVISED_TRAINING"
    assert result["training_state_source"] == "traditional_baseline_trajectory"
    assert result["sample_counts"] == {
        "training": 115200,
        "validation": 57600,
        "diagnostic_test": 57600,
    }
    assert result["planned_records"] == {
        "scenario_records": 432,
        "episode_records": 6912,
        "temporal_records": 2160,
        "temporal_episode_records": 34560,
        "control_episode_records": 576,
    }
    assert result["initialization_seeds"] == [8201, 8202, 8203]
    assert result["deployment_scales"] == [0.1, 0.25, 0.5]
    assert result["primary_scale"] == 0.5
    assert not result["rl_training_allowed"]
    assert not result["real_slm_actions"]


def test_seed_validation_rejects_cross_split_episode_reuse() -> None:
    experiment = _load_yaml(_project_path(CONFIG))
    settings = _effective_settings(experiment, quick=False)
    broken = deepcopy(settings)
    broken["validation_conditions"][0]["base_seed"] = broken[
        "training_conditions"
    ][0]["base_seed"]
    with pytest.raises(RuntimeError, match="overlap across"):
        _validate_seed_splits(experiment, broken)


def test_deployment_validation_requires_a_preregistered_primary_scale() -> None:
    experiment = _load_yaml(_project_path(CONFIG))
    settings = _effective_settings(experiment, quick=False)
    broken = deepcopy(settings)
    broken["primary_scale"] = 0.75
    with pytest.raises(RuntimeError, match="primary scale"):
        _validate_deployment_settings(experiment, broken)


def test_text_evidence_accepts_crlf_after_lf_canonicalization(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.txt"
    evidence.write_bytes("第一行\r\n第二行\r\n".encode("utf-8"))
    expected = _lf_canonical_sha256(evidence)
    assert _matches_exact_or_lf_canonical(evidence, expected)


def test_interpretation_requires_offline_and_primary_closed_loop_improvement() -> None:
    experiment = _load_yaml(_project_path(CONFIG))
    settings = _effective_settings(experiment, quick=True)
    offline = {
        "teacher_mlp_seed_8201": {
            "mse": 0.20,
            "skill_score_against_zero": 0.40,
            "mean_cosine_similarity": 0.40,
        },
        "baseline_mlp_seed_8201": {
            "mse": 0.08,
            "skill_score_against_zero": 0.70,
            "mean_cosine_similarity": 0.70,
        },
    }
    grouped = {
        "teacher_mlp_seed_8201": {
            "0.10": {"overall": {"relative_power_gain": -0.01}, "closed_loop_gate": "FAIL"},
            "0.50": {"overall": {"relative_power_gain": -0.10}, "closed_loop_gate": "FAIL"},
        },
        "baseline_mlp_seed_8201": {
            "0.10": {"overall": {"relative_power_gain": 0.01}, "closed_loop_gate": "FAIL"},
            "0.50": {"overall": {"relative_power_gain": 0.04}, "closed_loop_gate": "PASS"},
        },
    }
    result = _interpretation(
        grouped=grouped,
        offline_results=offline,
        teacher_summary={"closed_loop_gate": "PASS"},
        experiment=experiment,
        settings=settings,
        quick=False,
    )
    assert result["status"] == "BASELINE_STATE_CORRECTION_PASS"
    assert result["baseline_state_offline_correction_pass"]
    assert result["primary_scale_improvement_pass"]
    assert result["primary_scale_closed_loop_pass"]


def test_preflight_rejects_model_architecture_change(tmp_path: Path) -> None:
    experiment_path = _project_path(CONFIG)
    experiment = _load_yaml(experiment_path)
    broken = deepcopy(experiment)
    broken["model"]["hidden_size"] = 128
    settings = _effective_settings(broken, quick=False)
    settings["output_directory"] = str(tmp_path / "formal")
    with pytest.raises(RuntimeError, match="256-wide"):
        preflight_s4_baseline_state_imitation(
            experiment_path,
            broken,
            settings,
            quick=False,
        )
