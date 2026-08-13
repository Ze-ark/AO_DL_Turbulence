from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from src.rl.s4_action_range_sweep import (
    find_minimum_demonstrated_limit,
    run_s4_r1_action_range_sweep,
    select_limit_envelope,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "s4_r1_action_range_sweep_v1.yaml"


def _temporary_config(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    experiment = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    experiment["outputs"]["directory"] = str(tmp_path / "formal")
    experiment["outputs"]["quick_directory"] = str(tmp_path / "quick")
    path = tmp_path / "action_range.yaml"
    path.write_text(
        yaml.safe_dump(experiment, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path, experiment


def test_limit_envelope_keeps_all_metrics_from_selected_limit() -> None:
    candidates = {
        0.0125: {
            "power_in_bucket": torch.tensor([0.7, 0.9]),
            "strehl": torch.tensor([0.5, 0.8]),
            "selected_residual_action_limit_rad": torch.tensor([0.0125, 0.0125]),
        },
        0.025: {
            "power_in_bucket": torch.tensor([0.8, 0.6]),
            "strehl": torch.tensor([0.7, 0.4]),
            "selected_residual_action_limit_rad": torch.tensor([0.025, 0.025]),
        },
    }

    selected = select_limit_envelope(candidates)

    assert torch.allclose(selected["power_in_bucket"], torch.tensor([0.8, 0.9]))
    assert torch.allclose(selected["strehl"], torch.tensor([0.7, 0.8]))
    assert torch.allclose(
        selected["selected_residual_action_limit_rad"],
        torch.tensor([0.025, 0.0125]),
    )


def test_find_minimum_demonstrated_limit_uses_first_full_pass() -> None:
    summaries = [
        {"maximum_allowed_residual_action_rad": 0.025, "capacity_gate": "FAIL"},
        {"maximum_allowed_residual_action_rad": 0.05, "capacity_gate": "PASS"},
        {"maximum_allowed_residual_action_rad": 0.0375, "capacity_gate": "PASS"},
    ]

    assert find_minimum_demonstrated_limit(summaries) == pytest.approx(0.0375)


def test_action_range_preflight_refuses_rerun_after_source_change(
    tmp_path: Path,
) -> None:
    path, _ = _temporary_config(tmp_path)
    with pytest.raises(RuntimeError, match="oracle tracked source changed"):
        run_s4_r1_action_range_sweep(path, quick=True, preflight_only=True)


def test_action_range_preflight_rejects_missing_anchor(tmp_path: Path) -> None:
    path, experiment = _temporary_config(tmp_path)
    experiment["action"]["residual_action_limits_rad"] = [0.01875, 0.025, 0.05]
    path.write_text(
        yaml.safe_dump(experiment, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="oracle tracked source changed"):
        run_s4_r1_action_range_sweep(path, quick=True, preflight_only=True)


def test_action_range_preflight_rejects_limit_above_final_step(
    tmp_path: Path,
) -> None:
    path, experiment = _temporary_config(tmp_path)
    experiment["action"]["residual_action_limits_rad"].append(0.20)
    path.write_text(
        yaml.safe_dump(experiment, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="oracle tracked source changed"):
        run_s4_r1_action_range_sweep(path, quick=True, preflight_only=True)


def test_action_range_preflight_rejects_prior_seed_overlap(tmp_path: Path) -> None:
    path, experiment = _temporary_config(tmp_path)
    experiment["quick"]["physical_conditions"][0]["base_seed"] = 2599999
    path.write_text(
        yaml.safe_dump(experiment, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="oracle tracked source changed"):
        run_s4_r1_action_range_sweep(path, quick=True, preflight_only=True)


def test_quick_preflight_also_rejects_formal_seed_overlap(tmp_path: Path) -> None:
    path, experiment = _temporary_config(tmp_path)
    experiment["evaluation"]["physical_conditions"][0]["base_seed"] = 2599999
    path.write_text(
        yaml.safe_dump(experiment, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="oracle tracked source changed"):
        run_s4_r1_action_range_sweep(path, quick=True, preflight_only=True)


def test_quick_preflight_rejects_changed_formal_episode_count(tmp_path: Path) -> None:
    path, experiment = _temporary_config(tmp_path)
    experiment["evaluation"]["episodes_per_physical_condition"] = 8
    path.write_text(
        yaml.safe_dump(experiment, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="oracle tracked source changed"):
        run_s4_r1_action_range_sweep(path, quick=True, preflight_only=True)
