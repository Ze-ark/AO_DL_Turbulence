from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from src.rl.s4_oracle_bound import (
    oracle_normalized_residual,
    run_s4_r1_oracle_bound,
    select_hindsight_envelope,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "s4_r1_oracle_bound_v1.yaml"


def _temporary_config(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    experiment = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    experiment["outputs"]["directory"] = str(tmp_path / "formal")
    experiment["outputs"]["quick_directory"] = str(tmp_path / "quick")
    path = tmp_path / "oracle_bound.yaml"
    path.write_text(
        yaml.safe_dump(experiment, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path, experiment


def test_oracle_residual_projects_the_truth_target_into_r1_box() -> None:
    state = torch.zeros(2, 100)
    state[:, -20:-10] = 0.10
    state[:, -10:] = -0.02
    future = torch.full((2, 10), -0.30)

    normalized, target = oracle_normalized_residual(
        state=state,
        future_disturbance_modal=future,
        num_modes=10,
        residual_action_limit_rad=0.0125,
        modal_limit_rad=3.0,
        phase_scale=1.0,
    )

    assert torch.allclose(target, torch.full_like(target, 0.30))
    assert torch.all(normalized == 1)
    requested = normalized * 0.0125
    assert float(requested.abs().max()) == pytest.approx(0.0125, abs=1e-7)


def test_hindsight_envelope_keeps_all_metrics_from_the_selected_candidate() -> None:
    candidates = {
        0: {
            "power_in_bucket": torch.tensor([0.7, 0.9]),
            "strehl": torch.tensor([0.5, 0.8]),
        },
        2: {
            "power_in_bucket": torch.tensor([0.8, 0.6]),
            "strehl": torch.tensor([0.7, 0.4]),
        },
    }

    selected = select_hindsight_envelope(candidates)

    assert torch.allclose(selected["power_in_bucket"], torch.tensor([0.8, 0.9]))
    assert torch.allclose(selected["strehl"], torch.tensor([0.7, 0.8]))
    assert torch.allclose(
        selected["selected_preview_horizon_frames"], torch.tensor([2.0, 0.0])
    )


def test_oracle_bound_preflight_refuses_rerun_after_source_change(
    tmp_path: Path,
) -> None:
    path, _ = _temporary_config(tmp_path)
    with pytest.raises(RuntimeError, match="R1 tracked source changed"):
        run_s4_r1_oracle_bound(path, quick=True, preflight_only=True)


def test_oracle_bound_preflight_rejects_r1_action_change(tmp_path: Path) -> None:
    path, experiment = _temporary_config(tmp_path)
    experiment["action"]["residual_action_limit_rad"] = 0.02
    path.write_text(
        yaml.safe_dump(experiment, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="R1 tracked source changed"):
        run_s4_r1_oracle_bound(path, quick=True, preflight_only=True)


def test_oracle_bound_preflight_rejects_prior_seed_overlap(tmp_path: Path) -> None:
    path, experiment = _temporary_config(tmp_path)
    experiment["quick"]["physical_conditions"][0]["base_seed"] = 2499999
    path.write_text(
        yaml.safe_dump(experiment, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="R1 tracked source changed"):
        run_s4_r1_oracle_bound(path, quick=True, preflight_only=True)
