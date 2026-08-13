from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from src.rl.s4_registration_oracle import (
    _available_profile_ids,
    registration_aware_normalized_residual,
    registration_inverse_modal_map,
    run_s4_registration_oracle,
)
from src.simulation.config import load_s1_config
from src.simulation.hardware_effects import HardwareProfile, apply_registration_error
from src.simulation.modes import make_low_order_zernike_basis, synthesize_phase


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "s4_registration_oracle_v1.yaml"
ENVIRONMENT = ROOT / "configs" / "environment" / "s1_taylor_v1.yaml"


def _temporary_config(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    experiment = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    experiment["outputs"]["directory"] = str(tmp_path / "formal")
    experiment["outputs"]["quick_directory"] = str(tmp_path / "quick")
    path = tmp_path / "registration_oracle.yaml"
    path.write_text(
        yaml.safe_dump(experiment, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path, experiment


def test_nominal_registration_map_is_exact_phase_scale_inverse() -> None:
    config, _ = load_s1_config(ENVIRONMENT)
    profile = HardwareProfile(
        identifier="scaled",
        label="scaled",
        severity="test",
        required_for_gate=True,
        phase_scale=0.8,
    )

    mapping, diagnostics = registration_inverse_modal_map(
        config=config,
        profile=profile,
        rcond=1e-5,
        device=torch.device("cpu"),
    )

    assert torch.equal(mapping, torch.eye(config.num_modes) / 0.8)
    assert diagnostics["registration_present"] is False
    assert diagnostics["condition_number"] == 1.0
    assert diagnostics["relative_pupil_reconstruction_rmse"] == 0.0


def test_registration_inverse_reduces_full_pupil_error() -> None:
    config, _ = load_s1_config(ENVIRONMENT)
    profile = HardwareProfile(
        identifier="severe",
        label="severe",
        severity="test",
        required_for_gate=True,
        shift_x_pixels=2.0,
        shift_y_pixels=2.0,
        rotation_deg=1.0,
    )
    mapping, diagnostics = registration_inverse_modal_map(
        config=config,
        profile=profile,
        rcond=1e-5,
        device=torch.device("cpu"),
    )
    basis, pupil = make_low_order_zernike_basis(
        config.grid_size,
        config.pupil_radius_fraction,
        config.num_modes,
        torch.device("cpu"),
    )
    generator = torch.Generator().manual_seed(7)
    target_modal = torch.randn(32, config.num_modes, generator=generator) * 0.25
    desired = synthesize_phase(target_modal, basis)
    blind = apply_registration_error(
        synthesize_phase(target_modal, basis),
        shift_x_pixels=2.0,
        shift_y_pixels=2.0,
        rotation_deg=1.0,
    )
    aware = apply_registration_error(
        synthesize_phase(target_modal @ mapping.T, basis),
        shift_x_pixels=2.0,
        shift_y_pixels=2.0,
        rotation_deg=1.0,
    )
    blind_rmse = torch.sqrt((blind[:, pupil] - desired[:, pupil]).square().mean())
    aware_rmse = torch.sqrt((aware[:, pupil] - desired[:, pupil]).square().mean())

    assert aware_rmse < blind_rmse
    assert diagnostics["registration_present"] is True
    assert diagnostics["condition_number"] < 10


def test_registration_aware_residual_targets_inverse_mapped_command() -> None:
    state = torch.zeros(2, 100)
    state[:, -20:-10] = 0.10
    state[:, -10:] = -0.02
    future = torch.full((2, 10), -0.30)
    mapping = torch.eye(10) * 1.5

    normalized, target = registration_aware_normalized_residual(
        state=state,
        future_disturbance_modal=future,
        registration_mapping=mapping,
        num_modes=10,
        residual_action_limit_rad=0.05,
        modal_limit_rad=3.0,
    )

    assert torch.allclose(target, torch.full_like(target, 0.45))
    assert torch.all(normalized == 1)


def test_quick_summary_filters_formal_profile_lists_to_available_profiles() -> None:
    available = {"registration_severe": object()}

    assert _available_profile_ids(
        ["registration_moderate", "registration_severe"], available
    ) == ["registration_severe"]
    assert _available_profile_ids(
        ["nominal", "delay_3", "settling_050"], available
    ) == []


def test_closed_registration_oracle_rejects_rerun_after_tracked_source_change(
    tmp_path: Path,
) -> None:
    path, _ = _temporary_config(tmp_path)

    with pytest.raises(RuntimeError, match="tracked source changed"):
        run_s4_registration_oracle(path, quick=True, preflight_only=True)


def test_registration_oracle_rejects_target_profile_change(tmp_path: Path) -> None:
    path, experiment = _temporary_config(tmp_path)
    experiment["registration_inverse"]["target_profile_ids"] = [
        "registration_moderate"
    ]
    path.write_text(
        yaml.safe_dump(experiment, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="target profiles"):
        run_s4_registration_oracle(path, quick=True, preflight_only=True)


def test_registration_oracle_rejects_action_limit_change(tmp_path: Path) -> None:
    path, experiment = _temporary_config(tmp_path)
    experiment["action"]["residual_action_limit_rad"] = 0.075
    path.write_text(
        yaml.safe_dump(experiment, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="0.05 rad"):
        run_s4_registration_oracle(path, quick=True, preflight_only=True)


def test_registration_oracle_rejects_formal_seed_overlap(tmp_path: Path) -> None:
    path, experiment = _temporary_config(tmp_path)
    experiment["evaluation"]["physical_conditions"][0]["base_seed"] = 2699999
    path.write_text(
        yaml.safe_dump(experiment, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="protected seeds"):
        run_s4_registration_oracle(path, quick=True, preflight_only=True)
