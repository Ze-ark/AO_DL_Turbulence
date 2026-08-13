from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from src.rl.residual_control import ResidualTrackingController
from src.rl.s4_representation_capacity import (
    ANCHOR_MODES,
    ActionRepresentation,
    AnchoredResidualController,
    _effective_settings,
    _extract_anchor_observation,
    _future_disturbance_sequence,
    _rollout_representation_preview,
    _verify_fair_design,
    build_action_basis,
    clip_box_and_l2,
    representation_registration_inverse,
)
from src.rl.s4_training import _load_yaml
from src.simulation.config import S1EnvConfig
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.hardware_effects import HardwareProfile
from src.simulation.robust_control import RobustnessCondition


def _config(**changes) -> S1EnvConfig:
    values = {
        "grid_size": 32,
        "turbulence_grid_multiplier": 1,
        "batch_size": 2,
        "episode_length": 3,
        "num_modes": 10,
        "slm_delay_frames": 0,
        "slm_quantization_levels": 0,
        "slm_max_delta_rad": 10.0,
        "slm_phase_min_rad": -10.0,
        "slm_phase_max_rad": 10.0,
        "wind_speed_mps": 0.0,
    }
    values.update(changes)
    return S1EnvConfig(**values)


def _parameters() -> dict[str, float]:
    return {
        "gain": 0.25,
        "leak": 0.10,
        "tracking_gain": 0.50,
        "max_request_step_rad": 0.15,
    }


def test_action_bases_are_orthonormal_and_share_the_ten_mode_anchor():
    config = _config()
    anchor, pupil, anchor_diagnostics = build_action_basis(
        config,
        ActionRepresentation("zernike_10_anchor", "zernike", 10),
        torch.device("cpu"),
    )
    hybrid, hybrid_pupil, diagnostics = build_action_basis(
        config,
        ActionRepresentation("hybrid_spatial_256", "hybrid_spatial", 256),
        torch.device("cpu"),
    )

    assert torch.equal(pupil, hybrid_pupil)
    assert torch.equal(anchor, hybrid[:ANCHOR_MODES])
    assert anchor_diagnostics["max_orthonormality_error"] < 1e-8
    assert diagnostics["max_orthonormality_error"] < 1e-8


def test_high_dimension_box_is_projected_to_the_ten_mode_total_budget():
    residual_limit = 0.05
    total_limit = (ANCHOR_MODES**0.5) * residual_limit
    ten = torch.full((2, 10), residual_limit)
    high = torch.full((2, 256), residual_limit)

    ten_projected, ten_limited = clip_box_and_l2(
        ten,
        component_limit=residual_limit,
        l2_limit=total_limit,
    )
    high_projected, high_limited = clip_box_and_l2(
        high,
        component_limit=residual_limit,
        l2_limit=total_limit,
    )

    assert torch.equal(ten_projected, ten)
    assert torch.equal(ten_limited, torch.zeros_like(ten_limited))
    assert torch.allclose(
        torch.linalg.vector_norm(high_projected, dim=-1),
        torch.full((2,), total_limit),
        atol=1e-6,
    )
    assert torch.equal(high_limited, torch.ones_like(high_limited))


def test_ten_mode_anchored_composition_matches_existing_residual_controller():
    torch.manual_seed(4)
    observation = torch.randn(2, 2 * ANCHOR_MODES + 2) * 0.1
    target = torch.randn(2, ANCHOR_MODES) * 0.02
    existing = ResidualTrackingController(
        num_modes=ANCHOR_MODES,
        modal_limit_rad=3.0,
        history_frames=1,
        residual_action_limit_rad=0.05,
        final_action_step_limit_rad=0.15,
        gain=0.25,
        leak=0.10,
        tracking_gain=0.50,
    )
    state = existing.reset(observation)
    prior = state[:, -2 * ANCHOR_MODES : -ANCHOR_MODES]
    baseline_delta = state[:, -ANCHOR_MODES:]
    normalized = ((target - prior - baseline_delta) / 0.05).clamp(-1, 1)
    old_action = existing.compose_action(normalized)

    anchored = AnchoredResidualController(
        num_modes=ANCHOR_MODES,
        modal_limit_rad=3.0,
        residual_limit_rad=0.05,
        final_step_limit_rad=0.15,
        parameters=_parameters(),
    )
    anchored.reset(observation)
    new_action = anchored.compose_to_target(target)

    assert torch.allclose(new_action.baseline_delta_rad, old_action.baseline_delta_rad)
    assert torch.allclose(new_action.final_delta_rad, old_action.final_delta_rad)
    assert torch.allclose(new_action.realized_residual_rad, old_action.realized_residual_rad)


def test_environment_accepts_a_valid_basis_override():
    config = _config(num_modes=21)
    basis, _, _ = build_action_basis(
        config,
        ActionRepresentation("zernike_21", "zernike", 21),
        torch.device("cpu"),
    )
    environment = AdaptiveOpticsEnv(config, "cpu", basis_override=basis)
    observation, _ = environment.reset(seed=8)
    next_observation, _, _, _, _ = environment.step(torch.zeros(2, 21))

    assert observation.shape == (2, 44)
    assert next_observation.shape == (2, 44)


def test_256d_truth_alignment_uses_direct_turbulence_projection():
    config = _config(num_modes=256, episode_length=4)
    representation = ActionRepresentation(
        "hybrid_spatial_256",
        "hybrid_spatial",
        256,
    )
    basis, pupil, _ = build_action_basis(config, representation, torch.device("cpu"))
    profile = HardwareProfile(
        identifier="nominal",
        label="标称",
        severity="anchor",
        required_for_gate=True,
        slm_delay_frames=0,
        slm_quantization_levels=0,
        slm_max_delta_rad=10.0,
        observation_noise_std_rad=0.0,
        phase_scale=1.0,
    )
    condition = RobustnessCondition(
        identifier="truth_alignment_unit",
        regime="frozen_stationary",
        base_seed=91,
        wind_speed_mps=0.0,
        wind_direction_deg=0.0,
        slm_delay_frames=0,
    )
    experiment = _load_yaml(
        Path("configs/experiments/s4_representation_capacity_v1.yaml")
    )
    future = _future_disturbance_sequence(
        config=config,
        condition=condition,
        profile=profile,
        length=4,
        basis=basis,
        device=torch.device("cpu"),
    )
    _, alignment = _rollout_representation_preview(
        experiment=experiment,
        config=config,
        condition=condition,
        profile=profile,
        steps=3,
        preview_horizon_frames=1,
        future_disturbance=future,
        registration_mapping=torch.eye(256),
        basis=basis,
        device=torch.device("cpu"),
    )

    assert alignment == 0.0
    direct = AdaptiveOpticsEnv(
        config,
        "cpu",
        profile.effects_config(),
        basis_override=basis,
    )
    direct.reset(seed=condition.base_seed)
    truth = direct.oracle_disturbance_modal()
    assert truth.shape == (config.batch_size, config.num_modes)
    assert torch.isfinite(truth).all()


def test_anchor_observation_keeps_first_ten_residual_and_applied_modes():
    observation = torch.arange(2 * 36 + 2, dtype=torch.float32).unsqueeze(0)
    anchor = _extract_anchor_observation(observation, 36)

    expected = torch.cat((observation[:, :10], observation[:, 36:46], observation[:, -2:]), dim=1)
    assert torch.equal(anchor, expected)


def test_identity_registration_mapping_preserves_every_coordinate():
    config = _config()
    basis, pupil, _ = build_action_basis(
        config,
        ActionRepresentation("zernike_21", "zernike", 21),
        torch.device("cpu"),
    )
    profile = HardwareProfile(
        identifier="nominal",
        label="标称",
        severity="anchor",
        required_for_gate=True,
        phase_scale=1.0,
    )
    mapping, diagnostics = representation_registration_inverse(
        basis=basis,
        pupil=pupil,
        profile=profile,
        rcond=1e-5,
    )

    assert torch.equal(mapping, torch.eye(21))
    assert diagnostics["effective_rank"] == 21
    assert diagnostics["relative_pupil_reconstruction_rmse"] == 0.0


def test_quick_settings_only_select_declared_anchor_and_256_dimensions():
    experiment = _load_yaml(Path("configs/experiments/s4_representation_capacity_v1.yaml"))
    settings = _effective_settings(experiment, quick=True)

    assert [item["id"] for item in settings["representations"]] == [
        "zernike_10_anchor",
        "hybrid_spatial_256",
    ]
    assert settings["profile_ids"] == ["registration_severe"]
    assert settings["batch_size"] == 2


def test_fair_design_rejects_a_changed_physical_condition():
    experiment = _load_yaml(Path("configs/experiments/s4_representation_capacity_v1.yaml"))
    upstream = _load_yaml(Path("configs/experiments/s4_registration_oracle_v1.yaml"))
    changed = deepcopy(experiment)
    changed["evaluation"]["physical_conditions"][0]["wind_speed_mps"] = 9.0

    with pytest.raises(RuntimeError, match="physical conditions"):
        _verify_fair_design(changed, upstream)


def test_fair_design_rejects_a_looser_truth_alignment_tolerance():
    experiment = _load_yaml(Path("configs/experiments/s4_representation_capacity_v1.yaml"))
    upstream = _load_yaml(Path("configs/experiments/s4_registration_oracle_v1.yaml"))
    changed = deepcopy(experiment)
    changed["oracle"]["truth_alignment_tolerance"] = 2e-5

    with pytest.raises(RuntimeError, match="truth-alignment tolerance"):
        _verify_fair_design(changed, upstream)
