from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.rl.s4_high_order_capacity import (
    AddedModesOracleController,
    _effective_settings,
    _rollout_added_modes_preview,
    preflight_s4_high_order_capacity,
)
from src.rl.s4_representation_capacity import (
    ActionRepresentation,
    _future_disturbance_sequence,
    build_action_basis,
    representation_registration_inverse,
)
from src.rl.s4_training import _load_yaml, _profiles, _project_path
from src.simulation.config import load_s1_config
from src.simulation.robust_control import RobustnessCondition


CONFIG_PATH = Path("configs/experiments/s4_high_order_capacity_v1.yaml")


def test_added_modes_controller_forces_requested_anchor_residual_to_zero() -> None:
    controller = AddedModesOracleController(
        num_modes=21,
        anchor_modes=10,
        modal_limit_rad=1.0,
        history_frames=4,
        residual_action_limit_rad=0.05,
        final_action_step_limit_rad=0.15,
        gain=0.25,
        leak=0.10,
        tracking_gain=0.50,
    )
    observation = torch.zeros(2, 44)
    controller.reset(observation)
    target = torch.linspace(-0.3, 0.3, 21).repeat(2, 1)

    action, desired, _ = controller.compose_added_modes_to_target(target)

    assert torch.count_nonzero(desired[:, :10]) == 0
    assert torch.count_nonzero(action.requested_residual_rad[:, :10]) == 0
    assert torch.count_nonzero(action.requested_residual_rad[:, 10:]) > 0


def test_formal_preflight_locks_nonlearning_high_order_design(tmp_path: Path) -> None:
    experiment_path = _project_path(CONFIG_PATH)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=False)
    settings["output_directory"] = str(tmp_path / "formal_high_order_capacity")

    result = preflight_s4_high_order_capacity(
        experiment_path,
        experiment,
        settings,
        quick=False,
    )

    assert result["status"] == "READY_FOR_USER_SIMULATION"
    assert result["rollout_count"] == 288
    assert result["expected_scenario_record_rows"] == 288
    assert result["expected_episode_record_rows"] == 4608
    assert not result["training_allowed"]
    assert result["requested_rl_anchor_coordinates_forced_zero"]


def test_preflight_rejects_training_authorization(tmp_path: Path) -> None:
    experiment_path = _project_path(CONFIG_PATH)
    experiment = _load_yaml(experiment_path)
    changed = deepcopy(experiment)
    changed["metadata"]["allow_training"] = True
    settings = _effective_settings(changed, quick=True)
    settings["output_directory"] = str(tmp_path / "quick_high_order_capacity")

    with pytest.raises(RuntimeError, match="scope flags"):
        preflight_s4_high_order_capacity(
            experiment_path,
            changed,
            settings,
            quick=True,
        )


def test_small_cpu_rollout_keeps_requested_anchor_coordinates_zero() -> None:
    experiment = _load_yaml(_project_path(CONFIG_PATH))
    base_config, _ = load_s1_config(_project_path(experiment["environment_config"]))
    config = replace(base_config, num_modes=21, batch_size=2, episode_length=6)
    representation = ActionRepresentation.from_mapping(experiment["representations"][0])
    basis, pupil, _ = build_action_basis(config, representation, torch.device("cpu"))
    profile = _profiles(experiment, ["nominal"])[0]
    condition = RobustnessCondition.from_mapping(experiment["quick"]["physical_conditions"][0])
    config = profile.environment_config(condition.environment_config(config))
    future = _future_disturbance_sequence(
        config=config,
        condition=condition,
        profile=profile,
        length=4,
        basis=basis,
        device=torch.device("cpu"),
    )
    mapping, _ = representation_registration_inverse(
        basis=basis,
        pupil=pupil,
        profile=profile,
        rcond=float(experiment["registration_inverse"]["pseudoinverse_rcond"]),
    )

    result, alignment, requested_anchor = _rollout_added_modes_preview(
        experiment=experiment,
        config=config,
        condition=condition,
        profile=profile,
        steps=4,
        preview_horizon_frames=0,
        future_disturbance=future,
        registration_mapping=mapping,
        basis=basis,
        device=torch.device("cpu"),
    )

    assert alignment == 0
    assert requested_anchor == 0
    assert torch.isfinite(result["power_in_bucket"]).all()
    assert torch.count_nonzero(
        result["requested_anchor_residual_abs_mean_rad"]
    ) == 0
