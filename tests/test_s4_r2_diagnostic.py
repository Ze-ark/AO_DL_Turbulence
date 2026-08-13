from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.rl.s4_r2_diagnostic import (
    _effective_settings,
    _rollout_anchored_baseline,
    _rollout_variant,
    interpret_r2_diagnostic,
    preflight_s4_r2_residual_diagnostic,
)
from src.rl.s4_training import _load_yaml, _profiles, _project_path
from src.simulation.config import load_s1_config
from src.simulation.robust_control import RobustnessCondition


CONFIG_PATH = Path("configs/experiments/s4_residual_sac_r2_diagnostic_v1.yaml")


class _OnesActor:
    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            state.shape[0],
            36,
            device=state.device,
            dtype=state.dtype,
        )


def _grouped_record(
    representation_id: str,
    scale: float,
    gain: float,
    *,
    projection: float = 0.0,
    residual_projection: float | None = None,
    violation: float = 0.0,
) -> dict[str, object]:
    return {
        "representation_id": representation_id,
        "scale": scale,
        "relative_power_gain": gain,
        "candidate": {"violation_fraction": {"mean": violation}},
        "action_diagnostics": {
            "final_projection_fraction": {"mean": projection},
            "residual_l2_projection_fraction": {
                "mean": projection if residual_projection is None else residual_projection
            },
        },
    }


def test_interpretation_requires_positive_smaller_scale_for_amplitude_label() -> None:
    grouped = [
        _grouped_record("zernike_21", 0.0, 0.0),
        _grouped_record("zernike_21", 0.25, 0.03),
        _grouped_record("zernike_21", 0.5, 0.01),
        _grouped_record("zernike_21", 1.0, -0.25),
        _grouped_record("zernike_21", -1.0, -0.10),
    ]

    result = interpret_r2_diagnostic(
        grouped,
        zero_max_abs_by_representation={"zernike_21": 0.0},
        zero_tolerance=1e-6,
        projection_warning_fraction=0.10,
        violation_limit=0.05,
    )

    diagnosis = result["by_representation"]["zernike_21"]
    assert diagnosis["primary_label"] == "EXCESSIVE_ACTION_AMPLITUDE_SUPPORTED"
    assert diagnosis["amplitude_check"]["supports_excessive_action_amplitude"]
    assert not result["retraining_authorized"]
    assert not result["s4d3_authorized"]


def test_interpretation_detects_residual_l2_projection_even_if_final_step_is_safe() -> None:
    grouped = [
        _grouped_record("zernike_36", 0.0, 0.0),
        _grouped_record("zernike_36", 0.25, -0.10),
        _grouped_record("zernike_36", 0.5, -0.20),
        _grouped_record(
            "zernike_36",
            1.0,
            -0.70,
            projection=0.02,
            residual_projection=0.95,
        ),
        _grouped_record("zernike_36", -1.0, -0.80),
    ]

    result = interpret_r2_diagnostic(
        grouped,
        zero_max_abs_by_representation={"zernike_36": 0.0},
        zero_tolerance=1e-6,
        projection_warning_fraction=0.10,
        violation_limit=0.05,
    )

    diagnosis = result["by_representation"]["zernike_36"]
    assert diagnosis["primary_label"] == "SAFETY_PROJECTION_CONFLICT_SUPPORTED"
    assert diagnosis["projection_and_safety_check"]["supports_projection_conflict"]


def test_formal_preflight_accepts_six_audited_read_only_checkpoints(
    tmp_path: Path,
) -> None:
    experiment_path = _project_path(CONFIG_PATH)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=False)
    settings["output_directory"] = str(tmp_path / "formal_diagnostic")

    result = preflight_s4_r2_residual_diagnostic(
        experiment_path,
        experiment,
        settings,
        quick=False,
    )

    assert result["status"] == "READY_FOR_USER_DIAGNOSTIC"
    assert result["checkpoint_count"] == 6
    assert result["rollout_count"] == 504
    assert not result["training_allowed"]
    assert result["checkpoints_are_read_only"]


def test_preflight_rejects_algorithm_change_flag() -> None:
    experiment_path = _project_path(CONFIG_PATH)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=True)
    changed = deepcopy(experiment)
    changed["metadata"]["allow_algorithm_change"] = True

    with pytest.raises(RuntimeError, match="mutation and scope-expansion"):
        preflight_s4_r2_residual_diagnostic(
            experiment_path,
            changed,
            settings,
            quick=True,
        )


def test_zero_residual_matches_independent_anchored_baseline_on_cpu() -> None:
    experiment = _load_yaml(_project_path(CONFIG_PATH))
    base_config, _ = load_s1_config(_project_path(experiment["environment_config"]))
    config = replace(
        base_config,
        num_modes=21,
        batch_size=2,
        episode_length=4,
    )
    profile = _profiles(experiment, ["nominal"])[0]
    condition = RobustnessCondition.from_mapping(
        {
            "id": "unit_zero",
            "regime": "frozen_stationary",
            "base_seed": 3195000,
            "wind_speed_mps": 0.5,
            "wind_direction_deg": 30.0,
            "slm_delay_frames": 2,
        }
    )
    config = profile.environment_config(condition.environment_config(config))
    experiment["action"]["num_modes"] = 21

    baseline = _rollout_anchored_baseline(
        experiment,
        config,
        condition,
        profile,
        4,
        torch.device("cpu"),
    )
    zero = _rollout_variant(
        actor=None,
        scale=0.0,
        experiment=experiment,
        config=config,
        condition=condition,
        profile=profile,
        steps=4,
        device=torch.device("cpu"),
    )

    for metric in (
        "power_in_bucket",
        "measured_power_in_bucket",
        "strehl",
        "phase_rmse",
        "violation_fraction",
    ):
        assert torch.equal(zero[metric], baseline[metric])


def test_high_dimensional_action_telemetry_detects_l2_projection_on_cpu() -> None:
    experiment = _load_yaml(_project_path(CONFIG_PATH))
    base_config, _ = load_s1_config(_project_path(experiment["environment_config"]))
    config = replace(
        base_config,
        num_modes=36,
        batch_size=2,
        episode_length=2,
    )
    profile = _profiles(experiment, ["nominal"])[0]
    condition = RobustnessCondition.from_mapping(
        {
            "id": "unit_projection",
            "regime": "frozen_stationary",
            "base_seed": 3196000,
            "wind_speed_mps": 0.5,
            "wind_direction_deg": 30.0,
            "slm_delay_frames": 2,
        }
    )
    config = profile.environment_config(condition.environment_config(config))
    experiment["action"]["num_modes"] = 36

    result = _rollout_variant(
        actor=_OnesActor(),  # type: ignore[arg-type]
        scale=1.0,
        experiment=experiment,
        config=config,
        condition=condition,
        profile=profile,
        steps=2,
        device=torch.device("cpu"),
    )

    assert torch.all(result["residual_l2_projection_fraction"] == 1)
    expected_budget = 10**0.5 * 0.05
    assert float(result["requested_residual_l2_rad"].max()) <= expected_budget + 1e-5
