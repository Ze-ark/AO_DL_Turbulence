from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from src.rl.residual_control import AnchoredResidualTrackingController
from src.rl.s4_r2_training import run_s4_r2_residual_sac
from src.rl.s4_training import _paired_episode_records
from src.simulation.controllers import TrackingLeakyIntegratorController


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "s4_residual_sac_r2_v1.yaml"


def _observation(batch: int = 3, modes: int = 21) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1234)
    residual = 0.2 * torch.randn(batch, modes, generator=generator)
    applied = 0.1 * torch.randn(batch, modes, generator=generator)
    metrics = torch.tensor([[0.4, 0.6]]).expand(batch, -1)
    return torch.cat((residual, applied, metrics), dim=-1)


def _controller(modes: int = 21) -> AnchoredResidualTrackingController:
    return AnchoredResidualTrackingController(
        num_modes=modes,
        anchor_modes=10,
        modal_limit_rad=3.0,
        history_frames=4,
        residual_action_limit_rad=0.05,
        final_action_step_limit_rad=0.15,
        gain=0.25,
        leak=0.10,
        tracking_gain=0.50,
    )


def _temporary_config(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    experiment = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    experiment["outputs"]["directory"] = str(tmp_path / "formal")
    experiment["outputs"]["quick_directory"] = str(tmp_path / "quick")
    path = tmp_path / "r2.yaml"
    path.write_text(
        yaml.safe_dump(experiment, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path, experiment


def test_anchored_controller_state_and_zero_residual_preserve_ten_mode_baseline() -> None:
    observation = _observation()
    controller = _controller()
    state = controller.reset(observation)
    assert state.shape == (3, 210)

    anchor_observation = torch.cat(
        (observation[:, :10], observation[:, 21:31], observation[:, -2:]), dim=-1
    )
    baseline = TrackingLeakyIntegratorController(
        num_modes=10,
        modal_limit_rad=3.0,
        gain=0.25,
        leak=0.10,
        tracking_gain=0.50,
        max_request_step_rad=0.15,
    )
    baseline.reset(3, torch.device("cpu"), torch.float32)
    expected = baseline.action(anchor_observation)
    actual = controller.compose_action(torch.zeros(3, 21)).final_delta_rad
    assert torch.equal(actual[:, :10], expected)
    assert torch.count_nonzero(actual[:, 10:]) == 0


def test_anchored_controller_enforces_shared_ten_mode_l2_budget() -> None:
    controller = _controller(modes=36)
    controller.reset(_observation(modes=36))
    action = controller.compose_action(torch.ones(3, 36))

    residual_norm = torch.linalg.vector_norm(action.requested_residual_rad, dim=-1)
    final_norm = torch.linalg.vector_norm(action.final_delta_rad, dim=-1)
    assert torch.all(residual_norm <= 10**0.5 * 0.05 + 1e-7)
    assert torch.all(final_norm <= 10**0.5 * 0.15 + 1e-7)
    assert torch.all(action.final_delta_rad.abs() <= 0.15 + 1e-7)


def test_anchored_controller_keeps_step_and_request_limits_over_time() -> None:
    controller = _controller(modes=36)
    observation = _observation(modes=36)
    controller.reset(observation)
    for _ in range(80):
        action = controller.compose_action(torch.ones(3, 36))
        assert torch.all(action.final_delta_rad.abs() <= 0.15 + 1e-6)
        assert torch.all(
            torch.linalg.vector_norm(action.final_delta_rad, dim=-1)
            <= 10**0.5 * 0.15 + 1e-6
        )
        assert torch.all(controller.requested_modal.abs() <= 3.0 + 1e-6)
        assert torch.all(
            torch.linalg.vector_norm(controller.requested_modal, dim=-1)
            <= 10**0.5 * 3.0 + 1e-6
        )
        controller.advance_observation(observation)


def test_r2_preflight_accepts_only_passing_representations_without_cuda(
    tmp_path: Path,
) -> None:
    path, _ = _temporary_config(tmp_path)
    result = run_s4_r2_residual_sac(path, quick=True, preflight_only=True)

    assert result["status"] == "READY_FOR_QUICK_SMOKE"
    assert result["total_policy_runs"] == 2
    assert result["representation_contracts"] == [
        {"id": "zernike_21", "num_modes": 21, "state_size": 210, "action_size": 21},
        {"id": "zernike_36", "num_modes": 36, "state_size": 360, "action_size": 36},
    ]
    assert result["scientific_contract"]["frozen_baseline_anchor_modes"] == 10
    assert result["scientific_contract"]["per_episode_validation_records"] is True
    assert result["sealed_s4d3_access"] is False


def test_r2_preflight_rejects_total_budget_increase(tmp_path: Path) -> None:
    path, experiment = _temporary_config(tmp_path)
    experiment["action"]["total_budget_anchor_modes"] = 21
    path.write_text(
        yaml.safe_dump(experiment, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="total action budget anchor changed"):
        run_s4_r2_residual_sac(path, quick=True, preflight_only=True)


def test_episode_records_keep_pairing_keys_and_metric_deltas() -> None:
    candidate = {
        "power_in_bucket": torch.tensor([0.7, 0.8]),
        "measured_power_in_bucket": torch.tensor([0.69, 0.79]),
        "strehl": torch.tensor([0.6, 0.7]),
        "phase_rmse": torch.tensor([0.5, 0.4]),
        "violation_fraction": torch.tensor([0.01, 0.02]),
    }
    baseline = {key: value - 0.1 for key, value in candidate.items()}
    records = _paired_episode_records(
        "nominal", "r2_val", 2860000, candidate, baseline
    )

    assert [item["episode_seed"] for item in records] == [2860000, 2860001]
    assert records[0]["profile_id"] == "nominal"
    assert records[0]["condition_id"] == "r2_val"
    assert records[0]["delta_power_in_bucket"] == pytest.approx(0.1)
