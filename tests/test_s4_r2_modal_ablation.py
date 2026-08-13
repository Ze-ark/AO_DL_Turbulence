from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from src.rl.s4_r2_modal_ablation import (
    ModalMaskActor,
    _effective_settings,
    interpret_modal_ablation,
    preflight_s4_r2_modal_ablation,
)
from src.rl.s4_training import _load_yaml, _project_path


CONFIG_PATH = Path("configs/experiments/s4_residual_sac_r2_modal_ablation_v1.yaml")


class _StaticActor:
    def __init__(self, action: torch.Tensor) -> None:
        self.action = action

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        return self.action.expand(state.shape[0], -1)


def _record(
    representation_id: str,
    variant: str,
    gain: float,
    *,
    violation: float = 0.01,
) -> dict[str, object]:
    return {
        "representation_id": representation_id,
        "variant": variant,
        "relative_power_gain": gain,
        "candidate": {
            "violation_fraction": {"mean": violation},
        },
        "paired_delta_candidate_minus_baseline": {
            "power_in_bucket": {"mean": gain},
            "strehl": {"mean": gain},
            "phase_rmse": {"mean": -gain},
        },
        "action_diagnostics": {
            "residual_l2_projection_fraction": {"mean": 0.0},
            "final_projection_fraction": {"mean": 0.0},
        },
    }


def test_modal_mask_actor_preserves_declared_subspace() -> None:
    action = torch.arange(1, 7, dtype=torch.float32).unsqueeze(0)
    actor = _StaticActor(action)
    state = torch.zeros(2, 10)

    all_modes = ModalMaskActor(
        actor,  # type: ignore[arg-type]
        mask="all",
        num_modes=6,
        anchor_modes=2,
    ).deterministic(state)
    anchor_only = ModalMaskActor(
        actor,  # type: ignore[arg-type]
        mask="anchor_only",
        num_modes=6,
        anchor_modes=2,
    ).deterministic(state)
    added_only = ModalMaskActor(
        actor,  # type: ignore[arg-type]
        mask="added_only",
        num_modes=6,
        anchor_modes=2,
    ).deterministic(state)

    assert torch.equal(all_modes, action.expand(2, -1))
    assert torch.equal(anchor_only[:, :2], all_modes[:, :2])
    assert torch.count_nonzero(anchor_only[:, 2:]) == 0
    assert torch.count_nonzero(added_only[:, :2]) == 0
    assert torch.equal(added_only[:, 2:], all_modes[:, 2:])


def test_interpretation_identifies_anchor_mode_interference() -> None:
    grouped = [
        _record("zernike_21", "zero_residual", 0.0),
        _record("zernike_21", "all_modes_quarter", -0.10),
        _record("zernike_21", "anchor_only_quarter", -0.12),
        _record("zernike_21", "added_modes_only_quarter", 0.03),
    ]

    result = interpret_modal_ablation(
        grouped,
        zero_max_abs_by_representation={"zernike_21": 0.0},
        zero_tolerance=1e-6,
        violation_limit=0.05,
    )

    diagnosis = result["by_representation"]["zernike_21"]
    assert diagnosis["primary_label"] == "ANCHOR_MODE_INTERFERENCE"
    assert diagnosis["added_only_has_positive_gain"]
    assert not result["training_authorized"]
    assert not result["s4d3_authorized"]


def test_interpretation_identifies_both_destructive_subspaces() -> None:
    grouped = [
        _record("zernike_36", "zero_residual", 0.0),
        _record("zernike_36", "all_modes_quarter", -0.60),
        _record("zernike_36", "anchor_only_quarter", -0.20),
        _record("zernike_36", "added_modes_only_quarter", -0.50),
    ]

    result = interpret_modal_ablation(
        grouped,
        zero_max_abs_by_representation={"zernike_36": 0.0},
        zero_tolerance=1e-6,
        violation_limit=0.05,
    )

    assert (
        result["by_representation"]["zernike_36"]["primary_label"]
        == "BOTH_ACTION_SUBSPACES_DESTRUCTIVE"
    )


def test_formal_preflight_accepts_paired_read_only_modal_ablation(
    tmp_path: Path,
) -> None:
    experiment_path = _project_path(CONFIG_PATH)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=False)
    settings["output_directory"] = str(tmp_path / "formal_modal_ablation")

    result = preflight_s4_r2_modal_ablation(
        experiment_path,
        experiment,
        settings,
        quick=False,
    )

    assert result["status"] == "READY_FOR_USER_DIAGNOSTIC"
    assert result["rollout_count"] == 396
    assert result["formal_seed_pairing_with_upstream"]
    assert not result["training_allowed"]
    assert result["checkpoints_are_read_only"]


def test_preflight_rejects_changed_ablation_scale() -> None:
    experiment_path = _project_path(CONFIG_PATH)
    experiment = _load_yaml(experiment_path)
    changed = deepcopy(experiment)
    changed["variants"][0]["scale"] = 0.5
    settings = _effective_settings(changed, quick=True)

    with pytest.raises(RuntimeError, match="three declared masks at 25%"):
        preflight_s4_r2_modal_ablation(
            experiment_path,
            changed,
            settings,
            quick=True,
        )
