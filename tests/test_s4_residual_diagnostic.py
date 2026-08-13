from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from src.rl.s4_diagnostic import (
    _load_actor,
    interpret_diagnostic,
    run_s4_residual_diagnostic,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "s4_residual_sac_diagnostic_v1.yaml"


def _grouped_record(
    scale: float,
    power_gain: float,
    reward: float,
    *,
    cancellation: float = 0.0,
    cosine: float = 0.0,
) -> dict[str, object]:
    return {
        "variant": f"scale_{scale}",
        "scale": scale,
        "relative_power_gain": power_gain,
        "candidate": {"power_in_bucket": {"mean": 0.7 + power_gain}},
        "action_diagnostics": {
            "training_style_reward": {"mean": reward},
            "cancellation_fraction": {"mean": cancellation},
            "baseline_residual_cosine": {"mean": cosine},
        },
    }


def test_interpretation_separates_wiring_amplitude_and_direction_evidence() -> None:
    grouped = [
        _grouped_record(0.0, 0.0, 0.70),
        _grouped_record(0.25, -0.001, 0.699),
        _grouped_record(0.5, -0.002, 0.698),
        _grouped_record(1.0, -0.005, 0.695, cancellation=0.7, cosine=-0.3),
        _grouped_record(-1.0, 0.003, 0.703),
    ]
    result = interpret_diagnostic(grouped, zero_max_abs=0.0, zero_tolerance=1e-7)

    assert result["wiring_check"]["status"] == "PASS"
    assert result["amplitude_check"]["supports_excessive_action_amplitude"] is True
    assert result["direction_check"]["supports_wrong_direction"] is True
    assert result["s4d3_authorized"] is False
    assert result["retraining_authorized"] is False


def test_best_checkpoint_actor_loads_read_only_on_cpu() -> None:
    actor = _load_actor(
        {
            "path": "outputs/s4_residual_sac_v1/policy_seed_4101/checkpoint_best.pt",
        },
        torch.device("cpu"),
    )
    state = torch.zeros(2, 100)
    action = actor.deterministic(state)

    assert action.shape == (2, 10)
    assert torch.isfinite(action).all()
    assert all(parameter.requires_grad is False for parameter in actor.parameters())


def test_diagnostic_preflight_refuses_rerun_after_source_change(
    tmp_path: Path,
) -> None:
    experiment = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    experiment["outputs"]["quick_directory"] = str(tmp_path / "quick_diagnostic")
    temporary_config = tmp_path / "diagnostic.yaml"
    temporary_config.write_text(
        yaml.safe_dump(experiment, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="S4-D2 training source changed"):
        run_s4_residual_diagnostic(
            temporary_config,
            quick=True,
            preflight_only=True,
        )
