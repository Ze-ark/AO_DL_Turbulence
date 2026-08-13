from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.rl.s4_r1_training import run_s4_r1_residual_sac


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "s4_residual_sac_r1_v1.yaml"


def _temporary_config(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    experiment = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    experiment["outputs"]["directory"] = str(tmp_path / "formal")
    experiment["outputs"]["quick_directory"] = str(tmp_path / "quick")
    path = tmp_path / "r1.yaml"
    path.write_text(
        yaml.safe_dump(experiment, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path, experiment


def test_r1_preflight_refuses_rerun_after_source_change(
    tmp_path: Path,
) -> None:
    path, _ = _temporary_config(tmp_path)
    with pytest.raises(RuntimeError, match="diagnostic source changed"):
        run_s4_r1_residual_sac(path, quick=True, preflight_only=True)


def test_r1_preflight_rejects_reward_change(tmp_path: Path) -> None:
    path, experiment = _temporary_config(tmp_path)
    experiment["reward"]["residual_action_weight"] = 0.001
    path.write_text(
        yaml.safe_dump(experiment, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="diagnostic source changed"):
        run_s4_r1_residual_sac(path, quick=True, preflight_only=True)


def test_r1_preflight_rejects_second_action_change(tmp_path: Path) -> None:
    path, experiment = _temporary_config(tmp_path)
    experiment["action"]["final_action_step_limit_rad"] = 0.10
    path.write_text(
        yaml.safe_dump(experiment, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="diagnostic source changed"):
        run_s4_r1_residual_sac(path, quick=True, preflight_only=True)


def test_r1_preflight_rejects_protected_seed_overlap(tmp_path: Path) -> None:
    path, experiment = _temporary_config(tmp_path)
    experiment["quick"]["training_environment_seed_bases"] = [2200000]
    path.write_text(
        yaml.safe_dump(experiment, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="diagnostic source changed"):
        run_s4_r1_residual_sac(path, quick=True, preflight_only=True)
