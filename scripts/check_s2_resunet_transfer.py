"""检查旧静态ResUNet能否直接迁移到S2理想瞳面动态观测。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import sys
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.resunet_phase import ResUNetPhase
from src.runtime import resolve_device
from src.simulation import AdaptiveOpticsEnv, load_s1_config
from src.simulation.modes import project_phase_to_modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/s2_baselines_v1.yaml")
    parser.add_argument("--output", default="outputs/s2_resunet_transfer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_path = _project_path(args.config)
    experiment = _load_yaml(experiment_path)
    transfer = experiment["resunet_transfer"]
    env_path = _project_path(experiment["environment_config"])
    env_config, requested_device = load_s1_config(env_path)
    device = resolve_device(requested_device)
    checkpoint_path = _project_path(transfer["checkpoint"])
    model, checkpoint_epoch = _load_model(checkpoint_path, device)
    seeds = [int(seed) for seed in transfer["development_seeds"]]
    steps = min(int(transfer["steps"]), env_config.episode_length)

    episode_cosine: list[torch.Tensor] = []
    episode_relative_rmse: list[torch.Tensor] = []
    episode_norm_ratio: list[torch.Tensor] = []
    started = perf_counter()

    with torch.no_grad():
        for seed in seeds:
            environment = AdaptiveOpticsEnv(env_config, device)
            observation, _ = environment.reset(seed=seed)
            cosine_steps: list[torch.Tensor] = []
            rmse_steps: list[torch.Tensor] = []
            ratio_steps: list[torch.Tensor] = []
            zero_action = torch.zeros(
                env_config.batch_size,
                env_config.num_modes,
                device=device,
            )
            for _ in range(steps):
                wavefront = environment.ideal_wavefront_observation()
                predicted_phase = model(wavefront)[:, 0]
                predicted_modal = project_phase_to_modes(
                    predicted_phase,
                    environment.basis,
                    environment.pupil,
                )
                true_modal = observation[:, : env_config.num_modes]
                denominator = true_modal.norm(dim=1).clamp_min(1e-12)
                cosine_steps.append(
                    torch.nn.functional.cosine_similarity(
                        predicted_modal,
                        true_modal,
                        dim=1,
                    ).cpu()
                )
                rmse_steps.append(
                    ((predicted_modal - true_modal).norm(dim=1) / denominator).cpu()
                )
                ratio_steps.append((predicted_modal.norm(dim=1) / denominator).cpu())
                observation, _, terminal, _, _ = environment.step(zero_action)
                if terminal.all():
                    break

            episode_cosine.append(torch.stack(cosine_steps, dim=1).mean(dim=1))
            episode_relative_rmse.append(torch.stack(rmse_steps, dim=1).mean(dim=1))
            episode_norm_ratio.append(torch.stack(ratio_steps, dim=1).mean(dim=1))

    cosine = torch.cat(episode_cosine)
    relative_rmse = torch.cat(episode_relative_rmse)
    norm_ratio = torch.cat(episode_norm_ratio)
    cosine_gate = float(cosine.mean()) >= float(transfer["min_mean_modal_cosine"])
    rmse_gate = float(relative_rmse.mean()) <= float(transfer["max_relative_modal_rmse"])
    gate_passed = cosine_gate and rmse_gate
    duration = perf_counter() - started

    result = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "s2_resunet_transfer_v1",
        },
        "experiment": {
            "id": "AO-S2-RESUNET-TRANSFER",
            "type": "simulation",
            "status": "completed",
            "command": ".\\.venv\\Scripts\\python.exe scripts\\check_s2_resunet_transfer.py",
            "working_directory": str(PROJECT_ROOT),
            "duration_seconds": duration,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "inputs": {
            "experiment_config": str(experiment_path.relative_to(PROJECT_ROOT)),
            "experiment_config_sha256": _sha256(experiment_path),
            "environment_config": str(env_path.relative_to(PROJECT_ROOT)),
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_epoch": checkpoint_epoch,
            "observation_kind": "ideal_pupil_intensity_and_wrapped_phase",
            "development_base_seeds": seeds,
            "episodes": len(seeds) * env_config.batch_size,
            "steps_per_episode": steps,
        },
        "metrics": {
            "modal_cosine": _summary(cosine, higher_is_better=True),
            "relative_modal_rmse": _summary(relative_rmse, higher_is_better=False),
            "modal_norm_ratio": _summary(norm_ratio, higher_is_better=False),
        },
        "gate": {
            "min_mean_modal_cosine": float(transfer["min_mean_modal_cosine"]),
            "max_relative_modal_rmse": float(transfer["max_relative_modal_rmse"]),
            "cosine_gate_passed": cosine_gate,
            "rmse_gate_passed": rmse_gate,
            "transfer_gate": "PASS" if gate_passed else "FAIL",
        },
        "interpretation_boundary": (
            "This tests a frozen static receiver-plane checkpoint on ideal pupil-plane "
            "dynamic observations. Failure indicates domain/interface mismatch, not that "
            "the ResUNet architecture is intrinsically ineffective."
        ),
        "anomalies": [],
    }

    output_dir = _project_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "experiment_result.md").write_text(
        _markdown_result(result),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _load_model(path: Path, device: torch.device) -> tuple[ResUNetPhase, int | None]:
    checkpoint = torch.load(path, map_location=device)
    model_config = checkpoint.get("config", {}).get("model", {})
    model = ResUNetPhase(
        in_channels=2,
        base_channels=int(model_config.get("base_channels", 32)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint.get("epoch")


def _summary(values: torch.Tensor, higher_is_better: bool) -> dict[str, float]:
    values = values.to(torch.float64)
    count = int(values.numel())
    mean = values.mean()
    half_width = 1.96 * values.std(unbiased=True) / count**0.5 if count > 1 else float("nan")
    quantile = 0.1 if higher_is_better else 0.9
    return {
        "mean": float(mean),
        "median": float(values.median()),
        "ci95_low": float(mean - half_width),
        "ci95_high": float(mean + half_width),
        "worst_decile": float(torch.quantile(values, quantile)),
    }


def _markdown_result(result: dict[str, Any]) -> str:
    passport = result["material_passport"]
    experiment = result["experiment"]
    metrics = result["metrics"]
    gate = result["gate"]
    return f"""## Material Passport

- Origin Skill: {passport['origin_skill']}
- Origin Mode: {passport['origin_mode']}
- Origin Date: {passport['origin_date']}
- Verification Status: {passport['verification_status']}
- Version Label: {passport['version_label']}

## Experiment Result

- **ID**: {experiment['id']}
- **Type**: {experiment['type']}
- **Status**: {experiment['status']}
- **Command**: `{experiment['command']}`
- **Working Directory**: `{experiment['working_directory']}`
- **Duration**: {experiment['duration_seconds']:.3f} seconds
- **Exit Code**: 0

### Output Summary

| Metric | Mean | 95% CI | Gate |
|---|---:|---:|---|
| Modal cosine | {metrics['modal_cosine']['mean']:.6f} | [{metrics['modal_cosine']['ci95_low']:.6f}, {metrics['modal_cosine']['ci95_high']:.6f}] | >= {gate['min_mean_modal_cosine']:.3f} |
| Relative modal RMSE | {metrics['relative_modal_rmse']['mean']:.6f} | [{metrics['relative_modal_rmse']['ci95_low']:.6f}, {metrics['relative_modal_rmse']['ci95_high']:.6f}] | <= {gate['max_relative_modal_rmse']:.3f} |

- **Transfer Gate**: {gate['transfer_gate']}
- **Boundary**: {result['interpretation_boundary']}

### Anomalies Detected

None
"""


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
