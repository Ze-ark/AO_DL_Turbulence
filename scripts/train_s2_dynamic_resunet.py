"""训练并评价S2单帧动态ResUNet模态感知基线。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import sys
from typing import Any

import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.resunet_phase import ResUNetPhase
from src.runtime import resolve_device
from src.simulation import AdaptiveOpticsEnv, load_s1_config
from src.simulation.resunet_training import (
    WavefrontSupervisedData,
    generate_wavefront_supervised_data,
    modal_prediction_metrics,
    resunet_phase_modal_loss,
)
from src.training_progress import (
    advance_to,
    counted_progress,
    progress_bar,
    progress_message,
    update_progress,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/s2_baselines_v1.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_path = _project_path(args.config)
    experiment = _load_yaml(experiment_path)
    settings = experiment["resunet_dynamic"]
    env_path = _project_path(experiment["environment_config"])
    env_config, requested_device = load_s1_config(env_path)
    device = resolve_device(requested_device)
    _enable_determinism(20260803)
    started = perf_counter()
    progress_message(
        f"训练设备：{torch.cuda.get_device_name(device)}；"
        f"轮数：{int(settings['epochs'])}；批大小：{int(settings['batch_size'])}"
    )

    train_data = _generate_data_with_progress(
        "生成训练数据",
        env_config,
        device,
        [int(value) for value in settings["train_base_seeds"]],
        int(settings["frames_per_episode"]),
        float(settings["random_action_std_rad"]),
    )
    validation_data = _generate_data_with_progress(
        "生成验证数据",
        env_config,
        device,
        [int(value) for value in settings["validation_base_seeds"]],
        int(settings["frames_per_episode"]),
        float(settings["random_action_std_rad"]),
    )
    dataset_path = _project_path(settings["dataset_h5"])
    _export_dataset(dataset_path, train_data, validation_data, experiment_path, env_path)

    reference_environment = AdaptiveOpticsEnv(env_config, device)
    model = ResUNetPhase(
        in_channels=2,
        base_channels=int(settings["base_channels"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    shuffle_generator = torch.Generator().manual_seed(20260803)
    train_loader = DataLoader(
        TensorDataset(train_data.inputs, train_data.target_phase),
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        generator=shuffle_generator,
    )
    validation_loader = DataLoader(
        TensorDataset(validation_data.inputs, validation_data.target_phase),
        batch_size=int(settings["batch_size"]),
        shuffle=False,
    )
    checkpoint_path = _project_path(settings["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = _project_path(settings["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_validation_rmse = float("inf")
    best_epoch = 0
    total_epochs = int(settings["epochs"])

    epochs = progress_bar(
        range(1, total_epochs + 1),
        description="模型训练总进度",
        unit="轮",
    )
    for epoch in epochs:
        train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            reference_environment,
            float(settings["modal_loss_weight"]),
            optimizer,
            epoch=epoch,
            total_epochs=total_epochs,
            stage="训练",
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            device,
            reference_environment,
            float(settings["modal_loss_weight"]),
            optimizer=None,
            epoch=epoch,
            total_epochs=total_epochs,
            stage="验证",
        )
        row = {
            "epoch": epoch,
            "train_total": train_metrics["total"],
            "train_modal_rmse": train_metrics["relative_modal_rmse"],
            "validation_total": validation_metrics["total"],
            "validation_modal_cosine": validation_metrics["modal_cosine"],
            "validation_modal_rmse": validation_metrics["relative_modal_rmse"],
        }
        history.append(row)
        _write_history(output_dir / "loss_history.csv", history)
        improved = validation_metrics["relative_modal_rmse"] < best_validation_rmse
        if improved:
            best_validation_rmse = validation_metrics["relative_modal_rmse"]
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "experiment_config": experiment,
                    "environment_config": env_config.__dict__,
                    "epoch": epoch,
                    "observation_kind": "ideal_pupil_intensity_and_wrapped_phase",
                    "target_kind": "controllable_residual_modal_phase",
                },
                checkpoint_path,
            )
            progress_message(
                f"第 {epoch}/{total_epochs} 轮保存新的最佳权重："
                f"验证相对模态误差={best_validation_rmse:.6f}"
            )
        update_progress(
            epochs,
            device=device,
            metrics={
                "训练损失": train_metrics["total"],
                "验证误差": validation_metrics["relative_modal_rmse"],
                "最佳误差": best_validation_rmse,
            },
        )
        progress_message(json.dumps(row, ensure_ascii=False))

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    test_data = _generate_data_with_progress(
        "生成感知测试数据",
        env_config,
        device,
        [int(value) for value in settings["perception_test_base_seeds"]],
        int(settings["frames_per_episode"]),
        float(settings["random_action_std_rad"]),
    )
    test_metrics = _evaluate_by_episode(
        model,
        test_data,
        device,
        reference_environment,
        int(settings["batch_size"]),
    )
    cosine_passed = test_metrics["modal_cosine"]["mean"] >= float(
        settings["min_test_modal_cosine"]
    )
    rmse_passed = test_metrics["relative_modal_rmse"]["mean"] <= float(
        settings["max_test_relative_modal_rmse"]
    )
    perception_gate = "PASS" if cosine_passed and rmse_passed else "FAIL"
    duration = perf_counter() - started
    _write_history(output_dir / "loss_history.csv", history)

    result = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "s2_dynamic_resunet_v1",
        },
        "experiment": {
            "id": "AO-S2-DYNAMIC-RESUNET",
            "type": "training",
            "status": "completed",
            "command": ".\\.venv\\Scripts\\python.exe scripts\\train_s2_dynamic_resunet.py",
            "working_directory": str(PROJECT_ROOT),
            "duration_seconds": duration,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "inputs": {
            "experiment_config": str(experiment_path.relative_to(PROJECT_ROOT)),
            "experiment_config_sha256": _sha256(experiment_path),
            "environment_config": str(env_path.relative_to(PROJECT_ROOT)),
            "dataset_h5": str(dataset_path.relative_to(PROJECT_ROOT)),
            "dataset_sha256": _sha256(dataset_path),
            "train_samples": len(train_data.inputs),
            "validation_samples": len(validation_data.inputs),
            "test_samples": len(test_data.inputs),
            "train_episode_seeds": torch.unique(train_data.episode_seed).tolist(),
            "validation_episode_seeds": torch.unique(validation_data.episode_seed).tolist(),
            "test_episode_seeds": torch.unique(test_data.episode_seed).tolist(),
        },
        "training": {
            "best_epoch": best_epoch,
            "best_validation_relative_modal_rmse": best_validation_rmse,
            "history": history,
        },
        "test_metrics_by_episode": test_metrics,
        "gate": {
            "min_test_modal_cosine": float(settings["min_test_modal_cosine"]),
            "max_test_relative_modal_rmse": float(settings["max_test_relative_modal_rmse"]),
            "cosine_gate_passed": cosine_passed,
            "rmse_gate_passed": rmse_passed,
            "perception_gate": perception_gate,
        },
        "outputs": {
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "loss_history": str((output_dir / "loss_history.csv").relative_to(PROJECT_ROOT)),
        },
        "interpretation_boundary": (
            "This is a memoryless ResUNet trained on ideal simulated pupil-plane observations. "
            "It is not the old receiver-plane model, a temporal model, an RL policy, or real-SLM evidence."
        ),
        "anomalies": [],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "experiment_result.md").write_text(
        _markdown_result(result),
        encoding="utf-8",
    )
    print(json.dumps({"test_metrics_by_episode": test_metrics, "gate": result["gate"]}, ensure_ascii=False))


def _generate_data_with_progress(
    description: str,
    env_config: Any,
    device: torch.device,
    base_seeds: list[int],
    frames_per_episode: int,
    random_action_std_rad: float,
) -> WavefrontSupervisedData:
    """生成完整回合数据，并在 IDE 终端显示当前帧组进度和 ETA。"""
    total = len(base_seeds) * frames_per_episode
    with counted_progress(total=total, description=description, unit="步") as bar:
        return generate_wavefront_supervised_data(
            env_config,
            device,
            base_seeds,
            frames_per_episode,
            random_action_std_rad,
            progress_callback=lambda completed, _: advance_to(bar, completed),
        )


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    environment: AdaptiveOpticsEnv,
    modal_weight: float,
    optimizer: torch.optim.Optimizer | None,
    *,
    epoch: int,
    total_epochs: int,
    stage: str,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "modal_cosine": 0.0, "relative_modal_rmse": 0.0}
    count = 0
    batches = progress_bar(
        loader,
        description=f"{stage} {epoch}/{total_epochs}",
        unit="批",
        leave=False,
    )
    for inputs, target in batches:
        inputs = inputs.to(device)
        target = target.to(device)
        with torch.set_grad_enabled(training):
            predicted = model(inputs)
            losses = resunet_phase_modal_loss(
                predicted,
                target,
                environment.basis,
                environment.pupil,
                modal_weight,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                optimizer.step()
        metrics = modal_prediction_metrics(
            predicted.detach(),
            target,
            environment.basis,
            environment.pupil,
        )
        batch_size = inputs.shape[0]
        totals["total"] += float(losses["total"].detach()) * batch_size
        totals["modal_cosine"] += float(metrics["modal_cosine"].mean()) * batch_size
        totals["relative_modal_rmse"] += float(metrics["relative_modal_rmse"].mean()) * batch_size
        count += batch_size
        update_progress(
            batches,
            device=device,
            metrics={
                "损失": totals["total"] / count,
                "模态误差": totals["relative_modal_rmse"] / count,
            },
        )
    return {key: value / count for key, value in totals.items()}


def _evaluate_by_episode(
    model: torch.nn.Module,
    data: WavefrontSupervisedData,
    device: torch.device,
    environment: AdaptiveOpticsEnv,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    loader = DataLoader(
        TensorDataset(data.inputs, data.target_phase),
        batch_size=batch_size,
        shuffle=False,
    )
    cosine_parts: list[torch.Tensor] = []
    rmse_parts: list[torch.Tensor] = []
    with torch.no_grad():
        batches = progress_bar(
            loader,
            description="最终感知测试",
            unit="批",
            leave=True,
        )
        for inputs, target in batches:
            predicted = model(inputs.to(device))
            metrics = modal_prediction_metrics(
                predicted,
                target.to(device),
                environment.basis,
                environment.pupil,
            )
            cosine_parts.append(metrics["modal_cosine"].cpu())
            rmse_parts.append(metrics["relative_modal_rmse"].cpu())
            update_progress(
                batches,
                device=device,
                metrics={
                    "模态相似度": float(metrics["modal_cosine"].mean()),
                    "模态误差": float(metrics["relative_modal_rmse"].mean()),
                },
            )
    cosine = _episode_means(torch.cat(cosine_parts), data.episode_seed)
    rmse = _episode_means(torch.cat(rmse_parts), data.episode_seed)
    return {
        "modal_cosine": _summary(cosine, higher_is_better=True),
        "relative_modal_rmse": _summary(rmse, higher_is_better=False),
    }


def _episode_means(values: torch.Tensor, episode_seed: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [values[episode_seed == seed].mean() for seed in torch.unique(episode_seed)]
    )


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


def _export_dataset(
    path: Path,
    train: WavefrontSupervisedData,
    validation: WavefrontSupervisedData,
    experiment_path: Path,
    environment_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["stage"] = "S2"
        handle.attrs["observation_kind"] = "ideal_pupil_intensity_and_wrapped_phase"
        handle.attrs["target_kind"] = "controllable_residual_modal_phase"
        handle.attrs["experiment_config"] = str(experiment_path.relative_to(PROJECT_ROOT))
        handle.attrs["environment_config"] = str(environment_path.relative_to(PROJECT_ROOT))
        for split, data in (("train", train), ("validation", validation)):
            handle.create_dataset(f"{split}/input", data=data.inputs.numpy(), compression="gzip")
            handle.create_dataset(
                f"{split}/target_phase", data=data.target_phase.numpy(), compression="gzip"
            )
            handle.create_dataset(f"{split}/episode_seed", data=data.episode_seed.numpy())
            handle.create_dataset(f"{split}/step_id", data=data.step_id.numpy())


def _write_history(path: Path, history: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _markdown_result(result: dict[str, Any]) -> str:
    passport = result["material_passport"]
    experiment = result["experiment"]
    metrics = result["test_metrics_by_episode"]
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

| Metric | Episode Mean | 95% CI | Gate |
|---|---:|---:|---|
| Modal cosine | {metrics['modal_cosine']['mean']:.6f} | [{metrics['modal_cosine']['ci95_low']:.6f}, {metrics['modal_cosine']['ci95_high']:.6f}] | >= {gate['min_test_modal_cosine']:.3f} |
| Relative modal RMSE | {metrics['relative_modal_rmse']['mean']:.6f} | [{metrics['relative_modal_rmse']['ci95_low']:.6f}, {metrics['relative_modal_rmse']['ci95_high']:.6f}] | <= {gate['max_test_relative_modal_rmse']:.3f} |

- **Perception Gate**: {gate['perception_gate']}
- **Boundary**: {result['interpretation_boundary']}

### Anomalies Detected

None
"""


def _enable_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


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
