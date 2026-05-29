from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_holo import HoloH5Dataset, split_indices
from src.losses import compensation_loss
from src.models.resunet_phase import ResUNetPhase
from src.runtime import resolve_device


def train(config_path: str | Path) -> Path:
    config_path = _resolve_project_path(config_path)
    config = _load_config(config_path)
    device = resolve_device(config["runtime"].get("device", "cpu"))
    h5_path = _resolve_project_path(config["data"].get("simulated_h5_path", config["data"].get("h5_path")))
    dataset = HoloH5Dataset(h5_path)
    splits = split_indices(
        len(dataset),
        train_fraction=config["data"].get("train_fraction", 0.8),
        val_fraction=config["data"].get("val_fraction", 0.0),
        seed=config["data"].get("seed", 42),
    )
    train_loader = DataLoader(
        HoloH5Dataset(h5_path, indices=splits["train"]),
        batch_size=config["train"].get("batch_size", 4),
        shuffle=True,
    )
    test_loader = DataLoader(
        HoloH5Dataset(h5_path, indices=splits["test"]),
        batch_size=config["train"].get("batch_size", 4),
        shuffle=False,
    )

    model = ResUNetPhase(base_channels=config["model"].get("base_channels", 32)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"].get("learning_rate", 1e-3),
        weight_decay=config["train"].get("weight_decay", 1e-4),
    )

    output_dir = _resolve_project_path(config["train"].get("output_dir", "outputs/train_sim_gaussian_v1"))
    checkpoint_dir = _resolve_project_path(config["train"].get("checkpoint_dir", "checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "loss_history.csv"
    best_path = checkpoint_dir / "sim_gaussian_v1_best.pt"
    best_val = float("inf")

    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_total", "test_total"])
        for epoch in range(1, config["train"].get("epochs", 2) + 1):
            train_loss = _run_epoch(model, train_loader, device, optimizer)
            test_loss = _run_epoch(model, test_loader, device, optimizer=None)
            writer.writerow([epoch, train_loss, test_loss])
            if test_loss < best_val:
                best_val = test_loss
                torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, best_path)
            print(f"epoch={epoch} train_total={train_loss:.6f} test_total={test_loss:.6f}")

    return best_path


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        with torch.set_grad_enabled(training):
            phi_corr = model(batch["input"])
            losses = compensation_loss(
                phi_corr=phi_corr,
                input_intensity=batch["input_intensity"],
                input_phase=batch["input_phase"],
                target_intensity=batch["target_intensity"],
                target_phase=batch["target_phase"],
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                optimizer.step()
        batch_size = int(batch["input"].shape[0])
        total += float(losses["total"].detach().cpu().item()) * batch_size
        count += batch_size
    return total / max(count, 1)


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AO-DL phase compensation.")
    parser.add_argument("--config", default="configs/sim_gaussian_v1.yaml")
    args = parser.parse_args()
    print(f"saved_best_checkpoint={train(args.config)}")


if __name__ == "__main__":
    main()
