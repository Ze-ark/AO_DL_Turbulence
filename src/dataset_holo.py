from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class HoloH5Dataset(Dataset):
    """Dataset for MATLAB-exported turbulence compensation HDF5 files."""

    def __init__(self, h5_path: str | Path, indices: list[int] | None = None) -> None:
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 dataset not found: {self.h5_path}")

        with h5py.File(self.h5_path, "r") as f:
            _require_dataset(f, "input/intensity_turb")
            _require_dataset(f, "input/phase_turb")
            _require_dataset(f, "target/intensity_clean")
            _require_dataset(f, "target/phase_clean")
            shape = tuple(int(v) for v in f["input/intensity_turb"].shape)
            meta_length = int(np.prod(f["meta/frame_id"].shape)) if "meta/frame_id" in f else None
            self._frame_axis_last = meta_length is not None and len(shape) == 3 and shape[-1] == meta_length
            n = int(shape[-1] if self._frame_axis_last else shape[0])
            for key in ("input/phase_turb", "target/intensity_clean", "target/phase_clean"):
                key_shape = tuple(int(v) for v in f[key].shape)
                key_length = int(key_shape[-1] if self._frame_axis_last else key_shape[0])
                if key_length != n:
                    raise ValueError(f"Dataset {key} length does not match input/intensity_turb")

        self.indices = list(range(n)) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = self.indices[item]
        with h5py.File(self.h5_path, "r") as f:
            intensity = _read_frame(f, "input/intensity_turb", index, self._frame_axis_last)
            phase = _read_frame(f, "input/phase_turb", index, self._frame_axis_last)
            target_intensity = _read_frame(f, "target/intensity_clean", index, self._frame_axis_last)
            target_phase = _read_frame(f, "target/phase_clean", index, self._frame_axis_last)
            frame_id = _read_meta_scalar(f, "meta/frame_id", index, default=index)
            turbulence_strength = _read_meta_scalar(f, "meta/turbulence_strength", index, default=np.nan)
            r0 = _read_meta_scalar(f, "meta/r0_or_equivalent", index, default=np.nan)

        _validate_arrays(intensity, phase, target_intensity, target_phase)
        intensity_norm = _normalize_intensity(intensity)
        phase_wrapped = _wrap_phase(phase)

        return {
            "input": torch.from_numpy(np.stack([intensity_norm, phase_wrapped], axis=0)),
            "input_intensity": torch.from_numpy(intensity[None, ...]),
            "input_phase": torch.from_numpy(phase_wrapped[None, ...]),
            "target_intensity": torch.from_numpy(target_intensity[None, ...]),
            "target_phase": torch.from_numpy(_wrap_phase(target_phase)[None, ...]),
            "meta": {
                "frame_id": int(frame_id),
                "turbulence_strength": float(turbulence_strength),
                "r0_or_equivalent": float(r0),
            },
        }


class RealComplexH5Dataset(Dataset):
    """Unlabeled real-experiment complex-field HDF5 dataset."""

    def __init__(self, h5_path: str | Path, indices: list[int] | None = None) -> None:
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"Real validation HDF5 dataset not found: {self.h5_path}")

        with h5py.File(self.h5_path, "r") as f:
            _require_dataset(f, "input/intensity_turb")
            _require_dataset(f, "input/phase_turb")
            shape = tuple(int(v) for v in f["input/intensity_turb"].shape)
            meta_length = int(np.prod(f["meta/frame_id"].shape)) if "meta/frame_id" in f else None
            self._frame_axis_last = meta_length is not None and len(shape) == 3 and shape[-1] == meta_length
            n = int(shape[-1] if self._frame_axis_last else shape[0])
            phase_shape = tuple(int(v) for v in f["input/phase_turb"].shape)
            phase_length = int(phase_shape[-1] if self._frame_axis_last else phase_shape[0])
            if phase_length != n:
                raise ValueError("Dataset input/phase_turb length does not match input/intensity_turb")

        self.indices = list(range(n)) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = self.indices[item]
        with h5py.File(self.h5_path, "r") as f:
            intensity = _read_frame(f, "input/intensity_turb", index, self._frame_axis_last)
            phase = _read_frame(f, "input/phase_turb", index, self._frame_axis_last)
            frame_id = _read_meta_scalar(f, "meta/frame_id", index, default=index)
            temperature = _read_meta_scalar(f, "meta/temperature", index, default=np.nan)

        _validate_real_arrays(intensity, phase)
        intensity_norm = _normalize_intensity(intensity)
        phase_wrapped = _wrap_phase(phase)

        return {
            "input": torch.from_numpy(np.stack([intensity_norm, phase_wrapped], axis=0)),
            "input_intensity": torch.from_numpy(intensity[None, ...]),
            "input_phase": torch.from_numpy(phase_wrapped[None, ...]),
            "meta": {
                "frame_id": int(frame_id),
                "temperature": float(temperature),
            },
        }


def split_indices(
    length: int,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, list[int]]:
    if length <= 0:
        raise ValueError("length must be positive")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be less than 1")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(length).tolist()
    train_end = int(round(length * train_fraction))
    val_end = train_end + int(round(length * val_fraction))
    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }


def _require_dataset(f: h5py.File, key: str) -> None:
    if key not in f:
        raise KeyError(f"Missing required HDF5 dataset: {key}")


def _read_meta_scalar(f: h5py.File, key: str, index: int, default: float | int) -> float | int:
    if key not in f:
        return default
    values = np.asarray(f[key]).reshape(-1)
    if index >= values.shape[0]:
        return default
    return values[index].item()


def _read_frame(f: h5py.File, key: str, index: int, frame_axis_last: bool) -> np.ndarray:
    if frame_axis_last:
        return np.asarray(f[key][:, :, index], dtype=np.float32)
    return np.asarray(f[key][index], dtype=np.float32)


def _validate_arrays(*arrays: np.ndarray) -> None:
    shape = arrays[0].shape
    for array in arrays:
        if array.shape != shape:
            raise ValueError(f"All fields must share shape {shape}; got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("Dataset contains NaN or Inf values")
    if (arrays[0] < 0).any() or (arrays[2] < 0).any():
        raise ValueError("Intensity arrays must be non-negative")


def _validate_real_arrays(intensity: np.ndarray, phase: np.ndarray) -> None:
    if intensity.shape != phase.shape:
        raise ValueError(f"Intensity and phase must share shape; got {intensity.shape} and {phase.shape}")
    if not np.isfinite(intensity).all() or not np.isfinite(phase).all():
        raise ValueError("Dataset contains NaN or Inf values")
    if (intensity < 0).any():
        raise ValueError("Intensity arrays must be non-negative")


def _normalize_intensity(intensity: np.ndarray) -> np.ndarray:
    max_value = float(np.max(intensity))
    if max_value <= 0:
        return np.zeros_like(intensity, dtype=np.float32)
    return (intensity / max_value).astype(np.float32)


def _wrap_phase(phase: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * phase)).astype(np.float32)
