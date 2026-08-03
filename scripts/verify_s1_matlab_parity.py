"""比较 MATLAB 与 PyTorch 的同一泰勒冻结流时间步。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime import resolve_device
from src.simulation.turbulence import advance_taylor_frozen_flow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="data/processed/s1_matlab_reference.h5")
    parser.add_argument("--tolerance", type=float, default=1e-10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with h5py.File(args.reference, "r") as handle:
        phase = np.asarray(handle["phase_initial"], dtype=np.float64).T.copy()
        innovation = np.asarray(handle["innovation"], dtype=np.float64).T.copy()
        expected = np.asarray(handle["phase_next"], dtype=np.float64).T.copy()
        sample_pitch = _scalar_attribute(handle, "sample_pitch_m")
        shift_x = _scalar_attribute(handle, "shift_x_m")
        shift_y = _scalar_attribute(handle, "shift_y_m")
        rho = _scalar_attribute(handle, "rho")

    device = resolve_device("cuda")
    actual = advance_taylor_frozen_flow(
        torch.as_tensor(phase, device=device),
        shift_x,
        shift_y,
        sample_pitch,
        rho,
        torch.as_tensor(innovation, device=device),
    )
    error = np.max(np.abs(actual.cpu().numpy() - expected))
    print(f"device={device}")
    print(f"maximum_absolute_error={error:.3e}")
    print(f"tolerance={args.tolerance:.3e}")
    if error > args.tolerance:
        raise SystemExit("S1 MATLAB-PyTorch parity check failed")
    print("S1 MATLAB-PyTorch parity check passed")


def _scalar_attribute(handle: h5py.File, name: str) -> float:
    """兼容 MATLAB 写出的 1×1 HDF5 数值属性。"""
    return float(np.asarray(handle.attrs[name]).reshape(-1)[0])


if __name__ == "__main__":
    main()
