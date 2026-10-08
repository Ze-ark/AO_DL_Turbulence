"""O1 已知答案技术验证；无训练、动态环境、策略推理或真实数据访问。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml

from observation_bridge.adapter import (
    FieldOrientation, FrozenModalObservationBridge, O1Tolerances,
    SyntheticFieldContract, orient_complex_field,
)
from src.simulation.modes import project_phase_to_modes, synthesize_phase


def preflight(path: str | Path) -> tuple[dict[str, Any], FrozenModalObservationBridge, Path]:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    fixed = dict(schema="observation_bridge_o1_v1", scope="deterministic_synthetic_unit_verification",
                 device="cpu", seed=831001, grid_size=64, pupil_radius_fraction=0.4,
                 num_modes=21, coefficient_tolerance_rad=1e-5)
    if any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k, v in fixed.items()):
        raise ValueError("O1 fixed unit scope/geometry/seed/tolerance changed")
    if set(cfg) != set(fixed) | {"output_directory", "contract", "tolerances"}:
        raise ValueError("unknown O1 configuration fields")
    contract = dict(cfg["contract"])
    contract["orientation"] = FieldOrientation(**contract["orientation"])
    bridge = FrozenModalObservationBridge(SyntheticFieldContract(**contract), O1Tolerances(**cfg["tolerances"]))
    if bridge.contract.orientation != FieldOrientation(False, False, False, 1):
        raise ValueError("O1 canonical fixtures require explicitly canonical orientation")
    output = (ROOT / cfg["output_directory"]).resolve()
    output_root = (ROOT / "outputs").resolve()
    if output == output_root or not output.is_relative_to(output_root):
        raise ValueError("O1 output must be a new child of workspace outputs")
    if output.exists():
        raise FileExistsError(f"preserve existing O1 output: {output}")
    return cfg, bridge, output


def known_field(coefficients: torch.Tensor, bridge: FrozenModalObservationBridge,
                *, constant: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    phase = synthesize_phase(coefficients.double(), bridge.basis.double()) + constant
    # 独立检查已知答案的真实邻点差，而不是据包裹相位声称排除了空间混叠。
    pupil = bridge.pupil
    jumps = torch.cat((
        (phase[:, 1:] - phase[:, :-1])[:, pupil[1:] & pupil[:-1]].flatten(),
        (phase[:, :, 1:] - phase[:, :, :-1])[:, pupil[:, 1:] & pupil[:, :-1]].flatten(),
    ))
    if float(jumps.abs().max()) > bridge.tolerances.max_neighbor_jump_rad:
        raise ValueError("known fixture is spatially undersampled for O1")
    return torch.polar(torch.ones_like(phase), phase), phase


def check_cases(cfg: dict[str, Any], bridge: FrozenModalObservationBridge) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(cfg["seed"])
    cases: dict[str, Any] = {}
    canonical = torch.complex(torch.arange(15).reshape(1, 3, 5).double(),
                              torch.arange(30, 45).reshape(1, 3, 5).double())
    raw = canonical.conj().flip(-1).flip(-2).transpose(-2, -1)
    if not torch.equal(orient_complex_field(raw, FieldOrientation(True, True, True, -1)), canonical):
        raise RuntimeError("asymmetric non-square orientation fixture failed")
    cases["non_square_orientation"] = {"shape": [1, 3, 5], "exact_match": True}
    pulses = torch.cat((torch.eye(21), -torch.eye(21))) * 0.05
    mixed = torch.randn(2, 21, generator=generator) * 0.02
    wrapped = torch.zeros(1, 21)
    wrapped[0, 0], wrapped[0, 2], wrapped[0, 10] = 4.0, 0.7, 0.1
    for name, coefficients, constant in (
        ("signed_21_mode_pulses", pulses, 0.0),
        ("seeded_unwrapped_mixture", mixed, 0.0),
        ("wrapped_smooth_phase", wrapped, 0.0),
        ("wrapped_with_constant_reference", wrapped, 3.7),
    ):
        field, phase = known_field(coefficients, bridge, constant=constant)
        measured = bridge.measure(field)
        error = float((measured.residual_rad - coefficients).abs().max())
        if error > cfg["coefficient_tolerance_rad"]:
            raise RuntimeError(f"O1 coefficient tolerance failed: {name}={error}")
        if name.startswith("wrapped") and not bool((phase[:, bridge.pupil].abs() > torch.pi).any()):
            raise RuntimeError("wrapped fixture never crosses pi")
        cases[name] = {
            "samples": len(coefficients), "max_coefficient_error_rad": error,
            "max_fit_rmse_rad": float(measured.fit_rmse_rad.max()),
            "max_true_phase_abs_rad": float(phase[:, bridge.pupil].abs().max()),
            "max_neighbor_jump_rad": float(measured.max_neighbor_jump_rad.max()),
        }
        if name == "wrapped_smooth_phase":
            naive = project_phase_to_modes(field.angle(), bridge.basis.double(), bridge.pupil)
            cases[name]["naive_wrapped_projection_error_rad"] = float((naive - coefficients).abs().max())
    return cases


def run(path: str | Path, *, preflight_only: bool = False) -> dict[str, Any]:
    cfg, bridge, output = preflight(path)
    report = dict(scope=cfg["scope"], device="cpu", training_updates=0,
                  dynamic_environment_steps=0, policy_forward_calls=0, real_data_access=False,
                  confirmation_trajectories_accessed=False, real_slm_actions=False)
    if preflight_only:
        return dict(status="O1_UNIT_PREFLIGHT_ONLY", output_directory=str(output), **report)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        cases = check_cases(cfg, bridge)
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, encoding="utf-8", errors="replace", check=False)
        sources = ["observation_bridge/adapter.py", "observation_bridge/__init__.py",
                   "scripts/verify_observation_bridge_o1.py",
                   "src/simulation/modes.py", "src/rl/s4_representation_capacity.py"]
        config_path = Path(path) if Path(path).is_absolute() else ROOT / path
        result = dict(
            status="O1_SYNTHETIC_TECHNICAL_PASS_ONLY", **report, config=cfg,
            cases=cases, elapsed_seconds=time.perf_counter() - started,
            basis_diagnostics=bridge.basis_diagnostics,
            reference_fit_condition_number=bridge.design_condition_number,
            runtime=dict(python=platform.python_version(), torch=torch.__version__),
            git_head=git.stdout.strip() if git.returncode == 0 else None,
            source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
            config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            output_directory=str(output), real_accuracy_verified=False,
            full_holography_chain_verified=False, cuda_or_realtime_verified=False,
            next_action="Design O2 synthetic holography chain; do not train or rerun sealed C1.",
        )
        summary = output / "summary.json"
        summary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output / "SUCCESS.json").write_text(json.dumps(dict(
            status=result["status"], summary_sha256=hashlib.sha256(summary.read_bytes()).hexdigest(),
        ), indent=2) + "\n", encoding="utf-8")
        return result
    except Exception as exc:
        (output / "failure.json").write_text(json.dumps(dict(
            status="O1_TECHNICAL_FAILED", exception=type(exc).__name__, message=str(exc), **report,
        ), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="O1 人工光场技术单元验证（CPU；不是正式实验）")
    parser.add_argument("--config", default="configs/experiments/observation_bridge_o1_v1.yaml")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, preflight_only=args.preflight_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
