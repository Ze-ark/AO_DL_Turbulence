"""O2-A：CUDA 人工光学往返技术验证，零动态转移、零模型推理。

不访问真实 HDF5、确认轨迹或设备；真值只在测量返回后用于审计。
输出存在即拒绝；失败留痕且不自动重试。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml

from observation_bridge.adapter import FieldOrientation, SyntheticFieldContract
from observation_bridge.cuda_modal_observation import CudaModalObservationBridge, O2ModalTolerances
from observation_bridge.holography import SyntheticOffAxisSensor, SyntheticOpticsCalibration
from src.runtime import resolve_device
from src.simulation.modes import project_phase_to_modes, synthesize_phase

DESIGN_SHA256 = "08da140eb29d86ada740c33e4f2edfc1719a2aeba6e58c15964bfbc6608625e9"
BUNDLE_SHA256 = "ccdc31faaa155361d8bd3e19ffc5bb82705f425a0ef4c7f58dd7f70831efd956"
SCOPE = dict(training_updates=0, dynamic_environment_steps=0, policy_forward_calls=0,
             real_data_access=False, confirmation_trajectories_accessed=False, real_slm_actions=False)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sealed_bundle_sha256() -> str:
    paths = sorted((ROOT / "src").rglob("*.py"))
    paths.append(ROOT / "scripts" / "confirm_s4_r5_policy_r2.py")
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def check_sources(design: dict[str, Any]) -> dict[str, Any]:
    sources = design["frozen_source_identity"]
    for relative, expected in sources.items():
        if file_sha256(ROOT / relative) != expected:
            raise RuntimeError(f"frozen source identity changed: {relative}")
    archive = json.loads((ROOT / "configs/experiments/s4_r5_g2_c1_archive_v1.json").read_text(encoding="utf-8"))
    frozen = archive["frozen_files"]
    for relative, expected in frozen.items():
        if file_sha256(ROOT / relative) != expected:
            raise RuntimeError(f"sealed C1 file changed: {relative}")
    if sealed_bundle_sha256() != BUNDLE_SHA256:
        raise RuntimeError("sealed src bundle changed; do not weaken the hash guard")
    return dict(frozen_source_sha256=sources, frozen_C1_files_checked=len(frozen),
                sealed_source_bundle_sha256=BUNDLE_SHA256)


def preflight(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], Path, torch.device, dict[str, Any]]:
    config_path = Path(path) if Path(path).is_absolute() else ROOT / path
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fixed = dict(schema="observation_bridge_o2_optics_v1", scope="cuda_synthetic_optics_technical_only",
                 device="cuda", fixture_seed=832001, positive_fixture_count=47,
                 maximum_gpu_fixture_batch=4, design_path="configs/experiments/observation_bridge_o2_design_v1.json",
                 design_sha256=DESIGN_SHA256, sealed_source_bundle_sha256=BUNDLE_SHA256)
    if not isinstance(cfg, dict) or any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k, v in fixed.items()):
        raise ValueError("O2-A fixed scope, seed, budget or identity changed")
    if set(cfg) != set(fixed) | {"output_directory", "thresholds", "optics", "contract", "modal_tolerances"}:
        raise ValueError("unknown O2-A configuration fields")
    design_path = ROOT / cfg["design_path"]
    if file_sha256(design_path) != DESIGN_SHA256:
        raise RuntimeError("historical O2 design changed")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if cfg["thresholds"] != design["stage_A"]["thresholds"]:
        raise ValueError("predeclared O2-A thresholds changed")
    optics = dict(cfg["optics"])
    optics["orientation"] = FieldOrientation(**optics["orientation"])
    for name in ("reference_carrier_bins_yx", "crop_rows", "crop_columns"):
        optics[name] = tuple(optics[name])
    SyntheticOpticsCalibration(**optics).validate()
    contract = dict(cfg["contract"])
    contract["orientation"] = FieldOrientation(**contract["orientation"])
    SyntheticFieldContract(**contract).validate()
    if contract["orientation"] != FieldOrientation(False, False, False, 1):
        raise ValueError("O2-A fixtures require canonical reconstructed pupil axes")
    expected_tolerances = dict(min_relative_intensity=1e-4, max_neighbor_jump_rad=1.5, loop_consistency_rad=1e-4)
    if cfg["modal_tolerances"] != expected_tolerances:
        raise ValueError("predeclared O2-A modal guards changed")
    O2ModalTolerances(**cfg["modal_tolerances"]).validate()
    output = (ROOT / cfg["output_directory"]).resolve()
    output_root = (ROOT / "outputs").resolve()
    if output == output_root or not output.is_relative_to(output_root):
        raise ValueError("O2 output must be a new child of workspace outputs")
    if output.exists():
        raise FileExistsError(f"preserve existing O2-A output: {output}")
    sources = check_sources(design)
    device = resolve_device(cfg["device"])
    if device.type != "cuda":
        raise RuntimeError("O2-A requires CUDA; no CPU fallback")
    return cfg, design, output, device, sources


def make_components(cfg: dict[str, Any], device: torch.device) -> tuple[SyntheticOffAxisSensor, CudaModalObservationBridge]:
    optics = dict(cfg["optics"])
    optics["orientation"] = FieldOrientation(**optics["orientation"])
    for name in ("reference_carrier_bins_yx", "crop_rows", "crop_columns"):
        optics[name] = tuple(optics[name])
    contract = dict(cfg["contract"])
    contract["orientation"] = FieldOrientation(**contract["orientation"])
    return (SyntheticOffAxisSensor(SyntheticOpticsCalibration(**optics), device),
            CudaModalObservationBridge(SyntheticFieldContract(**contract),
                                      O2ModalTolerances(**cfg["modal_tolerances"]), device))


def fixture_groups(seed: int, device: torch.device) -> list[tuple[str, torch.Tensor, float]]:
    generator = torch.Generator(device=device).manual_seed(seed)
    pulses = torch.cat((torch.eye(21, device=device), -torch.eye(21, device=device))) * 0.05
    mixed = torch.randn(2, 21, generator=generator, device=device) * 0.02
    wrapped = torch.zeros(1, 21, device=device)
    wrapped[0, 0], wrapped[0, 2], wrapped[0, 10] = 4.0, 0.7, 0.1
    return [("zero_phase", torch.zeros(1, 21, device=device), 0.0),
            ("signed_mode_pulses", pulses, 0.0), ("seeded_mixtures", mixed, 0.0),
            ("wrapped_smooth_phase", wrapped, 0.0), ("wrapped_constant_reference", wrapped, 3.7)]


def audit_known_phase(phase: torch.Tensor, bridge: CudaModalObservationBridge) -> tuple[torch.Tensor, float]:
    """独立真值审计，不把答案作为 bridge.measure 的输入或失败回退。"""
    pupil, basis = bridge.pupil, bridge.basis.double()
    design = torch.cat((torch.ones(int(pupil.sum()), 1, dtype=torch.float64, device=phase.device),
                        basis[:, pupil].T), dim=1)
    # 审计重新用最小二乘求解，不调用测量器的预计算逆矩阵。
    solution = torch.linalg.lstsq(design, phase[:, pupil].T, driver="gels").solution.T
    jumps = torch.cat(((phase[:, 1:] - phase[:, :-1])[:, pupil[1:] & pupil[:-1]].flatten(),
                       (phase[:, :, 1:] - phase[:, :, :-1])[:, pupil[:, 1:] & pupil[:, :-1]].flatten()))
    return solution[:, 1:], float(jumps.abs().max())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@torch.no_grad()
def execute_checks(cfg: dict[str, Any], sensor: SyntheticOffAxisSensor, bridge: CudaModalObservationBridge,
                   output: Path) -> dict[str, Any]:
    started, completed = time.perf_counter(), 0
    cases: list[dict[str, Any]] = []
    group_results: dict[str, Any] = {}
    max_field_error, max_modal_error = 0.0, 0.0
    opposite_error = 0.0
    for name, all_coefficients, constant in fixture_groups(cfg["fixture_seed"], bridge.device):
        start_index = len(cases)
        for offset in range(0, len(all_coefficients), cfg["maximum_gpu_fixture_batch"]):
            coefficients = all_coefficients[offset:offset + cfg["maximum_gpu_fixture_batch"]]
            phase = synthesize_phase(coefficients.double(), bridge.basis.double()) + constant
            field = (torch.polar(torch.ones_like(phase), phase) * bridge.pupil).to(torch.complex64)
            intensity = sensor.render(field)
            reconstructed = sensor.reconstruct(intensity)
            measured = bridge.measure(reconstructed)
            # 从这行开始才读取相位真值审计；之前测量链不接收 phase/coefficients。
            expected, true_jump = audit_known_phase(phase, bridge)
            if true_jump > bridge.tolerances.max_neighbor_jump_rad:
                raise RuntimeError("known fixture sampling precondition failed; do not change the seed")
            field_errors = ((reconstructed - field).abs().square().sum(dim=(-2, -1))
                            / field.abs().square().sum(dim=(-2, -1))).sqrt()
            modal_errors = (measured.residual_rad.double() - expected).abs().amax(dim=1)
            if (float(field_errors.max()) > cfg["thresholds"]["relative_complex_field_l2_error_maximum"]
                    or float(modal_errors.max()) > cfg["thresholds"]["modal_coefficient_max_error_rad"]):
                raise RuntimeError(f"O2-A predeclared optical/modal tolerance failed: {name}/{offset}")
            legacy = project_phase_to_modes(phase, bridge.basis.double(), bridge.pupil)
            if name == "wrapped_smooth_phase":
                twin = sensor.reconstruct_opposite_for_test(intensity)
                twin_error = float((twin - field.conj()).abs().square().sum().sqrt() / field.abs().square().sum().sqrt())
                sign_error = float((bridge.measure(twin).residual_rad.double() + expected).abs().max())
                opposite_error = max(twin_error, sign_error)
                if twin_error > 1e-5 or sign_error > 1e-4:
                    raise RuntimeError("opposite sideband sign/boundary check failed")
            for index in range(len(coefficients)):
                row = dict(group=name, fixture_index=offset + index,
                           relative_complex_field_l2_error=float(field_errors[index]),
                           max_modal_coefficient_error_rad=float(modal_errors[index]),
                           max_known_coefficient_error_rad=float((measured.residual_rad[index] - coefficients[index]).abs().max()),
                           modal_fit_rmse_rad=float(measured.fit_rmse_rad[index]),
                           legacy_projection_difference_rad=float((legacy[index] - expected[index]).abs().max()),
                           max_true_phase_abs_rad=float(phase[index, bridge.pupil].abs().max()),
                           batch_true_neighbor_jump_rad=true_jump)
                cases.append(row)
            completed += len(coefficients)
            max_field_error = max(max_field_error, float(field_errors.max()))
            max_modal_error = max(max_modal_error, float(modal_errors.max()))
            torch.cuda.synchronize(bridge.device)
            elapsed = time.perf_counter() - started
            progress = dict(completed=completed, total=47, group=name, elapsed_seconds=elapsed,
                            eta_seconds=elapsed / completed * (47 - completed),
                            max_field_error=max_field_error, max_modal_error_rad=max_modal_error,
                            cuda_allocated_gb=torch.cuda.memory_allocated(bridge.device) / 2**30,
                            cuda_reserved_gb=torch.cuda.memory_reserved(bridge.device) / 2**30)
            with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(progress, ensure_ascii=False, allow_nan=False) + "\n")
            print(f"O2-A 技术检查 {completed}/47 | 光场误差={max_field_error:.3g} | "
                  f"相位系数误差={max_modal_error:.3g} rad | 剩余={progress['eta_seconds']:.1f}s | "
                  f"显存={progress['cuda_allocated_gb']:.2f}GB", flush=True)
        group_results[name] = dict(samples=len(cases) - start_index)
    if completed != cfg["positive_fixture_count"]:
        raise RuntimeError("O2-A fixture budget mismatch")
    return dict(cases=cases, groups=group_results, positive_fixtures_completed=completed,
                max_relative_complex_field_l2_error=max_field_error, max_modal_coefficient_error_rad=max_modal_error,
                opposite_sideband_field_and_sign_error_max=opposite_error,
                optical_checks_elapsed_seconds=time.perf_counter() - started)


def run(path: str | Path, *, preflight_only: bool = False) -> dict[str, Any]:
    cfg, _, output, device, sources = preflight(path)
    report = dict(scope=cfg["scope"], device=str(device), **SCOPE)
    if preflight_only:
        return dict(status="O2_A_TECHNICAL_PREFLIGHT_ONLY", output_directory=str(output), **report, **sources)
    # 确定性设置在执行函数中显式设置；导入不修改后端或分配显存。
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8", ":16:8"):
        raise RuntimeError("unsupported deterministic cuBLAS workspace setting")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    try:
        sensor, bridge = make_components(cfg, device)
        source_paths = ["observation_bridge/holography.py", "observation_bridge/cuda_modal_observation.py",
                        "observation_bridge/adapter.py", "scripts/verify_observation_bridge_o2_optics.py",
                        "tests/test_observation_bridge_o2.py", "src/rl/s4_representation_capacity.py", "src/runtime.py"]
        config_path = Path(path) if Path(path).is_absolute() else ROOT / path
        source_manifest = dict(**sources, new_source_sha256={p: file_sha256(ROOT / p) for p in source_paths},
                               config_sha256=file_sha256(config_path), design_sha256=DESIGN_SHA256)
        write_json(output / "source_manifest.json", source_manifest)
        write_json(output / "effective_config.json", cfg)
        write_json(output / "stream_manifest.json", dict(fixture_seed=cfg["fixture_seed"],
                                                         rng_device=str(device), dynamic_weather_streams=0))
        checks = execute_checks(cfg, sensor, bridge, output)
        # 执行后再查一次冻结边界，不能把已变化的历史当成成功。
        check_sources(json.loads((ROOT / cfg["design_path"]).read_text(encoding="utf-8")))
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, encoding="utf-8", errors="replace", check=False)
        status_git = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", check=False)
        result = dict(status="O2_A_SYNTHETIC_OPTICS_TECHNICAL_PASS_ONLY", **report, **checks,
                      output_directory=str(output), reference_fit_condition_number=bridge.design_condition_number,
                      basis_diagnostics=bridge.basis_diagnostics, source_manifest=source_manifest,
                      runtime=dict(python=platform.python_version(), torch=torch.__version__,
                                   cuda=torch.version.cuda, device_name=torch.cuda.get_device_name(device),
                                   deterministic=torch.are_deterministic_algorithms_enabled(),
                                   cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
                                   cudnn_enabled=torch.backends.cudnn.enabled, cudnn_benchmark=False, allow_tf32=False),
                      git_head=git.stdout.strip() if git.returncode == 0 else None,
                      git_dirty=bool(status_git.stdout.strip()) if status_git.returncode == 0 else None,
                      raw_camera_frames_saved=0, real_accuracy_verified=False, dynamic_closed_loop_verified=False,
                      real_time_verified=False, inverse_crime_limitation=True,
                      next_action="Prepare O2-B technical entry only; do not train or run development/confirmation automatically.")
        write_json(output / "summary.json", result)
        artifacts = {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()}
        write_json(output / "SUCCESS.json", dict(status=result["status"], summary_sha256=artifacts["summary.json"],
                                                 artifact_sha256=artifacts))
        return result
    except BaseException as exc:
        write_json(output / "failure.json", dict(status="O2_A_TECHNICAL_FAILED", exception=type(exc).__name__,
                                                 message=str(exc), traceback=traceback.format_exc(), **report))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="O2-A CUDA 人工全息技术检查（不训练、不跑动态回合）")
    parser.add_argument("--config", default="configs/experiments/observation_bridge_o2_optics_v1.yaml")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = run(args.config, preflight_only=args.preflight_only)
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "source_manifest", "basis_diagnostics")},
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
