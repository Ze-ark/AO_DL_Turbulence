"""O2-D1：固定静态光场的 CUDA 读出噪声标尺，不运行控制器或环境。

允许记录预先列出的测量保护拒绝；其他异常立即停止并保留现场。
真值只用于图像生成、源采样前提和测量返回后的隔离审计。
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback
from typing import Any

import torch
import yaml

from scripts import verify_observation_bridge_o2_optics as optics
from src.runtime import resolve_device
from src.simulation.modes import synthesize_phase

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/experiments/observation_bridge_o2_d1_read_noise_v1.yaml"
C_SUMMARY_SHA = "233daf9687fce241299c6e90e3d00c1b8da36e61a1fc65c9b48d5e6ca3a4eb32"
PINS = {
    "configs/experiments/observation_bridge_o2_optics_v1.yaml": "b117ae9e7a5eb0e6d7bf7f0a0ee259dd7303f8cdd00a7cbfadc08eed4d13e9f0",
    "observation_bridge/holography.py": "4d790fe67dbe76f38c078d0768ed18c6776c0e046fb3d5faca8e50bb435f34e0",
    "observation_bridge/cuda_modal_observation.py": "bccfb058dc546b0b433ce8765df9f1ae88e1c662da6764855f3774f6691a7f26",
    "scripts/verify_observation_bridge_o2_optics.py": "02cbdddacbc7d8e9c82350e9f61956833649d6d322634351dc3d98be07ef1f28",
    "observation_bridge/development.py": "b022b9c372091a7347b7dc561c747a5c219620eb638e6d179990cb210d7e09b5",
    "configs/experiments/observation_bridge_o2_development_v1.yaml": "2e9582993ee979e6b15079aa8b2110b7bbff7a29f498be5bf50204de53d90eb2",
}
# 只捕获已知的测量有效性拒绝；非有限值、设备和程序错误不在此清单。
MEASUREMENT_REJECTIONS = {
    "low intensity or hole inside frozen pupil": "low_intensity",
    "spatial phase jump exceeds sampling guard": "sampling_jump",
    "phase loop inconsistency; singular or ambiguous field": "phase_loop",
}
NUMERIC_COLUMNS = (
    "modal_rmse_rad", "modal_max_error_rad", "modal_delta_from_zero_rmse_rad", "representation_fit_rmse_rad",
    "minimum_relative_intensity", "max_neighbor_jump_rad",
)
ROW_COLUMNS = (
    "fixture_index", "fixture_id", "fixture_group", "repetition", "noise_level_index",
    "read_noise_std", "noise_seed", "noise_rng_after_draw_sha256", "status", "rejection_type", "rejection_reason",
    "modal_rmse_rad", "modal_max_error_rad", "modal_delta_from_zero_rmse_rad", "modal_bias_rad", "representation_fit_rmse_rad",
    "minimum_relative_intensity", "max_neighbor_jump_rad", "clean_intensity_mean",
    "noise_std_over_clean_mean", "negative_clip_fraction", "observation_latency_ms",
)


def read_config(path: str | Path = CONFIG) -> dict[str, Any]:
    target = Path(path) if Path(path).is_absolute() else ROOT / path
    return yaml.safe_load(target.read_text(encoding="utf-8"))


def validate_config(cfg: dict[str, Any]) -> None:
    expected = dict(
        schema="observation_bridge_o2_d1_read_noise_v1",
        scope="synthetic_static_read_noise_diagnostic_not_closed_loop", device="cuda",
        optics_config="configs/experiments/observation_bridge_o2_optics_v1.yaml", fixture_seed=832001,
        fixture_count=55, noise_distribution="additive_gaussian_then_clip_negative_intensity_to_zero",
        noise_units="arbitrary_synthetic_intensity", shared_draw_across_levels=True, measurement_batch_size=1,
        data=dict(fixture_indices=list(range(55)), repetitions=16,
                  noise_stds=[0.0, .001, .01, .1, .3, 1.0, 3.0, 10.0],
                  extra_fixture_seed=8500000, noise_seed_base=8501000),
        quick=dict(fixture_indices=[0, 1, 47], repetitions=1, noise_stds=[0.0, .001, 10.0],
                   extra_fixture_seed=8510000, noise_seed_base=8511000),
        thresholds=dict(zero_noise_modal_error_max_rad=.001, extra_component_norm_min=1e-8),
        boundary=dict(dynamic_environment_steps=0, model_loads=0, model_forward_calls=0, training_updates=0,
                      scientific_gain_analysis=False, independent_confirmation=False,
                      old_confirmation_trajectory_access=False, real_data_access=False, real_slm_actions=False,
                      historical_gate_reclassification=False, truth_fallback=False, automatic_retry=False),
    )
    if not isinstance(cfg, dict) or set(cfg) != set(expected) | {"output_directory", "quick_directory"} or any(
        json.dumps(cfg.get(k), sort_keys=True, allow_nan=False) != json.dumps(v, sort_keys=True)
        for k, v in expected.items()
    ):
        raise ValueError("O2-D1 fixed single-factor contract changed")


def budget(spec: dict[str, Any]) -> dict[str, int]:
    fields = len(spec["fixture_indices"])
    draws = fields * spec["repetitions"]
    return dict(selected_fixtures=fields, clean_images_generated=fields,
                standard_normal_draws=draws, measurement_attempts=draws * len(spec["noise_stds"]),
                attempts_per_noise_level=draws, dynamic_environment_steps=0, model_forward_calls=0,
                training_updates=0)


def noise_seed(spec: dict[str, Any], fixture_index: int, repetition: int) -> int:
    if fixture_index not in spec["fixture_indices"] or not 0 <= repetition < spec["repetitions"]:
        raise ValueError("unknown fixture/repetition identity")
    return spec["noise_seed_base"] + 55 * repetition + fixture_index


def stream_manifest(cfg: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    """纯种子清单与互斥检查；不加载旧轨迹，不调用旧实验入口。"""
    from observation_bridge import development as prior
    from scripts import confirm_s4_r5_g2_c1 as c1
    source = prior.short.frozen.source
    factories = [source.stream_manifest, source.d10.d9.stream_manifest,
                 source.d10.d9.d8.stream_manifest, source.d10.d9.d3.stream_manifest,
                 source.d10.d9.d8.d2.g2.stream_manifest]
    historical = [fn(quick=q) for fn in factories for q in (False, True)]
    c1cfg = prior.read_config(c1.CONFIG)
    historical.extend(c1.stream_manifest(c1cfg[k]) for k in ("data", "quick"))
    historical.append(prior.short.stream_manifest(prior.read_config(prior.short.CONFIG)))
    ccfg = prior.read_config(prior.CONFIG)
    historical.extend(prior._streams(ccfg, ccfg[k]) for k in ("data", "quick"))
    spec = cfg["quick" if quick else "data"]
    other = cfg["data" if quick else "quick"]

    def seeds(s: dict[str, Any]) -> set[int]:
        return {s["extra_fixture_seed"]} | {
            noise_seed(s, f, rep) for rep in range(s["repetitions"]) for f in s["fixture_indices"]}

    current = seeds(spec)
    if len(current) != budget(spec)["standard_normal_draws"] + 1:
        raise RuntimeError("O2-D1 internal seed collision")
    reserved = set(range(8470000, 8480000)) | set(range(8520000, 8530000))
    if current & (seeds(other) | reserved) or any(current & prior.short._integers(m) for m in historical):
        raise RuntimeError("O2-D1 historical/quick/unit seed collision")
    return dict(extra_fixture_seed=spec["extra_fixture_seed"], noise_seeds=sorted(current - {spec["extra_fixture_seed"]}),
                reused_O2_A_fixture_seed=cfg["fixture_seed"], reused_known_fixtures_not_new_weather=True,
                historical_manifests_checked=len(historical),
                disjointness_scope="declared_G2_C1_O2_B_O2_C_and_other_D1_mode_plus_reserved_unit_namespaces",
                technical_unit_namespace=8520000, shared_draw_across_levels=True)


def verify_prerequisites(cfg: dict[str, Any]) -> dict[str, Any]:
    from observation_bridge import development as prior
    for name, expected in PINS.items():
        if optics.file_sha256(ROOT / name) != expected:
            raise RuntimeError(f"O2-D1 frozen dependency changed: {name}")
    result = prior.verify_short_loop(prior.read_config(prior.CONFIG))
    directory = ROOT / "outputs/observation_bridge_o2_development_v1"
    summary, success = prior.short.read_json(directory / "summary.json"), prior.short.read_json(directory / "SUCCESS.json")
    names = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
             if p.is_file() and p.name != "SUCCESS.json"}
    if (optics.file_sha256(directory / "summary.json") != C_SUMMARY_SHA
            or success["summary_sha256"] != C_SUMMARY_SHA or len(names) != 426
            or set(success["artifact_sha256"]) != names or (directory / "failure.json").exists()):
        raise RuntimeError("O2-C completed prerequisite changed")
    for name, expected in success["artifact_sha256"].items():
        target = (directory / name).resolve()
        if not target.is_relative_to(directory.resolve()) or optics.file_sha256(target) != expected:
            raise RuntimeError("O2-C successful artifact changed")
    if (summary["completed_episodes"], summary["completed_physical_transitions"], summary["invalid_observations"]) != (624, 124800, 0):
        raise RuntimeError("O2-C prerequisite incomplete")
    if prior.short.read_json(directory / "source_manifest.json") != prior.source_manifest(prior.CONFIG):
        raise RuntimeError("O2-C executed source identity changed")
    calibration = prior.short.read_json(ROOT / "configs/experiments/observation_bridge_design_v1.json")["real_calibration"]
    if calibration["status"] != "UNVERIFIED_DO_NOT_INFER_DEFAULTS" or len(calibration) != 19 or any(
        v is not None for k, v in calibration.items() if k != "status"
    ):
        raise RuntimeError("real calibration unknowns changed; do not infer camera defaults")
    return dict(result, audited_C_summary_sha256=C_SUMMARY_SHA, C_artifacts_checked=426,
                dependency_sha256=PINS, real_calibration_unknown_fields=18)


def source_manifest(path: str | Path) -> dict[str, Any]:
    target = Path(path) if Path(path).is_absolute() else ROOT / path
    names = ["observation_bridge/read_noise_diagnostic.py", "scripts/diagnose_observation_bridge_o2_read_noise.py",
             "tests/test_observation_bridge_o2_read_noise.py"]
    return dict(new_source_sha256={n: optics.file_sha256(ROOT / n) for n in names},
                config_sha256=optics.file_sha256(target), frozen_dependency_sha256=PINS)


def configure_runtime() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8", ":16:8"):
        raise RuntimeError("unsupported deterministic cuBLAS workspace setting")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def preflight(path: str | Path = CONFIG, *, quick: bool = False) -> tuple:
    cfg = read_config(path)
    validate_config(cfg)
    spec = cfg["quick" if quick else "data"]
    output = (ROOT / cfg["quick_directory" if quick else "output_directory"]).resolve()
    if output == (ROOT / "outputs").resolve() or not output.is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("O2-D1 output must be a new child of workspace outputs")
    if output.exists():
        raise FileExistsError(f"preserve existing O2-D1 output: {output}")
    frozen = verify_prerequisites(cfg)
    streams = stream_manifest(cfg, quick=quick)
    configure_runtime()
    device = resolve_device(cfg["device"])
    if device.type != "cuda":
        raise RuntimeError("O2-D1 requires CUDA; no CPU fallback")
    report = dict(status="O2_D1_READY_FOR_TECHNICAL_SMOKE" if quick else "O2_D1_READY_FOR_USER_IDE",
                  quick=quick, **cfg["boundary"], **{k: v for k, v in budget(spec).items() if k not in cfg["boundary"]},
                  device=str(device), output_directory=str(output), frozen_sources=frozen, stream_manifest=streams,
                  preflight_image_generations=0, preflight_measurement_calls=0)
    return cfg, spec, output, device, report


def extra_phase_templates(basis: torch.Tensor, pupil: torch.Tensor, seed: int, norm_min: float) -> torch.Tensor:
    """静态几何构造；允许明确的小型 CPU 几何单元，不用于 CPU 光学结果。"""
    axis = torch.linspace(-1, 1, 64, dtype=torch.float64, device=basis.device)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    raw = (6 * math.pi * x).sin() * (4 * math.pi * y).cos()
    design = torch.cat((torch.ones(int(pupil.sum()), 1, device=basis.device, dtype=torch.float64),
                        basis[:, pupil].T.double()), dim=1)
    solver = "gels" if basis.device.type == "cuda" else "gelsd"
    solution = torch.linalg.lstsq(design, raw[pupil, None], driver=solver).solution[:, 0]
    outside = raw.clone()
    outside[pupil] -= design @ solution
    norm = outside[pupil].square().mean().sqrt()
    if not bool(torch.isfinite(norm)) or float(norm) < norm_min:
        raise RuntimeError("nonmodal template is singular; do not replace the function")
    outside = outside / norm
    generator = torch.Generator(device=basis.device).manual_seed(seed)
    coefficients = torch.randn(8, 21, dtype=torch.float64, generator=generator, device=basis.device) * .02
    amplitudes = torch.tensor([.1, -.1, .2, -.2, .4, -.4, .8, -.8], device=basis.device, dtype=torch.float64)
    return synthesize_phase(coefficients, basis.double()) + amplitudes[:, None, None] * outside


def make_fixtures(cfg: dict[str, Any], spec: dict[str, Any], bridge: Any) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    phases, identities = [], []
    for group, coefficients, constant in optics.fixture_groups(cfg["fixture_seed"], bridge.device):
        phase = synthesize_phase(coefficients.double(), bridge.basis.double()) + constant
        phases.append(phase)
        for i in range(len(phase)):
            identities.append(dict(fixture_index=len(identities), fixture_id=f"{group}_{i}", fixture_group="controlled"))
    phases.append(extra_phase_templates(bridge.basis, bridge.pupil, spec["extra_fixture_seed"],
                                       cfg["thresholds"]["extra_component_norm_min"]))
    for i in range(8):
        identities.append(dict(fixture_index=len(identities), fixture_id=f"outside_21_modes_{i}", fixture_group="outside_21_modes"))
    all_phases = torch.cat(phases)
    if len(all_phases) != cfg["fixture_count"] or not bool(torch.isfinite(all_phases).all()):
        raise RuntimeError("static fixture definition invalid")
    pupil = bridge.pupil
    jumps = torch.cat(((all_phases[:, 1:] - all_phases[:, :-1])[:, pupil[1:] & pupil[:-1]].flatten(),
                       (all_phases[:, :, 1:] - all_phases[:, :, :-1])[:, pupil[:, 1:] & pupil[:, :-1]].flatten()))
    if float(jumps.abs().max()) > bridge.tolerances.max_neighbor_jump_rad:
        raise RuntimeError("source sampling precondition failed; do not reduce phase amplitude")
    return all_phases[spec["fixture_indices"]], [identities[i] for i in spec["fixture_indices"]]


def rejection_type(error: ValueError) -> str | None:
    return MEASUREMENT_REJECTIONS.get(str(error))


def paired_intensity(clean: torch.Tensor, samples: torch.Tensor, std: float) -> tuple[torch.Tensor, float]:
    if clean.device.type != "cuda" or clean.device != samples.device or clean.shape != samples.shape:
        raise ValueError("paired camera tensors require the same CUDA shape/device")
    if not math.isfinite(std) or std < 0 or not bool(torch.isfinite(clean).all()) or not bool(torch.isfinite(samples).all()):
        raise ValueError("nonfinite read-noise input")
    noisy = clean + std * samples
    if not bool(torch.isfinite(noisy).all()):
        raise ValueError("nonfinite noisy intensity")
    return noisy.clamp_min(0), float((noisy < 0).float().mean())


def validate_rows(rows: list[dict[str, Any]], spec: dict[str, Any]) -> None:
    expected = {(f, rep, level) for f in spec["fixture_indices"] for rep in range(spec["repetitions"])
                for level in range(len(spec["noise_stds"]))}
    seen, draw_hashes = set(), {}
    for row in rows:
        key = row["fixture_index"], row["repetition"], row["noise_level_index"]
        if key not in expected or key in seen or set(row) != set(ROW_COLUMNS):
            raise ValueError("incomplete/duplicate/unknown reading identity")
        seen.add(key)
        f, rep, level = key
        if row["noise_seed"] != noise_seed(spec, f, rep) or row["read_noise_std"] != spec["noise_stds"][level]:
            raise ValueError("reading seed/noise level mismatch")
        group = "controlled" if f < 47 else "outside_21_modes"
        if row["fixture_group"] != group:
            raise ValueError("fixture group identity mismatch")
        token = row["noise_rng_after_draw_sha256"]
        if not isinstance(token, str) or len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
            raise ValueError("invalid noise draw fingerprint")
        draw_key = f, rep
        if draw_hashes.setdefault(draw_key, token) != token:
            raise ValueError("noise draw differs across levels")
        for column in ("clean_intensity_mean", "noise_std_over_clean_mean", "negative_clip_fraction", "observation_latency_ms"):
            if not math.isfinite(row[column]) or row[column] < 0:
                raise ValueError("nonfinite/negative measurement metadata")
        if row["clean_intensity_mean"] <= 0 or not 0 <= row["negative_clip_fraction"] <= 1:
            raise ValueError("invalid camera scale/clip fraction")
        if row["status"] == "valid":
            if row["rejection_type"] is not None or row["rejection_reason"] is not None:
                raise ValueError("valid observation has rejection reason")
            if any(not isinstance(row[c], (int, float)) or not math.isfinite(row[c]) or row[c] < 0 for c in NUMERIC_COLUMNS):
                raise ValueError("nonfinite valid observation statistic")
            if len(row["modal_bias_rad"]) != 21 or any(not math.isfinite(v) for v in row["modal_bias_rad"]):
                raise ValueError("nonfinite modal bias")
        elif row["status"] == "rejected":
            if spec["noise_stds"][level] == 0 or MEASUREMENT_REJECTIONS.get(row["rejection_reason"]) != row["rejection_type"]:
                raise ValueError("unknown/zero-noise observation rejection")
            if any(row[c] is not None for c in NUMERIC_COLUMNS) or row["modal_bias_rad"] is not None:
                raise ValueError("rejected reading has invented audit values")
        else:
            raise ValueError("unknown reading status")
    if seen != expected:
        raise ValueError("missing planned readings; do not analyze survivors only")


def valid_statistics(values: torch.Tensor) -> dict[str, Any]:
    """纯张量数学；CPU 仅供手写小数组单元，正式调用强制 CUDA。"""
    if values.ndim != 1 or len(values) == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError("finite nonempty valid statistics required")
    return dict(mean=float(values.mean()), p95=float(torch.quantile(values, .95)), maximum=float(values.max()))


def summarize(rows: list[dict[str, Any]], spec: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    if device.type != "cuda":
        raise ValueError("formal read-noise statistics require CUDA")
    validate_rows(rows, spec)
    cells = []
    for level, std in enumerate(spec["noise_stds"]):
        for group in ("all", "controlled", "outside_21_modes"):
            selected = [r for r in rows if r["noise_level_index"] == level and (group == "all" or r["fixture_group"] == group)]
            valid = [r for r in selected if r["status"] == "valid"]
            n = len(selected)
            errors = torch.tensor([r["modal_rmse_rad"] for r in valid], device=device, dtype=torch.float64)
            zero_deltas = torch.tensor([r["modal_delta_from_zero_rmse_rad"] for r in valid], device=device, dtype=torch.float64)
            bias = torch.tensor([r["modal_bias_rad"] for r in valid], device=device, dtype=torch.float64)
            clean_scale = torch.tensor([r["clean_intensity_mean"] for r in selected], device=device, dtype=torch.float64)
            clip = torch.tensor([r["negative_clip_fraction"] for r in selected], device=device, dtype=torch.float64)
            cells.append(dict(noise_level_index=level, read_noise_std=std, fixture_group=group,
                planned_attempts=n, valid_readings=len(valid), rejected_readings=n-len(valid),
                rejection_fraction=(n-len(valid))/n if n else None,
                valid_only_modal_rmse_rad=valid_statistics(errors) if len(valid) else None,
                valid_only_delta_from_zero_rmse_rad=valid_statistics(zero_deltas) if len(valid) else None,
                valid_only_modal_bias_rad=bias.mean(0).tolist() if len(valid) else None,
                all_attempts_mean_clean_intensity=float(clean_scale.mean()) if n else None,
                all_attempts_mean_negative_clip_fraction=float(clip.mean()) if n else None,
                conditional_on_valid_readings=True, whole_level_reliability_claim=False))
    return cells


def _runtime(device: torch.device) -> dict[str, Any]:
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                         text=True, encoding="utf-8", errors="replace", check=False)
    return dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(device), git_head=(git.stdout or "").strip(),
                deterministic=torch.are_deterministic_algorithms_enabled(), allow_tf32=False)


@torch.no_grad()
def execute(cfg: dict[str, Any], spec: dict[str, Any], output: Path, device: torch.device,
            report: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    if device.type != "cuda":
        raise RuntimeError("optical diagnostic requires CUDA")
    started = time.perf_counter()
    sensor, bridge = optics.make_components(read_config(cfg["optics_config"]), device)
    phases, identities = make_fixtures(cfg, spec, bridge)
    fields = (torch.polar(torch.ones_like(phases), phases) * bridge.pupil).to(torch.complex64)
    clean_images = sensor.render(fields)
    context["clean_images_generated"] = len(phases)
    targets: dict[int, torch.Tensor] = {}
    zero_readouts: dict[int, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    total = budget(spec)["measurement_attempts"]
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with (output / "measurements.csv").open("w", encoding="utf-8", newline="") as table, (
        output / "progress.jsonl").open("w", encoding="utf-8") as progress:
        writer = csv.DictWriter(table, fieldnames=ROW_COLUMNS)
        writer.writeheader()
        for rep in range(spec["repetitions"]):
            for position, identity in enumerate(identities):
                fixture = identity["fixture_index"]
                seed = noise_seed(spec, fixture, rep)
                generator = torch.Generator(device=device).manual_seed(seed)
                samples = torch.randn((1, 512, 512), generator=generator, device=device)
                # 只把随机状态元数据复制回 CPU，不回传整张观测/噪声图。
                draw_hash = hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()
                context["standard_normal_draws"] += 1
                clean = clean_images[position:position+1]
                mean = float(clean.double().mean())
                for level, std in enumerate(spec["noise_stds"]):
                    context.update(fixture_index=fixture, repetition=rep, noise_level_index=level)
                    intensity, clipped = paired_intensity(clean, samples, std)
                    row = dict(identity, repetition=rep, noise_level_index=level, read_noise_std=std,
                               noise_seed=seed, noise_rng_after_draw_sha256=draw_hash, status="valid",
                               rejection_type=None, rejection_reason=None, modal_bias_rad=None,
                               **{c: None for c in NUMERIC_COLUMNS}, clean_intensity_mean=mean,
                               noise_std_over_clean_mean=std/mean, negative_clip_fraction=clipped,
                               observation_latency_ms=0.0)
                    begin.record()
                    try:
                        measured = bridge.measure(sensor.reconstruct(intensity))
                    except ValueError as error:
                        category = rejection_type(error)
                        if category is None or std == 0:
                            raise
                        row.update(status="rejected", rejection_type=category, rejection_reason=str(error))
                    finally:
                        end.record()
                        end.synchronize()
                        row["observation_latency_ms"] = begin.elapsed_time(end)
                    # 从测量返回后才计算审计目标；拒绝的读数不补真值。
                    if row["status"] == "valid":
                        if fixture not in targets:
                            target, _ = optics.audit_known_phase(phases[position:position+1], bridge)
                            targets[fixture] = target[0]
                            zero_readouts[fixture] = measured.residual_rad[0].detach().clone()
                        if std == 0 and not torch.equal(measured.residual_rad[0], zero_readouts[fixture]):
                            raise RuntimeError("repeated zero-noise readout changed")
                        delta = measured.residual_rad[0].double() - targets[fixture]
                        row.update(modal_bias_rad=delta.tolist(), modal_rmse_rad=float(delta.square().mean().sqrt()),
                                   modal_max_error_rad=float(delta.abs().max()),
                                   modal_delta_from_zero_rmse_rad=float((measured.residual_rad[0].double()
                                       - zero_readouts[fixture].double()).square().mean().sqrt()),
                                   representation_fit_rmse_rad=float(measured.fit_rmse_rad[0]),
                                   minimum_relative_intensity=float(measured.minimum_relative_intensity[0]),
                                   max_neighbor_jump_rad=float(measured.max_neighbor_jump_rad[0]))
                        if std == 0 and row["modal_max_error_rad"] > cfg["thresholds"]["zero_noise_modal_error_max_rad"]:
                            raise RuntimeError("zero-noise modal error exceeds frozen guard")
                    rows.append(row)
                    writer.writerow({**row, "modal_bias_rad": json.dumps(row["modal_bias_rad"], allow_nan=False)})
                    table.flush()
                    context["measurement_attempts_completed"] += 1
                    context["valid_readings"] += row["status"] == "valid"
                    context["rejected_readings"] += row["status"] == "rejected"
                    elapsed = time.perf_counter() - started
                    n = context["measurement_attempts_completed"]
                    record = dict(context, elapsed_seconds=elapsed, attempts_total=total,
                                  attempts_per_second=n/max(elapsed, 1e-9), eta_seconds=elapsed*(total-n)/n,
                                  cuda_allocated_gib=torch.cuda.memory_allocated(device)/2**30,
                                  cuda_reserved_gib=torch.cuda.memory_reserved(device)/2**30)
                    progress.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
                    progress.flush()
                    if n % 32 == 0 or n == total:
                        valid_errors = [r["modal_rmse_rad"] for r in rows if r["status"] == "valid"]
                        average = sum(valid_errors)/len(valid_errors) if valid_errors else 0.0
                        print(f"O2-D1 {n}/{total} | 有效={context['valid_readings']} 无效={context['rejected_readings']} "
                              f"| 有效平均误差={average:.3g} rad | 剩余={record['eta_seconds']:.0f}s "
                              f"| 显存={record['cuda_allocated_gib']:.2f}/{record['cuda_reserved_gib']:.2f}GiB", flush=True)
    validate_rows(rows, spec)
    if context["standard_normal_draws"] != budget(spec)["standard_normal_draws"]:
        raise RuntimeError("random draw count mismatch")
    torch.save(dict(fixture_identities=identities, full_phase_rad=phases,
                    post_measurement_audit_target_rad=torch.stack([targets[i["fixture_index"]] for i in identities])),
               output / "fixtures.pt")
    quality = summarize(rows, spec, device)
    result = dict(report, status="O2_D1_TECHNICAL_SMOKE_ONLY" if report["quick"] else "O2_D1_DIAGNOSTIC_COMPLETE_REQUIRES_READ_ONLY_AUDIT",
                  completed_measurement_attempts=len(rows), valid_readings=context["valid_readings"],
                  rejected_readings=context["rejected_readings"], completed_standard_normal_draws=context["standard_normal_draws"],
                  cells=[] if report["quick"] else quality, technical_grid_validated=True,
                  no_survivor_only_reliability_claim=True, gain_threshold=None,
                  inverse_crime_limitation=True, real_camera_noise_calibrated=False, runtime=_runtime(device),
                  latency_scope="single_static_frame_reconstruction_and_modal_readout_excludes_render_noise_audit_IO",
                  elapsed_seconds=time.perf_counter()-started, next_action="Read-only audit; no automatic retry, training or closed-loop run.")
    return result


def run(path: str | Path = CONFIG, *, quick: bool = False, preflight_only: bool = False) -> dict[str, Any]:
    cfg, spec, output, device, report = preflight(path, quick=quick)
    if preflight_only:
        return report
    source = source_manifest(path)
    output.mkdir(parents=True, exist_ok=False)
    context = dict(clean_images_generated=0, standard_normal_draws=0,
                   measurement_attempts_completed=0, valid_readings=0, rejected_readings=0)
    try:
        optics.write_json(output / "effective_config.json", cfg)
        optics.write_json(output / "preflight.json", report)
        optics.write_json(output / "input_manifest.json", dict(frozen_sources=report["frozen_sources"],
                          stream_manifest=report["stream_manifest"]))
        optics.write_json(output / "source_manifest.json", source)
        result = execute(cfg, spec, output, device, report, context)
        if source_manifest(path) != source or verify_prerequisites(cfg) != report["frozen_sources"]:
            raise RuntimeError("source/prerequisite changed during diagnostic")
        optics.write_json(output / "summary.json", result)
        artifacts = {p.relative_to(output).as_posix(): optics.file_sha256(p) for p in output.rglob("*") if p.is_file()}
        optics.write_json(output / "SUCCESS.json", dict(status=result["status"],
                          summary_sha256=artifacts["summary.json"], artifact_sha256=artifacts))
        return result
    except BaseException as error:
        optics.write_json(output / "failure.json", dict(status="O2_D1_STOPPED_PRESERVE_OUTPUT", error_type=type(error).__name__,
                          error=str(error), traceback=traceback.format_exc(), last_context=context,
                          **cfg["boundary"], next_action="Preserve output and request diagnosis; do not retry automatically."))
        raise
