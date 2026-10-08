"""O2-D4：固定静态完整光场的 CUDA 泊松计数标尺，零控制/训练。

有限光照档独立抽样；零噪声不抽样。真相位只用于生成和返回后的审计。
仅记录预声明测量保护拒绝；其他异常停止、保留，不自动重试。
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
from typing import Any

import torch

from observation_bridge import noise_development as completed
from observation_bridge import read_noise_diagnostic as static
from src.runtime import resolve_device

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/experiments/observation_bridge_o2_d4_photon_noise_v1.yaml"
optics, short = completed.optics, completed.short
D3_PINS = {
    "observation_bridge/noise_development.py": "69a547eb32299b6c835e9a91254567e9fd29bdcf75055d14416ff01a2dc01dcc",
    "scripts/evaluate_observation_bridge_o2_noise_development.py": "e440fd819120961d58f49080c66c509e5fb7eb02d160046603d7966a2a34a42c",
    "tests/test_observation_bridge_o2_noise_development.py": "2fe899602b5a75d152fbb41f60117ef58af25f26874a4fc70b870a9eb168f470",
    completed.CONFIG: "7a81395189ab678de91811a6d60dda3a7cc31c9177b3a13d5e3cd06475f1faf0",
}
D3_OUTPUTS = {
    "outputs/observation_bridge_o2_d3_noise_development_v1":
        ("5fd5a7d803b93bc5811adbceba3eb55da3cd560174434a0ad0c030b6b70e8c72", 842, False),
    "outputs/observation_bridge_o2_d3_noise_development_v1_quick":
        ("dbf24f4879ea2eb39664454da9313ff1dcc2f0249a3decb8f1e9702f4409f687", 62, True),
}
NUMERIC_COLUMNS = static.NUMERIC_COLUMNS
COUNT_COLUMNS = ("expected_count_mean", "sampled_count_mean", "sampled_count_max", "zero_count_fraction")
ROW_COLUMNS = (
    "fixture_index", "fixture_id", "fixture_group", "repetition", "light_level_index", "counts_per_intensity_unit",
    "noise_seed", "noise_rng_after_draw_sha256", "counts_sha256", "clean_intensity_sha256", "status",
    "rejection_type", "rejection_reason", *NUMERIC_COLUMNS, "modal_bias_rad", "clean_intensity_mean",
    *COUNT_COLUMNS, "observation_latency_ms",
)


def read_config(path: str | Path = CONFIG) -> dict[str, Any]:
    return static.read_config(path)


def validate_config(cfg: dict[str, Any]) -> None:
    expected = dict(
        schema="observation_bridge_o2_d4_photon_noise_v1", scope="synthetic_static_photon_noise_not_closed_loop",
        device="cuda", optics_config="configs/experiments/observation_bridge_o2_optics_v1.yaml",
        fixture_seed=832001, fixture_count=55, measurement_batch_size=1,
        noise_distribution="poisson_counts_divided_by_fixed_exposure_scale",
        noise_units="synthetic_detected_counts_per_arbitrary_intensity_unit", count_dtype="float64",
        reconstructed_intensity_dtype="float32", shared_draw_across_levels=False, image_dependent_normalization=False,
        read_noise_std=0.0,
        data=dict(fixture_indices=list(range(55)), repetitions=16,
                  count_scales=[None, 1000000.0, 10000.0, 1000.0, 100.0, 10.0, 1.0],
                  extra_fixture_seed=8800000, noise_seed_base=8801000),
        quick=dict(fixture_indices=[0, 1, 47], repetitions=1, count_scales=[None, 1000.0, 1.0],
                   extra_fixture_seed=8860000, noise_seed_base=8861000),
        thresholds=dict(zero_noise_modal_error_max_rad=.001, extra_component_norm_min=1e-8),
        boundary=dict(dynamic_environment_steps=0, model_loads=0, model_forward_calls=0, training_updates=0,
                      scientific_gain_analysis=False, independent_confirmation=False,
                      old_confirmation_trajectory_access=False, real_data_access=False, real_slm_actions=False,
                      historical_gate_reclassification=False, truth_fallback=False, automatic_retry=False))
    if (not isinstance(cfg, dict) or set(cfg) != set(expected) | {"output_directory", "quick_directory"}
            or any(json.dumps(cfg.get(k), sort_keys=True, allow_nan=False) != json.dumps(v, sort_keys=True)
                   for k, v in expected.items())):
        raise ValueError("O2-D4 fixed single-factor contract changed")
    if any(not isinstance(cfg[k], str) or not cfg[k] for k in ("output_directory", "quick_directory")):
        raise ValueError("O2-D4 output paths required")
    if (ROOT / cfg["output_directory"]).resolve() == (ROOT / cfg["quick_directory"]).resolve():
        raise ValueError("O2-D4 formal and quick outputs must differ")


def budget(spec: dict[str, Any]) -> dict[str, int]:
    fields, levels = len(spec["fixture_indices"]), len(spec["count_scales"])
    return dict(selected_fixtures=fields, clean_images_generated=fields,
                measurement_attempts=fields * spec["repetitions"] * levels,
                attempts_per_light_level=fields * spec["repetitions"],
                poisson_draws=fields * spec["repetitions"] * (levels-1),
                dynamic_environment_steps=0, model_forward_calls=0, training_updates=0)


def noise_seed(spec: dict[str, Any], fixture: int, repetition: int, level: int) -> int | None:
    if (type(fixture) is not int or type(repetition) is not int or type(level) is not int
            or fixture not in spec["fixture_indices"] or not 0 <= repetition < spec["repetitions"]
            or not 0 <= level < len(spec["count_scales"])):
        raise ValueError("unknown fixture/repetition/light level")
    return None if level == 0 else spec["noise_seed_base"] + 55 * (spec["repetitions"] * (level-1) + repetition) + fixture


def stream_manifest(cfg: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    # D1 纯清单检查覆盖 15 份历史清单；将有限光照档展开为重复索引，不调用旧入口。
    probe = static.read_config()
    for mode in ("data", "quick"):
        spec = cfg[mode]
        probe[mode] = dict(spec, noise_stds=[0.0], repetitions=spec["repetitions"]*(len(spec["count_scales"])-1))
    static.stream_manifest(probe, quick=quick)
    old_static = static.read_config()
    old_technical = completed.technical.read_config()
    old_completed = completed.read_config()
    history = [static.stream_manifest(old_static, quick=q) for q in (False, True)]
    history += [completed.technical.stream_manifest(old_technical, quick=q) for q in (False, True)]
    history += [completed.stream_manifest(old_completed, quick=q) for q in (False, True)]

    def seeds(spec: dict[str, Any]) -> set[int]:
        return {spec["extra_fixture_seed"]} | {
            noise_seed(spec, f, r, level) for f in spec["fixture_indices"]
            for r in range(spec["repetitions"]) for level in range(1, len(spec["count_scales"]))}

    spec = cfg["quick" if quick else "data"]
    current = seeds(spec)
    if len(current) != budget(spec)["poisson_draws"]+1:
        raise RuntimeError("O2-D4 internal seed collision")
    reserved = set().union(*(set(range(n, n+10000)) for n in (8470000, 8520000, 8670000, 8770000, 8870000)))
    reserved |= {n+offset for base in (8670000, 8770000) for n in range(base, base+10000)
                 for offset in (50000000, 60000000, 170000000)}
    if (current & (seeds(cfg["data" if quick else "quick"]) | reserved)
            or any(current & short._integers(h) for h in history)):
        raise RuntimeError("O2-D4 historical/quick/unit seed collision")
    return dict(extra_fixture_seed=spec["extra_fixture_seed"],
                noise_seeds=sorted(current - {spec["extra_fixture_seed"]}),
                reused_O2_A_fixture_seed=cfg["fixture_seed"], reused_known_fixtures_not_new_weather=True,
                historical_manifests_checked=21, technical_unit_namespace=8870000,
                shared_draw_across_levels=False, zero_noise_draws=0,
                seed_formula="base+55*(repetitions*(finite_level_index-1)+repetition)+fixture_index",
                disjointness_scope="declared_G2_C1_B_C_D1_D2_D3_other_D4_mode_and_reserved_units")


def verify_prerequisites() -> dict[str, Any]:
    for name, digest in D3_PINS.items():
        if optics.file_sha256(ROOT / name) != digest:
            raise RuntimeError(f"O2-D4 frozen D3 source changed: {name}")
    result = completed.verify_prerequisites()
    cfg = completed.read_config()
    completed.validate_config(cfg)
    for relative, (digest, count, quick) in D3_OUTPUTS.items():
        directory = ROOT / relative
        summary = short.read_json(directory / "summary.json")
        success = short.read_json(directory / "SUCCESS.json")
        names = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
                 if p.is_file() and p.name != "SUCCESS.json"}
        status = "O2_D3_TECHNICAL_SMOKE_ONLY" if quick else "O2_D3_DEVELOPMENT_COMPLETE_REQUIRES_READ_ONLY_AUDIT"
        if (len(names) != count or set(success["artifact_sha256"]) != names
                or success["summary_sha256"] != digest or optics.file_sha256(directory / "summary.json") != digest
                or success["status"] != status or summary["status"] != status
                or (directory / "failure.json").exists() or (directory / "partial").exists()):
            raise RuntimeError("O2-D4 completed D3 seal changed")
        for name, expected in success["artifact_sha256"].items():
            target = (directory / name).resolve()
            if not target.is_relative_to(directory.resolve()) or optics.file_sha256(target) != expected:
                raise RuntimeError("O2-D4 completed D3 artifact changed")
        planned = completed.budget(cfg["quick" if quick else "data"])
        if (summary["completed_episodes"] != planned["complete_episodes"]
                or summary["completed_physical_transitions"] != planned["physical_transitions"]
                or summary["invalid_observations"] != 0 or summary["failed_or_truncated_episodes"] != 0
                or summary["replay_max_absolute_error"] != 0 or not summary["observation_quality"]["targets_met"]
                or short.read_json(directory / "effective_config.json") != cfg
                or short.read_json(directory / "source_manifest.json") != completed.source_manifest(completed.CONFIG)):
            raise RuntimeError("O2-D4 D3 prerequisite incomplete or source changed")
    return dict(result, D3_artifacts_checked=904, frozen_D3_source_sha256=D3_PINS,
                D3_output_summary_sha256={p:v[0] for p, v in D3_OUTPUTS.items()})


def source_manifest(path: str | Path) -> dict[str, Any]:
    target = Path(path) if Path(path).is_absolute() else ROOT / path
    names = ("observation_bridge/photon_noise_diagnostic.py", "scripts/diagnose_observation_bridge_o2_photon_noise.py",
             "tests/test_observation_bridge_o2_photon_noise.py")
    return dict(new_source_sha256={n:optics.file_sha256(ROOT / n) for n in names},
                config_sha256=optics.file_sha256(target), frozen_D3_source=completed.source_manifest(completed.CONFIG))


def preflight(path: str | Path = CONFIG, *, quick: bool = False) -> tuple:
    cfg = read_config(path)
    validate_config(cfg)
    spec = cfg["quick" if quick else "data"]
    output = (ROOT / cfg["quick_directory" if quick else "output_directory"]).resolve()
    if output == (ROOT / "outputs").resolve() or not output.is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("O2-D4 output must be a new child of outputs")
    if output.exists():
        raise FileExistsError(f"preserve existing O2-D4 output: {output}")
    frozen, streams = verify_prerequisites(), stream_manifest(cfg, quick=quick)
    static.configure_runtime()
    device = resolve_device(cfg["device"])
    if device.type != "cuda":
        raise RuntimeError("O2-D4 requires CUDA; no CPU fallback")
    report = dict(status="O2_D4_READY_FOR_TECHNICAL_SMOKE" if quick else "O2_D4_READY_FOR_USER_IDE",
                  quick=quick, **cfg["boundary"], **{k:v for k,v in budget(spec).items() if k not in cfg["boundary"]},
                  device=str(device), output_directory=str(output), frozen_sources=frozen, stream_manifest=streams,
                  preflight_image_generations=0, preflight_measurement_calls=0)
    return cfg, spec, output, device, report


def tensor_sha256(tensor: torch.Tensor) -> str:
    # 仅复制字节到主机做指纹，不在 CPU 做图像/误差计算。
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def photon_intensity(clean: torch.Tensor, scale: float | None, seed: int | None) -> tuple[torch.Tensor, dict[str, Any]]:
    """仅当前干净强度/固定尺度/显式种子，无相位或环境参数。"""
    if clean.device.type != "cuda" or clean.dtype != torch.float32 or clean.numel() == 0:
        raise ValueError("photon camera requires nonempty float32 CUDA intensity")
    if not bool(torch.isfinite(clean).all()) or bool((clean < 0).any()):
        raise ValueError("nonfinite/negative clean intensity")
    if scale is None:
        if seed is not None:
            raise ValueError("noiseless level must not consume a random draw")
        return clean.clone(), dict(noise_rng_after_draw_sha256=None, counts_sha256=None,
                                   **{k:None for k in COUNT_COLUMNS})
    if (type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0
            or type(seed) is not int or not 0 <= seed < 2**63):
        raise ValueError("positive finite fixed photon scale and explicit integer seed required")
    rates = clean.double() * scale
    if not bool(torch.isfinite(rates).all()):
        raise ValueError("nonfinite photon rate")
    generator = torch.Generator(device=clean.device).manual_seed(seed)
    counts = torch.poisson(rates, generator=generator)
    intensity = (counts/scale).float()
    if (not bool(torch.isfinite(counts).all()) or bool((counts < 0).any())
            or not torch.equal(counts, counts.floor()) or not bool(torch.isfinite(intensity).all())):
        raise ValueError("invalid photon counts/intensity")
    return intensity, dict(noise_rng_after_draw_sha256=tensor_sha256(generator.get_state()),
                           counts_sha256=tensor_sha256(counts), expected_count_mean=float(rates.mean()),
                           sampled_count_mean=float(counts.mean()), sampled_count_max=float(counts.max()),
                           zero_count_fraction=float(counts.eq(0).double().mean()))


def _is_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_rows(rows: list[dict[str, Any]], spec: dict[str, Any]) -> None:
    expected = {(f,r,l) for f in spec["fixture_indices"] for r in range(spec["repetitions"])
                for l in range(len(spec["count_scales"]))}
    seen: set[tuple] = set()
    clean_metadata: dict[int, tuple] = {}
    for row in rows:
        if set(row) != set(ROW_COLUMNS):
            raise ValueError("unknown/missing reading fields")
        key = row["fixture_index"], row["repetition"], row["light_level_index"]
        if any(type(k) is not int for k in key) or key not in expected or key in seen:
            raise ValueError("incomplete/duplicate/unknown reading identity")
        seen.add(key)
        f, r, l = key
        scale = spec["count_scales"][l]
        if row["noise_seed"] != noise_seed(spec, f, r, l) or row["counts_per_intensity_unit"] != scale:
            raise ValueError("reading seed/light scale mismatch")
        if (row["fixture_group"] != ("controlled" if f < 47 else "outside_21_modes")
                or not isinstance(row["fixture_id"], str) or not row["fixture_id"]):
            raise ValueError("fixture identity mismatch")
        clean = row["clean_intensity_mean"]
        latency = row["observation_latency_ms"]
        if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in (clean, latency)) or clean <= 0:
            raise ValueError("nonfinite/negative camera metadata")
        if not _is_hash(row["clean_intensity_sha256"]):
            raise ValueError("invalid clean image fingerprint")
        identity = row["fixture_id"], row["clean_intensity_sha256"], clean
        if clean_metadata.setdefault(f, identity) != identity:
            raise ValueError("clean fixture/image differs across light levels or repetitions")
        if scale is None:
            if any(row[k] is not None for k in (*COUNT_COLUMNS, "counts_sha256", "noise_rng_after_draw_sha256", "noise_seed")):
                raise ValueError("noiseless level invented photon draw")
        else:
            if any(not _is_hash(row[k]) for k in ("counts_sha256", "noise_rng_after_draw_sha256")):
                raise ValueError("invalid photon draw fingerprint")
            if any(type(row[k]) not in (float, int) or not math.isfinite(row[k]) or row[k] < 0 for k in COUNT_COLUMNS):
                raise ValueError("nonfinite/negative photon metadata")
            if (not 0 <= row["zero_count_fraction"] <= 1 or row["sampled_count_max"] < row["sampled_count_mean"]
                    or not math.isclose(row["expected_count_mean"], scale*clean, rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError("invalid fixed count scale/domain")
        if row["status"] == "valid":
            if row["rejection_type"] is not None or row["rejection_reason"] is not None:
                raise ValueError("valid reading has rejection reason")
            if any(type(row[c]) not in (float, int) or not math.isfinite(row[c]) or row[c] < 0 for c in NUMERIC_COLUMNS):
                raise ValueError("nonfinite valid statistic")
            bias = row["modal_bias_rad"]
            if not isinstance(bias, list) or len(bias) != 21 or any(type(v) not in (float,int) or not math.isfinite(v) for v in bias):
                raise ValueError("nonfinite modal bias")
        elif row["status"] == "rejected":
            if scale is None or static.MEASUREMENT_REJECTIONS.get(row["rejection_reason"]) != row["rejection_type"]:
                raise ValueError("unknown/zero-noise observation rejection")
            if any(row[c] is not None for c in NUMERIC_COLUMNS) or row["modal_bias_rad"] is not None:
                raise ValueError("rejected reading has invented audit values")
        else:
            raise ValueError("unknown reading status")
    if seen != expected:
        raise ValueError("missing planned readings; do not analyze survivors only")


def summarize(rows: list[dict[str, Any]], spec: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    if device.type != "cuda":
        raise ValueError("formal photon statistics require CUDA")
    validate_rows(rows, spec)
    cells = []
    for level, scale in enumerate(spec["count_scales"]):
        for group in ("all", "controlled", "outside_21_modes"):
            selected = [r for r in rows if r["light_level_index"] == level and (group == "all" or r["fixture_group"] == group)]
            valid = [r for r in selected if r["status"] == "valid"]
            n = len(selected)
            def tensor(column: str, records: list[dict[str, Any]]) -> torch.Tensor:
                return torch.tensor([r[column] for r in records], device=device, dtype=torch.float64)
            cell = dict(light_level_index=level, counts_per_intensity_unit=scale, fixture_group=group,
                        planned_attempts=n, valid_readings=len(valid), rejected_readings=n-len(valid),
                        rejection_fraction=(n-len(valid))/n if n else None,
                        rejection_counts={k:sum(r["rejection_type"] == k for r in selected) for k in static.MEASUREMENT_REJECTIONS.values()},
                        valid_only_modal_bias_rad=tensor("modal_bias_rad", valid).mean(0).tolist() if valid else None,
                        all_attempts_mean_clean_intensity=float(tensor("clean_intensity_mean", selected).mean()) if n else None,
                        conditional_on_valid_readings=True, whole_level_reliability_claim=False)
            for column in NUMERIC_COLUMNS:
                cell[f"valid_only_{column}"] = static.valid_statistics(tensor(column, valid)) if valid else None
            for column in COUNT_COLUMNS:
                cell[f"all_attempts_mean_{column}"] = float(tensor(column, selected).mean()) if n and scale is not None else None
            cells.append(cell)
    return cells


@torch.no_grad()
def execute(cfg: dict[str, Any], spec: dict[str, Any], output: Path, device: torch.device,
            report: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    if device.type != "cuda":
        raise RuntimeError("optical diagnostic requires CUDA")
    started = time.perf_counter()
    sensor, bridge = optics.make_components(read_config(cfg["optics_config"]), device)
    phases, identities = static.make_fixtures(cfg, spec, bridge)
    fields = (torch.polar(torch.ones_like(phases), phases)*bridge.pupil).to(torch.complex64)
    images = sensor.render(fields)
    context["clean_images_generated"] = len(phases)
    clean_hashes = [tensor_sha256(i) for i in images]
    clean_means = images.double().mean((-2,-1)).tolist()
    targets: dict[int, torch.Tensor] = {}
    zeros: dict[int, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    total = budget(spec)["measurement_attempts"]
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with (output/"measurements.csv").open("w", encoding="utf-8", newline="") as table, (
        output/"progress.jsonl").open("w", encoding="utf-8") as progress:
        writer = csv.DictWriter(table, fieldnames=ROW_COLUMNS)
        writer.writeheader()
        for rep in range(spec["repetitions"]):
            for position, identity in enumerate(identities):
                fixture = identity["fixture_index"]
                for level, scale in enumerate(spec["count_scales"]):
                    context.update(fixture_index=fixture, repetition=rep, light_level_index=level)
                    seed = noise_seed(spec, fixture, rep, level)
                    intensity, metadata = photon_intensity(images[position:position+1], scale, seed)
                    context["poisson_draws"] += scale is not None
                    row = dict(identity, repetition=rep, light_level_index=level, counts_per_intensity_unit=scale,
                               noise_seed=seed, **metadata, clean_intensity_sha256=clean_hashes[position],
                               status="valid", rejection_type=None, rejection_reason=None,
                               **{c:None for c in NUMERIC_COLUMNS}, modal_bias_rad=None,
                               clean_intensity_mean=clean_means[position], observation_latency_ms=0.0)
                    begin.record()
                    try:
                        measured = bridge.measure(sensor.reconstruct(intensity))
                    except ValueError as error:
                        category = static.rejection_type(error)
                        if category is None or scale is None:
                            raise
                        row.update(status="rejected", rejection_type=category, rejection_reason=str(error))
                    finally:
                        end.record()
                        end.synchronize()
                        row["observation_latency_ms"] = begin.elapsed_time(end)
                    # 审计目标只在读数返回后拟合；拒绝的观测不补真值/零误差。
                    if row["status"] == "valid":
                        if fixture not in targets:
                            if scale is not None:
                                raise RuntimeError("missing zero-noise reference")
                            target, _ = optics.audit_known_phase(phases[position:position+1], bridge)
                            targets[fixture] = target[0]
                            zeros[fixture] = measured.residual_rad[0].detach().clone()
                        if scale is None and not torch.equal(measured.residual_rad[0], zeros[fixture]):
                            raise RuntimeError("repeated zero-noise readout changed")
                        delta = measured.residual_rad[0].double()-targets[fixture]
                        row.update(modal_bias_rad=delta.tolist(), modal_rmse_rad=float(delta.square().mean().sqrt()),
                                   modal_max_error_rad=float(delta.abs().max()),
                                   modal_delta_from_zero_rmse_rad=float((measured.residual_rad[0].double()-zeros[fixture].double()).square().mean().sqrt()),
                                   representation_fit_rmse_rad=float(measured.fit_rmse_rad[0]),
                                   minimum_relative_intensity=float(measured.minimum_relative_intensity[0]),
                                   max_neighbor_jump_rad=float(measured.max_neighbor_jump_rad[0]))
                        if scale is None and row["modal_max_error_rad"] > cfg["thresholds"]["zero_noise_modal_error_max_rad"]:
                            raise RuntimeError("zero-noise modal error exceeds frozen guard")
                    rows.append(row)
                    writer.writerow({**row, "modal_bias_rad":json.dumps(row["modal_bias_rad"], allow_nan=False)})
                    table.flush()
                    context["measurement_attempts_completed"] += 1
                    context["valid_readings"] += row["status"] == "valid"
                    context["rejected_readings"] += row["status"] == "rejected"
                    elapsed = time.perf_counter()-started
                    n = context["measurement_attempts_completed"]
                    record = dict(context, elapsed_seconds=elapsed, attempts_total=total,
                                  attempts_per_second=n/max(elapsed,1e-9), eta_seconds=elapsed*(total-n)/n,
                                  cuda_allocated_gib=torch.cuda.memory_allocated(device)/2**30,
                                  cuda_reserved_gib=torch.cuda.memory_reserved(device)/2**30)
                    progress.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
                    progress.flush()
                    if n % 32 == 0 or n == total:
                        errors = [r["modal_rmse_rad"] for r in rows if r["status"] == "valid"]
                        average = sum(errors)/len(errors) if errors else 0.0
                        print(f"O2-D4 {n}/{total} | 有效={context['valid_readings']} 无效={context['rejected_readings']} "
                              f"| 有效平均误差={average:.3g}rad | {record['attempts_per_second']:.1f}次/s "
                              f"| 剩余={record['eta_seconds']:.0f}s | 显存={record['cuda_allocated_gib']:.2f}/{record['cuda_reserved_gib']:.2f}GiB", flush=True)
    validate_rows(rows, spec)
    if (context["poisson_draws"] != budget(spec)["poisson_draws"]
            or len(rows) != total or context["valid_readings"]+context["rejected_readings"] != total):
        raise RuntimeError("O2-D4 incomplete draw/measurement budget")
    torch.save(dict(fixture_identities=identities, full_phase_rad=phases,
                    post_measurement_audit_target_rad=torch.stack([targets[i["fixture_index"]] for i in identities]),
                    zero_readout_rad=torch.stack([zeros[i["fixture_index"]] for i in identities])), output/"fixtures.pt")
    cells = summarize(rows, spec, device)
    return dict(report, status="O2_D4_TECHNICAL_SMOKE_ONLY" if report["quick"] else "O2_D4_DIAGNOSTIC_COMPLETE_REQUIRES_READ_ONLY_AUDIT",
                completed_measurement_attempts=len(rows), completed_poisson_draws=context["poisson_draws"],
                valid_readings=context["valid_readings"], rejected_readings=context["rejected_readings"],
                cells=[] if report["quick"] else cells, technical_grid_validated=True,
                no_survivor_only_reliability_claim=True, gain_threshold=None, inverse_crime_limitation=True,
                real_camera_noise_calibrated=False, runtime=static._runtime(device),
                count_scale_scope="fixed_synthetic_detected_counts_not_real_exposure_power_or_QE_calibration",
                latency_scope="single_static_frame_reconstruction_and_modal_readout_excludes_render_noise_audit_IO",
                elapsed_seconds=time.perf_counter()-started,
                next_action="Read-only audit; no automatic retry, training or closed-loop run.")


def run(path: str | Path = CONFIG, *, quick: bool = False, preflight_only: bool = False) -> dict[str, Any]:
    cfg, spec, output, device, report = preflight(path, quick=quick)
    if preflight_only:
        return report
    source = source_manifest(path)
    output.mkdir(parents=True, exist_ok=False)
    context = dict(clean_images_generated=0, poisson_draws=0, measurement_attempts_completed=0,
                   valid_readings=0, rejected_readings=0)
    try:
        optics.write_json(output/"effective_config.json", cfg)
        optics.write_json(output/"preflight.json", report)
        optics.write_json(output/"input_manifest.json", dict(frozen_sources=report["frozen_sources"], stream_manifest=report["stream_manifest"]))
        optics.write_json(output/"source_manifest.json", source)
        result = execute(cfg, spec, output, device, report, context)
        if source_manifest(path) != source or verify_prerequisites() != report["frozen_sources"]:
            raise RuntimeError("source/prerequisite changed during diagnostic")
        optics.write_json(output/"summary.json", result)
        artifacts = {p.relative_to(output).as_posix():optics.file_sha256(p) for p in output.rglob("*") if p.is_file()}
        optics.write_json(output/"SUCCESS.json", dict(status=result["status"], summary_sha256=artifacts["summary.json"], artifact_sha256=artifacts))
        return result
    except BaseException as error:
        optics.write_json(output/"failure.json", dict(status="O2_D4_STOPPED_PRESERVE_OUTPUT", error_type=type(error).__name__,
                          error=str(error), traceback=traceback.format_exc(), last_context=context,
                          **cfg["boundary"], next_action="Preserve output and request diagnosis; do not retry automatically."))
        raise
