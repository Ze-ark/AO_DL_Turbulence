"""O2-D6：冻结控制器、单因素光子噪声、完整天气配对开发对照。

复用封存 D5 的因果回合，不调用旧实验入口；真值仅供独立审计。
统计单位为八份完整天气，不把帧、家族或初始化当作独立天气。
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
import traceback
from typing import Any

import torch

from observation_bridge import photon_closed_loop as technical
from src.runtime import resolve_device

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/experiments/observation_bridge_o2_d6_photon_development_v1.yaml"
prior, short, optics = technical.prior, technical.short, technical.optics
D5_PINS = {
    "observation_bridge/photon_closed_loop.py": "dd6331eacc2bb3afd60652d998456665942e0bb97c60abc67c28e68c211401d5",
    "scripts/verify_observation_bridge_o2_photon_closed_loop.py": "99e0d2a724558a06ec7df263e2e7bc6b6fb03bc4af30372f25f8b0f1f9f02e7d",
    "tests/test_observation_bridge_o2_photon_closed_loop.py": "24cd57847f2893853120082c11f5e83ac4a74b6b241b54c2809bd3f8919fd419",
    technical.CONFIG: "8c1ea31fb580fdbec8f87138be4665c809357893cc641444b9fae4b1144b1169",
}
D5_OUTPUTS = {
    "outputs/observation_bridge_o2_d5_photon_closed_loop_v1":
        ("afa5f10b2ed15885def783fc24c5076eacd8bdc72262f83b6dbb9743c2ffb76c", 477, False),
    "outputs/observation_bridge_o2_d5_photon_closed_loop_v1_quick":
        ("a114d6d3744dd197969fb0c1ef4b128fcc22fb8c7b9404aaf7d6934ce49de946", 87, True),
}


def read_config(path: str | Path = CONFIG) -> dict[str, Any]:
    return prior.read_config(path)


def validate_config(cfg: dict[str, Any]) -> None:
    # 仅从已固定指纹的 D5 合约继承不变项；不接受随旧文件漂移的默认值。
    if optics.file_sha256(technical.ROOT / technical.CONFIG) != D5_PINS[technical.CONFIG]:
        raise RuntimeError("O2-D6 frozen D5 configuration changed")
    expected = technical.read_config()
    for key in ("output_directory", "quick_directory"):
        expected.pop(key)
    expected.update(
        schema="observation_bridge_o2_d6_photon_development_v1",
        scope="synthetic_photon_noise_development_not_confirmation",
        data=dict(weather_seed_base=9000000, weather_count=8, weather_seed_stride=10000,
                  episode_length=200, camera_ids=["noiseless", "photon_k100", "photon_k10"]),
        quick=dict(weather_seed_base=9160000, weather_count=1, weather_seed_stride=10000,
                   episode_length=28, camera_ids=["noiseless", "photon_k10"]),
        statistics=dict(cluster="complete_weather", stratification="fixed_scorer_fold",
                        bootstrap_seed=9180000, bootstrap_repeats=5000, interval=.95,
                        interval_scope="pointwise_descriptive_not_simultaneous_not_confirmation"))
    expected["boundary"].pop("scientific_gain_analysis")
    if (not isinstance(cfg, dict) or set(cfg) != set(expected) | {"output_directory", "quick_directory"}
            or any(json.dumps(cfg.get(k), sort_keys=True, allow_nan=False) != json.dumps(v, sort_keys=True)
                   for k, v in expected.items())):
        raise ValueError("O2-D6 fixed photon development contract changed")
    if (any(not isinstance(cfg[k], str) or not cfg[k] for k in ("output_directory", "quick_directory"))
            or (ROOT / cfg["output_directory"]).resolve() == (ROOT / cfg["quick_directory"]).resolve()):
        raise ValueError("O2-D6 distinct output paths required")


def budget(spec: dict[str, Any]) -> dict[str, int]:
    return technical.budget(spec)


def verify_prerequisites() -> dict[str, Any]:
    for name, digest in D5_PINS.items():
        if optics.file_sha256(ROOT / name) != digest:
            raise RuntimeError(f"O2-D6 frozen D5 source changed: {name}")
    result = technical.verify_prerequisites()
    cfg = technical.read_config()
    technical.validate_config(cfg)
    for relative, (digest, count, quick) in D5_OUTPUTS.items():
        directory = ROOT / relative
        success, summary = short.read_json(directory / "SUCCESS.json"), short.read_json(directory / "summary.json")
        names = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
                 if p.is_file() and p.name != "SUCCESS.json"}
        status = "O2_D5_TECHNICAL_SMOKE_ONLY" if quick else "O2_D5_COMPLETE_REQUIRES_READ_ONLY_AUDIT"
        if (len(names) != count or set(success["artifact_sha256"]) != names
                or success["summary_sha256"] != digest or optics.file_sha256(directory / "summary.json") != digest
                or summary["status"] != status or success["status"] != status
                or (directory / "failure.json").exists() or (directory / "partial").exists()):
            raise RuntimeError("O2-D6 completed D5 seal changed")
        for name, expected in success["artifact_sha256"].items():
            target = (directory / name).resolve()
            if not target.is_relative_to(directory.resolve()) or optics.file_sha256(target) != expected:
                raise RuntimeError("O2-D6 completed D5 artifact changed")
        planned = budget(cfg["quick" if quick else "data"])
        if (short.read_json(directory / "effective_config.json") != cfg
                or short.read_json(directory / "source_manifest.json") != technical.source_manifest(technical.CONFIG)
                or summary["quick"] is not quick or summary["analysis"] != {}
                or summary["completed_episodes"] != planned["complete_episodes"]
                or summary["completed_physical_transitions"] != planned["physical_transitions"]
                or summary["completed_poisson_draws"] != planned["poisson_draws"]
                or summary["completed_paired_prefix_checks"] != planned["paired_prefix_checks"]
                or summary["invalid_observations"] != 0 or summary["failed_or_truncated_episodes"] != 0
                or summary["replay_max_absolute_error"] != 0 or not summary["observation_quality"]["targets_met"]):
            raise RuntimeError("O2-D6 D5 technical prerequisite not passed")
    return dict(result, D5_artifacts_checked=564, frozen_D5_source_sha256=D5_PINS,
                D5_output_summary_sha256={p: v[0] for p, v in D5_OUTPUTS.items()})


def stream_manifest(cfg: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    manifest = technical.stream_manifest(cfg, quick=quick)
    dcfg = technical.read_config()
    old = [technical.stream_manifest(dcfg, quick=q) for q in (False, True)]
    current = set().union(*(set(manifest[k]) for k in
                           ("weather_bases", "turbulence", "power", "unused_proxy_sensor", "camera")))
    unit = set(range(9170000, 9180000))
    reserved = unit | {n + o for n in unit for o in (50000000, 60000000, 180000000, 181000000, 182000000)}
    bootstrap = cfg["statistics"]["bootstrap_seed"]
    if (current & reserved or bootstrap in current or bootstrap in reserved
            or any((current | {bootstrap}) & short._integers(h) for h in old)):
        raise RuntimeError("O2-D6 D5/bootstrap/unit seed collision")
    # 用同一旧流检查约束统计种子；只生成清单，不执行旧回合。
    probe = {**cfg, "data": {**cfg["data"], "weather_seed_base": bootstrap, "weather_count": 1}}
    technical.stream_manifest(probe, quick=False)
    legacy_statistics = {prior.read_config(prior.CONFIG)["statistics"]["bootstrap_seed"],
                         technical.photon.completed.read_config()["statistics"]["bootstrap_seed"]}
    if bootstrap in legacy_statistics or bootstrap in short._integers(technical._streams(cfg, cfg["quick"])):
        raise RuntimeError("O2-D6 historical statistical/quick seed collision")
    manifest.update(historical_manifests_checked=25, technical_unit_namespace=9170000,
                    bootstrap_seed=bootstrap, bootstrap_draws_used=not quick,
                    disjointness_scope="declared_G2_C1_B_C_D1_D2_D3_D4_D5_other_D6_mode_and_reserved_units")
    return manifest


def source_manifest(path: str | Path) -> dict[str, Any]:
    target = Path(path) if Path(path).is_absolute() else ROOT / path
    names = ("observation_bridge/photon_development.py", "scripts/evaluate_observation_bridge_o2_photon_development.py",
             "tests/test_observation_bridge_o2_photon_development.py")
    return dict(new_source_sha256={n: optics.file_sha256(ROOT / n) for n in names},
                config_sha256=optics.file_sha256(target), frozen_D5_source=technical.source_manifest(technical.CONFIG),
                reused_statistical_source=optics.file_sha256(ROOT / "observation_bridge/development.py"))


def preflight(path: str | Path = CONFIG, *, quick: bool = False) -> tuple:
    cfg = read_config(path)
    validate_config(cfg)
    spec = cfg["quick" if quick else "data"]
    output = (ROOT / cfg["quick_directory" if quick else "output_directory"]).resolve()
    if output == (ROOT / "outputs").resolve() or not output.is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("O2-D6 output must be a new child of outputs")
    if output.exists():
        raise FileExistsError(f"preserve existing O2-D6 output: {output}")
    frozen, streams = verify_prerequisites(), stream_manifest(cfg, quick=quick)
    short.configure_runtime()
    device = resolve_device(cfg["device"])
    if device.type != "cuda":
        raise RuntimeError("O2-D6 requires CUDA; no CPU fallback")
    parent = read_config(cfg["parent"])
    policies, scorers, models = short.load_assets(device, parent)
    if any(w in m.get("train_weather", []) + m.get("held_out_weather", [])
           for m in models for w in streams["weather_bases"]):
        raise RuntimeError("O2-D6 weather used by frozen scorer")
    report = dict(status="O2_D6_READY_FOR_TECHNICAL_SMOKE" if quick else "O2_D6_READY_FOR_USER_IDE",
                  quick=quick, **cfg["boundary"], **budget(spec), episode_length=spec["episode_length"],
                  weather_count=spec["weather_count"], camera_ids=spec["camera_ids"], device=str(device),
                  loaded_policies=len(policies), loaded_scorers=len(scorers), stream_manifest=streams,
                  frozen_sources=frozen, output_directory=str(output), gain_threshold=None,
                  scientific_gain_analysis=not quick, preflight_model_forward_calls=0, preflight_environment_transitions=0,
                  synthetic_nominal_profile=short.nominal_profile(parent).as_record(),
                  fold_routing="weather_index_modulo_4_fixed_before_observation")
    return cfg, spec, output, device, parent, policies, scorers, models, report


def active_cameras(cfg: dict, spec: dict) -> list[dict]:
    return [c for c in cfg["camera_conditions"] if c["id"] in spec["camera_ids"]]


@torch.no_grad()
def episode_rows(trace: dict, audit: dict, record: dict, cfg: dict, spec: dict, filename: str) -> list[dict]:
    values = dict(power=audit["action_power"], strehl=audit["action_strehl"], phase_rmse=audit["action_phase_rmse"],
                  violation=audit["violation"], requested_applied_gap_rad=(audit["requested_modal"]-audit["applied_modal"]).abs().mean(-1),
                  requested_step_abs_rad=trace["requested_delta"].abs().mean(-1), requested_modal_abs_rad=trace["requested_modal"].abs().mean(-1),
                  observation_latency_ms=audit["observation_latency_ms"], decision_latency_ms=audit["decision_latency_ms"],
                  # 泊松计数/固定正尺度天然非负；没有高斯负值裁剪步骤。
                  camera_negative_clip_fraction=torch.zeros_like(audit["modal_rmse_rad"]), modal_rmse_rad=audit["modal_rmse_rad"],
                  representation_fit_rmse_rad=audit["fit_rmse_rad"])
    if any(v.device.type != "cuda" or not bool(torch.isfinite(v).all()) for v in values.values()):
        raise ValueError("O2-D6 episode metrics require finite CUDA values")
    means = {k: v.double().mean(0) for k, v in values.items()}
    return [dict(weather_seed=record["weather_seed"], weather_index=record["weather_index"], family=f, family_index=fi,
                 camera_condition=record["camera_condition"], counts_per_intensity_unit=record["counts_per_intensity_unit"],
                 read_noise_std=0.0, **{k: record[k] for k in ("controller", "member", "scorer_seed", "scorer_fold")},
                 episode_length=spec["episode_length"], camera_frames=spec["episode_length"]+1,
                 turbulence_stream_seed=record["weather_seed"]+fi, power_stream_seed=record["weather_seed"]+60000000,
                 camera_initial_frame_seed=technical.frame_seed(cfg, record["weather_seed"], record["camera_condition"], 0, fi),
                 camera_seed_sequence_sha256=record["camera_seed_sequence_sha256"], failed=False, truncated=False,
                 trajectory_file=filename, selected_nonoriginal_fraction=float(trace["choice"][:, fi].ne(0).float().mean()),
                 **{k: float(v[fi]) for k, v in means.items()}) for fi, f in enumerate(cfg["family_ids"])]


def validate_rows(rows: list[dict], cfg: dict, spec: dict) -> dict[tuple, dict]:
    weather = technical._streams(cfg, spec)["weather_bases"]
    branches = {b["controller"]: b for b in prior.controller_specs(cfg)}
    cameras = {c["id"]: c for c in active_cameras(cfg, spec)}
    expected = {(c, w, b, f) for c in cameras for w in weather for b in branches for f in cfg["family_ids"]}
    mapping: dict[tuple, dict] = {}
    for r in rows:
        key = r["camera_condition"], r["weather_seed"], r["controller"], r["family"]
        if key in mapping or key not in expected:
            raise ValueError("O2-D6 duplicated or unexpected episode")
        ci, w, bi, f = key
        wi, fi = weather.index(w), cfg["family_ids"].index(f)
        if (any(r[k] != v for k, v in branches[bi].items()) or r["weather_index"] != wi or r["family_index"] != fi
                or r["counts_per_intensity_unit"] != cameras[ci]["counts_per_intensity_unit"] or r["read_noise_std"] != 0.0
                or r["scorer_fold"] != (wi % 4 if branches[bi]["scorer_seed"] is not None else None)
                or r["episode_length"] != spec["episode_length"] or r["camera_frames"] != spec["episode_length"]+1
                or r["turbulence_stream_seed"] != w+fi or r["power_stream_seed"] != w+60000000
                or r["camera_initial_frame_seed"] != technical.frame_seed(cfg, w, ci, 0, fi)
                or not technical.photon._is_hash(r["camera_seed_sequence_sha256"])
                or r["trajectory_file"] != f"weather_{wi:02d}_{ci}_{bi}.pt"
                or r["failed"] is not False or r["truncated"] is not False
                or any(type(r[k]) not in (float, int) or not math.isfinite(r[k]) or r[k] < 0 for k in prior.METRICS)
                or any(not 0 <= r[k] <= 1 for k in ("power", "strehl", "violation", "selected_nonoriginal_fraction"))
                or r["camera_negative_clip_fraction"] != 0.0):
            raise ValueError("O2-D6 episode identity, photon provenance or metric mismatch")
        sequence = [technical.frame_seed(cfg, w, ci, t, j) for t in range(spec["episode_length"]+1) for j in range(3)]
        if r["camera_seed_sequence_sha256"] != hashlib.sha256(json.dumps(sequence).encode("utf-8")).hexdigest():
            raise ValueError("O2-D6 per-frame camera seed sequence changed")
        mapping[key] = r
    if set(mapping) != expected:
        raise ValueError("O2-D6 incomplete episode grid; no silent exclusion")
    return mapping


@torch.no_grad()
def summarize(rows: list[dict], cfg: dict, spec: dict, *, device: torch.device, quick: bool,
              draws: torch.Tensor | None = None) -> dict[str, Any]:
    mapping = validate_rows(rows, cfg, spec)
    if quick:
        if draws is not None:
            raise ValueError("O2-D6 quick must not bootstrap")
        return {}
    if device.type != "cuda":
        raise ValueError("O2-D6 scientific development statistics require CUDA")
    if draws is None:
        draws = prior.stratified_draws(8, repeats=cfg["statistics"]["bootstrap_repeats"],
                                      seed=cfg["statistics"]["bootstrap_seed"], device=device)
    if (draws.device.type != "cuda" or draws.dtype != torch.int64 or draws.shape != (cfg["statistics"]["bootstrap_repeats"], 8)
            or bool(((draws < 0) | (draws >= 8)).any())
            or any(not bool((draws[:, 2*f:2*f+2] % 4 == f).all()) for f in range(4))):
        raise ValueError("O2-D6 fixed-fold complete-weather draws required")
    cameras, ids = active_cameras(cfg, spec), [b["controller"] for b in prior.controller_specs(cfg)]
    weather = technical._streams(cfg, spec)["weather_bases"]
    values = {k: torch.tensor([[[[mapping[(c["id"], w, b, f)][k] for f in cfg["family_ids"]]
                                for w in weather] for b in ids] for c in cameras], device=device, dtype=torch.float64)
              for k in prior.METRICS}
    originals = [ids.index(f"original_{m}") for m in range(3)]
    currents = [ids.index(f"current_{m}_{s}") for m in range(3) for s in cfg["scorer_seeds"]]
    cells, family_table, clusters = {}, [], {}
    for ci, camera in enumerate(cameras):
        means = {k: dict(integrator=v[ci, 0], original=v[ci, originals].mean(0), current=v[ci, currents].mean(0))
                 for k, v in values.items()}
        power = {name: v.mean(-1) for name, v in means["power"].items()}
        if bool((power["integrator"] <= 0).any()):
            raise ValueError("nonpositive complete-weather integrator power")
        clusters[camera["id"]] = power
        comparisons = {}
        for left, right in (("original", "integrator"), ("current", "integrator"), ("current", "original")):
            diff = power[left]-power[right]
            part = dict(mean_power_difference=float(diff.mean()), descriptive_ci95=prior.paired_interval(diff, draws),
                        positive_weather_count=int((diff > 0).sum()), per_weather_power_difference=diff.tolist())
            if right == "integrator":
                part.update(relative_gain=float(diff.mean()/power[right].mean()),
                            relative_gain_descriptive_ci95=prior.paired_interval(diff, draws, power[right]))
            comparisons[f"{left}_vs_{right}"] = part
        cells[camera["id"]] = dict(counts_per_intensity_unit=camera["counts_per_intensity_unit"], weather_clusters=8,
            comparisons=comparisons, method_means={n: {k: float(v[n].mean()) for k, v in means.items()}
            for n in ("integrator", "original", "current")}, per_controller_means={b: {k: float(v[ci, bi].mean())
            for k, v in values.items()} for bi, b in enumerate(ids)}, metric_deltas={k: dict(
            current_vs_integrator=float((v["current"]-v["integrator"]).mean()),
            current_vs_original=float((v["current"]-v["original"]).mean())) for k, v in means.items()})
        for fi, f in enumerate(cfg["family_ids"]):
            base = means["power"]["integrator"][:, fi]
            if bool((base <= 0).any()):
                raise ValueError("nonpositive family integrator power")
            family_table.append(dict(camera_condition=camera["id"], family=f, weather_clusters=8,
                current_relative_gain_vs_integrator=float((means["power"]["current"][:, fi]-base).mean()/base.mean()),
                current_minus_original_power=float((means["power"]["current"][:, fi]-means["power"]["original"][:, fi]).mean())))
    zero, effects = clusters["noiseless"], {}
    for camera in cameras[1:]:
        p = clusters[camera["id"]]
        increment = (p["current"]-p["integrator"])-(zero["current"]-zero["integrator"])
        gain = lambda x: (x["current"].mean()-x["integrator"].mean())/x["integrator"].mean()
        boot = lambda x: (x["current"][draws].mean(1)-x["integrator"][draws].mean(1))/x["integrator"][draws].mean(1)
        change = boot(p)-boot(zero)
        effects[camera["id"]] = dict(counts_per_intensity_unit=camera["counts_per_intensity_unit"],
            method_power_changes={n: dict(noisy_minus_noiseless_power=float((p[n]-zero[n]).mean()),
                descriptive_ci95=prior.paired_interval(p[n]-zero[n], draws), per_weather_power_difference=(p[n]-zero[n]).tolist())
                for n in ("integrator", "original", "current")}, current_increment_change_noisy_minus_noiseless=dict(
                mean_power_difference=float(increment.mean()), descriptive_ci95=prior.paired_interval(increment, draws)),
            current_relative_gain_change_fraction=float(gain(p)-gain(zero)),
            current_relative_gain_change_descriptive_ci95_fraction=[float(v) for v in torch.quantile(change, change.new_tensor([.025, .975]))])
    result = dict(status="DEVELOPMENT_DESCRIPTIVE_NOT_CONFIRMATION", cells=cells, family_table=family_table,
        photon_noise_effects_vs_noiseless=effects, independent_weather_clusters=8, frame_independence_assumed=False,
        interval_scope="pointwise_descriptive_conditional_on_frozen_models_and_fixed_fold_routing_not_simultaneous",
        statistical_backend="cuda", seed_aggregation="average_of_separate_closed_loop_trajectories_not_ensemble_deployment",
        training_uncertainty_covered=False, statistical_power_guaranteed=False, multiple_comparison_significance_claims=False,
        gain_threshold=None, independent_confirmation=False, historical_gate_reclassification=False,
        quality_and_safety_not_implied_by_power_gain=True, photon_scale_units=cfg["noise_units"], real_camera_noise_calibrated=False)
    json.dumps(result, allow_nan=False)
    return result


@torch.no_grad()
def run(path: str | Path = CONFIG, *, quick: bool = False, preflight_only: bool = False) -> dict[str, Any]:
    cfg, spec, output, device, parent, policies, scorers, models, report = preflight(path, quick=quick)
    if preflight_only:
        return report
    output.mkdir(parents=True, exist_ok=False)
    context = dict(physical_transitions=0, completed_episode_batches=0, incomplete_batch_size=0)
    started = time.perf_counter()
    try:
        for directory in ("trajectories", "audit", "camera"):
            (output / directory).mkdir()
        for name, value in (("effective_config.json", cfg), ("preflight.json", report), ("model_manifest.json", models),
                            ("stream_manifest.json", report["stream_manifest"]), ("source_manifest.json", source_manifest(path))):
            optics.write_json(output / name, value)
        sensor, bridge = optics.make_components(read_config(cfg["optics_config"]), device)
        rows, records, manifest = [], [], []
        rmse: dict[str, list[torch.Tensor]] = {}
        ideal_max = replay_max = 0.0
        prefixes = policy_calls = scorer_calls = progress_count = poisson_draws = 0
        def progress(row: dict) -> None:
            nonlocal progress_count
            progress_count += 1
            elapsed = time.perf_counter()-started
            count, total = row["physical_transitions"], report["physical_transitions"]
            row.update(total_physical_transitions=total, elapsed_seconds=elapsed, eta_seconds=elapsed*(total-count)/count,
                       transitions_per_second=count/max(elapsed, 1e-9), cuda_allocated_gib=torch.cuda.memory_allocated(device)/2**30,
                       cuda_reserved_gib=torch.cuda.memory_reserved(device)/2**30)
            with (output / "progress.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
            if row["observation_step"] % 20 == 0 or row["observation_step"] == spec["episode_length"]:
                print(f"O2-D6 {'技术冒烟' if quick else '开发对照'} {count}/{total} | 天气{row['weather_seed']} "
                      f"{row['camera_condition']} {row['controller']} {row['observation_step']}/{spec['episode_length']}帧 | "
                      f"{row['transitions_per_second']:.1f}转移/s 剩余={row['eta_seconds']:.1f}s | "
                      f"显存={row['cuda_allocated_gib']:.2f}/{row['cuda_reserved_gib']:.2f}GiB", flush=True)
        for wi, seed in enumerate(report["stream_manifest"]["weather_bases"]):
            for camera in active_cameras(cfg, spec):
                originals, original_cameras = {}, {}
                for branch in prior.controller_specs(cfg):
                    policy = policies.get(branch["member"])
                    selector = None if branch["scorer_seed"] is None else scorers[wi % 4, branch["scorer_seed"]]
                    trace, audit, record, camera_rows = technical.rollout(cfg, spec, parent, branch, seed=seed, weather_index=wi,
                        camera=camera, sensor=sensor, bridge=bridge, policy=policy, selector=selector, progress=progress,
                        context=context, partial_directory=output / "partial")
                    context.update(phase="save_and_replay", incomplete_batch_size=0)
                    technical.validate_camera_rows(camera_rows, cfg, spec, weather=seed, camera=camera)
                    name = f"weather_{wi:02d}_{camera['id']}_{branch['controller']}"
                    for directory, value in (("trajectories", trace), ("audit", audit)):
                        technical.technical.save_tensors(output / directory / f"{name}.pt", value)
                    optics.write_json(output / "camera" / f"{name}.json", camera_rows)
                    saved = torch.load(output / "trajectories" / f"{name}.pt", map_location=device, weights_only=True)
                    record["replay"] = short.replay_visible(saved, {**cfg, "episode_length": spec["episode_length"]}, policy, selector)
                    replay_max = max(replay_max, record["replay"]["max_absolute_error"])
                    if selector is None and branch["member"] is not None:
                        originals[branch["member"]], original_cameras[branch["member"]] = trace, camera_rows
                    if selector is not None:
                        short.require_prefix(originals[branch["member"]], trace, cfg["selector_start_step"])
                        n = 3*(cfg["selector_start_step"]+1)
                        if original_cameras[branch["member"]][:n] != camera_rows[:n]:
                            raise RuntimeError("O2-D6 selector-disabled photon camera prefix differs")
                        record["paired_prefix_exact"] = record["paired_camera_prefix_exact"] = True
                        prefixes += 1
                    rmse.setdefault(camera["id"], []).append(audit["modal_rmse_rad"].detach().clone())
                    if camera["counts_per_intensity_unit"] is None:
                        ideal_max = max(ideal_max, record["modal_error_max_rad"])
                    part = episode_rows(trace, audit, record, cfg, spec, f"{name}.pt")
                    rows.extend(part)
                    records.append(record)
                    manifest.append(dict(file=f"{name}.pt", camera_file=f"{name}.json", **branch, weather_seed=seed,
                        camera_condition=camera["id"], visible_sha256=optics.file_sha256(output / "trajectories" / f"{name}.pt"),
                        audit_sha256=optics.file_sha256(output / "audit" / f"{name}.pt"),
                        camera_sha256=optics.file_sha256(output / "camera" / f"{name}.json")))
                    for filename, items in (("records.jsonl", part), ("batch_records.jsonl", [record])):
                        with (output / filename).open("a", encoding="utf-8") as handle:
                            for value in items:
                                handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n")
                    context["completed_episode_batches"] += 1
                    policy_calls += record["policy_forward_calls"]
                    scorer_calls += record["scorer_forward_calls"]
                    poisson_draws += record["poisson_draws"]
        technical.validate_records(records, cfg, spec)
        validate_rows(rows, cfg, spec)
        if (context["physical_transitions"] != report["physical_transitions"] or progress_count != report["batched_steps"]
                or len(rows) != report["complete_episodes"] or policy_calls != report["policy_forward_calls"]
                or scorer_calls != report["scorer_forward_calls"] or prefixes != report["paired_prefix_checks"]
                or poisson_draws != report["poisson_draws"]):
            raise RuntimeError("O2-D6 complete execution budget mismatch")
        context.update(phase="statistics", incomplete_batch_size=0)
        quality = technical.observation_quality(rmse, ideal_max, cfg)
        if not quality["targets_met"]:
            raise RuntimeError("O2-D6 technical observation targets not met; preserve all evidence")
        draws = None if quick else prior.stratified_draws(8, repeats=cfg["statistics"]["bootstrap_repeats"],
                                                        seed=cfg["statistics"]["bootstrap_seed"], device=device)
        analysis = summarize(rows, cfg, spec, device=device, quick=quick, draws=draws)
        if draws is not None:
            technical.technical.save_tensors(output / "bootstrap_draws.pt", {"weather_indices": draws})
        verify_prerequisites()
        if short.read_json(output / "source_manifest.json") != source_manifest(path):
            raise RuntimeError("O2-D6 source/config changed during execution")
        result = dict(report, status="O2_D6_TECHNICAL_SMOKE_ONLY" if quick else "O2_D6_DEVELOPMENT_COMPLETE_REQUIRES_READ_ONLY_AUDIT",
            completed_episodes=len(rows), completed_episode_batches=len(records),
            completed_physical_transitions=context["physical_transitions"], invalid_observations=0, failed_or_truncated_episodes=0,
            completed_paired_prefix_checks=prefixes, completed_poisson_draws=poisson_draws, replay_max_absolute_error=replay_max,
            completed_policy_forward_calls=policy_calls, completed_scorer_forward_calls=scorer_calls,
            replay_policy_forward_calls=policy_calls, replay_scorer_forward_calls=scorer_calls, replay_environment_transitions=0,
            observation_quality=quality, analysis=analysis, bootstrap_index_artifact=None if quick else "bootstrap_draws.pt",
            raw_camera_frames_saved=0, inverse_crime_limitation=True, real_accuracy_verified=False,
            real_camera_noise_calibrated=False, realtime_verified=False, equal_poisson_rng_end_state_not_assumed=True,
            elapsed_seconds=time.perf_counter()-started, runtime=technical.photon.static._runtime(device),
            latency_scope="batch-3 CUDA render/photon sampling/measurement including hash synchronization; decision excludes projection; not real exposure/end-to-end latency",
            next_action="Read-only audit; no automatic rerun, tuning, training, confirmation or hardware actions.")
        optics.write_json(output / "trajectory_manifest.json", manifest)
        optics.write_json(output / "summary.json", result)
        artifacts = {p.relative_to(output).as_posix(): optics.file_sha256(p) for p in output.rglob("*") if p.is_file()}
        optics.write_json(output / "SUCCESS.json", dict(status=result["status"], summary_sha256=artifacts["summary.json"], artifact_sha256=artifacts))
        return result
    except BaseException as exc:
        optics.write_json(output / "failure.json", dict(status="O2_D6_STOPPED_NO_AUTOMATIC_RETRY", exception=type(exc).__name__,
            message=str(exc), rejection_type=technical.photon.static.rejection_type(exc) if isinstance(exc, ValueError) else None,
            traceback=traceback.format_exc(), last_context=context, **cfg["boundary"]))
        raise
