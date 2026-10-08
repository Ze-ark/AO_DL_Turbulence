"""O2-D2：冻结控制器、四档人工相机噪声的短闭环技术检查。

复用封存 O2-C 的图像端口和 O2-B 的因果接口，不修改既有实现。
只检查读数/时序/重放；不得用此入口生成补偿收益排名。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback
from typing import Any, Callable

import torch

from observation_bridge import development as prior
from observation_bridge import read_noise_diagnostic as static
from src.rl.r4_observation import R4Interface
from src.rl.r4_trajectory import anchor_delta
from src.runtime import resolve_device

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/experiments/observation_bridge_o2_d2_noise_closed_loop_v1.yaml"
short = prior.short
optics = short.optics
PINS = {
    "observation_bridge/read_noise_diagnostic.py": "14819579f1b595d0b3ff5ed2ea50c776d6f99eff38745700f9b995d7a0eec660",
    "scripts/diagnose_observation_bridge_o2_read_noise.py": "f2f8241e0ffe64c1f7bc21a0569abe1c40901ef17c3d9948fbf6d125118960f1",
    "tests/test_observation_bridge_o2_read_noise.py": "93519f93f60e99a439b071ac49d17bf7687610b96e3ecbb9deb25c4a4100333f",
    static.CONFIG: "eed0b39e352c1f7ba2595e26b1444fc158897072947f6a5de482435475e98af1",
}
D1_SUMMARIES = {
    "outputs/observation_bridge_o2_d1_read_noise_v1": "f428db35849aa5f18dbf74441739c1c74074cfb5ee07349c94dcb0332d799373",
    "outputs/observation_bridge_o2_d1_read_noise_v1_quick": "0b126e9faf4519c2409b2defb0043d61647a74cac3cffe25dd2d3c267d2cfb79",
}


def read_config(path: str | Path = CONFIG) -> dict[str, Any]:
    return prior.read_config(path)


def validate_config(cfg: dict[str, Any]) -> None:
    expected = dict(
        schema="observation_bridge_o2_d2_noise_closed_loop_v1",
        scope="synthetic_read_noise_short_loop_technical_only", device="cuda",
        optics_config="configs/experiments/observation_bridge_o2_optics_v1.yaml",
        parent="configs/experiments/s4_r5_policy_training_v1.yaml",
        data=dict(weather_seed_base=8600000, weather_count=4, weather_seed_stride=10, episode_length=32,
                  camera_ids=["noiseless", "small_read_noise", "medium_read_noise", "large_read_noise"]),
        quick=dict(weather_seed_base=8660000, weather_count=1, weather_seed_stride=10, episode_length=28,
                   camera_ids=["noiseless", "large_read_noise"]),
        family_ids=["frozen", "boiling", "varying"],
        camera_conditions=[dict(id="noiseless", read_noise_std=0.0), dict(id="small_read_noise", read_noise_std=.001),
                           dict(id="medium_read_noise", read_noise_std=.3), dict(id="large_read_noise", read_noise_std=1.0)],
        camera_seed_offset=170000000,
        camera_noise_distribution="additive_gaussian_then_clip_negative_intensity_to_zero",
        camera_noise_units="arbitrary_synthetic_intensity", camera_draw_including_noiseless=True,
        batch_size=3, controller_branches=13, policy_initializations=[0, 1, 2],
        scorer_seeds=[7564000, 7564001, 7564002], selector_start_step=25,
        integrator=dict(gain=.15, leak=.1, tracking_gain=.5), policy_scale=1.75, candidate_epsilon=.1,
        thresholds=dict(ideal_modal_error_max_rad=.001, noisy_modal_rmse_mean_rad=.01,
                        noisy_modal_rmse_p95_rad=.02, replay_max_absolute_error=1e-6,
                        invalid_observations=0, failed_or_truncated_episodes=0), gain_threshold=None,
        boundary=dict(training_updates=0, scientific_gain_analysis=False, independent_confirmation=False,
                      old_confirmation_trajectory_access=False, real_data_access=False, real_slm_actions=False,
                      historical_gate_reclassification=False, truth_fallback=False, automatic_retry=False))
    if (not isinstance(cfg, dict) or set(cfg) != set(expected) | {"output_directory", "quick_directory"}
            or any(json.dumps(cfg.get(k), sort_keys=True, allow_nan=False) != json.dumps(v, sort_keys=True)
                   for k, v in expected.items())):
        raise ValueError("O2-D2 fixed technical contract changed")


def budget(spec: dict[str, Any]) -> dict[str, int]:
    groups = spec["weather_count"] * len(spec["camera_ids"])
    batches, steps = groups * 13, spec["episode_length"]
    return dict(episode_batches=batches, complete_episodes=batches * 3, batched_steps=batches * steps,
                physical_transitions=batches * steps * 3, policy_forward_calls=groups * 12 * steps,
                scorer_forward_calls=groups * 9 * max(0, steps - 25), paired_prefix_checks=groups * 9,
                camera_batch_draws=batches * (steps + 1), camera_family_frame_draws=batches * (steps + 1) * 3)


def verify_prerequisites() -> dict[str, Any]:
    for name, digest in PINS.items():
        if optics.file_sha256(ROOT / name) != digest:
            raise RuntimeError(f"O2-D2 frozen D1 source changed: {name}")
    dcfg = static.read_config()
    static.validate_config(dcfg)
    result = static.verify_prerequisites(dcfg)
    for relative, digest in D1_SUMMARIES.items():
        directory = ROOT / relative
        success = short.read_json(directory / "SUCCESS.json")
        names = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
                 if p.is_file() and p.name != "SUCCESS.json"}
        if (len(names) != 8 or set(success["artifact_sha256"]) != names
                or success["summary_sha256"] != digest or optics.file_sha256(directory / "summary.json") != digest
                or (directory / "failure.json").exists()):
            raise RuntimeError("O2-D2 completed D1 artifact seal changed")
        for name, expected in success["artifact_sha256"].items():
            target = (directory / name).resolve()
            if not target.is_relative_to(directory.resolve()) or optics.file_sha256(target) != expected:
                raise RuntimeError("O2-D2 completed D1 artifact changed")
        if (short.read_json(directory / "source_manifest.json") != static.source_manifest(static.CONFIG)
                or short.read_json(directory / "effective_config.json") != dcfg):
            raise RuntimeError("O2-D2 D1 executed source identity changed")
    result.update(D1_artifacts_checked=16, D1_summary_sha256=D1_SUMMARIES, frozen_D1_source_sha256=PINS)
    return result


def stream_manifest(cfg: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    # 复用纯清单检查（无环境/旧轨迹）；再补 D1 静态种子和新保留命名空间。
    manifest = prior.verify_streams(cfg, quick=quick)
    current = set().union(*(set(manifest[k]) for k in
                           ("weather_bases", "turbulence", "power", "unused_proxy_sensor", "camera")))
    dcfg = static.read_config()
    ccfg = prior.read_config(prior.CONFIG)
    history = [prior._streams(ccfg, ccfg[k]) for k in ("data", "quick")]
    history.extend(static.stream_manifest(dcfg, quick=q) for q in (False, True))
    reserved = set(range(8470000, 8480000)) | set(range(8520000, 8530000)) | set(range(8670000, 8680000))
    # 相机/功率流也保留单元天气相同的偏移命名空间。
    reserved |= {n + offset for n in range(8670000, 8680000) for offset in (50000000, 60000000, 170000000)}
    if current & reserved or any(current & short._integers(m) for m in history):
        raise RuntimeError("O2-D2 D1 or reserved unit seed collision")
    manifest.update(historical_manifests_checked=17, technical_unit_namespace=8670000,
                    disjointness_scope="declared_G2_C1_B_C_D1_other_D2_mode_and_reserved_units")
    return manifest


def source_manifest(path: str | Path) -> dict[str, Any]:
    target = Path(path) if Path(path).is_absolute() else ROOT / path
    names = ["observation_bridge/noise_closed_loop.py", "scripts/verify_observation_bridge_o2_noise_closed_loop.py",
             "tests/test_observation_bridge_o2_noise_closed_loop.py"]
    return dict(new_source_sha256={n: optics.file_sha256(ROOT / n) for n in names},
                config_sha256=optics.file_sha256(target), reused_C_source=prior.source_manifest(prior.CONFIG),
                frozen_D1_source_sha256=PINS)


def preflight(path: str | Path = CONFIG, *, quick: bool = False) -> tuple:
    cfg = read_config(path)
    validate_config(cfg)
    spec = cfg["quick" if quick else "data"]
    output = (ROOT / cfg["quick_directory" if quick else "output_directory"]).resolve()
    if output == (ROOT / "outputs").resolve() or not output.is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("O2-D2 output must be a new child of outputs")
    if output.exists():
        raise FileExistsError(f"preserve existing O2-D2 output: {output}")
    frozen = verify_prerequisites()
    streams = stream_manifest(cfg, quick=quick)
    short.configure_runtime()
    device = resolve_device(cfg["device"])
    if device.type != "cuda":
        raise RuntimeError("O2-D2 requires CUDA; no CPU fallback")
    parent = read_config(cfg["parent"])
    policies, scorers, models = short.load_assets(device, parent)
    if any(w in m.get("train_weather", []) + m.get("held_out_weather", [])
           for m in models for w in streams["weather_bases"]):
        raise RuntimeError("O2-D2 weather used by frozen scorer")
    report = dict(status="O2_D2_READY_FOR_TECHNICAL_SMOKE" if quick else "O2_D2_READY_FOR_USER_IDE",
                  quick=quick, **cfg["boundary"], **budget(spec), episode_length=spec["episode_length"],
                  weather_count=spec["weather_count"], camera_ids=spec["camera_ids"], device=str(device),
                  loaded_policies=len(policies), loaded_scorers=len(scorers), stream_manifest=streams,
                  frozen_sources=frozen, output_directory=str(output), gain_threshold=None,
                  preflight_model_forward_calls=0, preflight_environment_transitions=0,
                  synthetic_nominal_profile=short.nominal_profile(parent).as_record(),
                  fold_routing="weather_index_modulo_4_fixed_before_observation")
    return cfg, spec, output, device, parent, policies, scorers, models, report


class PairedReadNoise(prior.PairedReadNoise):
    """扩大到事前固定四档；抽样/负强度裁剪复用封存实现。"""

    def __init__(self, device: torch.device, seeds: list[int], noise_std: float):
        if (device.type != "cuda" or len(seeds) != 3 or len(set(seeds)) != 3
                or any(type(s) is not int for s in seeds) or type(noise_std) not in (float, int)
                or noise_std not in (0, .001, .3, 1)):
            raise ValueError("O2-D2 fixed four-level CUDA camera noise required")
        resolved = resolve_device(str(device))
        self.device = resolve_device(f"cuda:{torch.cuda.current_device() if resolved.index is None else resolved.index}")
        self.noise_std = float(noise_std)
        self.generators = [torch.Generator(device=self.device).manual_seed(seed) for seed in seeds]
        self.frames = 0


def save_tensors(path: Path, values: dict[str, torch.Tensor]) -> None:
    # 仅写磁盘时转 CPU，不是 CPU 测量/控制计算。
    torch.save({k: v.detach().cpu() for k, v in values.items()}, path)


@torch.no_grad()
def rollout(cfg: dict[str, Any], spec: dict[str, Any], parent: dict, branch: dict, *, seed: int,
            weather_index: int, camera: dict, sensor: Any, bridge: Any, policy: Any, selector: Any,
            progress: Callable[[dict], None], context: dict, partial_directory: Path) -> tuple[dict, dict, dict]:
    context.update(controller=branch["controller"], weather_seed=seed, camera_condition=camera["id"],
                   read_noise_std=camera["read_noise_std"], action_step=None, phase="reset",
                   current_batch_completed_steps=0, pending_action=False, camera_frames=0)
    env = short.make_environment(parent, bridge.basis, seed, spec["episode_length"])
    noise = PairedReadNoise(bridge.device, [seed + i + cfg["camera_seed_offset"] for i in range(3)], camera["read_noise_std"])
    port = prior.DevelopmentPort(env, sensor, bridge, noise, cfg["thresholds"]["ideal_modal_error_max_rad"])
    visible: dict[str, list[torch.Tensor]] = {}
    audits: dict[str, list[torch.Tensor]] = {}
    pending: dict[str, torch.Tensor] = {}
    try:
        readout, initial_audit = port.reset(seed)
        interface = R4Interface()
        interface.reset(readout.residual, episode_id=f"o2-d2-{seed}-{camera['id']}-{branch['controller']}")
        visible = {k: [] for k in ("history", "valid", "original", "selected", "choice", "prediction", "requested_delta",
                                  "requested_modal", "residual", "measured_power", "next_clock", "power_action_step", "power_arrival_step")}
        visible["residual"].append(readout.residual.clone())
        audits = {k: [v.clone()] for k, v in initial_audit.items()}
        def on_transition() -> None:
            context["physical_transitions"] += 3
        for step in range(spec["episode_length"]):
            context.update(action_step=step, phase="decision", pending_action=False)
            view = interface.snapshot()
            if readout.observation_step != step or view.observation_step != step:
                raise RuntimeError("O2-D2 causal observation clock mismatch")
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            original, selected, choice, prediction = short.choose_command(view.features, view.valid, policy, selector, step=step, cfg=cfg)
            end.record(); end.synchronize()
            decision_ms = start.elapsed_time(end)
            action = interface.issue(anchor_delta(view.features[:, -1], cfg["integrator"]), selected, step=step)
            pending = dict(history=view.features, valid=view.valid, original=original, selected=selected,
                           choice=choice, prediction=prediction, requested_delta=action.requested_delta_rad,
                           requested_modal=action.requested_modal_rad)
            context.update(phase="physical_step_and_next_observation", pending_action=True)
            readout, power, audit = port.step(action.requested_delta_rad, step, on_transition)
            transition = interface.observe_next(readout.residual, step=readout.observation_step, power=power)
            if not bool(transition.action_power_valid.all()):
                raise RuntimeError("O2-D2 causal action power missing")
            for key, value in dict(pending, measured_power=power.value,
                                   next_clock=transition.next_history.features[:, -1, 75:79],
                                   power_action_step=torch.tensor(power.action_step, device=bridge.device),
                                   power_arrival_step=torch.tensor(power.arrival_observation_step, device=bridge.device)).items():
                visible[key].append(value.detach().clone())
            visible["residual"].append(readout.residual.clone())
            audit["decision_latency_ms"] = readout.residual.new_full((3,), decision_ms)
            for key, value in audit.items():
                audits.setdefault(key, []).append(value.detach().clone())
            context.update(current_batch_completed_steps=step + 1, pending_action=False, camera_frames=noise.frames)
            pending = {}
            progress(dict(weather_seed=seed, camera_condition=camera["id"], controller=branch["controller"],
                          observation_step=step + 1, physical_transitions=context["physical_transitions"],
                          modal_error_max_rad=float(audit["modal_error_rad"].max())))
        trace, audit_trace = short._stack(visible), short._stack(audits)
        if noise.frames != spec["episode_length"] + 1 or any(
                v.is_floating_point() and not bool(torch.isfinite(v).all()) for d in (trace, audit_trace) for v in d.values()):
            raise RuntimeError("O2-D2 camera budget or finite trace check failed")
        record = dict(**branch, weather_seed=seed, weather_index=weather_index, camera_condition=camera["id"],
                      read_noise_std=camera["read_noise_std"], complete_episodes=3,
                      physical_transitions=spec["episode_length"] * 3, scorer_fold=weather_index % 4 if selector else None,
                      camera_frames_per_family=noise.frames, camera_final_rng_sha256=noise.final_state_sha256(),
                      modal_error_max_rad=float(audit_trace["modal_error_rad"].max()),
                      policy_forward_calls=spec["episode_length"] if policy is not None else 0,
                      scorer_forward_calls=max(0, spec["episode_length"] - cfg["selector_start_step"]) if selector else 0)
        return trace, audit_trace, record
    except BaseException:
        context.update(camera_frames=noise.frames, current_batch_environment_step=env.step_count,
                       incomplete_batch_size=3)
        # 保存未完成回合及最后已发请求；不补假读数，不生成成功标记，不吞原始错误。
        try:
            partial_directory.mkdir(parents=True, exist_ok=False)
            save_tensors(partial_directory / "visible.pt", short._stack(visible))
            save_tensors(partial_directory / "audit.pt", short._stack(audits))
            save_tensors(partial_directory / "pending_request.pt", pending)
        except BaseException as save_error:
            context["partial_save_error"] = repr(save_error)
        raise


def validate_records(records: list[dict], cfg: dict, spec: dict) -> None:
    branches = {b["controller"]: b for b in prior.controller_specs(cfg)}
    seeds = [spec["weather_seed_base"] + i * spec["weather_seed_stride"] for i in range(spec["weather_count"])]
    expected = {(s, c, b) for s in seeds for c in spec["camera_ids"] for b in branches}
    camera_stds = {c["id"]: c["read_noise_std"] for c in cfg["camera_conditions"]}
    seen = set()
    for r in records:
        key = (r["weather_seed"], r["camera_condition"], r["controller"])
        if key in seen or key not in expected:
            raise ValueError("O2-D2 duplicated or unexpected record")
        branch = branches[r["controller"]]
        wi = seeds.index(r["weather_seed"])
        if (any(r[k] != v for k, v in branch.items()) or r["weather_index"] != wi
                or r["scorer_fold"] != (wi % 4 if branch["scorer_seed"] is not None else None)
                or r["read_noise_std"] != camera_stds[r["camera_condition"]]
                or r["complete_episodes"] != 3 or r["physical_transitions"] != 3 * spec["episode_length"]
                or r["camera_frames_per_family"] != spec["episode_length"] + 1
                or r["policy_forward_calls"] != (spec["episode_length"] if branch["member"] is not None else 0)
                or r["scorer_forward_calls"] != (max(0, spec["episode_length"] - 25) if branch["scorer_seed"] is not None else 0)):
            raise ValueError("O2-D2 record identity/fold/budget mismatch")
        seen.add(key)
    if seen != expected:
        raise ValueError("O2-D2 incomplete grid; no silent exclusion")


def observation_quality(parts: dict[str, list[torch.Tensor]], ideal_max: float, cfg: dict) -> dict:
    known = {c["id"] for c in cfg["camera_conditions"]}
    if not parts or not set(parts).issubset(known) or any(not values for values in parts.values()):
        raise ValueError("O2-D2 empty or unknown quality cell")
    cells = []
    for camera in cfg["camera_conditions"]:
        if camera["id"] not in parts:
            continue
        values = torch.cat([x.reshape(-1).double() for x in parts[camera["id"]]])
        if values.device.type != "cuda" or not bool(torch.isfinite(values).all()):
            raise ValueError("O2-D2 quality statistics require finite CUDA observations")
        mean, p95 = float(values.mean()), float(torch.quantile(values, .95))
        target = camera["read_noise_std"] == 0 or (mean <= cfg["thresholds"]["noisy_modal_rmse_mean_rad"]
                    and p95 <= cfg["thresholds"]["noisy_modal_rmse_p95_rad"])
        cells.append(dict(camera_condition=camera["id"], read_noise_std=camera["read_noise_std"],
                          observed_family_frames=len(values), mean_modal_rmse_rad=mean, p95_modal_rmse_rad=p95,
                          targets_met=target, including_initial_frame=True))
    return dict(cells=cells, ideal_max_modal_error_rad=ideal_max,
                targets_met=ideal_max <= cfg["thresholds"]["ideal_modal_error_max_rad"] and all(c["targets_met"] for c in cells),
                noisy_error_used_to_replace_measurements=False,
                scope="technical_observation_targets_not_compensation_gain_gate",
                inference_unit="descriptive_correlated_frames_not_independent_weather_samples")


def run(path: str | Path = CONFIG, *, quick: bool = False, preflight_only: bool = False) -> dict[str, Any]:
    cfg, spec, output, device, parent, policies, scorers, models, report = preflight(path, quick=quick)
    if preflight_only:
        return report
    output.mkdir(parents=True, exist_ok=False)
    context = dict(physical_transitions=0, completed_episode_batches=0, incomplete_batch_size=0)
    started = time.perf_counter()
    try:
        for directory in ("trajectories", "audit"):
            (output / directory).mkdir()
        for name, value in (("effective_config.json", cfg), ("preflight.json", report), ("model_manifest.json", models),
                            ("stream_manifest.json", report["stream_manifest"]), ("source_manifest.json", source_manifest(path))):
            optics.write_json(output / name, value)
        sensor, bridge = optics.make_components(read_config(cfg["optics_config"]), device)
        records, manifest = [], []
        rmse: dict[str, list[torch.Tensor]] = {}
        ideal_max = replay_max = 0.0
        prefixes = policy_calls = scorer_calls = progress_count = 0
        rng_states: dict[int, list[str]] = {}
        def progress(row: dict) -> None:
            nonlocal progress_count
            progress_count += 1
            elapsed = time.perf_counter() - started
            count, total = row["physical_transitions"], report["physical_transitions"]
            row.update(total_physical_transitions=total, elapsed_seconds=elapsed, eta_seconds=elapsed / count * (total - count),
                       cuda_allocated_gb=torch.cuda.memory_allocated(device) / 2**30)
            with (output / "progress.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            if row["observation_step"] % 8 == 0 or row["observation_step"] == spec["episode_length"]:
                print(f"O2-D2 {'技术冒烟' if quick else '短闭环检查'} {count}/{total} | {row['camera_condition']} "
                      f"{row['controller']} {row['observation_step']}/{spec['episode_length']}帧 | "
                      f"观测误差={row['modal_error_max_rad']:.3g}rad | 剩余={row['eta_seconds']:.1f}s | "
                      f"显存={row['cuda_allocated_gb']:.2f}GB", flush=True)
        for wi, seed in enumerate(report["stream_manifest"]["weather_bases"]):
            for camera in (c for c in cfg["camera_conditions"] if c["id"] in spec["camera_ids"]):
                originals = {}
                for branch in prior.controller_specs(cfg):
                    policy = policies.get(branch["member"])
                    selector = None if branch["scorer_seed"] is None else scorers[wi % 4, branch["scorer_seed"]]
                    trace, audit, record = rollout(cfg, spec, parent, branch, seed=seed, weather_index=wi, camera=camera,
                        sensor=sensor, bridge=bridge, policy=policy, selector=selector, progress=progress,
                        context=context, partial_directory=output / "partial")
                    context.update(phase="save_and_replay", incomplete_batch_size=0)
                    name = f"weather_{wi:02d}_{camera['id']}_{branch['controller']}.pt"
                    for directory, value in (("trajectories", trace), ("audit", audit)):
                        save_tensors(output / directory / name, value)
                    saved = torch.load(output / "trajectories" / name, map_location=device, weights_only=True)
                    record["replay"] = short.replay_visible(saved, {**cfg, "episode_length": spec["episode_length"]}, policy, selector)
                    replay_max = max(replay_max, record["replay"]["max_absolute_error"])
                    if selector is None and branch["member"] is not None:
                        originals[branch["member"]] = trace
                    if selector is not None:
                        short.require_prefix(originals[branch["member"]], trace, cfg["selector_start_step"])
                        record["paired_prefix_exact"] = True
                        prefixes += 1
                    final_rng = record["camera_final_rng_sha256"]
                    if seed in rng_states and rng_states[seed] != final_rng:
                        raise RuntimeError("O2-D2 exogenous camera draw count differs across branches/levels")
                    rng_states[seed] = final_rng
                    rmse.setdefault(camera["id"], []).append(audit["modal_rmse_rad"].detach().clone())
                    if camera["read_noise_std"] == 0:
                        ideal_max = max(ideal_max, record["modal_error_max_rad"])
                    records.append(record)
                    manifest.append(dict(file=name, **branch, weather_seed=seed, camera_condition=camera["id"],
                                         visible_sha256=optics.file_sha256(output / "trajectories" / name),
                                         audit_sha256=optics.file_sha256(output / "audit" / name)))
                    with (output / "records.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    context["completed_episode_batches"] += 1
                    policy_calls += record["policy_forward_calls"]
                    scorer_calls += record["scorer_forward_calls"]
        validate_records(records, cfg, spec)
        if (context["physical_transitions"] != report["physical_transitions"] or progress_count != report["batched_steps"]
                or policy_calls != report["policy_forward_calls"] or scorer_calls != report["scorer_forward_calls"]
                or prefixes != report["paired_prefix_checks"]):
            raise RuntimeError("O2-D2 execution budget mismatch")
        quality = observation_quality(rmse, ideal_max, cfg)
        verify_prerequisites()
        if short.read_json(output / "source_manifest.json") != source_manifest(path):
            raise RuntimeError("O2-D2 source/config changed during execution")
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, encoding="utf-8", errors="replace", check=False)
        result = dict(report, status="O2_D2_TECHNICAL_SMOKE_ONLY" if quick else "O2_D2_COMPLETE_REQUIRES_READ_ONLY_AUDIT",
                      completed_episodes=len(records) * 3, completed_episode_batches=len(records),
                      completed_physical_transitions=context["physical_transitions"], invalid_observations=0,
                      failed_or_truncated_episodes=0, completed_paired_prefix_checks=prefixes,
                      camera_rng_pair_checks=len(records), replay_max_absolute_error=replay_max,
                      completed_policy_forward_calls=policy_calls, completed_scorer_forward_calls=scorer_calls,
                      replay_policy_forward_calls=policy_calls, replay_scorer_forward_calls=scorer_calls,
                      replay_environment_transitions=0, observation_quality=quality, analysis={},
                      raw_camera_frames_saved=0, inverse_crime_limitation=True, real_accuracy_verified=False,
                      real_camera_noise_calibrated=False, realtime_verified=False,
                      elapsed_seconds=time.perf_counter() - started,
                      runtime=dict(python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name(device), deterministic=True, allow_tf32=False,
                                   cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
                                   git_head=git.stdout.strip() if git.returncode == 0 else None),
                      latency_scope="batch-3 CUDA image measurement; decision excludes safety projection; excludes audit/IO/exposure; not real end-to-end latency",
                      next_action="Read-only audit. No automatic full development, training, confirmation or hardware actions.")
        optics.write_json(output / "trajectory_manifest.json", manifest)
        optics.write_json(output / "summary.json", result)
        artifacts = {p.relative_to(output).as_posix(): optics.file_sha256(p) for p in output.rglob("*") if p.is_file()}
        optics.write_json(output / "SUCCESS.json", dict(status=result["status"], summary_sha256=artifacts["summary.json"], artifact_sha256=artifacts))
        return result
    except BaseException as exc:
        optics.write_json(output / "failure.json", dict(status="O2_D2_STOPPED_NO_AUTOMATIC_RETRY", exception=type(exc).__name__,
                         message=str(exc), rejection_type=static.rejection_type(exc) if isinstance(exc, ValueError) else None,
                         traceback=traceback.format_exc(), last_context=context, **cfg["boundary"]))
        raise
