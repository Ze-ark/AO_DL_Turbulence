"""O2-C 有限开发对照：因果人工相机、冻结模型、完整天气配对。

不得调用旧实验入口或载入确认轨迹。真值只用于生成图和独立审计，
噪声后模态误差不会补回控制读数，不按未来安全标签筛候选。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback
from typing import Any, Callable

import torch
import yaml

from observation_bridge import closed_loop as short
from src.rl.r4_observation import R4Interface
from src.rl.r4_trajectory import anchor_delta
from src.runtime import resolve_device
from src.simulation.modes import project_phase_to_modes

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/experiments/observation_bridge_o2_development_v1.yaml"
B_SUMMARY_SHA = "700c353763ea260317f8e76460bbdfa8162a56451ef4ef6fa244cd586d29ef23"
METRICS = ("power", "strehl", "phase_rmse", "violation", "requested_applied_gap_rad",
           "requested_step_abs_rad", "requested_modal_abs_rad", "observation_latency_ms",
           "decision_latency_ms", "camera_negative_clip_fraction", "modal_rmse_rad",
           "representation_fit_rmse_rad")


def read_config(path: str | Path) -> dict[str, Any]:
    target = Path(path) if Path(path).is_absolute() else ROOT / path
    return yaml.safe_load(target.read_text(encoding="utf-8"))


def validate_config(cfg: dict[str, Any]) -> None:
    expected = dict(
        schema="observation_bridge_o2_development_v1", scope="synthetic_holography_development_not_confirmation",
        device="cuda", design_path="configs/experiments/observation_bridge_o2_design_v1.json",
        design_sha256=short.optics.DESIGN_SHA256,
        short_loop_summary="outputs/observation_bridge_o2_closed_loop_quick_v1/summary.json",
        short_loop_summary_sha256=B_SUMMARY_SHA,
        optics_config="configs/experiments/observation_bridge_o2_optics_v1.yaml",
        parent="configs/experiments/s4_r5_policy_training_v1.yaml",
        data=dict(weather_seed_base=8400000, weather_count=8, weather_seed_stride=10, episode_length=200),
        quick=dict(weather_seed_base=8460000, weather_count=1, weather_seed_stride=10, episode_length=28),
        family_ids=["frozen", "boiling", "varying"],
        camera_conditions=[dict(id="noiseless", read_noise_std=0.0), dict(id="small_read_noise", read_noise_std=.001)],
        camera_seed_offset=170000000,
        camera_noise_distribution="additive_gaussian_then_clip_negative_intensity_to_zero",
        camera_draw_including_noiseless=True, batch_size=3, controller_branches=13,
        policy_initializations=[0, 1, 2], scorer_seeds=[7564000, 7564001, 7564002],
        selector_start_step=25, integrator=dict(gain=.15, leak=.1, tracking_gain=.5),
        policy_scale=1.75, candidate_epsilon=.1,
        thresholds=dict(ideal_modal_error_max_rad=.001, noisy_modal_rmse_mean_rad=.01,
                        noisy_modal_rmse_p95_rad=.02, replay_max_absolute_error=1e-6,
                        invalid_observations=0, failed_or_truncated_episodes=0),
        statistics=dict(cluster="complete_weather", stratification="fixed_scorer_fold", bootstrap_seed=8490000,
                        bootstrap_repeats=5000, interval=.95), gain_threshold=None,
        boundary=dict(training_updates=0, independent_confirmation=False, old_confirmation_trajectory_access=False,
                      real_data_access=False, real_slm_actions=False, historical_gate_reclassification=False,
                      automatic_retry=False))
    excluded = {"output_directory", "quick_directory"}
    if set(cfg) != set(expected) | excluded or any(
            json.dumps(cfg.get(k), sort_keys=True) != json.dumps(v, sort_keys=True) for k, v in expected.items()):
        raise ValueError("O2-C fixed development contract changed")


def controller_specs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return short.frozen.controller_specs(dict(policy_initializations=3, scorer_seeds=cfg["scorer_seeds"]))


def budget(spec: dict[str, Any]) -> dict[str, int]:
    batches = spec["weather_count"] * 2 * 13
    steps = spec["episode_length"]
    return dict(episode_batches=batches, complete_episodes=batches * 3,
                batched_steps=batches * steps, physical_transitions=batches * steps * 3,
                policy_forward_calls=spec["weather_count"] * 2 * 12 * steps,
                scorer_forward_calls=spec["weather_count"] * 2 * 9 * max(0, steps - 25),
                camera_batch_draws=batches * (steps + 1), camera_family_frame_draws=batches * (steps + 1) * 3)


def verify_short_loop(cfg: dict[str, Any]) -> dict[str, Any]:
    """只校验 B 已有文件；不调用 B 的 run/preflight，不重跑短闭环。"""
    bcfg = read_config(short.CONFIG)
    short.validate_config(bcfg)
    prerequisite = short.verify_optics_prerequisite(bcfg)
    target = ROOT / cfg["short_loop_summary"]
    if short.optics.file_sha256(target) != B_SUMMARY_SHA:
        raise RuntimeError("O2-B audited summary changed")
    summary = short.read_json(target)
    success = short.read_json(target.parent / "SUCCESS.json")
    expected_files = {p.relative_to(target.parent).as_posix() for p in target.parent.rglob("*")
                      if p.is_file() and p.name != "SUCCESS.json"}
    if ((target.parent / "failure.json").exists() or len(expected_files) != 34
            or set(success["artifact_sha256"]) != expected_files
            or success["summary_sha256"] != B_SUMMARY_SHA
            or success["status"] != "O2_B_SYNTHETIC_CLOSED_LOOP_TECHNICAL_PASS_ONLY"):
        raise RuntimeError("O2-B prerequisite incomplete")
    for name, digest in success["artifact_sha256"].items():
        item = (target.parent / name).resolve()
        if not item.is_relative_to(target.parent.resolve()) or short.optics.file_sha256(item) != digest:
            raise RuntimeError("O2-B successful artifact changed")
    if short.read_json(target.parent / "source_manifest.json") != short.source_manifest(short.CONFIG):
        raise RuntimeError("O2-B executed source/config changed")
    if (summary["complete_episodes"], summary["physical_transitions"], summary["paired_prefix_checks"],
            summary["replay_max_absolute_error"], summary["invalid_observations"]) != (39, 1248, 9, 0, 0):
        raise RuntimeError("O2-B technical prerequisite did not pass")
    return dict(**prerequisite, audited_short_loop_summary_sha256=B_SUMMARY_SHA,
                short_loop_artifacts_checked=34)


def _streams(cfg: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    weather = [spec["weather_seed_base"] + i * spec["weather_seed_stride"] for i in range(spec["weather_count"])]
    return dict(weather_bases=weather, turbulence=[w + i for w in weather for i in range(3)],
                power=[w + 60000000 for w in weather], unused_proxy_sensor=[w + 50000000 for w in weather],
                camera=[w + i + cfg["camera_seed_offset"] for w in weather for i in range(3)],
                fold_assignment={str(w): i % 4 for i, w in enumerate(weather)},
                controllers_share_exogenous_streams=True, camera_conditions_share_standard_normal_draws=True,
                camera_frame_draws_per_family_per_branch=spec["episode_length"] + 1,
                fixed_draw_including_noiseless_and_initial_frame=True, proxy_random_draws=0)


def verify_streams(cfg: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    from scripts import confirm_s4_r5_g2_c1 as c1
    s = short.frozen.source
    factories = [s.stream_manifest, s.d10.d9.stream_manifest, s.d10.d9.d8.stream_manifest,
                 s.d10.d9.d3.stream_manifest, s.d10.d9.d8.d2.g2.stream_manifest]
    historical = [fn(quick=q) for fn in factories for q in (False, True)]
    old_cfg = read_config(c1.CONFIG)
    historical.extend(c1.stream_manifest(old_cfg[k]) for k in ("data", "quick"))
    historical.append(short.stream_manifest(read_config(short.CONFIG)))
    spec = cfg["quick" if quick else "data"]
    manifest = _streams(cfg, spec)
    other = _streams(cfg, cfg["data" if quick else "quick"])
    categories = ("weather_bases", "turbulence", "power", "unused_proxy_sensor", "camera")
    current = set().union(*(set(manifest[k]) for k in categories))
    unique_count = sum(len(manifest[k]) for k in categories if k != "weather_bases")
    if len(set().union(*(set(manifest[k]) for k in categories if k != "weather_bases"))) != unique_count:
        raise RuntimeError("O2-C internal random stream collision")
    if any(current & short._integers(old) for old in historical) or current & short._integers(other):
        raise RuntimeError("O2-C exact historical/quick stream collision")
    manifest.update(historical_manifests_checked=len(historical),
                    disjointness_scope="declared_G2_C1_O2_B_and_other_C_mode_streams_plus_reserved_namespace",
                    technical_unit_namespace=8470000)
    return manifest


def source_manifest(path: str | Path) -> dict[str, Any]:
    target = Path(path) if Path(path).is_absolute() else ROOT / path
    paths = ["observation_bridge/development.py", "scripts/evaluate_observation_bridge_o2_development.py",
             "tests/test_observation_bridge_o2_development.py"]
    return dict(new_source_sha256={p: short.optics.file_sha256(ROOT / p) for p in paths},
                config_sha256=short.optics.file_sha256(target),
                reused_short_loop_sources=short.source_manifest(short.CONFIG))


def preflight(path: str | Path = CONFIG, *, quick: bool = False) -> tuple:
    cfg = read_config(path)
    validate_config(cfg)
    spec = cfg["quick" if quick else "data"]
    output = (ROOT / cfg["quick_directory" if quick else "output_directory"]).resolve()
    output_root = (ROOT / "outputs").resolve()
    if output == output_root or not output.is_relative_to(output_root):
        raise ValueError("O2-C output must be a new child of workspace outputs")
    if output.exists():
        raise FileExistsError(f"preserve existing O2-C output: {output}")
    sources = verify_short_loop(cfg)
    streams = verify_streams(cfg, quick=quick)
    short.configure_runtime()
    device = resolve_device("cuda")
    if device.type != "cuda":
        raise RuntimeError("O2-C requires CUDA; no CPU fallback")
    parent = read_config(cfg["parent"])
    profile = short.nominal_profile(parent)
    policies, scorers, models = short.load_assets(device, parent)
    if any(w in row.get("train_weather", []) + row.get("held_out_weather", [])
           for row in models for w in streams["weather_bases"]):
        raise RuntimeError("O2-C weather was used by frozen scorer")
    report = dict(status="O2_C_READY_FOR_TECHNICAL_SMOKE" if quick else "O2_C_READY_FOR_USER_IDE",
                  quick=quick, **cfg["boundary"], **budget(spec), episode_length=spec["episode_length"],
                  weather_count=spec["weather_count"], device=str(device), loaded_policies=len(policies),
                  loaded_scorers=len(scorers), stream_manifest=streams, frozen_sources=sources,
                  synthetic_nominal_profile=profile.as_record(), output_directory=str(output),
                  preflight_model_forward_calls=0, preflight_environment_transitions=0,
                  gain_threshold=None, scientific_gain_analysis=not quick,
                  fold_routing="weather_index_modulo_4_fixed_before_observation")
    return cfg, spec, output, device, parent, policies, scorers, models, report


class PairedReadNoise:
    """每家族独立 GPU 随机流；每帧总会抽样，强度负值裁剪被记录。"""

    def __init__(self, device: torch.device, seeds: list[int], noise_std: float):
        if device.type != "cuda" or len(seeds) != 3 or len(set(seeds)) != 3 or noise_std not in (0.0, .001):
            raise ValueError("O2-C fixed CUDA camera noise contract required")
        resolved = resolve_device(str(device))
        self.device = resolve_device(f"cuda:{torch.cuda.current_device() if resolved.index is None else resolved.index}")
        self.noise_std = noise_std
        self.generators = [torch.Generator(device=self.device).manual_seed(seed) for seed in seeds]
        self.frames = 0

    @torch.no_grad()
    def apply(self, intensity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (intensity.device != self.device or intensity.shape != (3, 512, 512)
                or intensity.dtype != torch.float32 or not bool(torch.isfinite(intensity).all())
                or bool((intensity < 0).any())):
            raise ValueError("invalid current CUDA camera intensity")
        samples = torch.stack([torch.randn((512, 512), device=self.device, generator=g,
                                          dtype=intensity.dtype) for g in self.generators])
        noisy = intensity + self.noise_std * samples
        fraction = (noisy < 0).float().mean((-2, -1))
        self.frames += 1
        return noisy.clamp_min(0), fraction

    def final_state_sha256(self) -> list[str]:
        # 仅回合结束记录随机流身份；测量张量不会去 CPU 处理后再回 GPU。
        return [hashlib.sha256(g.get_state().cpu().numpy().tobytes()).hexdigest() for g in self.generators]


class DevelopmentPort(short.HolographicEnvironmentPort):
    def __init__(self, env, sensor, bridge, noise: PairedReadNoise, max_modal_error: float):
        super().__init__(env, sensor, bridge, max_modal_error)
        self.noise = noise

    @torch.no_grad()
    def observe(self) -> tuple[short.SensorReadout, dict[str, torch.Tensor]]:
        if self.env.slm.current_phase is None:
            raise RuntimeError("unknown actuator initial state")
        phase = self.env._current_turbulence_window() + self.env.slm.current_phase
        field = torch.polar(self.env.pupil.to(phase.dtype).expand_as(phase), phase)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        intensity, clipped = self.noise.apply(self.sensor.render(field))
        measured = self.bridge.measure(self.sensor.reconstruct(intensity))
        end.record()
        end.synchronize()
        latency_ms = start.elapsed_time(end)
        # 先完成图像读数，再独立查真值；噪声误差只记录，不补回、不逐帧筛选。
        target, true_jump = short.optics.audit_known_phase(phase.double(), self.bridge)
        if true_jump > self.bridge.tolerances.max_neighbor_jump_rad:
            raise RuntimeError("O2-C known spatial sampling precondition failed")
        difference = measured.residual_rad.double() - target
        error = difference.abs().amax(-1)
        if self.noise.noise_std == 0 and float(error.max()) > self.max_modal_error:
            raise RuntimeError("O2-C ideal modal observation tolerance failed")
        audit = dict(joint_target_rad=target, legacy_projection_rad=project_phase_to_modes(phase, self.env.basis, self.env.pupil),
                     modal_error_rad=error, modal_rmse_rad=difference.square().mean(-1).sqrt(),
                     fit_rmse_rad=measured.fit_rmse_rad, max_wrapped_neighbor_jump_rad=measured.max_neighbor_jump_rad,
                     batch_true_neighbor_jump_rad=error.new_full(error.shape, true_jump),
                     camera_negative_clip_fraction=clipped,
                     camera_draw_index=torch.full((3,), self.noise.frames - 1, dtype=torch.int64, device=self.bridge.device),
                     observation_latency_ms=error.new_full(error.shape, latency_ms))
        return short.SensorReadout(measured.residual_rad, self.env.step_count), audit


@torch.no_grad()
def rollout(cfg: dict[str, Any], spec: dict[str, Any], parent: dict, branch: dict, *, seed: int,
            weather_index: int, camera: dict, sensor, bridge, policy, selector,
            progress: Callable[[dict], None], context: dict) -> tuple[dict, dict, list[dict], dict]:
    context.update(controller=branch["controller"], weather_seed=seed, camera_condition=camera["id"], action_step=None)
    env = short.make_environment(parent, bridge.basis, seed, spec["episode_length"])
    camera_seeds = [seed + i + cfg["camera_seed_offset"] for i in range(3)]
    noise = PairedReadNoise(bridge.device, camera_seeds, camera["read_noise_std"])
    port = DevelopmentPort(env, sensor, bridge, noise, cfg["thresholds"]["ideal_modal_error_max_rad"])
    readout, initial_audit = port.reset(seed)
    interface = R4Interface()
    interface.reset(readout.residual, episode_id=f"o2-c-{seed}-{camera['id']}-{branch['controller']}")
    visible = {k: [] for k in ("history", "valid", "original", "selected", "choice", "prediction", "requested_delta",
                              "requested_modal", "residual", "measured_power", "next_clock", "power_action_step", "power_arrival_step")}
    visible["residual"].append(readout.residual.clone())
    audits = {k: [v.clone()] for k, v in initial_audit.items()}
    def on_transition() -> None:
        context["physical_transitions"] += 3
    for step in range(spec["episode_length"]):
        context["action_step"] = step
        view = interface.snapshot()
        if readout.observation_step != step or view.observation_step != step:
            raise RuntimeError("O2-C current observation clock mismatch")
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        original, selected, choice, prediction = short.choose_command(view.features, view.valid, policy, selector, step=step, cfg=cfg)
        end.record(); end.synchronize()
        decision_ms = start.elapsed_time(end)
        action = interface.issue(anchor_delta(view.features[:, -1], cfg["integrator"]), selected, step=step)
        readout, power, audit = port.step(action.requested_delta_rad, step, on_transition)
        transition = interface.observe_next(readout.residual, step=readout.observation_step, power=power)
        if not bool(transition.action_power_valid.all()):
            raise RuntimeError("O2-C causal action power missing")
        fields = dict(history=view.features, valid=view.valid, original=original, selected=selected, choice=choice,
                      prediction=prediction, requested_delta=action.requested_delta_rad, requested_modal=action.requested_modal_rad,
                      measured_power=power.value, next_clock=transition.next_history.features[:, -1, 75:79],
                      power_action_step=torch.tensor(power.action_step, device=bridge.device),
                      power_arrival_step=torch.tensor(power.arrival_observation_step, device=bridge.device))
        for key, value in fields.items():
            visible[key].append(value.detach().clone())
        visible["residual"].append(readout.residual.clone())
        audit["decision_latency_ms"] = readout.residual.new_full((3,), decision_ms)
        for key, value in audit.items():
            audits.setdefault(key, []).append(value.detach().clone())
        progress(dict(weather_seed=seed, camera_condition=camera["id"], controller=branch["controller"],
                      observation_step=step + 1, physical_transitions=context["physical_transitions"],
                      modal_error_max_rad=float(audit["modal_error_rad"].max())))
    trace, audit_trace = short._stack(visible), short._stack(audits)
    if noise.frames != spec["episode_length"] + 1:
        raise RuntimeError("O2-C fixed camera frame draw budget mismatch")
    for store in (trace, audit_trace):
        if any(value.is_floating_point() and not bool(torch.isfinite(value).all()) for value in store.values()):
            raise RuntimeError("O2-C nonfinite saved trajectory")
    values = dict(power=audit_trace["action_power"], strehl=audit_trace["action_strehl"],
                  phase_rmse=audit_trace["action_phase_rmse"], violation=audit_trace["violation"],
                  requested_applied_gap_rad=(audit_trace["requested_modal"] - audit_trace["applied_modal"]).abs().mean(-1),
                  requested_step_abs_rad=trace["requested_delta"].abs().mean(-1),
                  requested_modal_abs_rad=trace["requested_modal"].abs().mean(-1),
                  observation_latency_ms=audit_trace["observation_latency_ms"], decision_latency_ms=audit_trace["decision_latency_ms"],
                  camera_negative_clip_fraction=audit_trace["camera_negative_clip_fraction"], modal_rmse_rad=audit_trace["modal_rmse_rad"],
                  representation_fit_rmse_rad=audit_trace["fit_rmse_rad"])
    means = {k: v.double().mean(0) for k, v in values.items()}
    rows = []
    for family_index, family in enumerate(cfg["family_ids"]):
        rows.append(dict(weather_seed=seed, weather_index=weather_index, family=family, family_index=family_index,
                         camera_condition=camera["id"], **branch, episode_length=spec["episode_length"],
                         scorer_fold=weather_index % 4 if selector is not None else None,
                         turbulence_stream_seed=seed + family_index, power_stream_seed=seed + 60000000,
                         camera_stream_seed=camera_seeds[family_index], camera_frames=noise.frames,
                         failed=False, truncated=False, **{k: float(v[family_index]) for k, v in means.items()},
                         selected_nonoriginal_fraction=float(trace["choice"][:, family_index].ne(0).float().mean())))
    record = dict(controller=branch["controller"], weather_seed=seed, weather_index=weather_index,
                  camera_condition=camera["id"], complete_episodes=3, physical_transitions=spec["episode_length"] * 3,
                  scorer_fold=weather_index % 4 if selector is not None else None,
                  camera_frames_per_family=noise.frames, camera_final_rng_sha256=noise.final_state_sha256(),
                  modal_error_max_rad=float(audit_trace["modal_error_rad"].max()),
                  policy_forward_calls=spec["episode_length"] if policy is not None else 0,
                  scorer_forward_calls=max(0, spec["episode_length"] - cfg["selector_start_step"]) if selector is not None else 0)
    return trace, audit_trace, rows, record


def stratified_draws(weather_count: int, *, repeats: int, seed: int, device: torch.device) -> torch.Tensor:
    """按固定模型折重采样整份天气；绝不重采样帧或单个家族。"""
    if weather_count != 8 or type(repeats) is not int or repeats < 1:
        raise ValueError("O2-C descriptive bootstrap requires eight weather clusters")
    generator = torch.Generator(device=device).manual_seed(seed)
    draws = []
    for fold in range(4):
        group = torch.arange(fold, weather_count, 4, device=device)
        indexes = torch.randint(len(group), (repeats, len(group)), generator=generator, device=device)
        draws.append(group[indexes])
    return torch.cat(draws, dim=1)


def paired_interval(difference: torch.Tensor, draws: torch.Tensor,
                    denominator: torch.Tensor | None = None) -> list[float]:
    if difference.shape != (8,) or draws.ndim != 2 or draws.shape[1] != 8:
        raise ValueError("complete-weather paired statistic shape mismatch")
    values = difference[draws].mean(1)
    if denominator is not None:
        if denominator.shape != (8,) or bool((denominator <= 0).any()):
            raise ValueError("nonpositive integrator bucket power")
        values = values / denominator[draws].mean(1)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("nonfinite weather bootstrap")
    return [float(v) for v in torch.quantile(values.double(), values.new_tensor([.025, .975], dtype=torch.float64))]


def validate_rows(rows: list[dict], cfg: dict[str, Any], spec: dict[str, Any]) -> dict[tuple, dict]:
    weather = _streams(cfg, spec)["weather_bases"]
    branches = {b["controller"]: b for b in controller_specs(cfg)}
    expected = {(c["id"], w, b, f) for c in cfg["camera_conditions"] for w in weather
                for b in branches for f in cfg["family_ids"]}
    mapping = {}
    for row in rows:
        key = (row["camera_condition"], row["weather_seed"], row["controller"], row["family"])
        if key in mapping or key not in expected:
            raise ValueError("O2-C duplicated or unexpected complete episode")
        wi, fi = weather.index(row["weather_seed"]), cfg["family_ids"].index(row["family"])
        branch = branches[row["controller"]]
        if (row["weather_index"] != wi or row["family_index"] != fi or row["episode_length"] != spec["episode_length"]
                or row["camera_frames"] != spec["episode_length"] + 1
                or row["scorer_fold"] != (wi % 4 if branch["scorer_seed"] is not None else None)
                or any(row[k] != v for k, v in branch.items())
                or row["turbulence_stream_seed"] != row["weather_seed"] + fi
                or row["camera_stream_seed"] != row["weather_seed"] + fi + cfg["camera_seed_offset"]
                or row["power_stream_seed"] != row["weather_seed"] + 60000000
                or row["failed"] is not False or row["truncated"] is not False
                or any(type(row[k]) not in (float, int) or not math.isfinite(row[k]) for k in METRICS)
                or any(not 0 <= row[k] <= 1 for k in ("violation", "camera_negative_clip_fraction", "selected_nonoriginal_fraction"))):
            raise ValueError("O2-C episode identity, seed, metric or clock mismatch")
        mapping[key] = row
    if set(mapping) != expected:
        raise ValueError("O2-C incomplete episode grid; no silent exclusion allowed")
    return mapping


@torch.no_grad()
def summarize(rows: list[dict], cfg: dict[str, Any], spec: dict[str, Any], *,
              device: torch.device, quick: bool) -> dict[str, Any]:
    mapping = validate_rows(rows, cfg, spec)
    if quick:
        return {}  # 技术冒烟不输出科学收益、区间或排名。
    if device.type != "cuda":
        raise ValueError("O2-C scientific development statistics require CUDA")
    cameras = [c["id"] for c in cfg["camera_conditions"]]
    ids = [b["controller"] for b in controller_specs(cfg)]
    weather = _streams(cfg, spec)["weather_bases"]
    families = cfg["family_ids"]
    values = {k: torch.tensor([[[[mapping[(c, w, b, f)][k] for f in families] for w in weather]
                               for b in ids] for c in cameras], dtype=torch.float64, device=device) for k in METRICS}
    if any(not bool(torch.isfinite(value).all()) for value in values.values()):
        raise ValueError("nonfinite O2-C development statistics")
    originals = [ids.index(f"original_{m}") for m in range(3)]
    currents = [ids.index(f"current_{m}_{s}") for m in range(3) for s in cfg["scorer_seeds"]]
    draws = stratified_draws(8, repeats=cfg["statistics"]["bootstrap_repeats"],
                             seed=cfg["statistics"]["bootstrap_seed"], device=device)
    cells, group_table, power_clusters = {}, [], {}
    for ci, camera in enumerate(cameras):
        means = {k: dict(integrator=value[ci, 0], original=value[ci, originals].mean(0),
                         current=value[ci, currents].mean(0)) for k, value in values.items()}
        power = {name: value.mean(-1) for name, value in means["power"].items()}
        power_clusters[camera] = power
        base = power["integrator"]
        if bool((base <= 0).any()):
            raise ValueError("nonpositive complete-weather integrator power")
        comparisons = {}
        for left, right in (("original", "integrator"), ("current", "integrator"), ("current", "original")):
            difference = power[left] - power[right]
            part = dict(mean_power_difference=float(difference.mean()),
                        descriptive_ci95=paired_interval(difference, draws),
                        positive_weather_count=int((difference > 0).sum()),
                        per_weather_power_difference=difference.tolist())
            if right == "integrator":
                part.update(relative_gain=float(difference.mean() / base.mean()),
                            relative_gain_descriptive_ci95=paired_interval(difference, draws, base))
            comparisons[f"{left}_vs_{right}"] = part
        cells[camera] = dict(weather_clusters=8, comparisons=comparisons,
                            method_means={name: {k: float(value[name].mean()) for k, value in means.items()}
                                          for name in ("integrator", "original", "current")},
                            per_controller_means={b: {k: float(value[ci, bi].mean()) for k, value in values.items()}
                                                  for bi, b in enumerate(ids)},
                            metric_deltas={k: dict(current_vs_integrator=float((v["current"] - v["integrator"]).mean()),
                                                  current_vs_original=float((v["current"] - v["original"]).mean()))
                                           for k, v in means.items()})
        for fi, family in enumerate(families):
            base_family = means["power"]["integrator"][:, fi]
            group_table.append(dict(camera_condition=camera, family=family, weather_clusters=8,
                                    current_relative_gain_vs_integrator=float((means["power"]["current"][:, fi] - base_family).mean() / base_family.mean()),
                                    current_minus_original_power=float((means["power"]["current"][:, fi] - means["power"]["original"][:, fi]).mean())))
    noise_effect = {}
    for method in ("integrator", "original", "current"):
        difference = power_clusters["small_read_noise"][method] - power_clusters["noiseless"][method]
        noise_effect[method] = dict(noisy_minus_noiseless_power=float(difference.mean()),
                                    descriptive_ci95=paired_interval(difference, draws))
    gain_change = ((power_clusters["small_read_noise"]["current"] - power_clusters["small_read_noise"]["integrator"])
                   - (power_clusters["noiseless"]["current"] - power_clusters["noiseless"]["integrator"]))
    return dict(status="DEVELOPMENT_DESCRIPTIVE_NOT_CONFIRMATION", cells=cells, family_table=group_table,
                camera_noise_effect=noise_effect,
                current_increment_change_noisy_minus_noiseless=dict(mean_power_difference=float(gain_change.mean()),
                                                                    descriptive_ci95=paired_interval(gain_change, draws)),
                interval_scope="small_development_sample_conditional_on_frozen_models_and_fixed_fold_routing",
                independent_weather_clusters=8, frame_independence_assumed=False,
                seed_aggregation="average_of_separate_closed_loop_trajectories_not_ensemble_deployment",
                training_uncertainty_covered=False, statistical_power_guaranteed=False,
                multiple_comparison_significance_claims=False, gain_threshold=None,
                independent_confirmation=False, historical_gate_reclassification=False)


def observation_quality(ideal_max: float, noisy_rmse: list[torch.Tensor], cfg: dict[str, Any]) -> dict[str, Any]:
    values = torch.cat([v.flatten().double() for v in noisy_rmse])
    if not bool(torch.isfinite(values).all()):
        raise ValueError("nonfinite observation audit")
    mean, p95 = float(values.mean()), float(torch.quantile(values, .95))
    targets = cfg["thresholds"]
    checks = dict(ideal_modal_error=ideal_max <= targets["ideal_modal_error_max_rad"],
                  noisy_mean_modal_rmse=mean <= targets["noisy_modal_rmse_mean_rad"],
                  noisy_p95_modal_rmse=p95 <= targets["noisy_modal_rmse_p95_rad"])
    return dict(ideal_modal_error_max_rad=ideal_max, noisy_modal_rmse_mean_rad=mean,
                noisy_modal_rmse_p95_rad=p95, noisy_family_observations=int(values.numel()),
                targets_met=all(checks.values()), checks=checks,
                scope="technical_observation_targets_not_compensation_gain_gate",
                noisy_error_used_to_replace_measurements=False)


@torch.no_grad()
def run(path: str | Path = CONFIG, *, quick: bool = False, preflight_only: bool = False) -> dict[str, Any]:
    cfg, spec, output, device, parent, policies, scorers, models, report = preflight(path, quick=quick)
    if preflight_only:
        return report
    output.mkdir(parents=True, exist_ok=False)
    context: dict[str, Any] = dict(physical_transitions=0, completed_episode_batches=0,
                                  weather_seed=None, camera_condition=None, controller=None, action_step=None)
    started = time.perf_counter()
    try:
        for directory in ("trajectories", "audit"):
            (output / directory).mkdir()
        for name, value in (("effective_config.json", cfg), ("preflight.json", report), ("model_manifest.json", models),
                            ("stream_manifest.json", report["stream_manifest"]), ("source_manifest.json", source_manifest(path))):
            short.optics.write_json(output / name, value)
        sensor, bridge = short.optics.make_components(read_config(cfg["optics_config"]), device)
        rows, batch_records, trajectories = [], [], []
        noisy_rmse: list[torch.Tensor] = []
        ideal_max = replay_max = 0.0
        prefixes = policy_calls = scorer_calls = progress_count = 0
        rng_states: dict[int, list[str]] = {}
        def progress(row: dict) -> None:
            nonlocal progress_count
            progress_count += 1
            elapsed = time.perf_counter() - started
            count, total = row["physical_transitions"], report["physical_transitions"]
            row.update(total_physical_transitions=total, elapsed_seconds=elapsed,
                       eta_seconds=elapsed / count * (total - count),
                       transitions_per_second=count / max(elapsed, 1e-9),
                       cuda_allocated_gb=torch.cuda.memory_allocated(device) / 2**30)
            with (output / "progress.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            if row["observation_step"] % 10 == 0 or row["observation_step"] == spec["episode_length"]:
                print(f"O2-C {'技术冒烟' if quick else '开发对照'} {count}/{total} | 天气{row['weather_seed']} "
                      f"{row['camera_condition']} {row['controller']} {row['observation_step']}/{spec['episode_length']}帧 | "
                      f"速度={row['transitions_per_second']:.1f}转移/s | 剩余={row['eta_seconds']:.1f}s | "
                      f"显存={row['cuda_allocated_gb']:.2f}GB", flush=True)
        for wi, seed in enumerate(report["stream_manifest"]["weather_bases"]):
            for camera in cfg["camera_conditions"]:
                originals = {}
                for branch in controller_specs(cfg):
                    policy = policies.get(branch["member"])
                    selector = None if branch["scorer_seed"] is None else scorers[wi % 4, branch["scorer_seed"]]
                    trace, audit, part_rows, record = rollout(cfg, spec, parent, branch, seed=seed, weather_index=wi,
                        camera=camera, sensor=sensor, bridge=bridge, policy=policy, selector=selector, progress=progress, context=context)
                    name = f"weather_{wi:02d}_{camera['id']}_{branch['controller']}.pt"
                    for directory, value in (("trajectories", trace), ("audit", audit)):
                        torch.save({k: v.cpu() for k, v in value.items()}, output / directory / name)
                    loop_cfg = {**cfg, "episode_length": spec["episode_length"]}
                    saved = torch.load(output / "trajectories" / name, map_location=device, weights_only=True)
                    record["replay"] = short.replay_visible(saved, loop_cfg, policy, selector)
                    replay_max = max(replay_max, record["replay"]["max_absolute_error"])
                    if selector is None and branch["member"] is not None:
                        originals[branch["member"]] = trace
                    if selector is not None:
                        short.require_prefix(originals[branch["member"]], trace, cfg["selector_start_step"])
                        record["paired_prefix_exact"] = True
                        prefixes += 1
                    final_rng = record["camera_final_rng_sha256"]
                    if seed in rng_states and rng_states[seed] != final_rng:
                        raise RuntimeError("O2-C camera exogenous draw count differs across controllers/conditions")
                    rng_states[seed] = final_rng
                    if camera["id"] == "noiseless":
                        ideal_max = max(ideal_max, record["modal_error_max_rad"])
                    else:
                        noisy_rmse.append(audit["modal_rmse_rad"].detach().clone())
                    for row in part_rows:
                        row["trajectory_file"] = name
                    rows.extend(part_rows); batch_records.append(record)
                    trajectories.append(dict(file=name, weather_seed=seed, camera_condition=camera["id"], **branch,
                                             scorer_fold=record["scorer_fold"],
                                             visible_sha256=short.optics.file_sha256(output / "trajectories" / name),
                                             audit_sha256=short.optics.file_sha256(output / "audit" / name)))
                    for filename, items in (("records.jsonl", part_rows), ("batch_records.jsonl", [record])):
                        with (output / filename).open("a", encoding="utf-8") as handle:
                            for row in items:
                                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    context["completed_episode_batches"] += 1
                    policy_calls += record["policy_forward_calls"]; scorer_calls += record["scorer_forward_calls"]
        if (context["physical_transitions"] != report["physical_transitions"] or len(rows) != report["complete_episodes"]
                or len(batch_records) != report["episode_batches"] or progress_count != report["batched_steps"]
                or policy_calls != report["policy_forward_calls"] or scorer_calls != report["scorer_forward_calls"]
                or prefixes != spec["weather_count"] * 2 * 9):
            raise RuntimeError("O2-C complete execution budget mismatch")
        quality = observation_quality(ideal_max, noisy_rmse, cfg)
        analysis = summarize(rows, cfg, spec, device=device, quick=quick)
        verify_short_loop(cfg)
        if short.read_json(output / "source_manifest.json") != source_manifest(path):
            raise RuntimeError("O2-C source/config changed during execution")
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, encoding="utf-8", errors="replace", check=False)
        result = dict(report, status="O2_C_TECHNICAL_SMOKE_ONLY" if quick else "O2_C_DEVELOPMENT_COMPLETE_REQUIRES_READ_ONLY_AUDIT",
                      completed_episodes=len(rows), completed_episode_batches=len(batch_records),
                      completed_physical_transitions=context["physical_transitions"], invalid_observations=0,
                      failed_or_truncated_episodes=0, paired_prefix_checks=prefixes, camera_rng_pair_checks=len(batch_records),
                      completed_policy_forward_calls=policy_calls, completed_scorer_forward_calls=scorer_calls,
                      replay_policy_forward_calls=policy_calls, replay_scorer_forward_calls=scorer_calls,
                      replay_environment_transitions=0, replay_max_absolute_error=replay_max,
                      observation_quality=quality, analysis=analysis, records=rows, raw_camera_frames_saved=0,
                      inverse_crime_limitation=True, real_accuracy_verified=False, realtime_verified=False,
                      elapsed_seconds=time.perf_counter() - started,
                      runtime=dict(python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name(device), deterministic=True, allow_tf32=False,
                                   cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
                                   batch_layout="three_families_one_nominal_profile", git_head=git.stdout.strip() if git.returncode==0 else None),
                      latency_scope="CUDA batch-3 optical generation/read-noise/reconstruction/modal fit; decision excludes safety projection; both exclude audit_truth_fit_and_IO; not camera exposure or end-to-end real-time latency",
                      next_action="Read-only audit; no automatic rerun, training, confirmation or hardware actions.")
        short.optics.write_json(output / "trajectory_manifest.json", trajectories)
        short.optics.write_json(output / "summary.json", result)
        artifacts = {p.relative_to(output).as_posix(): short.optics.file_sha256(p) for p in output.rglob("*") if p.is_file()}
        short.optics.write_json(output / "SUCCESS.json", dict(status=result["status"], summary_sha256=artifacts["summary.json"], artifact_sha256=artifacts))
        return result
    except BaseException as exc:
        short.optics.write_json(output / "failure.json", dict(status="O2_C_STOPPED_NO_AUTOMATIC_RETRY", exception=type(exc).__name__,
                                 message=str(exc), traceback=traceback.format_exc(), last_context=context,
                                 incomplete_batch_size=(3 if context["controller"] is not None
                                     and context["completed_episode_batches"] < report["episode_batches"] else 0),
                                 **cfg["boundary"]))
        raise
