"""O2-D7：固定四格相机噪声、冻结控制器、严格因果的短闭环技术检查。

只复用封存模块的纯函数、模型加载与物理接口，不调用旧 run/preflight。
相机先产生光子计数，再加独立读出噪声；两种尺度均未做真实相机标定。
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
import traceback
from typing import Any, Callable

import torch

from observation_bridge import photon_development as completed
from src.rl.r4_observation import R4Interface
from src.rl.r4_trajectory import anchor_delta
from src.runtime import resolve_device

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/experiments/observation_bridge_o2_d7_combined_noise_closed_loop_v1.yaml"
technical = completed.technical
short, optics, prior, photon = technical.short, technical.optics, technical.prior, technical.photon
D6_PINS = {
    "observation_bridge/photon_development.py": "95163b6c8ff7aa79878ccf50a091d804b5b1d15ea9d178ad7a8d8c46929e86b8",
    "scripts/evaluate_observation_bridge_o2_photon_development.py": "63f6c58c145a9f89bce384da609749f9fa23d08d2c1b87ad2f5ab636e21e343c",
    "tests/test_observation_bridge_o2_photon_development.py": "6aea63e52fbbb0f6ea3cdc9c15d024baf3f2ad79680430dac051560dbef4bf62",
    completed.CONFIG: "c00b59275d4182d196bb7b358ea3f192ac9dc3273880e6212ec6185247fad051",
}
D6_OUTPUTS = {
    "outputs/observation_bridge_o2_d6_photon_development_v1":
        ("4f04173f68700678af309880fb5d3bc0cbae5b933bc9d7c92f3b309c76a8fdcb", 947, False),
    "outputs/observation_bridge_o2_d6_photon_development_v1_quick":
        ("d1d9ffa70f1e61053be32e8b9183b8fb25e89eb1b27c0bf7fece5a5c7a07929b", 88, True),
}


def read_config(path: str | Path = CONFIG) -> dict[str, Any]:
    return prior.read_config(path)


def validate_config(cfg: dict[str, Any]) -> None:
    # 用固定指纹的 D5 合约继承控制与技术目标，不更改任何封存源码。
    if optics.file_sha256(technical.ROOT / technical.CONFIG) != completed.D5_PINS[technical.CONFIG]:
        raise RuntimeError("O2-D7 frozen D5 configuration changed")
    expected = technical.read_config()
    for key in ("output_directory", "quick_directory", "camera_seed_offset", "camera_level_seed_stride",
                "noise_distribution", "noise_units", "read_noise_std", "shared_draw_across_levels"):
        expected.pop(key)
    expected.update(
        schema="observation_bridge_o2_d7_combined_noise_closed_loop_v1",
        scope="synthetic_combined_noise_short_loop_technical_only",
        data=dict(weather_seed_base=9200000, weather_count=4, weather_seed_stride=10000,
                  episode_length=32, camera_ids=["noiseless", "read_only", "photon_only", "combined"]),
        quick=dict(weather_seed_base=9360000, weather_count=1, weather_seed_stride=10000,
                   episode_length=28, camera_ids=["noiseless", "combined"]),
        camera_conditions=[dict(id="noiseless", counts_per_intensity_unit=None, read_noise_std=0.0),
                           dict(id="read_only", counts_per_intensity_unit=None, read_noise_std=1.0),
                           dict(id="photon_only", counts_per_intensity_unit=10.0, read_noise_std=0.0),
                           dict(id="combined", counts_per_intensity_unit=10.0, read_noise_std=1.0)],
        photon_seed_offset=200000000, read_seed_offset=210000000,
        noise_order="poisson_counts_divided_by_k_then_add_gaussian_then_clip_negative_to_zero",
        noise_units="arbitrary_synthetic_intensity_not_calibrated_camera_electrons",
        component_seed_identity_shared_across_conditions=True, noise_components_independent=True)
    if (not isinstance(cfg, dict) or set(cfg) != set(expected) | {"output_directory", "quick_directory"}
            or any(json.dumps(cfg.get(k), sort_keys=True, allow_nan=False) != json.dumps(v, sort_keys=True)
                   for k, v in expected.items())):
        raise ValueError("O2-D7 fixed four-cell technical contract changed")
    if (any(not isinstance(cfg[k], str) or not cfg[k] for k in ("output_directory", "quick_directory"))
            or (ROOT / cfg["output_directory"]).resolve() == (ROOT / cfg["quick_directory"]).resolve()):
        raise ValueError("O2-D7 distinct output paths required")


def budget(cfg: dict[str, Any], spec: dict[str, Any]) -> dict[str, int]:
    cameras = [c for c in cfg["camera_conditions"] if c["id"] in spec["camera_ids"]]
    groups, steps = spec["weather_count"] * len(cameras), spec["episode_length"]
    batches = groups * 13
    component_frames = spec["weather_count"] * 13 * (steps + 1) * 3
    return dict(episode_batches=batches, complete_episodes=batches * 3, batched_steps=batches * steps,
                physical_transitions=batches * steps * 3, policy_forward_calls=groups * 12 * steps,
                scorer_forward_calls=groups * 9 * max(0, steps - 25), paired_prefix_checks=groups * 9,
                camera_batch_frames=batches * (steps + 1), camera_family_frames=batches * (steps + 1) * 3,
                poisson_draws=component_frames * sum(c["counts_per_intensity_unit"] is not None for c in cameras),
                read_noise_draws=component_frames * sum(c["read_noise_std"] > 0 for c in cameras))


def component_seed(cfg: dict[str, Any], weather: int, camera_id: str, frame: int, family: int,
                   component: str) -> int | None:
    cameras = {c["id"]: c for c in cfg["camera_conditions"]}
    if (camera_id not in cameras or type(weather) is not int or weather < 0
            or type(frame) is not int or frame < 0 or type(family) is not int or not 0 <= family < 3
            or component not in ("photon", "read")):
        raise ValueError("O2-D7 unknown camera/component/frame/family seed identity")
    camera = cameras[camera_id]
    active = camera["counts_per_intensity_unit"] is not None if component == "photon" else camera["read_noise_std"] > 0
    return (weather + cfg[f"{component}_seed_offset"] + frame * cfg["camera_frame_seed_stride"] + family
            if active else None)


def _streams(cfg: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    weather = [spec["weather_seed_base"] + i * spec["weather_seed_stride"] for i in range(spec["weather_count"])]
    cameras = [c for c in cfg["camera_conditions"] if c["id"] in spec["camera_ids"]]
    streams: dict[str, Any] = dict(weather_bases=weather, turbulence=[w + f for w in weather for f in range(3)],
                                 power=[w + 60000000 for w in weather], unused_proxy_sensor=[w + 50000000 for w in weather])
    # 条件间共享同一噪声分量身份，种子清单只列每个独立分量一次，不误报重复。
    for component in ("photon", "read"):
        active = next((c for c in cameras if (c["counts_per_intensity_unit"] is not None
                       if component == "photon" else c["read_noise_std"] > 0)), None)
        streams[f"{component}_camera"] = ([] if active is None else
            [component_seed(cfg, w, active["id"], t, f, component) for w in weather
             for t in range(spec["episode_length"] + 1) for f in range(3)])
    return streams


def stream_manifest(cfg: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    from scripts import confirm_s4_r5_g2_c1 as c1
    s = short.frozen.source
    factories = [s.stream_manifest, s.d10.d9.stream_manifest, s.d10.d9.d8.stream_manifest,
                 s.d10.d9.d3.stream_manifest, s.d10.d9.d8.d2.g2.stream_manifest]
    history = [fn(quick=q) for fn in factories for q in (False, True)]
    old = read_config(c1.CONFIG)
    history.extend(c1.stream_manifest(old[k]) for k in ("data", "quick"))
    history.append(short.stream_manifest(read_config(short.CONFIG)))
    old = prior.read_config(prior.CONFIG)
    history.extend(prior._streams(old, old[k]) for k in ("data", "quick"))
    for module in (photon.static, technical.technical, photon.completed, photon, technical, completed):
        old = module.read_config()
        history.extend(module.stream_manifest(old, quick=q) for q in (False, True))
    spec = cfg["quick" if quick else "data"]
    manifest, other = _streams(cfg, spec), _streams(cfg, cfg["data" if quick else "quick"])
    independent = [v for k in ("turbulence", "power", "unused_proxy_sensor", "photon_camera", "read_camera")
                   for v in manifest[k]]
    if len(independent) != len(set(independent)):
        raise RuntimeError("O2-D7 internal random stream collision")
    current = set(independent) | set(manifest["weather_bases"])
    units = (8470000, 8520000, 8670000, 8770000, 8870000, 8970000, 9170000, 9370000)
    reserved = set().union(*(set(range(n, n + 10000)) for n in units))
    reserved |= {n + offset for b in units for n in range(b, b + 10000)
                 for offset in (50000000, 60000000, 160000000, 170000000, 180000000,
                                181000000, 182000000, 200000000, 210000000)}
    reserved |= {9180000, prior.read_config(prior.CONFIG)["statistics"]["bootstrap_seed"],
                 photon.completed.read_config()["statistics"]["bootstrap_seed"]}
    if (current & (short._integers(other) | reserved)
            or any(current & short._integers(h) for h in history)):
        raise RuntimeError("O2-D7 historical/quick/unit/statistics seed collision")
    return dict(manifest, historical_manifests_checked=len(history), technical_unit_namespace=9370000,
                fold_assignment={str(w): i % 4 for i, w in enumerate(manifest["weather_bases"])},
                seed_formula="weather+component_offset+observation_frame*10+family_index",
                component_seed_identity_shared_across_conditions=True, noise_components_independent=True,
                seed_identity_shared_across_controllers=True, zero_noise_draws=0,
                equal_poisson_rng_end_state_not_assumed=True, frame_randomness_reseeded=True,
                disjointness_scope="declared_G2_C1_B_C_D1_D2_D3_D4_D5_D6_other_D7_mode_and_reserved_units")


def verify_prerequisites() -> dict[str, Any]:
    for name, digest in D6_PINS.items():
        if optics.file_sha256(ROOT / name) != digest:
            raise RuntimeError(f"O2-D7 frozen D6 source changed: {name}")
    result = completed.verify_prerequisites()
    cfg = completed.read_config()
    completed.validate_config(cfg)
    for relative, (digest, count, quick) in D6_OUTPUTS.items():
        directory = ROOT / relative
        summary, success = short.read_json(directory / "summary.json"), short.read_json(directory / "SUCCESS.json")
        names = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
                 if p.is_file() and p.name != "SUCCESS.json"}
        status = "O2_D6_TECHNICAL_SMOKE_ONLY" if quick else "O2_D6_DEVELOPMENT_COMPLETE_REQUIRES_READ_ONLY_AUDIT"
        if (len(names) != count or set(success["artifact_sha256"]) != names or summary["status"] != status
                or success["status"] != status or success["summary_sha256"] != digest
                or optics.file_sha256(directory / "summary.json") != digest
                or (directory / "failure.json").exists() or (directory / "partial").exists()):
            raise RuntimeError("O2-D7 completed D6 seal changed")
        for name, expected in success["artifact_sha256"].items():
            target = (directory / name).resolve()
            if not target.is_relative_to(directory.resolve()) or optics.file_sha256(target) != expected:
                raise RuntimeError("O2-D7 completed D6 artifact changed")
        planned = completed.budget(cfg["quick" if quick else "data"])
        if (short.read_json(directory / "effective_config.json") != cfg
                or short.read_json(directory / "source_manifest.json") != completed.source_manifest(completed.CONFIG)
                or summary["quick"] is not quick or summary["completed_episodes"] != planned["complete_episodes"]
                or summary["completed_physical_transitions"] != planned["physical_transitions"]
                or summary["completed_poisson_draws"] != planned["poisson_draws"]
                or summary["completed_paired_prefix_checks"] != planned["paired_prefix_checks"]
                or summary["invalid_observations"] != 0 or summary["failed_or_truncated_episodes"] != 0
                or summary["replay_max_absolute_error"] != 0 or not summary["observation_quality"]["targets_met"]
                or summary["analysis"].get("status") != (None if quick else "DEVELOPMENT_DESCRIPTIVE_NOT_CONFIRMATION")):
            raise RuntimeError("O2-D7 D6 completed technical/source identity changed")
    return dict(result, D6_artifacts_checked=1035, frozen_D6_source_sha256=D6_PINS,
                D6_output_summary_sha256={p: v[0] for p, v in D6_OUTPUTS.items()})


def source_manifest(path: str | Path) -> dict[str, Any]:
    target = Path(path) if Path(path).is_absolute() else ROOT / path
    names = ("observation_bridge/combined_noise_closed_loop.py",
             "scripts/verify_observation_bridge_o2_combined_noise_closed_loop.py",
             "tests/test_observation_bridge_o2_combined_noise_closed_loop.py")
    return dict(new_source_sha256={n: optics.file_sha256(ROOT / n) for n in names},
                config_sha256=optics.file_sha256(target), frozen_D6_source=completed.source_manifest(completed.CONFIG),
                reused_causal_sources=technical.source_manifest(technical.CONFIG))


def preflight(path: str | Path = CONFIG, *, quick: bool = False) -> tuple:
    cfg = read_config(path)
    validate_config(cfg)
    spec = cfg["quick" if quick else "data"]
    output = (ROOT / cfg["quick_directory" if quick else "output_directory"]).resolve()
    output_root = (ROOT / "outputs").resolve()
    if output == output_root or not output.is_relative_to(output_root):
        raise ValueError("O2-D7 output must be a new child of outputs")
    if output.exists():
        raise FileExistsError(f"preserve existing O2-D7 output: {output}")
    frozen, streams = verify_prerequisites(), stream_manifest(cfg, quick=quick)
    short.configure_runtime()
    device = resolve_device(cfg["device"])
    if device.type != "cuda":
        raise RuntimeError("O2-D7 requires CUDA; no CPU fallback")
    parent = read_config(cfg["parent"])
    policies, scorers, models = short.load_assets(device, parent)
    if any(w in m.get("train_weather", []) + m.get("held_out_weather", [])
           for m in models for w in streams["weather_bases"]):
        raise RuntimeError("O2-D7 weather used by frozen scorer")
    report = dict(status="O2_D7_READY_FOR_TECHNICAL_SMOKE" if quick else "O2_D7_READY_FOR_USER_IDE",
                  quick=quick, **cfg["boundary"], **budget(cfg, spec), episode_length=spec["episode_length"],
                  weather_count=spec["weather_count"], camera_ids=spec["camera_ids"], device=str(device),
                  loaded_policies=len(policies), loaded_scorers=len(scorers), stream_manifest=streams,
                  frozen_sources=frozen, output_directory=str(output), gain_threshold=None,
                  preflight_model_forward_calls=0, preflight_environment_transitions=0,
                  synthetic_nominal_profile=short.nominal_profile(parent).as_record(),
                  fold_routing="weather_index_modulo_4_fixed_before_observation")
    return cfg, spec, output, device, parent, policies, scorers, models, report


@torch.no_grad()
def combined_intensity(clean: torch.Tensor, scale: float | None, read_std: float,
                       photon_seed: int | None, read_seed: int | None) -> tuple[torch.Tensor, dict[str, Any]]:
    """只接受当前强度和固定相机尺度；不存在真值、标签或未来观测参数。"""
    if (clean.device.type != "cuda" or clean.dtype != torch.float32 or clean.numel() == 0
            or not bool(torch.isfinite(clean).all()) or bool((clean < 0).any())):
        raise ValueError("O2-D7 finite nonnegative float32 CUDA intensity required")
    if (type(read_std) not in (int, float) or not math.isfinite(read_std) or read_std < 0
            or (read_std == 0 and read_seed is not None)
            or (read_std > 0 and (type(read_seed) is not int or not 0 <= read_seed < 2**63))
            or (photon_seed is not None and photon_seed == read_seed)):
        raise ValueError("O2-D7 finite read scale and independent explicit component seeds required")
    image, metadata = photon.photon_intensity(clean, scale, photon_seed)
    read_hash = read_rng_hash = None
    if read_std > 0:
        generator = torch.Generator(device=clean.device).manual_seed(read_seed)
        normal = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        image = image + read_std * normal
        read_hash, read_rng_hash = photon.tensor_sha256(normal), photon.tensor_sha256(generator.get_state())
    if not bool(torch.isfinite(image).all()):
        raise ValueError("O2-D7 nonfinite preclip camera intensity")
    negative = float(image.lt(0).double().mean())
    preclip_hash = photon.tensor_sha256(image)
    image = image.clamp_min(0)
    return image, dict(metadata, read_normal_sha256=read_hash, read_rng_after_draw_sha256=read_rng_hash,
                       negative_clip_fraction=negative, preclip_intensity_sha256=preclip_hash,
                       measured_intensity_sha256=photon.tensor_sha256(image))


class CombinedCamera:
    """每帧分别重置两类噪声种子；控制器分叉不改变下一帧种子身份。"""

    def __init__(self, device: torch.device, cfg: dict[str, Any], weather: int, camera: dict[str, Any]):
        if device.type != "cuda" or type(weather) is not int or camera not in cfg["camera_conditions"]:
            raise ValueError("O2-D7 fixed CUDA combined camera required")
        resolved = resolve_device(str(device))
        self.device = resolve_device(f"cuda:{torch.cuda.current_device() if resolved.index is None else resolved.index}")
        self.cfg, self.weather, self.camera = cfg, weather, dict(camera)
        self.frames = self.poisson_draws = self.read_noise_draws = 0
        self.records: list[dict[str, Any]] = []

    @torch.no_grad()
    def apply(self, clean: torch.Tensor) -> torch.Tensor:
        if (clean.device != self.device or clean.shape != (3, 512, 512) or clean.dtype != torch.float32
                or not bool(torch.isfinite(clean).all()) or bool((clean < 0).any())):
            raise ValueError("O2-D7 invalid current CUDA camera intensity")
        images = []
        for family in range(3):
            pseed, rseed = (component_seed(self.cfg, self.weather, self.camera["id"], self.frames, family, kind)
                            for kind in ("photon", "read"))
            image, metadata = combined_intensity(clean[family], self.camera["counts_per_intensity_unit"],
                                                  self.camera["read_noise_std"], pseed, rseed)
            images.append(image)
            self.poisson_draws += pseed is not None
            self.read_noise_draws += rseed is not None
            self.records.append(dict(observation_frame=self.frames, family_index=family,
                photon_seed=pseed, read_seed=rseed, camera_condition=self.camera["id"],
                counts_per_intensity_unit=self.camera["counts_per_intensity_unit"], read_noise_std=self.camera["read_noise_std"],
                clean_intensity_sha256=photon.tensor_sha256(clean[family]),
                clean_intensity_mean=float(clean[family].double().mean()), **metadata))
        self.frames += 1
        return torch.stack(images)

    def seed_sequence_sha256(self) -> str:
        return hashlib.sha256(json.dumps([[r["photon_seed"], r["read_seed"]] for r in self.records]).encode("utf-8")).hexdigest()


class CombinedPort(short.HolographicEnvironmentPort):
    def __init__(self, env: Any, sensor: Any, bridge: Any, camera: CombinedCamera, max_modal_error: float):
        super().__init__(env, sensor, bridge, max_modal_error)
        self.camera = camera

    @torch.no_grad()
    def observe(self) -> tuple[Any, dict[str, torch.Tensor]]:
        if self.env.slm.current_phase is None:
            raise RuntimeError("unknown actuator initial state")
        phase = self.env._current_turbulence_window() + self.env.slm.current_phase
        field = torch.polar(self.env.pupil.to(phase.dtype).expand_as(phase), phase)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        measured = self.bridge.measure(self.sensor.reconstruct(self.camera.apply(self.sensor.render(field))))
        end.record()
        end.synchronize()
        # 真值只在完成读出后查询并保存，不传给控制器，不修补偏差读数。
        target, true_jump = optics.audit_known_phase(phase.double(), self.bridge)
        if true_jump > self.bridge.tolerances.max_neighbor_jump_rad:
            raise RuntimeError("O2-D7 known spatial sampling precondition failed")
        difference = measured.residual_rad.double() - target
        error = difference.abs().amax(-1)
        ideal = self.camera.camera["counts_per_intensity_unit"] is None and self.camera.camera["read_noise_std"] == 0
        if ideal and float(error.max()) > self.max_modal_error:
            raise RuntimeError("O2-D7 ideal modal observation tolerance failed")
        rows = self.camera.records[-3:]
        audit = dict(joint_target_rad=target,
            legacy_projection_rad=short.project_phase_to_modes(phase, self.env.basis, self.env.pupil),
            modal_error_rad=error, modal_rmse_rad=difference.square().mean(-1).sqrt(), fit_rmse_rad=measured.fit_rmse_rad,
            max_wrapped_neighbor_jump_rad=measured.max_neighbor_jump_rad,
            batch_true_neighbor_jump_rad=error.new_full(error.shape, true_jump),
            camera_frame_index=torch.full((3,), self.camera.frames - 1, dtype=torch.int64, device=self.bridge.device),
            camera_negative_clip_fraction=error.new_tensor([r["negative_clip_fraction"] for r in rows]),
            observation_latency_ms=error.new_full(error.shape, start.elapsed_time(end)))
        for component in ("photon", "read"):
            audit[f"camera_{component}_seed"] = torch.tensor(
                [r[f"{component}_seed"] if r[f"{component}_seed"] is not None else -1 for r in rows],
                dtype=torch.int64, device=self.bridge.device)
        return short.SensorReadout(measured.residual_rad, self.env.step_count), audit


@torch.no_grad()
def rollout(cfg: dict[str, Any], spec: dict[str, Any], parent: dict, branch: dict, *, seed: int,
            weather_index: int, camera: dict, sensor: Any, bridge: Any, policy: Any, selector: Any,
            progress: Callable[[dict], None], context: dict, partial_directory: Path) -> tuple[dict, dict, dict, list[dict]]:
    context.update(controller=branch["controller"], weather_seed=seed, camera_condition=camera["id"],
                   counts_per_intensity_unit=camera["counts_per_intensity_unit"], read_noise_std=camera["read_noise_std"],
                   action_step=None, phase="reset", current_batch_completed_steps=0, pending_action=False, camera_frames=0)
    env = short.make_environment(parent, bridge.basis, seed, spec["episode_length"])
    noise = CombinedCamera(bridge.device, cfg, seed, camera)
    port = CombinedPort(env, sensor, bridge, noise, cfg["thresholds"]["ideal_modal_error_max_rad"])
    visible: dict[str, list[torch.Tensor]] = {}
    audits: dict[str, list[torch.Tensor]] = {}
    pending: dict[str, torch.Tensor] = {}
    try:
        readout, initial_audit = port.reset(seed)
        interface = R4Interface()
        interface.reset(readout.residual, episode_id=f"o2-d7-{seed}-{camera['id']}-{branch['controller']}")
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
                raise RuntimeError("O2-D7 causal observation clock mismatch")
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            original, selected, choice, prediction = short.choose_command(
                view.features, view.valid, policy, selector, step=step, cfg=cfg)
            end.record()
            end.synchronize()
            action = interface.issue(anchor_delta(view.features[:, -1], cfg["integrator"]), selected, step=step)
            pending = dict(history=view.features, valid=view.valid, original=original, selected=selected,
                           choice=choice, prediction=prediction, requested_delta=action.requested_delta_rad,
                           requested_modal=action.requested_modal_rad)
            context.update(phase="physical_step_and_next_observation", pending_action=True)
            readout, power, audit = port.step(action.requested_delta_rad, step, on_transition)
            transition = interface.observe_next(readout.residual, step=readout.observation_step, power=power)
            if not bool(transition.action_power_valid.all()):
                raise RuntimeError("O2-D7 causal action power missing")
            for key, value in dict(pending, measured_power=power.value,
                next_clock=transition.next_history.features[:, -1, 75:79],
                power_action_step=torch.tensor(power.action_step, device=bridge.device),
                power_arrival_step=torch.tensor(power.arrival_observation_step, device=bridge.device)).items():
                visible[key].append(value.detach().clone())
            visible["residual"].append(readout.residual.clone())
            audit["decision_latency_ms"] = readout.residual.new_full((3,), start.elapsed_time(end))
            for key, value in audit.items():
                audits.setdefault(key, []).append(value.detach().clone())
            context.update(current_batch_completed_steps=step + 1, pending_action=False, camera_frames=noise.frames)
            pending = {}
            progress(dict(weather_seed=seed, camera_condition=camera["id"], controller=branch["controller"],
                          observation_step=step + 1, physical_transitions=context["physical_transitions"],
                          modal_error_max_rad=float(audit["modal_error_rad"].max())))
        trace, audit_trace = short._stack(visible), short._stack(audits)
        frames = 3 * (spec["episode_length"] + 1)
        if (noise.frames != spec["episode_length"] + 1
                or noise.poisson_draws != (frames if camera["counts_per_intensity_unit"] is not None else 0)
                or noise.read_noise_draws != (frames if camera["read_noise_std"] > 0 else 0)
                or any(v.is_floating_point() and not bool(torch.isfinite(v).all())
                       for values in (trace, audit_trace) for v in values.values())):
            raise RuntimeError("O2-D7 camera budget or finite trace check failed")
        record = dict(**branch, weather_seed=seed, weather_index=weather_index, camera_condition=camera["id"],
            counts_per_intensity_unit=camera["counts_per_intensity_unit"], read_noise_std=camera["read_noise_std"],
            complete_episodes=3, physical_transitions=spec["episode_length"] * 3,
            scorer_fold=weather_index % 4 if selector is not None else None,
            camera_frames_per_family=noise.frames, poisson_draws=noise.poisson_draws, read_noise_draws=noise.read_noise_draws,
            camera_seed_sequence_sha256=noise.seed_sequence_sha256(),
            modal_error_max_rad=float(audit_trace["modal_error_rad"].max()),
            policy_forward_calls=spec["episode_length"] if policy is not None else 0,
            scorer_forward_calls=max(0, spec["episode_length"] - cfg["selector_start_step"]) if selector is not None else 0)
        return trace, audit_trace, record, noise.records
    except BaseException:
        context.update(camera_frames=noise.frames, poisson_draws_in_current_batch=noise.poisson_draws,
                       read_noise_draws_in_current_batch=noise.read_noise_draws, current_batch_environment_step=env.step_count,
                       incomplete_batch_size=3)
        try:
            partial_directory.mkdir(parents=True, exist_ok=False)
            for name, values in (("visible", short._stack(visible)), ("audit", short._stack(audits)), ("pending_request", pending)):
                technical.technical.save_tensors(partial_directory / f"{name}.pt", values)
            optics.write_json(partial_directory / "camera_frames.json", noise.records)
        except BaseException as save_error:
            context["partial_save_error"] = repr(save_error)
        raise


def validate_camera_rows(rows: list[dict], cfg: dict, spec: dict, *, weather: int, camera: dict) -> None:
    if len(rows) != 3 * (spec["episode_length"] + 1):
        raise ValueError("O2-D7 incomplete camera frame metadata")
    for i, row in enumerate(rows):
        frame, family = divmod(i, 3)
        scale, std = camera["counts_per_intensity_unit"], camera["read_noise_std"]
        if (row["observation_frame"] != frame or row["family_index"] != family
                or row["camera_condition"] != camera["id"] or row["counts_per_intensity_unit"] != scale
                or row["read_noise_std"] != std
                or any(row[f"{kind}_seed"] != component_seed(cfg, weather, camera["id"], frame, family, kind)
                       for kind in ("photon", "read"))
                or any(not photon._is_hash(row[k]) for k in ("clean_intensity_sha256", "preclip_intensity_sha256", "measured_intensity_sha256"))
                or type(row["clean_intensity_mean"]) not in (int, float) or not 0 <= row["clean_intensity_mean"] < float("inf")
                or type(row["negative_clip_fraction"]) not in (int, float) or not 0 <= row["negative_clip_fraction"] <= 1):
            raise ValueError("O2-D7 camera identity/seed/intensity mismatch")
        if scale is None:
            if any(row[k] is not None for k in (*photon.COUNT_COLUMNS, "counts_sha256", "noise_rng_after_draw_sha256")):
                raise ValueError("O2-D7 absent photon component consumed a draw")
        elif (not photon._is_hash(row["counts_sha256"]) or not photon._is_hash(row["noise_rng_after_draw_sha256"])
                or any(type(row[k]) not in (float, int) or not 0 <= row[k] < float("inf") for k in photon.COUNT_COLUMNS)
                or row["zero_count_fraction"] > 1 or row["sampled_count_max"] < row["sampled_count_mean"]
                or row["sampled_count_max"] != int(row["sampled_count_max"])
                or abs(row["expected_count_mean"] - scale * row["clean_intensity_mean"]) > 1e-9):
            raise ValueError("O2-D7 photon count metadata mismatch")
        hashes = (row["read_normal_sha256"], row["read_rng_after_draw_sha256"])
        if (std == 0 and any(h is not None for h in hashes)) or (std > 0 and any(not photon._is_hash(h) for h in hashes)):
            raise ValueError("O2-D7 read component draw metadata mismatch")
        if std == 0 and (row["negative_clip_fraction"] != 0 or row["preclip_intensity_sha256"] != row["measured_intensity_sha256"]):
            raise ValueError("O2-D7 nonnegative photon-only component unexpectedly clipped")
        if scale is None and std == 0 and row["measured_intensity_sha256"] != row["clean_intensity_sha256"]:
            raise ValueError("O2-D7 ideal intensity changed")


def seed_sequence(cfg: dict, weather: int, camera_id: str, length: int) -> str:
    pairs = [[component_seed(cfg, weather, camera_id, t, f, kind) for kind in ("photon", "read")]
             for t in range(length + 1) for f in range(3)]
    return hashlib.sha256(json.dumps(pairs).encode("utf-8")).hexdigest()


def validate_records(records: list[dict], cfg: dict, spec: dict) -> None:
    branches = {b["controller"]: b for b in prior.controller_specs(cfg)}
    seeds = _streams(cfg, spec)["weather_bases"]
    cameras = {c["id"]: c for c in cfg["camera_conditions"]}
    expected = {(s, c, b) for s in seeds for c in spec["camera_ids"] for b in branches}
    seen: set[tuple] = set()
    for row in records:
        key = row["weather_seed"], row["camera_condition"], row["controller"]
        if key in seen or key not in expected:
            raise ValueError("O2-D7 duplicated or unexpected record")
        branch, wi, camera = branches[key[2]], seeds.index(key[0]), cameras[key[1]]
        frames = 3 * (spec["episode_length"] + 1)
        if (any(row[k] != v for k, v in branch.items()) or row["weather_index"] != wi
                or row["scorer_fold"] != (wi % 4 if branch["scorer_seed"] is not None else None)
                or row["counts_per_intensity_unit"] != camera["counts_per_intensity_unit"]
                or row["read_noise_std"] != camera["read_noise_std"] or row["complete_episodes"] != 3
                or row["physical_transitions"] != 3 * spec["episode_length"]
                or row["camera_frames_per_family"] != spec["episode_length"] + 1
                or row["poisson_draws"] != (frames if camera["counts_per_intensity_unit"] is not None else 0)
                or row["read_noise_draws"] != (frames if camera["read_noise_std"] > 0 else 0)
                or row["camera_seed_sequence_sha256"] != seed_sequence(cfg, key[0], key[1], spec["episode_length"])
                or row["policy_forward_calls"] != (spec["episode_length"] if branch["member"] is not None else 0)
                or row["scorer_forward_calls"] != (max(0, spec["episode_length"] - 25) if branch["scorer_seed"] is not None else 0)):
            raise ValueError("O2-D7 record identity/fold/budget mismatch")
        seen.add(key)
    if seen != expected:
        raise ValueError("O2-D7 incomplete grid; no silent exclusion")


def observation_quality(parts: dict[str, list[torch.Tensor]], ideal_max: float, cfg: dict) -> dict:
    known = {c["id"] for c in cfg["camera_conditions"]}
    if not parts or not set(parts).issubset(known) or any(not v for v in parts.values()) or not 0 <= ideal_max < float("inf"):
        raise ValueError("O2-D7 empty/unknown/nonfinite quality cell")
    cells = []
    for camera in cfg["camera_conditions"]:
        if camera["id"] not in parts:
            continue
        values = torch.cat([x.reshape(-1).double() for x in parts[camera["id"]]])
        if values.device.type != "cuda" or values.numel() == 0 or not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
            raise ValueError("O2-D7 quality statistics require finite nonnegative CUDA observations")
        mean, p95 = float(values.mean()), float(torch.quantile(values, .95))
        ideal = camera["counts_per_intensity_unit"] is None and camera["read_noise_std"] == 0
        target = ideal or (mean <= cfg["thresholds"]["noisy_modal_rmse_mean_rad"]
                          and p95 <= cfg["thresholds"]["noisy_modal_rmse_p95_rad"])
        cells.append(dict(camera_condition=camera["id"], counts_per_intensity_unit=camera["counts_per_intensity_unit"],
            read_noise_std=camera["read_noise_std"], observed_family_frames=len(values), mean_modal_rmse_rad=mean,
            p95_modal_rmse_rad=p95, targets_met=target, including_initial_frame=True))
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
        for directory in ("trajectories", "audit", "camera"):
            (output / directory).mkdir()
        for name, value in (("effective_config.json", cfg), ("preflight.json", report), ("model_manifest.json", models),
                           ("stream_manifest.json", report["stream_manifest"]), ("source_manifest.json", source_manifest(path))):
            optics.write_json(output / name, value)
        sensor, bridge = optics.make_components(read_config(cfg["optics_config"]), device)
        records, manifest = [], []
        rmse: dict[str, list[torch.Tensor]] = {}
        initial_clean: dict[int, list[str]] = {}
        ideal_max = replay_max = 0.0
        prefixes = policy_calls = scorer_calls = progress_count = poisson_draws = read_draws = 0

        def progress(row: dict) -> None:
            nonlocal progress_count
            progress_count += 1
            elapsed = time.perf_counter() - started
            count, total = row["physical_transitions"], report["physical_transitions"]
            row.update(total_physical_transitions=total, elapsed_seconds=elapsed, eta_seconds=elapsed * (total - count) / count,
                       transitions_per_second=count / max(elapsed, 1e-9),
                       cuda_allocated_gib=torch.cuda.memory_allocated(device) / 2**30,
                       cuda_reserved_gib=torch.cuda.memory_reserved(device) / 2**30)
            with (output / "progress.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            if row["observation_step"] % 8 == 0 or row["observation_step"] == spec["episode_length"]:
                print(f"O2-D7 {'技术冒烟' if quick else '联合噪声短闭环'} {count}/{total} | {row['camera_condition']} "
                      f"{row['controller']} {row['observation_step']}/{spec['episode_length']}帧 | "
                      f"观测最大误差={row['modal_error_max_rad']:.3g}rad | {row['transitions_per_second']:.1f}转移/s "
                      f"剩余={row['eta_seconds']:.1f}s | 显存={row['cuda_allocated_gib']:.2f}/{row['cuda_reserved_gib']:.2f}GiB", flush=True)

        for wi, seed in enumerate(report["stream_manifest"]["weather_bases"]):
            for camera in (c for c in cfg["camera_conditions"] if c["id"] in spec["camera_ids"]):
                originals, original_cameras = {}, {}
                for branch in prior.controller_specs(cfg):
                    policy = policies.get(branch["member"])
                    selector = None if branch["scorer_seed"] is None else scorers[wi % 4, branch["scorer_seed"]]
                    trace, audit, record, camera_rows = rollout(cfg, spec, parent, branch, seed=seed, weather_index=wi,
                        camera=camera, sensor=sensor, bridge=bridge, policy=policy, selector=selector, progress=progress,
                        context=context, partial_directory=output / "partial")
                    context.update(phase="save_and_replay", incomplete_batch_size=0)
                    validate_camera_rows(camera_rows, cfg, spec, weather=seed, camera=camera)
                    clean_hashes = [r["clean_intensity_sha256"] for r in camera_rows[:3]]
                    if initial_clean.setdefault(seed, clean_hashes) != clean_hashes:
                        raise RuntimeError("O2-D7 factorial conditions do not share initial clean weather")
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
                        n = 3 * (cfg["selector_start_step"] + 1)
                        if original_cameras[branch["member"]][:n] != camera_rows[:n]:
                            raise RuntimeError("O2-D7 selector-disabled combined camera prefix differs")
                        record["paired_prefix_exact"] = record["paired_camera_prefix_exact"] = True
                        prefixes += 1
                    rmse.setdefault(camera["id"], []).append(audit["modal_rmse_rad"].detach().clone())
                    if camera["id"] == "noiseless":
                        ideal_max = max(ideal_max, record["modal_error_max_rad"])
                    records.append(record)
                    manifest.append(dict(file=f"{name}.pt", camera_file=f"{name}.json", **branch, weather_seed=seed,
                        camera_condition=camera["id"], visible_sha256=optics.file_sha256(output / "trajectories" / f"{name}.pt"),
                        audit_sha256=optics.file_sha256(output / "audit" / f"{name}.pt"),
                        camera_sha256=optics.file_sha256(output / "camera" / f"{name}.json")))
                    with (output / "records.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    context["completed_episode_batches"] += 1
                    policy_calls += record["policy_forward_calls"]
                    scorer_calls += record["scorer_forward_calls"]
                    poisson_draws += record["poisson_draws"]
                    read_draws += record["read_noise_draws"]
        validate_records(records, cfg, spec)
        if (context["physical_transitions"] != report["physical_transitions"] or progress_count != report["batched_steps"]
                or policy_calls != report["policy_forward_calls"] or scorer_calls != report["scorer_forward_calls"]
                or prefixes != report["paired_prefix_checks"] or poisson_draws != report["poisson_draws"]
                or read_draws != report["read_noise_draws"]):
            raise RuntimeError("O2-D7 execution budget mismatch")
        quality = observation_quality(rmse, ideal_max, cfg)
        verify_prerequisites()
        if short.read_json(output / "source_manifest.json") != source_manifest(path):
            raise RuntimeError("O2-D7 source/config changed during execution")
        result = dict(report, status="O2_D7_TECHNICAL_SMOKE_ONLY" if quick else "O2_D7_COMPLETE_REQUIRES_READ_ONLY_AUDIT",
            completed_episodes=len(records) * 3, completed_episode_batches=len(records),
            completed_physical_transitions=context["physical_transitions"], invalid_observations=0, failed_or_truncated_episodes=0,
            completed_paired_prefix_checks=prefixes, completed_poisson_draws=poisson_draws,
            completed_read_noise_draws=read_draws, replay_max_absolute_error=replay_max,
            completed_policy_forward_calls=policy_calls, completed_scorer_forward_calls=scorer_calls,
            replay_policy_forward_calls=policy_calls, replay_scorer_forward_calls=scorer_calls, replay_environment_transitions=0,
            observation_quality=quality, analysis={}, raw_camera_frames_saved=0, inverse_crime_limitation=True,
            real_accuracy_verified=False, real_camera_noise_calibrated=False, realtime_verified=False,
            equal_poisson_rng_end_state_not_assumed=True, elapsed_seconds=time.perf_counter() - started,
            runtime=photon.static._runtime(device),
            latency_scope="batch-3 CUDA render/Poisson/Gaussian/measurement with hash synchronizations; decision excludes projection; excludes exposure/IO/audit; not real end-to-end latency",
            next_action="Read-only audit. No automatic development, training, confirmation or hardware actions.")
        optics.write_json(output / "trajectory_manifest.json", manifest)
        optics.write_json(output / "summary.json", result)
        artifacts = {p.relative_to(output).as_posix(): optics.file_sha256(p) for p in output.rglob("*") if p.is_file()}
        optics.write_json(output / "SUCCESS.json", dict(status=result["status"], summary_sha256=artifacts["summary.json"], artifact_sha256=artifacts))
        return result
    except BaseException as exc:
        optics.write_json(output / "failure.json", dict(status="O2_D7_STOPPED_NO_AUTOMATIC_RETRY", exception=type(exc).__name__,
            message=str(exc), rejection_type=photon.static.rejection_type(exc) if isinstance(exc, ValueError) else None,
            traceback=traceback.format_exc(), last_context=context, **cfg["boundary"]))
        raise
