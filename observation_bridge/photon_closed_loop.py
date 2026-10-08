"""O2-D5：冻结模型的单因素光子噪声短闭环技术检查。

复用封存的 D4 泊松相机与 B 因果接口。每家族/光照/帧独立显式种子；
不假定不同强度图像消耗相同的泊松随机流，不重跑旧入口，不评价收益。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import traceback
from typing import Any, Callable

import torch

from observation_bridge import photon_noise_diagnostic as photon
from observation_bridge import noise_closed_loop as technical
from src.rl.r4_observation import R4Interface
from src.rl.r4_trajectory import anchor_delta
from src.runtime import resolve_device

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/experiments/observation_bridge_o2_d5_photon_closed_loop_v1.yaml"
short, optics, prior = technical.short, technical.optics, technical.prior
D4_PINS = {
    "observation_bridge/photon_noise_diagnostic.py": "da76ab537b60ae392c54e4af984f1887de5782da1e25579cd48f9fcdc45cacd7",
    "scripts/diagnose_observation_bridge_o2_photon_noise.py": "f174463aa68b7f8fd64cb18d28679ae8f468fc2edaba359138b59c235ad46d74",
    "tests/test_observation_bridge_o2_photon_noise.py": "503869a0af0d642d47c55ff7d52d22246e09e134a866df619c53682f2136a147",
    photon.CONFIG: "4fae174bbf492949c0539fade5c8db4148d42184faf4c964574adbad260dea65",
}
D4_OUTPUTS = {
    "outputs/observation_bridge_o2_d4_photon_noise_v1":
        ("d94f8f2dad73ba1821f8bdb88cf86e579f7c504d8514a2e3081ccc2395f087b6", False),
    "outputs/observation_bridge_o2_d4_photon_noise_v1_quick":
        ("d0993c00f4304c456b52236b0aa530c152a0e28f133b2ee852407a134f722e73", True),
}


def read_config(path: str | Path = CONFIG) -> dict[str, Any]:
    return photon.read_config(path)


def validate_config(cfg: dict[str, Any]) -> None:
    expected = dict(
        schema="observation_bridge_o2_d5_photon_closed_loop_v1",
        scope="synthetic_photon_noise_short_loop_technical_only", device="cuda",
        optics_config="configs/experiments/observation_bridge_o2_optics_v1.yaml",
        parent="configs/experiments/s4_r5_policy_training_v1.yaml",
        data=dict(weather_seed_base=8900000, weather_count=4, weather_seed_stride=1000,
                  episode_length=32, camera_ids=["noiseless", "photon_k100", "photon_k10"]),
        quick=dict(weather_seed_base=8960000, weather_count=1, weather_seed_stride=1000,
                   episode_length=28, camera_ids=["noiseless", "photon_k10"]),
        family_ids=["frozen", "boiling", "varying"],
        camera_conditions=[dict(id="noiseless", counts_per_intensity_unit=None),
                           dict(id="photon_k100", counts_per_intensity_unit=100.0),
                           dict(id="photon_k10", counts_per_intensity_unit=10.0)],
        camera_seed_offset=180000000, camera_level_seed_stride=1000000, camera_frame_seed_stride=10,
        noise_distribution="poisson_counts_divided_by_fixed_exposure_scale",
        noise_units="synthetic_detected_counts_per_arbitrary_intensity_unit", read_noise_std=0.0,
        image_dependent_normalization=False, shared_draw_across_levels=False, zero_noise_draws=0,
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
        raise ValueError("O2-D5 fixed single-factor technical contract changed")
    if any(not isinstance(cfg[k], str) or not cfg[k] for k in ("output_directory", "quick_directory")):
        raise ValueError("O2-D5 output paths required")
    if (ROOT / cfg["output_directory"]).resolve() == (ROOT / cfg["quick_directory"]).resolve():
        raise ValueError("O2-D5 formal and quick outputs must differ")


def budget(spec: dict[str, Any]) -> dict[str, int]:
    groups = spec["weather_count"] * len(spec["camera_ids"])
    finite_groups = spec["weather_count"] * sum(c != "noiseless" for c in spec["camera_ids"])
    batches, steps = groups * 13, spec["episode_length"]
    return dict(episode_batches=batches, complete_episodes=batches * 3, batched_steps=batches * steps,
                physical_transitions=batches * steps * 3, policy_forward_calls=groups * 12 * steps,
                scorer_forward_calls=groups * 9 * max(0, steps - 25), paired_prefix_checks=groups * 9,
                camera_batch_frames=batches * (steps + 1), camera_family_frames=batches * (steps + 1) * 3,
                poisson_draws=finite_groups * 13 * (steps + 1) * 3)


def frame_seed(cfg: dict[str, Any], weather: int, camera_id: str, frame: int, family: int) -> int | None:
    ids = [c["id"] for c in cfg["camera_conditions"]]
    if (camera_id not in ids or type(weather) is not int or type(frame) is not int or frame < 0
            or type(family) is not int or not 0 <= family < 3):
        raise ValueError("O2-D5 unknown camera/frame/family seed identity")
    level = ids.index(camera_id)
    return None if level == 0 else (weather + cfg["camera_seed_offset"]
        + level * cfg["camera_level_seed_stride"] + frame * cfg["camera_frame_seed_stride"] + family)


def _streams(cfg: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    weather = [spec["weather_seed_base"] + i * spec["weather_seed_stride"] for i in range(spec["weather_count"])]
    return dict(weather_bases=weather, turbulence=[w+i for w in weather for i in range(3)],
                power=[w+60000000 for w in weather], unused_proxy_sensor=[w+50000000 for w in weather],
                camera=[frame_seed(cfg,w,c,t,f) for w in weather for c in spec["camera_ids"] if c != "noiseless"
                        for t in range(spec["episode_length"]+1) for f in range(3)])


def stream_manifest(cfg: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    # 历史检查是纯种子清单计算，不调用任何旧实验入口或加载旧轨迹。
    prior.verify_streams(cfg, quick=quick)
    history = [prior._streams(prior.read_config(prior.CONFIG), prior.read_config(prior.CONFIG)[k])
               for k in ("data", "quick")]
    for module in (photon.static, technical, photon.completed, photon):
        old = module.read_config()
        history.extend(module.stream_manifest(old, quick=q) for q in (False, True))
    spec = cfg["quick" if quick else "data"]
    manifest, other = _streams(cfg, spec), _streams(cfg, cfg["data" if quick else "quick"])
    unique = [s for k in ("turbulence", "power", "unused_proxy_sensor", "camera") for s in manifest[k]]
    if len(set(unique)) != len(unique):
        raise RuntimeError("O2-D5 internal random stream collision")
    current = set(unique) | set(manifest["weather_bases"])
    unit_bases = (8470000, 8520000, 8670000, 8770000, 8870000, 8970000)
    reserved = set().union(*(set(range(n, n+10000)) for n in unit_bases))
    reserved |= {n+o for b in (8670000,8770000,8970000) for n in range(b,b+10000)
                 for o in (50000000,60000000,170000000,180000000,181000000,182000000)}
    if (current & (short._integers(other) | reserved)
            or any(current & short._integers(h) for h in history)):
        raise RuntimeError("O2-D5 historical/quick/unit seed collision")
    return dict(manifest, historical_manifests_checked=23, technical_unit_namespace=8970000,
                fold_assignment={str(w):i%4 for i,w in enumerate(manifest["weather_bases"])},
                seed_formula="weather+180000000+level_index*1000000+observation_frame*10+family_index",
                seed_identity_shared_within_level_across_controllers=True, shared_draw_across_levels=False,
                equal_poisson_rng_end_state_not_assumed=True, zero_noise_draws=0,
                frame_randomness_reseeded_to_avoid_rate_dependent_cross_frame_rng_drift=True,
                disjointness_scope="declared_G2_C1_B_C_D1_D2_D3_D4_other_D5_mode_and_reserved_units")


def verify_prerequisites() -> dict[str, Any]:
    for name, digest in D4_PINS.items():
        if optics.file_sha256(ROOT/name) != digest:
            raise RuntimeError(f"O2-D5 frozen D4 source changed: {name}")
    result = photon.verify_prerequisites()
    cfg = photon.read_config()
    photon.validate_config(cfg)
    for relative, (digest, quick) in D4_OUTPUTS.items():
        directory = ROOT/relative
        summary, success = short.read_json(directory/"summary.json"), short.read_json(directory/"SUCCESS.json")
        names = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
                 if p.is_file() and p.name != "SUCCESS.json"}
        status = "O2_D4_TECHNICAL_SMOKE_ONLY" if quick else "O2_D4_DIAGNOSTIC_COMPLETE_REQUIRES_READ_ONLY_AUDIT"
        if (len(names) != 8 or set(success["artifact_sha256"]) != names
                or success["summary_sha256"] != digest or optics.file_sha256(directory/"summary.json") != digest
                or summary["status"] != status or success["status"] != status
                or (directory/"failure.json").exists() or (directory/"partial").exists()):
            raise RuntimeError("O2-D5 D4 completed artifact seal changed")
        for name, expected in success["artifact_sha256"].items():
            target = (directory/name).resolve()
            if not target.is_relative_to(directory.resolve()) or optics.file_sha256(target) != expected:
                raise RuntimeError("O2-D5 D4 completed artifact changed")
        planned = photon.budget(cfg["quick" if quick else "data"])
        if (summary["completed_measurement_attempts"] != planned["measurement_attempts"]
                or summary["completed_poisson_draws"] != planned["poisson_draws"]
                or summary["rejected_readings"] != (0 if quick else 3)
                or summary["valid_readings"] != planned["measurement_attempts"]-(0 if quick else 3)
                or not summary["technical_grid_validated"]
                or short.read_json(directory/"effective_config.json") != cfg
                or short.read_json(directory/"source_manifest.json") != photon.source_manifest(photon.CONFIG)):
            raise RuntimeError("O2-D5 D4 source/grid identity changed")
        if not quick:
            safe = [c for c in summary["cells"] if c["counts_per_intensity_unit"] in (None,100.0,10.0)]
            if len(safe) != 9 or any(c["rejected_readings"] != 0 for c in safe):
                raise RuntimeError("O2-D5 selected D4 light levels not rejection-free")
    return dict(result, D4_artifacts_checked=16, frozen_D4_source_sha256=D4_PINS,
                D4_output_summary_sha256={p:v[0] for p,v in D4_OUTPUTS.items()})


def source_manifest(path: str | Path) -> dict[str, Any]:
    target = Path(path) if Path(path).is_absolute() else ROOT/path
    names = ("observation_bridge/photon_closed_loop.py", "scripts/verify_observation_bridge_o2_photon_closed_loop.py",
             "tests/test_observation_bridge_o2_photon_closed_loop.py")
    return dict(new_source_sha256={n:optics.file_sha256(ROOT/n) for n in names},
                config_sha256=optics.file_sha256(target), frozen_D4_source=photon.source_manifest(photon.CONFIG),
                reused_causal_sources=short.source_manifest(short.CONFIG))


def preflight(path: str | Path = CONFIG, *, quick: bool = False) -> tuple:
    cfg = read_config(path)
    validate_config(cfg)
    spec = cfg["quick" if quick else "data"]
    output = (ROOT/cfg["quick_directory" if quick else "output_directory"]).resolve()
    if output == (ROOT/"outputs").resolve() or not output.is_relative_to((ROOT/"outputs").resolve()):
        raise ValueError("O2-D5 output must be a new child of outputs")
    if output.exists():
        raise FileExistsError(f"preserve existing O2-D5 output: {output}")
    frozen, streams = verify_prerequisites(), stream_manifest(cfg, quick=quick)
    short.configure_runtime()
    device = resolve_device(cfg["device"])
    if device.type != "cuda":
        raise RuntimeError("O2-D5 requires CUDA; no CPU fallback")
    parent = read_config(cfg["parent"])
    policies, scorers, models = short.load_assets(device, parent)
    if any(w in m.get("train_weather", [])+m.get("held_out_weather", [])
           for m in models for w in streams["weather_bases"]):
        raise RuntimeError("O2-D5 weather used by frozen scorer")
    report = dict(status="O2_D5_READY_FOR_TECHNICAL_SMOKE" if quick else "O2_D5_READY_FOR_USER_IDE",
                  quick=quick, **cfg["boundary"], **budget(spec), episode_length=spec["episode_length"],
                  weather_count=spec["weather_count"], camera_ids=spec["camera_ids"], device=str(device),
                  loaded_policies=len(policies), loaded_scorers=len(scorers), stream_manifest=streams,
                  frozen_sources=frozen, output_directory=str(output), gain_threshold=None,
                  preflight_model_forward_calls=0, preflight_environment_transitions=0,
                  synthetic_nominal_profile=short.nominal_profile(parent).as_record(),
                  fold_routing="weather_index_modulo_4_fixed_before_observation")
    return cfg,spec,output,device,parent,policies,scorers,models,report


class PhotonCamera:
    """只接受当前强度；每帧种子由时钟固定，控制器改变强度不会改变下一帧种子。"""

    def __init__(self, device: torch.device, cfg: dict[str, Any], weather: int, camera: dict[str, Any]):
        if (device.type != "cuda" or type(weather) is not int
                or camera not in cfg["camera_conditions"]):
            raise ValueError("O2-D5 fixed CUDA photon camera required")
        resolved = resolve_device(str(device))
        self.device = resolve_device(f"cuda:{torch.cuda.current_device() if resolved.index is None else resolved.index}")
        self.cfg, self.weather, self.camera = cfg,weather,camera
        self.frames = self.draws = 0
        self.records: list[dict[str, Any]] = []

    @torch.no_grad()
    def apply(self, clean: torch.Tensor) -> torch.Tensor:
        if (clean.device != self.device or clean.shape != (3,512,512) or clean.dtype != torch.float32
                or not bool(torch.isfinite(clean).all()) or bool((clean < 0).any())):
            raise ValueError("O2-D5 invalid current CUDA camera intensity")
        images, rows = [],[]
        for family in range(3):
            seed = frame_seed(self.cfg,self.weather,self.camera["id"],self.frames,family)
            image, metadata = photon.photon_intensity(clean[family],self.camera["counts_per_intensity_unit"],seed)
            images.append(image)
            self.draws += seed is not None
            rows.append(dict(observation_frame=self.frames,family_index=family,noise_seed=seed,
                             camera_condition=self.camera["id"],counts_per_intensity_unit=self.camera["counts_per_intensity_unit"],
                             clean_intensity_sha256=photon.tensor_sha256(clean[family]),
                             clean_intensity_mean=float(clean[family].double().mean()), **metadata))
        self.records.extend(rows)
        self.frames += 1
        return torch.stack(images)

    def seed_sequence_sha256(self) -> str:
        return hashlib.sha256(json.dumps([r["noise_seed"] for r in self.records]).encode("utf-8")).hexdigest()


class PhotonPort(short.HolographicEnvironmentPort):
    def __init__(self, env: Any, sensor: Any, bridge: Any, camera: PhotonCamera, max_modal_error: float):
        super().__init__(env,sensor,bridge,max_modal_error)
        self.camera = camera

    @torch.no_grad()
    def observe(self) -> tuple[Any, dict[str, torch.Tensor]]:
        if self.env.slm.current_phase is None:
            raise RuntimeError("unknown actuator initial state")
        phase = self.env._current_turbulence_window()+self.env.slm.current_phase
        field = torch.polar(self.env.pupil.to(phase.dtype).expand_as(phase),phase)
        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        measured = self.bridge.measure(self.sensor.reconstruct(self.camera.apply(self.sensor.render(field))))
        end.record(); end.synchronize()
        # 相机返回后独立查真值；不回传给 choose_command，不补读数、不逐帧剔除误差。
        target,true_jump = optics.audit_known_phase(phase.double(),self.bridge)
        if true_jump > self.bridge.tolerances.max_neighbor_jump_rad:
            raise RuntimeError("O2-D5 known spatial sampling precondition failed")
        difference = measured.residual_rad.double()-target
        error = difference.abs().amax(-1)
        if self.camera.camera["counts_per_intensity_unit"] is None and float(error.max()) > self.max_modal_error:
            raise RuntimeError("O2-D5 ideal modal observation tolerance failed")
        rows = self.camera.records[-3:]
        audit = dict(joint_target_rad=target,
                     legacy_projection_rad=short.project_phase_to_modes(phase,self.env.basis,self.env.pupil),
                     modal_error_rad=error,modal_rmse_rad=difference.square().mean(-1).sqrt(),
                     fit_rmse_rad=measured.fit_rmse_rad,max_wrapped_neighbor_jump_rad=measured.max_neighbor_jump_rad,
                     batch_true_neighbor_jump_rad=error.new_full(error.shape,true_jump),
                     camera_frame_index=torch.full((3,),self.camera.frames-1,dtype=torch.int64,device=self.bridge.device),
                     camera_noise_seed=torch.tensor([r["noise_seed"] if r["noise_seed"] is not None else -1 for r in rows],
                                                    dtype=torch.int64,device=self.bridge.device),
                     observation_latency_ms=error.new_full(error.shape,start.elapsed_time(end)))
        return short.SensorReadout(measured.residual_rad,self.env.step_count),audit


@torch.no_grad()
def rollout(cfg: dict[str, Any], spec: dict[str, Any], parent: dict, branch: dict, *, seed: int,
            weather_index: int, camera: dict, sensor: Any, bridge: Any, policy: Any, selector: Any,
            progress: Callable[[dict],None], context: dict, partial_directory: Path) -> tuple[dict,dict,dict,list[dict]]:
    context.update(controller=branch["controller"],weather_seed=seed,camera_condition=camera["id"],
                   counts_per_intensity_unit=camera["counts_per_intensity_unit"],action_step=None,phase="reset",
                   current_batch_completed_steps=0,pending_action=False,camera_frames=0)
    env = short.make_environment(parent,bridge.basis,seed,spec["episode_length"])
    noise = PhotonCamera(bridge.device,cfg,seed,camera)
    port = PhotonPort(env,sensor,bridge,noise,cfg["thresholds"]["ideal_modal_error_max_rad"])
    visible: dict[str,list[torch.Tensor]] = {}
    audits: dict[str,list[torch.Tensor]] = {}
    pending: dict[str,torch.Tensor] = {}
    try:
        readout,initial_audit = port.reset(seed)
        interface = R4Interface()
        interface.reset(readout.residual,episode_id=f"o2-d5-{seed}-{camera['id']}-{branch['controller']}")
        visible = {k:[] for k in ("history","valid","original","selected","choice","prediction","requested_delta",
                   "requested_modal","residual","measured_power","next_clock","power_action_step","power_arrival_step")}
        visible["residual"].append(readout.residual.clone())
        audits = {k:[v.clone()] for k,v in initial_audit.items()}
        def on_transition() -> None:
            context["physical_transitions"] += 3
        for step in range(spec["episode_length"]):
            context.update(action_step=step,phase="decision",pending_action=False)
            view = interface.snapshot()
            if readout.observation_step != step or view.observation_step != step:
                raise RuntimeError("O2-D5 causal observation clock mismatch")
            start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            original,selected,choice,prediction = short.choose_command(view.features,view.valid,policy,selector,step=step,cfg=cfg)
            end.record(); end.synchronize()
            action = interface.issue(anchor_delta(view.features[:,-1],cfg["integrator"]),selected,step=step)
            pending = dict(history=view.features,valid=view.valid,original=original,selected=selected,
                           choice=choice,prediction=prediction,requested_delta=action.requested_delta_rad,
                           requested_modal=action.requested_modal_rad)
            context.update(phase="physical_step_and_next_observation",pending_action=True)
            readout,power,audit = port.step(action.requested_delta_rad,step,on_transition)
            transition = interface.observe_next(readout.residual,step=readout.observation_step,power=power)
            if not bool(transition.action_power_valid.all()):
                raise RuntimeError("O2-D5 causal action power missing")
            for key,value in dict(pending,measured_power=power.value,
                next_clock=transition.next_history.features[:,-1,75:79],
                power_action_step=torch.tensor(power.action_step,device=bridge.device),
                power_arrival_step=torch.tensor(power.arrival_observation_step,device=bridge.device)).items():
                visible[key].append(value.detach().clone())
            visible["residual"].append(readout.residual.clone())
            audit["decision_latency_ms"] = readout.residual.new_full((3,),start.elapsed_time(end))
            for key,value in audit.items():
                audits.setdefault(key,[]).append(value.detach().clone())
            context.update(current_batch_completed_steps=step+1,pending_action=False,camera_frames=noise.frames)
            pending = {}
            progress(dict(weather_seed=seed,camera_condition=camera["id"],controller=branch["controller"],
                          observation_step=step+1,physical_transitions=context["physical_transitions"],
                          modal_error_max_rad=float(audit["modal_error_rad"].max())))
        trace,audit_trace = short._stack(visible),short._stack(audits)
        expected_draws = 0 if camera["counts_per_intensity_unit"] is None else 3*(spec["episode_length"]+1)
        if (noise.frames != spec["episode_length"]+1 or noise.draws != expected_draws
                or any(v.is_floating_point() and not bool(torch.isfinite(v).all())
                       for d in (trace,audit_trace) for v in d.values())):
            raise RuntimeError("O2-D5 camera budget or finite trace check failed")
        record = dict(**branch,weather_seed=seed,weather_index=weather_index,camera_condition=camera["id"],
                      counts_per_intensity_unit=camera["counts_per_intensity_unit"],read_noise_std=0.0,complete_episodes=3,
                      physical_transitions=spec["episode_length"]*3,scorer_fold=weather_index%4 if selector else None,
                      camera_frames_per_family=noise.frames,poisson_draws=noise.draws,
                      camera_seed_sequence_sha256=noise.seed_sequence_sha256(),
                      modal_error_max_rad=float(audit_trace["modal_error_rad"].max()),
                      policy_forward_calls=spec["episode_length"] if policy is not None else 0,
                      scorer_forward_calls=max(0,spec["episode_length"]-cfg["selector_start_step"]) if selector else 0)
        return trace,audit_trace,record,noise.records
    except BaseException:
        context.update(camera_frames=noise.frames,poisson_draws_in_current_batch=noise.draws,
                       current_batch_environment_step=env.step_count,incomplete_batch_size=3)
        try:
            partial_directory.mkdir(parents=True,exist_ok=False)
            for name,values in (("visible",short._stack(visible)),("audit",short._stack(audits)),("pending_request",pending)):
                technical.save_tensors(partial_directory/f"{name}.pt",values)
            optics.write_json(partial_directory/"camera_frames.json",noise.records)
        except BaseException as save_error:
            context["partial_save_error"] = repr(save_error)
        raise


def validate_camera_rows(rows: list[dict], cfg: dict, spec: dict, *, weather: int, camera: dict) -> None:
    if len(rows) != 3*(spec["episode_length"]+1):
        raise ValueError("O2-D5 incomplete camera frame metadata")
    for i,r in enumerate(rows):
        frame,family = divmod(i,3)
        seed = frame_seed(cfg,weather,camera["id"],frame,family)
        scale = camera["counts_per_intensity_unit"]
        if (r["observation_frame"] != frame or r["family_index"] != family or r["noise_seed"] != seed
                or r["camera_condition"] != camera["id"] or r["counts_per_intensity_unit"] != scale
                or not photon._is_hash(r["clean_intensity_sha256"])
                or type(r["clean_intensity_mean"]) not in (int,float) or not 0 <= r["clean_intensity_mean"] < float("inf")):
            raise ValueError("O2-D5 camera identity/seed/clean intensity mismatch")
        if scale is None:
            if any(r[k] is not None for k in (*photon.COUNT_COLUMNS,"counts_sha256","noise_rng_after_draw_sha256")):
                raise ValueError("O2-D5 noiseless camera consumed a draw")
        else:
            if (not photon._is_hash(r["counts_sha256"]) or not photon._is_hash(r["noise_rng_after_draw_sha256"])
                    or any(type(r[k]) not in (float,int) or not 0 <= r[k] < float("inf") for k in photon.COUNT_COLUMNS)
                    or r["zero_count_fraction"] > 1 or r["sampled_count_max"] < r["sampled_count_mean"]
                    or r["sampled_count_max"] != int(r["sampled_count_max"])
                    or abs(r["expected_count_mean"]-scale*r["clean_intensity_mean"]) > 1e-9):
                raise ValueError("O2-D5 photon count metadata mismatch")


def validate_records(records: list[dict], cfg: dict, spec: dict) -> None:
    branches = {b["controller"]:b for b in prior.controller_specs(cfg)}
    seeds = _streams(cfg,spec)["weather_bases"]
    cameras = {c["id"]:c for c in cfg["camera_conditions"]}
    expected = {(s,c,b) for s in seeds for c in spec["camera_ids"] for b in branches}
    seen: set[tuple] = set()
    for r in records:
        key = r["weather_seed"],r["camera_condition"],r["controller"]
        if key in seen or key not in expected:
            raise ValueError("O2-D5 duplicated or unexpected record")
        branch,wi = branches[key[2]],seeds.index(key[0])
        camera = cameras[key[1]]
        seed_hash = hashlib.sha256(json.dumps([frame_seed(cfg,key[0],key[1],t,f)
                    for t in range(spec["episode_length"]+1) for f in range(3)]).encode("utf-8")).hexdigest()
        if (any(r[k] != v for k,v in branch.items()) or r["weather_index"] != wi
                or r["scorer_fold"] != (wi%4 if branch["scorer_seed"] is not None else None)
                or r["counts_per_intensity_unit"] != camera["counts_per_intensity_unit"] or r["read_noise_std"] != 0.0
                or r["complete_episodes"] != 3 or r["physical_transitions"] != 3*spec["episode_length"]
                or r["camera_frames_per_family"] != spec["episode_length"]+1
                or r["poisson_draws"] != (0 if camera["counts_per_intensity_unit"] is None else 3*(spec["episode_length"]+1))
                or r["camera_seed_sequence_sha256"] != seed_hash
                or r["policy_forward_calls"] != (spec["episode_length"] if branch["member"] is not None else 0)
                or r["scorer_forward_calls"] != (max(0,spec["episode_length"]-25) if branch["scorer_seed"] is not None else 0)):
            raise ValueError("O2-D5 record identity/fold/budget mismatch")
        seen.add(key)
    if seen != expected:
        raise ValueError("O2-D5 incomplete grid; no silent exclusion")


def observation_quality(parts: dict[str,list[torch.Tensor]], ideal_max: float, cfg: dict) -> dict:
    known = {c["id"] for c in cfg["camera_conditions"]}
    if (not parts or not set(parts).issubset(known) or any(not v for v in parts.values())
            or not 0 <= ideal_max < float("inf")):
        raise ValueError("O2-D5 empty/unknown/nonfinite quality cell")
    cells = []
    for camera in cfg["camera_conditions"]:
        if camera["id"] not in parts:
            continue
        values = torch.cat([x.reshape(-1).double() for x in parts[camera["id"]]])
        if values.device.type != "cuda" or values.numel() == 0 or not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
            raise ValueError("O2-D5 quality statistics require finite nonnegative CUDA observations")
        mean,p95 = float(values.mean()),float(torch.quantile(values,.95))
        target = camera["counts_per_intensity_unit"] is None or (mean <= cfg["thresholds"]["noisy_modal_rmse_mean_rad"]
                                                       and p95 <= cfg["thresholds"]["noisy_modal_rmse_p95_rad"])
        cells.append(dict(camera_condition=camera["id"],counts_per_intensity_unit=camera["counts_per_intensity_unit"],
                          observed_family_frames=len(values),mean_modal_rmse_rad=mean,p95_modal_rmse_rad=p95,
                          targets_met=target,including_initial_frame=True))
    return dict(cells=cells,ideal_max_modal_error_rad=ideal_max,
                targets_met=ideal_max <= cfg["thresholds"]["ideal_modal_error_max_rad"] and all(c["targets_met"] for c in cells),
                noisy_error_used_to_replace_measurements=False,
                scope="technical_observation_targets_not_compensation_gain_gate",
                inference_unit="descriptive_correlated_frames_not_independent_weather_samples")


def run(path: str | Path = CONFIG, *, quick: bool = False, preflight_only: bool = False) -> dict[str, Any]:
    cfg,spec,output,device,parent,policies,scorers,models,report = preflight(path,quick=quick)
    if preflight_only:
        return report
    output.mkdir(parents=True,exist_ok=False)
    context = dict(physical_transitions=0,completed_episode_batches=0,incomplete_batch_size=0)
    started = time.perf_counter()
    try:
        for directory in ("trajectories","audit","camera"):
            (output/directory).mkdir()
        for name,value in (("effective_config.json",cfg),("preflight.json",report),("model_manifest.json",models),
                           ("stream_manifest.json",report["stream_manifest"]),("source_manifest.json",source_manifest(path))):
            optics.write_json(output/name,value)
        sensor,bridge = optics.make_components(read_config(cfg["optics_config"]),device)
        records,manifest = [],[]
        rmse: dict[str,list[torch.Tensor]] = {}
        ideal_max = replay_max = 0.0
        prefixes = policy_calls = scorer_calls = progress_count = draws = 0
        def progress(row: dict) -> None:
            nonlocal progress_count
            progress_count += 1
            elapsed = time.perf_counter()-started
            count,total = row["physical_transitions"],report["physical_transitions"]
            row.update(total_physical_transitions=total,elapsed_seconds=elapsed,eta_seconds=elapsed*(total-count)/count,
                       transitions_per_second=count/max(elapsed,1e-9),
                       cuda_allocated_gib=torch.cuda.memory_allocated(device)/2**30,
                       cuda_reserved_gib=torch.cuda.memory_reserved(device)/2**30)
            with (output/"progress.jsonl").open("a",encoding="utf-8") as handle:
                handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
            if row["observation_step"]%8 == 0 or row["observation_step"] == spec["episode_length"]:
                print(f"O2-D5 {'技术冒烟' if quick else '短闭环检查'} {count}/{total} | {row['camera_condition']} "
                      f"{row['controller']} {row['observation_step']}/{spec['episode_length']}帧 | "
                      f"观测最大误差={row['modal_error_max_rad']:.3g}rad | {row['transitions_per_second']:.1f}转移/s "
                      f"剩余={row['eta_seconds']:.1f}s | 显存={row['cuda_allocated_gib']:.2f}/{row['cuda_reserved_gib']:.2f}GiB",flush=True)
        for wi,seed in enumerate(report["stream_manifest"]["weather_bases"]):
            for camera in (c for c in cfg["camera_conditions"] if c["id"] in spec["camera_ids"]):
                originals,original_cameras = {},{}
                for branch in prior.controller_specs(cfg):
                    policy = policies.get(branch["member"])
                    selector = None if branch["scorer_seed"] is None else scorers[wi%4,branch["scorer_seed"]]
                    trace,audit,record,camera_rows = rollout(cfg,spec,parent,branch,seed=seed,weather_index=wi,camera=camera,
                        sensor=sensor,bridge=bridge,policy=policy,selector=selector,progress=progress,
                        context=context,partial_directory=output/"partial")
                    context.update(phase="save_and_replay",incomplete_batch_size=0)
                    validate_camera_rows(camera_rows,cfg,spec,weather=seed,camera=camera)
                    name = f"weather_{wi:02d}_{camera['id']}_{branch['controller']}"
                    for directory,value in (("trajectories",trace),("audit",audit)):
                        technical.save_tensors(output/directory/f"{name}.pt",value)
                    optics.write_json(output/"camera"/f"{name}.json",camera_rows)
                    saved = torch.load(output/"trajectories"/f"{name}.pt",map_location=device,weights_only=True)
                    record["replay"] = short.replay_visible(saved,{**cfg,"episode_length":spec["episode_length"]},policy,selector)
                    replay_max = max(replay_max,record["replay"]["max_absolute_error"])
                    if selector is None and branch["member"] is not None:
                        originals[branch["member"]],original_cameras[branch["member"]] = trace,camera_rows
                    if selector is not None:
                        short.require_prefix(originals[branch["member"]],trace,cfg["selector_start_step"])
                        if original_cameras[branch["member"]][:3*(cfg["selector_start_step"]+1)] != camera_rows[:3*(cfg["selector_start_step"]+1)]:
                            raise RuntimeError("O2-D5 selector-disabled photon camera prefix differs")
                        record["paired_prefix_exact"] = record["paired_camera_prefix_exact"] = True
                        prefixes += 1
                    rmse.setdefault(camera["id"],[]).append(audit["modal_rmse_rad"].detach().clone())
                    if camera["counts_per_intensity_unit"] is None:
                        ideal_max = max(ideal_max,record["modal_error_max_rad"])
                    records.append(record)
                    manifest.append(dict(file=f"{name}.pt",camera_file=f"{name}.json",**branch,
                                         weather_seed=seed,camera_condition=camera["id"],
                                         visible_sha256=optics.file_sha256(output/"trajectories"/f"{name}.pt"),
                                         audit_sha256=optics.file_sha256(output/"audit"/f"{name}.pt"),
                                         camera_sha256=optics.file_sha256(output/"camera"/f"{name}.json")))
                    with (output/"records.jsonl").open("a",encoding="utf-8") as handle:
                        handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")
                    context["completed_episode_batches"] += 1
                    policy_calls += record["policy_forward_calls"]
                    scorer_calls += record["scorer_forward_calls"]
                    draws += record["poisson_draws"]
        validate_records(records,cfg,spec)
        if (context["physical_transitions"] != report["physical_transitions"] or progress_count != report["batched_steps"]
                or policy_calls != report["policy_forward_calls"] or scorer_calls != report["scorer_forward_calls"]
                or prefixes != report["paired_prefix_checks"] or draws != report["poisson_draws"]):
            raise RuntimeError("O2-D5 execution budget mismatch")
        quality = observation_quality(rmse,ideal_max,cfg)
        verify_prerequisites()
        if short.read_json(output/"source_manifest.json") != source_manifest(path):
            raise RuntimeError("O2-D5 source/config changed during execution")
        result = dict(report,status="O2_D5_TECHNICAL_SMOKE_ONLY" if quick else "O2_D5_COMPLETE_REQUIRES_READ_ONLY_AUDIT",
                      completed_episodes=len(records)*3,completed_episode_batches=len(records),
                      completed_physical_transitions=context["physical_transitions"],invalid_observations=0,
                      failed_or_truncated_episodes=0,completed_paired_prefix_checks=prefixes,
                      completed_poisson_draws=draws,replay_max_absolute_error=replay_max,
                      completed_policy_forward_calls=policy_calls,completed_scorer_forward_calls=scorer_calls,
                      replay_policy_forward_calls=policy_calls,replay_scorer_forward_calls=scorer_calls,
                      replay_environment_transitions=0,observation_quality=quality,analysis={},
                      raw_camera_frames_saved=0,inverse_crime_limitation=True,real_accuracy_verified=False,
                      real_camera_noise_calibrated=False,realtime_verified=False,
                      equal_poisson_rng_end_state_not_assumed=True,elapsed_seconds=time.perf_counter()-started,
                      runtime=photon.static._runtime(device),
                      latency_scope="batch-3 CUDA render/photon sampling/measurement with hash synchronizations; decision excludes projection; excludes exposure/IO/audit; not real end-to-end latency",
                      next_action="Read-only audit. No automatic development, training, confirmation or hardware actions.")
        optics.write_json(output/"trajectory_manifest.json",manifest)
        optics.write_json(output/"summary.json",result)
        artifacts = {p.relative_to(output).as_posix():optics.file_sha256(p) for p in output.rglob("*") if p.is_file()}
        optics.write_json(output/"SUCCESS.json",dict(status=result["status"],summary_sha256=artifacts["summary.json"],artifact_sha256=artifacts))
        return result
    except BaseException as exc:
        optics.write_json(output/"failure.json",dict(status="O2_D5_STOPPED_NO_AUTOMATIC_RETRY",exception=type(exc).__name__,
                         message=str(exc),rejection_type=photon.static.rejection_type(exc) if isinstance(exc,ValueError) else None,
                         traceback=traceback.format_exc(),last_context=context,**cfg["boundary"]))
        raise
