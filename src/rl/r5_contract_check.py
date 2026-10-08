"""R5-0诊断入口：无模型训练、无硬件连接；正式与冒烟输出隔离。"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import time
import traceback

import torch

from src.rl.r4_control import NominalCalibration
from src.rl.r4_interface_smoke import seed_blocks, write_json
from src.rl.r4_observation import R4Interface, PowerMeasurement, simulation_residual_proxy
from src.rl.r4_trajectory import anchor_delta
from src.rl.r5_physics_adapter import DifferentiableSlm, causal_policy_features, measured_objective
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _load_yaml, _profiles, _project_path, _file_sha256, _relative
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.optics import focal_plane_metrics
from src.simulation.modes import make_pupil_mask
from src.simulation.robust_control import RobustnessCondition
from src.training_progress import counted_progress, update_progress

CONTRACT = "configs/experiments/s4_r5_physics_check_v2_sources.json"
BOUNDARY = dict(training_updates=0, real_slm_actions=False, confirmation_access=False,
                scientific_ranking=False, automatic_retry=False)


def used_seed_blocks(obj: dict, *, own_effective_config: bool = False) -> set[int]:
    """本实验配置同时声明两种预算；实际使用的随机流以预检streams记录为准。"""
    if own_effective_config:
        return set()
    if "streams" in obj and obj.get("seed_audit") == "conservative_blocks_and_known_offsets":
        values = [s for group in obj["streams"].values() for s in group]
        if any(type(s) is not int or s < 0 for s in values):
            raise ValueError("invalid historical stream record")
        return {s//10000 for s in values}
    return seed_blocks(obj)


def validate_config(cfg: dict) -> None:
    if (cfg.get("stage") != "S4-D2-R5-0" or cfg.get("design_only", False)
            or cfg.get("runtime") != {"device": "cuda", "require_cuda": True}
            or cfg.get("boundary") != BOUNDARY):
        raise ValueError("R5-0 requires frozen CUDA diagnostic configuration")
    if (cfg["formal"] != dict(seed_base=5200000, family_indices=[0, 1, 2],
                              weather_per_family=4, steps=200, gradient_directions=4)
            or cfg["quick"] != dict(seed_base=5280000, family_indices=[0],
                                     weather_per_family=1, steps=16, gradient_directions=1)):
        raise ValueError("R5-0 fixed diagnostic budget changed")
    if cfg["direction_scales"] != [.1, .25, .5] or cfg["primary_scale"] != .25:
        raise ValueError("R5-0 direction contract changed")


def preflight(path: str | Path, quick: bool = False) -> tuple[dict, dict]:
    path = _project_path(path).resolve()
    cfg = _load_yaml(path)
    validate_config(cfg)
    frozen = json.loads(_project_path(CONTRACT).read_text(encoding="utf-8"))
    if _relative(path) not in frozen:
        raise ValueError("configuration is not source-frozen")
    for name, digest in frozen.items():
        if _file_sha256(_project_path(name)) != digest:
            raise RuntimeError(f"frozen R5 source changed: {name}")
    output = _project_path(cfg["outputs"]["quick" if quick else "formal"]).resolve()
    expected = _project_path("outputs/s4_r5_physics_check_v1" + ("_quick" if quick else "")).resolve()
    if output != expected:
        raise ValueError("R5 output directory changed")
    if output.exists():
        raise FileExistsError(f"preserve existing R5 output: {output}")
    spec = cfg["quick" if quick else "formal"]
    seeds = [spec["seed_base"] + 4096*f+i for f in spec["family_indices"]
             for i in range(spec["weather_per_family"])]
    # 环境创新与功率噪声、观测噪声显式分离；仅扫描配置/元数据，不读旧测试轨迹。
    streams = {kind: [s+offset for s in seeds] for kind, offset in
               (("weather", 0), ("sensor", 50000000), ("power", 60000000))}
    flattened = [s for values in streams.values() for s in values]
    if len(flattened) != len(set(flattened)):
        raise ValueError("R5 random streams collide")
    historical: set[int] = set()
    paths = list(_project_path("configs").rglob("*.yaml"))
    for name in ("effective_config.json", "data_manifest.json", "preflight.json"):
        paths += list(_project_path("outputs").glob("*/"+name))
    exempt = {path, _project_path(cfg["design"]).resolve()}
    own_outputs = {_project_path(p).resolve() for p in cfg["outputs"].values()}
    for old in paths:
        if old.resolve() in exempt:
            continue
        obj = _load_yaml(old) if old.suffix == ".yaml" else json.loads(old.read_text(encoding="utf-8"))
        historical |= used_seed_blocks(obj, own_effective_config=(
            old.name == "effective_config.json" and old.parent.resolve() in own_outputs))
    # 同时保守展开历史中常用噪声/激励偏移；宁可停下审计，也不忽略潜在碰撞。
    expanded = {b+offset for b in historical for offset in (0, 5000, 6000, 7000)}
    if {s//10000 for s in flattened} & expanded:
        raise RuntimeError("R5 historical seed block collision")
    design = _load_yaml(_project_path(cfg["design"]))
    upstream = design["upstream"]
    if _file_sha256(_project_path(upstream["summary"])) != upstream["summary_sha256"]:
        raise RuntimeError("R4 upstream summary changed")
    if json.loads(_project_path(upstream["summary"]).read_text(encoding="utf-8"))["status"] != upstream["required_status"]:
        raise RuntimeError("R4 upstream status changed")
    device = resolve_device(cfg["runtime"]["device"])
    if device.type != "cuda":
        raise RuntimeError("CUDA required; no CPU fallback")
    parent = _load_yaml(_project_path(cfg["physical_source"]))
    profiles = _profiles(parent, cfg["quick_profile_ids"] if quick else parent["profile_ids"])
    count = len(seeds)*len(profiles)
    expected_steps = count*spec["steps"]*10 + len(spec["family_indices"])*len(profiles)*spec["gradient_directions"]*3*cfg["gradient_steps"]
    return cfg, dict(status="READY", quick=quick, output=str(output), device=str(device),
                     streams=streams, expected_forward_samples=expected_steps,
                     frozen_files=frozen, seed_audit="conservative_blocks_and_known_offsets",
                     **BOUNDARY)


class Progress:
    def __init__(self, output: Path, total: int, device: torch.device, timeout: float):
        self.output, self.device, self.total, self.timeout = output, device, total, timeout
        self.started = time.perf_counter()
        self.counts: dict[str, int] = defaultdict(int)
        self.backward_calls = 0
        self.bar = counted_progress(total=total, description="R5-0物理与梯度检查", unit="样本步")
        self.log = (output/"progress.jsonl").open("x", encoding="utf-8")

    def tick(self, stage: str, count: int = 1, error: float = 0.) -> None:
        self.counts[stage] += count
        self.bar.update(count)
        elapsed = time.perf_counter()-self.started
        if elapsed > self.timeout:
            raise TimeoutError("R5 diagnostic timeout; preserve output, no automatic retry")
        if self.bar.n % 20 == 0 or self.bar.n == self.total:
            self.log.write(json.dumps(dict(stage=stage, completed=self.bar.n, total=self.total,
                elapsed_seconds=elapsed, eta_seconds=elapsed/max(1,self.bar.n)*(self.total-self.bar.n),
                cuda_allocated_gb=torch.cuda.memory_allocated(self.device)/1024**3,
                maximum_alignment_error=error), ensure_ascii=False)+"\n")
            self.log.flush()
            update_progress(self.bar, device=self.device, metrics={"误差": error})

    def close(self) -> None:
        self.log.close()
        self.bar.close()


def make_environment(config, profile, basis, surrogate: bool) -> AdaptiveOpticsEnv:
    env = AdaptiveOpticsEnv(config, basis.device, profile.effects_config(), basis_override=basis)
    if surrogate:
        env.slm = DifferentiableSlm(config, profile.effects_config())
    return env


def parity(config, profile, basis, seed: int, tracker: Progress) -> dict:
    with torch.no_grad():
        old = make_environment(config, profile, basis, False)
        new = make_environment(config, profile, basis, True)
        raw1, _ = old.reset(seed=seed)
        raw2, _ = new.reset(seed=seed)
        errors: dict[str, float] = defaultdict(float)
        causal = True
        for t in range(config.episode_length):
            # 独立预置压力请求，不作为控制器性能或安全结果。
            request = torch.zeros_like(old.requested_modal)
            if t % 8 == 1: request[:, 10] = .14
            elif t % 8 == 2: request[:, 10] = -.14
            elif t % 8 >= 3: request[:] = .14 if (t//8) % 2 == 0 else -.14
            for raw in (raw1, raw2):
                poisoned = raw.clone(); poisoned[:, 21:] = float("nan")
                clean = simulation_residual_proxy(raw, generator=torch.Generator(device=basis.device).manual_seed(seed), noise_std_rad=0)
                dirty = simulation_residual_proxy(poisoned, generator=torch.Generator(device=basis.device).manual_seed(seed), noise_std_rad=0)
                interface = R4Interface(); view = interface.reset(clean, episode_id="causal-probe")
                causal = causal and torch.equal(clean, dirty) and bool(torch.isfinite(causal_policy_features(view.features, view.valid)).all())
            raw1, _, term1, trunc1, info1 = old.step(request)
            raw2, _, term2, trunc2, info2 = new.step(request)
            pairs = dict(observation=(raw1, raw2), actual_phase=(old.slm.current_phase, new.slm.current_phase),
                         queue=(old.slm.command_queue, new.slm.command_queue), turbulence=(old.turbulence_phase,new.turbulence_phase))
            for key in ("requested_modal", "applied_modal", "delayed_modal", "registered_modal",
                        "measured_power_in_bucket", "reward_power_in_bucket", "saturated_fraction", "slew_limited_fraction"):
                pairs[key] = info1[key], info2[key]
            for key, (left,right) in pairs.items():
                error = float((left-right).abs().max()) if left.numel() else 0.
                if not math.isfinite(error): raise RuntimeError(f"nonfinite parity: {key}")
                errors[key] = max(errors[key],error)
            if not torch.equal(term1,term2) or bool(trunc1.any()|trunc2.any()) or bool(term1.all()) != (t == config.episode_length-1):
                raise RuntimeError("termination timing mismatch")
            tracker.tick("B_parity",2,max(errors.values()))
        return dict(seed=seed, profile=profile.identifier, errors=dict(errors), causal_features_pass=causal)


def smooth_gradient(config, profile, basis, family: int, directions: int, cfg: dict, tracker: Progress) -> list[dict]:
    """受控光学数值探针，不把人工小相位当作正式湍流性能。"""
    config = replace(config, slm_quantization_levels=0)
    basis = basis.double()
    pupil = make_pupil_mask(config.grid_size, config.pupil_radius_fraction, basis.device, basis.dtype)
    records = []
    def evaluate(value):
        model = DifferentiableSlm(config, profile.effects_config())
        model.reset((1, config.grid_size, config.grid_size), basis.device, basis.dtype)
        scores = []
        previous = torch.zeros_like(value)
        for t in range(cfg["gradient_steps"]):
            request = torch.einsum("bm,mhw->bhw", .0125*value, basis[10:])
            phase, _ = model.step(request)
            disturbance = (.3*math.cos((family+1)*t*.03)*basis[10]+.2*basis[11])[None]
            power = focal_plane_metrics(disturbance+phase,pupil,config.bucket_radius_pixels)["power_in_bucket"]
            scores.append(measured_objective(power,value,previous,cfg["action_weight"],cfg["smooth_weight"]))
            previous = value
            tracker.tick("C_continuous")
        return torch.stack(scores).mean()
    for index in range(directions):
        direction = torch.cos(torch.arange(11,device=basis.device,dtype=basis.dtype)+index+.3)[None]
        direction = direction/direction.norm()
        value = torch.full((1,11),.2,device=basis.device,dtype=basis.dtype,requires_grad=True)
        grad = torch.autograd.grad(evaluate(value), value)[0]
        tracker.backward_calls += 1
        auto = (grad*direction).sum()
        epsilon = cfg["finite_difference_epsilon"]
        with torch.no_grad():
            fd = (evaluate(value.detach()+epsilon*direction)-evaluate(value.detach()-epsilon*direction))/(2*epsilon)
        error = float((auto-fd).abs())
        records.append(dict(family=family,profile=profile.identifier,direction=index,
            automatic=float(auto),finite_difference=float(fd),error=error,
            passed=math.isfinite(error) and error <= cfg["gradient_atol"]+cfg["gradient_rtol"]*abs(float(fd))))
    return records


def closed_return(config, profile, basis, seed: int, correction: torch.Tensor,
                  surrogate: bool, cfg: dict, tracker: Progress) -> tuple[torch.Tensor, dict]:
    env = make_environment(config,profile,basis,surrogate)
    raw, _ = env.reset(seed=seed)
    sensor = torch.Generator(device=basis.device).manual_seed(seed+50000000)
    proxy = lambda x: simulation_residual_proxy(x,generator=sensor,noise_std_rad=profile.observation_noise_std_rad)
    interface = R4Interface(calibration=NominalCalibration())
    interface.reset(proxy(raw),episode_id=f"weather-{seed}")
    rewards, powers = [], []
    safety = defaultdict(list)
    previous = torch.zeros_like(correction)
    for t in range(config.episode_length):
        view = interface.snapshot()
        causal_policy_features(view.features,view.valid)
        action = interface.issue(anchor_delta(view.features[:,-1],cfg["anchor"]),correction,step=t)
        raw, _, terminated, truncated, info = env.step(action.requested_delta_rad)
        if not torch.allclose(action.requested_modal_rad,info["requested_modal"],atol=1e-6,rtol=0):
            raise RuntimeError("requested action alignment failed")
        if bool(terminated.all()) != (t==config.episode_length-1) or bool(truncated.any()):
            raise RuntimeError("incomplete diagnostic episode")
        interface.observe_next(proxy(raw),step=t+1,power=PowerMeasurement(info["measured_power_in_bucket"],t,t+1))
        rewards.append(measured_objective(info["measured_power_in_bucket"],action.normalized_correction,
                                         previous,cfg["action_weight"],cfg["smooth_weight"]))
        previous = action.normalized_correction
        powers.append(info["reward_power_in_bucket"].detach())
        for key in ("violation_fraction","saturated_fraction","slew_limited_fraction"):
            safety[key].append(info[key].detach())
        tracker.tick("D_gradient" if surrogate else "D_hard")
    value = torch.stack(rewards).mean()
    if not bool(torch.isfinite(value)): raise RuntimeError("nonfinite return")
    return value, dict(power=float(torch.stack(powers).mean()),
                       **{k:float(torch.stack(v).mean()) for k,v in safety.items()})


def direction_summary(records: list[dict], minimum_positive: int, cfg: dict) -> dict:
    weather, profiles = defaultdict(list), defaultdict(list)
    versus_zero, safety = [], []
    key = str(cfg["primary_scale"])
    for row in records:
        branches = row["branches"]
        positive, negative, zero = branches[key], branches["-"+key], branches["0.0"]
        difference = positive["objective"]-negative["objective"]
        weather[row["seed"]].append(difference)
        profiles[row["profile"]].append(difference)
        versus_zero.append(positive["objective"]-zero["objective"])
        safety.append(all(positive[k]-zero[k] <= cfg["safety_increase_maximum"]
                          for k in ("violation_fraction","saturated_fraction","slew_limited_fraction")))
    average = lambda x: sum(x)/len(x)
    count = sum(average(v)>0 for v in weather.values())
    profile_means = {k:average(v) for k,v in profiles.items()}
    passed = (count>=minimum_positive and all(v>=0 for v in profile_means.values())
              and average(versus_zero)>0 and all(safety))
    return dict(passed=passed,positive_weather=count,total_weather=len(weather),
                profile_mean_signed_differences=profile_means,mean_positive_vs_zero=average(versus_zero),
                all_paired_safety_checks=all(safety),ties=sum(average(v)==0 for v in weather.values()))


def run(path: str | Path, *, quick: bool = False, preflight_only: bool = False) -> dict:
    cfg, report = preflight(path,quick)
    if preflight_only:
        return {k:v for k,v in report.items() if k!="frozen_files"}
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = resolve_device("cuda")
    output = Path(report["output"]); output.mkdir(parents=True,exist_ok=False)
    write_json(output/"preflight.json",report)
    write_json(output/"effective_config.json",cfg)
    write_json(output/"source_manifest.json",report["frozen_files"])
    tracker = Progress(output,report["expected_forward_samples"],device,cfg["timeout_seconds"])
    try:
        spec = cfg["quick" if quick else "formal"]
        parent = _load_yaml(_project_path(cfg["physical_source"]))
        base, _ = load_s1_config(_project_path(parent["environment_config"]))
        base = replace(base,num_modes=21,batch_size=1,episode_length=spec["steps"])
        basis, _, _ = build_action_basis(base,ActionRepresentation("r5_zernike21","zernike",21),device)
        profiles = _profiles(parent,cfg["quick_profile_ids"] if quick else parent["profile_ids"])
        alignments, gradients, directions = [], [], []
        for f in spec["family_indices"]:
            family = parent["families"][f]
            condition = RobustnessCondition.from_mapping(dict(family,base_seed=spec["seed_base"]+4096*f))
            for profile in profiles:
                config = profile.environment_config(condition.environment_config(base))
                gradients += smooth_gradient(config,profile,basis,f,spec["gradient_directions"],cfg,tracker)
                write_json(output/"continuous_gradients.json",gradients)
                if not all(x["passed"] for x in gradients):
                    raise RuntimeError("continuous gradient check failed; do not retry automatically")
                for i in range(spec["weather_per_family"]):
                    seed = spec["seed_base"]+4096*f+i
                    alignment = parity(config,profile,basis,seed,tracker)
                    alignments.append(alignment)
                    write_json(output/"forward_alignment.json",alignments)
                    if not alignment["causal_features_pass"] or max(alignment["errors"].values())>cfg["forward_atol"]:
                        raise RuntimeError("causal/forward alignment failed; preserve diagnostic")
                    correction = torch.zeros(1,11,device=device,requires_grad=True)
                    value, _ = closed_return(config,profile,basis,seed,correction,True,cfg,tracker)
                    grad = torch.autograd.grad(value,correction)[0]
                    tracker.backward_calls += 1
                    if not bool(torch.isfinite(grad).all()) or float(grad.abs().max())<=cfg["gradient_zero_threshold"]:
                        raise RuntimeError("surrogate gradient nonfinite or uninformative; stop")
                    direction = (grad/grad.abs().max()).detach()
                    branches = {}
                    with torch.no_grad():
                        for scale in [0.]+[s*sign for s in cfg["direction_scales"] for sign in (1,-1)]:
                            score, metrics = closed_return(config,profile,basis,seed,direction*scale,False,cfg,tracker)
                            branches[str(scale)] = dict(objective=float(score),**metrics)
                    directions.append(dict(seed=seed,family=f,profile=profile.identifier,
                        gradient=grad.detach().cpu().tolist(),branches=branches))
                    write_json(output/"hard_direction_checks.json",directions)
        analysis = direction_summary(directions,1 if quick else 9,cfg)
        if sum(tracker.counts.values())!=report["expected_forward_samples"]:
            raise RuntimeError("diagnostic sample accounting mismatch")
        for name,digest in report["frozen_files"].items():
            if _file_sha256(_project_path(name))!=digest: raise RuntimeError(f"source changed during run: {name}")
        result = dict(status="QUICK_CHECK_COMPLETED_NOT_A_GATE" if quick else
                      ("R5_0_CHECK_PASS" if analysis["passed"] else "R5_0_DIRECTION_FAIL"),
            quick=quick,checks=dict(causal_feature_boundary=True,hard_forward_parity=True,
                continuous_gradient=True,hard_direction=analysis),
            counters=dict(forward_samples=dict(tracker.counts),backward_calls=tracker.backward_calls),
            elapsed_seconds=time.perf_counter()-tracker.started,gpu=torch.cuda.get_device_name(device),
            training_information="privileged_physics_gradient_diagnostic_not_deployable_controller",
            policy_leakage_verified=False, policy_leakage_note="R5 policy not implemented; only feature boundary tested",
            material_passport=dict(origin_skill="academic-research-suite",origin_mode="run",
                verification_status="UNVERIFIED",version_label="r5_0_physics_check_v1"),
            **BOUNDARY,next_action="Stop for read-only audit; no automatic training or retry.")
        write_json(output/"summary.json",result)
        return result
    except Exception as exc:
        write_json(output/"failure.json",dict(error=str(exc),traceback=traceback.format_exc(),
            counters=dict(tracker.counts),backward_calls=tracker.backward_calls,automatic_retry=False))
        raise
    finally:
        tracker.close()
