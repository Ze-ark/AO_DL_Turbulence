"""R5-1：在独立开发天气上筛选传统预测基座，不训练策略。"""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
import json, os, shutil, time, traceback
import hashlib
from pathlib import Path
import torch
from dataclasses import replace
from src.rl.r4_baselines import BaselineSpec, baseline_delta
from src.rl.r4_control import NominalCalibration
from src.rl.r4_dynamics import ARXDynamics
from src.rl.r4_dynamics_experiment import Progress, safe_git_record
from src.rl.r4_interface_smoke import write_json
from src.rl.r4_observation import PowerMeasurement, R4Interface, simulation_residual_proxy
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _profiles, _project_path, _relative
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.robust_control import RobustnessCondition

METRICS = ("reward_power_in_bucket", "reward_strehl", "violation_fraction",
           "reward_phase_rmse", "saturated_fraction", "slew_limited_fraction")

def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def _write_artifact_closure(out: Path) -> str:
    artifacts = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.json", "SUCCESS.json"}:
            artifacts[_relative(path)] = _file_sha256(path)
    write_json(out / "artifact_manifest.json", artifacts)
    return _file_sha256(out / "artifact_manifest.json")

def _candidates(cfg: dict) -> list[BaselineSpec]:
    result = [BaselineSpec(**item) for item in cfg["candidate_order"]]
    if len(result) != 9 or len(set(result)) != 9:
        raise ValueError("R5-1 requires nine unique frozen candidates")
    for item in result: item.validate()
    return result

def _select(means: torch.Tensor, reference: int, max_increase: float) -> dict:
    average = means.double().mean(2).mean(2).mean(1)
    increase = average[:, 2] - average[reference, 2]
    eligible = increase <= max_increase
    eligible[reference] = True
    power = torch.where(eligible, average[:, 0], torch.full_like(average[:, 0], -torch.inf))
    selected = int(torch.argmax(power)) if bool(eligible.any()) else None
    return dict(status="BASELINE_SELECTED_REQUIRES_READ_ONLY_AUDIT" if selected is not None else "NO_ELIGIBLE_BASELINE",
                selected_index=selected, reference_index=reference, eligible=eligible.tolist(),
                violation_increase=increase.tolist(), candidate_means=average.tolist(),
                interpretation="development_selection_not_independent_confirmation")

def _preflight(path: str | Path, quick: bool) -> tuple[dict, dict, dict]:
    cfg = _load_yaml(_project_path(path))
    if cfg.get("stage") != "S4-D2-R5-1" or cfg.get("runtime") != {"device":"cuda", "formal_owner":"user_ide", "automatic_retry":False}:
        raise ValueError("R5-1 frozen CUDA configuration required")
    upstream = _project_path(cfg["upstream"])
    summary = _read(upstream / "summary.json")
    if _file_sha256(upstream / "summary.json") != cfg["upstream_summary_sha256"] or summary.get("status") != "R5_0_CHECK_PASS" and summary.get("status") != "R5_0_CHECK_PASS":
        raise RuntimeError("R5-0 prerequisite is missing or changed")
    if cfg["boundary"] != {"no_r5_confirm_weather":True, "no_policy_training":True, "no_online_identification":True, "no_hardware_actions":True}:
        raise ValueError("R5-1 boundary changed")
    specs = _candidates(cfg)
    p = _load_yaml(_project_path(cfg["parent"]))
    d = cfg["development"]
    n = cfg["quick"]["weather_per_family"] if quick else d["weather_per_family"]
    steps = cfg["quick"]["steps"] if quick else d["steps"]
    batch = cfg["quick"]["batch_size"] if quick else d["batch_size"]
    profiles = cfg["quick"]["profile_ids"] if quick else d["profile_ids"]
    starts = [d["seed_base"] + x for x in d["family_offsets"]]
    used = {s+i for s in starts for i in range(n)}
    reserved_r50 = set(range(5200000, 5210000))
    if used & reserved_r50 or n % batch or len(profiles) not in (2, 6):
        raise ValueError("R5-1 seed or batch contract violation")
    physical = len(specs) * len(p["families"]) * n * len(profiles) * steps
    expected = cfg["budget"]["quick_physical_transitions"] if quick else cfg["budget"]["formal_physical_transitions"]
    if physical != expected: raise ValueError(f"frozen budget mismatch: {physical} != {expected}")
    out = _project_path(cfg["quick_directory"] if quick else cfg["output_directory"])
    if out.exists(): raise FileExistsError(f"preserve existing R5-1 output: {out}")
    if shutil.disk_usage(_project_path(".")).free < (128 if quick else 2048)*1024**2: raise RuntimeError("insufficient disk")
    device = resolve_device("cuda")
    settings = dict(parent=p, specs=specs, starts=starts, weather=n, steps=steps, batch=batch, profiles=profiles)
    return cfg, settings, dict(status="READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_IDE", quick=quick,
        device=str(device), physical_transitions=physical, candidates=len(specs), weather=n*3,
        training_updates=0, confirmation_access=False, real_slm_actions=False, s4d3_access=False)

@torch.no_grad()
def _execute(cfg: dict, s: dict, report: dict, out: Path) -> dict:
    device = resolve_device("cuda"); parent = s["parent"]
    base, _ = load_s1_config(_project_path(parent["environment_config"]))
    base = replace(base, num_modes=21, batch_size=s["batch"], episode_length=s["steps"])
    basis, _, _ = build_action_basis(base, ActionRepresentation("r5_zernike21", "zernike", 21), device)
    cal = NominalCalibration(**parent["nominal_calibration"]); linear=[]
    for i in range(3):
        state=torch.load(_project_path(f"{cfg['predictor_source']}/linear_{i}.pt"), map_location=device, weights_only=True)["state_dict"]
        model=ARXDynamics(state["x_mean"],state["x_scale"],state["y_mean"],state["y_scale"]).to(device); model.load_state_dict(state); linear.append(model.eval().requires_grad_(False))
    means=torch.zeros(9,3,len(s["profiles"]),s["weather"],len(METRICS),device=device,dtype=torch.float64)
    progress=Progress(out,device); profiles=_profiles(parent,s["profiles"]); physical=0
    write_json(out/"candidates.json", [asdict(x) for x in s["specs"]]); write_json(out/"seeds.json", {"family_starts":s["starts"],"weather_per_family":s["weather"],"split":"r5_1_development"})
    progress.phase("R5-1传统基座筛选", report["physical_transitions"]//s["batch"])
    try:
        for ci,spec in enumerate(s["specs"]):
            for fi,family in enumerate(parent["families"]):
                for pi,profile in enumerate(profiles):
                    for offset in range(0,s["weather"],s["batch"]):
                        seeds=[s["starts"][fi]+offset+j for j in range(s["batch"])]
                        condition=RobustnessCondition.from_mapping(dict(family,base_seed=seeds[0]))
                        env=AdaptiveOpticsEnv(profile.environment_config(condition.environment_config(base)),device,profile.effects_config(),basis_override=basis)
                        raw,_=env.reset(seed=seeds[0]); sensor=torch.Generator(device=device).manual_seed(seeds[0]+parent["data"]["sensor_seed_offset"])
                        proxy=lambda x: simulation_residual_proxy(x,generator=sensor,noise_std_rad=profile.observation_noise_std_rad)
                        interface=R4Interface(calibration=cal); interface.reset(proxy(raw),episode_id=str(seeds[0])); rec={k:[] for k in METRICS}
                        for step in range(s["steps"]):
                            snap=interface.snapshot(); delta,_=baseline_delta(spec,snap.features,snap.valid,linear,cal); action=interface.issue(delta,delta.new_zeros(s["batch"],11),step=step)
                            raw,_,terminated,truncated,info=env.step(action.requested_delta_rad); physical+=s["batch"]
                            interface.observe_next(proxy(raw),step=step+1,power=PowerMeasurement(info["measured_power_in_bucket"],step,step+1))
                            if bool(truncated.any()) or not bool((terminated==(step==s["steps"]-1)).all()): raise RuntimeError("incomplete R5-1 episode")
                            for k in METRICS: rec[k].append(info[k].cpu())
                            progress.tick({"候选":ci+1,"候选总数":9,"物理转移":physical})
                        vals=torch.stack([torch.stack(rec[k],1).to(device).double().mean(1) for k in METRICS],-1)
                        means[ci,fi,pi,offset:offset+s["batch"]]=vals
        if physical != report["physical_transitions"]: raise RuntimeError("R5-1 budget mismatch after execution")
        selection={} if report["quick"] else _select(means,cfg["reference_index"],cfg["violation_increase_max"])
        torch.save({"means":means.cpu(),"metric_order":METRICS},out/"candidate_metrics.pt")
        write_json(out/"selection.json",selection)
        result=dict(status="QUICK_SMOKE_ONLY" if report["quick"] else selection["status"],selection=selection,physical_transitions=physical,candidates=9,training_updates=0,confirmation_access=False,real_slm_actions=False,s4d3_access=False,material_passport=dict(origin_skill="academic-research-suite / experiment-agent",origin_mode="run",origin_date=datetime.now(timezone.utc).isoformat(),verification_status="UNVERIFIED",version_label="r5_1_baseline_selection_v1"),next_action="停止等待只读审计；不自动训练策略")
        manifest_sha = _write_artifact_closure(out)
        result["artifact_manifest_sha256"] = manifest_sha
        result["next_action"] = "停止等待只读审计；不自动训练策略"
        write_json(out/"summary.json",result)
        write_json(out/"SUCCESS.json", {"summary_sha256": _file_sha256(out/"summary.json"), "artifact_manifest_sha256": manifest_sha})
        progress.close(); return result
    except Exception:
        write_json(out/"failure.json",dict(traceback=traceback.format_exc(),automatic_retry=False)); raise
    finally: progress.close()

def run(config_path: str | Path, *, quick=False, preflight_only=False) -> dict:
    cfg,s,report=_preflight(config_path,quick)
    if preflight_only:return report
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8"); torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False
    out=_project_path(cfg["quick_directory"] if quick else cfg["output_directory"]); out.mkdir(parents=True,exist_ok=False); write_json(out/"preflight.json",report); write_json(out/"config.json",cfg); write_json(out/"runtime.json",dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),git=safe_git_record()))
    return _execute(cfg,s,report,out)

