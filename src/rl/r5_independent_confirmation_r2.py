"""R5 固定 1.25 倍残差的全新天气确认；正式运行由用户在 IDE 启动。"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import statistics
import traceback

import torch

from src.rl.r4_dynamics_experiment import Progress, safe_git_record
from src.rl.r4_interface_smoke import write_json
from src.rl.r5_margin_development import _rollout, effective_stream_seeds
from src.rl.r5_policy_training import ResidualGRUPolicy
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _profiles, _project_path
from src.runtime import resolve_device
from src.simulation.config import load_s1_config


ROOT = Path(__file__).resolve().parents[2]
METRICS = ("power", "strehl", "phase_rmse", "violation", "saturation", "slew_limited")


def source_bundle_sha256() -> str:
    """冻结所有仿真/控制源码和本入口；与有无 Git 提交无关。"""
    paths = sorted((ROOT / "src").rglob("*.py"))
    paths.append(ROOT / "scripts" / "confirm_s4_r5_policy_r2.py")
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest()


def development_choice(records: list[dict], expected_weather: int = 32) -> dict:
    """只从 D1 全部三份模型选择一个统一倍率，绝不按档位或模型挑选。"""
    families = ("frozen", "boiling", "varying")
    profiles = ("nominal", "delay_3", "settling_050", "registration_moderate",
                "registration_severe", "combined_moderate")
    expected = 10 * expected_weather * len(families) * len(profiles)
    keys = [(r["controller"], r["family"], r["profile"], r["weather_seed"])
            for r in records]
    if len(records) != expected or len(set(keys)) != expected:
        raise RuntimeError("D1 records incomplete or duplicated")
    by_key = dict(zip(keys, records))
    weather = sorted({r["weather_seed"] for r in records})
    if weather != [5240000 + 10 * i for i in range(expected_weather)]:
        raise RuntimeError("D1 weather layout changed")
    baseline = [by_key[("integrator", family, profile, seed)]
                for family in families for profile in profiles for seed in weather]
    baseline_power = statistics.fmean(row["power"] for row in baseline)
    results = {}
    for scale in (0.75, 1.0, 1.25):
        paired = [(by_key[(f"policy_{member}_scale_{scale}", family, profile, seed)],
                   by_key[("integrator", family, profile, seed)])
                  for member in range(3) for family in families
                  for profile in profiles for seed in weather]
        delta = {metric: statistics.fmean(a[metric] - b[metric] for a, b in paired)
                 for metric in METRICS}
        safe = (delta["strehl"] >= 0 and delta["phase_rmse"] <= 0
                and all(delta[metric] <= 0.001 for metric in
                        ("violation", "saturation", "slew_limited")))
        results[scale] = {"relative_power_gain": delta["power"] / baseline_power,
                          "safe_on_development": safe}
    eligible = [scale for scale, result in results.items()
                if result["safe_on_development"]]
    if not eligible:
        raise RuntimeError("no development scale satisfies the safety screen")
    selected = max(eligible, key=lambda scale: results[scale]["relative_power_gain"])
    if selected != 1.25:
        raise RuntimeError("frozen D1 scale selection changed")
    return {"selected_scale": selected, "baseline_power": baseline_power,
            "scale_results": {str(scale): result for scale, result in results.items()}}


def preflight(path: str | Path, quick: bool) -> tuple[dict, dict, Path, torch.device]:
    cfg = _load_yaml(_project_path(path))
    if (cfg["stage"] != "S4-D2-R5-3-R2" or cfg["selected_scale"] != 1.25
            or cfg["runtime"] != {"device": "cuda", "formal_owner": "user_ide",
                                  "automatic_retry": False}
            or cfg["data"] != {"seed_base": 5700000, "seed_stride": 10,
                               "weather_count": 32, "episode_length": 200}
            or cfg["quick"] != {"seed_base": 5710000, "weather_count": 1,
                                "episode_length": 16,
                                "profile_ids": ["nominal", "combined_moderate"]}
            or cfg["statistics"] != {"bootstrap_seed": 5723456,
                                      "bootstrap_repeats": 20000}
            or cfg["thresholds"] != {"relative_power_gain": 0.01,
                                      "maximum_safety_increase": 0.001}
            or cfg["boundary"] != {"training_updates": 0, "real_slm_actions": False,
                                    "automatic_retry": False, "old_confirmation_access": False}):
        raise ValueError("R5-R2 frozen confirmation contract changed")
    if source_bundle_sha256() != cfg["source_bundle_sha256"]:
        raise RuntimeError("confirmation source bundle changed")
    parent_path = _project_path(cfg["parent"])
    if _file_sha256(parent_path) != cfg["parent_sha256"]:
        raise RuntimeError("R5 training configuration changed")
    parent = _load_yaml(parent_path)
    training = _project_path(cfg["training_output"])
    training_summary = training / "summary.json"
    if (_file_sha256(training_summary) != cfg["training_summary_sha256"]
            or json.loads(training_summary.read_text(encoding="utf-8"))["status"]
            != "R5_2_TRAINING_COMPLETE_REQUIRES_AUDIT"):
        raise RuntimeError("R5 training summary changed")
    if set(cfg["checkpoints"]) != {f"policy_{i}_02000.pt" for i in range(3)}:
        raise RuntimeError("three frozen final policy members required")
    for name, digest in cfg["checkpoints"].items():
        if _file_sha256(training / "checkpoints" / name) != digest:
            raise RuntimeError(f"R5 checkpoint changed: {name}")
    development = _project_path(cfg["development_output"])
    for name, digest in cfg["development_hashes"].items():
        if _file_sha256(development / name) != digest:
            raise RuntimeError(f"D1 evidence changed: {name}")
    dev_summary = json.loads((development / "summary.json").read_text(encoding="utf-8"))
    if (dev_summary["status"] != "DEVELOPMENT_DIAGNOSTIC_REQUIRES_AUDIT"
            or dev_summary["confirmation_access"] is not False
            or dev_summary["training_updates"] != 0):
        raise RuntimeError("D1 is not a development-only source")
    records = [json.loads(line) for line in
               (development / "records.jsonl").open(encoding="utf-8")]
    choice = development_choice(records)
    if choice["selected_scale"] != cfg["selected_scale"]:
        raise RuntimeError("predeclared global scale disagrees with D1")
    spec = cfg["quick"] if quick else cfg["data"]
    expected_seed_base = 5710000 if quick else 5700000
    count, steps = int(spec["weather_count"]), int(spec["episode_length"])
    if (int(spec["seed_base"]) != expected_seed_base
            or int(cfg["data"]["seed_stride"]) != 10
            or (count, steps) != ((1, 16) if quick else (32, 200))):
        raise ValueError("confirmation seed or episode contract changed")
    families = parent["families"][:1] if quick else parent["families"]
    profiles = _profiles(parent, spec["profile_ids"] if quick else parent["profile_ids"])
    streams = effective_stream_seeds(expected_seed_base, count, 10,
                                     len(profiles), len(families))
    if len(families) != (1 if quick else 3) or len(profiles) != (2 if quick else 6):
        raise ValueError("confirmation family/profile layout changed")
    # 旧训练、D1 与两次已查看确认均在 5.7M 以下；传感器/功率流使用不同大偏移。
    if any(seed < 5700000 or seed >= 5720000 for seed in streams):
        raise ValueError("confirmation seed namespace changed")
    output = _project_path(cfg["quick_directory"] if quick else cfg["output_directory"])
    if output.exists():
        raise FileExistsError(f"preserve confirmation output: {output}")
    device = resolve_device("cuda")
    report = {"status": "READY_FOR_DIAGNOSTIC" if quick else "READY_FOR_USER_IDE",
              "quick": quick, "device": str(device), "weather_count": count,
              "families": len(families), "profiles": len(profiles), "controllers": 4,
              "episode_length": steps, "unique_turbulence_streams": len(streams),
              "physical_transitions": 4 * count * len(families) * len(profiles) * steps,
              "selected_scale": cfg["selected_scale"], "development_choice": choice,
              **cfg["boundary"]}
    return cfg, report, output, device


def summarize(rows: list[dict], cfg: dict, device: torch.device) -> dict:
    families = ("frozen", "boiling", "varying")
    profiles = ("nominal", "delay_3", "settling_050", "registration_moderate",
                "registration_severe", "combined_moderate")
    weather = [5700000 + 10 * i for i in range(32)]
    controllers = ("integrator", "policy_0_scale_1.25", "policy_1_scale_1.25",
                   "policy_2_scale_1.25")
    by_key = {(r["controller"], r["family"], r["profile"], r["weather_seed"]): r
              for r in rows}
    if len(rows) != 2304 or len(by_key) != 2304:
        raise RuntimeError("confirmation rows incomplete or duplicated")
    values = {}
    for metric in METRICS:
        values[metric] = torch.tensor(
            [[[[by_key[(controller, family, profile, seed)][metric]
                for profile in profiles] for seed in weather] for family in families]
             for controller in controllers], device=device, dtype=torch.float64)
    power = values["power"]
    baseline = power[0].mean()
    delta = power[1:] - power[0]
    difference = delta.mean()
    family_weather = delta.mean(dim=(0, 3))
    repeats = int(cfg["statistics"]["bootstrap_repeats"])
    if repeats != 20000:
        raise ValueError("bootstrap budget changed")
    generator = torch.Generator(device=device).manual_seed(int(cfg["statistics"]["bootstrap_seed"]))
    indices = torch.randint(32, (repeats, 3, 32), generator=generator, device=device)
    draws = torch.gather(family_weather[None].expand(repeats, -1, -1), 2, indices)
    ci = torch.quantile(draws.mean(dim=(1, 2)),
                        torch.tensor([0.025, 0.975], device=device, dtype=torch.float64))
    other = {metric: float((value[1:] - value[0]).mean())
             for metric, value in values.items() if metric != "power"}
    member_gain = delta.mean(dim=(1, 2, 3))
    family_gain = delta.mean(dim=(0, 2, 3))
    profile_gain = delta.mean(dim=(0, 1, 2))
    family_profile_gain = delta.mean(dim=(0, 2))
    relative = float(difference / baseline)
    threshold = cfg["thresholds"]
    gates = {
        "mean_relative_power_at_least_1pct": relative >= threshold["relative_power_gain"],
        "paired_ci_lower_positive": float(ci[0]) > 0,
        "each_initialization_positive": bool((member_gain > 0).all()),
        "each_family_positive": bool((family_gain > 0).all()),
        "each_profile_positive": bool((profile_gain > 0).all()),
        "strehl_not_lower": other["strehl"] >= 0,
        "phase_rmse_not_higher": other["phase_rmse"] <= 0,
        "violation_within_limit": other["violation"] <= threshold["maximum_safety_increase"],
        "saturation_within_limit": other["saturation"] <= threshold["maximum_safety_increase"],
        "slew_within_limit": other["slew_limited"] <= threshold["maximum_safety_increase"],
    }
    return {"baseline_power": float(baseline), "policy_power": float(power[1:].mean()),
            "absolute_power_gain": float(difference), "relative_power_gain": relative,
            "absolute_power_gain_ci95": [float(x) for x in ci],
            "member_absolute_gains": [float(x) for x in member_gain],
            "family_absolute_gains": dict(zip(families, [float(x) for x in family_gain])),
            "profile_absolute_gains": dict(zip(profiles, [float(x) for x in profile_gain])),
            "positive_family_profile_cells": int((family_profile_gain > 0).sum()),
            "other_deltas": other, "preliminary_gates": gates,
            "preliminary_all_gates": all(gates.values()),
            "bootstrap_unit": "weather_within_family; profiles_and_initializations_paired"}


def run(path: str | Path = "configs/experiments/s4_r5_independent_confirmation_r2.yaml",
        *, quick: bool = False, preflight_only: bool = False) -> dict:
    cfg, report, output, device = preflight(path, quick)
    if preflight_only:
        return report
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "preflight.json", report)
    write_json(output / "config.json", cfg)
    write_json(output / "runtime.json", {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                                         "gpu": torch.cuda.get_device_name(), "git": safe_git_record(),
                                         "source_bundle_sha256": source_bundle_sha256()})
    progress = Progress(output, device)
    try:
        parent = _load_yaml(_project_path(cfg["parent"]))
        spec = cfg["quick"] if quick else cfg["data"]
        families = parent["families"][:1] if quick else parent["families"]
        profiles = _profiles(parent, spec["profile_ids"] if quick else parent["profile_ids"])
        base, _ = load_s1_config(_project_path(parent["environment_config"]))
        base = replace(base, num_modes=21, batch_size=1, episode_length=report["episode_length"])
        basis, _, _ = build_action_basis(base, ActionRepresentation("r5_zernike21", "zernike", 21), device)
        policies = []
        for member in range(3):
            checkpoint = torch.load(_project_path(cfg["training_output"]) / "checkpoints"
                                    / f"policy_{member}_02000.pt", map_location=device, weights_only=True)
            if checkpoint["init"] != member or checkpoint["update"] != 2000:
                raise RuntimeError("R5 checkpoint identity mismatch")
            policy = ResidualGRUPolicy(parent["policy"]["hidden_size"],
                                       parent["policy"]["output_size"]).to(device)
            policy.load_state_dict(checkpoint["state_dict"])
            policy.eval()
            policies.append(policy)
        branches = [("integrator", 0.0, None)] + [
            (f"policy_{member}_scale_1.25", cfg["selected_scale"], policy)
            for member, policy in enumerate(policies)]
        progress.phase("R5-R2确认快速冒烟" if quick else "R5-R2全新天气独立确认",
                       4 * report["weather_count"] * report["episode_length"])
        rows = []
        with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
            for label, scale, policy in branches:
                for weather_index in range(report["weather_count"]):
                    seed = int(spec["seed_base"]) + 10 * weather_index
                    batch = _rollout(seed, label, float(scale), policy, report["episode_length"],
                                     basis, base, families, profiles,
                                     parent["data"]["sensor_seed_offset"], progress)
                    for row in batch:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows.extend(batch)
                    handle.flush()
        expected = 4 * report["weather_count"] * report["families"] * report["profiles"]
        if len(rows) != expected:
            raise RuntimeError("confirmation record count mismatch")
        analysis = {} if quick else summarize(rows, cfg, device)
        result = {"status": "QUICK_SMOKE_NO_CONCLUSION" if quick
                  else "R5_R2_CONFIRMATION_COMPLETE_REQUIRES_AUDIT",
                  "records": len(rows), "controllers": [x[0] for x in branches],
                  "weather_count": report["weather_count"],
                  "physical_transitions": report["physical_transitions"],
                  "analysis": analysis, "confirmation_access": not quick,
                  "material_passport": {"origin_skill": "academic-research-suite / experiment-agent",
                                        "origin_mode": "run", "verification_status": "UNVERIFIED"},
                  **cfg["boundary"],
                  "next_action": "停止并等待只读审计；预计算门槛不等于最终结论"}
        write_json(output / "summary.json", result)
        write_json(output / "SUCCESS.json", {
            "summary_sha256": _file_sha256(output / "summary.json"),
            "records_sha256": _file_sha256(output / "records.jsonl"),
            "progress_sha256": _file_sha256(output / "progress.jsonl")})
        return result
    except Exception:
        write_json(output / "failure.json", {"traceback": traceback.format_exc(),
                                             "automatic_retry": False})
        raise
    finally:
        progress.close()
