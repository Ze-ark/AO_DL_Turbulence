"""R5-R3 冻结 1.75 倍策略的新天气独立确认；正式运行仅由用户在 IDE 启动。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_s4_r5_margin_d2 import (
    CorrectionClampTelemetry, FAMILIES, PROFILES, summarize_development,
)
from src.rl.r4_dynamics_experiment import Progress, safe_git_record
from src.rl.r4_interface_smoke import write_json
from src.rl.r5_independent_confirmation_r2 import METRICS, source_bundle_sha256
from src.rl.r5_margin_development import _rollout, effective_stream_seeds
from src.rl.r5_policy_training import ResidualGRUPolicy
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _profiles, _project_path
from src.runtime import resolve_device
from src.simulation.config import load_s1_config


CONFIG = "configs/experiments/s4_r5_independent_confirmation_r3.yaml"
SCALE = 1.75
FORMAL_BASE = 6_300_000
QUICK_BASE = 6_310_000
FROZEN_SOURCE = "ccdc31faaa155361d8bd3e19ffc5bb82705f425a0ef4c7f58dd7f70831efd956"


def stream_manifest(quick: bool) -> dict[str, list[int]]:
    base, count, profiles, families = (QUICK_BASE, 1, 2, 1) if quick else (FORMAL_BASE, 64, 6, 3)
    weather = [base + 10 * i for i in range(count)]
    turbulence = effective_stream_seeds(base, count, 10, profiles, families)
    sensor = [seed + profile * 1000 + 50_000_000 for seed in weather for profile in range(profiles)]
    power = [seed + profile * 1000 + 60_000_000 for seed in weather for profile in range(profiles)]
    if (len(turbulence) != count * profiles * families
            or len(sensor) != len(set(sensor)) or len(power) != len(set(power))
            or not (6_300_000 <= min(turbulence) <= max(turbulence) < 6_310_000) and not quick
            or quick and not (6_310_000 <= min(turbulence) <= max(turbulence) < 6_320_000)):
        raise RuntimeError("R3 随机流边界错误")
    return {"weather_bases": weather, "turbulence": turbulence, "sensor": sensor, "power": power}


def _contract(cfg: dict) -> None:
    if (cfg.get("stage") != "S4-D2-R5-3-R3"
            or cfg.get("selected_scale") != SCALE
            or cfg.get("source_bundle_sha256") != FROZEN_SOURCE
            or cfg.get("runtime") != {"device": "cuda", "formal_owner": "user_ide", "automatic_retry": False}
            or cfg.get("data") != {"seed_base": FORMAL_BASE, "seed_stride": 10,
                                   "weather_count": 64, "episode_length": 200}
            or cfg.get("quick") != {"seed_base": QUICK_BASE, "weather_count": 1,
                                    "episode_length": 16,
                                    "profile_ids": ["nominal", "combined_moderate"]}
            or cfg.get("statistics") != {"bootstrap_seed": 6_323_456, "bootstrap_repeats": 20_000}
            or cfg.get("thresholds") != {"relative_power_gain": .01,
                                         "maximum_safety_increase": .001}
            or cfg.get("boundary") != {"training_updates": 0, "real_slm_actions": False,
                                       "automatic_retry": False, "old_confirmation_access": False}
            or cfg.get("output_directory") != "outputs/s4_r5_independent_confirmation_r3_v1"
            or cfg.get("quick_directory") != "outputs/s4_r5_independent_confirmation_r3_v1_quick"):
        raise ValueError("R3 冻结确认合同被修改")


def _validate_development(cfg: dict) -> dict:
    if _file_sha256(_project_path(cfg["development_config"])) != cfg["development_config_sha256"]:
        raise RuntimeError("D2 配置已改变")
    if _file_sha256(_project_path("scripts/diagnose_s4_r5_margin_d2.py")) != cfg["development_entry_sha256"]:
        raise RuntimeError("D2 入口已改变")
    d2cfg = _load_yaml(_project_path(cfg["development_config"]))
    if (d2cfg["parent"] != cfg["parent"] or d2cfg["parent_sha256"] != cfg["parent_sha256"]
            or d2cfg["training_output"] != cfg["training_output"]
            or d2cfg["checkpoints"] != cfg["checkpoints"]):
        raise RuntimeError("D2 与冻结训练来源不一致")
    root = _project_path(cfg["development_output"])
    if set(cfg["development_hashes"]) != {"summary.json", "records.jsonl", "stream_manifest.json", "SUCCESS.json"}:
        raise RuntimeError("D2 证据清单不完整")
    for name, digest in cfg["development_hashes"].items():
        if _file_sha256(root / name) != digest:
            raise RuntimeError(f"D2 证据已改变: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    success = json.loads((root / "SUCCESS.json").read_text(encoding="utf-8"))
    if (success != {"summary_sha256": cfg["development_hashes"]["summary.json"]}
            or summary["records_sha256"] != cfg["development_hashes"]["records.jsonl"]
            or summary["stream_manifest_sha256"] != cfg["development_hashes"]["stream_manifest.json"]
            or summary["status"] != "DEVELOPMENT_ONLY_REQUIRES_AUDIT"
            or summary["confirmation_access"] is not False
            or summary["training_updates"] != 0):
        raise RuntimeError("D2 完整性或信息隔离不成立")
    with (root / "records.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    recomputed = summarize_development(rows, d2cfg, quick=False)
    if (summary["development_analysis"] != recomputed
            or recomputed["selected_scale"] != SCALE
            or recomputed["development_go_for_new_confirmation_design"] is not True):
        raise RuntimeError("D2 预先规则未选出统一 1.75 倍")
    return {"selected_scale": SCALE,
            "development_relative_power_gain": next(x["relative_power_gain"] for x in recomputed["candidate_comparisons"] if x["scale"] == SCALE),
            "development_only": True}


def preflight(path: str | Path = CONFIG, *, quick: bool = False) -> tuple[dict, dict, Path, torch.device]:
    cfg = _load_yaml(_project_path(path))
    _contract(cfg)
    if source_bundle_sha256() != FROZEN_SOURCE:
        raise RuntimeError("R5 物理或策略源码已改变")
    if _file_sha256(_project_path(cfg["parent"])) != cfg["parent_sha256"]:
        raise RuntimeError("训练配置已改变")
    parent = _load_yaml(_project_path(cfg["parent"]))
    if ([x["id"] for x in parent["families"]] != list(FAMILIES)
            or parent["profile_ids"] != list(PROFILES)):
        raise RuntimeError("湍流/硬件分组已改变")
    training = _project_path(cfg["training_output"])
    training_summary = training / "summary.json"
    if (_file_sha256(training_summary) != cfg["training_summary_sha256"]
            or json.loads(training_summary.read_text(encoding="utf-8"))["status"]
            != "R5_2_TRAINING_COMPLETE_REQUIRES_AUDIT"):
        raise RuntimeError("冻结训练摘要已改变")
    if set(cfg["checkpoints"]) != {f"policy_{i}_02000.pt" for i in range(3)}:
        raise RuntimeError("必须有三份冻结策略")
    for name, digest in cfg["checkpoints"].items():
        if _file_sha256(training / "checkpoints" / name) != digest:
            raise RuntimeError(f"冻结权重已改变: {name}")
    development = _validate_development(cfg)
    streams = stream_manifest(quick)
    formal_streams = set(stream_manifest(False)["turbulence"])
    quick_streams = set(stream_manifest(True)["turbulence"])
    if formal_streams & quick_streams or min(formal_streams | quick_streams) < 6_300_000:
        raise RuntimeError("新天气与旧实验或冒烟重叠")
    output = _project_path(cfg["quick_directory"] if quick else cfg["output_directory"])
    if output.exists():
        raise FileExistsError(f"保留已有确认输出，不覆盖/续跑: {output}")
    device = resolve_device("cuda")
    count, steps = (1, 16) if quick else (64, 200)
    report = {"status": "READY_FOR_DIAGNOSTIC" if quick else "READY_FOR_USER_IDE",
              "quick": quick, "device": str(device), "weather_count": count,
              "families": 1 if quick else 3, "profiles": 2 if quick else 6,
              "controllers": 4, "episode_length": steps,
              "physical_transitions": 4 * count * (1 if quick else 3) * (2 if quick else 6) * steps,
              "unique_turbulence_streams": len(streams["turbulence"]),
              "unique_sensor_streams": len(streams["sensor"]),
              "unique_power_streams": len(streams["power"]),
              "config_sha256": _file_sha256(_project_path(path)),
              "entry_sha256": _file_sha256(Path(__file__)),
              "source_bundle_sha256": FROZEN_SOURCE,
              "development_selection": development,
              "selected_scale": SCALE, **cfg["boundary"]}
    return cfg, report, output, device


def summarize(rows: list[dict], cfg: dict, device: torch.device) -> dict:
    weather = [FORMAL_BASE + 10 * i for i in range(64)]
    controllers = ("integrator",) + tuple(f"policy_{i}_scale_{SCALE}" for i in range(3))
    by_key = {(r["controller"], r["family"], r["profile"], r["weather_seed"]): r for r in rows}
    expected = {(controller, family, profile, seed) for controller in controllers
                for family in FAMILIES for profile in PROFILES for seed in weather}
    if len(rows) != len(expected) or set(by_key) != expected:
        raise RuntimeError("R3 确认记录缺失、重复或分组错位")
    for row in rows:
        if (row["scale"] != (0.0 if row["controller"] == "integrator" else SCALE)
                or any(not math.isfinite(row[metric]) for metric in METRICS)
                or not 0 <= row["normalized_correction_clipped_fraction"] <= 1):
            raise RuntimeError("R3 确认数值或倍率错误")
    values = {metric: torch.tensor(
        [[[[by_key[(controller, family, profile, seed)][metric] for profile in PROFILES]
           for seed in weather] for family in FAMILIES] for controller in controllers],
        device=device, dtype=torch.float64) for metric in METRICS}
    power = values["power"]
    baseline = power[0].mean()
    if baseline <= 0:
        raise RuntimeError("积分器桶内功率非正")
    delta = power[1:] - power[0]
    difference = delta.mean()
    family_weather = delta.mean(dim=(0, 3))
    generator = torch.Generator(device=device).manual_seed(cfg["statistics"]["bootstrap_seed"])
    repeats = cfg["statistics"]["bootstrap_repeats"]
    indices = torch.randint(64, (repeats, 3, 64), generator=generator, device=device)
    samples = torch.gather(family_weather[None].expand(repeats, -1, -1), 2, indices)
    ci = torch.quantile(samples.mean(dim=(1, 2)),
                        torch.tensor([.025, .975], device=device, dtype=torch.float64))
    other = {metric: float((value[1:] - value[0]).mean())
             for metric, value in values.items() if metric != "power"}
    member = delta.mean(dim=(1, 2, 3))
    family = delta.mean(dim=(0, 2, 3))
    profile = delta.mean(dim=(0, 1, 2))
    family_profile = delta.mean(dim=(0, 2))
    relative = float(difference / baseline)
    limit = cfg["thresholds"]["maximum_safety_increase"]
    gates = {"mean_relative_power_at_least_1pct": relative >= cfg["thresholds"]["relative_power_gain"],
             "paired_ci_lower_positive": float(ci[0]) > 0,
             "each_initialization_positive": bool((member > 0).all()),
             "each_family_positive": bool((family > 0).all()),
             "each_profile_positive": bool((profile > 0).all()),
             "strehl_not_lower": other["strehl"] >= 0,
             "phase_rmse_not_higher": other["phase_rmse"] <= 0,
             "violation_within_limit": other["violation"] <= limit,
             "saturation_within_limit": other["saturation"] <= limit,
             "slew_within_limit": other["slew_limited"] <= limit}
    clips = [by_key[(controller, family_name, profile_name, seed)]["normalized_correction_clipped_fraction"]
             for controller in controllers[1:] for family_name in FAMILIES
             for profile_name in PROFILES for seed in weather]
    return {"baseline_power": float(baseline), "policy_power": float(power[1:].mean()),
            "absolute_power_gain": float(difference), "relative_power_gain": relative,
            "absolute_power_gain_ci95": [float(x) for x in ci],
            "member_absolute_gains": [float(x) for x in member],
            "family_absolute_gains": dict(zip(FAMILIES, [float(x) for x in family])),
            "profile_absolute_gains": dict(zip(PROFILES, [float(x) for x in profile])),
            "positive_family_profile_cells": int((family_profile > 0).sum()),
            "normalized_correction_clipped_fraction": math.fsum(clips) / len(clips),
            "other_deltas": other, "preliminary_gates": gates,
            "preliminary_all_gates": all(gates.values()),
            "bootstrap_unit": "weather_within_family; profiles_and_initializations_paired"}


def run(path: str | Path = CONFIG, *, quick: bool = False, preflight_only: bool = False) -> dict:
    cfg, report, output, device = preflight(path, quick=quick)
    if preflight_only:
        return report
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "preflight.json", report)
    write_json(output / "config.json", cfg)
    streams = stream_manifest(quick)
    write_json(output / "stream_manifest.json", streams)
    write_json(output / "runtime.json", {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                                          "gpu": torch.cuda.get_device_name(), "git": safe_git_record(),
                                          "source_bundle_sha256": FROZEN_SOURCE})
    progress = Progress(output, device)
    started = time.perf_counter()
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
                raise RuntimeError("R5 权重身份不匹配")
            policy = ResidualGRUPolicy(parent["policy"]["hidden_size"],
                                       parent["policy"]["output_size"]).to(device)
            policy.load_state_dict(checkpoint["state_dict"])
            policy.eval()
            policies.append(policy)
        branches = [("integrator", 0.0, None)] + [
            (f"policy_{member}_scale_{SCALE}", SCALE, policy)
            for member, policy in enumerate(policies)]
        progress.phase("R5-R3快速冒烟" if quick else "R5-R3独立确认",
                       4 * report["weather_count"] * report["episode_length"])
        rows: list[dict] = []
        with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
            for label, scale, policy in branches:
                for seed in streams["weather_bases"]:
                    meter = None if policy is None else CorrectionClampTelemetry(policy, scale)
                    episode = _rollout(seed, label, scale, meter, report["episode_length"],
                                       basis, base, families, profiles,
                                       parent["data"]["sensor_seed_offset"], progress)
                    rates = ([0.0] * len(episode) if meter is None
                             else meter.rates(report["episode_length"], len(episode)))
                    for row, rate in zip(episode, rates, strict=True):
                        row["normalized_correction_clipped_fraction"] = rate
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        rows.append(row)
                    handle.flush()
        expected = 4 * report["weather_count"] * report["families"] * report["profiles"]
        if len(rows) != expected:
            raise RuntimeError("R3 确认记录数量错误")
        analysis = {} if quick else summarize(rows, cfg, device)
        result = {"status": "QUICK_SMOKE_NO_CONCLUSION" if quick
                  else "R5_R3_CONFIRMATION_COMPLETE_REQUIRES_AUDIT",
                  "records": len(rows), "controllers": [label for label, _, _ in branches],
                  "weather_count": report["weather_count"],
                  "physical_transitions": report["physical_transitions"],
                  "elapsed_seconds": time.perf_counter() - started,
                  "analysis": analysis, "confirmation_access": not quick,
                  "config_sha256": report["config_sha256"],
                  "entry_sha256": report["entry_sha256"],
                  "source_bundle_sha256": FROZEN_SOURCE,
                  "material_passport": {"origin_skill": "academic-research-suite / experiment-agent",
                                        "origin_mode": "run", "verification_status": "UNVERIFIED"},
                  **cfg["boundary"],
                  "next_action": "停止并等待只读审计；预计算门槛不等于最终结论"}
        write_json(output / "summary.json", result)
        write_json(output / "SUCCESS.json", {
            "summary_sha256": _file_sha256(output / "summary.json"),
            "records_sha256": _file_sha256(output / "records.jsonl"),
            "progress_sha256": _file_sha256(output / "progress.jsonl"),
            "stream_manifest_sha256": _file_sha256(output / "stream_manifest.json")})
        return result
    except Exception:
        write_json(output / "failure.json", {"traceback": traceback.format_exc(),
                                             "automatic_retry": False})
        raise
    finally:
        progress.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--quick", action="store_true", help="16帧CUDA冒烟，不得作性能结论")
    parser.add_argument("--preflight-only", action="store_true", help="只读检查，不生成输出")
    args = parser.parse_args()
    print(json.dumps(run(args.config, quick=args.quick, preflight_only=args.preflight_only),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
