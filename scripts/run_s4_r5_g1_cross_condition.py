"""R5-G1 冻结策略跨条件复核；正式长时仿真仅由用户在 IDE 启动。"""
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

from scripts.confirm_s4_r5_policy_r3 import FAMILIES, PROFILES, SCALE, stream_manifest as r3_stream_manifest
from scripts.diagnose_s4_r5_margin_d2 import CorrectionClampTelemetry
from src.rl.r4_dynamics_experiment import Progress, safe_git_record
from src.rl.r4_interface_smoke import write_json
from src.rl.r5_independent_confirmation_r2 import METRICS, source_bundle_sha256
from src.rl.r5_margin_development import _rollout, effective_stream_seeds
from src.rl.r5_policy_training import ResidualGRUPolicy
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _profiles, _project_path
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.robust_control import RobustnessCondition


CONFIG = "configs/experiments/s4_r5_g1_cross_condition_v1.yaml"
HARDWARE_PROFILES = ("delay_1", "quantization_64", "lut_scale_085",
                     "observation_noise_010", "power_noise_003", "combined_severe")
FROZEN_SOURCE = "ccdc31faaa155361d8bd3e19ffc5bb82705f425a0ef4c7f58dd7f70831efd956"
ARM_SEEDS = {"wind": (6_400_000, 6_410_000), "hardware": (6_500_000, 6_510_000)}


def _contract(cfg: dict) -> None:
    if (cfg.get("stage") != "S4-D2-R5-G1"
            or cfg.get("source_bundle_sha256") != FROZEN_SOURCE
            or cfg.get("r3_config") != "configs/experiments/s4_r5_independent_confirmation_r3.yaml"
            or cfg.get("r3_config_sha256") != "5818c371c49f96244e6fea27f2d4c05afe0ab22437d79123f689311fc9e54f7f"
            or cfg.get("r3_entry_sha256") != "6954404e3e8e20e65a1a6cc1fc2f2be4b80b01ad7c3e8e624001ad53d8a4ed0c"
            or cfg.get("r3_output") != "outputs/s4_r5_independent_confirmation_r3_v1"
            or set(cfg.get("r3_hashes", {})) != {"summary.json", "records.jsonl", "stream_manifest.json", "SUCCESS.json"}
            or cfg.get("selected_scale") != SCALE
            or cfg.get("wind_speed_multiplier") != 1.10
            or cfg.get("hardware_shift_profiles") != list(HARDWARE_PROFILES)
            or cfg.get("runtime") != {"device": "cuda", "formal_owner": "user_ide", "automatic_retry": False}
            or cfg.get("data") != {"seed_stride": 10, "weather_count": 64,
                                   "episode_length": 200, "wind_seed_base": 6_400_000,
                                   "hardware_seed_base": 6_500_000}
            or cfg.get("quick") != {"weather_count": 1, "episode_length": 16,
                                    "wind_seed_base": 6_410_000, "hardware_seed_base": 6_510_000,
                                    "wind_profiles": ["nominal", "combined_moderate"],
                                    "hardware_profiles": ["quantization_64", "combined_severe"]}
            or cfg.get("statistics") != {"bootstrap_seed_wind": 6_423_456,
                                         "bootstrap_seed_hardware": 6_523_456,
                                         "bootstrap_repeats": 20_000}
            or cfg.get("thresholds") != {"relative_power_gain": .01,
                                         "maximum_safety_increase": .001}
            or cfg.get("outputs") != {"wind": "outputs/s4_r5_g1_wind_v1",
                                      "wind_quick": "outputs/s4_r5_g1_wind_v1_quick",
                                      "hardware": "outputs/s4_r5_g1_hardware_v1",
                                      "hardware_quick": "outputs/s4_r5_g1_hardware_v1_quick"}
            or cfg.get("boundary") != {"training_updates": 0, "real_slm_actions": False,
                                       "automatic_retry": False, "r3_results_read_only": True,
                                       "r3_trajectory_tuning": False}):
        raise ValueError("G1 事前冻结的跨条件合同被修改")


def stream_manifest(arm: str, quick: bool, sensor_offset: int = 50_000_000) -> dict[str, list[int]]:
    if arm not in ARM_SEEDS or sensor_offset != 50_000_000:
        raise ValueError("G1 随机流参数错误")
    base = ARM_SEEDS[arm][1 if quick else 0]
    count, families, profiles = (1, 1, 2) if quick else (64, 3, 6)
    weather = [base + 10 * i for i in range(count)]
    turbulence = effective_stream_seeds(base, count, 10, profiles, families)
    sensor = [seed + 1000 * i + sensor_offset for seed in weather for i in range(profiles)]
    power = [seed + 1000 * i + 60_000_000 for seed in weather for i in range(profiles)]
    if (len(turbulence) != count * families * profiles
            or len(sensor) != len(set(sensor)) or len(power) != len(set(power))
            or min(turbulence) < base or max(turbulence) >= base + 10_000):
        raise RuntimeError("G1 随机流冲突")
    return {"weather_bases": weather, "turbulence": turbulence,
            "sensor": sensor, "power": power}


def _verify_r3_source(cfg: dict) -> dict:
    if source_bundle_sha256() != FROZEN_SOURCE:
        raise RuntimeError("冻结物理/控制源码已改变")
    if _file_sha256(_project_path(cfg["r3_config"])) != cfg["r3_config_sha256"]:
        raise RuntimeError("R3 配置已改变")
    if _file_sha256(_project_path("scripts/confirm_s4_r5_policy_r3.py")) != cfg["r3_entry_sha256"]:
        raise RuntimeError("R3 入口已改变")
    r3_cfg = _load_yaml(_project_path(cfg["r3_config"]))
    if (r3_cfg["selected_scale"] != SCALE
            or r3_cfg["source_bundle_sha256"] != FROZEN_SOURCE
            or r3_cfg["data"]["seed_base"] != 6_300_000):
        raise RuntimeError("R3 策略身份与天气不符")
    root = _project_path(cfg["r3_output"])
    for name, digest in cfg["r3_hashes"].items():
        if _file_sha256(root / name) != digest:
            raise RuntimeError(f"R3 证据已改变: {name}")
    success = json.loads((root / "SUCCESS.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if ((root / "failure.json").exists()
            or success["summary_sha256"] != cfg["r3_hashes"]["summary.json"]
            or success["records_sha256"] != cfg["r3_hashes"]["records.jsonl"]
            or success["stream_manifest_sha256"] != cfg["r3_hashes"]["stream_manifest.json"]
            or summary["status"] != "R5_R3_CONFIRMATION_COMPLETE_REQUIRES_AUDIT"
            or summary["analysis"]["preliminary_all_gates"] is not True
            or summary["confirmation_access"] is not True
            or summary["training_updates"] != 0):
        raise RuntimeError("R3 已审计确认的完整性不成立")
    if set(r3_stream_manifest(False)["turbulence"]) & set(stream_manifest("wind", False)["turbulence"]):
        raise RuntimeError("G1 风速天气与 R3 重叠")
    parent_path = _project_path(r3_cfg["parent"])
    if _file_sha256(parent_path) != r3_cfg["parent_sha256"]:
        raise RuntimeError("R5 训练配置已改变")
    training = _project_path(r3_cfg["training_output"])
    training_summary = training / "summary.json"
    if (_file_sha256(training_summary) != r3_cfg["training_summary_sha256"]
            or json.loads(training_summary.read_text(encoding="utf-8"))["status"]
            != "R5_2_TRAINING_COMPLETE_REQUIRES_AUDIT"):
        raise RuntimeError("R5 训练摘要已改变")
    if set(r3_cfg["checkpoints"]) != {f"policy_{i}_02000.pt" for i in range(3)}:
        raise RuntimeError("三份冻结策略缺失")
    for name, digest in r3_cfg["checkpoints"].items():
        if _file_sha256(training / "checkpoints" / name) != digest:
            raise RuntimeError(f"策略权重已改变: {name}")
    return r3_cfg


def _conditions(parent: dict, arm: str, quick: bool, multiplier: float) -> tuple[list[dict], list]:
    families = parent["families"][:1] if quick else parent["families"]
    if arm == "wind":
        families = [dict(row, wind_speed_mps=float(row["wind_speed_mps"]) * multiplier)
                    for row in families]
        profile_ids = ["nominal", "combined_moderate"] if quick else list(PROFILES)
    else:
        profile_ids = ["quantization_64", "combined_severe"] if quick else list(HARDWARE_PROFILES)
    profiles = _profiles(parent, profile_ids)
    if ([row["id"] for row in families] != list(FAMILIES[:1] if quick else FAMILIES)
            or [profile.identifier for profile in profiles] != profile_ids):
        raise RuntimeError("G1 条件顺序被改变")
    return families, profiles


def preflight(path: str | Path = CONFIG, *, arm: str, quick: bool = False) -> tuple[dict, dict, dict, Path, torch.device]:
    cfg = _load_yaml(_project_path(path))
    _contract(cfg)
    if arm not in ARM_SEEDS:
        raise ValueError("arm 必须为 wind 或 hardware")
    r3_cfg = _verify_r3_source(cfg)
    parent = _load_yaml(_project_path(r3_cfg["parent"]))
    if ([row["id"] for row in parent["families"]] != list(FAMILIES)
            or parent["profile_ids"] != list(PROFILES)
            or parent["data"]["sensor_seed_offset"] != 50_000_000):
        raise RuntimeError("R5 因果环境/分组合同已改变")
    base, _ = load_s1_config(_project_path(parent["environment_config"]))
    families, profiles = _conditions(parent, arm, quick, cfg["wind_speed_multiplier"])
    episode_length = 16 if quick else 200
    maximum_displacements = []
    for family in families:
        condition = RobustnessCondition.from_mapping(dict(family, base_seed=ARM_SEEDS[arm][1 if quick else 0]))
        simulation = condition.environment_config(replace(base, episode_length=episode_length))
        displacement = (simulation.wind_speed_mps * (1 + simulation.wind_speed_modulation_fraction)
                        * simulation.dt_s * episode_length / simulation.sample_pitch_m)
        maximum_displacements.append(displacement)
        if displacement >= simulation.turbulence_grid_size:
            raise RuntimeError("相位屏会在完整回合内重复")
    all_streams = {key: stream_manifest(test_arm, test_quick)["turbulence"]
                   for key, test_arm, test_quick in (("wind", "wind", False),
                                                     ("wind_quick", "wind", True),
                                                     ("hardware", "hardware", False),
                                                     ("hardware_quick", "hardware", True))}
    history = set(r3_stream_manifest(False)["turbulence"])
    selected = stream_manifest(arm, quick)
    if (len(set().union(*map(set, all_streams.values()))) != sum(map(len, all_streams.values()))
            or history & set().union(*map(set, all_streams.values()))):
        raise RuntimeError("G1 正式/冒烟或 R3 天气冲突")
    output = _project_path(cfg["outputs"][arm + ("_quick" if quick else "")])
    if output.exists():
        raise FileExistsError(f"保留现有 G1 输出，不覆盖/重跑: {output}")
    device = resolve_device("cuda")
    weather_count = 1 if quick else 64
    report = {"status": "READY_FOR_DIAGNOSTIC" if quick else "READY_FOR_USER_IDE",
              "arm": arm, "quick": quick, "device": str(device),
              "weather_count": weather_count, "episode_length": episode_length,
              "families": len(families), "profiles": len(profiles), "controllers": 4,
              "physical_transitions": 4 * weather_count * len(families) * len(profiles) * episode_length,
              "unique_turbulence_streams": len(selected["turbulence"]),
              "unique_sensor_streams": len(selected["sensor"]),
              "unique_power_streams": len(selected["power"]),
              "maximum_displacement_pixels": maximum_displacements,
              "turbulence_grid_pixels": simulation.turbulence_grid_size,
              "family_ids": [family["id"] for family in families],
              "profile_ids": [profile.identifier for profile in profiles],
              "selected_scale": SCALE, "config_sha256": _file_sha256(_project_path(path)),
              "entry_sha256": _file_sha256(Path(__file__)),
              "source_bundle_sha256": FROZEN_SOURCE, **cfg["boundary"]}
    return cfg, r3_cfg, report, output, device


def summarize(rows: list[dict], cfg: dict, *, arm: str, device: torch.device) -> dict:
    if arm not in ARM_SEEDS:
        raise ValueError("无效 G1 测试臂")
    weather = [ARM_SEEDS[arm][0] + 10 * i for i in range(64)]
    profiles = PROFILES if arm == "wind" else HARDWARE_PROFILES
    controllers = ("integrator",) + tuple(f"policy_{i}_scale_{SCALE}" for i in range(3))
    by_key = {(row["controller"], row["family"], row["profile"], row["weather_seed"]): row
              for row in rows}
    expected = {(controller, family, profile, seed) for controller in controllers
                for family in FAMILIES for profile in profiles for seed in weather}
    if len(rows) != len(expected) or set(by_key) != expected:
        raise RuntimeError("G1 回合缺失、重复或条件错位")
    for row in rows:
        if (row["arm"] != arm
                or row["scale"] != (0.0 if row["controller"] == "integrator" else SCALE)
                or row["turbulence_stream_seed"] != (row["weather_seed"]
                    + profiles.index(row["profile"]) * 1000 + FAMILIES.index(row["family"]))
                or any(not math.isfinite(row[metric]) for metric in METRICS)
                or not 0 <= row["normalized_correction_clipped_fraction"] <= 1):
            raise RuntimeError("G1 非有限指标、错位种子或错误动作")
    values = {metric: torch.tensor(
        [[[[by_key[(controller, family, profile, seed)][metric] for profile in profiles]
           for seed in weather] for family in FAMILIES] for controller in controllers],
        device=device, dtype=torch.float64) for metric in METRICS}
    power = values["power"]
    baseline = power[0].mean()
    if baseline <= 0:
        raise RuntimeError("G1 积分器功率非正")
    delta = power[1:] - power[0]
    family_weather = delta.mean(dim=(0, 3))
    generator = torch.Generator(device=device).manual_seed(cfg["statistics"][f"bootstrap_seed_{arm}"])
    repeats = cfg["statistics"]["bootstrap_repeats"]
    indices = torch.randint(64, (repeats, 3, 64), generator=generator, device=device)
    samples = torch.gather(family_weather[None].expand(repeats, -1, -1), 2, indices)
    ci = torch.quantile(samples.mean(dim=(1, 2)),
                        torch.tensor([.025, .975], device=device, dtype=torch.float64))
    other = {metric: float((value[1:] - value[0]).mean()) for metric, value in values.items()
             if metric != "power"}
    member = delta.mean(dim=(1, 2, 3))
    family = delta.mean(dim=(0, 2, 3))
    profile = delta.mean(dim=(0, 1, 2))
    relative = float(delta.mean() / baseline)
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
    policy_rows = [by_key[(controller, family_name, profile_name, seed)]
                   for controller in controllers[1:] for family_name in FAMILIES
                   for profile_name in profiles for seed in weather]
    mean = lambda key: math.fsum(row[key] for row in policy_rows) / len(policy_rows)
    return {"baseline_power": float(baseline), "policy_power": float(power[1:].mean()),
            "absolute_power_gain": float(delta.mean()), "relative_power_gain": relative,
            "absolute_power_gain_ci95": [float(x) for x in ci],
            "member_absolute_gains": [float(x) for x in member],
            "family_absolute_gains": dict(zip(FAMILIES, [float(x) for x in family])),
            "profile_absolute_gains": dict(zip(profiles, [float(x) for x in profile])),
            "positive_family_profile_cells": int((delta.mean(dim=(0, 2)) > 0).sum()),
            "normalized_correction_clipped_fraction": mean("normalized_correction_clipped_fraction"),
            "requested_applied_gap_abs": mean("requested_applied_gap_abs"),
            "policy_forward_seconds_per_step": mean("policy_forward_seconds_per_step"),
            "other_deltas": other, "preliminary_gates": gates,
            "preliminary_all_gates": all(gates.values()),
            "bootstrap_unit": "weather_within_family; profiles_and_initializations_paired"}


def run(path: str | Path = CONFIG, *, arm: str, quick: bool = False,
        preflight_only: bool = False) -> dict:
    cfg, r3_cfg, report, output, device = preflight(path, arm=arm, quick=quick)
    if preflight_only:
        return report
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "preflight.json", report)
    write_json(output / "config.json", cfg)
    streams = stream_manifest(arm, quick)
    write_json(output / "stream_manifest.json", streams)
    write_json(output / "runtime.json", {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                                          "gpu": torch.cuda.get_device_name(), "git": safe_git_record(),
                                          "source_bundle_sha256": FROZEN_SOURCE})
    progress = Progress(output, device)
    started = time.perf_counter()
    try:
        parent = _load_yaml(_project_path(r3_cfg["parent"]))
        families, profiles = _conditions(parent, arm, quick, cfg["wind_speed_multiplier"])
        base, _ = load_s1_config(_project_path(parent["environment_config"]))
        base = replace(base, num_modes=21, batch_size=1, episode_length=report["episode_length"])
        basis, _, _ = build_action_basis(base, ActionRepresentation("r5_zernike21", "zernike", 21), device)
        policies = []
        for member in range(3):
            checkpoint = torch.load(_project_path(r3_cfg["training_output"]) / "checkpoints"
                                    / f"policy_{member}_02000.pt", map_location=device, weights_only=True)
            if checkpoint["init"] != member or checkpoint["update"] != 2000:
                raise RuntimeError("R5 冻结权重身份错误")
            policy = ResidualGRUPolicy(parent["policy"]["hidden_size"],
                                       parent["policy"]["output_size"]).to(device)
            policy.load_state_dict(checkpoint["state_dict"])
            policy.eval()
            policies.append(policy)
        branches = [("integrator", 0.0, None)] + [
            (f"policy_{member}_scale_{SCALE}", SCALE, policy)
            for member, policy in enumerate(policies)]
        progress.phase(f"R5-G1-{arm}-快速冒烟" if quick else f"R5-G1-{arm}-正式复核",
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
                        row["arm"] = arm
                        row["normalized_correction_clipped_fraction"] = rate
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        rows.append(row)
                    handle.flush()
        expected = 4 * report["weather_count"] * report["families"] * report["profiles"]
        if len(rows) != expected:
            raise RuntimeError("G1 分组回合数量不足")
        analysis = {} if quick else summarize(rows, cfg, arm=arm, device=device)
        result = {"status": "QUICK_SMOKE_NO_CONCLUSION" if quick
                  else "R5_G1_ARM_COMPLETE_REQUIRES_AUDIT",
                  "arm": arm, "records": len(rows),
                  "controllers": [label for label, _, _ in branches],
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
                  "next_action": "停止并等待只读审计；两臂不得合并掩盖失败"}
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
    parser.add_argument("--arm", choices=("wind", "hardware"), required=True)
    parser.add_argument("--quick", action="store_true", help="16帧CUDA诊断冒烟，不产生性能结论")
    parser.add_argument("--preflight-only", action="store_true", help="只读检查，不生成输出")
    args = parser.parse_args()
    print(json.dumps(run(args.config, arm=args.arm, quick=args.quick,
                         preflight_only=args.preflight_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
