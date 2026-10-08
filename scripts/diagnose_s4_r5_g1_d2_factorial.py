"""G1-D2：新天气下风速×硬件档位四格配对开发诊断；正式运行由用户启动。"""
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

from scripts import diagnose_s4_r5_g1_hardware_mechanism as d1
from scripts import run_s4_r5_g1_cross_condition as g1
from scripts.diagnose_s4_r5_margin_d2 import CorrectionClampTelemetry
from src.rl.r4_dynamics_experiment import Progress, safe_git_record
from src.rl.r4_interface_smoke import write_json
from src.rl.r5_independent_confirmation_r2 import source_bundle_sha256
from src.rl.r5_margin_development import _rollout, effective_stream_seeds
from src.rl.r5_policy_training import ResidualGRUPolicy
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _profiles, _project_path
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.robust_control import RobustnessCondition


CONFIG = "configs/experiments/s4_r5_g1_d2_factorial_v1.yaml"
WINDS = ("original", "faster_110")
HARDWARE = ("original", "shift")
FAMILIES = g1.FAMILIES
ORIGINAL_PROFILES = g1.PROFILES
SHIFT_PROFILES = g1.HARDWARE_PROFILES
SCALE = g1.SCALE
FORMAL_BASE = 6_700_000
QUICK_BASE = 6_710_000
FROZEN_SOURCE = g1.FROZEN_SOURCE
D1_CONFIG_SHA256 = "37d0591555dae5e0cb9da64c976edff3d14fa617440f6251fefa566432739162"
D1_ENTRY_SHA256 = "8f38330709366fa331648f0ace1da15509c4b6d7a39458f9429ded92b3b0fad9"
D1_OUTPUT_HASHES = {
    "summary.json": "2d098007e85fcc9119fd27e3c1a5addc270487de472765579f23fe16aceed7ad",
    "records.jsonl": "4746a2d4c5d8f247c53f03a69ceb53b2b0931d4378d4d33dc7d9fe1b3171e5dd",
    "progress.jsonl": "83c179155e2b1bcd1e28f83a27d5fa0be88683a968cd0506f63e415adf08dc44",
    "stream_manifest.json": "8d78f96a873a5eb7b53fbd2596f277756ac6dfd0223f8854cabed25e4a0a1527",
    "SUCCESS.json": "0d35eae882316583775c4cd984c6b0dc92c0db4185b483dc44687b19176ea82f",
}
METRICS = d1.METRICS


def _contract(cfg: dict) -> None:
    if (cfg.get("stage") != "S4-D2-R5-G1-D2"
            or cfg.get("purpose") != "development_only_paired_wind_hardware_factorial"
            or cfg.get("runtime") != {"device": "cuda", "formal_owner": "user_ide", "automatic_retry": False}
            or cfg.get("g1_d1_config") != d1.CONFIG
            or cfg.get("g1_d1_config_sha256") != D1_CONFIG_SHA256
            or cfg.get("g1_d1_entry_sha256") != D1_ENTRY_SHA256
            or cfg.get("g1_d1_output") != "outputs/s4_r5_g1_hardware_mechanism_v1"
            or cfg.get("g1_d1_output_hashes") != D1_OUTPUT_HASHES
            or cfg.get("selected_scale") != SCALE
            or cfg.get("wind_speed_multiplier") != 1.10
            or cfg.get("original_profiles") != list(ORIGINAL_PROFILES)
            or cfg.get("shift_profiles") != list(SHIFT_PROFILES)
            or cfg.get("data") != {"seed_base": FORMAL_BASE, "seed_stride": 10,
                                   "weather_count": 24, "episode_length": 200}
            or cfg.get("quick") != {"seed_base": QUICK_BASE, "weather_count": 1,
                                    "episode_length": 16}
            or cfg.get("statistics") != {"bootstrap_seed": 6_723_456,
                                         "bootstrap_repeats": 5_000}
            or cfg.get("output_directory") != "outputs/s4_r5_g1_d2_factorial_v1"
            or cfg.get("quick_directory") != "outputs/s4_r5_g1_d2_factorial_v1_quick_r1"
            or cfg.get("boundary") != {"confirmation_access": False,
                                       "training_updates": 0, "real_slm_actions": False,
                                       "automatic_retry": False,
                                       "historical_results_read_only": True}):
        raise ValueError("G1-D2 四格配对开发诊断合同被改变")


def _verify_lineage(cfg: dict) -> tuple[dict, dict, dict]:
    if _file_sha256(_project_path(cfg["g1_d1_config"])) != D1_CONFIG_SHA256:
        raise RuntimeError("G1-D1 冻结配置已改变")
    if _file_sha256(_project_path("scripts/diagnose_s4_r5_g1_hardware_mechanism.py")) != D1_ENTRY_SHA256:
        raise RuntimeError("G1-D1 冻结入口已改变")
    d1_cfg = _load_yaml(_project_path(cfg["g1_d1_config"]))
    d1._contract(d1_cfg)
    _, r3_cfg = d1._verify_lineage(d1_cfg)  # 同时验证 R3、训练权重、G1 双臂证据。
    if source_bundle_sha256() != FROZEN_SOURCE:
        raise RuntimeError("冻结物理/控制源码已改变")
    root = _project_path(cfg["g1_d1_output"])
    if (root / "failure.json").exists():
        raise RuntimeError("G1-D1 历史输出存在失败标记")
    for name, digest in D1_OUTPUT_HASHES.items():
        if _file_sha256(root / name) != digest:
            raise RuntimeError(f"G1-D1 历史证据已改变: {name}")
    success = json.loads((root / "SUCCESS.json").read_text(encoding="utf-8"))
    for key, name in (("summary_sha256", "summary.json"),
                      ("records_sha256", "records.jsonl"),
                      ("progress_sha256", "progress.jsonl"),
                      ("stream_manifest_sha256", "stream_manifest.json")):
        if success.get(key) != D1_OUTPUT_HASHES[name]:
            raise RuntimeError("G1-D1 成功标记与历史证据不符")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("status") != "DEVELOPMENT_DIAGNOSTIC_REQUIRES_AUDIT"
            or summary.get("records") != 3456
            or summary.get("training_updates") != 0
            or summary.get("confirmation_access") is not False
            or summary.get("real_slm_actions") is not False
            or summary.get("g1_config_sha256") != d1.G1_CONFIG_SHA256
            or summary.get("frozen_source_bundle_sha256") != FROZEN_SOURCE):
        raise RuntimeError("G1-D1 已审计开发诊断状态与冻结证据不符")
    parent = _load_yaml(_project_path(r3_cfg["parent"]))
    return d1_cfg, r3_cfg, parent


def stream_manifest(quick: bool) -> dict[str, list[int]]:
    base, count = (QUICK_BASE, 1) if quick else (FORMAL_BASE, 24)
    weather = [base + 10 * index for index in range(count)]
    turbulence = effective_stream_seeds(base, count, 10, 6, 3)
    sensor = [seed + 1_000 * slot + 50_000_000 for seed in weather for slot in range(6)]
    power = [seed + 1_000 * slot + 60_000_000 for seed in weather for slot in range(6)]
    if (len(set(turbulence)) != count * 18
            or len(set(sensor)) != count * 6
            or len(set(power)) != count * 6
            or min(turbulence) < base or max(turbulence) >= base + 10_000):
        raise RuntimeError("G1-D2 随机流冲突")
    return {"weather_bases": weather, "turbulence": turbulence,
            "sensor": sensor, "power": power}


def _conditions(parent: dict, wind: str, hardware: str, multiplier: float) -> tuple[list[dict], list]:
    if wind not in WINDS or hardware not in HARDWARE:
        raise ValueError("G1-D2 因子水平无效")
    families = [dict(item) for item in parent["families"]]
    if wind == "faster_110":
        families = [dict(item, wind_speed_mps=float(item["wind_speed_mps"]) * multiplier)
                    for item in families]
    profile_ids = ORIGINAL_PROFILES if hardware == "original" else SHIFT_PROFILES
    profiles = _profiles(parent, profile_ids)
    if ([item["id"] for item in families] != list(FAMILIES)
            or [item.identifier for item in profiles] != list(profile_ids)):
        raise RuntimeError("G1-D2 湍流或硬件顺序已改变")
    return families, profiles


def preflight(path: str | Path = CONFIG, *, quick: bool = False
              ) -> tuple[dict, dict, dict, dict, Path, torch.device]:
    cfg = _load_yaml(_project_path(path))
    _contract(cfg)
    _, r3_cfg, parent = _verify_lineage(cfg)
    if (parent["profile_ids"] != list(ORIGINAL_PROFILES)
            or [family["id"] for family in parent["families"]] != list(FAMILIES)
            or parent["data"]["sensor_seed_offset"] != 50_000_000):
        raise RuntimeError("R5 原始档位、湍流或观测流合同已改变")
    spec = cfg["quick" if quick else "data"]
    formal, smoke = stream_manifest(False), stream_manifest(True)
    historical_manifests = [g1.r3_stream_manifest(False), g1.r3_stream_manifest(True)]
    for arm in ("wind", "hardware"):
        for old_quick in (False, True):
            historical_manifests.append(g1.stream_manifest(arm, old_quick))
    for old_quick in (False, True):
        historical_manifests.append(d1.stream_manifest(old_quick))
    for stream in ("turbulence", "sensor", "power"):
        history = set().union(*(set(manifest[stream]) for manifest in historical_manifests))
        if (set(formal[stream]) & set(smoke[stream])
                or history & (set(formal[stream]) | set(smoke[stream]))):
            raise RuntimeError(f"G1-D2 {stream} 随机流与 R3/G1/G1-D1 或自身冒烟重叠")
    training_data = parent["data"]
    if (training_data["train_seed_base"] != 5_500_000
            or training_data["family_offsets"] != [0, 4096, 8192]
            or training_data["episodes_per_family"] != 256
            or training_data["train_seed_base"] + max(training_data["family_offsets"])
            + 10 * training_data["episodes_per_family"] + 6_000 >= FORMAL_BASE):
        raise RuntimeError("G1-D2 无法证明与 R5 训练随机流分离")
    base, _ = load_s1_config(_project_path(parent["environment_config"]))
    displacements = []
    for wind in WINDS:
        families, _ = _conditions(parent, wind, "original", cfg["wind_speed_multiplier"])
        for family in families:
            condition = RobustnessCondition.from_mapping(dict(family, base_seed=spec["seed_base"]))
            simulation = condition.environment_config(replace(base, episode_length=spec["episode_length"]))
            displacement = (simulation.wind_speed_mps * (1 + simulation.wind_speed_modulation_fraction)
                            * simulation.dt_s * spec["episode_length"] / simulation.sample_pitch_m)
            if displacement >= simulation.turbulence_grid_size:
                raise RuntimeError("相位屏在完整回合内重复")
            displacements.append(displacement)
    output = _project_path(cfg["quick_directory"] if quick else cfg["output_directory"])
    if output.exists():
        raise FileExistsError(f"保留已有 G1-D2 开发输出，不覆盖/重跑: {output}")
    device = resolve_device("cuda")
    report = {
        "status": "READY_FOR_DIAGNOSTIC" if quick else "READY_FOR_USER_IDE",
        "quick": quick, "device": str(device), "weather_count": spec["weather_count"],
        "episode_length": spec["episode_length"], "families": 3, "profile_slots": 6,
        "factorial_cells": 4, "controllers_per_cell": 4,
        "physical_transitions": 4 * 4 * spec["weather_count"] * 3 * 6 * spec["episode_length"],
        "unique_turbulence_streams_per_cell": len(stream_manifest(quick)["turbulence"]),
        "shared_streams_between_all_four_cells": True,
        "maximum_displacement_pixels": displacements,
        "turbulence_grid_pixels": simulation.turbulence_grid_size,
        "original_profile_ids": list(ORIGINAL_PROFILES), "shift_profile_ids": list(SHIFT_PROFILES),
        "profile_slots_are_physical_equivalence": False,
        "selected_scale": SCALE, "integrator_gain": .15, "integrator_leak": .10,
        "config_sha256": _file_sha256(_project_path(path)),
        "entry_sha256": _file_sha256(Path(__file__)),
        "frozen_source_bundle_sha256": FROZEN_SOURCE,
        "g1_d1_config_sha256": D1_CONFIG_SHA256, "g1_d1_entry_sha256": D1_ENTRY_SHA256,
        **cfg["boundary"],
    }
    return cfg, r3_cfg, parent, report, output, device


def _paired_ci(effect: torch.Tensor, draws: torch.Tensor, quantiles: torch.Tensor) -> list[float]:
    """Effect axis order: family, weather; resample whole weather across all families."""
    selected = effect[:, draws]  # family × repeat × weather；全部 family 共用同一天气索引。
    return [float(x) for x in torch.quantile(selected.mean(dim=(0, 2)), quantiles)]


def summarize(rows: list[dict], cfg: dict, *, quick: bool, device: torch.device) -> dict:
    spec = cfg["quick" if quick else "data"]
    weather = [spec["seed_base"] + 10 * index for index in range(spec["weather_count"])]
    controllers = ("integrator",) + tuple(f"policy_{index}_scale_{SCALE}" for index in range(3))
    by_key = {(row["wind_condition"], row["hardware_condition"], row["controller"],
               row["family"], row["slot"], row["weather_seed"]): row for row in rows}
    expected = {(wind, hardware, controller, family, slot, seed)
                for wind in WINDS for hardware in HARDWARE for controller in controllers
                for family in FAMILIES for slot in range(6) for seed in weather}
    if len(rows) != len(expected) or set(by_key) != expected:
        raise RuntimeError("G1-D2 四格配对回合缺失、重复或条件错位")
    for row in rows:
        profiles = ORIGINAL_PROFILES if row["hardware_condition"] == "original" else SHIFT_PROFILES
        if (row["profile"] != profiles[row["slot"]]
                or row["scale"] != (0.0 if row["controller"] == "integrator" else SCALE)
                or row["turbulence_stream_seed"] != row["weather_seed"] + row["slot"] * 1_000 + FAMILIES.index(row["family"])
                or any(not math.isfinite(row[name]) for name in METRICS)
                or not 0 <= row["normalized_correction_clipped_fraction"] <= 1):
            raise RuntimeError("G1-D2 档位、随机流或指标错位")
    values = {name: torch.tensor(
        [[[[[[by_key[(wind, hardware, controller, family, slot, seed)][name]
                for slot in range(6)] for seed in weather] for family in FAMILIES]
            for controller in controllers] for hardware in HARDWARE] for wind in WINDS],
        device=device, dtype=torch.float64) for name in METRICS}
    # wind × hardware × controller × family × weather × slot
    power = values["power"]
    if bool((power[:, :, 0] <= 0).any()):
        raise RuntimeError("G1-D2 积分器桶内功率非正")
    gains = power[:, :, 1:].mean(dim=2) - power[:, :, 0]
    # family × weather × slot；六槽仅作成对设计分组，不暗示两个硬件集合物理等价。
    wind_effect = .5 * ((gains[1, 0] - gains[0, 0]) + (gains[1, 1] - gains[0, 1]))
    hardware_effect = .5 * ((gains[0, 1] - gains[0, 0]) + (gains[1, 1] - gains[1, 0]))
    interaction = (gains[1, 1] - gains[1, 0]) - (gains[0, 1] - gains[0, 0])
    repeats = cfg["statistics"]["bootstrap_repeats"]
    generator = torch.Generator(device=device).manual_seed(cfg["statistics"]["bootstrap_seed"])
    draws = torch.randint(len(weather), (repeats, len(weather)), device=device, generator=generator)
    quantiles = torch.tensor([.025, .975], device=device, dtype=torch.float64)
    effects = {}
    for name, tensor in (("wind_main", wind_effect), ("hardware_main", hardware_effect),
                         ("interaction", interaction)):
        effects[name] = {
            "absolute_policy_increment_effect": float(tensor.mean()),
            "exploratory_paired_ci95": _paired_ci(tensor.mean(dim=-1), draws, quantiles),
            "family_effects": {family: float(tensor[index].mean()) for index, family in enumerate(FAMILIES)},
            "slot_effects": [float(tensor[:, :, slot].mean()) for slot in range(6)],
        }
    cells = {}
    for wi, wind in enumerate(WINDS):
        for hi, hardware in enumerate(HARDWARE):
            baseline = float(power[wi, hi, 0].mean())
            policy = float(power[wi, hi, 1:].mean())
            cell_gains = gains[wi, hi]
            key = f"{wind}__{hardware}"
            cells[key] = {
                "integrator_power": baseline, "policy_power": policy,
                "policy_minus_integrator_power": float(cell_gains.mean()),
                "relative_power_gain": float(cell_gains.mean()) / baseline,
                "exploratory_paired_ci95": _paired_ci(cell_gains.mean(dim=-1), draws, quantiles),
                "member_absolute_gains": [float((power[wi, hi, member + 1] - power[wi, hi, 0]).mean())
                                          for member in range(3)],
                "family_absolute_gains": {family: float(cell_gains[index].mean())
                                          for index, family in enumerate(FAMILIES)},
                "slot_absolute_gains": [float(cell_gains[:, :, slot].mean()) for slot in range(6)],
                "other_deltas": {metric: float((value[wi, hi, 1:] - value[wi, hi, 0]).mean())
                                 for metric, value in values.items() if metric != "power"},
                "raw_metrics": {
                    metric: {"integrator": float(value[wi, hi, 0].mean()),
                             "policy": float(value[wi, hi, 1:].mean())}
                    for metric, value in values.items() if metric != "power"},
                "policy_diagnostics": {
                    metric: float(values[metric][wi, hi, 1:].mean())
                    for metric in ("normalized_correction_clipped_fraction",
                                   "requested_applied_gap_abs", "policy_forward_seconds_per_step",
                                   "policy_forward_p95_seconds")},
                "original_or_shift_profile_ids": (list(ORIGINAL_PROFILES) if hi == 0 else list(SHIFT_PROFILES)),
            }
    return {
        "status": "EXPLORATORY_DEVELOPMENT_NO_CONFIRMATION_GATE",
        "cells": cells, "factorial_effects": effects,
        "paired_unit": "same_weather_across_all_families_slots_and_members; bootstrap_complete_weather",
        "same_random_streams_all_four_cells": True,
        "profile_slot_warning": "槽位只用于配对；原档位与新档位并非物理等价。",
        "multiple_comparisons_adjusted": False,
        "latency_scope": "CUDA policy forward only, not real SLM end-to-end latency",
        "independent_confirmation": False,
    }


def run(path: str | Path = CONFIG, *, quick: bool = False,
        preflight_only: bool = False) -> dict:
    cfg, r3_cfg, parent, report, output, device = preflight(path, quick=quick)
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
    write_json(output / "runtime.json", {
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "git": safe_git_record(),
        "frozen_source_bundle_sha256": FROZEN_SOURCE,
    })
    progress = Progress(output, device)
    started = time.perf_counter()
    try:
        spec = cfg["quick" if quick else "data"]
        base, _ = load_s1_config(_project_path(parent["environment_config"]))
        base = replace(base, num_modes=21, batch_size=1, episode_length=spec["episode_length"])
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
        progress.phase("G1-D2 快速冒烟" if quick else "G1-D2 四格配对开发诊断",
                       4 * len(branches) * spec["weather_count"] * spec["episode_length"])
        rows: list[dict] = []
        with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
            for wind in WINDS:
                for hardware in HARDWARE:
                    families, profiles = _conditions(parent, wind, hardware, cfg["wind_speed_multiplier"])
                    for label, scale, policy in branches:
                        for seed in streams["weather_bases"]:
                            meter = None if policy is None else CorrectionClampTelemetry(policy, scale)
                            episode = _rollout(seed, label, scale, meter, spec["episode_length"],
                                               basis, base, families, profiles,
                                               parent["data"]["sensor_seed_offset"], progress)
                            rates = ([0.0] * len(episode) if meter is None
                                     else meter.rates(spec["episode_length"], len(episode)))
                            for row, rate in zip(episode, rates, strict=True):
                                slot = next(index for index, item in enumerate(profiles)
                                            if item.identifier == row["profile"])
                                row.update({"wind_condition": wind, "hardware_condition": hardware,
                                            "slot": slot,
                                            "normalized_correction_clipped_fraction": rate})
                                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                                rows.append(row)
                            handle.flush()
        expected = 4 * len(branches) * spec["weather_count"] * 3 * 6
        if len(rows) != expected:
            raise RuntimeError("G1-D2 完整分组回合不足")
        analysis = {} if quick else summarize(rows, cfg, quick=False, device=device)
        result = {
            "status": "QUICK_SMOKE_NO_CONCLUSION" if quick
                      else "DEVELOPMENT_DIAGNOSTIC_REQUIRES_AUDIT",
            "records": len(rows), "weather_count": spec["weather_count"],
            "completed_group_episodes": len(rows), "failed_group_episodes": 0,
            "episode_length": spec["episode_length"],
            "physical_transitions": report["physical_transitions"],
            "wind_conditions": list(WINDS), "hardware_conditions": list(HARDWARE),
            "controllers": [label for label, _, _ in branches],
            "elapsed_seconds": time.perf_counter() - started,
            "analysis": analysis,
            "config_sha256": report["config_sha256"], "entry_sha256": report["entry_sha256"],
            "frozen_source_bundle_sha256": FROZEN_SOURCE,
            "g1_d1_config_sha256": D1_CONFIG_SHA256,
            "g1_d1_entry_sha256": D1_ENTRY_SHA256,
            "material_passport": {"origin_skill": "academic-research-suite / experiment-agent",
                                  "origin_mode": "run", "verification_status": "UNVERIFIED"},
            **cfg["boundary"],
            "next_action": "停止并等待只读审计；开发诊断不能改判 G1 或充当独立确认",
        }
        write_json(output / "summary.json", result)
        write_json(output / "SUCCESS.json", {
            "summary_sha256": _file_sha256(output / "summary.json"),
            "records_sha256": _file_sha256(output / "records.jsonl"),
            "progress_sha256": _file_sha256(output / "progress.jsonl"),
            "stream_manifest_sha256": _file_sha256(output / "stream_manifest.json"),
        })
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
    parser.add_argument("--quick", action="store_true", help="16 帧 CUDA 冒烟，不产生性能结论")
    parser.add_argument("--preflight-only", action="store_true", help="只读预检，不生成输出")
    args = parser.parse_args()
    print(json.dumps(run(args.config, quick=args.quick,
                         preflight_only=args.preflight_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
