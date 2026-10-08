"""G1-D3：只补 D2 天气的标称硬件参照臂；正式运行由用户启动。"""
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

from scripts import diagnose_s4_r5_g1_d2_factorial as d2
from scripts import diagnose_s4_r5_g1_hardware_mechanism as d1
from scripts.diagnose_s4_r5_margin_d2 import CorrectionClampTelemetry
from src.rl.r4_dynamics_experiment import Progress, safe_git_record
from src.rl.r4_interface_smoke import write_json
from src.rl.r5_independent_confirmation_r2 import source_bundle_sha256
from src.rl.r5_margin_development import _rollout, effective_stream_seeds
from src.rl.r5_policy_training import ResidualGRUPolicy
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _project_path
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.robust_control import RobustnessCondition


CONFIG = "configs/experiments/s4_r5_g1_d3_nominal_bridge_v1.yaml"
D2_CONFIG_SHA256 = "ab092f22b9a7649ed5a43f1d4fa185c11d49fccdce9c6f5531200f264f9db315"
D2_ENTRY_SHA256 = "5ab735ef8195929f622e63e48dc2d49e6f6beedfd0bd4c74c0cb8f2de52ceaec"
D2_OUTPUT_HASHES = {
    "summary.json": "ebcc3ceef90e648cffeb631be38297cc1ccc18ec7d449da7959666779cf0ceed",
    "records.jsonl": "b5a7d80c6e0d178fc66545bde3861068f4e126469153cae9a19ba64ebd5f1d56",
    "progress.jsonl": "20c808fe688a8ad914022d0c9dfeda8a7d3ba2f624daeef77bd91a4c5cc70150",
    "stream_manifest.json": "09fb204e7fdcab9c24426fe96bc0324b21dd378c7803550bf1edb8919e69a80c",
    "SUCCESS.json": "72bffaf8570db17a333c7cc826052eda5832115fcf913fce368c3af6acabbaa5",
}
WINDS = d2.WINDS
FAMILIES = d2.FAMILIES
HARDWARE = ("original", "shift", "nominal_bridge")
SCALE = d2.SCALE
METRICS = d2.METRICS
FORMAL_BASE = d2.FORMAL_BASE
QUICK_BASE = 6_720_000


def _contract(cfg: dict) -> None:
    if (cfg.get("stage") != "S4-D2-R5-G1-D3"
            or cfg.get("purpose") != "development_only_nominal_bridge_on_d2_weather"
            or cfg.get("runtime") != {"device": "cuda", "formal_owner": "user_ide", "automatic_retry": False}
            or cfg.get("g1_d2_config") != d2.CONFIG
            or cfg.get("g1_d2_config_sha256") != D2_CONFIG_SHA256
            or cfg.get("g1_d2_entry_sha256") != D2_ENTRY_SHA256
            or cfg.get("g1_d2_output") != "outputs/s4_r5_g1_d2_factorial_v1"
            or cfg.get("g1_d2_output_hashes") != D2_OUTPUT_HASHES
            or cfg.get("selected_scale") != SCALE
            or cfg.get("wind_speed_multiplier") != 1.10
            or cfg.get("data") != {"seed_base": FORMAL_BASE, "seed_stride": 10,
                                   "weather_count": 24, "episode_length": 200}
            or cfg.get("quick") != {"seed_base": QUICK_BASE, "weather_count": 1,
                                    "episode_length": 16}
            or cfg.get("statistics") != {"bootstrap_seed": 6_733_456, "bootstrap_repeats": 5_000}
            or cfg.get("output_directory") != "outputs/s4_r5_g1_d3_nominal_bridge_v1"
            or cfg.get("quick_directory") != "outputs/s4_r5_g1_d3_nominal_bridge_v1_quick"
            or cfg.get("boundary") != {"confirmation_access": False, "training_updates": 0,
                                       "real_slm_actions": False, "automatic_retry": False,
                                       "g1_d2_results_read_only": True,
                                       "reuse_d2_development_weather": True}):
        raise ValueError("G1-D3 共同标称参照合同被改变")


def _verify_lineage(cfg: dict) -> tuple[dict, dict, dict]:
    if _file_sha256(_project_path(cfg["g1_d2_config"])) != D2_CONFIG_SHA256:
        raise RuntimeError("G1-D2 冻结配置已改变")
    if _file_sha256(_project_path("scripts/diagnose_s4_r5_g1_d2_factorial.py")) != D2_ENTRY_SHA256:
        raise RuntimeError("G1-D2 冻结入口已改变")
    d2_cfg = _load_yaml(_project_path(cfg["g1_d2_config"]))
    d2._contract(d2_cfg)
    _, r3_cfg, parent = d2._verify_lineage(d2_cfg)
    if source_bundle_sha256() != d2.FROZEN_SOURCE:
        raise RuntimeError("冻结物理或控制源码已改变")
    root = _project_path(cfg["g1_d2_output"])
    if (root / "failure.json").exists():
        raise RuntimeError("G1-D2 历史输出存在失败标记")
    for name, digest in D2_OUTPUT_HASHES.items():
        if _file_sha256(root / name) != digest:
            raise RuntimeError(f"G1-D2 历史证据已改变: {name}")
    success = json.loads((root / "SUCCESS.json").read_text(encoding="utf-8"))
    for key, name in (("summary_sha256", "summary.json"), ("records_sha256", "records.jsonl"),
                      ("progress_sha256", "progress.jsonl"),
                      ("stream_manifest_sha256", "stream_manifest.json")):
        if success.get(key) != D2_OUTPUT_HASHES[name]:
            raise RuntimeError("G1-D2 成功标记与证据不符")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("status") != "DEVELOPMENT_DIAGNOSTIC_REQUIRES_AUDIT"
            or summary.get("records") != 6912 or summary.get("failed_group_episodes") != 0
            or summary.get("weather_count") != 24 or summary.get("episode_length") != 200
            or summary.get("confirmation_access") is not False
            or summary.get("real_slm_actions") is not False
            or summary.get("training_updates") != 0
            or summary.get("config_sha256") != D2_CONFIG_SHA256
            or summary.get("entry_sha256") != D2_ENTRY_SHA256
            or summary.get("frozen_source_bundle_sha256") != d2.FROZEN_SOURCE):
        raise RuntimeError("G1-D2 已审计状态与冻结证据不符")
    recorded = json.loads((root / "stream_manifest.json").read_text(encoding="utf-8"))
    if recorded != d2.stream_manifest(False):
        raise RuntimeError("G1-D2 天气随机流清单不符")
    return d2_cfg, r3_cfg, parent


def _manifest(quick: bool) -> dict[str, list[int]]:
    if not quick:
        return d2.stream_manifest(False)
    weather = [QUICK_BASE]
    turbulence = effective_stream_seeds(QUICK_BASE, 1, 10, 6, 3)
    sensor = [QUICK_BASE + 1_000 * slot + 50_000_000 for slot in range(6)]
    power = [QUICK_BASE + 1_000 * slot + 60_000_000 for slot in range(6)]
    return {"weather_bases": weather, "turbulence": turbulence,
            "sensor": sensor, "power": power}


def _nominal_profiles(parent: dict) -> list:
    clones, _ = d1._profile_pairs(parent)
    if len(clones) != 6 or len({item.identifier for item in clones}) != 6:
        raise RuntimeError("六槽标称硬件克隆不完整")
    return clones


def preflight(path: str | Path = CONFIG, *, quick: bool = False
              ) -> tuple[dict, dict, dict, dict, Path, torch.device]:
    cfg = _load_yaml(_project_path(path))
    _contract(cfg)
    d2_cfg, r3_cfg, parent = _verify_lineage(cfg)
    if (parent["profile_ids"] != list(d2.ORIGINAL_PROFILES)
            or [item["id"] for item in parent["families"]] != list(FAMILIES)
            or parent["data"]["sensor_seed_offset"] != 50_000_000):
        raise RuntimeError("R5 冻结环境或随机流合同已改变")
    _nominal_profiles(parent)
    selected, smoke = _manifest(quick), _manifest(True)
    formal = _manifest(False)
    for name in ("turbulence", "sensor", "power"):
        if set(formal[name]) & set(smoke[name]):
            raise RuntimeError("G1-D3 快速冒烟与 D2 开发天气冲突")
    spec = cfg["quick" if quick else "data"]
    base, _ = load_s1_config(_project_path(parent["environment_config"]))
    displacements = []
    for wind in WINDS:
        families, _ = d2._conditions(parent, wind, "original", cfg["wind_speed_multiplier"])
        for family in families:
            condition = RobustnessCondition.from_mapping(dict(family, base_seed=spec["seed_base"]))
            simulation = condition.environment_config(replace(base, episode_length=spec["episode_length"]))
            displacement = (simulation.wind_speed_mps * (1 + simulation.wind_speed_modulation_fraction)
                            * simulation.dt_s * spec["episode_length"] / simulation.sample_pitch_m)
            if displacement >= simulation.turbulence_grid_size:
                raise RuntimeError("相位屏在回合内重复")
            displacements.append(displacement)
    output = _project_path(cfg["quick_directory"] if quick else cfg["output_directory"])
    if output.exists():
        raise FileExistsError(f"保留已有 G1-D3 输出，不覆盖或重跑: {output}")
    device = resolve_device("cuda")
    report = {
        "status": "READY_FOR_DIAGNOSTIC" if quick else "READY_FOR_USER_IDE",
        "quick": quick, "device": str(device), "weather_count": spec["weather_count"],
        "episode_length": spec["episode_length"], "wind_conditions": list(WINDS),
        "profile_slots": 6, "families": 3, "controllers": 4,
        "physical_transitions": 2 * 4 * spec["weather_count"] * 3 * 6 * spec["episode_length"],
        "reuses_d2_development_weather": not quick,
        "random_streams_shared_with_d2_by_design": not quick,
        "quick_streams_disjoint_from_d2": True,
        "maximum_displacement_pixels": displacements,
        "turbulence_grid_pixels": simulation.turbulence_grid_size,
        "config_sha256": _file_sha256(_project_path(path)),
        "entry_sha256": _file_sha256(Path(__file__)),
        "g1_d2_config_sha256": D2_CONFIG_SHA256, "g1_d2_entry_sha256": D2_ENTRY_SHA256,
        "frozen_source_bundle_sha256": d2.FROZEN_SOURCE,
        **cfg["boundary"],
    }
    return cfg, r3_cfg, parent, report, output, device


def _effect_tensors(gain: torch.Tensor) -> dict[str, torch.Tensor]:
    """Input order: wind × (old, new, nominal) × family × weather × slot."""
    old_nominal = gain[:, 0] - gain[:, 2]
    new_nominal = gain[:, 1] - gain[:, 2]
    new_old = gain[:, 1] - gain[:, 0]
    if not torch.allclose(new_nominal - old_nominal, new_old, atol=1e-12, rtol=0):
        raise RuntimeError("G1-D3 标称桥接恒等式不成立")
    return {"old_minus_nominal": old_nominal, "new_minus_nominal": new_nominal,
            "new_minus_old": new_old}


def _ci(effect: torch.Tensor, draws: torch.Tensor) -> list[float]:
    """family × weather；同一被抽中的完整天气携带全部湍流类别。"""
    selected = torch.gather(effect[None].expand(draws.shape[0], -1, -1), 2,
                            draws[:, None, :].expand(-1, effect.shape[0], -1))
    quantiles = torch.tensor([.025, .975], device=effect.device, dtype=torch.float64)
    return [float(value) for value in torch.quantile(selected.mean(dim=(1, 2)), quantiles)]


def summarize(new_rows: list[dict], cfg: dict, device: torch.device) -> dict:
    d2_path = _project_path(cfg["g1_d2_output"]) / "records.jsonl"
    with d2_path.open(encoding="utf-8") as handle:
        historical = [json.loads(line) for line in handle]
    all_rows = historical + new_rows
    weather = _manifest(False)["weather_bases"]
    controllers = ("integrator",) + tuple(f"policy_{i}_scale_{SCALE}" for i in range(3))
    key = lambda row: (row["wind_condition"], row["hardware_condition"], row["controller"],
                       row["family"], row["slot"], row["weather_seed"])
    by_key = {key(row): row for row in all_rows}
    expected = {(wind, hardware, controller, family, slot, seed)
                for wind in WINDS for hardware in HARDWARE for controller in controllers
                for family in FAMILIES for slot in range(6) for seed in weather}
    if len(all_rows) != len(expected) or len(by_key) != len(expected) or set(by_key) != expected:
        raise RuntimeError("G1-D3 三种硬件参照的配对记录缺失或重复")
    # 标称槽位名称由 D1 冻结入口定义；这里不从历史观测中挑选槽位。
    nominal = [f"nominal_for_{item}" for item in d2.SHIFT_PROFILES]
    profile_lists = {"original": d2.ORIGINAL_PROFILES,
                     "shift": d2.SHIFT_PROFILES, "nominal_bridge": nominal}
    for row in all_rows:
        if (row["profile"] != profile_lists[row["hardware_condition"]][row["slot"]]
                or row["turbulence_stream_seed"] != row["weather_seed"] + 1000 * row["slot"] + FAMILIES.index(row["family"])
                or row["scale"] != (0.0 if row["controller"] == "integrator" else SCALE)
                or any(not math.isfinite(row[name]) for name in METRICS)
                or not 0 <= row["normalized_correction_clipped_fraction"] <= 1):
            raise RuntimeError("G1-D3 硬件槽位、随机流或指标错位")
    values = {name: torch.tensor(
        [[[[[[by_key[(wind, hardware, controller, family, slot, seed)][name]
                for slot in range(6)] for seed in weather] for family in FAMILIES]
            for controller in controllers] for hardware in HARDWARE] for wind in WINDS],
        device=device, dtype=torch.float64) for name in METRICS}
    power = values["power"]
    if bool((power[:, :, 0] <= 0).any()):
        raise RuntimeError("G1-D3 积分器桶内功率非正")
    gain = power[:, :, 1:].mean(dim=2) - power[:, :, 0]
    effects = _effect_tensors(gain)
    d2_summary = json.loads((_project_path(cfg["g1_d2_output"]) / "summary.json").read_text(encoding="utf-8"))
    for wi, wind in enumerate(WINDS):
        for hi, hardware in enumerate(HARDWARE[:2]):
            old_cell = d2_summary["analysis"]["cells"][f"{wind}__{hardware}"]
            if abs(float(gain[wi, hi].mean()) - old_cell["policy_minus_integrator_power"]) > 1e-12:
                raise RuntimeError("G1-D3 与 D2 冻结功率摘要不对齐")
    repeats = cfg["statistics"]["bootstrap_repeats"]
    generator = torch.Generator(device=device).manual_seed(cfg["statistics"]["bootstrap_seed"])
    draws = torch.randint(len(weather), (repeats, len(weather)), device=device, generator=generator)
    cells = {}
    for wi, wind in enumerate(WINDS):
        for hi, hardware in enumerate(HARDWARE):
            baseline = float(power[wi, hi, 0].mean())
            policy = float(power[wi, hi, 1:].mean())
            cells[f"{wind}__{hardware}"] = {
                "integrator_power": baseline, "policy_power": policy,
                "policy_minus_integrator_power": float(gain[wi, hi].mean()),
                "relative_power_gain": float(gain[wi, hi].mean()) / baseline,
                "strehl_delta": float((values["strehl"][wi, hi, 1:] - values["strehl"][wi, hi, 0]).mean()),
                "phase_rmse_delta": float((values["phase_rmse"][wi, hi, 1:] - values["phase_rmse"][wi, hi, 0]).mean()),
                "violation_delta": float((values["violation"][wi, hi, 1:] - values["violation"][wi, hi, 0]).mean()),
                "saturation_delta": float((values["saturation"][wi, hi, 1:] - values["saturation"][wi, hi, 0]).mean()),
                "slew_limited_delta": float((values["slew_limited"][wi, hi, 1:] - values["slew_limited"][wi, hi, 0]).mean()),
                "requested_applied_gap_delta": float((values["requested_applied_gap_abs"][wi, hi, 1:] - values["requested_applied_gap_abs"][wi, hi, 0]).mean()),
                "policy_clipped_fraction": float(values["normalized_correction_clipped_fraction"][wi, hi, 1:].mean()),
                "policy_forward_seconds_per_step": float(values["policy_forward_seconds_per_step"][wi, hi, 1:].mean()),
                "policy_forward_p95_seconds": float(values["policy_forward_p95_seconds"][wi, hi, 1:].mean()),
                "other_deltas": {
                    metric: float((value[wi, hi, 1:] - value[wi, hi, 0]).mean())
                    for metric, value in values.items() if metric != "power"},
                "raw_metrics": {
                    metric: {"integrator": float(value[wi, hi, 0].mean()),
                             "policy": float(value[wi, hi, 1:].mean())}
                    for metric, value in values.items() if metric != "power"},
                "family_absolute_gains": {family: float(gain[wi, hi, fi].mean())
                                          for fi, family in enumerate(FAMILIES)},
                "slot_absolute_gains": [float(gain[wi, hi, :, :, slot].mean()) for slot in range(6)],
            }
    contrasts = {}
    for name, tensor in effects.items():
        contrasts[name] = {
            "overall_absolute_advantage_change": float(tensor.mean()),
            "exploratory_paired_ci95": _ci(tensor.mean(dim=(0, 3)), draws),
            "by_wind": {wind: {"absolute_advantage_change": float(tensor[wi].mean()),
                               "exploratory_paired_ci95": _ci(tensor[wi].mean(dim=-1), draws),
                               "family_effects": {family: float(tensor[wi, fi].mean())
                                                  for fi, family in enumerate(FAMILIES)},
                               "slot_effects": [float(tensor[wi, :, :, slot].mean())
                                                for slot in range(6)]}
                        for wi, wind in enumerate(WINDS)},
        }
    return {
        "status": "EXPLORATORY_DEVELOPMENT_NO_CONFIRMATION_GATE",
        "cells": cells, "contrasts": contrasts,
        "bridge_identity_checked_per_weather_family_slot": True,
        "nominal_clones_share_physical_parameters": True,
        "paired_unit": "same_weather_across_all_families_slots_and_members; bootstrap_complete_weather",
        "weather_was_previously_used_for_d2_development": True,
        "profile_slot_warning": "不同硬件档位并非物理等价；槽位仅用于随机流配对。",
        "multiple_comparisons_adjusted": False,
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
    streams = _manifest(quick)
    write_json(output / "stream_manifest.json", streams)
    write_json(output / "runtime.json", {
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "git": safe_git_record(),
        "frozen_source_bundle_sha256": d2.FROZEN_SOURCE,
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
        profiles = _nominal_profiles(parent)
        progress.phase("G1-D3 快速冒烟" if quick else "G1-D3 标称参照开发诊断",
                       2 * len(branches) * spec["weather_count"] * spec["episode_length"])
        rows: list[dict] = []
        with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
            for wind in WINDS:
                families, _ = d2._conditions(parent, wind, "original", cfg["wind_speed_multiplier"])
                for label, scale, policy in branches:
                    for seed in streams["weather_bases"]:
                        meter = None if policy is None else CorrectionClampTelemetry(policy, scale)
                        episode = _rollout(seed, label, scale, meter, spec["episode_length"],
                                           basis, base, families, profiles,
                                           parent["data"]["sensor_seed_offset"], progress)
                        rates = ([0.0] * len(episode) if meter is None
                                 else meter.rates(spec["episode_length"], len(episode)))
                        for row, rate in zip(episode, rates, strict=True):
                            slot = next(i for i, item in enumerate(profiles)
                                        if item.identifier == row["profile"])
                            row.update({"wind_condition": wind, "hardware_condition": "nominal_bridge",
                                        "slot": slot, "normalized_correction_clipped_fraction": rate})
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                            rows.append(row)
                        handle.flush()
        expected = 2 * len(branches) * spec["weather_count"] * 3 * 6
        if len(rows) != expected:
            raise RuntimeError("G1-D3 标称参照分组回合不足")
        analysis = {} if quick else summarize(rows, cfg, device)
        result = {
            "status": "QUICK_SMOKE_NO_CONCLUSION" if quick else "DEVELOPMENT_BRIDGE_REQUIRES_AUDIT",
            "records": len(rows), "completed_group_episodes": len(rows), "failed_group_episodes": 0,
            "weather_count": spec["weather_count"], "episode_length": spec["episode_length"],
            "physical_transitions": report["physical_transitions"],
            "wind_conditions": list(WINDS), "new_hardware_condition": "nominal_bridge",
            "controllers": [label for label, _, _ in branches],
            "elapsed_seconds": time.perf_counter() - started, "analysis": analysis,
            "config_sha256": report["config_sha256"], "entry_sha256": report["entry_sha256"],
            "frozen_source_bundle_sha256": d2.FROZEN_SOURCE,
            "g1_d2_config_sha256": D2_CONFIG_SHA256, "g1_d2_entry_sha256": D2_ENTRY_SHA256,
            "material_passport": {"origin_skill": "academic-research-suite / experiment-agent",
                                  "origin_mode": "run", "verification_status": "UNVERIFIED"},
            **cfg["boundary"],
            "next_action": "停止并等待只读审计；共同标称参照是开发诊断，不改判 G1",
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
