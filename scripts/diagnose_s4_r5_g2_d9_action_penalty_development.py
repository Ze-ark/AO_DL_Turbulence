"""G2-D9：新天气评价动作惩罚单因素训练；正式评价由用户在 IDE 启动。"""
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

from scripts import diagnose_s4_r5_g2_d3_matched_development as d3
from scripts import train_s4_r5_g2_d8_action_penalty_zero as d8
from scripts.diagnose_s4_r5_margin_d2 import CorrectionClampTelemetry
from src.rl.r4_dynamics_experiment import safe_git_record
from src.rl.r4_interface_smoke import write_json
from src.rl.r5_margin_development import _rollout
from src.rl.r5_policy_training import ResidualGRUPolicy
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _project_path
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.robust_control import RobustnessCondition


CONFIG = "configs/experiments/s4_r5_g2_d9_action_penalty_development_v1.yaml"
CONDITIONS = d3.CONDITIONS
FAMILIES = d3.FAMILIES
PROFILES = d3.PROFILES
SCALE = 1.75
FORMAL_BASE, QUICK_BASE = 7_500_000, 7_520_000
WEATHER_COUNT = 32
CONTROLLERS = (("integrator",) + tuple(f"{d8.COMPARATOR}_{i}" for i in range(3))
               + tuple(f"{d8.ARM}_{i}" for i in range(3)))
METRICS = d3.METRICS
D8_CONFIG_SHA256 = "34a592e3895e144f6b5f369330c183399a2aadb2251eaf3ef0ed38f54150e9ac"
D8_ENTRY_SHA256 = "6d91c8b7669c4f3aeaa9624297cbffca197c430587285b6f752f9866d1ab0e57"
D8_HASHES = {
    "summary.json": "95dc607e06fd00493f450a122df7314004e11cbf681ccb3d5a51ca31ad178961",
    "losses.jsonl": "48cf5c0db9d41f92df5b254d0891e4cfda1ea176e3dd5b60c79538d04e7e7027",
    "progress.jsonl": "ee20f5eec693adfa44a3342103200c15de96664e67b5e0a5779f58920be2b193",
    "stream_manifest.json": "6312e5ec18ba93685dd4383c3ddc3689f4297ea60c28bb3a819a97adac026f58",
    "checkpoint_manifest.json": "3e5b738b7b18c4a38fc1150c65fb84a96a37451d464e97111b7a0d352bb2d9b7",
    "SUCCESS.json": "11f92fd8f1a1247a2c8529e8e7a4536a313b96bfb42d4053142f1b3e195aa94c",
    "runtime.json": "1ac4ba074a15184a46793fd2be80dde6eea1d8ea00653ab1a5542edfa2597867",
    "config.json": "9d69da3509ffec2e305a3c962ea5e35077a7d4ab5cd88d2b55c3b05c715a1e88",
    "preflight.json": "409ebce9aca1631688851c53dbea3a7affd4d20112ffc6bace523b787ad8649e",
}
D8_FINAL = {
    "action_penalty_0_policy_0_00512.pt": "af2453ef2b5c20f231d7286f0d6c597b258f944e9a295c2e1470d8a4b79b2523",
    "action_penalty_0_policy_1_00512.pt": "71ab8b375b4b942a37e567906891ec1fbf4e2dd335c9677233d4dac4bea4a471",
    "action_penalty_0_policy_2_00512.pt": "4eac6814f28d7ef49a4ebe933342bb86e77c46a64d80cd7eaf8f5a5434e3ee94",
}
HELPER_HASHES = {
    "scripts/diagnose_s4_r5_g2_d3_matched_development.py": "91c40144b47845118876052e7e68009ae402c218544e76c6b6cf195edf5322ca",
    "scripts/diagnose_s4_r5_margin_d2.py": "5205e3cc6772a11b0d2df33d807ca08efa70bdab7eb2aaa1cac4acfb8051719d",
}


def _contract(cfg: dict) -> None:
    expected = {
        "stage": "S4-D2-R5-G2-D9",
        "purpose": "fresh_weather_action_penalty_single_factor_closed_loop_development",
        "runtime": {"device": "cuda", "formal_owner": "user_ide", "automatic_retry": False},
        "d8_config": d8.CONFIG, "d8_config_sha256": D8_CONFIG_SHA256,
        "d8_entry_sha256": D8_ENTRY_SHA256,
        "d8_output": "outputs/s4_r5_g2_d8_action_penalty_zero_v1",
        "d8_output_hashes": D8_HASHES, "d8_final_checkpoints": D8_FINAL,
        "evaluation_helpers_sha256": HELPER_HASHES,
        "source_bundle_sha256": d8.SOURCE_BUNDLE_SHA256,
        "controller_ids": list(CONTROLLERS), "deployment_scale": SCALE,
        "hardware_conditions": list(CONDITIONS),
        "data": {"seed_base": FORMAL_BASE, "seed_stride": 10,
                 "weather_count": WEATHER_COUNT, "episode_length": 200},
        "quick": {"seed_base": QUICK_BASE, "weather_count": 1, "episode_length": 16},
        "reserved_confirmation_seed_base": d8.CONFIRMATION_BASE,
        "statistics": {"bootstrap_seed": 7_533_456, "bootstrap_repeats": 5_000,
                       "primary_interval_per_condition": .975},
        "thresholds": {"minimum_relative_gain": .0105, "maximum_safety_increase": .001},
        "output_directory": "outputs/s4_r5_g2_d9_action_penalty_development_v1",
        "quick_directory": "outputs/s4_r5_g2_d9_action_penalty_development_v1_quick",
        "boundary": {"confirmation_access": False, "training_updates": 0,
                     "real_slm_actions": False, "automatic_retry": False,
                     "historical_results_read_only": True},
    }
    if cfg != expected:
        raise ValueError("G2-D9 冻结开发评价合同已改变")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_training(cfg: dict) -> tuple[dict, dict, dict[int, str]]:
    """只读验证新旧训练证据；从未调用训练或独立确认入口。"""
    if (_file_sha256(_project_path(d8.CONFIG)) != D8_CONFIG_SHA256
            or _file_sha256(Path(d8.__file__)) != D8_ENTRY_SHA256):
        raise RuntimeError("G2-D8 冻结配置或训练入口变化")
    for path, digest in HELPER_HASHES.items():
        if _file_sha256(_project_path(path)) != digest:
            raise RuntimeError(f"G2-D9 冻结评价工具变化: {path}")
    training_cfg = _load_yaml(_project_path(d8.CONFIG))
    d8._contract(training_cfg)
    _, r3_cfg, parent, sources, comparators = d8._verify_d2(training_cfg)
    output = _project_path(cfg["d8_output"])
    if (output / "failure.json").exists():
        raise RuntimeError("G2-D8 训练有失败标记")
    for name, digest in D8_HASHES.items():
        if _file_sha256(output / name) != digest:
            raise RuntimeError(f"G2-D8 训练证据变化: {name}")
    success = _read_json(output / "SUCCESS.json")
    for key, name in (("summary_sha256", "summary.json"), ("losses_sha256", "losses.jsonl"),
                      ("progress_sha256", "progress.jsonl"),
                      ("stream_manifest_sha256", "stream_manifest.json"),
                      ("checkpoint_manifest_sha256", "checkpoint_manifest.json")):
        if success.get(key) != D8_HASHES[name]:
            raise RuntimeError("G2-D8 成功标记与训练证据不符")
    summary = _read_json(output / "summary.json")
    if (summary.get("status") != "DEVELOPMENT_TRAINING_COMPLETE_REQUIRES_AUDIT"
            or summary.get("arm") != d8.ARM or summary.get("comparator") != d8.COMPARATOR
            or summary.get("initializations") != 3 or summary.get("training_updates") != 1536
            or summary.get("physical_transitions") != 5_529_600 or summary.get("checkpoints") != 24
            or summary.get("config_sha256") != D8_CONFIG_SHA256
            or summary.get("entry_sha256") != D8_ENTRY_SHA256
            or summary.get("frozen_source_bundle_sha256") != d8.SOURCE_BUNDLE_SHA256
            or summary.get("source_checkpoint_sha256") != {str(k): v for k, v in sources.items()}
            or summary.get("comparator_final_checkpoint_sha256") != {str(k): v for k, v in comparators.items()}
            or any(summary.get(key) is not False for key in
                   ("confirmation_access", "development_evaluation_access", "real_slm_actions"))):
        raise RuntimeError("G2-D8 训练预算、来源或边界不符")
    if (_read_json(output / "config.json") != training_cfg
            or _read_json(output / "stream_manifest.json") != d8.stream_manifest(quick=False)):
        raise RuntimeError("G2-D8 保存配置或天气流不符")
    runtime = _read_json(output / "runtime.json")
    if ({key: runtime.get(key) for key in d8.REFERENCE_RUNTIME} != d8.REFERENCE_RUNTIME
            or runtime.get("deterministic_algorithms") is not True
            or runtime.get("allow_tf32") is not False):
        raise RuntimeError("G2-D8 运行环境不符")
    manifest = _read_json(output / "checkpoint_manifest.json")
    expected = {f"{d8.ARM}_policy_{member}_{update:05d}.pt"
                for member in range(3) for update in range(64, 513, 64)}
    if set(manifest) != expected or {p.name for p in (output / "checkpoints").iterdir()} != expected:
        raise RuntimeError("G2-D8 检查点网格不完整或有额外文件")
    for name, digest in manifest.items():
        path = output / "checkpoints" / name
        if _file_sha256(path) != digest:
            raise RuntimeError(f"G2-D8 检查点变化: {name}")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        member, update = int(name.split("_")[-2]), int(name.split("_")[-1].split(".")[0])
        if (saved.get("arm") != d8.ARM or saved.get("init") != member
                or saved.get("update") != update or saved.get("action_weight") != 0.0
                or saved.get("training_scale") != SCALE or saved.get("deployment_scale") != SCALE
                or saved.get("config_sha256") != D8_CONFIG_SHA256
                or saved.get("source_checkpoint_sha256") != sources[member]
                or saved.get("comparator_final_checkpoint_sha256") != comparators[member]
                or not all(bool(torch.isfinite(t).all()) for t in saved["state_dict"].values())):
            raise RuntimeError(f"G2-D8 检查点身份或参数不符: {name}")
    if any(manifest.get(name) != digest for name, digest in D8_FINAL.items()):
        raise RuntimeError("G2-D8 冻结末次权重变化")
    # 验证每份初始化都覆盖 1..512，且日志包含实际训练条件和天气。
    losses: dict[int, list[float]] = {member: [] for member in range(3)}
    rows = [json.loads(line) for line in (output / "losses.jsonl").read_text(encoding="utf-8").splitlines()]
    progress = [json.loads(line) for line in (output / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != 1536 or len(progress) != 1536:
        raise RuntimeError("G2-D8 训练日志数量不足")
    for index, (row, tick) in enumerate(zip(rows, progress, strict=True)):
        member, zero_update = divmod(index, 512)
        update = zero_update + 1
        condition, seed = d8.schedule(update, quick=False)
        if (row.get("initialization") != member or row.get("update") != update
                or row.get("condition") != condition or row.get("weather_seed") != seed
                or row.get("arm") != d8.ARM or row.get("action_weight") != 0.0
                or row.get("training_scale") != SCALE or row.get("deployment_scale") != SCALE
                or not math.isfinite(row["loss"])
                or tick.get("completed") != index + 1 or tick.get("total") != 1536):
            raise RuntimeError("G2-D8 更新顺序、天气或损失不符")
        losses[member].append(row["loss"])
    for member, result in enumerate(summary["results"]):
        if (result.get("initialization") != member or result.get("updates") != 512
                or not math.isclose(result["mean_training_loss"], sum(losses[member]) / 512,
                                    abs_tol=1e-10)):
            raise RuntimeError("G2-D8 汇总损失与日志不符")
    return r3_cfg, parent, comparators


def stream_manifest(quick: bool) -> dict:
    base, count = (QUICK_BASE, 1) if quick else (FORMAL_BASE, WEATHER_COUNT)
    weather = [base + 10 * i for i in range(count)]
    streams = {
        "weather_bases": weather,
        "turbulence": [seed + 1000 * slot + family for seed in weather
                       for slot in range(6) for family in range(3)],
        "sensor": [seed + 1000 * slot + 50_000_000 for seed in weather for slot in range(6)],
        "power": [seed + 1000 * slot + 60_000_000 for seed in weather for slot in range(6)],
        "shared_between_controllers_and_conditions": True,
    }
    if any(len(set(streams[name])) != count * width
           for name, width in (("turbulence", 18), ("sensor", 6), ("power", 6))):
        raise RuntimeError("G2-D9 随机流碰撞")
    return streams


def _verify_stream_separation() -> None:
    formal, smoke = stream_manifest(False), stream_manifest(True)
    old = (d8.stream_manifest(quick=False), d8.stream_manifest(quick=True),
           d8.d2.stream_manifest(quick=True), d3.stream_manifest(False), d3.stream_manifest(True),
           d8.d2.g2.stream_manifest(False), d8.d2.g2.stream_manifest(True))
    # D4-D7 的随机流全部位于 7,300,000..7,369,999；只验证区间，不打开结果。
    for name, offset in (("turbulence", 0), ("sensor", 50_000_000), ("power", 60_000_000)):
        prior = set().union(*(set(item[name]) for item in old))
        new = set(formal[name]) | set(smoke[name])
        if (set(formal[name]) & set(smoke[name]) or prior & new
                or any(7_300_000 + offset <= seed < 7_370_000 + offset for seed in new)
                or any(seed >= d8.CONFIRMATION_BASE + offset for seed in new)):
            raise RuntimeError(f"G2-D9 {name} 流与旧阶段或预留确认天气重叠")


def preflight(path: str | Path = CONFIG, *, quick: bool = False
              ) -> tuple[dict, dict, dict, Path, torch.device]:
    cfg = _load_yaml(_project_path(path))
    _contract(cfg)
    _, parent, comparators = _verify_training(cfg)
    _verify_stream_separation()
    if (parent["data"]["sensor_seed_offset"] != 50_000_000
            or [item["id"] for item in parent["families"]] != list(FAMILIES)):
        raise RuntimeError("G2-D9 因果观测或湍流家族变化")
    nominal, shifted = d8.d2.g2.d1._profile_pairs(parent)
    if ([p.identifier for p in shifted] != list(PROFILES)
            or [p.identifier for p in nominal] != [f"nominal_for_{p}" for p in PROFILES]):
        raise RuntimeError("G2-D9 标称与新硬件槽位不符")
    spec = cfg["quick" if quick else "data"]
    base, _ = load_s1_config(_project_path(parent["environment_config"]))
    displacement = []
    for family in parent["families"]:
        condition = RobustnessCondition.from_mapping(dict(family, base_seed=spec["seed_base"]))
        simulation = condition.environment_config(replace(base, episode_length=spec["episode_length"]))
        pixels = (simulation.wind_speed_mps * (1 + simulation.wind_speed_modulation_fraction)
                  * simulation.dt_s * spec["episode_length"] / simulation.sample_pitch_m)
        if pixels >= simulation.turbulence_grid_size:
            raise RuntimeError("G2-D9 相位屏在完整回合内重复")
        displacement.append(pixels)
    output = _project_path(cfg["quick_directory" if quick else "output_directory"])
    if output.exists():
        raise FileExistsError(f"保留已有 G2-D9 输出，不覆盖或重跑: {output}")
    device = resolve_device("cuda")
    runtime = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(device)}
    if runtime != d8.REFERENCE_RUNTIME:
        raise RuntimeError("G2-D9 运行环境与冻结训练环境不符")
    episodes = len(CONDITIONS) * len(CONTROLLERS) * spec["weather_count"] * len(FAMILIES) * 6
    report = {
        "status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_IDE",
        "quick": quick, "device": str(device), "weather_count": spec["weather_count"],
        "episode_length": spec["episode_length"], "complete_episodes": episodes,
        "physical_transitions": episodes * spec["episode_length"],
        "controller_ids": list(CONTROLLERS), "hardware_conditions": list(CONDITIONS),
        "config_sha256": _file_sha256(_project_path(path)),
        "entry_sha256": _file_sha256(Path(__file__)),
        "d8_training_output_hashes": D8_HASHES, "d8_final_checkpoints": D8_FINAL,
        "d2_final_checkpoints": {str(k): v for k, v in comparators.items()},
        "frozen_source_bundle_sha256": d8.SOURCE_BUNDLE_SHA256,
        "maximum_displacement_pixels": displacement,
        "turbulence_grid_pixels": simulation.turbulence_grid_size,
        **cfg["boundary"],
    }
    return cfg, parent, report, output, device


def summarize(rows: list[dict], cfg: dict, *, device: torch.device) -> dict:
    """同一完整天气聚类；三初始化、三湍流、六槽不当作独立天气。"""
    weather = stream_manifest(False)["weather_bases"]
    keys = {(r["hardware_condition"], r["controller"], r["family"], r["slot"],
             r["weather_seed"]): r for r in rows}
    expected = {(condition, controller, family, slot, seed)
                for condition in CONDITIONS for controller in CONTROLLERS
                for family in FAMILIES for slot in range(6) for seed in weather}
    if len(rows) != len(expected) or set(keys) != expected:
        raise RuntimeError("G2-D9 完整回合配对缺失或重复")
    for row in rows:
        slot = row["slot"]
        profile = f"nominal_for_{PROFILES[slot]}" if row["hardware_condition"] == "nominal_clone" else PROFILES[slot]
        if (row["profile"] != profile or row["episode_length"] != 200
                or row["scale"] != (0.0 if row["controller"] == "integrator" else SCALE)
                or row["turbulence_stream_seed"] != row["weather_seed"] + 1000 * slot
                + FAMILIES.index(row["family"])
                or any(not math.isfinite(row[name]) for name in METRICS)
                or any(not 0 <= row[name] <= 1 for name in
                       ("violation", "saturation", "slew_limited", "normalized_correction_clipped_fraction"))):
            raise RuntimeError("G2-D9 回合长度、档位、随机流、动作或指标错位")
    values = {name: torch.tensor(
        [[[[[keys[(condition, controller, family, slot, seed)][name]
              for slot in range(6)] for seed in weather] for family in FAMILIES]
          for controller in CONTROLLERS] for condition in CONDITIONS],
        dtype=torch.float64, device=device) for name in METRICS}
    # condition × controller × family × weather × slot。
    power = values["power"]
    if bool((power[:, 0] <= 0).any()):
        raise RuntimeError("G2-D9 积分器桶内功率非正")
    generator = torch.Generator(device=device).manual_seed(cfg["statistics"]["bootstrap_seed"])
    draws = torch.randint(len(weather), (cfg["statistics"]["bootstrap_repeats"], len(weather)),
                          device=device, generator=generator)
    cells = {}
    for index, condition in enumerate(CONDITIONS):
        baseline, old, new = power[index, 0], power[index, 1:4], power[index, 4:7]
        effect = new - old
        baseline_mean = float(baseline.mean())
        relative = float((new - baseline).mean()) / baseline_mean
        interval = d3._paired_interval(effect.mean(dim=(0, 1, 3)), draws)
        member_effects = [float(effect[i].mean()) for i in range(3)]
        family_effects = {family: float(effect[:, i].mean()) for i, family in enumerate(FAMILIES)}
        old_deltas = {name: float((tensor[index, 4:7] - tensor[index, 1:4]).mean())
                      for name, tensor in values.items() if name != "power"}
        baseline_deltas = {name: float((tensor[index, 4:7] - tensor[index, 0]).mean())
                           for name, tensor in values.items() if name != "power"}
        criteria = {
            "new_minus_old_power_positive": float(effect.mean()) > 0,
            "paired_weather_simultaneous_ci_lower_positive": interval[0] > 0,
            "each_member_effect_positive": all(item > 0 for item in member_effects),
            "each_family_effect_positive": all(item > 0 for item in family_effects.values()),
            "new_relative_gain_at_least_1_05_percent": relative >= cfg["thresholds"]["minimum_relative_gain"],
            "strehl_non_decrease_vs_old_and_integrator":
                old_deltas["strehl"] >= 0 and baseline_deltas["strehl"] >= 0,
            "phase_rmse_non_increase_vs_old_and_integrator":
                old_deltas["phase_rmse"] <= 0 and baseline_deltas["phase_rmse"] <= 0,
            "safety_increase_at_most_0_001_vs_old_and_integrator": all(
                old_deltas[name] <= cfg["thresholds"]["maximum_safety_increase"]
                and baseline_deltas[name] <= cfg["thresholds"]["maximum_safety_increase"]
                for name in ("violation", "saturation", "slew_limited")),
        }
        criteria["all"] = all(criteria.values())
        cells[condition] = {
            "integrator_power": baseline_mean, "old_power": float(old.mean()),
            "new_power": float(new.mean()), "new_relative_gain": relative,
            "old_relative_gain": float((old - baseline).mean()) / baseline_mean,
            "new_minus_old_absolute_power": float(effect.mean()),
            "new_minus_old_paired_weather_ci97_5": interval,
            "member_effects": member_effects, "family_effects": family_effects,
            "slot_effects": [float(effect[..., slot].mean()) for slot in range(6)],
            "new_minus_old_metric_deltas": old_deltas,
            "new_minus_integrator_metric_deltas": baseline_deltas,
            "controller_mean_metrics": {
                controller: {name: float(tensor[index, i].mean()) for name, tensor in values.items()}
                for i, controller in enumerate(CONTROLLERS)},
            "continue_criteria": criteria,
        }
    return {
        "status": "EXPLORATORY_DEVELOPMENT_NOT_INDEPENDENT_CONFIRMATION",
        "cells": cells,
        "continue_criteria": {**{name: cells[name]["continue_criteria"] for name in CONDITIONS},
                              "all": all(cells[name]["continue_criteria"]["all"] for name in CONDITIONS)},
        "paired_unit": "same_complete_weather_family_slot_member_across_controllers",
        "independent_weather_count_per_condition": len(weather),
        "two_condition_familywise_ci": "Bonferroni: 97.5% interval per primary condition",
        "independent_confirmation": False,
        "latency_scope": "CUDA batch-18 policy forward plus clamp telemetry; excludes safety projection, environment and optical I/O",
    }


@torch.no_grad()
def run(path: str | Path = CONFIG, *, quick: bool = False,
        preflight_only: bool = False) -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cfg, parent, report, output, device = preflight(path, quick=quick)
    if preflight_only:
        return report
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    created = False
    progress: d3.SparseProgress | None = None
    started = time.perf_counter()
    try:
        output.mkdir(parents=True, exist_ok=False)
        created = True
        write_json(output / "preflight.json", report)
        write_json(output / "config.json", cfg)
        write_json(output / "stream_manifest.json", stream_manifest(quick))
        write_json(output / "runtime.json", {
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device), "git": safe_git_record(),
            "frozen_source_bundle_sha256": d8.SOURCE_BUNDLE_SHA256,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        })
        spec = cfg["quick" if quick else "data"]
        base, _ = load_s1_config(_project_path(parent["environment_config"]))
        base = replace(base, num_modes=21, batch_size=1, episode_length=spec["episode_length"])
        basis, _, _ = build_action_basis(base, ActionRepresentation("r5_zernike21", "zernike", 21), device)
        branches: list[tuple[str, ResidualGRUPolicy | None]] = [("integrator", None)]
        training_cfg = _load_yaml(_project_path(d8.CONFIG))
        for arm, folder in ((d8.COMPARATOR, _project_path(training_cfg["d2_output"])),
                            (d8.ARM, _project_path(cfg["d8_output"]))):
            for member in range(3):
                name = f"{arm}_policy_{member}_00512.pt"
                path_ = folder / "checkpoints" / name
                expected_hash = (D8_FINAL[name] if arm == d8.ARM
                                 else report["d2_final_checkpoints"][str(member)])
                if _file_sha256(path_) != expected_hash:
                    raise RuntimeError("G2-D9 权重在预检后变化")
                saved = torch.load(path_, map_location=device, weights_only=True)
                if (saved["arm"], saved["init"], saved["update"], saved["deployment_scale"]) != (arm, member, 512, SCALE):
                    raise RuntimeError("G2-D9 评价权重身份不符")
                policy = ResidualGRUPolicy(parent["policy"]["hidden_size"], parent["policy"]["output_size"]).to(device)
                policy.load_state_dict(saved["state_dict"])
                policy.eval().requires_grad_(False)
                branches.append((f"{arm}_{member}", policy))
        if [label for label, _ in branches] != list(CONTROLLERS):
            raise RuntimeError("G2-D9 控制器顺序不符")
        nominal, shifted = d8.d2.g2.d1._profile_pairs(parent)
        progress = d3.SparseProgress(output, device, spec["episode_length"])
        progress.phase("G2-D9 CUDA 技术冒烟" if quick else "G2-D9 新天气动作惩罚闭环开发评价",
                       report["physical_transitions"] // 18)
        rows = []
        with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
            for condition, profiles in zip(CONDITIONS, (nominal, shifted), strict=True):
                slots = {profile.identifier: i for i, profile in enumerate(profiles)}
                for label, policy in branches:
                    for seed in stream_manifest(quick)["weather_bases"]:
                        meter = None if policy is None else CorrectionClampTelemetry(policy, SCALE)
                        episodes = _rollout(seed, label, 0.0 if meter is None else SCALE, meter,
                                            spec["episode_length"], basis, base, parent["families"],
                                            profiles, parent["data"]["sensor_seed_offset"], progress)
                        rates = [0.0] * len(episodes) if meter is None else meter.rates(spec["episode_length"], len(episodes))
                        for row, rate in zip(episodes, rates, strict=True):
                            row.update({"hardware_condition": condition, "slot": slots[row["profile"]],
                                        "episode_length": spec["episode_length"],
                                        "normalized_correction_clipped_fraction": rate})
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                            rows.append(row)
                        handle.flush()
        if len(rows) != report["complete_episodes"]:
            raise RuntimeError("G2-D9 完整回合总数不足")
        analysis = {} if quick else summarize(rows, cfg, device=device)
        result = {
            "status": "QUICK_SMOKE_NO_SCIENTIFIC_CONCLUSION" if quick else "DEVELOPMENT_COMPARISON_REQUIRES_READ_ONLY_AUDIT",
            "records": len(rows), "completed_group_episodes": len(rows), "failed_group_episodes": 0,
            "physical_transitions": report["physical_transitions"],
            "weather_count": spec["weather_count"], "episode_length": spec["episode_length"],
            "controller_ids": list(CONTROLLERS), "hardware_conditions": list(CONDITIONS),
            "deployment_scale": SCALE, "analysis": analysis,
            "elapsed_seconds": time.perf_counter() - started,
            "config_sha256": report["config_sha256"], "entry_sha256": report["entry_sha256"],
            "d8_training_output_hashes": D8_HASHES, "d8_final_checkpoints": D8_FINAL,
            "d2_final_checkpoints": report["d2_final_checkpoints"],
            "frozen_source_bundle_sha256": d8.SOURCE_BUNDLE_SHA256,
            "material_passport": {"origin_skill": "academic-research-suite / experiment-agent",
                                  "origin_mode": "run", "verification_status": "UNVERIFIED"},
            **cfg["boundary"],
            "next_action": "停止并等待只读开发审计；双条件通过后才设计一次新的独立确认",
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
        if created:
            write_json(output / "failure.json", {"traceback": traceback.format_exc(), "automatic_retry": False})
        raise
    finally:
        if progress is not None:
            progress.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--quick", action="store_true", help="16 帧 CUDA 技术冒烟")
    parser.add_argument("--preflight-only", action="store_true", help="只读预检，不创建输出")
    args = parser.parse_args()
    print(json.dumps(run(args.config, quick=args.quick, preflight_only=args.preflight_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
