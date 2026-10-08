"""G2-D3：全新天气比较等预算续训策略；正式闭环评价仅由用户在 IDE 启动。"""
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

from scripts import train_s4_r5_g2_d2_matched as train
from scripts.diagnose_s4_r5_margin_d2 import CorrectionClampTelemetry
from src.rl.r4_dynamics_experiment import safe_git_record
from src.rl.r4_interface_smoke import write_json
from src.rl.r5_independent_confirmation_r2 import source_bundle_sha256
from src.rl.r5_margin_development import _rollout
from src.rl.r5_policy_training import ResidualGRUPolicy
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _project_path
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.robust_control import RobustnessCondition
from src.training_progress import counted_progress, update_progress


CONFIG = "configs/experiments/s4_r5_g2_d3_matched_development_v1.yaml"
CONDITIONS = ("nominal_clone", "hardware_shift")
FAMILIES = train.g2.FAMILIES
PROFILES = train.g2.PROFILES
CONTROLLERS = ("integrator",) + tuple(f"original_{i}" for i in range(3)) + tuple(
    f"{arm}_{i}" for arm, _ in train.ARMS for i in range(3))
SCALE = 1.75
FORMAL_BASE = 7_100_000
QUICK_BASE = 7_110_000
WEATHER_COUNT = 32
TRAIN_CONFIG_SHA256 = "297535002d1df67e4b2ce29e8a4d749c361bcda0c94e6e199195cdb2748dc63a"
TRAIN_ENTRY_SHA256 = "f747952ae9d4d736e78dba7e4168ae41e49d876fc26a86ca8a602304a96d0706"
TRAIN_HASHES = {
    "summary.json": "68471420177832f2f883448a8a3ab01b87c5481773e515570aca308201d22f8e",
    "losses.jsonl": "a447df87d3bbcefb94726a71f8ac0126275f03cfd7c8d36f6c9c90948445ff02",
    "progress.jsonl": "86d83bfd42f711b5f1c9e230a8ba7d0ff078b374f4c94cee348118765f4b33b9",
    "stream_manifest.json": "6312e5ec18ba93685dd4383c3ddc3689f4297ea60c28bb3a819a97adac026f58",
    "checkpoint_manifest.json": "a1d5c713dd2c35599e0c397659fc07c4e2c1d6f01acce3e529d1c4517a710e5f",
    "SUCCESS.json": "db66743be8529b70a81dd85e9d7a0acaf27a03d90d2037e8c9a4ef62f13d9287",
}
METRICS = ("power", "strehl", "phase_rmse", "violation", "saturation", "slew_limited",
           "correction_abs", "requested_step_abs", "requested_modal_abs", "applied_modal_abs",
           "requested_applied_gap_abs", "normalized_correction_clipped_fraction",
           "policy_forward_seconds_per_step", "policy_forward_p95_seconds")


class SparseProgress:
    """逐帧显示进度、每条完整回合写一行日志，避免百万行进度文件。"""

    def __init__(self, output: Path, device: torch.device, interval: int):
        self.path = output / "progress.jsonl"
        self.device = device
        self.interval = interval
        self.bar = None
        self.started = time.perf_counter()

    def phase(self, title: str, total: int) -> None:
        self.title = title
        self.phase_started = time.perf_counter()
        self.bar = counted_progress(total=total, description=title, unit="帧")

    def tick(self, metrics: dict[str, float] | None = None) -> None:
        if self.bar is None:
            raise RuntimeError("进度阶段尚未开始")
        self.bar.update(1)
        if self.bar.n % self.interval and self.bar.n != self.bar.total:
            return
        values = metrics or {}
        update_progress(self.bar, device=self.device, metrics=values)
        elapsed = time.perf_counter() - self.phase_started
        record = {"phase": self.title, "completed": self.bar.n, "total": self.bar.total,
                  "elapsed_seconds": time.perf_counter() - self.started,
                  "eta_seconds": elapsed / self.bar.n * (self.bar.total - self.bar.n),
                  "cuda_allocated_gb": torch.cuda.memory_allocated(self.device) / 1024**3,
                  **values}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()


def _contract(cfg: dict) -> None:
    expected = {
        "stage": "S4-D2-R5-G2-D3",
        "purpose": "fresh_weather_equal_deployment_paired_closed_loop_development",
        "runtime": {"device": "cuda", "formal_owner": "user_ide", "automatic_retry": False},
        "training_config": train.CONFIG, "training_config_sha256": TRAIN_CONFIG_SHA256,
        "training_entry_sha256": TRAIN_ENTRY_SHA256,
        "training_output": "outputs/s4_r5_g2_d2_matched_training_v1",
        "training_output_hashes": TRAIN_HASHES, "source_bundle_sha256": train.FROZEN_SOURCE,
        "controller_ids": list(CONTROLLERS), "deployment_scale": SCALE,
        "hardware_conditions": list(CONDITIONS),
        "data": {"seed_base": FORMAL_BASE, "seed_stride": 10,
                 "weather_count": WEATHER_COUNT, "episode_length": 200},
        "quick": {"seed_base": QUICK_BASE, "weather_count": 1, "episode_length": 16},
        "statistics": {"bootstrap_seed": 7_123_456, "bootstrap_repeats": 5_000,
                       "primary_interval_per_condition": 0.975},
        "thresholds": {"minimum_relative_gain": .0105, "maximum_safety_increase": .001},
        "output_directory": "outputs/s4_r5_g2_d3_matched_development_v1",
        "quick_directory": "outputs/s4_r5_g2_d3_matched_development_v1_quick",
        "boundary": {"confirmation_access": False, "training_updates": 0,
                     "real_slm_actions": False, "automatic_retry": False,
                     "historical_results_read_only": True},
    }
    if cfg != expected:
        raise ValueError("G2-D3 冻结开发评价合同已改变")


def _verify_lineage(cfg: dict) -> tuple[dict, dict]:
    if (_file_sha256(_project_path(cfg["training_config"])) != TRAIN_CONFIG_SHA256
            or _file_sha256(_project_path("scripts/train_s4_r5_g2_d2_matched.py")) != TRAIN_ENTRY_SHA256
            or source_bundle_sha256() != train.FROZEN_SOURCE):
        raise RuntimeError("G2-D2 配置、入口或冻结物理源码变化")
    training_cfg = _load_yaml(_project_path(cfg["training_config"]))
    train._contract(training_cfg)
    r3_cfg, parent = train._verify_lineage(training_cfg)
    output = _project_path(cfg["training_output"])
    if (output / "failure.json").exists():
        raise RuntimeError("G2-D2 训练有失败标记")
    for name, digest in TRAIN_HASHES.items():
        if _file_sha256(output / name) != digest:
            raise RuntimeError(f"G2-D2 训练证据变化: {name}")
    success = json.loads((output / "SUCCESS.json").read_text(encoding="utf-8"))
    for key, name in (("summary_sha256", "summary.json"), ("losses_sha256", "losses.jsonl"),
                      ("progress_sha256", "progress.jsonl"),
                      ("stream_manifest_sha256", "stream_manifest.json"),
                      ("checkpoint_manifest_sha256", "checkpoint_manifest.json")):
        if success.get(key) != TRAIN_HASHES[name]:
            raise RuntimeError("G2-D2 SUCCESS 与训练文件不一致")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("status") != "DEVELOPMENT_TRAINING_COMPLETE_REQUIRES_AUDIT"
            or summary.get("training_updates") != 3072 or summary.get("checkpoints") != 48
            or summary.get("physical_transitions") != 11_059_200
            or summary.get("config_sha256") != TRAIN_CONFIG_SHA256
            or summary.get("entry_sha256") != TRAIN_ENTRY_SHA256
            or summary.get("confirmation_access") is not False
            or summary.get("development_evaluation_access") is not False
            or summary.get("real_slm_actions") is not False):
        raise RuntimeError("G2-D2 训练完成性或边界不符")
    manifest = json.loads((output / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    expected = {f"{arm}_policy_{member}_{update:05d}.pt"
                for arm, _ in train.ARMS for member in range(3) for update in range(64, 513, 64)}
    if set(manifest) != expected or {p.name for p in (output / "checkpoints").glob("*.pt")} != expected:
        raise RuntimeError("G2-D2 检查点集合不完整或有额外文件")
    for name, digest in manifest.items():
        if _file_sha256(output / "checkpoints" / name) != digest:
            raise RuntimeError(f"G2-D2 检查点哈希变化: {name}")
    return r3_cfg, parent


def stream_manifest(quick: bool) -> dict[str, list[int]]:
    base, count = (QUICK_BASE, 1) if quick else (FORMAL_BASE, WEATHER_COUNT)
    weather = [base + 10 * index for index in range(count)]
    turbulence = [seed + 1000 * slot + family for seed in weather
                  for slot in range(6) for family in range(3)]
    sensor = [seed + 1000 * slot + 50_000_000 for seed in weather for slot in range(6)]
    power = [seed + 1000 * slot + 60_000_000 for seed in weather for slot in range(6)]
    if (len(set(turbulence)) != 18 * count or len(set(sensor)) != 6 * count
            or len(set(power)) != 6 * count or max(turbulence) >= base + 10_000):
        raise RuntimeError("G2-D3 随机流碰撞")
    return {"weather_bases": weather, "turbulence": turbulence,
            "sensor": sensor, "power": power,
            "shared_between_controllers_and_conditions": True}


def preflight(path: str | Path = CONFIG, *, quick: bool = False
              ) -> tuple[dict, dict, dict, dict, Path, torch.device]:
    cfg = _load_yaml(_project_path(path))
    _contract(cfg)
    r3_cfg, parent = _verify_lineage(cfg)
    if (parent["data"]["sensor_seed_offset"] != 50_000_000
            or [item["id"] for item in parent["families"]] != list(FAMILIES)):
        raise RuntimeError("R5 因果观测或湍流家族变化")
    nominal, shifted = train.g2.d1._profile_pairs(parent)
    if ([p.identifier for p in shifted] != list(PROFILES)
            or [p.identifier for p in nominal] != [f"nominal_for_{p}" for p in PROFILES]):
        raise RuntimeError("同槽标称/新硬件档位无法配对")
    formal, smoke = stream_manifest(False), stream_manifest(True)
    old_streams = (train.stream_manifest(quick=False), train.stream_manifest(quick=True),
                   train.g2.stream_manifest(False), train.g2.stream_manifest(True))
    for name in ("turbulence", "sensor", "power"):
        seen = set().union(*(set(item[name]) for item in old_streams))
        if (set(formal[name]) & set(smoke[name])
                or seen & (set(formal[name]) | set(smoke[name]))):
            raise RuntimeError(f"G2-D3 {name} 随机流与训练或旧开发重叠")
    spec = cfg["quick" if quick else "data"]
    base, _ = load_s1_config(_project_path(parent["environment_config"]))
    displacement = []
    for family in parent["families"]:
        condition = RobustnessCondition.from_mapping(dict(family, base_seed=spec["seed_base"]))
        simulation = condition.environment_config(replace(base, episode_length=spec["episode_length"]))
        pixels = (simulation.wind_speed_mps * (1 + simulation.wind_speed_modulation_fraction)
                  * simulation.dt_s * spec["episode_length"] / simulation.sample_pitch_m)
        if pixels >= simulation.turbulence_grid_size:
            raise RuntimeError("G2-D3 相位屏在回合内重复")
        displacement.append(pixels)
    output = _project_path(cfg["quick_directory" if quick else "output_directory"])
    if output.exists():
        raise FileExistsError(f"保留已有 G2-D3 输出，不覆盖或重跑: {output}")
    device = resolve_device("cuda")
    total = len(CONDITIONS) * len(CONTROLLERS) * spec["weather_count"] * len(FAMILIES) * 6
    report = {
        "status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_IDE",
        "quick": quick, "device": str(device), "weather_count": spec["weather_count"],
        "episode_length": spec["episode_length"], "controller_ids": list(CONTROLLERS),
        "hardware_conditions": list(CONDITIONS), "complete_episodes": total,
        "physical_transitions": total * spec["episode_length"],
        "maximum_displacement_pixels": displacement,
        "turbulence_grid_pixels": simulation.turbulence_grid_size,
        "config_sha256": _file_sha256(_project_path(path)),
        "entry_sha256": _file_sha256(Path(__file__)),
        "training_summary_sha256": TRAIN_HASHES["summary.json"],
        "training_checkpoint_manifest_sha256": TRAIN_HASHES["checkpoint_manifest.json"],
        "frozen_source_bundle_sha256": train.FROZEN_SOURCE,
        **cfg["boundary"],
    }
    return cfg, r3_cfg, parent, report, output, device


def _paired_interval(weather_values: torch.Tensor, draws: torch.Tensor) -> list[float]:
    samples = weather_values[draws].mean(-1)
    bounds = torch.tensor((.0125, .9875), dtype=samples.dtype, device=samples.device)
    return [float(value) for value in torch.quantile(samples, bounds)]


def summarize(rows: list[dict], cfg: dict, *, device: torch.device) -> dict:
    weather = stream_manifest(False)["weather_bases"]
    keys = {(r["hardware_condition"], r["controller"], r["family"], r["slot"],
             r["weather_seed"]): r for r in rows}
    expected = {(condition, controller, family, slot, seed)
                for condition in CONDITIONS for controller in CONTROLLERS
                for family in FAMILIES for slot in range(6) for seed in weather}
    if len(rows) != len(expected) or set(keys) != expected:
        raise RuntimeError("G2-D3 完整回合配对缺失或重复")
    for row in rows:
        profile = (f"nominal_for_{PROFILES[row['slot']]}"
                   if row["hardware_condition"] == "nominal_clone" else PROFILES[row["slot"]])
        if (row["profile"] != profile
                or row["scale"] != (0.0 if row["controller"] == "integrator" else SCALE)
                or row["turbulence_stream_seed"] != row["weather_seed"] + 1000 * row["slot"]
                + FAMILIES.index(row["family"])
                or any(not math.isfinite(row[name]) for name in METRICS)
                or not 0 <= row["normalized_correction_clipped_fraction"] <= 1):
            raise RuntimeError("G2-D3 档位、随机流、动作或指标错位")
    values = {name: torch.tensor(
        [[[[[keys[(condition, controller, family, slot, seed)][name]
              for slot in range(6)] for seed in weather] for family in FAMILIES]
          for controller in CONTROLLERS] for condition in CONDITIONS],
        device=device, dtype=torch.float64) for name in METRICS}
    # condition × controller × family × weather × slot。
    power = values["power"]
    if bool((power[:, 0] <= 0).any()):
        raise RuntimeError("G2-D3 积分器桶内功率非正")
    generator = torch.Generator(device=device).manual_seed(cfg["statistics"]["bootstrap_seed"])
    draws = torch.randint(len(weather), (cfg["statistics"]["bootstrap_repeats"], len(weather)),
                          generator=generator, device=device)
    cells = {}
    for condition_index, condition in enumerate(CONDITIONS):
        baseline = power[condition_index, 0]
        original = power[condition_index, 1:4]
        unmatched = power[condition_index, 4:7]
        matched = power[condition_index, 7:10]
        effect = matched - unmatched  # member × family × weather × slot，完整天气聚类。
        baseline_mean = float(baseline.mean())
        metric_deltas = {name: float((tensor[condition_index, 7:10]
                                      - tensor[condition_index, 4:7]).mean())
                         for name, tensor in values.items() if name != "power"}
        baseline_deltas = {name: float((tensor[condition_index, 7:10]
                                        - tensor[condition_index, 0]).mean())
                           for name, tensor in values.items() if name != "power"}
        relative = float((matched - baseline).mean()) / baseline_mean
        interval = _paired_interval(effect.mean(dim=(0, 1, 3)), draws)
        member_effects = [float(effect[i].mean()) for i in range(3)]
        family_effects = {family: float(effect[:, i].mean())
                          for i, family in enumerate(FAMILIES)}
        safety = ("violation", "saturation", "slew_limited")
        criteria = {
            "matched_minus_unmatched_power_positive": float(effect.mean()) > 0,
            "paired_weather_simultaneous_ci_lower_positive": interval[0] > 0,
            "each_member_effect_positive": all(item > 0 for item in member_effects),
            "each_family_effect_positive": all(item > 0 for item in family_effects.values()),
            "matched_relative_gain_at_least_1_05_percent": relative >= cfg["thresholds"]["minimum_relative_gain"],
            "strehl_non_decrease_vs_unmatched_and_integrator":
                metric_deltas["strehl"] >= 0 and baseline_deltas["strehl"] >= 0,
            "phase_rmse_non_increase_vs_unmatched_and_integrator":
                metric_deltas["phase_rmse"] <= 0 and baseline_deltas["phase_rmse"] <= 0,
            "safety_increase_at_most_0_001_vs_unmatched_and_integrator": all(
                metric_deltas[name] <= cfg["thresholds"]["maximum_safety_increase"]
                and baseline_deltas[name] <= cfg["thresholds"]["maximum_safety_increase"]
                for name in safety),
        }
        criteria["all"] = all(criteria.values())
        cells[condition] = {
            "integrator_power": baseline_mean, "original_power": float(original.mean()),
            "unmatched_power": float(unmatched.mean()), "matched_power": float(matched.mean()),
            "original_relative_gain": float((original - baseline).mean()) / baseline_mean,
            "unmatched_relative_gain": float((unmatched - baseline).mean()) / baseline_mean,
            "matched_relative_gain": relative,
            "matched_minus_unmatched_absolute_power": float(effect.mean()),
            "matched_minus_unmatched_paired_weather_ci97_5": interval,
            "member_mapping_effects": member_effects, "family_mapping_effects": family_effects,
            "slot_mapping_effects": [float(effect[..., slot].mean()) for slot in range(6)],
            "matched_minus_unmatched_metric_deltas": metric_deltas,
            "matched_minus_integrator_metric_deltas": baseline_deltas,
            "continue_criteria": criteria,
        }
    return {
        "status": "EXPLORATORY_DEVELOPMENT_NOT_INDEPENDENT_CONFIRMATION",
        "cells": cells, "continue_criteria": {
            "nominal_clone": cells["nominal_clone"]["continue_criteria"],
            "hardware_shift": cells["hardware_shift"]["continue_criteria"],
            "all": all(cells[name]["continue_criteria"]["all"] for name in CONDITIONS)},
        "paired_unit": "same_complete_weather_family_slot_member_across_all_controllers",
        "two_condition_familywise_ci": "Bonferroni: 97.5% interval per primary condition",
        "independent_confirmation": False,
        "latency_scope": "CUDA policy forward only; excludes projection, environment, camera, SLM and optical I/O",
    }


def run(path: str | Path = CONFIG, *, quick: bool = False,
        preflight_only: bool = False) -> dict:
    cfg, r3_cfg, parent, report, output, device = preflight(path, quick=quick)
    if preflight_only:
        return report
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    created = False
    progress: SparseProgress | None = None
    started = time.perf_counter()
    try:
        output.mkdir(parents=True, exist_ok=False)
        created = True
        write_json(output / "preflight.json", report)
        write_json(output / "config.json", cfg)
        write_json(output / "stream_manifest.json", stream_manifest(quick))
        write_json(output / "runtime.json", {
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "git": safe_git_record(),
            "frozen_source_bundle_sha256": train.FROZEN_SOURCE,
        })
        spec = cfg["quick" if quick else "data"]
        base, _ = load_s1_config(_project_path(parent["environment_config"]))
        base = replace(base, num_modes=21, batch_size=1, episode_length=spec["episode_length"])
        basis, _, _ = build_action_basis(base, ActionRepresentation("r5_zernike21", "zernike", 21), device)
        branches = [("integrator", None)]
        for member in range(3):
            old_path = _project_path(r3_cfg["training_output"]) / "checkpoints" / f"policy_{member}_02000.pt"
            old = torch.load(old_path, map_location=device, weights_only=True)
            if old["init"] != member or old["update"] != 2000:
                raise RuntimeError("旧 R5-2 策略身份错误")
            policy = ResidualGRUPolicy(parent["policy"]["hidden_size"],
                                       parent["policy"]["output_size"]).to(device)
            policy.load_state_dict(old["state_dict"])
            policy.eval()
            branches.append((f"original_{member}", policy))
        for arm, scale in train.ARMS:
            for member in range(3):
                file = _project_path(cfg["training_output"]) / "checkpoints" / f"{arm}_policy_{member}_00512.pt"
                saved = torch.load(file, map_location=device, weights_only=True)
                if (saved["arm"] != arm or saved["init"] != member or saved["update"] != 512
                        or saved["training_scale"] != scale or saved["deployment_scale"] != SCALE):
                    raise RuntimeError("G2-D2 最终检查点身份错误")
                policy = ResidualGRUPolicy(parent["policy"]["hidden_size"],
                                           parent["policy"]["output_size"]).to(device)
                policy.load_state_dict(saved["state_dict"])
                policy.eval()
                branches.append((f"{arm}_{member}", policy))
        assert [label for label, _ in branches] == list(CONTROLLERS)
        nominal, shifted = train.g2.d1._profile_pairs(parent)
        progress = SparseProgress(output, device, spec["episode_length"])
        progress.phase("G2-D3 CUDA 快速冒烟" if quick else "G2-D3 新天气配对闭环开发评价",
                       report["physical_transitions"] // (len(FAMILIES) * 6))
        rows: list[dict] = []
        with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
            for condition, profiles in zip(CONDITIONS, (nominal, shifted), strict=True):
                for label, policy in branches:
                    for seed in stream_manifest(quick)["weather_bases"]:
                        meter = None if policy is None else CorrectionClampTelemetry(policy, SCALE)
                        episodes = _rollout(seed, label, 0.0 if meter is None else SCALE, meter,
                                            spec["episode_length"], basis, base, parent["families"],
                                            profiles, parent["data"]["sensor_seed_offset"], progress)
                        clipped = ([0.0] * len(episodes) if meter is None
                                   else meter.rates(spec["episode_length"], len(episodes)))
                        slots = {profile.identifier: i for i, profile in enumerate(profiles)}
                        for row, rate in zip(episodes, clipped, strict=True):
                            row.update({"hardware_condition": condition, "slot": slots[row["profile"]],
                                        "normalized_correction_clipped_fraction": rate})
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                            rows.append(row)
                        handle.flush()
        if len(rows) != report["complete_episodes"]:
            raise RuntimeError("G2-D3 闭环回合总数不足")
        analysis = {} if quick else summarize(rows, cfg, device=device)
        result = {
            "status": "QUICK_SMOKE_NO_SCIENTIFIC_CONCLUSION" if quick
                      else "DEVELOPMENT_COMPARISON_REQUIRES_READ_ONLY_AUDIT",
            "records": len(rows), "completed_group_episodes": len(rows), "failed_group_episodes": 0,
            "physical_transitions": report["physical_transitions"],
            "weather_count": spec["weather_count"], "episode_length": spec["episode_length"],
            "controller_ids": list(CONTROLLERS), "hardware_conditions": list(CONDITIONS),
            "deployment_scale": SCALE, "analysis": analysis,
            "elapsed_seconds": time.perf_counter() - started,
            "config_sha256": report["config_sha256"], "entry_sha256": report["entry_sha256"],
            "training_output_hashes": TRAIN_HASHES,
            "frozen_source_bundle_sha256": train.FROZEN_SOURCE,
            "material_passport": {"origin_skill": "academic-research-suite / experiment-agent",
                                  "origin_mode": "run", "verification_status": "UNVERIFIED"},
            **cfg["boundary"],
            "next_action": "停止并等待只读开发审计；仅双条件开发门槛可决定是否设计新的独立确认",
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
            try:
                write_json(output / "failure.json", {"traceback": traceback.format_exc(),
                                                      "automatic_retry": False})
            except Exception:
                pass
        raise
    finally:
        if progress is not None:
            progress.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--quick", action="store_true", help="16 帧 CUDA 冒烟，不形成科学结论")
    parser.add_argument("--preflight-only", action="store_true", help="只读预检")
    args = parser.parse_args()
    print(json.dumps(run(args.config, quick=args.quick,
                         preflight_only=args.preflight_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
