"""G2-D7：冻结策略完整闭环硬前向倍率有限差分；仅开发机制诊断。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import diagnose_s4_r5_g2_d6_full_objective as d6
from src.rl.r4_dynamics_experiment import safe_git_record
from src.rl.r4_interface_smoke import write_json
from src.rl.r4_observation import PowerMeasurement
from src.rl.r4_trajectory import anchor_delta
from src.rl.r5_physics_adapter import measured_objective
from src.rl.r5_policy_training import ResidualGRUPolicy
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _project_path
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.robust_control import RobustnessCondition


CONFIG = "configs/experiments/s4_r5_g2_d7_hard_forward_v1.yaml"
D6_CONFIG_SHA256 = "c7efffbe49077c0e040bd2514901fc84cb19f151715b21f0efd2065a803cb09d"
D6_ENTRY_SHA256 = "11474a3da009a52f5c04f0fec9fcf98164bc7f255c7c91245b66db65d21b4a80"
D6_OUTPUT_HASHES = {
    "summary.json": "086eadd4901260b377e1e9b2a030c7d9bb97525d09cc479a47880b2561be901b",
    "groups.jsonl": "7957471129147abc204b70355ba4ca53fb94158087943a6481e0e6119ad2f00e",
    "records.jsonl": "7d8d9251bdff3b18740e5deff6eb9e588db9a71b35ed44df41538e9517bf78fa",
    "progress.jsonl": "3f4d86ae98ccfeded852dde1b6d94fa420fcb40c327c7ee7a4407862cb1a06ec",
    "stream_manifest.json": "290ca7adc5b711a44db8289b1d2f97e87ef6af9c554d89fe58a52fec1718759e",
    "SUCCESS.json": "48afc1bd594a385203c7faa85c46c0484c11983c12478a84dda671fecc35c002",
}
ARMS = d6.ARMS
CONDITIONS = d6.CONDITIONS
SCALES = (1.6625, 1.8375)
CENTER = 1.75
SPAN = SCALES[1] - SCALES[0]
FORMAL_BASE, QUICK_BASE = d6.FORMAL_BASE, 7_360_000
METRICS = d6.METRICS
POWER_TERMS = ("measured_power", "physical_power", "action_penalty",
               "smooth_penalty", "training_objective")


def _contract(cfg: dict) -> None:
    expected = {
        "stage": "S4-D2-R5-G2-D7",
        "purpose": "frozen_full_horizon_hard_forward_scale_finite_difference_development_diagnostic",
        "runtime": {"device": "cuda", "diagnostic_owner": "assistant_under_user_goal",
                    "automatic_retry": False},
        "g2_d6_config": d6.CONFIG, "g2_d6_config_sha256": D6_CONFIG_SHA256,
        "g2_d6_entry_sha256": D6_ENTRY_SHA256,
        "g2_d6_output": "outputs/s4_r5_g2_d6_full_objective_v1",
        "g2_d6_output_hashes": D6_OUTPUT_HASHES,
        "controllers": list(ARMS), "hardware_conditions": list(CONDITIONS),
        "deployment_scales": list(SCALES), "center_deployment_scale": CENTER,
        "direction_epsilon": 1e-6, "metric_delta_epsilon": 1e-7,
        "data": {"seed_base": FORMAL_BASE, "seed_stride": 10, "weather_count": 2,
                 "episode_length": 200, "initializations": 3},
        "quick": {"seed_base": QUICK_BASE, "weather_count": 1,
                  "episode_length": 16, "initializations": 1},
        "output_directory": "outputs/s4_r5_g2_d7_hard_forward_v1",
        "quick_directory": "outputs/s4_r5_g2_d7_hard_forward_v1_quick",
        "boundary": {"confirmation_access": False, "training_updates": 0,
                     "optimizer_updates": 0, "real_slm_actions": False,
                     "automatic_retry": False, "historical_results_read_only": True,
                     "independent_confirmation": False},
    }
    if cfg != expected:
        raise ValueError("G2-D7 冻结硬前向开发诊断合同已改变")


def stream_manifest(quick: bool) -> dict:
    seeds = ([QUICK_BASE] if quick else d6.stream_manifest(False)["weather_bases"])
    return {
        "weather_bases": seeds,
        "turbulence": [seed + 1000 * slot + family for seed in seeds
                       for slot in range(6) for family in range(3)],
        "sensor": [seed + 1000 * slot + 50_000_000 for seed in seeds
                   for slot in range(6)],
        "power": [seed + 1000 * slot + 60_000_000 for seed in seeds
                  for slot in range(6)],
        "same_seed_restarts_each_scale_arm_and_member": True,
        "formal_weather_reuses_hash_locked_d6_center": not quick,
    }


def _check_stream_isolation() -> None:
    formal, quick = stream_manifest(False), stream_manifest(True)
    if formal["weather_bases"] != d6.stream_manifest(False)["weather_bases"]:
        raise RuntimeError("G2-D7 正式天气未与冻结 G2-D6 中心配对")
    old = (d6.stream_manifest(False), d6.stream_manifest(True),
           d6.d5.stream_manifest(False), d6.d5.stream_manifest(True),
           d6.d5.d4.stream_manifest(False), d6.d5.d4.stream_manifest(True),
           d6.d5.d4.d3.stream_manifest(False), d6.d5.d4.d3.stream_manifest(True),
           d6.d5.d4.d3.train.stream_manifest(quick=False),
           d6.d5.d4.d3.train.stream_manifest(quick=True),
           d6.d5.d4.d3.train.g2.stream_manifest(False),
           d6.d5.d4.d3.train.g2.stream_manifest(True))
    for name in ("turbulence", "sensor", "power"):
        quick_seeds = set(quick[name])
        if (len(quick_seeds) != len(quick[name])
                or quick_seeds & set(formal[name])
                or quick_seeds & set().union(*(set(item[name]) for item in old))):
            raise RuntimeError(f"G2-D7 {name} 快速冒烟与开发或历史随机流重叠")


def _jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _d6_center(cfg: dict) -> tuple[dict, dict, dict, dict]:
    """在创建任何 D7 输出前锁定并审计 D6 中心及其上游来源。"""
    if (_file_sha256(_project_path(cfg["g2_d6_config"])) != D6_CONFIG_SHA256
            or _file_sha256(Path(d6.__file__)) != D6_ENTRY_SHA256):
        raise RuntimeError("G2-D6 冻结配置或入口哈希变化")
    d6_cfg = _load_yaml(_project_path(cfg["g2_d6_config"]))
    d6._contract(d6_cfg)
    parent, train_cfg = d6._verify_d5_and_lineage(d6_cfg)
    root = _project_path(cfg["g2_d6_output"])
    if (root / "failure.json").exists():
        raise RuntimeError("G2-D6 冻结中心含失败标记")
    for name, digest in D6_OUTPUT_HASHES.items():
        if _file_sha256(root / name) != digest:
            raise RuntimeError(f"G2-D6 冻结中心哈希变化: {name}")
    success = json.loads((root / "SUCCESS.json").read_text(encoding="utf-8"))
    for key, name in (("summary_sha256", "summary.json"),
                      ("groups_sha256", "groups.jsonl"),
                      ("records_sha256", "records.jsonl"),
                      ("progress_sha256", "progress.jsonl"),
                      ("stream_manifest_sha256", "stream_manifest.json")):
        if success.get(key) != D6_OUTPUT_HASHES[name]:
            raise RuntimeError(f"G2-D6 SUCCESS 与冻结中心不一致: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("status") != "DEVELOPMENT_DIAGNOSTIC_REQUIRES_READ_ONLY_AUDIT"
            or summary.get("groups") != 24 or summary.get("complete_episodes") != 432
            or summary.get("physical_transitions") != 86_400
            or summary.get("config_sha256") != D6_CONFIG_SHA256
            or summary.get("entry_sha256") != D6_ENTRY_SHA256
            or summary.get("training_updates") != 0
            or summary.get("optimizer_updates") != 0
            or summary.get("confirmation_access") is not False
            or summary.get("real_slm_actions") is not False):
        raise RuntimeError("G2-D6 冻结中心完成性或安全边界不符")
    groups, rows = _jsonl(root / "groups.jsonl"), _jsonl(root / "records.jsonl")
    d6._summarize(groups, rows, d6_cfg, quick=False)
    index = {(group["hardware_condition"], group["weather_seed"],
              group["member"], group["controller"]): group for group in groups}
    for group in groups:
        if (group["deployment_scale"] != CENTER or group["episode_length"] != 200
                or not math.isclose(group["group_mean"]["training_objective"],
                                    group["group_mean"]["measured_power"]
                                    - group["group_mean"]["action_penalty"]
                                    - group["group_mean"]["smooth_penalty"],
                                    rel_tol=1e-5, abs_tol=1e-7)):
            raise RuntimeError("G2-D6 中心倍率或目标分解不符")
    return parent, train_cfg, index, summary


def preflight(path: str | Path = CONFIG, *, quick: bool = False
              ) -> tuple[dict, dict, dict, dict, Path, torch.device, dict]:
    cfg = _load_yaml(_project_path(path))
    _contract(cfg)
    parent, train_cfg, center, _ = _d6_center(cfg)
    d6._check_stream_isolation()
    _check_stream_isolation()
    if (parent["data"]["sensor_seed_offset"] != 50_000_000
            or [family["id"] for family in parent["families"]] != list(d6.d5.d4.d3.FAMILIES)):
        raise RuntimeError("G2-D7 因果观测或湍流家族变化")
    nominal, shifted = d6.d5.d4.d3.train.g2.d1._profile_pairs(parent)
    if ([item.identifier for item in shifted] != list(d6.d5.d4.d3.PROFILES)
            or [item.identifier for item in nominal]
            != [f"nominal_for_{name}" for name in d6.d5.d4.d3.PROFILES]):
        raise RuntimeError("G2-D7 硬件档位配对变化")
    spec = cfg["quick" if quick else "data"]
    base, _ = load_s1_config(_project_path(parent["environment_config"]))
    for family in parent["families"]:
        condition = RobustnessCondition.from_mapping(dict(family, base_seed=spec["seed_base"]))
        simulation = condition.environment_config(replace(base, episode_length=spec["episode_length"]))
        displacement = (simulation.wind_speed_mps
                        * (1 + simulation.wind_speed_modulation_fraction)
                        * simulation.dt_s * spec["episode_length"] / simulation.sample_pitch_m)
        if displacement >= simulation.turbulence_grid_size:
            raise RuntimeError("G2-D7 相位屏在完整回合内重复")
    output = _project_path(cfg["quick_directory" if quick else "output_directory"])
    if output.exists():
        raise FileExistsError(f"保留已有 G2-D7 输出，不覆盖或重跑: {output}")
    device = resolve_device("cuda")
    groups = len(CONDITIONS) * spec["weather_count"] * spec["initializations"] * len(ARMS) * len(SCALES)
    report = {
        "status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_IDE",
        "device": str(device), "quick": quick, "weather_count": spec["weather_count"],
        "episode_length": spec["episode_length"], "batched_closed_loops": groups,
        "complete_episodes": groups * len(parent["families"]) * len(nominal),
        "batched_environment_steps": groups * spec["episode_length"],
        "physical_transitions": groups * len(parent["families"]) * len(nominal)
                                * spec["episode_length"],
        "gradient_comparison": ("D6 surrogate dJ/dscale at 1.75 versus "
                                "D7 hard-forward (J_plus-J_minus)/0.175") if not quick
                               else "not_performed_on_independent_quick_weather",
        "config_sha256": _file_sha256(_project_path(path)),
        "entry_sha256": _file_sha256(Path(__file__)),
        "g2_d6_output_hashes": D6_OUTPUT_HASHES,
        "training_checkpoint_manifest_sha256": d6.d5.d4.d3.TRAIN_HASHES["checkpoint_manifest.json"],
        "frozen_source_bundle_sha256": d6.d5.d4.d3.train.FROZEN_SOURCE,
        **cfg["boundary"],
    }
    return cfg, parent, train_cfg, report, output, device, center


@torch.no_grad()
def _hard_rollout(state: tuple, policy: ResidualGRUPolicy, *, scale_value: float,
                  action_weight: float, smooth_weight: float,
                  progress: d6.d5.d4.d3.SparseProgress) -> dict:
    """完全硬前向，与 D6 同一闭环和原训练目标，绝不构建反向图。"""
    env, interface = state
    batch = env.config.batch_size
    previous = torch.zeros((batch, 11), device=env.device, dtype=env.basis.dtype)
    frames: dict[str, list[torch.Tensor]] = {name: [] for name in METRICS}
    for step in range(env.config.episode_length):
        view = interface.snapshot()
        raw_action = policy(view.features, view.valid)
        correction = raw_action * scale_value
        baseline = anchor_delta(view.features[:, -1],
                                {"gain": .15, "leak": .10, "tracking_gain": .50})
        action = interface.issue(baseline, correction, step=step)
        raw, _, terminated, truncated, info = env.step(action.requested_delta_rad)
        interface.observe_next(env.proxy(raw), step=step + 1,
                               power=PowerMeasurement(info["measured_power_in_bucket"],
                                                      step, step + 1))
        normalized = action.normalized_correction
        measured = info["measured_power_in_bucket"]
        frames["measured_power"].append(measured)
        frames["action_penalty"].append(action_weight * normalized.square().mean(-1))
        frames["smooth_penalty"].append(
            smooth_weight * (normalized - previous).square().mean(-1))
        frames["training_objective"].append(
            measured_objective(measured, normalized, previous, action_weight, smooth_weight))
        frames["physical_power"].append(info["reward_power_in_bucket"])
        frames["violation"].append(info["violation_fraction"])
        frames["saturation"].append(info["saturated_fraction"])
        frames["slew_limited"].append(info["slew_limited_fraction"])
        frames["correction_clipped_fraction"].append(
            correction.abs().gt(1).float().mean(-1))
        frames["requested_applied_gap_abs"].append(
            (info["requested_modal"] - info["applied_modal"]).abs().mean(-1))
        previous = normalized
        if bool(truncated.any()) or bool(terminated.all()) != (step == env.config.episode_length - 1):
            raise RuntimeError("G2-D7 完整闭环回合提前或错误终止")
        progress.tick()
    matrices = {name: torch.stack(values, dim=1) for name, values in frames.items()}
    for name, values in matrices.items():
        if (values.shape != (batch, env.config.episode_length)
                or not bool(torch.isfinite(values).all()) or values.requires_grad):
            raise RuntimeError(f"G2-D7 {name} 硬前向帧矩阵不完整、非有限或含反向图")
    full_objective = matrices["training_objective"].mean()
    components = (matrices["measured_power"].mean()
                  - matrices["action_penalty"].mean()
                  - matrices["smooth_penalty"].mean())
    if not torch.allclose(full_objective, components, rtol=1e-5, atol=1e-7):
        raise RuntimeError("G2-D7 原训练目标均值分解不闭合")
    return {
        "per_episode": {name: values.mean(dim=1).tolist() for name, values in matrices.items()},
        "per_frame_batch_mean": {name: values.mean(dim=0).tolist()
                                 for name, values in matrices.items() if name in POWER_TERMS},
        "group_mean": {name: float(values.mean()) for name, values in matrices.items()},
        "full_training_objective": float(full_objective),
    }


def _direction(value: float, epsilon: float) -> int:
    return 1 if value > epsilon else -1 if value < -epsilon else 0


def _compare_triplet(minus: dict, center: dict, plus: dict, cfg: dict) -> dict:
    """每臂每成员独立比较，天气不能被展开成伪独立样本。"""
    m, c, p = (group["group_mean"] for group in (minus, center, plus))
    delta = {name: p[name] - m[name] for name in METRICS}
    derivative = {name: delta[name] / SPAN for name in METRICS}
    one_sided = {name: {"minus_to_center": c[name] - m[name],
                        "center_to_plus": p[name] - c[name]} for name in METRICS}
    surrogate = center["d_objective_d_deployment_scale"]
    hard_sign = _direction(derivative["training_objective"], cfg["direction_epsilon"])
    proxy_sign = _direction(surrogate, cfg["direction_epsilon"])
    agreement = ("INDETERMINATE_FLAT" if 0 in (hard_sign, proxy_sign)
                 else "AGREE" if hard_sign == proxy_sign
                 else "OPPOSITE_REQUIRES_STE_CHECK")
    penalty_increase = delta["action_penalty"] + delta["smooth_penalty"]
    reversal = (delta["physical_power"] > cfg["metric_delta_epsilon"]
                and delta["training_objective"] < -cfg["metric_delta_epsilon"]
                and penalty_increase - delta["measured_power"]
                > cfg["metric_delta_epsilon"])
    return {
        "hardware_condition": minus["hardware_condition"],
        "weather_seed": minus["weather_seed"], "member": minus["member"],
        "controller": minus["controller"],
        "source_state_sha256": minus["source_state_sha256"],
        "scales": {"minus": SCALES[0], "center_from_d6": CENTER, "plus": SCALES[1]},
        "means": {"minus": m, "center_from_d6": c, "plus": p},
        "plus_minus_delta": delta, "hard_centered_derivative": derivative,
        "one_sided_deltas": one_sided,
        "d6_surrogate_d_objective_d_scale": surrogate,
        "objective_direction_agreement": agreement,
        "physical_up_objective_down_penalty_reversal": reversal,
        "penalty_increase_minus_measured_power_increase": penalty_increase
                                                        - delta["measured_power"],
    }


def _summarize(groups: list[dict], rows: list[dict], cfg: dict, center: dict,
               *, quick: bool) -> tuple[dict, list[dict]]:
    spec = cfg["quick" if quick else "data"]
    seeds = stream_manifest(quick)["weather_bases"]
    expected = {(condition, seed, member, arm, scale)
                for condition in CONDITIONS for seed in seeds
                for member in range(spec["initializations"])
                for arm in ARMS for scale in SCALES}
    index = {(g["hardware_condition"], g["weather_seed"], g["member"],
              g["controller"], g["deployment_scale"]): g for g in groups}
    if len(groups) != len(expected) or set(index) != expected:
        raise RuntimeError("G2-D7 两侧组别缺失或重复")
    for condition in CONDITIONS:
        for seed in seeds:
            for member in range(spec["initializations"]):
                common = [index[condition, seed, member, arm, scale]
                          for arm in ARMS for scale in SCALES]
                if len({g["source_state_sha256"] for g in common}) != 1:
                    raise RuntimeError("G2-D7 两臂两倍率初态哈希不一致")
                if not quick and common[0]["source_state_sha256"] != center[
                        condition, seed, member, ARMS[0]]["source_state_sha256"]:
                    raise RuntimeError("G2-D7 两侧与 G2-D6 中心初态哈希不一致")
    for group in groups:
        mean = group["group_mean"]
        if (set(mean) != set(METRICS) or group["episode_length"] != spec["episode_length"]
                or not math.isfinite(group["full_training_objective"])
                or any(not math.isfinite(value) for value in mean.values())
                or not math.isclose(mean["training_objective"],
                                    mean["measured_power"] - mean["action_penalty"]
                                    - mean["smooth_penalty"], rel_tol=1e-5, abs_tol=1e-7)
                or not math.isclose(mean["training_objective"],
                                    group["full_training_objective"], rel_tol=1e-5, abs_tol=1e-7)
                or set(group["per_frame_batch_mean"]) != set(POWER_TERMS)):
            raise RuntimeError("G2-D7 组均值或目标分解无效")
        for name, trace in group["per_frame_batch_mean"].items():
            if (len(trace) != spec["episode_length"]
                    or any(not math.isfinite(value) for value in trace)
                    or not math.isclose(statistics.fmean(trace), mean[name],
                                        rel_tol=1e-5, abs_tol=1e-7)):
                raise RuntimeError("G2-D7 逐帧与回合均值不一致")
    expected_rows = {(condition, seed, member, arm, scale, family, slot)
                     for condition in CONDITIONS for seed in seeds
                     for member in range(spec["initializations"])
                     for arm in ARMS for scale in SCALES
                     for family in d6.d5.d4.d3.FAMILIES
                     for slot in range(len(d6.d5.d4.d3.PROFILES))}
    row_index = {(r["hardware_condition"], r["weather_seed"], r["member"],
                  r["controller"], r["deployment_scale"], r["family"], r["slot"]): r
                 for r in rows}
    if len(rows) != len(expected_rows) or set(row_index) != expected_rows:
        raise RuntimeError("G2-D7 回合记录索引缺失或重复")
    for key, row in row_index.items():
        condition, seed, member, arm, scale, family, slot = key
        group = index[condition, seed, member, arm, scale]
        profile = (f"nominal_for_{d6.d5.d4.d3.PROFILES[slot]}"
                   if condition == "nominal_clone" else d6.d5.d4.d3.PROFILES[slot])
        if (row["source_state_sha256"] != group["source_state_sha256"]
                or row["profile"] != profile
                or row["turbulence_stream_seed"] != seed + 1000 * slot
                + d6.d5.d4.d3.FAMILIES.index(family)
                or row["episode_length"] != spec["episode_length"]
                or any(not math.isfinite(row[name]) for name in METRICS)
                or not math.isclose(row["training_objective"],
                                    row["measured_power"] - row["action_penalty"]
                                    - row["smooth_penalty"], rel_tol=1e-5, abs_tol=1e-7)):
            raise RuntimeError("G2-D7 回合流、来源状态或目标分解不一致")
    if quick:
        return ({"status": "QUICK_SMOKE_ONLY_NO_D6_CENTER_COMPARISON",
                 "unit": "shortened_16_frame_simulation_only",
                 "interpretation_boundary": "independent quick weather; no full-horizon or scientific conclusion"}, [])
    comparisons = []
    cells = {}
    for condition in CONDITIONS:
        weather_cells = {}
        for seed in seeds:
            arm_cells = {}
            for arm in ARMS:
                members = []
                for member in range(spec["initializations"]):
                    middle = center[condition, seed, member, arm]
                    pair = [index[condition, seed, member, arm, scale] for scale in SCALES]
                    if any(g["source_state_sha256"] != middle["source_state_sha256"] for g in pair):
                        raise RuntimeError("G2-D7 与 G2-D6 中心初态哈希不一致")
                    comparison = _compare_triplet(pair[0], middle, pair[1], cfg)
                    members.append(comparison)
                    comparisons.append(comparison)
                arm_cells[arm] = {
                    "members": len(members),
                    "mean_hard_centered_derivative": {
                        name: statistics.fmean(item["hard_centered_derivative"][name]
                                               for item in members) for name in METRICS},
                    "mean_d6_surrogate_d_objective_d_scale": statistics.fmean(
                        item["d6_surrogate_d_objective_d_scale"] for item in members),
                    "objective_direction_agreement_counts": {
                        label: sum(item["objective_direction_agreement"] == label
                                   for item in members)
                        for label in ("AGREE", "OPPOSITE_REQUIRES_STE_CHECK", "INDETERMINATE_FLAT")},
                    "physical_up_objective_down_penalty_reversal_count": sum(
                        item["physical_up_objective_down_penalty_reversal"] for item in members),
                }
            weather_cells[str(seed)] = arm_cells
        cells[condition] = weather_cells
    return ({
        "status": "EXPLORATORY_HARD_FORWARD_FINITE_DIFFERENCE_NO_GATE",
        "cells_by_condition_weather_and_arm": cells,
        "finite_difference_definition": "(J_1.8375-J_1.6625)/(1.8375-1.6625); each full 200-frame hard forward independently reset",
        "center_definition": "hash-locked G2-D6 1.75 output and surrogate gradient; not rerun in G2-D7",
        "unit": "one_complete_200_frame_episode_per_family_slot_member; only two independent weather clusters",
        "interpretation_boundary": "frozen pure-simulation development mechanism; no retraining, independent confirmation, real SLM, natural atmosphere, significance test, or RL rank change",
    }, comparisons)


def _execute(cfg: dict, parent: dict, train_cfg: dict, report: dict, output: Path,
             device: torch.device, center: dict, quick: bool) -> dict:
    spec = cfg["quick" if quick else "data"]
    base, _ = load_s1_config(_project_path(parent["environment_config"]))
    base = replace(base, num_modes=21, batch_size=1, episode_length=spec["episode_length"])
    basis, _, _ = build_action_basis(base, ActionRepresentation("r5_zernike21", "zernike", 21), device)
    nominal, shifted = d6.d5.d4.d3.train.g2.d1._profile_pairs(parent)
    policies = {}
    checkpoints = _project_path("outputs/s4_r5_g2_d2_matched_training_v1") / "checkpoints"
    for arm in ARMS:
        for member in range(spec["initializations"]):
            saved = torch.load(checkpoints / f"{arm}_policy_{member}_00512.pt",
                               map_location=device, weights_only=True)
            if (saved["arm"] != arm or saved["init"] != member or saved["update"] != 512
                    or saved["training_scale"] != dict(d6.d5.d4.d3.train.ARMS)[arm]
                    or saved["deployment_scale"] != CENTER):
                raise RuntimeError("G2-D7 冻结策略身份不符")
            policy = ResidualGRUPolicy(parent["policy"]["hidden_size"],
                                       parent["policy"]["output_size"]).to(device)
            policy.load_state_dict(saved["state_dict"])
            # 与 D6 相同的无 dropout GRU 模式；no_grad 保证不计算梯度或更新。
            policy.train().requires_grad_(False)
            policies[arm, member] = policy
    progress = d6.d5.d4.d3.SparseProgress(output, device, spec["episode_length"])
    progress.phase("G2-D7 CUDA 快速冒烟" if quick else "G2-D7 完整闭环硬前向开发诊断",
                   report["batched_environment_steps"])
    started = time.perf_counter()
    groups, rows = [], []
    try:
        with ((output / "groups.jsonl").open("w", encoding="utf-8") as group_file,
              (output / "records.jsonl").open("w", encoding="utf-8") as row_file):
            for condition, profiles in zip(CONDITIONS, (nominal, shifted), strict=True):
                for seed in stream_manifest(quick)["weather_bases"]:
                    for member in range(spec["initializations"]):
                        states = {scale: d6._fresh_state(
                            seed, spec["episode_length"], basis, base,
                            parent["families"], profiles, parent["data"]["sensor_seed_offset"])
                                  for scale in SCALES}
                        # 先核对两侧同源；第二臂随后再独立重建相同随机流。
                        first = states[SCALES[0]]
                        second = states[SCALES[1]]
                        source_hash = d6.d5.d4._same_state(first, second)
                        if not quick and source_hash != center[
                                condition, seed, member, ARMS[0]]["source_state_sha256"]:
                            raise RuntimeError("G2-D7 初态与冻结 G2-D6 中心不一致")
                        for arm in ARMS:
                            if arm != ARMS[0]:
                                states = {scale: d6._fresh_state(
                                    seed, spec["episode_length"], basis, base,
                                    parent["families"], profiles,
                                    parent["data"]["sensor_seed_offset"])
                                          for scale in SCALES}
                                if d6.d5.d4._same_state(states[SCALES[0]], states[SCALES[1]]) != source_hash:
                                    raise RuntimeError("G2-D7 两臂独立重置初态不一致")
                            for scale in SCALES:
                                state = states[scale]
                                result = _hard_rollout(
                                    state, policies[arm, member], scale_value=scale,
                                    action_weight=train_cfg["objective"]["action_weight"],
                                    smooth_weight=train_cfg["objective"]["smooth_weight"],
                                    progress=progress)
                                states[scale] = None
                                del state
                                group = {
                                    "hardware_condition": condition, "weather_seed": seed,
                                    "member": member, "controller": arm,
                                    "source_state_sha256": source_hash,
                                    "deployment_scale": scale, "episode_length": spec["episode_length"],
                                    "group_mean": result["group_mean"],
                                    "per_frame_batch_mean": result["per_frame_batch_mean"],
                                    "full_training_objective": result["full_training_objective"],
                                }
                                group_file.write(json.dumps(group, ensure_ascii=False) + "\n")
                                groups.append(group)
                                for slot, profile in enumerate(profiles):
                                    for family_index, family in enumerate(d6.d5.d4.d3.FAMILIES):
                                        index = slot * len(d6.d5.d4.d3.FAMILIES) + family_index
                                        row = {
                                            "hardware_condition": condition, "weather_seed": seed,
                                            "member": member, "controller": arm,
                                            "deployment_scale": scale, "family": family,
                                            "slot": slot, "profile": profile.identifier,
                                            "turbulence_stream_seed": seed + 1000 * slot + family_index,
                                            "source_state_sha256": source_hash,
                                            "episode_length": spec["episode_length"],
                                            **{name: result["per_episode"][name][index]
                                               for name in METRICS},
                                        }
                                        row_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                                        rows.append(row)
                                group_file.flush()
                                row_file.flush()
        if progress.bar.n != report["batched_environment_steps"]:
            raise RuntimeError("G2-D7 实际硬前向步数与预注册预算不符")
        analysis, comparisons = _summarize(groups, rows, cfg, center, quick=quick)
        with (output / "comparisons.jsonl").open("w", encoding="utf-8") as handle:
            for item in comparisons:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        result = {
            "status": "QUICK_SMOKE_NO_SCIENTIFIC_CONCLUSION" if quick
                      else "DEVELOPMENT_DIAGNOSTIC_REQUIRES_READ_ONLY_AUDIT",
            "groups": len(groups), "complete_episodes": len(rows),
            "physical_transitions": report["physical_transitions"],
            "comparisons": len(comparisons), "analysis": analysis,
            "elapsed_seconds": time.perf_counter() - started,
            "config_sha256": report["config_sha256"],
            "entry_sha256": report["entry_sha256"],
            "g2_d6_output_hashes": D6_OUTPUT_HASHES,
            "training_checkpoint_manifest_sha256": report["training_checkpoint_manifest_sha256"],
            "frozen_source_bundle_sha256": report["frozen_source_bundle_sha256"],
            "material_passport": {"origin_skill": "academic-research-suite / experiment-agent",
                                  "origin_mode": "run", "verification_status": "UNVERIFIED"},
            **cfg["boundary"],
            "next_action": "停止并等待只读审计；不训练、不改判 G2-D3、不打开独立确认集",
        }
        write_json(output / "summary.json", result)
        write_json(output / "SUCCESS.json", {
            "summary_sha256": _file_sha256(output / "summary.json"),
            "groups_sha256": _file_sha256(output / "groups.jsonl"),
            "records_sha256": _file_sha256(output / "records.jsonl"),
            "comparisons_sha256": _file_sha256(output / "comparisons.jsonl"),
            "progress_sha256": _file_sha256(output / "progress.jsonl"),
            "stream_manifest_sha256": _file_sha256(output / "stream_manifest.json"),
        })
        return result
    finally:
        progress.close()


def run(path: str | Path = CONFIG, *, quick: bool = False,
        preflight_only: bool = False) -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cfg, parent, train_cfg, report, output, device, center = preflight(path, quick=quick)
    if preflight_only:
        return report
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    created = False
    try:
        output.mkdir(parents=True, exist_ok=False)
        created = True
        write_json(output / "preflight.json", report)
        write_json(output / "config.json", cfg)
        write_json(output / "stream_manifest.json", stream_manifest(quick))
        write_json(output / "runtime.json", {
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device), "git": safe_git_record(),
            "diagnostic_owner": cfg["runtime"]["diagnostic_owner"],
            "frozen_source_bundle_sha256": d6.d5.d4.d3.train.FROZEN_SOURCE,
        })
        return _execute(cfg, parent, train_cfg, report, output, device, center, quick)
    except Exception:
        if created:
            try:
                write_json(output / "failure.json", {"traceback": traceback.format_exc(),
                                                      "automatic_retry": False})
            except Exception:
                pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--quick", action="store_true", help="独立天气 16 帧 CUDA 技术冒烟")
    parser.add_argument("--preflight-only", action="store_true", help="只读预检，不创建输出")
    args = parser.parse_args()
    print(json.dumps(run(args.config, quick=args.quick,
                         preflight_only=args.preflight_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
