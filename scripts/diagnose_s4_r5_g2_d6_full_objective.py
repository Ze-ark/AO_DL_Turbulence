"""G2-D6：冻结策略的完整闭环训练目标和部署倍率径向梯度开发诊断。"""
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

from scripts import diagnose_s4_r5_g2_d5_gradient as d5
from src.rl.r4_control import NominalCalibration
from src.rl.r4_dynamics_experiment import safe_git_record
from src.rl.r4_interface_smoke import write_json
from src.rl.r4_observation import PowerMeasurement, R4Interface
from src.rl.r4_trajectory import anchor_delta
from src.rl.r5_batched_environment import R5BatchedEnvironment
from src.rl.r5_physics_adapter import measured_objective
from src.rl.r5_policy_training import ResidualGRUPolicy
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _project_path
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.robust_control import RobustnessCondition


CONFIG = "configs/experiments/s4_r5_g2_d6_full_objective_v1.yaml"
D5_CONFIG_SHA256 = "47462a70a4cda9cd94c710a313d399ab080531be9d87bcd5c4d22d77e2f62269"
D5_ENTRY_SHA256 = "41b1e72dc55a7914b9b36f9c381536dee27fc5ec84ddc1aee1f7402dbaf325f5"
D5_OUTPUT_HASHES = {
    "summary.json": "94b7de9e7efbfc4fd23d2a8a6c707d59dff82652b099d7baed4a63a5d64d7e12",
    "records.jsonl": "cabe12682e31263c1e03ce2aed05cc4962aa2f8e6e5827fde57040702e1ff315",
    "progress.jsonl": "d363f57190a93200abecfe8653340e109731c74e6d24638f5b93439766390c53",
    "stream_manifest.json": "1663d1cf346cefd23c4192d009f3b6cd1bbb61812548163e7ecf1be9a5342539",
    "SUCCESS.json": "e6967e63aa264b89dfe9275e3df8182b52ec4ec844644f76166687dd5d6b7aa3",
}
ARMS = ("train_scale_1", "train_scale_1_75")
CONDITIONS = ("nominal_clone", "hardware_shift")
FORMAL_BASE, QUICK_BASE = 7_340_000, 7_350_000
METRICS = ("measured_power", "action_penalty", "smooth_penalty", "training_objective",
           "physical_power", "violation", "saturation", "slew_limited",
           "correction_clipped_fraction", "requested_applied_gap_abs")


def _contract(cfg: dict) -> None:
    expected = {
        "stage": "S4-D2-R5-G2-D6",
        "purpose": "frozen_full_horizon_training_objective_radial_development_diagnostic",
        "runtime": {"device": "cuda", "diagnostic_owner": "assistant_under_user_goal",
                    "automatic_retry": False},
        "g2_d5_config": d5.CONFIG, "g2_d5_config_sha256": D5_CONFIG_SHA256,
        "g2_d5_entry_sha256": D5_ENTRY_SHA256,
        "g2_d5_output": "outputs/s4_r5_g2_d5_gradient_v1",
        "g2_d5_output_hashes": D5_OUTPUT_HASHES,
        "controllers": list(ARMS), "hardware_conditions": list(CONDITIONS),
        "deployment_scale": d5.d4.d3.SCALE,
        "data": {"seed_base": FORMAL_BASE, "seed_stride": 10, "weather_count": 2,
                 "episode_length": 200, "initializations": 3},
        "quick": {"seed_base": QUICK_BASE, "weather_count": 1,
                  "episode_length": 16, "initializations": 1},
        "output_directory": "outputs/s4_r5_g2_d6_full_objective_v1",
        "quick_directory": "outputs/s4_r5_g2_d6_full_objective_v1_quick_r1",
        "boundary": {"confirmation_access": False, "training_updates": 0,
                     "optimizer_updates": 0, "real_slm_actions": False,
                     "automatic_retry": False, "historical_results_read_only": True,
                     "independent_confirmation": False},
    }
    if cfg != expected:
        raise ValueError("G2-D6 冻结完整闭环开发诊断合同已改变")


def stream_manifest(quick: bool) -> dict:
    seeds = ([QUICK_BASE] if quick else [FORMAL_BASE, FORMAL_BASE + 10])
    return {
        "weather_bases": seeds,
        "turbulence": [seed + 1000 * slot + family for seed in seeds
                       for slot in range(6) for family in range(3)],
        "sensor": [seed + 1000 * slot + 50_000_000 for seed in seeds
                   for slot in range(6)],
        "power": [seed + 1000 * slot + 60_000_000 for seed in seeds
                  for slot in range(6)],
        "shared_between_arms_members_and_conditions": True,
        "same_seed_restarts_each_closed_loop": True,
    }


def _verify_d5_and_lineage(cfg: dict) -> tuple[dict, dict]:
    if (_file_sha256(_project_path(cfg["g2_d5_config"])) != D5_CONFIG_SHA256
            or _file_sha256(Path(d5.__file__)) != D5_ENTRY_SHA256):
        raise RuntimeError("G2-D5 冻结配置或入口变化")
    d5_cfg = _load_yaml(_project_path(cfg["g2_d5_config"]))
    d5._contract(d5_cfg)
    if (_file_sha256(_project_path(d5_cfg["g2_d4_config"])) != d5.D4_CONFIG_SHA256
            or _file_sha256(Path(d5.d4.__file__)) != d5.D4_ENTRY_SHA256):
        raise RuntimeError("G2-D4 冻结配置或入口变化")
    d4_cfg = _load_yaml(_project_path(d5_cfg["g2_d4_config"]))
    d5.d4._contract(d4_cfg)
    d3_cfg = _load_yaml(_project_path(d4_cfg["g2_d3_config"]))
    d5.d4.d3._contract(d3_cfg)
    _, parent = d5.d4.d3._verify_lineage(d3_cfg)
    for root, hashes, label in (
        (_project_path(d5_cfg["g2_d4_output"]), d5.D4_OUTPUT_HASHES, "G2-D4"),
        (_project_path(cfg["g2_d5_output"]), D5_OUTPUT_HASHES, "G2-D5"),
    ):
        if (root / "failure.json").exists():
            raise RuntimeError(f"{label} 历史输出有失败标记")
        for name, digest in hashes.items():
            if _file_sha256(root / name) != digest:
                raise RuntimeError(f"{label} 历史证据变化: {name}")
        success = json.loads((root / "SUCCESS.json").read_text(encoding="utf-8"))
        for key, name in (("summary_sha256", "summary.json"),
                          ("records_sha256", "records.jsonl"),
                          ("progress_sha256", "progress.jsonl"),
                          ("stream_manifest_sha256", "stream_manifest.json")):
            if success.get(key) != hashes[name]:
                raise RuntimeError(f"{label} SUCCESS 与历史文件不一致")
    prior = json.loads((_project_path(cfg["g2_d5_output"]) / "summary.json").read_text(
        encoding="utf-8"))
    if (prior.get("status") != "DEVELOPMENT_DIAGNOSTIC_REQUIRES_READ_ONLY_AUDIT"
            or prior.get("records") != 864 or prior.get("paired_source_states") != 432
            or prior.get("physical_transitions") != 45_360
            or prior.get("config_sha256") != D5_CONFIG_SHA256
            or prior.get("entry_sha256") != D5_ENTRY_SHA256
            or prior.get("training_updates") != 0
            or prior.get("confirmation_access") is not False
            or prior.get("real_slm_actions") is not False):
        raise RuntimeError("G2-D5 历史完成性或安全边界不符")
    train_cfg = _load_yaml(_project_path(d5.d4.d3.train.CONFIG))
    d5.d4.d3.train._contract(train_cfg)
    if train_cfg["objective"] != {"action_weight": 0.01, "smooth_weight": 0.001,
                                  "discount": 1.0}:
        raise RuntimeError("G2-D2 原训练目标权重变化")
    return parent, train_cfg


def _check_stream_isolation() -> None:
    now = (stream_manifest(False), stream_manifest(True))
    old = (d5.stream_manifest(False), d5.stream_manifest(True),
           d5.d4.stream_manifest(False), d5.d4.stream_manifest(True),
           d5.d4.d3.stream_manifest(False), d5.d4.d3.stream_manifest(True),
           d5.d4.d3.train.stream_manifest(quick=False),
           d5.d4.d3.train.stream_manifest(quick=True),
           d5.d4.d3.train.g2.stream_manifest(False),
           d5.d4.d3.train.g2.stream_manifest(True))
    for name in ("turbulence", "sensor", "power"):
        formal, quick = (set(item[name]) for item in now)
        previous = set().union(*(set(item[name]) for item in old))
        if (len(formal) != len(now[0][name]) or len(quick) != len(now[1][name])
                or formal & quick or (formal | quick) & previous):
            raise RuntimeError(f"G2-D6 {name} 随机流与旧开发或快速冒烟重叠")


def preflight(path: str | Path = CONFIG, *, quick: bool = False
              ) -> tuple[dict, dict, dict, dict, Path, torch.device]:
    cfg = _load_yaml(_project_path(path))
    _contract(cfg)
    parent, train_cfg = _verify_d5_and_lineage(cfg)
    _check_stream_isolation()
    if (parent["data"]["sensor_seed_offset"] != 50_000_000
            or [family["id"] for family in parent["families"]] != list(d5.d4.d3.FAMILIES)):
        raise RuntimeError("R5 因果观测或湍流家族变化")
    nominal, shifted = d5.d4.d3.train.g2.d1._profile_pairs(parent)
    if ([item.identifier for item in shifted] != list(d5.d4.d3.PROFILES)
            or [item.identifier for item in nominal]
            != [f"nominal_for_{name}" for name in d5.d4.d3.PROFILES]):
        raise RuntimeError("G2-D6 硬件档位配对变化")
    spec = cfg["quick" if quick else "data"]
    base, _ = load_s1_config(_project_path(parent["environment_config"]))
    for family in parent["families"]:
        condition = RobustnessCondition.from_mapping(dict(family, base_seed=spec["seed_base"]))
        simulation = condition.environment_config(replace(base, episode_length=spec["episode_length"]))
        displacement = (simulation.wind_speed_mps
                        * (1 + simulation.wind_speed_modulation_fraction)
                        * simulation.dt_s * spec["episode_length"] / simulation.sample_pitch_m)
        if displacement >= simulation.turbulence_grid_size:
            raise RuntimeError("G2-D6 相位屏在完整回合内重复")
    output = _project_path(cfg["quick_directory" if quick else "output_directory"])
    if output.exists():
        raise FileExistsError(f"保留已有 G2-D6 输出，不覆盖或重跑: {output}")
    device = resolve_device("cuda")
    groups = len(CONDITIONS) * spec["weather_count"] * spec["initializations"] * len(ARMS)
    report = {
        "status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_IDE",
        "device": str(device), "quick": quick, "weather_count": spec["weather_count"],
        "episode_length": spec["episode_length"], "batched_closed_loops": groups,
        "complete_episodes": groups * len(parent["families"]) * len(nominal),
        "batched_environment_steps": groups * spec["episode_length"],
        "physical_transitions": groups * len(parent["families"]) * len(nominal)
                                * spec["episode_length"],
        "training_objective": (
            f"mean_over_{spec['episode_length']}_frames_and_18_episodes_of_"
            "measured_power_minus_action_penalty_minus_smooth_penalty"),
        "gradient": "d_full_closed_loop_mean_training_objective_d_global_deployment_scale_at_1_75",
        "config_sha256": _file_sha256(_project_path(path)),
        "entry_sha256": _file_sha256(Path(__file__)),
        "g2_d5_output_hashes": D5_OUTPUT_HASHES,
        "training_checkpoint_manifest_sha256": d5.d4.d3.TRAIN_HASHES["checkpoint_manifest.json"],
        "frozen_source_bundle_sha256": d5.d4.d3.train.FROZEN_SOURCE,
        **cfg["boundary"],
    }
    return cfg, parent, train_cfg, report, output, device


def _fresh_state(seed: int, steps: int, basis: torch.Tensor, base, families: list[dict],
                 profiles: list, sensor_offset: int) -> tuple[R5BatchedEnvironment, R4Interface]:
    condition = RobustnessCondition.from_mapping(dict(families[0], base_seed=seed))
    env_cfg = replace(condition.environment_config(base),
                      batch_size=len(families) * len(profiles), episode_length=steps)
    env = R5BatchedEnvironment(env_cfg, basis.device, basis, families, profiles, sensor_offset)
    raw, _ = env.reset(seed=seed)
    interface = R4Interface(calibration=NominalCalibration())
    interface.reset(env.proxy(raw), episode_id=f"r5-{seed}")
    return env, interface


def _rollout(state: tuple[R5BatchedEnvironment, R4Interface], policy: ResidualGRUPolicy,
             *, scale_value: float, action_weight: float, smooth_weight: float,
             progress: d5.d4.d3.SparseProgress) -> dict:
    env, interface = state
    batch = env.config.batch_size
    scale = torch.tensor(scale_value, device=env.device, dtype=env.basis.dtype,
                         requires_grad=True)
    previous = torch.zeros((batch, 11), device=env.device, dtype=scale.dtype)
    frames: dict[str, list[torch.Tensor]] = {name: [] for name in METRICS}
    scores: list[torch.Tensor] = []
    for step in range(env.config.episode_length):
        view = interface.snapshot()
        raw_action = policy(view.features, view.valid)
        correction = raw_action * scale
        baseline = anchor_delta(view.features[:, -1],
                                {"gain": .15, "leak": .10, "tracking_gain": .50})
        action = interface.issue(baseline, correction, step=step)
        raw, _, terminated, truncated, info = env.step(action.requested_delta_rad)
        interface.observe_next(env.proxy(raw), step=step + 1,
                               power=PowerMeasurement(info["measured_power_in_bucket"],
                                                      step, step + 1))
        normalized = action.normalized_correction
        measured = info["measured_power_in_bucket"]
        action_penalty = action_weight * normalized.square().mean(-1)
        smooth_penalty = smooth_weight * (normalized - previous).square().mean(-1)
        objective = measured_objective(measured, normalized, previous,
                                       action_weight, smooth_weight)
        frames["measured_power"].append(measured)
        frames["action_penalty"].append(action_penalty)
        frames["smooth_penalty"].append(smooth_penalty)
        frames["training_objective"].append(objective)
        frames["physical_power"].append(info["reward_power_in_bucket"])
        frames["violation"].append(info["violation_fraction"])
        frames["saturation"].append(info["saturated_fraction"])
        frames["slew_limited"].append(info["slew_limited_fraction"])
        frames["correction_clipped_fraction"].append(correction.abs().gt(1).float().mean(-1))
        frames["requested_applied_gap_abs"].append(
            (info["requested_modal"] - info["applied_modal"]).abs().mean(-1))
        scores.append(objective.mean())  # 与 G2-D2 / _rollout_batch 的逐帧批均值一致。
        previous = normalized
        if bool(truncated.any()) or bool(terminated.all()) != (step == env.config.episode_length - 1):
            raise RuntimeError("G2-D6 完整闭环回合提前或错误终止")
        progress.tick()
    full_objective = torch.stack(scores).mean()  # 原训练使用 discount=1.0。
    matrices = {name: torch.stack(values, dim=1) for name, values in frames.items()}
    for name, values in matrices.items():
        if values.shape != (batch, env.config.episode_length) or not bool(torch.isfinite(values).all()):
            raise RuntimeError(f"G2-D6 {name} 帧矩阵不完整或非有限")
    component_check = (matrices["measured_power"].mean()
                       - matrices["action_penalty"].mean()
                       - matrices["smooth_penalty"].mean())
    if not torch.allclose(full_objective, component_check, rtol=1e-5, atol=1e-7):
        raise RuntimeError("G2-D6 原训练目标均值分解不闭合")
    if not full_objective.requires_grad:
        raise RuntimeError("G2-D6 完整闭环目标与全局部署倍率断开")
    derivative = torch.autograd.grad(full_objective, scale, allow_unused=False)[0]
    if not bool(torch.isfinite(derivative)):
        raise RuntimeError("G2-D6 完整闭环训练目标梯度非有限")
    return {
        "per_episode": {name: values.detach().mean(dim=1).tolist()
                        for name, values in matrices.items()},
        "per_frame_batch_mean": {name: values.detach().mean(dim=0).tolist()
                                 for name, values in matrices.items()
                                 if name in ("measured_power", "action_penalty", "smooth_penalty",
                                             "training_objective", "physical_power")},
        "group_mean": {name: float(values.detach().mean()) for name, values in matrices.items()},
        "full_training_objective": float(full_objective.detach()),
        "d_objective_d_deployment_scale": float(derivative.detach()),
        "radial_d_objective_d_alpha": float((scale.detach() * derivative.detach())),
    }


def _summarize(groups: list[dict], rows: list[dict], cfg: dict, *, quick: bool) -> dict:
    spec = cfg["quick" if quick else "data"]
    seeds = stream_manifest(quick)["weather_bases"]
    expected = {(condition, seed, member, arm)
                for condition in CONDITIONS for seed in seeds
                for member in range(spec["initializations"]) for arm in ARMS}
    index = {(g["hardware_condition"], g["weather_seed"], g["member"], g["controller"]): g
             for g in groups}
    if len(groups) != len(expected) or set(index) != expected:
        raise RuntimeError("G2-D6 组别缺失或重复")
    for condition in CONDITIONS:
        for seed in seeds:
            for member in range(spec["initializations"]):
                pair = [index[condition, seed, member, arm] for arm in ARMS]
                if len({g["source_state_sha256"] for g in pair}) != 1:
                    raise RuntimeError("G2-D6 两臂初态哈希不一致")
    for group in groups:
        numeric = [group["full_training_objective"],
                   group["d_objective_d_deployment_scale"],
                   group["radial_d_objective_d_alpha"],
                   *group["group_mean"].values()]
        if (set(group["group_mean"]) != set(METRICS)
                or any(not math.isfinite(value) for value in numeric)
                or not math.isclose(group["radial_d_objective_d_alpha"],
                                    group["deployment_scale"]
                                    * group["d_objective_d_deployment_scale"],
                                    rel_tol=1e-5, abs_tol=1e-7)):
            raise RuntimeError("G2-D6 组梯度、均值或径向换算无效")
        for name, trace in group["per_frame_batch_mean"].items():
            if (name not in METRICS or len(trace) != spec["episode_length"]
                    or any(not math.isfinite(value) for value in trace)
                    or not math.isclose(statistics.fmean(trace), group["group_mean"][name],
                                        rel_tol=1e-5, abs_tol=1e-7)):
                raise RuntimeError("G2-D6 逐帧均值与完整回合均值不一致")
    expected_rows = {
        (condition, seed, member, arm, family, slot)
        for condition in CONDITIONS for seed in seeds
        for member in range(spec["initializations"]) for arm in ARMS
        for family in d5.d4.d3.FAMILIES for slot in range(len(d5.d4.d3.PROFILES))
    }
    row_index = {(row["hardware_condition"], row["weather_seed"], row["member"],
                  row["controller"], row["family"], row["slot"]): row for row in rows}
    if len(rows) != len(expected_rows) or set(row_index) != expected_rows:
        raise RuntimeError("G2-D6 完整回合记录索引缺失或重复")
    for key, row in row_index.items():
        condition, seed, member, arm, family, slot = key
        group = index[condition, seed, member, arm]
        profile = (f"nominal_for_{d5.d4.d3.PROFILES[slot]}"
                   if condition == "nominal_clone" else d5.d4.d3.PROFILES[slot])
        if (row["source_state_sha256"] != group["source_state_sha256"]
                or row["profile"] != profile
                or row["turbulence_stream_seed"] != seed + 1000 * slot
                + d5.d4.d3.FAMILIES.index(family)
                or any(not math.isfinite(row[name]) for name in METRICS)
                or not math.isclose(row["training_objective"],
                                    row["measured_power"] - row["action_penalty"]
                                    - row["smooth_penalty"], rel_tol=1e-5, abs_tol=1e-7)):
            raise RuntimeError("G2-D6 完整回合流、来源状态或目标分解不一致")
    cells: dict[str, dict] = {}
    for condition in CONDITIONS:
        weather_cells = {}
        for seed in seeds:
            arms = {}
            for arm in ARMS:
                selected = [index[condition, seed, member, arm]
                            for member in range(spec["initializations"])]
                arms[arm] = {
                    "members": len(selected),
                    "mean": {name: statistics.fmean(g["group_mean"][name] for g in selected)
                             for name in METRICS},
                    "mean_d_objective_d_deployment_scale": statistics.fmean(
                        g["d_objective_d_deployment_scale"] for g in selected),
                    "mean_radial_d_objective_d_alpha": statistics.fmean(
                        g["radial_d_objective_d_alpha"] for g in selected),
                }
            delta = {name: arms[ARMS[1]]["mean"][name] - arms[ARMS[0]]["mean"][name]
                     for name in METRICS}
            weather_cells[str(seed)] = {"arms": arms,
                                        "matched_minus_unmatched_mean": delta,
                                        "paired_member_deltas": [
                                            {name: index[condition, seed, member, ARMS[1]]["group_mean"][name]
                                             - index[condition, seed, member, ARMS[0]]["group_mean"][name]
                                             for name in METRICS}
                                            for member in range(spec["initializations"])]}
        cells[condition] = weather_cells
    return {
        "status": "EXPLORATORY_FULL_HORIZON_OBJECTIVE_NO_GATE",
        "cells_by_condition_and_weather": cells,
        "objective_definition": (
            "same_G2_D2_rollout_batch_terms_and_weights_at_common_deployment_scale_1_75; "
            f"horizon={spec['episode_length']}; quick_smoke_is_shortened"),
        "gradient_definition": (
            f"autograd_d_mean_{spec['episode_length']}_frame_measured_objective_"
            "d_global_scale_at_1_75_with_frozen_policy_and_full_feedback; "
            "uses_differentiable_SLM_surrogate_not_hard_finite_difference"),
        "radial_definition": "dJ/dalpha_for_correction=raw*1.75*alpha_at_alpha=1",
        "unit": (f"one_complete_{spec['episode_length']}_frame_episode_per_family_slot_member; "
                 "weather seeds are independent clusters"),
        "interpretation_boundary": "new development weather only; no optimizer step, hard finite difference, independent confirmation, real SLM or natural atmosphere",
    }


def _execute(cfg: dict, parent: dict, train_cfg: dict, report: dict, output: Path,
             device: torch.device, quick: bool) -> dict:
    spec = cfg["quick" if quick else "data"]
    base, _ = load_s1_config(_project_path(parent["environment_config"]))
    base = replace(base, num_modes=21, batch_size=1, episode_length=spec["episode_length"])
    basis, _, _ = build_action_basis(base, ActionRepresentation("r5_zernike21", "zernike", 21), device)
    nominal, shifted = d5.d4.d3.train.g2.d1._profile_pairs(parent)
    policies = {}
    training_output = _project_path("outputs/s4_r5_g2_d2_matched_training_v1") / "checkpoints"
    for arm in ARMS:
        for member in range(spec["initializations"]):
            name = f"{arm}_policy_{member}_00512.pt"
            saved = torch.load(training_output / name, map_location=device, weights_only=True)
            if (saved["arm"] != arm or saved["init"] != member or saved["update"] != 512
                    or saved["training_scale"] != dict(d5.d4.d3.train.ARMS)[arm]
                    or saved["deployment_scale"] != cfg["deployment_scale"]):
                raise RuntimeError("G2-D6 冻结策略身份不符")
            policy = ResidualGRUPolicy(parent["policy"]["hidden_size"],
                                       parent["policy"]["output_size"]).to(device)
            policy.load_state_dict(saved["state_dict"])
            # cuDNN 的 GRU 输入梯度反传要求训练模式；此模型无 dropout，
            # 因而前向与 eval 相同。冻结权重但保留对历史观测和动作倍率的梯度。
            policy.train().requires_grad_(False)
            policies[arm, member] = policy
    progress = d5.d4.d3.SparseProgress(output, device, spec["episode_length"])
    progress.phase("G2-D6 CUDA 快速冒烟" if quick else "G2-D6 完整闭环目标开发诊断",
                   report["batched_environment_steps"])
    started = time.perf_counter()
    groups, rows = [], []
    try:
        with ((output / "groups.jsonl").open("w", encoding="utf-8") as group_file,
              (output / "records.jsonl").open("w", encoding="utf-8") as row_file):
            for condition, profiles in zip(CONDITIONS, (nominal, shifted), strict=True):
                for seed in stream_manifest(quick)["weather_bases"]:
                    for member in range(spec["initializations"]):
                        states = [_fresh_state(seed, spec["episode_length"], basis, base,
                                               parent["families"], profiles,
                                               parent["data"]["sensor_seed_offset"])
                                  for _ in ARMS]
                        source_hash = d5.d4._same_state(states[0], states[1])
                        for arm_index, arm in enumerate(ARMS):
                            state = states[arm_index]
                            result = _rollout(
                                state, policies[arm, member],
                                scale_value=cfg["deployment_scale"],
                                action_weight=train_cfg["objective"]["action_weight"],
                                smooth_weight=train_cfg["objective"]["smooth_weight"],
                                progress=progress)
                            states[arm_index] = None  # 放开完整反向图，不让第一臂占住第二臂显存。
                            del state
                            group = {
                                "hardware_condition": condition, "weather_seed": seed,
                                "member": member, "controller": arm,
                                "source_state_sha256": source_hash,
                                "deployment_scale": cfg["deployment_scale"],
                                "episode_length": spec["episode_length"],
                                "group_mean": result["group_mean"],
                                "per_frame_batch_mean": result["per_frame_batch_mean"],
                                "full_training_objective": result["full_training_objective"],
                                "d_objective_d_deployment_scale": result["d_objective_d_deployment_scale"],
                                "radial_d_objective_d_alpha": result["radial_d_objective_d_alpha"],
                            }
                            if not math.isclose(group["group_mean"]["training_objective"],
                                                group["full_training_objective"], rel_tol=1e-5,
                                                abs_tol=1e-7):
                                raise RuntimeError("G2-D6 组均值与原训练评分不一致")
                            group_file.write(json.dumps(group, ensure_ascii=False) + "\n")
                            groups.append(group)
                            for slot, profile in enumerate(profiles):
                                for family_index, family in enumerate(d5.d4.d3.FAMILIES):
                                    index = slot * len(d5.d4.d3.FAMILIES) + family_index
                                    row = {
                                        "hardware_condition": condition,
                                        "weather_seed": seed,
                                        "member": member, "controller": arm,
                                        "family": family, "slot": slot,
                                        "profile": profile.identifier,
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
            raise RuntimeError("G2-D6 实际仿真步数与预注册预算不符")
        analysis = _summarize(groups, rows, cfg, quick=quick)
        result = {
            "status": "QUICK_SMOKE_NO_SCIENTIFIC_CONCLUSION" if quick
                      else "DEVELOPMENT_DIAGNOSTIC_REQUIRES_READ_ONLY_AUDIT",
            "groups": len(groups), "complete_episodes": len(rows),
            "physical_transitions": report["physical_transitions"],
            "analysis": analysis,
            "elapsed_seconds": time.perf_counter() - started,
            "config_sha256": report["config_sha256"],
            "entry_sha256": report["entry_sha256"],
            "g2_d5_output_hashes": D5_OUTPUT_HASHES,
            "training_checkpoint_manifest_sha256": report["training_checkpoint_manifest_sha256"],
            "frozen_source_bundle_sha256": report["frozen_source_bundle_sha256"],
            "material_passport": {"origin_skill": "academic-research-suite / experiment-agent",
                                  "origin_mode": "run", "verification_status": "UNVERIFIED"},
            **cfg["boundary"],
            "next_action": "停止并等待只读开发机制审计；不改判G2-D3或打开独立确认集",
        }
        write_json(output / "summary.json", result)
        write_json(output / "SUCCESS.json", {
            "summary_sha256": _file_sha256(output / "summary.json"),
            "groups_sha256": _file_sha256(output / "groups.jsonl"),
            "records_sha256": _file_sha256(output / "records.jsonl"),
            "progress_sha256": _file_sha256(output / "progress.jsonl"),
            "stream_manifest_sha256": _file_sha256(output / "stream_manifest.json"),
        })
        return result
    finally:
        progress.close()


def run(path: str | Path = CONFIG, *, quick: bool = False,
        preflight_only: bool = False) -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cfg, parent, train_cfg, report, output, device = preflight(path, quick=quick)
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
            "frozen_source_bundle_sha256": d5.d4.d3.train.FROZEN_SOURCE,
        })
        return _execute(cfg, parent, train_cfg, report, output, device, quick)
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
    parser.add_argument("--quick", action="store_true", help="16 帧 CUDA 技术冒烟，不作科学排名")
    parser.add_argument("--preflight-only", action="store_true", help="只读预检，不创建输出")
    args = parser.parse_args()
    print(json.dumps(run(args.config, quick=args.quick,
                         preflight_only=args.preflight_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
