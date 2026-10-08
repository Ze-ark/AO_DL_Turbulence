"""G2-D5：冻结策略的延迟物理梯度与共同随机流硬前向开发诊断。"""
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

from scripts import diagnose_s4_r5_g2_d4_same_state as d4
from src.rl.r4_dynamics_experiment import safe_git_record
from src.rl.r4_interface_smoke import write_json
from src.rl.r4_observation import R4Interface
from src.rl.r4_trajectory import anchor_delta
from src.rl.r5_batched_environment import R5BatchedEnvironment
from src.rl.r5_physics_adapter import measured_objective
from src.rl.r5_policy_training import ResidualGRUPolicy
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _project_path
from src.runtime import resolve_device
from src.simulation.config import load_s1_config


CONFIG = "configs/experiments/s4_r5_g2_d5_gradient_v1.yaml"
D4_CONFIG_SHA256 = "7eb356b5e3b333fd92eaaa6900b0768838613967adc1253df7b047cc0546eb97"
D4_ENTRY_SHA256 = "74e2ccd1318fbf76cf67d9aa040451e1300bf2bb7369b27f7c81074dc6d6a649"
D4_OUTPUT_HASHES = {
    "summary.json": "16ff93da219e51210d9749382b99e1697cf63ec3cd196d53a165572dd8fbf358",
    "records.jsonl": "e1b8008dbef5a71b4577ab98ca563f7b686f0d0d291ae791990b7555f321382c",
    "progress.jsonl": "76733ca894b5ca9fa0bb681993baa3d1f075e45c4fe14befaf49b98d14a13bbd",
    "stream_manifest.json": "77e14f3c8e2026ba2e0337078e29d1469fb90e3327faeddd84dd4ecfcfa65f63",
    "SUCCESS.json": "a0df7951b8bf46ad9bd1941c17e63e6adc525ab2e0171534af1a118c925075c3",
}
FORMAL_BASE, QUICK_BASE = 7_320_000, 7_330_000
BRANCHES = ("gradient", "alpha_minus", "alpha_plus")


def _contract(cfg: dict) -> None:
    expected = {
        "stage": "S4-D2-R5-G2-D5",
        "purpose": "frozen_same_state_delayed_physical_gradient_and_hard_forward_probe",
        "runtime": {"device": "cuda", "formal_owner": "user_ide", "automatic_retry": False},
        "g2_d4_config": d4.CONFIG, "g2_d4_config_sha256": D4_CONFIG_SHA256,
        "g2_d4_entry_sha256": D4_ENTRY_SHA256,
        "g2_d4_output": "outputs/s4_r5_g2_d4_same_state_v1",
        "g2_d4_output_hashes": D4_OUTPUT_HASHES,
        "controllers": list(d4.ARMS), "deployment_scale": d4.d3.SCALE,
        "hard_forward_alpha_epsilon": 0.05, "gradient_zero_tolerance": 1e-9,
        "data": {"seed_base": FORMAL_BASE, "seed_stride": 10, "weather_count": 2,
                 "episode_length": 200, "probe_steps": [0, 25], "hold_steps": 5,
                 "initializations": 3},
        "quick": {"seed_base": QUICK_BASE, "weather_count": 1,
                  "episode_length": 16, "probe_steps": [0], "hold_steps": 5,
                  "initializations": 1},
        "output_directory": "outputs/s4_r5_g2_d5_gradient_v1",
        "quick_directory": "outputs/s4_r5_g2_d5_gradient_v1_quick",
        "boundary": {"confirmation_access": False, "training_updates": 0,
                     "real_slm_actions": False, "automatic_retry": False,
                     "historical_results_read_only": True,
                     "independent_confirmation": False},
    }
    if cfg != expected:
        raise ValueError("G2-D5 冻结开发诊断合同已改变")


def stream_manifest(quick: bool) -> dict:
    base, count = (QUICK_BASE, 1) if quick else (FORMAL_BASE, 2)
    weather = [base + 10 * i for i in range(count)]
    return {
        "weather_bases": weather,
        "turbulence": [seed + 1000 * slot + family for seed in weather
                       for slot in range(6) for family in range(3)],
        "sensor": [seed + 1000 * slot + 50_000_000 for seed in weather
                   for slot in range(6)],
        "power": [seed + 1000 * slot + 60_000_000 for seed in weather
                  for slot in range(6)],
        "shared_between_arms_and_alpha_branches": True,
    }


def preflight(path: str | Path = CONFIG, *, quick: bool = False
              ) -> tuple[dict, dict, dict, Path, torch.device]:
    cfg = _load_yaml(_project_path(path))
    _contract(cfg)
    if (_file_sha256(_project_path(cfg["g2_d4_config"])) != D4_CONFIG_SHA256
            or _file_sha256(Path(d4.__file__)) != D4_ENTRY_SHA256):
        raise RuntimeError("G2-D4 冻结配置或入口变化")
    d4_cfg = _load_yaml(_project_path(cfg["g2_d4_config"]))
    d4._contract(d4_cfg)
    d3_cfg = _load_yaml(_project_path(d4_cfg["g2_d3_config"]))
    d4.d3._contract(d3_cfg)
    _, parent = d4.d3._verify_lineage(d3_cfg)
    old_output = _project_path(cfg["g2_d4_output"])
    if (old_output / "failure.json").exists():
        raise RuntimeError("G2-D4 历史输出有失败标记")
    for name, digest in D4_OUTPUT_HASHES.items():
        if _file_sha256(old_output / name) != digest:
            raise RuntimeError(f"G2-D4 历史证据变化: {name}")
    success = json.loads((old_output / "SUCCESS.json").read_text(encoding="utf-8"))
    for key, name in (("summary_sha256", "summary.json"),
                      ("records_sha256", "records.jsonl"),
                      ("progress_sha256", "progress.jsonl"),
                      ("stream_manifest_sha256", "stream_manifest.json")):
        if success.get(key) != D4_OUTPUT_HASHES[name]:
            raise RuntimeError("G2-D4 SUCCESS 与历史文件不一致")
    prior = json.loads((old_output / "summary.json").read_text(encoding="utf-8"))
    if (prior.get("status") != "DEVELOPMENT_DIAGNOSTIC_REQUIRES_READ_ONLY_AUDIT"
            or prior.get("records") != 3456 or prior.get("paired_source_states") != 1728
            or prior.get("physical_transitions") != 233280
            or prior.get("confirmation_access") is not False
            or prior.get("real_slm_actions") is not False
            or prior.get("training_updates") != 0):
        raise RuntimeError("G2-D4 历史计数或安全边界不符")
    current = (stream_manifest(False), stream_manifest(True))
    old = (d4.stream_manifest(False), d4.stream_manifest(True),
           d4.d3.stream_manifest(False), d4.d3.stream_manifest(True),
           d4.d3.train.stream_manifest(quick=False),
           d4.d3.train.stream_manifest(quick=True),
           d4.d3.train.g2.stream_manifest(False),
           d4.d3.train.g2.stream_manifest(True))
    for name in ("turbulence", "sensor", "power"):
        left, right = set(current[0][name]), set(current[1][name])
        previous = set().union(*(set(item[name]) for item in old))
        if (len(left) != len(current[0][name]) or len(right) != len(current[1][name])
                or left & right or (left | right) & previous):
            raise RuntimeError(f"G2-D5 {name} 随机流与历史开发或快速冒烟重叠")
    spec = cfg["quick" if quick else "data"]
    nominal, shifted = d4.d3.train.g2.d1._profile_pairs(parent)
    if (len(nominal) != 6 or len(shifted) != 6
            or max(p.slm_delay_frames for p in nominal + shifted) >= spec["hold_steps"]
            or any(step + spec["hold_steps"] > spec["episode_length"]
                   for step in spec["probe_steps"])):
        raise RuntimeError("G2-D5 硬件延迟或探针时域超出五帧窗口")
    output = _project_path(cfg["quick_directory" if quick else "output_directory"])
    if output.exists():
        raise FileExistsError(f"保留已有 G2-D5 输出，不覆盖或重跑: {output}")
    device = resolve_device("cuda")
    pairs = (len(d4.d3.CONDITIONS) * spec["weather_count"] * len(spec["probe_steps"])
             * spec["initializations"] * len(d4.d3.FAMILIES) * len(d4.d3.PROFILES))
    batched_steps = (len(d4.d3.CONDITIONS) * spec["weather_count"]
                     * spec["initializations"] * len(d4.ARMS) * len(BRANCHES)
                     * sum(step + spec["hold_steps"] for step in spec["probe_steps"]))
    report = {
        "status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_IDE",
        "device": str(device), "quick": quick, "paired_source_states": pairs,
        "records": pairs * len(d4.ARMS), "batched_environment_steps": batched_steps,
        "physical_transitions": batched_steps * len(d4.d3.FAMILIES) * len(d4.d3.PROFILES),
        "config_sha256": _file_sha256(_project_path(path)),
        "entry_sha256": _file_sha256(Path(__file__)),
        "g2_d4_output_hashes": D4_OUTPUT_HASHES,
        "training_checkpoint_manifest_sha256": d4.d3.TRAIN_HASHES["checkpoint_manifest.json"],
        "frozen_source_bundle_sha256": d4.d3.train.FROZEN_SOURCE,
        **cfg["boundary"],
    }
    return cfg, parent, report, output, device


def _gradient(target: torch.Tensor, correction: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if not target.requires_grad:
        return torch.zeros_like(correction), False
    gradient = torch.autograd.grad(target.sum(), correction, retain_graph=True,
                                   allow_unused=True)[0]
    return (torch.zeros_like(correction), False) if gradient is None else (gradient, True)


def _require_finite_tensors(values: dict[str, torch.Tensor], label: str) -> None:
    for name, value in values.items():
        if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"G2-D5 {label}非有限张量: {name}")


def _gradient_branch(state: tuple[R5BatchedEnvironment, R4Interface],
                     policy: ResidualGRUPolicy, *, step: int, hold_steps: int,
                     scale: float, action_weight: float, smooth_weight: float,
                     progress: d4.d3.SparseProgress) -> dict[str, torch.Tensor]:
    env, interface = state
    view = interface.snapshot()
    with torch.no_grad():
        raw = policy(view.features, view.valid)
        baseline = anchor_delta(view.features[:, -1],
                                {"gain": .15, "leak": .10, "tracking_gain": .50})
    correction = (raw * scale).detach().requires_grad_(True)
    previous = view.features[:, -1, 63:74].detach()
    action = interface.issue(baseline, correction, step=step)
    physical, measured = [], []
    for lag in range(hold_steps):
        delta = action.requested_delta_rad if lag == 0 else torch.zeros_like(action.requested_delta_rad)
        _, _, terminated, truncated, info = env.step(delta)
        if bool(truncated.any()) or bool(terminated.any()) != (step + lag + 1 == env.config.episode_length):
            raise RuntimeError("G2-D5 梯度探针回合不完整")
        physical.append(info["reward_power_in_bucket"])
        measured.append(info["measured_power_in_bucket"])
        progress.tick()
    normalized = action.normalized_correction
    action_penalty = action_weight * normalized.square().mean(-1)
    smooth_penalty = smooth_weight * (normalized - previous).square().mean(-1)
    immediate = measured_objective(measured[0], normalized, previous,
                                   action_weight, smooth_weight)
    per_lag = [_gradient(value, correction) for value in physical]
    gradients = torch.stack([value for value, _ in per_lag], dim=1)
    measured_gradient, measured_connected = _gradient(measured[0], correction)
    action_gradient, action_connected = _gradient(action_penalty, correction)
    smooth_gradient, smooth_connected = _gradient(smooth_penalty, correction)
    immediate_gradient, immediate_connected = _gradient(immediate, correction)
    if not torch.allclose(immediate_gradient,
                          measured_gradient - action_gradient - smooth_gradient,
                          rtol=1e-4, atol=1e-7):
        raise RuntimeError("G2-D5 即时训练目标梯度分解不闭合")
    result = {
        "raw": raw.detach(), "correction": correction.detach(),
        "normalized": normalized.detach(), "previous": previous,
        "physical": torch.stack(physical, dim=1).detach(),
        "measured_first": measured[0].detach(),
        "objective_first": immediate.detach(),
        "action_penalty": action_penalty.detach(),
        "smooth_penalty": smooth_penalty.detach(),
        "physical_gradient": gradients.detach(),
        "measured_first_gradient": measured_gradient.detach(),
        "action_penalty_gradient": action_gradient.detach(),
        "smooth_penalty_gradient": smooth_gradient.detach(),
        "objective_first_gradient": immediate_gradient.detach(),
        "requested_delta": action.requested_delta_rad.detach(),
        # 对 power.sum() 求导：仅表示整批计算图连通，不代表每条样本独立连通。
        "physical_gradient_graph_connected_batch_by_lag": [connected for _, connected in per_lag],
        "measured_first_gradient_connected": measured_connected,
        "action_penalty_gradient_connected": action_connected,
        "smooth_penalty_gradient_connected": smooth_connected,
        "objective_first_gradient_connected": immediate_connected,
    }
    _require_finite_tensors(result, "梯度分支")
    return result


@torch.no_grad()
def _hard_branch(state: tuple[R5BatchedEnvironment, R4Interface], raw: torch.Tensor,
                 *, alpha: float, scale: float, step: int, hold_steps: int,
                 action_weight: float, smooth_weight: float,
                 progress: d4.d3.SparseProgress) -> dict[str, torch.Tensor]:
    env, interface = state
    view = interface.snapshot()
    baseline = anchor_delta(view.features[:, -1],
                            {"gain": .15, "leak": .10, "tracking_gain": .50})
    action = interface.issue(baseline, raw * scale * alpha, step=step)
    previous = view.features[:, -1, 63:74]
    physical, measured = [], []
    for lag in range(hold_steps):
        delta = action.requested_delta_rad if lag == 0 else torch.zeros_like(action.requested_delta_rad)
        _, _, terminated, truncated, info = env.step(delta)
        if bool(truncated.any()) or bool(terminated.any()) != (step + lag + 1 == env.config.episode_length):
            raise RuntimeError("G2-D5 硬前向探针回合不完整")
        physical.append(info["reward_power_in_bucket"])
        measured.append(info["measured_power_in_bucket"])
        progress.tick()
    result = {
        "physical": torch.stack(physical, dim=1),
        "objective_first": measured_objective(measured[0], action.normalized_correction,
                                               previous, action_weight, smooth_weight),
    }
    _require_finite_tensors(result, "硬前向分支")
    return result


def _rows(gradient: dict[str, torch.Tensor], minus: dict[str, torch.Tensor],
          plus: dict[str, torch.Tensor], *, condition: str, seed: int, step: int,
          member: int, arm: str, profiles: list, state_hash: str,
          epsilon: float, tolerance: float) -> list[dict]:
    correction = gradient["correction"]
    radial = (gradient["physical_gradient"] * correction[:, None, :]).sum(-1)
    hard = (plus["physical"] - minus["physical"]) / (2 * epsilon)
    objective_radial = (gradient["objective_first_gradient"] * correction).sum(-1)
    action_radial = (gradient["action_penalty_gradient"] * correction).sum(-1)
    smooth_radial = (gradient["smooth_penalty_gradient"] * correction).sum(-1)
    _require_finite_tensors({"radial": radial, "hard": hard,
                             "objective_radial": objective_radial,
                             "action_radial": action_radial,
                             "smooth_radial": smooth_radial}, "径向导数")
    rows = []
    for slot, profile in enumerate(profiles):
        for family_index, family in enumerate(d4.d3.FAMILIES):
            index = slot * len(d4.d3.FAMILIES) + family_index
            autograd_values = radial[index].tolist()
            hard_values = hard[index].tolist()
            for lag in range(len(autograd_values)):
                if lag < profile.slm_delay_frames and abs(autograd_values[lag]) > tolerance:
                    raise RuntimeError("G2-D5 延迟到达前已有动作对物理功率的梯度")
                if lag < profile.slm_delay_frames and abs(hard_values[lag]) > tolerance:
                    raise RuntimeError("G2-D5 延迟到达前硬前向物理功率已变化")
            comparable = [abs(a) > tolerance and abs(b) > tolerance
                          for a, b in zip(autograd_values, hard_values, strict=True)]
            agreement = [(a * b > 0) if good else None
                         for a, b, good in zip(autograd_values, hard_values,
                                               comparable, strict=True)]
            row = {
                "hardware_condition": condition, "weather_seed": seed,
                "probe_step": step, "member": member, "controller": arm,
                "family": family, "slot": slot, "profile": profile.identifier,
                "slm_delay_frames": profile.slm_delay_frames,
                "source_state_sha256": state_hash,
                "deployment_scale": d4.d3.SCALE,
                "hold_rule": "one_policy_action_then_zero_request_increment",
                "gradient_wrt": "unclipped_11d_scaled_correction",
                "raw_correction": gradient["raw"][index].tolist(),
                "scaled_correction": correction[index].tolist(),
                "normalized_correction": gradient["normalized"][index].tolist(),
                "requested_delta_rad": gradient["requested_delta"][index].tolist(),
                "correction_zero": bool(correction[index].abs().max() <= tolerance),
                "correction_clipped_any": bool(correction[index].abs().gt(1).any()),
                "physical_power_by_lag": gradient["physical"][index].tolist(),
                "hard_alpha_minus_power_by_lag": minus["physical"][index].tolist(),
                "hard_alpha_plus_power_by_lag": plus["physical"][index].tolist(),
                "grad_physical_power_by_lag": gradient["physical_gradient"][index].tolist(),
                "physical_gradient_graph_connected_batch_by_lag": gradient["physical_gradient_graph_connected_batch_by_lag"],
                "radial_grad_physical_power_by_lag": autograd_values,
                "hard_central_radial_power_by_lag": hard_values,
                "radial_direction_agrees_by_lag": agreement,
                "radial_comparable_by_lag": comparable,
                "measured_power_first": float(gradient["measured_first"][index]),
                "training_objective_first": float(gradient["objective_first"][index]),
                "hard_alpha_minus_objective_first": float(minus["objective_first"][index]),
                "hard_alpha_plus_objective_first": float(plus["objective_first"][index]),
                "action_penalty": float(gradient["action_penalty"][index]),
                "smooth_penalty": float(gradient["smooth_penalty"][index]),
                "grad_measured_power_first": gradient["measured_first_gradient"][index].tolist(),
                "measured_first_gradient_connected": gradient["measured_first_gradient_connected"],
                "grad_action_penalty": gradient["action_penalty_gradient"][index].tolist(),
                "action_penalty_gradient_connected": gradient["action_penalty_gradient_connected"],
                "grad_smooth_penalty": gradient["smooth_penalty_gradient"][index].tolist(),
                "smooth_penalty_gradient_connected": gradient["smooth_penalty_gradient_connected"],
                "grad_training_objective_first": gradient["objective_first_gradient"][index].tolist(),
                "objective_first_gradient_connected": gradient["objective_first_gradient_connected"],
                "radial_grad_action_penalty": float(action_radial[index]),
                "radial_grad_smooth_penalty": float(smooth_radial[index]),
                "radial_grad_training_objective_first": float(objective_radial[index]),
            }
            for key, value in row.items():
                if isinstance(value, float) and not math.isfinite(value):
                    raise RuntimeError(f"G2-D5 非有限指标: {key}")
            rows.append(row)
    return rows


def summarize(rows: list[dict], cfg: dict, *, quick: bool) -> dict:
    spec = cfg["quick" if quick else "data"]
    keys = {(r["hardware_condition"], r["weather_seed"], r["probe_step"],
             r["member"], r["family"], r["slot"]): {} for r in rows}
    expected = {
        (condition, spec["seed_base"] + spec.get("seed_stride", 10) * weather,
         step, member, family, slot)
        for condition in d4.d3.CONDITIONS
        for weather in range(spec["weather_count"])
        for step in spec["probe_steps"]
        for member in range(spec["initializations"])
        for family in d4.d3.FAMILIES
        for slot in range(len(d4.d3.PROFILES))
    }
    if set(keys) != expected or len(rows) != len(expected) * len(d4.ARMS):
        raise RuntimeError("G2-D5 同状态记录数量或索引不完整")
    for row in rows:
        key = (row["hardware_condition"], row["weather_seed"], row["probe_step"],
               row["member"], row["family"], row["slot"])
        group = keys[key]
        if row["controller"] not in d4.ARMS or row["controller"] in group:
            raise RuntimeError("G2-D5 控制器记录重复")
        group[row["controller"]] = row
    if any(set(group) != set(d4.ARMS)
           or len({r["source_state_sha256"] for r in group.values()}) != 1
           for group in keys.values()):
        raise RuntimeError("G2-D5 同状态配对缺失或哈希不一致")
    cells = {}
    for condition in d4.d3.CONDITIONS:
        selected = [r for r in rows if r["hardware_condition"] == condition]
        strata = {}
        for name, subset in (
            ("all", selected),
            ("train_scale_1", [r for r in selected if r["controller"] == "train_scale_1"]),
            ("train_scale_1_75", [r for r in selected if r["controller"] == "train_scale_1_75"]),
            ("unclipped_nonzero", [r for r in selected
                                   if not r["correction_clipped_any"] and not r["correction_zero"]]),
            ("clipped", [r for r in selected if r["correction_clipped_any"]]),
            ("zero_action", [r for r in selected if r["correction_zero"]]),
        ):
            per_lag = []
            for lag in range(spec["hold_steps"]):
                comparable = [r for r in subset if r["radial_comparable_by_lag"][lag]]
                per_lag.append({
                    "lag": lag,
                    "samples": len(subset),
                    "comparable": len(comparable),
                    "direction_agreement_fraction": (
                        sum(r["radial_direction_agrees_by_lag"][lag] for r in comparable)
                        / len(comparable) if comparable else None),
                    "mean_surrogate_radial": (statistics.fmean(
                        r["radial_grad_physical_power_by_lag"][lag] for r in subset)
                        if subset else None),
                    "mean_hard_central_radial": (statistics.fmean(
                        r["hard_central_radial_power_by_lag"][lag] for r in subset)
                        if subset else None),
                    "surrogate_radial_zero": sum(
                        abs(r["radial_grad_physical_power_by_lag"][lag])
                        <= cfg["gradient_zero_tolerance"] for r in subset),
                    "hard_radial_zero": sum(
                        abs(r["hard_central_radial_power_by_lag"][lag])
                        <= cfg["gradient_zero_tolerance"] for r in subset),
                })
            strata[name] = {"samples": len(subset), "per_lag": per_lag}
        cells[condition] = {
            "records": len(selected), "strata": strata,
            "mean_radial_grad_action_penalty": statistics.fmean(
                r["radial_grad_action_penalty"] for r in selected),
            "mean_radial_grad_smooth_penalty": statistics.fmean(
                r["radial_grad_smooth_penalty"] for r in selected),
            "mean_radial_grad_training_objective_first": statistics.fmean(
                r["radial_grad_training_objective_first"] for r in selected),
        }
    return {
        "status": "DEVELOPMENT_GRADIENT_MECHANISM_NO_GATE",
        "cells": cells,
        "hard_forward_alpha_epsilon": cfg["hard_forward_alpha_epsilon"],
        "gradient_zero_tolerance": cfg["gradient_zero_tolerance"],
        "estimand": "one-action five-frame radial power response at identical frozen integrator states",
        "interpretation_boundary": (
            "STE quantization surrogate gradient is not the mathematical derivative of "
            "the hard forward; this short probe is not the full 200-frame training objective, "
            "closed-loop ranking or independent confirmation"),
    }


def _execute(cfg: dict, parent: dict, report: dict, output: Path,
             device: torch.device, quick: bool) -> dict:
    spec = cfg["quick" if quick else "data"]
    base, _ = load_s1_config(_project_path(parent["environment_config"]))
    base = replace(base, num_modes=21, batch_size=1, episode_length=spec["episode_length"])
    basis, _, _ = build_action_basis(base, ActionRepresentation("r5_zernike21", "zernike", 21), device)
    policies = {}
    training_output = _project_path("outputs/s4_r5_g2_d2_matched_training_v1") / "checkpoints"
    for arm in d4.ARMS:
        for member in range(spec["initializations"]):
            name = f"{arm}_policy_{member}_00512.pt"
            saved = torch.load(training_output / name, map_location=device, weights_only=True)
            if (saved["arm"] != arm or saved["init"] != member or saved["update"] != 512
                    or saved["training_scale"] != dict(d4.d3.train.ARMS)[arm]
                    or saved["deployment_scale"] != d4.d3.SCALE):
                raise RuntimeError("G2-D5 冻结策略身份不符")
            policy = ResidualGRUPolicy(parent["policy"]["hidden_size"],
                                       parent["policy"]["output_size"]).to(device)
            policy.load_state_dict(saved["state_dict"])
            policy.eval().requires_grad_(False)
            policies[arm, member] = policy
    training_cfg = _load_yaml(_project_path(d4.d3.train.CONFIG))
    action_weight = training_cfg["objective"]["action_weight"]
    smooth_weight = training_cfg["objective"]["smooth_weight"]
    nominal, shifted = d4.d3.train.g2.d1._profile_pairs(parent)
    progress = d4.d3.SparseProgress(output, device, 10 if quick else 100)
    progress.phase("G2-D5 CUDA 快速冒烟" if quick else "G2-D5 同状态梯度开发诊断",
                   report["batched_environment_steps"])
    started = time.perf_counter()
    rows = []
    try:
        with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
            for condition, profiles in zip(d4.d3.CONDITIONS, (nominal, shifted), strict=True):
                for seed in stream_manifest(quick)["weather_bases"]:
                    for step in spec["probe_steps"]:
                        for member in range(spec["initializations"]):
                            states = []
                            for _arm in d4.ARMS:
                                for _branch in BRANCHES:
                                    with torch.no_grad():
                                        states.append(d4._replay_integrator(
                                            seed, step, spec["episode_length"], basis, base,
                                            parent["families"], profiles,
                                            parent["data"]["sensor_seed_offset"], progress))
                            state_hash = d4._same_state(states[0], states[1])
                            for state in states[2:]:
                                if d4._same_state(states[0], state) != state_hash:
                                    raise RuntimeError("G2-D5 六路起点随机流或物理状态不一致")
                            for arm_index, arm in enumerate(d4.ARMS):
                                gradient_state, minus_state, plus_state = states[
                                    arm_index * len(BRANCHES):(arm_index + 1) * len(BRANCHES)]
                                gradient = _gradient_branch(
                                    gradient_state, policies[arm, member], step=step,
                                    hold_steps=spec["hold_steps"], scale=d4.d3.SCALE,
                                    action_weight=action_weight, smooth_weight=smooth_weight,
                                    progress=progress)
                                minus = _hard_branch(
                                    minus_state, gradient["raw"],
                                    alpha=1 - cfg["hard_forward_alpha_epsilon"],
                                    scale=d4.d3.SCALE, step=step,
                                    hold_steps=spec["hold_steps"], action_weight=action_weight,
                                    smooth_weight=smooth_weight, progress=progress)
                                plus = _hard_branch(
                                    plus_state, gradient["raw"],
                                    alpha=1 + cfg["hard_forward_alpha_epsilon"],
                                    scale=d4.d3.SCALE, step=step,
                                    hold_steps=spec["hold_steps"], action_weight=action_weight,
                                    smooth_weight=smooth_weight, progress=progress)
                                for row in _rows(
                                        gradient, minus, plus, condition=condition, seed=seed,
                                        step=step, member=member, arm=arm, profiles=profiles,
                                        state_hash=state_hash,
                                        epsilon=cfg["hard_forward_alpha_epsilon"],
                                        tolerance=cfg["gradient_zero_tolerance"]):
                                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                                    rows.append(row)
                            handle.flush()
        if progress.bar.n != report["batched_environment_steps"]:
            raise RuntimeError("G2-D5 实际仿真步数与预注册预算不符")
        analysis = summarize(rows, cfg, quick=quick)
        result = {
            "status": "QUICK_SMOKE_NO_SCIENTIFIC_CONCLUSION" if quick
                      else "DEVELOPMENT_DIAGNOSTIC_REQUIRES_READ_ONLY_AUDIT",
            "records": len(rows), "paired_source_states": report["paired_source_states"],
            "physical_transitions": report["physical_transitions"],
            "analysis": analysis, "elapsed_seconds": time.perf_counter() - started,
            "config_sha256": report["config_sha256"],
            "entry_sha256": report["entry_sha256"],
            "g2_d4_output_hashes": D4_OUTPUT_HASHES,
            "training_checkpoint_manifest_sha256": report["training_checkpoint_manifest_sha256"],
            "frozen_source_bundle_sha256": report["frozen_source_bundle_sha256"],
            "material_passport": {"origin_skill": "academic-research-suite / experiment-agent",
                                  "origin_mode": "run", "verification_status": "UNVERIFIED"},
            **cfg["boundary"],
            "next_action": "停止并等待只读机制审计；不重训、不打开独立确认集或硬件",
        }
        write_json(output / "summary.json", result)
        write_json(output / "SUCCESS.json", {
            "summary_sha256": _file_sha256(output / "summary.json"),
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
    cfg, parent, report, output, device = preflight(path, quick=quick)
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
            "frozen_source_bundle_sha256": d4.d3.train.FROZEN_SOURCE,
        })
        return _execute(cfg, parent, report, output, device, quick)
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
    parser.add_argument("--quick", action="store_true", help="小型 CUDA 技术冒烟，不作科学排名")
    parser.add_argument("--preflight-only", action="store_true", help="只读预检，不创建输出")
    args = parser.parse_args()
    print(json.dumps(run(args.config, quick=args.quick,
                         preflight_only=args.preflight_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
