"""G2-D11-R1：修正请求安全遥测的延迟语义；正式对照由用户在 IDE 启动。"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import diagnose_s4_r5_g2_d4_same_state as d4
from scripts import diagnose_s4_r5_g2_d10_clipping_association as d10
from src.rl.r4_dynamics_experiment import safe_git_record
from src.rl.r4_interface_smoke import write_json
from src.rl.r4_observation import PowerMeasurement, R4Interface
from src.rl.r4_trajectory import anchor_delta
from src.rl.r5_batched_environment import R5BatchedEnvironment
from src.rl.r5_policy_training import ResidualGRUPolicy
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _project_path
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.robust_control import RobustnessCondition

CONFIG = "configs/experiments/s4_r5_g2_d11_same_state_r1.yaml"
FORMAL_BASE, QUICK_BASE = 7_540_000, 7_550_000
CONDITIONS, FAMILIES, PROFILES = d10.d9.CONDITIONS, d10.d9.FAMILIES, d10.d9.PROFILES
CANDIDATES = ("original", "equivalent_clamp", "clipped_inward") + tuple(
    f"coordinate_{i:02d}_{direction}" for i in range(11) for direction in ("minus", "plus"))
HELPERS = {
    "scripts/diagnose_s4_r5_g2_d11_same_state.py": "1f5dcf0211f3c78c6c7a6e6610fc65ea648317f302bcf2a9b685601ee4de0426",
    "configs/experiments/s4_r5_g2_d11_same_state_v1.yaml": "8942bdb570f2944b709d1b1efb96260aaf7a7f45cb16c4567506466baf6b85bd",
    "scripts/diagnose_s4_r5_g2_d4_same_state.py": "74e2ccd1318fbf76cf67d9aa040451e1300bf2bb7369b27f7c81074dc6d6a649",
    "scripts/diagnose_s4_r5_g2_d10_clipping_association.py": "4cb263718237a02472b8e4777ec59883437fe30401f46ae35537a5cc58c6e7ba",
    d10.CONFIG: "3f3447288c6fb8189248b8da4a72ec711af538f94c75e75721a8fb8032a6bb98",
}
D10_HASHES = {
    "summary.json": "71930f20bf8d1a08a8587addb4525db65057ef2cf6bfe77ca672e503bd8d3376",
    "group_table.json": "d2adaacecaf68acbbf543cac33bdd3eccaee7b3fe822f0d19ecfb3433d78e491",
    "progress.jsonl": "ba2efb7f8a189f7eb13b533cf10b80610e31fab0f7f02394453e58855b67b6fb",
}
TRACE_METRICS = {"power": "reward_power_in_bucket", "measured_power": "measured_power_in_bucket",
                 "strehl": "reward_strehl", "phase_rmse": "reward_phase_rmse",
                 "violation": "violation_fraction", "saturation": "saturated_fraction",
                 "slew": "slew_limited_fraction"}

# 饱和是当前请求裁剪；违规是请求裁剪与延迟执行安全的混合量。
# 二者完整记录并参与安全筛选，但不受“执行延迟前不变”的物理断言约束。
DELAYED_METRICS = ("power", "measured_power", "strehl", "phase_rmse", "slew")
DELAYED_INDICES = tuple(list(TRACE_METRICS).index(key) for key in DELAYED_METRICS)
FAILED_V1_HASHES = {
    "config.json": "b19429cfd0c3957ca479ad899a8e8e6df5d3c36cf62dcf89f34e0f003f2947a7",
    "failure.json": "82d9637c7c282719cb310cce6fba98d9779b08c087dbcd233ae395afea9959d5",
    "preflight.json": "236aec9f60edde8f7747afd630df4d676366845fb71399eacfab9586697a3bd5",
    "progress.jsonl": "ab952be7d00464fd8712fc466a9a40a4fc1d69a956714f18f629269ec403e814",
    "records.jsonl": "7e474f2fd8cf50e2d352762fa4f317fb0a15713cf8909b06e2b17dcf938df054",
    "runtime.json": "4929f1873ab47b8d4ac1fb51c282cd0cbb2443647107a73c8a3cd1f0209a13d6",
    "stream_manifest.json": "33015f7dc5f9d286ee2b22adea1a53c111a538da86e75dc1d7f3453c854a7bb9",
}


def _contract(cfg: dict) -> None:
    expected = {
        "stage": "S4-D2-R5-G2-D11-R1",
        "purpose": "frozen_d8_policy_on_policy_same_state_feasible_action_probe",
        "runtime": {"device": "cuda", "formal_owner": "user_ide", "automatic_retry": False},
        "deployment_scale": 1.75, "source_state": "complete_frozen_d8_policy_episode",
        "candidate_rule": "original_equivalent_clamp_boundary_inward_and_22_coordinate_steps",
        "candidate_epsilon": .1,
        "data": {"seed_base": FORMAL_BASE, "seed_stride": 10, "weather_count": 8,
                 "initializations": 3, "episode_length": 200, "probe_steps": [25, 75, 150], "hold_steps": 12},
        "quick": {"seed_base": QUICK_BASE, "seed_stride": 10, "weather_count": 1,
                  "initializations": 1, "episode_length": 16, "probe_steps": [5], "hold_steps": 8},
        "statistics": {"bootstrap_seed": 7_563_456, "bootstrap_repeats": 5000,
                       "descriptive_interval": .95, "cluster": "complete_weather"},
        "tolerances": {"replay": 1e-7, "safety_increase": .001},
        "output_directory": "outputs/s4_r5_g2_d11_same_state_r1",
        "quick_directory": "outputs/s4_r5_g2_d11_same_state_r1_quick",
        "boundary": {"confirmation_access": False, "training_updates": 0,
                     "real_slm_actions": False, "automatic_retry": False, "historical_results_read_only": True,
                     "independent_confirmation": False, "deployable_candidate_selection": False,
                     "gate_reclassification": False},
    }
    if cfg != expected:
        raise ValueError("G2-D11 冻结同状态诊断合同变化")


def stream_manifest(quick: bool) -> dict:
    seeds = [QUICK_BASE] if quick else [FORMAL_BASE + 10 * i for i in range(8)]
    return {"weather_bases": seeds,
            "turbulence": [seed + 1000 * slot + family for seed in seeds for slot in range(6) for family in range(3)],
            "sensor": [seed + 1000 * slot + 50_000_000 for seed in seeds for slot in range(6)],
            "power": [seed + 1000 * slot + 60_000_000 for seed in seeds for slot in range(6)],
            "paired_candidates_share_full_generator_states": True}


def verify_streams() -> None:
    a, b = stream_manifest(False), stream_manifest(True)
    old = (d10.d9.stream_manifest(False), d10.d9.stream_manifest(True),
           d10.d9.d8.stream_manifest(quick=False), d10.d9.d8.stream_manifest(quick=True),
           d10.d9.d3.stream_manifest(False), d10.d9.d3.stream_manifest(True),
           d10.d9.d8.d2.g2.stream_manifest(False), d10.d9.d8.d2.g2.stream_manifest(True))
    for name, offset in (("turbulence", 0), ("sensor", 50_000_000), ("power", 60_000_000)):
        current = set(a[name]) | set(b[name])
        previous = set().union(*(set(item[name]) for item in old))
        if (len(set(a[name])) != len(a[name]) or len(set(b[name])) != len(b[name])
                or set(a[name]) & set(b[name]) or current & previous
                or any(7_300_000 + offset <= s < 7_370_000 + offset for s in current)
                or any(s >= d10.d9.d8.CONFIRMATION_BASE + offset for s in current)):
            raise RuntimeError(f"G2-D11 随机流与历史或确认预留重叠: {name}")


def verify_sources() -> dict:
    for path, digest in HELPERS.items():
        if _file_sha256(_project_path(path)) != digest:
            raise RuntimeError(f"G2-D11 冻结工具变化: {path}")
    if (_file_sha256(_project_path(d10.d9.CONFIG)) != d10.D9_CONFIG_SHA256
            or _file_sha256(Path(d10.d9.__file__)) != d10.D9_ENTRY_SHA256):
        raise RuntimeError("G2-D9 冻结来源变化")
    source_cfg = _load_yaml(_project_path(d10.d9.CONFIG))
    d10.d9._contract(source_cfg)
    _, parent, _ = d10.d9._verify_training(source_cfg)
    for root, hashes in (("outputs/s4_r5_g2_d9_action_penalty_development_v1", d10.D9_HASHES),
                         ("outputs/s4_r5_g2_d10_clipping_association_v1", D10_HASHES)):
        directory = _project_path(root)
        if (directory / "failure.json").exists():
            raise RuntimeError("G2-D11 上游有失败标记")
        for name, digest in hashes.items():
            if _file_sha256(directory / name) != digest:
                raise RuntimeError(f"G2-D11 上游证据变化: {root}/{name}")
    d10_root = _project_path("outputs/s4_r5_g2_d10_clipping_association_v1")
    success = json.loads((d10_root / "SUCCESS.json").read_text(encoding="utf-8"))
    for key, name in (("summary_sha256", "summary.json"), ("group_table_sha256", "group_table.json"),
                      ("progress_sha256", "progress.jsonl")):
        if success.get(key) != D10_HASHES[name]:
            raise RuntimeError("G2-D10 成功证据不符")
    failed_root = _project_path("outputs/s4_r5_g2_d11_same_state_v1")
    for name, digest in FAILED_V1_HASHES.items():
        if _file_sha256(failed_root / name) != digest:
            raise RuntimeError(f"G2-D11-R1 原始失败证据变化: {name}")
    return parent


def budget(spec: dict) -> dict:
    episode_batches = 2 * spec["weather_count"] * spec["initializations"]
    source_states = episode_batches * len(spec["probe_steps"]) * 18
    steps = episode_batches * (spec["episode_length"] + len(spec["probe_steps"]) * len(CANDIDATES) * spec["hold_steps"])
    return {"paired_source_states": source_states, "records": source_states * len(CANDIDATES),
            "batched_environment_steps": steps, "physical_transitions": steps * 18,
            "complete_source_episodes": episode_batches * 18,
            "probe_branch_segments": source_states * len(CANDIDATES),
            "policy_forward_calls": episode_batches * spec["episode_length"],
            "policy_forward_sample_steps": episode_batches * spec["episode_length"] * 18}


def preflight(path: str | Path = CONFIG, *, quick: bool = False) -> tuple[dict, dict, dict, Path, torch.device]:
    cfg = _load_yaml(_project_path(path))
    _contract(cfg)
    parent = verify_sources()
    verify_streams()
    spec = cfg["quick" if quick else "data"]
    nominal, shifted = d10.d9.d8.d2.g2.d1._profile_pairs(parent)
    if parent["data"]["sensor_seed_offset"] != 50_000_000:
        raise RuntimeError("G2-D11 观测随机流偏移变化")
    if ([p.identifier for p in shifted] != list(PROFILES)
            or [f["id"] for f in parent["families"]] != list(FAMILIES)):
        raise RuntimeError("G2-D11 硬件或天气分组变化")
    if (max(p.slm_delay_frames for p in nominal + shifted) >= spec["hold_steps"]
            or max(spec["probe_steps"]) + spec["hold_steps"] > spec["episode_length"]):
        raise RuntimeError("G2-D11 动作响应窗口不足或超过回合")
    output = _project_path(cfg["quick_directory" if quick else "output_directory"])
    if output.exists():
        raise FileExistsError(f"保留已有 G2-D11 输出，禁止覆盖: {output}")
    device = resolve_device("cuda")
    report = {"status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_IDE",
              "quick": quick, "device": str(device), **budget(spec),
              "config_sha256": _file_sha256(_project_path(path)),
              "entry_sha256": _file_sha256(Path(__file__)), "helper_hashes": HELPERS,
              "failed_v1_hashes": FAILED_V1_HASHES,
              "delay_invariant_metrics": list(DELAYED_METRICS) + ["applied_modal"],
              "immediate_request_telemetry": ["saturation", "violation"],
              "repair_scope": "assertion_semantics_only_no_safety_threshold_or_policy_change",
              "d10_hashes": D10_HASHES, "d8_checkpoints": d10.d9.D8_FINAL,
              "frozen_source_bundle_sha256": d10.d9.d8.SOURCE_BUNDLE_SHA256, **cfg["boundary"]}
    return cfg, parent, report, output, device


def candidate_commands(command: torch.Tensor, epsilon: float) -> dict[str, torch.Tensor]:
    if (command.ndim != 2 or command.shape[1] != 11 or not command.is_floating_point()
            or not bool(torch.isfinite(command).all()) or not math.isfinite(epsilon) or not 0 < epsilon < 1):
        raise ValueError("无效的同状态候选动作")
    normalized = command.clamp(-1, 1)
    inward = torch.where(command.abs() > 1, normalized * (1 - epsilon), normalized)
    result = {"original": command.clone(), "equivalent_clamp": normalized, "clipped_inward": inward}
    for index in range(11):
        for sign, label in ((-1, "minus"), (1, "plus")):
            alternative = normalized.clone()
            alternative[:, index] = (alternative[:, index] + sign * epsilon).clamp(-1, 1)
            result[f"coordinate_{index:02d}_{label}"] = alternative
    return result


def fork_state(state: tuple) -> tuple:
    """复制完整环境/接口；CUDA 随机数生成器显式复制，禁止共享可变状态。"""
    env, interface = state
    generators = [g for attr in ("generators", "sensor_generators", "power_generators") for g in getattr(env, attr)]
    generators.append(env.measurement_generator)
    memo = {}
    for generator in generators:
        cloned = torch.Generator(device=generator.device)
        cloned.set_state(generator.get_state().clone())
        memo[id(generator)] = cloned
    cloned_state = copy.deepcopy((env, interface), memo)
    require_same_state(state, cloned_state)
    return cloned_state


def require_same_state(left: tuple, right: tuple) -> str:
    digest = d4._same_state(left, right)
    a, ai = left
    b, bi = right
    if a.step_count != b.step_count or ai._pending is not None or bi._pending is not None:
        raise RuntimeError("G2-D11 时间或待处理动作不一致")
    for av, bv in ((ai.requested, bi.requested), (ai.estimator.current, bi.estimator.current),
                   (ai._correction, bi._correction), (ai._power, bi._power)):
        if not torch.equal(av, bv):
            raise RuntimeError("G2-D11 因果接口状态不一致")
    if ai._power_step != bi._power_step or ai.estimator.next_command_step != bi.estimator.next_command_step:
        raise RuntimeError("G2-D11 时间戳不一致")
    if len(ai.estimator._queue) != len(bi.estimator._queue) or any(
            not torch.equal(x, y) for x, y in zip(ai.estimator._queue, bi.estimator._queue, strict=True)):
        raise RuntimeError("G2-D11 名义执行器队列不一致")
    return digest


def probe_branch(state: tuple, command: torch.Tensor, *, step: int, hold_steps: int, progress) -> dict:
    env, interface = state
    view = interface.snapshot()
    baseline = anchor_delta(view.features[:, -1], {"gain": .15, "leak": .10, "tracking_gain": .50})
    action = interface.issue(baseline, command, step=step)
    traces = {key: [] for key in TRACE_METRICS}
    applied, gaps = [], []
    for lag in range(hold_steps):
        delta = action.requested_delta_rad if lag == 0 else torch.zeros_like(action.requested_delta_rad)
        _, _, term, trunc, info = env.step(delta)
        if bool(trunc.any()) or bool(term.any()) != (step + lag + 1 == env.config.episode_length):
            raise RuntimeError("G2-D11 探针回合时间错位")
        for key, source in TRACE_METRICS.items():
            traces[key].append(info[source])
        applied.append(info["applied_modal"])
        gaps.append((action.requested_modal_rad - info["applied_modal"]).abs().mean(-1))
        progress.tick({"窗口步": float(lag + 1), "平均桶内功率": float(info["reward_power_in_bucket"].mean())})
    result = {key: torch.stack(value, dim=1) for key, value in traces.items()}
    result.update(command=command.clone(), normalized=action.normalized_correction,
                  requested_delta=action.requested_delta_rad, requested_modal=action.requested_modal_rad,
                  applied_modal=torch.stack(applied, dim=1), requested_applied_gap=torch.stack(gaps, dim=1))
    if any(not bool(torch.isfinite(value).all()) for value in result.values()):
        raise RuntimeError("G2-D11 非有限探针结果")
    return result


def probe_rows(results: dict, *, condition: str, seed: int, step: int, member: int,
               profiles: list, state_hash: str, replay_tolerance: float) -> list[dict]:
    original = results["original"]
    for key in original:
        if key != "command" and not torch.equal(original[key], results["equivalent_clamp"][key]):
            raise RuntimeError(f"G2-D11 等价裁剪对照不相同: {key}")
    rows = []
    for name in CANDIDATES:
        value = results[name]
        for slot, profile in enumerate(profiles):
            for fi, family in enumerate(FAMILIES):
                i = slot * 3 + fi
                for key in (*DELAYED_METRICS, "applied_modal"):
                    # 只约束延迟执行后的物理/测量量；当前请求安全遥测仍完整保留。
                    if not torch.allclose(value[key][i, :profile.slm_delay_frames],
                                          original[key][i, :profile.slm_delay_frames], atol=replay_tolerance, rtol=0):
                        raise RuntimeError(f"G2-D11 延迟到达前候选已影响 {key}")
                row = {"hardware_condition": condition, "weather_seed": seed, "probe_step": step,
                       "member": member, "family": family, "slot": slot, "profile": profile.identifier,
                       "candidate": name, "source_state_sha256": state_hash,
                       "slm_delay_frames": profile.slm_delay_frames, "hold_steps": value["power"].shape[1],
                       "deployment_scale": 1.75, "hold_rule": "one_action_then_zero_request_increment",
                       "original_clipped_fraction": float(original["command"][i].abs().gt(1).float().mean())}
                for key, tensor in value.items():
                    row[key] = tensor[i].detach().cpu().tolist()
                delay = profile.slm_delay_frames
                row["post_arrival_power"] = float(value["power"][i, delay:].double().mean())
                for key in ("violation", "saturation", "slew"):
                    row[f"{key}_max"] = float(value[key][i].max())
                rows.append(row)
    return rows


def summarize(rows: list[dict], cfg: dict, *, device: torch.device, quick: bool = False) -> dict:
    spec = cfg["quick" if quick else "data"]
    expected = {(condition, seed, step, member, family, slot)
                for condition in CONDITIONS for seed in stream_manifest(quick)["weather_bases"]
                for step in spec["probe_steps"] for member in range(spec["initializations"])
                for family in FAMILIES for slot in range(6)}
    if len(rows) != len(expected) * len(CANDIDATES):
        raise RuntimeError("G2-D11 记录数量不完整")
    try:
        traces = torch.tensor([[r[k] for k in TRACE_METRICS] for r in rows], dtype=torch.float64, device=device)
        delays = torch.tensor([r["slm_delay_frames"] for r in rows], device=device)
        command = torch.tensor([r["command"] for r in rows], dtype=torch.float64, device=device)
        normalized = torch.tensor([r["normalized"] for r in rows], dtype=torch.float64, device=device)
        requested = torch.tensor([r["requested_delta"] for r in rows], dtype=torch.float64, device=device)
        modal = torch.tensor([r["requested_modal"] for r in rows], dtype=torch.float64, device=device)
        applied = torch.tensor([r["applied_modal"] for r in rows], dtype=torch.float64, device=device)
    except (ValueError, TypeError, KeyError) as exc:
        raise RuntimeError("G2-D11 轨迹字段或形状无效") from exc
    if (traces.shape != (len(rows), len(TRACE_METRICS), spec["hold_steps"])
            or command.shape != (len(rows), 11) or normalized.shape != command.shape
            or requested.shape != (len(rows), 21) or modal.shape != requested.shape
            or applied.shape != (len(rows), spec["hold_steps"], 21)
            or any(not bool(torch.isfinite(t).all()) for t in (traces, command, normalized, requested, modal, applied))
            or bool(((delays < 0) | (delays >= spec["hold_steps"])).any())
            or not torch.equal(normalized, command.clamp(-1, 1))
            or bool((requested.abs() > .15 + 1e-6).any()) or bool((modal.abs() > 3 + 1e-6).any())
            or bool((requested.norm(dim=-1) > math.sqrt(10) * .15 + 1e-6).any())
            or bool((modal.norm(dim=-1) > math.sqrt(10) * 3 + 1e-6).any())):
        raise RuntimeError("G2-D11 轨迹、归一化或动作安全范围不符")
    arrival = torch.arange(spec["hold_steps"], device=device)[None, :] >= delays[:, None]
    power_means = (traces[:, 0] * arrival).sum(-1) / arrival.sum(-1)
    saved_means = torch.tensor([r["post_arrival_power"] for r in rows], dtype=torch.float64, device=device)
    maxima = torch.tensor([[r[f"{k}_max"] for k in ("violation", "saturation", "slew")] for r in rows], dtype=torch.float64, device=device)
    if (not bool(torch.isfinite(saved_means).all() & torch.isfinite(maxima).all())
            or not torch.allclose(power_means, saved_means, atol=1e-7, rtol=0)
            or not torch.allclose(traces[:, 4:7].amax(-1), maxima, atol=1e-7, rtol=0)
            or bool(((traces[:, 4:7] < 0) | (traces[:, 4:7] > 1)).any())):
        raise RuntimeError("G2-D11 延迟均值或安全轨迹不符")
    indices = {id(row): i for i, row in enumerate(rows)}
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("hardware_condition", "weather_seed", "probe_step", "member", "family", "slot"))
        candidate = row["candidate"]
        if key not in expected:
            raise RuntimeError("G2-D11 非预定状态、天气或槽位")
        if candidate not in CANDIDATES or candidate in groups.setdefault(key, {}):
            raise RuntimeError("G2-D11 候选重复或未知")
        expected_profile = f"nominal_for_{PROFILES[row['slot']]}" if row["hardware_condition"] == "nominal_clone" else PROFILES[row["slot"]]
        if (row["profile"] != expected_profile or row["hold_steps"] != spec["hold_steps"]
                or row["deployment_scale"] != 1.75 or row["hold_rule"] != "one_action_then_zero_request_increment"
                or not isinstance(row["source_state_sha256"], str) or len(row["source_state_sha256"]) != 64
                or not math.isfinite(row["original_clipped_fraction"]) or not 0 <= row["original_clipped_fraction"] <= 1):
            raise RuntimeError("G2-D11 来源身份、窗口或档位不符")
        groups[key][candidate] = row
    if set(groups) != expected or any(set(g) != set(CANDIDATES) for g in groups.values()):
        raise RuntimeError("G2-D11 同状态配对缺失")
    descriptors = []
    for key, variants in groups.items():
        source = variants["original"]
        if any((r["source_state_sha256"], r["profile"], r["slm_delay_frames"]) !=
               (source["source_state_sha256"], source["profile"], source["slm_delay_frames"]) for r in variants.values()):
            raise RuntimeError("G2-D11 配对来源哈希或档位不一致")
        if abs(variants["equivalent_clamp"]["post_arrival_power"] - source["post_arrival_power"]) > cfg["tolerances"]["replay"]:
            raise RuntimeError("G2-D11 等价对照功率不一致")
        original_index = indices[id(source)]
        eq_index = indices[id(variants["equivalent_clamp"])]
        expected_commands = candidate_commands(command[original_index:original_index + 1], cfg["candidate_epsilon"])
        clipping = float(command[original_index].abs().gt(1).double().mean())
        if (not torch.equal(traces[original_index], traces[eq_index])
                or not torch.equal(requested[original_index], requested[eq_index])
                or not torch.equal(modal[original_index], modal[eq_index])
                or not torch.equal(applied[original_index], applied[eq_index])):
            raise RuntimeError("G2-D11 等价对照轨迹不相同")
        for variant in variants.values():
            index = indices[id(variant)]
            delay = source["slm_delay_frames"]
            if (not torch.allclose(command[index], expected_commands[variant["candidate"]][0], atol=1e-7, rtol=0)
                    or abs(variant["original_clipped_fraction"] - clipping) > 1e-7):
                raise RuntimeError("G2-D11 候选动作或原裁剪计数不符")
            if (not torch.allclose(traces[index, DELAYED_INDICES, :delay],
                                   traces[original_index, DELAYED_INDICES, :delay],
                                   atol=cfg["tolerances"]["replay"], rtol=0)
                    or not torch.allclose(applied[index, :delay], applied[original_index, :delay],
                                          atol=cfg["tolerances"]["replay"], rtol=0)):
                raise RuntimeError("G2-D11 延迟到达前已有候选效应")
        safe = [r for r in variants.values() if all(r[f"{k}_max"] <= source[f"{k}_max"] + cfg["tolerances"]["safety_increase"]
                                                   for k in ("violation", "saturation", "slew"))]
        row = dict(zip(("condition", "weather", "step", "member", "family", "slot"), key, strict=True))
        row.update(clipping=source["original_clipped_fraction"],
                   inward=variants["clipped_inward"]["post_arrival_power"] - source["post_arrival_power"],
                   oracle=max(r["post_arrival_power"] for r in variants.values()) - source["post_arrival_power"],
                   safe_oracle=max(r["post_arrival_power"] for r in safe) - source["post_arrival_power"],
                   inward_safety_pass=variants["clipped_inward"] in safe)
        descriptors.append(row)
    cells = {}
    for condition in CONDITIONS:
        selected = [r for r in descriptors if r["condition"] == condition]
        weather = stream_manifest(quick)["weather_bases"]
        means = torch.tensor([[sum(r[k] for r in selected if r["weather"] == seed) /
                               sum(r["weather"] == seed for r in selected)
                               for k in ("inward", "oracle", "safe_oracle")] for seed in weather],
                             dtype=torch.float64, device=device)
        generator = torch.Generator(device=device).manual_seed(cfg["statistics"]["bootstrap_seed"])
        draws = torch.randint(len(weather), (cfg["statistics"]["bootstrap_repeats"], len(weather)), generator=generator, device=device)
        inward_ci = torch.quantile(means[draws, 0].mean(1), means.new_tensor([.025, .975])).tolist()
        cells[condition] = {"paired_states": len(selected), "weather_clusters": len(weather),
                            "fixed_clipped_inward_power_delta": float(means[:, 0].mean()),
                            "fixed_clipped_inward_descriptive_ci95": inward_ci,
                            "inward_safety_pass_fraction": sum(r["inward_safety_pass"] for r in selected) / len(selected),
                            "oracle_candidate_gap": float(means[:, 1].mean()),
                            "safety_filtered_oracle_gap": float(means[:, 2].mean()),
                            "clipped_state_fraction": sum(r["clipping"] > 0 for r in selected) / len(selected)}
    table = []
    for condition in CONDITIONS:
        for family in FAMILIES:
            for slot in range(6):
                part = [r for r in descriptors if (r["condition"], r["family"], r["slot"]) == (condition, family, slot)]
                table.append({"hardware_condition": condition, "family": family, "slot": slot, "states": len(part),
                              **{f"mean_{k}": sum(r[k] for r in part) / len(part) for k in ("clipping", "inward", "oracle", "safe_oracle")}})
    return {"status": "DEVELOPMENT_SAME_STATE_MECHANISM_NO_PERFORMANCE_GATE", "cells": cells, "group_table": table,
            "oracle_scope": "truth_selected_local_candidate_set_only_optimistically_biased_not_deployable_not_global_capacity",
            "scope": "one_impulse_then_fixed_request_not_complete_closed_loop_increment",
            "interval_scope": "descriptive_only_no_new_gate", "prior_d9_gate_reclassified": False}


@torch.no_grad()
def execute(cfg: dict, parent: dict, report: dict, output: Path, device: torch.device, quick: bool) -> dict:
    spec = cfg["quick" if quick else "data"]
    base, _ = load_s1_config(_project_path(parent["environment_config"]))
    base = replace(base, num_modes=21, batch_size=1, episode_length=spec["episode_length"])
    basis, _, _ = build_action_basis(base, ActionRepresentation("r5_zernike21", "zernike", 21), device)
    policies = []
    for member in range(spec["initializations"]):
        name = f"action_penalty_0_policy_{member}_00512.pt"
        checkpoint = _project_path("outputs/s4_r5_g2_d8_action_penalty_zero_v1/checkpoints") / name
        if _file_sha256(checkpoint) != d10.d9.D8_FINAL[name]:
            raise RuntimeError("G2-D11 末次权重变化")
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        if (saved["arm"], saved["init"], saved["update"], saved["deployment_scale"]) != ("action_penalty_0", member, 512, 1.75):
            raise RuntimeError("G2-D11 策略身份不符")
        policy = ResidualGRUPolicy(parent["policy"]["hidden_size"], parent["policy"]["output_size"]).to(device)
        policy.load_state_dict(saved["state_dict"])
        policies.append(policy.eval().requires_grad_(False))
    nominal, shifted = d10.d9.d8.d2.g2.d1._profile_pairs(parent)
    progress = d10.d9.d3.SparseProgress(output, device, 10 if quick else 200)
    progress.phase("G2-D11 CUDA 技术冒烟" if quick else "G2-D11 同状态动作开发诊断", report["batched_environment_steps"])
    rows = []
    completed = 0
    started = time.perf_counter()
    try:
        with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
            for condition, profiles in zip(CONDITIONS, (nominal, shifted), strict=True):
                for seed in stream_manifest(quick)["weather_bases"]:
                    for member, policy in enumerate(policies):
                        first = RobustnessCondition.from_mapping(dict(parent["families"][0], base_seed=seed))
                        env = R5BatchedEnvironment(replace(first.environment_config(base), episode_length=spec["episode_length"]),
                                                   device, basis, parent["families"], profiles, parent["data"]["sensor_seed_offset"])
                        raw, _ = env.reset(seed=seed)
                        interface = R4Interface()
                        interface.reset(env.proxy(raw), episode_id=f"g2-d11-{condition}-{seed}-{member}")
                        for step in range(spec["episode_length"]):
                            view = interface.snapshot()
                            command = policy(view.features, view.valid) * 1.75
                            if step in spec["probe_steps"]:
                                state_hash = require_same_state((env, interface), (env, interface))
                                variants = {}
                                for name, candidate in candidate_commands(command, cfg["candidate_epsilon"]).items():
                                    branch = fork_state((env, interface))
                                    variants[name] = probe_branch(branch, candidate, step=step, hold_steps=spec["hold_steps"], progress=progress)
                                    del branch
                                group_rows = probe_rows(variants, condition=condition, seed=seed, step=step, member=member,
                                                        profiles=profiles, state_hash=state_hash, replay_tolerance=cfg["tolerances"]["replay"])
                                for row in group_rows:
                                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                                rows.extend(group_rows)
                                handle.flush()
                                del variants
                            action = interface.issue(anchor_delta(view.features[:, -1], {"gain": .15, "leak": .10, "tracking_gain": .50}), command, step=step)
                            raw, _, term, trunc, info = env.step(action.requested_delta_rad)
                            interface.observe_next(env.proxy(raw), step=step + 1, power=PowerMeasurement(info["measured_power_in_bucket"], step, step + 1))
                            if bool(trunc.any()) or bool(term.all()) != (step + 1 == spec["episode_length"]):
                                raise RuntimeError("G2-D11 来源回合不完整")
                            progress.tick({"来源回合步": float(step + 1), "平均桶内功率": float(info["reward_power_in_bucket"].mean())})
                        completed += 18
        if (len(rows) != report["records"] or progress.bar.n != report["batched_environment_steps"]
                or completed != report["complete_source_episodes"]):
            raise RuntimeError("G2-D11 实际预算不符")
        # 冒烟也检查全网格与统计 CUDA 路径，但不保留其科学分析值。
        checked = summarize(rows, cfg, device=device, quick=quick)
        analysis = {} if quick else checked
        result = {**report, "status": "QUICK_SMOKE_NO_SCIENTIFIC_CONCLUSION" if quick else "DEVELOPMENT_DIAGNOSTIC_REQUIRES_READ_ONLY_AUDIT",
                  "completed_source_episodes": completed, "analysis": analysis,
                  "elapsed_seconds": time.perf_counter() - started,
                  "material_passport": {"origin_skill": "academic-research-suite / experiment-agent", "origin_mode": "run",
                                        "origin_date": datetime.now(timezone.utc).date().isoformat(), "verification_status": "UNVERIFIED",
                                        "version_label": "g2_d11_same_state_r1"},
                  "next_action": "停止等待只读机制审计；不凭真值最佳候选宣称可部署算法达标"}
        write_json(output / "summary.json", result)
        write_json(output / "SUCCESS.json", {f"{name}_sha256": _file_sha256(output / filename)
                                             for name, filename in (("summary", "summary.json"), ("records", "records.jsonl"),
                                                                    ("progress", "progress.jsonl"), ("stream_manifest", "stream_manifest.json"))})
        return result
    finally:
        progress.close()


def run(path: str | Path = CONFIG, *, quick: bool = False, preflight_only: bool = False) -> dict:
    cfg, parent, report, output, device = preflight(path, quick=quick)
    if preflight_only:
        return report
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    created = False
    try:
        output.mkdir(parents=True, exist_ok=False)
        created = True
        write_json(output / "preflight.json", report)
        write_json(output / "config.json", cfg)
        write_json(output / "stream_manifest.json", stream_manifest(quick))
        write_json(output / "runtime.json", {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                                             "gpu": torch.cuda.get_device_name(device), "git": safe_git_record(),
                                             "deterministic_algorithms": True, "allow_tf32": False})
        return execute(cfg, parent, report, output, device, quick)
    except Exception:
        if created:
            write_json(output / "failure.json", {"traceback": traceback.format_exc(), "automatic_retry": False})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, quick=args.quick, preflight_only=args.preflight_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
