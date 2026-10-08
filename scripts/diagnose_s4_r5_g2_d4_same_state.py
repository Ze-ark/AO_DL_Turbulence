"""G2-D4：冻结策略的同状态动作与延迟功率诊断；正式运行由用户在 IDE 启动。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
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

from scripts import diagnose_s4_r5_g2_d3_matched_development as d3
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


CONFIG = "configs/experiments/s4_r5_g2_d4_same_state_v1.yaml"
D3_CONFIG_SHA256 = "635874d1a6c698275375ee41c30fbb389daf2ca8aa207230ba8c1ac9087f96d7"
ARMS = ("train_scale_1", "train_scale_1_75")
FORMAL_BASE, QUICK_BASE = 7_300_000, 7_310_000
METRICS = (
    "raw_abs", "scaled_abs", "normalized_abs", "clipped_fraction",
    "action_penalty", "smooth_penalty", "measured_power_first",
    "physical_power_first", "physical_power_last", "training_objective_first",
    "requested_applied_gap_first", "violation_max", "saturation_max", "slew_max",
)


def _contract(cfg: dict) -> None:
    expected = {
        "stage": "S4-D2-R5-G2-D4",
        "purpose": "frozen_policy_same_state_action_and_delayed_power_diagnostic",
        "runtime": {"device": "cuda", "formal_owner": "user_ide", "automatic_retry": False},
        "g2_d3_config": d3.CONFIG, "g2_d3_config_sha256": D3_CONFIG_SHA256,
        "controllers": list(ARMS), "deployment_scale": d3.SCALE,
        "source_state": "frozen_integrator_prefix",
        "data": {"seed_base": FORMAL_BASE, "seed_stride": 10, "weather_count": 4,
                 "episode_length": 200, "probe_steps": [0, 25, 75, 150], "hold_steps": 5},
        "quick": {"seed_base": QUICK_BASE, "weather_count": 1,
                  "episode_length": 16, "probe_steps": [0, 5], "hold_steps": 5},
        "output_directory": "outputs/s4_r5_g2_d4_same_state_v1",
        "quick_directory": "outputs/s4_r5_g2_d4_same_state_v1_quick_r1",
        "boundary": {"confirmation_access": False, "training_updates": 0,
                     "real_slm_actions": False, "automatic_retry": False,
                     "historical_results_read_only": True, "independent_confirmation": False},
    }
    if cfg != expected:
        raise ValueError("G2-D4 冻结诊断合同已改变")


def stream_manifest(quick: bool) -> dict[str, list[int] | bool]:
    base, count = (QUICK_BASE, 1) if quick else (FORMAL_BASE, 4)
    weather = [base + 10 * index for index in range(count)]
    return {
        "weather_bases": weather,
        "turbulence": [seed + 1000 * slot + family for seed in weather
                       for slot in range(6) for family in range(3)],
        "sensor": [seed + 1000 * slot + 50_000_000 for seed in weather for slot in range(6)],
        "power": [seed + 1000 * slot + 60_000_000 for seed in weather for slot in range(6)],
        "shared_between_arms": True,
    }


def preflight(path: str | Path = CONFIG, *, quick: bool = False
              ) -> tuple[dict, dict, dict, dict, Path, torch.device]:
    cfg = _load_yaml(_project_path(path))
    _contract(cfg)
    if _file_sha256(_project_path(cfg["g2_d3_config"])) != D3_CONFIG_SHA256:
        raise RuntimeError("G2-D3 冻结配置变化")
    d3_cfg = _load_yaml(_project_path(cfg["g2_d3_config"]))
    d3._contract(d3_cfg)
    _, parent = d3._verify_lineage(d3_cfg)
    nominal, shifted = d3.train.g2.d1._profile_pairs(parent)
    if ([item.identifier for item in shifted] != list(d3.PROFILES)
            or [item.identifier for item in nominal] != [f"nominal_for_{name}" for name in d3.PROFILES]):
        raise RuntimeError("G2-D4 同槽硬件档位不符")
    current = (stream_manifest(False), stream_manifest(True))
    historical = (d3.stream_manifest(False), d3.stream_manifest(True),
                  d3.train.stream_manifest(quick=False), d3.train.stream_manifest(quick=True),
                  d3.train.g2.stream_manifest(False), d3.train.g2.stream_manifest(True))
    for name in ("turbulence", "sensor", "power"):
        new = [set(item[name]) for item in current]
        old = set().union(*(set(item[name]) for item in historical))
        if (len(new[0]) != len(current[0][name]) or len(new[1]) != len(current[1][name])
                or new[0] & new[1] or (new[0] | new[1]) & old):
            raise RuntimeError(f"G2-D4 {name} 随机流碰撞")
    spec = cfg["quick" if quick else "data"]
    if any(step + spec["hold_steps"] > spec["episode_length"] for step in spec["probe_steps"]):
        raise RuntimeError("探针超过完整回合")
    output = _project_path(cfg["quick_directory" if quick else "output_directory"])
    if output.exists():
        raise FileExistsError(f"保留已有 G2-D4 输出，不覆盖或重跑: {output}")
    device = resolve_device("cuda")
    pairs = (len(d3.CONDITIONS) * spec["weather_count"] * len(spec["probe_steps"])
             * 3 * len(d3.FAMILIES) * len(d3.PROFILES))
    batched_steps = (len(ARMS) * len(d3.CONDITIONS) * spec["weather_count"] * 3
                     * sum(step + spec["hold_steps"] for step in spec["probe_steps"]))
    report = {
        "status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_IDE",
        "quick": quick, "device": str(device), "paired_source_states": pairs,
        "records": pairs * len(ARMS), "batched_environment_steps": batched_steps,
        "physical_transitions": batched_steps * len(d3.FAMILIES) * len(d3.PROFILES),
        "config_sha256": _file_sha256(_project_path(path)),
        "entry_sha256": _file_sha256(Path(__file__)),
        "training_checkpoint_manifest_sha256": d3.TRAIN_HASHES["checkpoint_manifest.json"],
        "frozen_source_bundle_sha256": d3.train.FROZEN_SOURCE,
        **cfg["boundary"],
    }
    return cfg, d3_cfg, parent, report, output, device


def _replay_integrator(seed: int, steps: int, horizon: int, basis: torch.Tensor,
                       base, families: list[dict], profiles: list,
                       sensor_offset: int, progress: d3.SparseProgress
                       ) -> tuple[R5BatchedEnvironment, R4Interface]:
    condition = RobustnessCondition.from_mapping(dict(families[0], base_seed=seed))
    env_cfg = replace(condition.environment_config(base),
                      batch_size=len(families) * len(profiles), episode_length=horizon)
    env = R5BatchedEnvironment(env_cfg, basis.device, basis, families, profiles, sensor_offset)
    raw, _ = env.reset(seed=seed)
    interface = R4Interface()
    interface.reset(env.proxy(raw), episode_id=f"g2-d4-integrator-{seed}")
    for step in range(steps):
        view = interface.snapshot()
        baseline = anchor_delta(view.features[:, -1],
                                {"gain": .15, "leak": .10, "tracking_gain": .50})
        correction = view.features.new_zeros((len(view.features), 11))
        action = interface.issue(baseline, correction, step=step)
        raw, _, terminated, truncated, info = env.step(action.requested_delta_rad)
        interface.observe_next(env.proxy(raw), step=step + 1,
                               power=PowerMeasurement(info["measured_power_in_bucket"],
                                                      step, step + 1))
        if bool(terminated.any()) or bool(truncated.any()):
            raise RuntimeError("积分器前缀意外终止")
        progress.tick()
    return env, interface


def _same_state(left: tuple[R5BatchedEnvironment, R4Interface],
                right: tuple[R5BatchedEnvironment, R4Interface]) -> str:
    a, ai = left
    b, bi = right
    av, bv = ai.snapshot(), bi.snapshot()
    if (av.observation_step != bv.observation_step
            or not torch.equal(av.features, bv.features)
            or not torch.equal(av.valid, bv.valid)
            or not torch.equal(a.turbulence_phase, b.turbulence_phase)
            or not torch.equal(a.requested_modal, b.requested_modal)
            or not torch.equal(a.slm.state.phase, b.slm.state.phase)
            or not torch.equal(a.slm.state.queue, b.slm.state.queue)):
        raise RuntimeError("G2-D4 两臂的观测或仿真物理状态不同")
    for attr in ("generators", "sensor_generators", "power_generators"):
        x, y = getattr(a, attr), getattr(b, attr)
        if len(x) != len(y) or any(not torch.equal(g.get_state(), h.get_state())
                                   for g, h in zip(x, y, strict=True)):
            raise RuntimeError(f"G2-D4 两臂的 {attr} 随机流状态不同")
    if not torch.equal(a.measurement_generator.get_state(),
                       b.measurement_generator.get_state()):
        raise RuntimeError("G2-D4 两臂的功率随机流状态不同")
    digest = hashlib.sha256()
    for tensor in (av.features, av.valid, a.requested_modal, a.slm.state.phase):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _branch(state: tuple[R5BatchedEnvironment, R4Interface], policy: ResidualGRUPolicy,
            *, scale: float, step: int, hold_steps: int,
            action_weight: float, smooth_weight: float,
            progress: d3.SparseProgress) -> dict[str, torch.Tensor]:
    env, interface = state
    view = interface.snapshot()
    raw = policy(view.features, view.valid)
    command = raw * scale
    baseline = anchor_delta(view.features[:, -1],
                            {"gain": .15, "leak": .10, "tracking_gain": .50})
    action = interface.issue(baseline, command, step=step)
    previous = view.features[:, -1, 63:74]
    physics, measurement, applied, violation, saturation, slew = [], [], [], [], [], []
    for lag in range(hold_steps):
        delta = action.requested_delta_rad if lag == 0 else torch.zeros_like(action.requested_delta_rad)
        _, _, terminated, truncated, info = env.step(delta)
        if bool(truncated.any()) or bool(terminated.any()) != (step + lag + 1 == env.config.episode_length):
            raise RuntimeError("G2-D4 探针回合不完整")
        physics.append(info["reward_power_in_bucket"])
        measurement.append(info["measured_power_in_bucket"])
        applied.append(info["applied_modal"])
        violation.append(info["violation_fraction"])
        saturation.append(info["saturated_fraction"])
        slew.append(info["slew_limited_fraction"])
        progress.tick()
    normalized = action.normalized_correction
    action_penalty = action_weight * normalized.square().mean(-1)
    smooth_penalty = smooth_weight * (normalized - previous).square().mean(-1)
    objective = measured_objective(measurement[0], normalized, previous,
                                   action_weight, smooth_weight)
    if not torch.allclose(objective, measurement[0] - action_penalty - smooth_penalty,
                          rtol=0, atol=1e-7):
        raise RuntimeError("训练目标分解不闭合")
    return {
        "raw": raw, "scaled": command, "normalized": normalized, "previous": previous,
        "requested_delta": action.requested_delta_rad,
        "requested_modal": action.requested_modal_rad,
        "applied_first": applied[0], "applied_last": applied[-1],
        "raw_abs": raw.abs().mean(-1), "scaled_abs": command.abs().mean(-1),
        "normalized_abs": normalized.abs().mean(-1),
        "clipped_fraction": command.abs().gt(1).float().mean(-1),
        "action_penalty": action_penalty, "smooth_penalty": smooth_penalty,
        "measured_power_first": measurement[0], "physical_power_first": physics[0],
        "physical_power_last": physics[-1], "training_objective_first": objective,
        "requested_applied_gap_first": (action.requested_modal_rad - applied[0]).abs().mean(-1),
        "violation_max": torch.stack(violation).amax(0),
        "saturation_max": torch.stack(saturation).amax(0),
        "slew_max": torch.stack(slew).amax(0),
        "physical_power_by_lag": torch.stack(physics, dim=1),
        "measured_power_by_lag": torch.stack(measurement, dim=1),
    }


def _rows(result: dict[str, torch.Tensor], *, condition: str, seed: int, step: int,
          member: int, arm: str, profiles: list, state_hash: str) -> list[dict]:
    matrix_names = ("raw", "scaled", "normalized", "previous", "requested_delta",
                    "requested_modal", "applied_first", "applied_last",
                    "physical_power_by_lag", "measured_power_by_lag")
    rows = []
    for slot, profile in enumerate(profiles):
        for family_index, family in enumerate(d3.FAMILIES):
            index = slot * len(d3.FAMILIES) + family_index
            row = {"hardware_condition": condition, "weather_seed": seed,
                   "probe_step": step, "member": member, "controller": arm,
                   "family": family, "slot": slot, "profile": profile.identifier,
                   "source_state_sha256": state_hash, "deployment_scale": d3.SCALE,
                   "hold_rule": "one_policy_action_then_zero_request_increment"}
            for name in METRICS:
                row[name] = float(result[name][index].item())
                if not math.isfinite(row[name]):
                    raise RuntimeError(f"G2-D4 非有限指标: {name}")
            for name in matrix_names:
                row[name] = result[name][index].detach().cpu().tolist()
            rows.append(row)
    return rows


def summarize(rows: list[dict], cfg: dict) -> dict:
    expected = (len(d3.CONDITIONS) * cfg["data"]["weather_count"]
                * len(cfg["data"]["probe_steps"]) * 3 * len(d3.FAMILIES)
                * len(d3.PROFILES))
    if len(rows) != expected * len(ARMS):
        raise RuntimeError("G2-D4 逐探针记录数量不完整")
    spec = cfg["data"]
    expected_keys = {
        (condition, spec["seed_base"] + spec.get("seed_stride", 10) * weather, step,
         member, family, slot)
        for condition in d3.CONDITIONS
        for weather in range(spec["weather_count"])
        for step in spec["probe_steps"]
        for member in range(3)
        for family in d3.FAMILIES
        for slot in range(len(d3.PROFILES))
    }
    pairs: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        key = (row["hardware_condition"], row["weather_seed"], row["probe_step"],
               row["member"], row["family"], row["slot"])
        arm = row["controller"]
        if arm not in ARMS or arm in pairs.setdefault(key, {}):
            raise RuntimeError("G2-D4 控制器重复或不在合同中")
        pairs[key][arm] = row
    if set(pairs) != expected_keys or any(set(pair) != set(ARMS) for pair in pairs.values()):
        raise RuntimeError("G2-D4 同状态配对缺失")
    differences: dict[str, dict[str, list[float]]] = {
        condition: {metric: [] for metric in METRICS} for condition in d3.CONDITIONS}
    shrink: dict[str, int] = {condition: 0 for condition in d3.CONDITIONS}
    disagreement: dict[str, int] = {condition: 0 for condition in d3.CONDITIONS}
    count: dict[str, int] = {condition: 0 for condition in d3.CONDITIONS}
    for key, pair in pairs.items():
        condition = key[0]
        a, b = pair[ARMS[0]], pair[ARMS[1]]
        if a["source_state_sha256"] != b["source_state_sha256"]:
            raise RuntimeError("G2-D4 配对观测哈希不同")
        count[condition] += 1
        shrink[condition] += b["raw_abs"] < a["raw_abs"]
        objective_delta = b["training_objective_first"] - a["training_objective_first"]
        delayed_delta = b["physical_power_last"] - a["physical_power_last"]
        disagreement[condition] += objective_delta * delayed_delta < 0
        for metric in METRICS:
            differences[condition][metric].append(b[metric] - a[metric])
    return {
        "status": "DEVELOPMENT_MECHANISM_DIAGNOSTIC_NO_GATE",
        "comparison": "train_scale_1_75 minus train_scale_1 on identical integrator-source states",
        "cells": {condition: {
            "paired_states": count[condition],
            "matched_raw_output_smaller_fraction": shrink[condition] / count[condition],
            "first_objective_vs_delayed_power_opposite_fraction": disagreement[condition] / count[condition],
            "mean_differences": {metric: statistics.fmean(values)
                                 for metric, values in differences[condition].items()},
        } for condition in d3.CONDITIONS},
        "interpretation_boundary": "one action then fixed request; not closed-loop ranking or independent confirmation",
    }


@torch.no_grad()
def _execute(cfg: dict, parent: dict, report: dict, output: Path,
             device: torch.device, quick: bool) -> dict:
    spec = cfg["quick" if quick else "data"]
    base, _ = load_s1_config(_project_path(parent["environment_config"]))
    base = replace(base, num_modes=21, batch_size=1, episode_length=spec["episode_length"])
    basis, _, _ = build_action_basis(base, ActionRepresentation("r5_zernike21", "zernike", 21), device)
    policies = {}
    for arm in ARMS:
        for member in range(3):
            name = f"{arm}_policy_{member}_00512.pt"
            saved = torch.load(_project_path("outputs/s4_r5_g2_d2_matched_training_v1")
                               / "checkpoints" / name, map_location=device, weights_only=True)
            training_scale = dict(d3.train.ARMS)[arm]
            if (saved["arm"] != arm or saved["init"] != member or saved["update"] != 512
                    or saved["training_scale"] != training_scale
                    or saved["deployment_scale"] != d3.SCALE):
                raise RuntimeError("G2-D4 冻结检查点身份不符")
            policy = ResidualGRUPolicy(parent["policy"]["hidden_size"],
                                       parent["policy"]["output_size"]).to(device)
            policy.load_state_dict(saved["state_dict"])
            policy.eval()
            policies[arm, member] = policy
    training_cfg = _load_yaml(_project_path(d3.train.CONFIG))
    action_weight = training_cfg["objective"]["action_weight"]
    smooth_weight = training_cfg["objective"]["smooth_weight"]
    nominal, shifted = d3.train.g2.d1._profile_pairs(parent)
    progress = d3.SparseProgress(output, device, 10 if quick else 100)
    progress.phase("G2-D4 CUDA 快速冒烟" if quick else "G2-D4 同状态开发诊断",
                   report["batched_environment_steps"])
    rows: list[dict] = []
    started = time.perf_counter()
    try:
        with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
            for condition, profiles in zip(d3.CONDITIONS, (nominal, shifted), strict=True):
                for seed in stream_manifest(quick)["weather_bases"]:
                    for step in spec["probe_steps"]:
                        for member in range(3):
                            states = [
                                _replay_integrator(seed, step, spec["episode_length"], basis,
                                                   base, parent["families"], profiles,
                                                   parent["data"]["sensor_seed_offset"], progress)
                                for _ in ARMS
                            ]
                            state_hash = _same_state(states[0], states[1])
                            for arm, state in zip(ARMS, states, strict=True):
                                result = _branch(state, policies[arm, member], scale=d3.SCALE,
                                                 step=step, hold_steps=spec["hold_steps"],
                                                 action_weight=action_weight,
                                                 smooth_weight=smooth_weight, progress=progress)
                                for row in _rows(result, condition=condition, seed=seed,
                                                 step=step, member=member, arm=arm,
                                                 profiles=profiles, state_hash=state_hash):
                                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                                    rows.append(row)
                            handle.flush()
        if progress.bar.n != report["batched_environment_steps"]:
            raise RuntimeError("G2-D4 实际仿真步数与预注册预算不符")
        analysis = summarize(rows, cfg if not quick else dict(cfg, data=spec))
        result = {
            "status": "QUICK_SMOKE_NO_SCIENTIFIC_CONCLUSION" if quick
                      else "DEVELOPMENT_DIAGNOSTIC_REQUIRES_READ_ONLY_AUDIT",
            "records": len(rows), "paired_source_states": report["paired_source_states"],
            "physical_transitions": report["physical_transitions"],
            "analysis": analysis, "elapsed_seconds": time.perf_counter() - started,
            "config_sha256": report["config_sha256"], "entry_sha256": report["entry_sha256"],
            "training_checkpoint_manifest_sha256": report["training_checkpoint_manifest_sha256"],
            "material_passport": {"origin_skill": "academic-research-suite / experiment-agent",
                                  "origin_mode": "run", "verification_status": "UNVERIFIED"},
            **cfg["boundary"],
            "next_action": "停止并等待只读机制审计；不得凭本诊断更改独立确认门槛",
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
    cfg, _, parent, report, output, device = preflight(path, quick=quick)
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
        write_json(output / "runtime.json", {
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device), "git": safe_git_record(),
            "frozen_source_bundle_sha256": d3.train.FROZEN_SOURCE,
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
    parser.add_argument("--quick", action="store_true", help="短 CUDA 冒烟，不形成科学结论")
    parser.add_argument("--preflight-only", action="store_true", help="只读预检")
    args = parser.parse_args()
    print(json.dumps(run(args.config, quick=args.quick,
                         preflight_only=args.preflight_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
