"""G2-D13：冻结当前观测打分器的完整闭环开发对照；正式运行由用户启动。"""
from __future__ import annotations

import argparse
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
from scripts import evaluate_s4_r5_g2_d12_held_out as local
from src.rl.r4_observation import PowerMeasurement, R4Interface
from src.rl.r4_trajectory import anchor_delta
from src.rl.r5_batched_environment import R5BatchedEnvironment
from src.rl.r5_policy_training import ResidualGRUPolicy
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.robust_control import RobustnessCondition

training = local.training
source = training.pairing.source
CONFIG = "configs/experiments/s4_r5_g2_d13_closed_loop_v1.yaml"
LOCAL_ENTRY_SHA256 = "7129d7c2f073a1178406cea671e0b7768ef6742f5b408b8a9be0fd17196910e0"
LOCAL_CONFIG_SHA256 = "d1835bb2724d40c1b3c72adc554f0b31081899bf27c903b800e6ed24eb6740a2"
LOCAL_HASHES = {
    "summary": "66f05ac35170253c30f67b9c5bcf5d0263950bd306dce8276d50330398f98cf1",
    "progress": "95e2615380f1d45ef50bd65ab2387581387741f52a0cbe946933d7d5342f96f7",
    "records": "032c481330097198a5a9d1d1ec6e4e4bb20644f0d49452ac8212d8737b9d6645",
    "predictions": "4d6d64de45263887c5fda276368c7554a02d20a879e81eba1a5b72584440f17b",
    "evaluation": "969492f476e94e8ef598e82cc763b561a8c8d89e0f6b78c56b76cfbc5bb48f7b",
    "manifest": "61bbc9912d6282e79043170a250aa103af65d7868ca93f609fb5f754c4a615e2",
    "weather_metrics": "df94851a59eec2512c6ba89e549eca708ad6d0431234db2a29cc3dbb599ed1c7",
    "group_metrics": "bcabac3d99ce0ea482f47f566739a373ee4da20026bdbb8c5d75852d01cc2e4a",
    "config": "e548c1732940a42e5f0568b43d87f5329cebb341f52c4c4ec01e38403a8d7b06",
}
METRICS = ("power", "measured_power", "strehl", "phase_rmse", "violation", "saturation",
           "slew", "correction_abs", "requested_step_abs", "requested_modal_abs",
           "applied_modal_abs", "requested_applied_gap_abs")
INFO_METRICS = dict(source.TRACE_METRICS)


def contract(cfg: dict) -> None:
    expected = {"stage": "S4-D2-R5-G2-D13",
        "purpose": "frozen_current_scorer_crossfit_complete_episode_development",
        "runtime": {"device": "cuda", "formal_owner": "user_ide", "automatic_retry": False},
        "selector": {"kind": "current", "candidate_epsilon": .1, "deployment_scale": 1.75,
            "selection": "argmax_with_original_zero", "checkpoint": "last", "future_safety_filter": False},
        "data": {"weather_source": "g2_d12_complete_weather_folds", "weather_count": 8,
            "policy_initializations": 3, "scorer_seeds": [7564000, 7564001, 7564002],
            "episode_length": 200, "selector_start_step": 25},
        "quick": {"weather_count": 1, "policy_initializations": 1, "scorer_seeds": [7564999],
            "episode_length": 16, "selector_start_step": 8, "technical_only": True},
        "statistics": {"bootstrap_seed": 7566000, "bootstrap_repeats": 5000, "interval": .975,
            "interval_scope": "descriptive_conditional_on_frozen_crossfit_models"},
        "reference_thresholds": {"minimum_relative_gain": .0105, "maximum_safety_increase": .001},
        "output_directory": "outputs/s4_r5_g2_d13_closed_loop_v1",
        "quick_directory": "outputs/s4_r5_g2_d13_closed_loop_v1_quick",
        "boundary": {"training_updates": 0, "confirmation_access": False, "real_slm_actions": False,
            "independent_confirmation": False, "gate_reclassification": False, "automatic_retry": False}}
    if cfg != expected:
        raise ValueError("G2-D13 冻结闭环合同变化")


def controller_specs(spec: dict) -> list[dict]:
    result = [{"controller": "integrator", "member": None, "scorer_seed": None}]
    for member in range(spec["policy_initializations"]):
        result.append({"controller": f"original_{member}", "member": member, "scorer_seed": None})
        result.extend({"controller": f"current_{member}_{seed}", "member": member, "scorer_seed": seed}
                      for seed in spec["scorer_seeds"])
    return result


def budget(spec: dict) -> dict:
    batches = 2 * spec["weather_count"] * len(controller_specs(spec))
    policy_batches = 2 * spec["weather_count"] * spec["policy_initializations"] * (1 + len(spec["scorer_seeds"]))
    scorer_batches = 2 * spec["weather_count"] * spec["policy_initializations"] * len(spec["scorer_seeds"])
    return {"episode_batches": batches, "complete_episodes": batches * 18,
        "batched_environment_steps": batches * spec["episode_length"],
        "physical_transitions": batches * spec["episode_length"] * 18,
        "policy_forward_calls": policy_batches * spec["episode_length"],
        "scorer_forward_calls": scorer_batches * (spec["episode_length"] - spec["selector_start_step"]),
        "batch_size": 18}


def verify_local_evaluation() -> None:
    if (source._file_sha256(Path(local.__file__)) != LOCAL_ENTRY_SHA256
            or source._file_sha256(source._project_path(local.CONFIG)) != LOCAL_CONFIG_SHA256):
        raise RuntimeError("C 冻结代码或配置变化")
    cfg = source._load_yaml(source._project_path(local.CONFIG)); local.contract(cfg)
    root = source._project_path(cfg["output_directory"])
    if (root / "failure.json").exists():
        raise RuntimeError("C 存在失败标记")
    success = local.read_json(root / "SUCCESS.json")
    for key, digest in LOCAL_HASHES.items():
        suffix = ".pt" if key in ("predictions", "evaluation") else ".jsonl" if key in ("records", "progress") else ".json"
        if source._file_sha256(root / (key + suffix)) != digest or success.get(key + "_sha256") != digest:
            raise RuntimeError(f"C 冻结完成证据变化: {key}")
    summary = local.read_json(root / "summary.json")
    if (summary["quick"] or summary["record_rows"] != 15552 or summary["training_updates"] != 0
            or summary["new_environment_transitions"] != 0 or summary["confirmation_access"]
            or summary["real_slm_actions"] or not summary["scientific_held_out_evaluation_completed"]):
        raise RuntimeError("C 完成性或边界不符")


def route_fold(seed: int, splits: list[dict], *, quick: bool) -> dict:
    matches = [s for s in splits if seed in s["train_weather" if quick else "held_out_weather"]]
    if len(matches) != 1 or (not quick and seed in matches[0]["train_weather"]):
        raise ValueError("闭环天气须使用唯一未见该天气的折")
    return matches[0]


def load_assets(device: torch.device, *, quick: bool) -> tuple[dict, dict, dict, list[dict], list[dict]]:
    # 来源校验可核对标签文件哈希，但下面不加载 targets/evaluation/predictions，更不把标签送入选择器。
    verify_local_evaluation()
    root, _, summary = local.verify_training(quick)
    dataset = training.verify_dataset(quick)
    inputs = torch.load(dataset / "inputs.pt", map_location=device, weights_only=True)
    x, valid, command = training.features(inputs)
    rows = [json.loads(line) for line in (dataset / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = training.weather_splits(rows, quick=quick)
    if local.read_json(root / "splits.json") != splits:
        raise RuntimeError("B 折分与 A 不符")
    parent = source.verify_sources()
    seeds = [7564999] if quick else [7564000, 7564001, 7564002]
    models = {}; manifest = []
    for split in splits:
        expected = training.fit_normalizer(x, valid, command, torch.tensor(split["train_indices"], device=device))
        for seed in seeds:
            name = f"fold_{split['fold']}_current_seed_{seed}_{2 if quick else 1000:05d}.pt"
            ck = torch.load(root / "checkpoints" / name, map_location=device, weights_only=True)
            local.validate_checkpoint(ck, split, "current", seed, quick=quick)
            if any(not torch.equal(v, expected[k]) for k, v in ck["normalizer"].items()):
                raise RuntimeError("标准化不是本折训练天气统计")
            model = training.CandidateScorer("current").to(device)
            model.load_state_dict(ck["state_dict"])
            if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise RuntimeError("打分器参数非有限")
            models[split["fold"], seed] = (model.eval().requires_grad_(False), ck["normalizer"])
            manifest.append({"fold": split["fold"], "kind": "current", "seed": seed,
                "checkpoint": name, "checkpoint_sha256": summary["checkpoint_hashes"][name],
                "train_weather": split["train_weather"], "held_out_weather": split["held_out_weather"],
                "technical_in_sample_only": quick})
    policies = {}
    for member in range(1 if quick else 3):
        name = f"action_penalty_0_policy_{member}_00512.pt"
        path = source._project_path("outputs/s4_r5_g2_d8_action_penalty_zero_v1/checkpoints") / name
        if source._file_sha256(path) != source.d10.d9.D8_FINAL[name]:
            raise RuntimeError("原 D8 权重变化")
        ck = torch.load(path, map_location=device, weights_only=True)
        if (ck["arm"], ck["init"], ck["update"], ck["deployment_scale"]) != ("action_penalty_0", member, 512, 1.75):
            raise RuntimeError("D8 策略身份不符")
        policy = ResidualGRUPolicy(parent["policy"]["hidden_size"], parent["policy"]["output_size"]).to(device)
        policy.load_state_dict(ck["state_dict"])
        policies[member] = policy.eval().requires_grad_(False)
    return parent, policies, models, splits, manifest


def select_command(history: torch.Tensor, valid: torch.Tensor, original: torch.Tensor,
                   model: training.CandidateScorer, normalizer: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """只接收动作前因果白名单；不接收环境、真值、安全、档位或天气元数据。"""
    x, valid, command = training.features({"history": history, "valid": valid, "command": original})
    prediction = local.forward_causal(model, x, valid, command, normalizer)
    choice = training.candidate_choice(prediction)
    candidates = torch.stack(list(source.candidate_commands(original, .1).values()), dim=1)
    selected = candidates[torch.arange(len(original), device=original.device), choice]
    return selected, choice, prediction


def preflight(path: str | Path = CONFIG, *, quick: bool = False) -> tuple:
    cfg = source._load_yaml(source._project_path(path)); contract(cfg)
    output = source._project_path(cfg["quick_directory" if quick else "output_directory"])
    if output.exists():
        raise FileExistsError(f"保留已有 G2-D13 输出，禁止覆盖: {output}")
    device = resolve_device("cuda")
    source.verify_streams()
    assets = load_assets(device, quick=quick)
    spec = cfg["quick" if quick else "data"]
    streams = source.stream_manifest(quick)
    if len(streams["weather_bases"]) != spec["weather_count"]:
        raise RuntimeError("完整天气预算不符")
    for seed in streams["weather_bases"]:
        route_fold(seed, assets[3], quick=quick)
    report = {"status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_IDE",
        "quick": quick, "device": str(device), **budget(spec), "weather_count": spec["weather_count"],
        "episode_length": spec["episode_length"], "selector_start_step": spec["selector_start_step"],
        "controller_ids": [b["controller"] for b in controller_specs(spec)],
        "entry_sha256": source._file_sha256(Path(__file__)),
        "config_sha256": source._file_sha256(source._project_path(path)),
        "local_evaluation_hashes": LOCAL_HASHES, "preflight_model_forward_calls": 0,
        "preflight_environment_transitions": 0, "scorer_last_checkpoints": len(assets[2]),
        "weather_scope": "technical_in_sample" if quick else "reused_development_weather_crossfit_not_confirmation",
        **cfg["boundary"]}
    return cfg, spec, output, device, assets, report


def trace_means(trace: dict[str, torch.Tensor], steps: int) -> dict[str, torch.Tensor]:
    if any(trace[k].shape != (steps, 18) for k in METRICS):
        raise ValueError("完整回合轨迹长度或批量不符")
    if any(not bool(torch.isfinite(trace[k]).all()) for k in METRICS):
        raise ValueError("完整回合科学指标非有限")
    return {k: trace[k].double().mean(0) for k in METRICS}


def require_prefix(original: dict, candidate: dict, prefix: int) -> None:
    """选择器启用前必须与原策略逐张量相同，防止随机流或状态配对错误。"""
    a, b = original["metrics_and_actions"], candidate["metrics_and_actions"]
    for key in (*METRICS, "history_frame", "history_valid", "original_command", "selected_command", "requested_delta", "requested_modal", "applied_modal"):
        if not torch.equal(a[key][:prefix], b[key][:prefix]):
            raise RuntimeError(f"选择器启用前配对轨迹不一致: {key}")


@torch.no_grad()
def rollout(seed: int, branch: dict, condition: str, profiles: list, spec: dict, base,
            basis: torch.Tensor, parent: dict, policy, selector, progress) -> tuple[list[dict], dict]:
    first = RobustnessCondition.from_mapping(dict(parent["families"][0], base_seed=seed))
    env = R5BatchedEnvironment(replace(first.environment_config(base), episode_length=spec["episode_length"]),
        basis.device, basis, parent["families"], profiles, parent["data"]["sensor_seed_offset"])
    raw, _ = env.reset(seed=seed)
    interface = R4Interface(); interface.reset(env.proxy(raw), episode_id=f"g2-d13-{condition}-{seed}-{branch['controller']}")
    trace = {k: [] for k in (*METRICS, "history_frame", "history_valid", "original_command", "selected_command", "requested_delta",
                            "requested_modal", "applied_modal", "choice", "prediction_scaled")}
    times = []; calls = 0
    for step in range(spec["episode_length"]):
        view = interface.snapshot()
        baseline = anchor_delta(view.features[:, -1], {"gain": .15, "leak": .10, "tracking_gain": .50})
        torch.cuda.synchronize(basis.device); tick = time.perf_counter()
        original = raw.new_zeros((18, 11)) if policy is None else policy(view.features, view.valid) * 1.75
        selected = original; choice = torch.zeros(18, dtype=torch.long, device=basis.device)
        prediction = raw.new_zeros((18, 23))
        if selector is not None and step >= spec["selector_start_step"]:
            selected, choice, prediction = select_command(view.features, view.valid, original, *selector)
            calls += 1
        torch.cuda.synchronize(basis.device); times.append(time.perf_counter() - tick)
        action = interface.issue(baseline, selected, step=step)
        raw, _, terminated, truncated, info = env.step(action.requested_delta_rad)
        if bool(truncated.any()) or bool(terminated.any()) != (step + 1 == spec["episode_length"]):
            raise RuntimeError("闭环回合未完整结束")
        if step + 1 == spec["episode_length"] and not bool(terminated.all()):
            raise RuntimeError("批量回合结束不一致")
        # 环境 info 仅在动作选择完成后用于日志及合法的已到达功率；不传给选择器。
        interface.observe_next(env.proxy(raw), step=step + 1,
            power=PowerMeasurement(info["measured_power_in_bucket"], step, step + 1))
        values = {k: info[v] for k, v in INFO_METRICS.items()}
        values.update(correction_abs=action.normalized_correction.abs().mean(-1),
            requested_step_abs=action.requested_delta_rad.abs().mean(-1),
            requested_modal_abs=info["requested_modal"].abs().mean(-1),
            applied_modal_abs=info["applied_modal"].abs().mean(-1),
            requested_applied_gap_abs=(info["requested_modal"] - info["applied_modal"]).abs().mean(-1),
            history_frame=view.features[:, -1], history_valid=view.valid,
            original_command=original, selected_command=selected, requested_delta=action.requested_delta_rad,
            requested_modal=info["requested_modal"], applied_modal=info["applied_modal"], choice=choice,
            prediction_scaled=prediction)
        for key, value in values.items():
            trace[key].append(value.detach().clone())
        progress.tick({"天气种子": float(seed), "回合步": float(step + 1),
                       "平均桶内功率": float(info["reward_power_in_bucket"].mean())})
    tensors = {k: torch.stack(v) for k, v in trace.items()}
    if any(not bool(torch.isfinite(v).all()) for v in tensors.values()):
        raise RuntimeError("闭环动作、预测或读数非有限")
    means = trace_means(tensors, spec["episode_length"])
    rows = []
    sorted_times = sorted(times)
    for slot, profile in enumerate(profiles):
        for fi, family in enumerate(parent["families"]):
            i = slot * 3 + fi
            rows.append({**branch, "hardware_condition": condition, "weather_seed": seed, "slot": slot,
                "family": family["id"], "profile": profile.identifier, "episode_length": spec["episode_length"],
                "turbulence_stream_seed": seed + 1000 * slot + fi,
                "sensor_stream_seed": seed + 1000 * slot + parent["data"]["sensor_seed_offset"],
                "power_stream_seed": seed + 1000 * slot + 60_000_000,
                "selector_calls": calls, "selected_nonoriginal_fraction": float((tensors["choice"][:, i] != 0).double().mean()),
                "decision_seconds_per_batch": sum(times) / len(times),
                "decision_p95_seconds_per_batch": sorted_times[math.ceil(.95 * len(times)) - 1],
                **{k: float(v[i]) for k, v in means.items()}})
    return rows, {"branch": branch, "condition": condition, "weather_seed": seed,
                  "steps": spec["episode_length"], "metrics_and_actions": {k: v.cpu() for k, v in tensors.items()}}


def bootstrap_interval(numerator: torch.Tensor, draws: torch.Tensor,
                       denominator: torch.Tensor | None = None) -> list[float]:
    samples = numerator[draws].mean(1)
    if denominator is not None:
        if bool((denominator <= 0).any()):
            raise ValueError("相对收益基准功率必须为正")
        samples = samples / denominator[draws].mean(1)
    if not bool(torch.isfinite(samples).all()):
        raise ValueError("重采样统计非有限")
    return torch.quantile(samples, samples.new_tensor([.0125, .9875])).tolist()


def summarize(rows: list[dict], cfg: dict, *, device: torch.device, quick: bool = False) -> dict:
    spec = cfg["quick" if quick else "data"]
    branches = controller_specs(spec); ids = [b["controller"] for b in branches]
    weather = source.stream_manifest(quick)["weather_bases"]
    keys = {(r["hardware_condition"], r["weather_seed"], r["controller"], r["family"], r["slot"]): r for r in rows}
    expected = {(c, w, b, f, s) for c in source.CONDITIONS for w in weather for b in ids for f in source.FAMILIES for s in range(6)}
    if len(rows) != len(expected) or set(keys) != expected:
        raise ValueError("闭环完整网格缺失或重复")
    branch_map = {b["controller"]: b for b in branches}
    for row in rows:
        slot, seed = row["slot"], row["weather_seed"]
        profile = f"nominal_for_{source.PROFILES[slot]}" if row["hardware_condition"] == "nominal_clone" else source.PROFILES[slot]
        if (row["profile"] != profile or row["episode_length"] != spec["episode_length"]
                or any(row[k] != branch_map[row["controller"]][k] for k in ("member", "scorer_seed"))
                or row["turbulence_stream_seed"] != seed + 1000 * slot + source.FAMILIES.index(row["family"])
                or row["sensor_stream_seed"] != seed + 1000 * slot + 50_000_000
                or row["power_stream_seed"] != seed + 1000 * slot + 60_000_000
                or any(not math.isfinite(row[k]) for k in METRICS)
                or any(not 0 <= row[k] <= 1 for k in ("violation", "saturation", "slew"))
                or not 0 <= row["selected_nonoriginal_fraction"] <= 1
                or row["selector_calls"] != (0 if row["scorer_seed"] is None else spec["episode_length"] - spec["selector_start_step"])):
            raise ValueError("闭环动作身份、种子、指标或预算错位")
        fold = None if row["scorer_seed"] is None else (-1 if quick else weather.index(seed) // 2)
        if row["scorer_fold"] != fold:
            raise ValueError("闭环模型折路由错误")
    values = {k: torch.tensor([[[[keys[(c, w, b, f, s)][k] for f in source.FAMILIES for s in range(6)]
        for w in weather] for b in ids] for c in source.CONDITIONS], dtype=torch.float64, device=device) for k in METRICS}
    if quick:
        return {}  # 技术集合不输出科学值或放行判断。
    originals = [ids.index(f"original_{m}") for m in range(3)]
    currents = [ids.index(f"current_{m}_{s}") for m in range(3) for s in spec["scorer_seeds"]]
    generator = torch.Generator(device=device).manual_seed(cfg["statistics"]["bootstrap_seed"])
    draws = torch.randint(len(weather), (cfg["statistics"]["bootstrap_repeats"], len(weather)), generator=generator, device=device)
    cells = {}; table = []
    for ci, condition in enumerate(source.CONDITIONS):
        means = {k: {"integrator": v[ci, 0], "original": v[ci, originals].mean(0),
                     "current": v[ci, currents].mean(0)} for k, v in values.items()}
        original = means["power"]["original"]; current = means["power"]["current"]
        base = means["power"]["integrator"]
        if bool((base <= 0).any()):
            raise ValueError("积分器桶内功率非正")
        effect = (current - original).mean(1)
        seed_effects = {}
        for seed in spec["scorer_seeds"]:
            indexes = [ids.index(f"current_{m}_{seed}") for m in range(3)]
            seed_effects[str(seed)] = float((values["power"][ci, indexes].mean(0) - original).mean())
        changes = {k: {"vs_original": float((v["current"] - v["original"]).mean()),
                       "vs_integrator": float((v["current"] - v["integrator"]).mean())} for k, v in means.items()}
        relative = float((current - base).mean() / base.mean())
        safety = cfg["reference_thresholds"]["maximum_safety_increase"]
        cells[condition] = {"weather_clusters": len(weather),
            "method_means": {name: {k: float(v[name].mean()) for k, v in means.items()} for name in ("integrator", "original", "current")},
            "current_minus_original_power": float(effect.mean()),
            "current_minus_original_descriptive_ci97_5": bootstrap_interval(effect, draws),
            "positive_weather_count": int((effect > 0).sum()),
            "original_relative_gain_vs_integrator": float((original - base).mean() / base.mean()),
            "current_relative_gain_vs_integrator": relative,
            "current_relative_gain_descriptive_ci97_5": bootstrap_interval((current - base).mean(1), draws, base.mean(1)),
            "by_scorer_seed_current_minus_original": seed_effects, "metric_deltas": changes,
            "reference_checks_not_gate_reclassification": {
                "relative_gain_at_least_1_05_percent": relative >= cfg["reference_thresholds"]["minimum_relative_gain"],
                "current_minus_original_positive": float(effect.mean()) > 0,
                "each_scorer_seed_effect_positive": all(v > 0 for v in seed_effects.values()),
                "strehl_non_decrease_vs_both": all(v >= 0 for v in changes["strehl"].values()),
                "phase_rmse_non_increase_vs_both": all(v <= 0 for v in changes["phase_rmse"].values()),
                "mean_safety_increase_at_most_0_001_vs_both": all(v <= safety for k in ("violation", "saturation", "slew") for v in changes[k].values())}}
        for fi, family in enumerate(source.FAMILIES):
            for slot in range(6):
                idx = fi * 6 + slot
                table.append({"hardware_condition": condition, "family": family, "slot": slot,
                    "weather_count": len(weather), "current_minus_original_power": float((current - original)[:, idx].mean()),
                    "metric_deltas": {k: {"vs_original": float((v["current"] - v["original"])[:, idx].mean()),
                                          "vs_integrator": float((v["current"] - v["integrator"])[:, idx].mean())} for k, v in means.items()}})
    return {"status": "DEVELOPMENT_CLOSED_LOOP_NO_AUTOMATIC_GATE_PROMOTION", "cells": cells, "group_table": table,
        "interval_scope": cfg["statistics"]["interval_scope"], "training_uncertainty_covered": False,
        "prior_d9_gate_reclassified": False, "independent_confirmation": False,
        "seed_aggregation": "average_of_separate_closed_loop_trajectories_not_ensemble_deployment"}


@torch.no_grad()
def run(path: str | Path = CONFIG, *, quick: bool = False, preflight_only: bool = False) -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    cfg, spec, output, device, assets, report = preflight(path, quick=quick)
    if preflight_only:
        return report
    parent, policies, models, splits, manifest = assets
    started = time.perf_counter(); created = False; progress = None
    try:
        output.mkdir(parents=True, exist_ok=False); created = True
        (output / "trajectories").mkdir()
        source.write_json(output / "preflight.json", report); source.write_json(output / "config.json", cfg)
        source.write_json(output / "model_manifest.json", manifest)
        source.write_json(output / "stream_manifest.json", source.stream_manifest(quick))
        source.write_json(output / "runtime.json", {"torch": str(torch.__version__), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device), "git": source.safe_git_record(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "allow_tf32": False})
        base, _ = load_s1_config(source._project_path(parent["environment_config"]))
        base = replace(base, num_modes=21, batch_size=1, episode_length=spec["episode_length"])
        basis, _, _ = build_action_basis(base, ActionRepresentation("r5_zernike21", "zernike", 21), device)
        nominal, shifted = source.d10.d9.d8.d2.g2.d1._profile_pairs(parent)
        progress = source.d10.d9.d3.SparseProgress(output, device, spec["episode_length"])
        progress.phase("G2-D13 CUDA 技术冒烟" if quick else "G2-D13 完整闭环开发对照", report["batched_environment_steps"])
        rows = []; trace_manifest = []; completed = 0; prefix_checks = 0
        policy_calls = 0; scorer_calls = 0
        with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
            for condition, profiles in zip(source.CONDITIONS, (nominal, shifted), strict=True):
                for seed in source.stream_manifest(quick)["weather_bases"]:
                    split = route_fold(seed, splits, quick=quick)
                    original_traces = {}
                    for branch in controller_specs(spec):
                        policy = policies.get(branch["member"])
                        selector = None if branch["scorer_seed"] is None else models[split["fold"], branch["scorer_seed"]]
                        part, trace = rollout(seed, branch, condition, profiles, spec, base, basis, parent, policy, selector, progress)
                        policy_calls += spec["episode_length"] if policy is not None else 0
                        scorer_calls += part[0]["selector_calls"]
                        if branch["member"] is not None and selector is None:
                            original_traces[branch["member"]] = trace
                        if selector is not None:
                            require_prefix(original_traces[branch["member"]], trace, spec["selector_start_step"])
                            prefix_checks += 1
                        name = f"batch_{completed:04d}.pt"; trace_path = output / "trajectories" / name
                        torch.save(trace, trace_path)
                        trace_manifest.append({"file": name, "sha256": source._file_sha256(trace_path),
                            "condition": condition, "weather_seed": seed, **branch, "scorer_fold": split["fold"] if selector else None})
                        for row in part:
                            row.update(trajectory_file=name, scorer_fold=split["fold"] if selector else None)
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        handle.flush(); rows.extend(part); completed += 1
        if (len(rows) != report["complete_episodes"] or completed != report["episode_batches"]
                or progress.bar.n != report["batched_environment_steps"]
                or policy_calls != report["policy_forward_calls"] or scorer_calls != report["scorer_forward_calls"]):
            raise RuntimeError("实际闭环预算不符")
        analysis = summarize(rows, cfg, device=device, quick=quick)
        source.write_json(output / "trajectory_manifest.json", trace_manifest)
        result = {**report, "status": "QUICK_SMOKE_NO_SCIENTIFIC_CONCLUSION" if quick else "COMPLETE_EPISODE_DEVELOPMENT_REQUIRES_READ_ONLY_AUDIT",
            "completed_episodes": len(rows), "completed_episode_batches": completed, "failed_episodes": 0,
            "exact_original_prefix_checks": prefix_checks,
            "completed_policy_forward_calls": policy_calls, "completed_scorer_forward_calls": scorer_calls,
            "analysis": analysis, "elapsed_seconds": time.perf_counter() - started,
            "decision_latency_scope": "CUDA batch-18 policy plus optional scorer/candidate selection; excludes projection/optics/I-O; no real-time hardware claim",
            "material_passport": {"origin_skill": "academic-research-suite / experiment-agent", "origin_mode": "run",
                "origin_date": datetime.now(timezone.utc).date().isoformat(), "verification_status": "UNVERIFIED", "version_label": "g2_d13_closed_loop_v1"},
            "next_action": "停止等待只读完整回合审计；不自动加训、调阈值或打开独立确认"}
        source.write_json(output / "summary.json", result)
        source.write_json(output / "SUCCESS.json", {f"{key}_sha256": source._file_sha256(output / file)
            for key, file in (("summary", "summary.json"), ("records", "records.jsonl"), ("progress", "progress.jsonl"),
                ("model_manifest", "model_manifest.json"), ("trajectory_manifest", "trajectory_manifest.json"),
                ("stream_manifest", "stream_manifest.json"), ("config", "config.json"))})
        return result
    except Exception:
        if created:
            source.write_json(output / "failure.json", {"traceback": traceback.format_exc(), "automatic_retry": False})
        raise
    finally:
        if progress is not None:
            progress.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--quick", action="store_true", help="16帧技术冒烟，不是科学评价")
    parser.add_argument("--preflight-only", action="store_true", help="只读预检，不创建输出或生成回合")
    args = parser.parse_args()
    print(json.dumps(run(args.config, quick=args.quick, preflight_only=args.preflight_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
