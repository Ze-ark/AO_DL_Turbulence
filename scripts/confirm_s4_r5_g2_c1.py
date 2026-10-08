"""G2-C1：冻结微调方法、全新天气一次性确认；正式运行由用户在 IDE 启动。"""
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
from scripts import evaluate_s4_r5_g2_d13_closed_loop as frozen
from src.runtime import resolve_device

source = frozen.source
CONFIG = "configs/experiments/s4_r5_g2_c1_confirmation_v1.yaml"
D13_ENTRY = "bbcac8d5ae80d6d4670591a11cb8e051bd5a873bd1a4c45789f9ce5df4e3933c"
D13_CONFIG = "8c5ca5312c0fd35968d05108d5fb8e1670fb9c3a8ccb0c0f509265e62397ae55"
D13_HASHES = {
    "summary": "b121413aa582df6dfc4bd141a7ecc071e9d7d85e32e59d02a6a0dc027059704b",
    "records": "5fcd7dc412966bfdf192b161680de2d5ad0214df08b5072dae1121e54971c0e8",
    "progress": "32d33b0673d66bf743a7298e1d87af7d016451196bb455fb66eb0b40ff4e5c89",
    "model_manifest": "877d56e59a2c0a18c5b0c41da0fdf4855c0d420b5956afc1f384b45ea15ae7b1",
    "trajectory_manifest": "9819a2ce8e79bd9cbddedf3051f4bbe2d68b2ade034a5a93c798dd155602e97f",
    "stream_manifest": "33015f7dc5f9d286ee2b22adea1a53c111a538da86e75dc1d7f3453c854a7bb9",
    "config": "6b0eaa6b21afd8eb7b850b5e2f081e73548613362210265e10c26ca226d1efe3",
}


def contract(cfg: dict) -> None:
    expected = {
        "stage": "S4-D2-R5-G2-C1",
        "purpose": "frozen_candidate_selector_fresh_weather_independent_confirmation",
        "runtime": {"device": "cuda", "formal_owner": "user_ide", "automatic_retry": False},
        "selector": {"kind": "current", "candidate_epsilon": .1, "deployment_scale": 1.75,
            "selection": "argmax_with_original_zero", "checkpoint": "last", "future_safety_filter": False},
        "data": {"seed_base": 7800000, "seed_stride": 10, "weather_count": 64,
            "policy_initializations": 3, "scorer_seeds": [7564000, 7564001, 7564002],
            "episode_length": 200, "selector_start_step": 25, "fold_assignment": "weather_index_modulo_4"},
        "quick": {"seed_base": 7810000, "seed_stride": 10, "weather_count": 1,
            "policy_initializations": 1, "scorer_seeds": [7564000], "episode_length": 32,
            "selector_start_step": 25, "fold_assignment": "weather_index_modulo_4", "technical_only": True},
        "statistics": {"bootstrap_seed": 7820000, "bootstrap_repeats": 20000, "interval": .95,
            "cluster": "complete_weather", "stratification": "assigned_scorer_fold",
            "interval_scope": "conditional_on_frozen_models_no_retraining_uncertainty"},
        "thresholds": {"minimum_relative_gain": .01, "maximum_safety_increase": .001},
        "historical_development_reference": {"minimum_relative_gain": .0105, "gate_for_this_confirmation": False},
        "output_directory": "outputs/s4_r5_g2_c1_confirmation_v1",
        "quick_directory": "outputs/s4_r5_g2_c1_confirmation_v1_quick",
        "boundary": {"training_updates": 0, "old_confirmation_access": False, "real_slm_actions": False,
            "historical_gate_reclassification": False, "automatic_retry": False, "formal_runs_per_version": 1},
    }
    if cfg != expected:
        raise ValueError("G2-C1 事前冻结确认合同变化")


def stream_manifest(spec: dict) -> dict:
    weather = [spec["seed_base"] + spec["seed_stride"] * i for i in range(spec["weather_count"])]
    return {"weather_bases": weather,
        "turbulence": [w + 1000 * slot + family for w in weather for slot in range(6) for family in range(3)],
        "sensor": [w + 1000 * slot + 50_000_000 for w in weather for slot in range(6)],
        "power": [w + 1000 * slot + 60_000_000 for w in weather for slot in range(6)],
        "scorer_fold_by_weather": {str(w): i % 4 for i, w in enumerate(weather)},
        "paired_conditions_and_controllers_share_streams": True,
        "assignment_uses_no_observations_or_outcomes": True}


def require_disjoint_streams(a: dict, b: dict) -> None:
    for key in ("turbulence", "sensor", "power"):
        if (len(set(a[key])) != len(a[key]) or len(set(b[key])) != len(b[key])
                or set(a[key]) & set(b[key])):
            raise ValueError(f"确认/技术/历史随机流重叠或内部重复: {key}")


def verify_streams(cfg: dict) -> None:
    formal, technical = stream_manifest(cfg["data"]), stream_manifest(cfg["quick"])
    require_disjoint_streams(formal, technical)
    source.verify_streams()
    old = [source.stream_manifest(q) for q in (False, True)]
    old += [source.d10.d9.stream_manifest(q) for q in (False, True)]
    old += [source.d10.d9.d8.stream_manifest(quick=q) for q in (False, True)]
    old += [source.d10.d9.d3.stream_manifest(q) for q in (False, True)]
    old += [source.d10.d9.d8.d2.g2.stream_manifest(q) for q in (False, True)]
    for previous in old:
        require_disjoint_streams(formal, previous)
        require_disjoint_streams(technical, previous)
    # 旧 R5 各阶段及未使用的 720/760 万确认区间保持封存，不读取其科学结果。
    for current in (formal, technical):
        for key, offset in (("turbulence", 0), ("sensor", 50_000_000), ("power", 60_000_000)):
            if any(s - offset < 7_800_000 for s in current[key]):
                raise ValueError("不得占用历史/预留随机流命名空间")


def assigned_fold(seed: int, spec: dict, splits: list[dict]) -> int:
    """全新天气事前均衡分配四份折模型；不是根据收益选最好的折。"""
    manifest = stream_manifest(spec)
    if seed not in manifest["weather_bases"] or sorted(s["fold"] for s in splits) != list(range(4)):
        raise ValueError("确认天气或四折身份不符")
    if any(seed in s["train_weather"] or seed in s["held_out_weather"] for s in splits):
        raise ValueError("确认天气不得复用训练或开发天气")
    return manifest["scorer_fold_by_weather"][str(seed)]


def verify_development() -> dict:
    if (source._file_sha256(Path(frozen.__file__)) != D13_ENTRY
            or source._file_sha256(source._project_path(frozen.CONFIG)) != D13_CONFIG):
        raise RuntimeError("D13 冻结控制实现/配置变化")
    cfg = source._load_yaml(source._project_path(frozen.CONFIG)); frozen.contract(cfg)
    root = source._project_path(cfg["output_directory"])
    if (root / "failure.json").exists():
        raise RuntimeError("D13 有失败标记，不能作为确认来源")
    success = frozen.local.read_json(root / "SUCCESS.json")
    for key, digest in D13_HASHES.items():
        name = key + (".jsonl" if key in ("records", "progress") else ".json")
        if source._file_sha256(root / name) != digest or success.get(key + "_sha256") != digest:
            raise RuntimeError(f"D13 完成证据变化: {key}")
    summary = frozen.local.read_json(root / "summary.json")
    if (summary["quick"] or summary["completed_episodes"] != 3744
            or summary["completed_episode_batches"] != 208 or summary["failed_episodes"] != 0
            or summary["training_updates"] != 0 or summary["independent_confirmation"]
            or summary["real_slm_actions"] or summary["entry_sha256"] != D13_ENTRY):
        raise RuntimeError("D13 完成性/边界不符")
    traces = frozen.local.read_json(root / "trajectory_manifest.json")
    names = {p.name for p in (root / "trajectories").glob("*.pt")}
    if len(traces) != 208 or {t["file"] for t in traces} != names or len(names) != 208:
        raise RuntimeError("D13 完整轨迹集合缺失或重复")
    for item in traces:
        if Path(item["file"]).name != item["file"]:
            raise ValueError("轨迹清单路径非法")
        if source._file_sha256(root / "trajectories" / item["file"]) != item["sha256"]:
            raise RuntimeError("D13 轨迹哈希变化")
    return summary


def preflight(path: str | Path = CONFIG, *, quick: bool = False) -> tuple:
    cfg = source._load_yaml(source._project_path(path)); contract(cfg)
    output = source._project_path(cfg["quick_directory" if quick else "output_directory"])
    if output.exists() or source._project_path(cfg["output_directory"]).exists():
        raise FileExistsError(f"保留已有 G2-C1 输出，禁止覆盖/重新确认: {output}")
    device = resolve_device("cuda")
    verify_streams(cfg); development = verify_development()
    # 技术冒烟也加载正式冻结权重，绝不训练一个专用冒烟模型。
    assets = frozen.load_assets(device, quick=False)
    if assets[4] != frozen.local.read_json(source._project_path("outputs/s4_r5_g2_d13_closed_loop_v1/model_manifest.json")):
        raise RuntimeError("确认的全部末次打分器与 D13 不一致")
    spec = cfg["quick" if quick else "data"]
    for seed in stream_manifest(spec)["weather_bases"]:
        assigned_fold(seed, spec, assets[3])
    report = {"status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_IDE",
        "quick": quick, "device": str(device), **frozen.budget(spec),
        "weather_count": spec["weather_count"], "episode_length": spec["episode_length"],
        "selector_start_step": spec["selector_start_step"],
        "controller_ids": [b["controller"] for b in frozen.controller_specs(spec)],
        "entry_sha256": source._file_sha256(Path(__file__)),
        "config_sha256": source._file_sha256(source._project_path(path)),
        "frozen_d13_entry_sha256": D13_ENTRY, "development_hashes": D13_HASHES,
        "preflight_model_forward_calls": 0, "preflight_environment_transitions": 0,
        "scorer_last_checkpoints": len(assets[2]), "independent_confirmation": not quick,
        "confirmation_access": False,  # 只读预检尚未生成确认回合。
        "weather_scope": "new_technical_weather" if quick else "fresh_unseen_weather_not_yet_generated",
        "development_complete_episodes": development["completed_episodes"], **cfg["boundary"]}
    return cfg, spec, output, device, assets, report


def bootstrap_draws(spec: dict, statistics: dict, device: torch.device) -> torch.Tensor:
    """在固定的四折路由内按完整天气配对重采样，保持四折等权。"""
    count = spec["weather_count"]
    if count % 4 or count < 8:
        raise ValueError("正式确认天气须均衡分配四折")
    generator = torch.Generator(device=device).manual_seed(statistics["bootstrap_seed"])
    parts = []
    for fold in range(4):
        indexes = torch.arange(fold, count, 4, device=device)
        choices = torch.randint(len(indexes), (statistics["bootstrap_repeats"], len(indexes)),
            device=device, generator=generator)
        parts.append(indexes[choices])
    return torch.cat(parts, dim=1)


def interval(numerator: torch.Tensor, draws: torch.Tensor,
             denominator: torch.Tensor | None = None) -> list[float]:
    samples = numerator[draws].mean(1)
    if denominator is not None:
        if bool((denominator <= 0).any()):
            raise ValueError("积分器基准功率必须为正")
        samples = samples / denominator[draws].mean(1)
    if not bool(torch.isfinite(samples).all()):
        raise ValueError("重采样统计非有限")
    return torch.quantile(samples, samples.new_tensor([.025, .975])).tolist()


def validate_rows(rows: list[dict], cfg: dict, *, quick: bool) -> tuple[list[int], list[str], dict]:
    spec = cfg["quick" if quick else "data"]
    weather = stream_manifest(spec)["weather_bases"]
    branches = {b["controller"]: b for b in frozen.controller_specs(spec)}
    keys = {(r["hardware_condition"], r["weather_seed"], r["controller"], r["family"], r["slot"]): r for r in rows}
    expected = {(c, w, b, f, s) for c in source.CONDITIONS for w in weather
        for b in branches for f in source.FAMILIES for s in range(6)}
    if len(rows) != len(expected) or set(keys) != expected:
        raise ValueError("确认完整配对网格缺失或重复")
    for row in rows:
        seed, slot, family = row["weather_seed"], row["slot"], row["family"]
        branch = branches[row["controller"]]
        selector = branch["scorer_seed"] is not None
        profile = f"nominal_for_{source.PROFILES[slot]}" if row["hardware_condition"] == "nominal_clone" else source.PROFILES[slot]
        if (row["profile"] != profile or row["episode_length"] != spec["episode_length"]
                or any(row[k] != branch[k] for k in ("member", "scorer_seed"))
                or row["scorer_fold"] != (weather.index(seed) % 4 if selector else None)
                or row["turbulence_stream_seed"] != seed + 1000 * slot + source.FAMILIES.index(family)
                or row["sensor_stream_seed"] != seed + 1000 * slot + 50_000_000
                or row["power_stream_seed"] != seed + 1000 * slot + 60_000_000
                or row["selector_calls"] != (spec["episode_length"] - spec["selector_start_step"] if selector else 0)
                or any(not math.isfinite(row[k]) for k in frozen.METRICS)
                or any(not 0 <= row[k] <= 1 for k in ("violation", "saturation", "slew", "selected_nonoriginal_fraction"))
                or any(not math.isfinite(row[k]) or row[k] < 0 for k in
                    ("decision_seconds_per_batch", "decision_p95_seconds_per_batch"))):
            raise ValueError("确认身份、分折、随机流、指标或预算错位")
    return weather, list(branches), keys


def summarize(rows: list[dict], cfg: dict, *, device: torch.device, quick: bool = False) -> dict:
    weather, ids, keys = validate_rows(rows, cfg, quick=quick)
    if quick:
        return {}  # 技术检查不输出科学门槛或科学结论。
    spec = cfg["data"]
    values = {k: torch.tensor([[[[keys[(c, w, b, f, s)][k] for f in source.FAMILIES for s in range(6)]
        for w in weather] for b in ids] for c in source.CONDITIONS], dtype=torch.float64, device=device)
        for k in frozen.METRICS}
    originals = [ids.index(f"original_{m}") for m in range(3)]
    currents = [ids.index(f"current_{m}_{s}") for m in range(3) for s in spec["scorer_seeds"]]
    draws = bootstrap_draws(spec, cfg["statistics"], device)
    cells = {}; groups = []
    for ci, condition in enumerate(source.CONDITIONS):
        means = {k: {"integrator": v[ci, 0], "original": v[ci, originals].mean(0),
                     "current": v[ci, currents].mean(0)} for k, v in values.items()}
        base, current, original = (means["power"][k] for k in ("integrator", "current", "original"))
        if bool((base <= 0).any()):
            raise ValueError("积分器基准功率必须为正")
        advantage = current - base
        relative = float(advantage.mean() / base.mean())
        absolute_ci = interval(advantage.mean(1), draws)
        deltas = {k: {"vs_integrator": float((v["current"] - v["integrator"]).mean()),
                       "vs_original": float((v["current"] - v["original"]).mean())} for k, v in means.items()}
        member_gains = {str(m): float((values["power"][ci,
            [ids.index(f"current_{m}_{s}") for s in spec["scorer_seeds"]]].mean(0) - base).mean()) for m in range(3)}
        scorer_gains = {str(s): float((values["power"][ci,
            [ids.index(f"current_{m}_{s}") for m in range(3)]].mean(0) - base).mean()) for s in spec["scorer_seeds"]}
        combination_gains = {ids[i]: float((values["power"][ci, i] - base).mean()) for i in currents}
        family_gains = {f: float(advantage[:, j * 6:(j + 1) * 6].mean()) for j, f in enumerate(source.FAMILIES)}
        slot_gains = {str(s): float(advantage[:, s::6].mean()) for s in range(6)}
        checks = {
            "mean_relative_gain_at_least_1_percent": relative >= cfg["thresholds"]["minimum_relative_gain"],
            "paired_absolute_power_ci95_lower_positive": absolute_ci[0] > 0,
            "each_policy_initialization_positive": all(v > 0 for v in member_gains.values()),
            "each_scorer_seed_positive": all(v > 0 for v in scorer_gains.values()),
            "each_frozen_combination_positive": all(v > 0 for v in combination_gains.values()),
            "each_turbulence_family_positive": all(v > 0 for v in family_gains.values()),
            "each_hardware_slot_positive": all(v > 0 for v in slot_gains.values()),
            "strehl_non_decrease_vs_integrator": deltas["strehl"]["vs_integrator"] >= 0,
            "phase_rmse_non_increase_vs_integrator": deltas["phase_rmse"]["vs_integrator"] <= 0,
            "mean_safety_increase_at_most_0_001_vs_integrator": all(deltas[k]["vs_integrator"] <=
                cfg["thresholds"]["maximum_safety_increase"] for k in ("violation", "saturation", "slew")),
        }
        cells[condition] = {"weather_clusters": len(weather), "independent_units": "complete_weather",
            "method_means": {name: {k: float(v[name].mean()) for k, v in means.items()}
                for name in ("integrator", "original", "current")},
            "current_relative_gain_vs_integrator": relative,
            "current_minus_integrator_power": float(advantage.mean()),
            "current_minus_integrator_ci95": absolute_ci,
            "current_relative_gain_ci95": interval(advantage.mean(1), draws, base.mean(1)),
            "original_relative_gain_vs_integrator": float((original - base).mean() / base.mean()),
            "current_minus_original_power": float((current - original).mean()),
            "current_minus_original_ci95_secondary": interval((current - original).mean(1), draws),
            "by_policy_initialization_power_advantage": member_gains,
            "by_scorer_seed_power_advantage": scorer_gains,
            "by_frozen_combination_power_advantage": combination_gains,
            "by_family_power_advantage": family_gains, "by_slot_power_advantage": slot_gains,
            "metric_deltas": deltas, "precomputed_checks_require_audit": checks,
            "precomputed_all_checks_pass": all(checks.values())}
        for j, family in enumerate(source.FAMILIES):
            for slot in range(6):
                idx = j * 6 + slot
                groups.append({"hardware_condition": condition, "family": family, "slot": slot,
                    "weather_count": len(weather), "current_minus_integrator_power": float(advantage[:, idx].mean()),
                    "current_minus_original_power": float((current - original)[:, idx].mean()),
                    "metric_deltas": {k: {"vs_integrator": float((v["current"] - v["integrator"])[:, idx].mean()),
                        "vs_original": float((v["current"] - v["original"])[:, idx].mean())} for k, v in means.items()}})
    return {"status": "FRESH_WEATHER_CONFIRMATION_PRECOMPUTED_REQUIRES_READ_ONLY_AUDIT", "cells": cells,
        "group_table": groups, "both_conditions_precomputed_pass": all(c["precomputed_all_checks_pass"] for c in cells.values()),
        "interval_scope": cfg["statistics"]["interval_scope"], "training_uncertainty_covered": False,
        "interval_is_not_a_claim_that_gain_exceeds_1_percent": True,
        "historical_multiple_attempts_corrected": False, "historical_gate_reclassification": False,
        "seed_aggregation": "average_of_separate_closed_loop_trajectories_not_ensemble_deployment",
        "secondary_increment_vs_original_is_not_a_new_gate": True}


@torch.no_grad()
def run(path: str | Path = CONFIG, *, quick: bool = False, preflight_only: bool = False) -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    cfg, spec, output, device, assets, report = preflight(path, quick=quick)
    if preflight_only:
        return report
    parent, policies, models, splits, model_manifest = assets
    started = time.perf_counter(); created = False; progress = None
    try:
        output.mkdir(parents=True, exist_ok=False); created = True
        (output / "trajectories").mkdir()
        source.write_json(output / "preflight.json", report); source.write_json(output / "config.json", cfg)
        source.write_json(output / "model_manifest.json", model_manifest)
        source.write_json(output / "stream_manifest.json", stream_manifest(spec))
        source.write_json(output / "runtime.json", {"torch": str(torch.__version__), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device), "git": source.safe_git_record(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "allow_tf32": False})
        base, _ = frozen.load_s1_config(source._project_path(parent["environment_config"]))
        base = replace(base, num_modes=21, batch_size=1, episode_length=spec["episode_length"])
        basis, _, _ = frozen.build_action_basis(base, frozen.ActionRepresentation("r5_zernike21", "zernike", 21), device)
        nominal, shifted = source.d10.d9.d8.d2.g2.d1._profile_pairs(parent)
        progress = source.d10.d9.d3.SparseProgress(output, device, spec["episode_length"])
        progress.phase("G2-C1 CUDA 技术冒烟" if quick else "G2-C1 全新天气独立确认", report["batched_environment_steps"])
        rows = []; trace_manifest = []; completed = 0; prefix_checks = 0
        policy_calls = 0; scorer_calls = 0
        with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
            for condition, profiles in zip(source.CONDITIONS, (nominal, shifted), strict=True):
                for seed in stream_manifest(spec)["weather_bases"]:
                    fold = assigned_fold(seed, spec, splits); original_traces = {}
                    for branch in frozen.controller_specs(spec):
                        policy = policies.get(branch["member"])
                        selector = None if branch["scorer_seed"] is None else models[fold, branch["scorer_seed"]]
                        part, trace = frozen.rollout(seed, branch, condition, profiles, spec, base, basis, parent, policy, selector, progress)
                        policy_calls += spec["episode_length"] if policy is not None else 0
                        scorer_calls += part[0]["selector_calls"]
                        if branch["member"] is not None and selector is None:
                            original_traces[branch["member"]] = trace
                        if selector is not None:
                            frozen.require_prefix(original_traces[branch["member"]], trace, spec["selector_start_step"])
                            prefix_checks += 1
                        name = f"batch_{completed:04d}.pt"; trace_path = output / "trajectories" / name
                        trace.update(stage=cfg["stage"], scorer_fold=fold if selector else None)
                        torch.save(trace, trace_path)
                        trace_manifest.append({"file": name, "sha256": source._file_sha256(trace_path),
                            "condition": condition, "weather_seed": seed, **branch, "scorer_fold": fold if selector else None})
                        for row in part:
                            row.update(trajectory_file=name, scorer_fold=fold if selector else None)
                            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        handle.flush(); rows.extend(part); completed += 1
        if (len(rows) != report["complete_episodes"] or completed != report["episode_batches"]
                or progress.bar.n != report["batched_environment_steps"]
                or policy_calls != report["policy_forward_calls"] or scorer_calls != report["scorer_forward_calls"]):
            raise RuntimeError("实际确认预算不符")
        analysis = summarize(rows, cfg, device=device, quick=quick)
        source.write_json(output / "trajectory_manifest.json", trace_manifest)
        result = {**report, "status": "QUICK_SMOKE_NO_SCIENTIFIC_CONCLUSION" if quick else "INDEPENDENT_CONFIRMATION_REQUIRES_READ_ONLY_AUDIT",
            "confirmation_access": not quick, "independent_confirmation": not quick,
            "weather_scope": "technical_only_not_confirmation" if quick else "fresh_confirmation_weather_generated_once",
            "completed_episodes": len(rows), "completed_episode_batches": completed, "failed_episodes": 0,
            "exact_original_prefix_checks": prefix_checks, "completed_policy_forward_calls": policy_calls,
            "completed_scorer_forward_calls": scorer_calls, "analysis": analysis,
            "elapsed_seconds": time.perf_counter() - started,
            "decision_latency_scope": "CUDA batch-18 policy and selector; excludes projection/optics/I-O; no real-time hardware claim",
            "material_passport": {"origin_skill": "academic-research-suite / experiment-agent", "origin_mode": "run",
                "origin_date": datetime.now(timezone.utc).date().isoformat(), "verification_status": "UNVERIFIED", "version_label": "g2_c1_confirmation_v1"},
            "next_action": "停止等待只读审计；不重跑、不换种子补考、不调参数、不改历史判定"}
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
    parser.add_argument("--quick", action="store_true", help="32 帧独立技术天气检查，不生成确认结论")
    parser.add_argument("--preflight-only", action="store_true", help="只读预检，无模型前向/物理转移/输出创建")
    args = parser.parse_args()
    print(json.dumps(run(args.config, quick=args.quick, preflight_only=args.preflight_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
