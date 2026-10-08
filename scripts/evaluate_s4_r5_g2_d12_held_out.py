"""G2-D12-C：冻结末次权重、完整天气折外候选评价；不训练或重算光学轨迹。"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_s4_r5_g2_d12_candidate_scorer as training
from src.runtime import resolve_device

CONFIG = "configs/experiments/s4_r5_g2_d12_held_out_evaluation_v1.yaml"
TRAINING_ENTRY_SHA256 = "1cecdabe52966e3a750b9fcd1295d074d3ff344d8aea434d2081fc3ef09aa3a9"
TRAINING_CONFIG_SHA256 = "402c04bcfb2817689bf34cd4ebc6d104ecef447b5944e3f0af984993c263e05d"
TRAINING_HASHES = {
    False: {"summary": "07aac07012924084a4a0976e520f4ad2034986559357da885d9d355730890312",
        "progress": "e803e3473a5b108c5f6dfcfc4270c7a1e5bcf3d2e3b504ceafc06bd54c95d1cb",
        "splits": "752cf531c40275da2f3faa692cae473dd1f8a11387a95df7b1a0164acdc85984",
        "config": "eeed7a0b7a281ee966561221ad9120bc9bae48ef016940e8a5684a51c3bc0a59"},
    True: {"summary": "5f9429812605bed81e03e789684a745d1d48d77e80ab3a9ebafd336f434b6b2e",
        "progress": "09eb7ef51a32d8a011d69ab36c58495d992d602ed66f6b94e9bcb801fc4675f4",
        "splits": "9ad4b6d94f73cbdfd77f3d97c539603a07cf9d03a1f7fd608061e1afbc3af615",
        "config": "eeed7a0b7a281ee966561221ad9120bc9bae48ef016940e8a5684a51c3bc0a59"},
}
METHODS = ("original", "training_constant", "current", "history", "truth_oracle_reference")
PAIRS = ((2, 0), (2, 1), (3, 0), (3, 1), (3, 2))


def contract(cfg: dict) -> None:
    expected = {"stage": "S4-D2-R5-G2-D12-C",
        "purpose": "frozen_last_checkpoint_weather_held_out_local_evaluation",
        "runtime": {"device": "cuda", "formal_owner": "user_ide", "automatic_retry": False},
        "training_config": training.CONFIG,
        "inference": {"batch_size": 128, "checkpoint": "last", "seed_aggregation": "mean_of_individual_choices", "future_safety_filter": False},
        "statistics": {"weather_bootstrap_seed": 7565000, "weather_bootstrap_repeats": 5000,
            "positive_delta_epsilon": 1e-7, "safety_tolerance": .001,
            "interval_scope": "descriptive_conditional_on_frozen_crossfit_models"},
        "methods": list(METHODS), "quick": {"batch_size": 8, "scientific_analysis": False},
        "output_directory": "outputs/s4_r5_g2_d12_held_out_evaluation_v1",
        "quick_directory": "outputs/s4_r5_g2_d12_held_out_evaluation_v1_quick",
        "boundary": {"training_updates": 0, "new_environment_transitions": 0,
            "confirmation_access": False, "real_slm_actions": False, "independent_confirmation": False,
            "gate_reclassification": False, "automatic_retry": False}}
    if cfg != expected:
        raise ValueError("G2-D12-C 冻结评价合同变化")


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_training(quick: bool) -> tuple[Path, dict, dict]:
    src = training.pairing.source
    if (src._file_sha256(Path(training.__file__)) != TRAINING_ENTRY_SHA256
            or src._file_sha256(src._project_path(training.CONFIG)) != TRAINING_CONFIG_SHA256):
        raise RuntimeError("冻结 B 代码或配置变化")
    cfg = src._load_yaml(src._project_path(training.CONFIG)); training.contract(cfg)
    root = src._project_path(cfg["quick_directory" if quick else "output_directory"])
    if (root / "failure.json").exists():
        raise RuntimeError("B 训练存在失败标记")
    success = read_json(root / "SUCCESS.json")
    for key, digest in TRAINING_HASHES[quick].items():
        filename = key + (".jsonl" if key == "progress" else ".json")
        if src._file_sha256(root / filename) != digest or success.get(key + "_sha256") != digest:
            raise RuntimeError(f"冻结 B 证据变化: {filename}")
    summary = read_json(root / "summary.json")
    if (read_json(root / "config.json") != cfg or summary["quick"] != quick
            or summary["completed_updates"] != (4 if quick else 24000)
            or summary["model_fits"] != (2 if quick else 24) or summary["analysis"] != {}
            or summary["held_out_evaluation"] or summary["confirmation_access"] or summary["real_slm_actions"]
            or summary["entry_sha256"] != TRAINING_ENTRY_SHA256
            or summary["config_sha256"] != TRAINING_CONFIG_SHA256
            or summary["dataset_hashes"] != training.DATA_HASHES[quick]):
        raise RuntimeError("B 训练完成性或边界不符")
    actual = {p.name for p in (root / "checkpoints").iterdir()}
    if actual != set(summary["checkpoint_hashes"]) or len(actual) != (2 if quick else 96):
        raise RuntimeError("B 检查点集合不完整")
    for name, digest in summary["checkpoint_hashes"].items():
        if src._file_sha256(root / "checkpoints" / name) != digest:
            raise RuntimeError(f"B 检查点变化: {name}")
    return root, cfg, summary


def evaluation_indices(split: dict, rows: list[dict], *, quick: bool) -> list[int]:
    indices = split["train_indices" if quick else "held_out_indices"]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("评价索引为空或重复")
    if not quick:
        if set(indices) & set(split["train_indices"]):
            raise ValueError("训练与评价状态重叠")
        for i in indices:
            row = rows[i]
            if (row["weather_fold"] != split["fold"] or row["weather_seed"] not in split["held_out_weather"]
                    or row["weather_seed"] in split["train_weather"]):
                raise ValueError("跨折使用已见天气模型")
    return indices


def validate_checkpoint(ck: dict, split: dict, kind: str, seed: int, *, quick: bool) -> None:
    if (ck["kind"] != kind or ck["seed"] != seed or ck["fold"] != split["fold"]
            or ck["update"] != (2 if quick else 1000) or ck["held_out_evaluation"]
            or ck["train_weather"] != split["train_weather"] or ck["held_out_weather"] != split["held_out_weather"]
            or ck["target_scale"] != 10000. or ck["feature_columns"] != list(training.FEATURE_COLUMNS)
            or ck["candidate_order"] != list(training.pairing.source.CANDIDATES)
            or ck["dataset_hashes"] != training.DATA_HASHES[quick]
            or type(ck["training_only_constant_candidate"]) is not int
            or not 0 <= ck["training_only_constant_candidate"] < 25):
        raise ValueError("检查点折、输入、末次预算或训练来源不符")
    normalizer = ck["normalizer"]
    if set(normalizer) != {"mean", "std", "command_mean", "command_std"}:
        raise ValueError("标准化字段错误")
    for key, width in (("mean", 76), ("std", 76), ("command_mean", 11), ("command_std", 11)):
        value = normalizer[key]
        if (value.shape != (width,) or not bool(torch.isfinite(value).all())
                or ("std" in key and not bool((value > 0).all()))):
            raise ValueError("标准化参数非有限或形状错误")


def forward_causal(model, x: torch.Tensor, valid: torch.Tensor, command: torch.Tensor,
                   normalizer: dict) -> torch.Tensor:
    """只接收因果张量；签名中没有元数据、未来功率或安全标签。"""
    nx, nc = training.normalize(x, valid, command, normalizer)
    return model(nx, valid, nc)


def method_choices(model_choices: torch.Tensor, constant: torch.Tensor,
                   power_labels: torch.Tensor) -> torch.Tensor:
    n, _, seeds = model_choices.shape
    if (model_choices.shape != (n, 2, seeds) or constant.shape != (n,) or power_labels.shape != (n, 25)
            or model_choices.dtype != torch.long or constant.dtype != torch.long
            or bool(((model_choices < 0) | (model_choices >= 25)).any())
            or bool(((constant < 0) | (constant >= 25)).any())
            or not bool(torch.isfinite(power_labels).all())):
        raise ValueError("候选选择网格错误")
    # 只有最后一个不可部署参照利用标签选动作；绝不更改此前模型的选择。
    return torch.stack((constant.new_zeros(n, seeds), constant[:, None].expand(n, seeds),
        model_choices[:, 0], model_choices[:, 1], power_labels.argmax(-1)[:, None].expand(n, seeds)), 1)


def score_choices(choices: torch.Tensor, targets: dict, *, tolerance: float) -> dict:
    n, methods, seeds = choices.shape
    if choices.dtype != torch.long or methods != 5 or bool(((choices < 0) | (choices >= 25)).any()):
        raise ValueError("评分选择格式错误")
    for key, shape in (("power_delta", (n, 25)), ("measured_power_delta", (n, 25)),
                       ("safety_maxima", (n, 25, 3)), ("safety_pass", (n, 25))):
        if targets[key].shape != shape or not bool(torch.isfinite(targets[key]).all()):
            raise ValueError("评分标签格式错误")
    safety = targets["safety_maxima"]
    if (bool(((safety < 0) | (safety > 1)).any()) or targets["safety_pass"].dtype != torch.bool
            or not torch.equal(targets["safety_pass"], (safety <= safety[:, :1] + tolerance).all(-1))):
        raise ValueError("相对安全标签不一致")
    index = torch.arange(n, device=choices.device)[:, None, None]
    power = targets["power_delta"][index, choices]
    chosen_safety = safety[index, choices]
    return {"choices": choices, "power_delta": power,
        "measured_power_delta": targets["measured_power_delta"][index, choices],
        "safety_maxima": chosen_safety, "safety_delta": chosen_safety - safety[:, 0, None, None],
        "safety_pass": targets["safety_pass"][index, choices],
        "oracle_regret": targets["power_delta"].amax(-1)[:, None, None] - power}


def metrics(part: dict, method: int, epsilon: float) -> dict:
    power = part["power_delta"][:, method]
    delta = part["safety_delta"][:, method]
    return {"mean_power_delta": float(power.mean()),
        "mean_measured_power_delta": float(part["measured_power_delta"][:, method].mean()),
        "mean_oracle_regret": float(part["oracle_regret"][:, method].mean()),
        "positive_power_fraction_state_seed": float((power > epsilon).double().mean()),
        "negative_power_fraction_state_seed": float((power < -epsilon).double().mean()),
        "original_action_fraction_state_seed": float((part["choices"][:, method] == 0).double().mean()),
        "relative_safety_pass_fraction_state_seed": float(part["safety_pass"][:, method].double().mean()),
        "safety_delta_maxima": delta.amax((0, 1)).tolist(),
        "safety_absolute_maxima": part["safety_maxima"][:, method].amax((0, 1)).tolist()}


def descriptive_interval(values: torch.Tensor, draws: torch.Tensor) -> dict:
    if values.ndim != 1 or len(values) != 8 or not bool(torch.isfinite(values).all()):
        raise ValueError("描述区间必须使用八个天气均值")
    if draws.ndim != 2 or draws.shape[1] != 8 or bool(((draws < 0) | (draws >= 8)).any()):
        raise ValueError("天气重采样格式错误")
    quantiles = torch.quantile(values[draws].mean(1), values.new_tensor([.025, .975]))
    return {"mean": float(values.mean()), "descriptive_ci95": quantiles.tolist(),
            "positive_weather_count": int((values > 0).sum()), "weather_count": 8}


def summarize(rows: list[dict], evaluation: dict, predictions: torch.Tensor,
              targets: dict, cfg: dict, seeds: list[int]) -> tuple[dict, list, list]:
    device = predictions.device; statistics = cfg["statistics"]
    epsilon = statistics["positive_delta_epsilon"]
    weather = training.pairing.source.stream_manifest(False)["weather_bases"]
    weather_table = []; group_table = []; cells = {}
    generator = torch.Generator(device=device).manual_seed(statistics["weather_bootstrap_seed"])
    draws = torch.randint(8, (statistics["weather_bootstrap_repeats"], 8), generator=generator, device=device)
    subset = lambda ids: {k: v[ids] for k, v in evaluation.items()}
    for condition in training.pairing.source.CONDITIONS:
        ids = torch.tensor([i for i, r in enumerate(rows) if r["hardware_condition"] == condition], device=device)
        if len(ids) != 1296:
            raise ValueError("两条件评价网格不完整")
        part = subset(ids); method_metrics = {name: metrics(part, mi, epsilon) for mi, name in enumerate(METHODS)}
        weather_means = []
        for seed in weather:
            wi = torch.tensor([i for i, r in enumerate(rows) if r["hardware_condition"] == condition and r["weather_seed"] == seed], device=device)
            if len(wi) != 162:
                raise ValueError("每天气状态网格不完整")
            wp = subset(wi)
            weather_means.append(torch.stack((wp["power_delta"].mean((0, 2)), wp["measured_power_delta"].mean((0, 2))), -1))
            weather_table.append({"hardware_condition": condition, "weather_seed": seed, "states": 162,
                "methods": {name: metrics(wp, mi, epsilon) for mi, name in enumerate(METHODS)}})
        means = torch.stack(weather_means)
        comparisons = {f"{METHODS[a]}_vs_{METHODS[b]}": {
            name: descriptive_interval(means[:, a, vi] - means[:, b, vi], draws)
            for vi, name in enumerate(("power_delta", "measured_power_delta"))} for a, b in PAIRS}
        by_seed = []
        for si, seed in enumerate(seeds):
            by_seed.append({"seed": seed, "methods": {name: {
                "mean_power_delta": float(part["power_delta"][:, mi, si].mean()),
                "mean_measured_power_delta": float(part["measured_power_delta"][:, mi, si].mean()),
                "relative_safety_pass_fraction": float(part["safety_pass"][:, mi, si].double().mean())}
                for mi, name in enumerate(METHODS)}})
        calibration = {}
        for ki, kind in enumerate(("current", "history")):
            error = predictions[ids, ki].double() / 10000. - targets["power_delta"][ids, None, 2:]
            calibration[kind] = {"power_delta_rmse": float(error.square().mean().sqrt())}
        oracle = method_metrics["truth_oracle_reference"]["mean_power_delta"]
        cells[condition] = {"states": 1296, "independent_weather_count": 8, "methods": method_metrics,
            "paired_comparisons": comparisons, "by_training_seed": by_seed, "calibration": calibration,
            "oracle_capture_fraction": {kind: method_metrics[kind]["mean_power_delta"] / oracle if oracle > 0 else None
                for kind in ("training_constant", "current", "history")},
            "oracle_capture_null_reason": None if oracle > 0 else "no_positive_oracle_mean_gap"}
        for family in training.pairing.source.FAMILIES:
            for slot in range(6):
                for step in (25, 75, 150):
                    gi = torch.tensor([i for i, r in enumerate(rows) if r["hardware_condition"] == condition
                        and r["family"] == family and r["slot"] == slot and r["probe_step"] == step], device=device)
                    if len(gi) != 24:
                        raise ValueError("分组状态网格不完整")
                    gp = subset(gi)
                    group_table.append({"hardware_condition": condition, "family": family, "slot": slot,
                        "probe_step": step, "states": 24, "weather_count": 8,
                        "methods": {name: metrics(gp, mi, epsilon) for mi, name in enumerate(METHODS)}})
    return {"status": "EXPLORATORY_LOCAL_CROSSFIT_NO_GATE_RECLASSIFICATION", "cells": cells,
        "interval_scope": statistics["interval_scope"], "formal_hypothesis_tests": 0,
        "interval_warning": "仅固定交叉拟合模型下的描述性天气重采样；训练折重叠且未重新拟合，不是独立确认或训练不确定性区间",
        "scope": "single_action_then_held_request_on_frozen_d8_states_not_full_closed_loop",
        "oracle_scope": "truth_selected_candidate_set_only_not_deployable_or_global_upper_bound",
        "seed_aggregation": "mean_of_individual_choice_outcomes_not_best_seed_or_ensemble_selection",
        "continuation_gate_evaluated": False, "prior_d9_gate_reclassified": False}, weather_table, group_table


def preflight(path: str | Path = CONFIG, *, quick: bool = False) -> tuple:
    src = training.pairing.source
    cfg = src._load_yaml(src._project_path(path)); contract(cfg)
    root, parent, summary = verify_training(quick)
    dataset = training.verify_dataset(quick)
    output = src._project_path(cfg["quick_directory" if quick else "output_directory"])
    if output.exists():
        raise FileExistsError(f"保留已有 G2-D12-C 输出，禁止覆盖: {output}")
    device = resolve_device("cuda")
    x, valid, command, labels, splits = training.load_dataset(dataset, device, quick=quick)
    rows = [json.loads(line) for line in (dataset / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    if read_json(root / "splits.json") != splits:
        raise RuntimeError("B 折分与 A 数据不一致")
    seeds = parent["quick" if quick else "training"]["seeds"]
    batch_size = cfg["quick" if quick else "inference"]["batch_size"]
    models = {}; manifest = []; calls = 0
    for split in splits:
        indices = evaluation_indices(split, rows, quick=quick)
        train_ids = torch.tensor(split["train_indices"], device=device)
        expected_norm = training.fit_normalizer(x, valid, command, train_ids)
        constant = int(labels[train_ids].mean(0).argmax())
        for seed in seeds:
            for kind in parent["models"]:
                name = f"fold_{split['fold']}_{kind}_seed_{seed}_{2 if quick else 1000:05d}.pt"
                ck = torch.load(root / "checkpoints" / name, map_location=device, weights_only=True)
                validate_checkpoint(ck, split, kind, seed, quick=quick)
                if (any(not torch.equal(v, expected_norm[k]) for k, v in ck["normalizer"].items())
                        or ck["training_only_constant_candidate"] != constant):
                    raise RuntimeError("标准化或固定方向未严格来源于训练天气")
                model = training.CandidateScorer(kind).to(device)
                model.load_state_dict(ck["state_dict"])
                if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise RuntimeError("检查点权重非有限")
                models[split["fold"], kind, seed] = (model.eval().requires_grad_(False), ck)
                manifest.append({"fold": split["fold"], "kind": kind, "seed": seed,
                    "checkpoint": name, "checkpoint_sha256": summary["checkpoint_hashes"][name],
                    "train_weather": split["train_weather"], "held_out_weather": split["held_out_weather"],
                    "indices": indices, "technical_in_sample_only": quick})
                calls += math.ceil(len(indices) / batch_size)
    report = {"status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_IDE", "quick": quick,
        "device": str(device), "states": len(rows), "last_checkpoints": len(models),
        "model_forward_calls": calls, "model_forward_samples": len(rows) * 2 * len(seeds),
        "record_rows": len(rows) * 2 * len(seeds), "batch_size": batch_size,
        "entry_sha256": src._file_sha256(Path(__file__)), "config_sha256": src._file_sha256(src._project_path(path)),
        "training_hashes": TRAINING_HASHES[quick], "dataset_hashes": training.DATA_HASHES[quick],
        "preflight_model_forward_calls": 0, "scientific_held_out_evaluation_requested": not quick, **cfg["boundary"]}
    return cfg, output, device, (x, valid, command, splits, rows, seeds, dataset), models, manifest, report


@torch.no_grad()
def execute(cfg: dict, output: Path, device: torch.device, data: tuple,
            models: dict, manifest: list, report: dict) -> dict:
    src = training.pairing.source
    x, valid, command, splits, rows, seeds, dataset = data
    n = len(rows); seed_count = len(seeds); batch_size = report["batch_size"]
    predictions = x.new_full((n, 2, seed_count, 23), float("nan"))
    selected = torch.full((n, 2, seed_count), -1, dtype=torch.long, device=device)
    constants = torch.full((n,), -1, dtype=torch.long, device=device)
    started = time.perf_counter(); calls = 0; samples = 0; timings = []
    with (output / "progress.jsonl").open("x", encoding="utf-8") as log:
        for split in splits:
            ids = evaluation_indices(split, rows, quick=report["quick"])
            for ki, kind in enumerate(("current", "history")):
                for si, seed in enumerate(seeds):
                    model, ck = models[split["fold"], kind, seed]
                    constants[ids] = ck["training_only_constant_candidate"]
                    forward_ms = 0.
                    for offset in range(0, len(ids), batch_size):
                        batch = torch.tensor(ids[offset:offset + batch_size], device=device)
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record()
                        pred = forward_causal(model, x[batch], valid[batch], command[batch], ck["normalizer"])
                        end.record(); end.synchronize(); forward_ms += begin.elapsed_time(end)
                        # 固定训练规则，在读取本轮未来评价标签之前完成选择。
                        choice = training.candidate_choice(pred)
                        predictions[batch, ki, si] = pred; selected[batch, ki, si] = choice
                        calls += 1; samples += len(batch); elapsed = time.perf_counter() - started
                        row = {"completed_calls": calls, "total_calls": report["model_forward_calls"],
                            "completed_predictions": samples, "total_predictions": report["model_forward_samples"],
                            "fold": split["fold"], "kind": kind, "seed": seed, "batch_size": len(batch),
                            "elapsed_seconds": elapsed, "eta_seconds": elapsed / calls * (report["model_forward_calls"] - calls),
                            "cuda_allocated_gb": torch.cuda.memory_allocated(device) / 1e9}
                        log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"); log.flush()
                        if calls == 1 or calls % 6 == 0 or calls == report["model_forward_calls"]:
                            print(f"G2-D12-C {calls}/{report['model_forward_calls']} 批次 "
                                f"{samples}/{report['model_forward_samples']}次预测 | {kind} 折{split['fold']} "
                                f"种子{seed} 剩余{row['eta_seconds']:.1f}秒 显存{row['cuda_allocated_gb']:.3f}GB", flush=True)
                    timings.append({"fold": split["fold"], "kind": kind, "seed": seed,
                        "sample_predictions": len(ids), "cuda_batched_inference_ms": forward_ms,
                        "milliseconds_per_prediction_amortized": forward_ms / len(ids),
                        "scope": "batched_indexing_normalization_and_network_only_not_hardware_or_end_to_end_control_latency"})
    if (calls != report["model_forward_calls"] or samples != report["model_forward_samples"]
            or not bool(torch.isfinite(predictions).all()) or bool((selected < 0).any()) or bool((constants < 0).any())):
        raise RuntimeError("折外预测网格或计数未完成")
    torch.save({"predictions_scaled": predictions.cpu(), "model_choices": selected.cpu()}, output / "predictions.pt")
    # 模型选择已冻结，标签仅从这里开始用于本轮结果评分（训练来源审计另只检查训练标签）。
    targets = torch.load(dataset / "targets.pt", map_location=device, weights_only=True)
    choices = method_choices(selected, constants, targets["power_delta"])
    evaluation = score_choices(choices, targets, tolerance=cfg["statistics"]["safety_tolerance"])
    torch.save({k: v.cpu() for k, v in evaluation.items()}, output / "evaluation.pt")
    record_count = 0
    cpu_evaluation = {k: v.cpu() for k, v in evaluation.items()}; cpu_predictions = predictions.cpu(); cpu_selected = selected.cpu()
    with (output / "records.jsonl").open("x", encoding="utf-8") as handle:
        for i, meta in enumerate(rows):
            for ki, kind in enumerate(("current", "history")):
                mi = ki + 2
                for si, seed in enumerate(seeds):
                    choice = int(cpu_selected[i, ki, si])
                    score = 0. if choice < 2 else float(cpu_predictions[i, ki, si, choice - 2]) / 10000.
                    row = {**meta, "model_kind": kind, "training_seed": seed, "selected_candidate_index": choice,
                        "selected_candidate": src.CANDIDATES[choice], "predicted_power_delta": score,
                        "power_delta": float(cpu_evaluation["power_delta"][i, mi, si]),
                        "measured_power_delta": float(cpu_evaluation["measured_power_delta"][i, mi, si]),
                        "safety_maxima": cpu_evaluation["safety_maxima"][i, mi, si].tolist(),
                        "safety_delta": cpu_evaluation["safety_delta"][i, mi, si].tolist(),
                        "relative_safety_pass": bool(cpu_evaluation["safety_pass"][i, mi, si]),
                        "oracle_regret": float(cpu_evaluation["oracle_regret"][i, mi, si]),
                        "future_labels_used_for_model_choice": False, "technical_in_sample_only": report["quick"]}
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"); record_count += 1
    if record_count != report["record_rows"]:
        raise RuntimeError("选择记录未完整保存")
    if report["quick"]:
        analysis, weather_table, group_table = {}, [], []
    else:
        analysis, weather_table, group_table = summarize(rows, evaluation, predictions, targets, cfg, seeds)
    src.write_json(output / "weather_metrics.json", weather_table)
    src.write_json(output / "group_metrics.json", group_table)
    result = {**report, "status": "QUICK_TECHNICAL_ONLY_NO_HELD_OUT_SCIENCE" if report["quick"] else "HELD_OUT_LOCAL_EVALUATION_COMPLETE_REQUIRES_READ_ONLY_AUDIT",
        "completed_model_forward_calls": calls, "completed_model_forward_samples": samples,
        "scientific_held_out_evaluation_completed": not report["quick"],
        "saved_records": record_count, "methods": list(METHODS), "training_seeds": seeds,
        "analysis": analysis, "timings": timings, "elapsed_seconds": time.perf_counter() - started,
        "unassessed_metrics": ["strehl", "phase_rmse", "full_episode_return", "real_hardware"],
        "material_passport": {"origin_skill": "academic-research-suite / experiment-agent", "origin_mode": "run",
            "origin_date": time.strftime("%Y-%m-%d"), "verification_status": "UNVERIFIED", "version_label": "g2_d12_c_evaluation_v1"},
        "next_action": "停止等待只读折外评价审计；不自动挑种子、重训、完整闭环或打开独立确认集"}
    # 明确禁止 JSON 中出现 NaN/Infinity。
    json.dumps(result, allow_nan=False)
    src.write_json(output / "summary.json", result)
    src.write_json(output / "SUCCESS.json", {f"{key}_sha256": src._file_sha256(output / file)
        for key, file in (("summary", "summary.json"), ("progress", "progress.jsonl"), ("records", "records.jsonl"),
            ("predictions", "predictions.pt"), ("evaluation", "evaluation.pt"), ("manifest", "manifest.json"),
            ("weather_metrics", "weather_metrics.json"), ("group_metrics", "group_metrics.json"), ("config", "config.json"))})
    return result


def run(path: str | Path = CONFIG, *, quick: bool = False, preflight_only: bool = False) -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    cfg, output, device, data, models, manifest, report = preflight(path, quick=quick)
    if preflight_only:
        return report
    output.mkdir(parents=True, exist_ok=False)
    src = training.pairing.source
    src.write_json(output / "config.json", cfg); src.write_json(output / "preflight.json", report)
    src.write_json(output / "manifest.json", manifest)
    src.write_json(output / "runtime.json", {"python": platform.python_version(), "torch": str(torch.__version__),
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
        "deterministic_algorithms": True, "tf32": False})
    try:
        return execute(cfg, output, device, data, models, manifest, report)
    except BaseException:
        src.write_json(output / "failure.json", {"error": traceback.format_exc(), "automatic_retry": False})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--quick", action="store_true", help="仅技术集合，无科学汇总")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, quick=args.quick, preflight_only=args.preflight_only), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
