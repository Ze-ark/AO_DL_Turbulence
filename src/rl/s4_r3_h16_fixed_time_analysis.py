"""A9冻结模型评估与完整回合配对统计。"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from src.rl.s4_r3_h16_fixed_time_contract import ARMS, BOUNDARY, budget
from src.rl.s4_r3_h16_fixed_time_training import (
    FixedTimeProbe, episode_groups, metrics, predictions, write_rows,
)
from src.rl.s4_r3_h16_gradient_conflict import _episode_gradient, _model_digest
from src.rl.s4_r3_h16_reward_pairwise_probe import RewardPairDataset, _runtime_fields
from src.rl.s4_training import _file_sha256, _project_path, _relative, json_safe
from src.training_progress import counted_progress, update_progress

REWARD_ARMS = ARMS[:3]
PRIMARY_IDS = ("shared_mae_deterioration", "split_attenuation", "regression_only_attenuation")


def error_changes(means: np.ndarray, *, floor: float) -> np.ndarray:
    """末两轴为初始化、组、时点中的时点；返回各初始化相对变化。"""
    if means.shape[-1] != 2 or not np.isfinite(means).all() or (means < 0).any():
        raise ValueError("A9 invalid error tensor")
    valid = means[..., 0] > floor
    result = np.full(means.shape[:-1], np.nan)
    np.divide(means[..., 1], means[..., 0], out=result, where=valid)
    return result - 1


def primary_comparisons(errors: np.ndarray, training_errors: np.ndarray,
                        episode_keys: list[tuple[str, int]], s: dict) -> tuple[list[dict], dict]:
    """数组轴为策略×初始化×三种收益组×早晚时点×完整回合。"""
    n = len(episode_keys)
    shape = (len(s["policy_seeds"]), len(s["replicate_indices"]), 3, 2)
    if errors.shape != (*shape, n) or training_errors.shape[:4] != shape or training_errors.ndim != 5:
        raise ValueError("A9 primary tensor coverage mismatch")
    if len(set(episode_keys)) != n or len({k[1] for k in episode_keys}) != n:
        raise ValueError("A9 duplicate episode key")
    if not np.isfinite(errors).all() or not np.isfinite(training_errors).all():
        raise ValueError("A9 non-finite errors")
    cfg = s["statistics"]
    point_per_init = error_changes(errors.mean(-1), floor=cfg["minimum_mae_denominator"])
    train_per_init = error_changes(training_errors.mean(-1), floor=cfg["minimum_mae_denominator"])
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    strata: dict[str, list[int]] = defaultdict(list)
    for i, (family, _) in enumerate(episode_keys):
        strata[family].append(i)
    # 同一回合抽样权重一次生成，跨演员、组别、初始化与时点完整保留配对。
    weights = np.zeros((cfg["bootstrap_replicates"], n), dtype=np.float64)
    for indices in strata.values():
        weights[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1/len(indices)), size=len(weights))
    weights /= n
    boot_means = np.tensordot(weights, errors, axes=([1], [-1]))
    boot_per_init = error_changes(boot_means, floor=cfg["minimum_mae_denominator"])
    boot_d = boot_per_init.mean(axis=2)
    point = point_per_init.mean(axis=1)
    rows = []
    for p, policy in enumerate(s["policy_seeds"]):
        for j, name in enumerate(PRIMARY_IDS):
            value = point[p, 0] if j == 0 else point[p, 0]-point[p, j]
            draws = boot_d[:, p, 0] if j == 0 else boot_d[:, p, 0]-boot_d[:, p, j]
            per_init = point_per_init[p, :, 0] if j == 0 else point_per_init[p, :, 0]-point_per_init[p, :, j]
            evaluable = bool(np.isfinite(value) and np.isfinite(draws).all())
            tail = cfg["familywise_alpha"]/(2*cfg["family_size"])
            quantiles = np.quantile(draws, [0.025, 0.975, tail, 1-tail]) if evaluable else [None]*4
            rows.append(dict(policy_seed=policy, metric=name, status="OK" if evaluable else "NOT_EVALUABLE_ZERO_DENOMINATOR",
                             estimate=float(value) if evaluable else None, ci95_low=quantiles[0], ci95_high=quantiles[1],
                             familywise_ci_low=quantiles[2], familywise_ci_high=quantiles[3],
                             per_initialization=[float(x) if np.isfinite(x) else None for x in per_init],
                             training_shared_relative_change=[float(x) if np.isfinite(x) else None for x in train_per_init[p, :, 0]],
                             replicates=len(weights), episodes=n, family_size=cfg["family_size"],
                             familywise_alpha=cfg["familywise_alpha"], checkpoint_paired=True,
                             splits_paired=False, unit="complete_episode_seed", stratified_by="condition_family"))
    return rows, interpret(rows, s)


def interpret(rows: list[dict], s: dict) -> dict:
    expected = {(p, m) for p in s["policy_seeds"] for m in PRIMARY_IDS}
    if len(rows) != len(expected) or {(r["policy_seed"], r["metric"]) for r in rows} != expected:
        raise RuntimeError("A9 missing or duplicate primary comparisons")
    cfg = s["statistics"]
    supported: dict[str, list[int]] = {m: [] for m in PRIMARY_IDS}
    def enough(values: list[Any], predicate: Any) -> bool:
        return sum(x is not None and predicate(x) for x in values) >= cfg["minimum_direction_agreeing_initializations"]
    for policy in s["policy_seeds"]:
        group = {r["metric"]: r for r in rows if r["policy_seed"] == policy}
        main = group[PRIMARY_IDS[0]]
        train = main["training_shared_relative_change"]
        simultaneous = sum(t is not None and v is not None and t < 0 and v > 0
                           for t, v in zip(train, main["per_initialization"], strict=True))
        main_ok = (main["status"] == "OK" and main["familywise_ci_low"] > .05
                   and all(t is not None for t in train) and float(np.mean(train)) <= -cfg["minimum_training_relative_mae_decrease"]
                   and simultaneous >= cfg["minimum_direction_agreeing_initializations"])
        if main_ok:
            supported[PRIMARY_IDS[0]].append(policy)
        for name in PRIMARY_IDS[1:]:
            row = group[name]
            if main_ok and row["status"] == "OK" and row["familywise_ci_low"] > .05 and enough(row["per_initialization"], lambda x: x > 0):
                supported[name].append(policy)
    yes = len(supported[PRIMARY_IDS[0]]) >= cfg["minimum_supporting_policy_seeds"]
    return dict(status="QUICK_SMOKE_ONLY" if s["quick"] else (
        "FIXED_TIME_GENERALIZATION_DEGRADATION_SUPPORTED" if yes else "FIXED_TIME_GENERALIZATION_DEGRADATION_NOT_CONFIRMED"),
        supporting_policy_seeds=supported,
        attenuation_supported={m: not s["quick"] and len(supported[m]) >= cfg["minimum_supporting_policy_seeds"] for m in PRIMARY_IDS[1:]},
        uncertainty="conditional_on_fixed_training_sets_and_initializations",
        checkpoint_selection_causality_proven=False, gradient_causality_proven=False,
        full_rl_authorized=False, s4d3_authorized=False, real_hardware_authorized=False)


def _load_model(record: dict, device: torch.device) -> tuple[FixedTimeProbe, dict]:
    path = _project_path(record["path"])
    if _file_sha256(path) != record["sha256"]:
        raise RuntimeError("A9 model changed after freeze")
    ck = torch.load(path, map_location=device, weights_only=False)
    for key in ("policy_seed", "replicate", "arm", "update", "role", "initialization_seed", "batch_order_seed"):
        if ck[key] != record[key]:
            raise RuntimeError("A9 checkpoint identity mismatch")
    if ck["target_horizon"] != 16 or ck["label_source"] != "empirical_reward_returns" or ck["independent_confirmation_used_for_selection"]:
        raise RuntimeError("A9 checkpoint evidence boundary changed")
    if any(ck[k] != v for k, v in BOUNDARY.items()):
        raise RuntimeError("A9 checkpoint update boundary changed")
    if not all(bool(torch.isfinite(t).all()) for t in ck["probe"].values()):
        raise RuntimeError("A9 non-finite checkpoint")
    norms = ck["normalization"]
    if any(not bool(torch.isfinite(t).all()) for t in norms.values()) or any(not bool((norms[k] > 0).all()) for k in ("feature_scale", "target_scale")):
        raise RuntimeError("A9 invalid normalization")
    model = FixedTimeProbe(ck["config"]["feature_size"], ck["config"]["hidden_sizes"][0], ck["arm"]).to(device)
    model.load_state_dict(ck["probe"])
    model.eval()
    return model, ck


def evaluate_all(inventory: list[dict], data: dict[int, dict[str, RewardPairDataset]],
                 s: dict, output: Path, device: torch.device) -> dict:
    endpoints, selected, subgroups, integrity = [], [], [], []
    errors = {}
    keys = {}
    progress_start = time.perf_counter()
    bar = counted_progress(total=len(inventory)*4, description="A9 固定模型评估", unit="组")
    grad_bar = counted_progress(total=budget(s)["gradient_records"], description="A9 回合梯度", unit="回合")
    gradient_rows = []
    with (output / "evaluation_progress.jsonl").open("w", encoding="utf-8") as log, (output / "gradient_progress.jsonl").open("w", encoding="utf-8") as glog:
        for rec in inventory:
            model, ck = _load_model(rec, device)
            digest = _model_digest(model)
            identity = {k: rec[k] for k in ("policy_seed", "replicate", "arm", "role", "update")}
            for split_name, dataset in data[rec["policy_seed"]].items():
                value, score = predictions(model, dataset, ck["normalization"], device)
                row = dict(**identity, split=split_name, **metrics(value, score, dataset, ck["training_reward_mean"]))
                (endpoints if rec["role"] == "fixed" else selected).append(row)
                prediction_path = _project_path(rec["path"]).with_name(f"predictions_{rec['role']}_{rec['update']:05d}_{split_name}.pt")
                torch.save(dict(**identity, split=split_name, reward_prediction=value, ranking_logit=score,
                                reward_delta=dataset.reward_delta, power_delta=dataset.power_delta, rows=dataset.rows), prediction_path)
                groups = episode_groups(dataset)
                if value is not None and rec["role"] == "fixed" and rec["update"] in (s["primary_early_update"], s["primary_late_update"]) and split_name in ("training", "confirmation_id"):
                    key = rec["policy_seed"], rec["replicate"], rec["arm"], rec["update"], split_name
                    error = (value.double()-dataset.reward_delta.double()).abs()
                    errors[key] = np.array([float(error[ix].mean()) for _, _, ix in groups])
                    keys[key] = [(c.split("_")[-1], seed) for c, seed, _ in groups]
                for field in ("condition_id", "profile_id", "probe_step"):
                    for group in sorted({r[field] for r in dataset.rows}):
                        ix = [i for i, r in enumerate(dataset.rows) if r[field] == group]
                        part = RewardPairDataset(dataset.features[ix], dataset.reward_delta[ix], dataset.power_delta[ix], [dataset.rows[i] for i in ix])
                        subgroups.append(dict(**identity, split=split_name, group_kind=field, group_id=group,
                                              **metrics(None if value is None else value[ix], score[ix], part, ck["training_reward_mean"])))
                if rec["role"] == "fixed" and rec["arm"] in ("shared", "split") and split_name != "selection":
                    for condition, episode, ix in groups:
                        grow = dict(**identity, split=split_name, condition_id=condition, episode_seed=episode, pairs=len(ix))
                        try:
                            relation = _episode_gradient(model, dataset, ix, ck,
                                positive_weight=torch.tensor(ck["positive_class_weight"], device=device),
                                classification_weight=.25, huber_delta=s["huber_delta"], device=device)
                            grow.update(status="OK", **relation)
                        except RuntimeError as exc:
                            if "zero shared-trunk gradient" not in str(exc):
                                raise
                            grow.update(status="UNDEFINED_ZERO_NORM", gradient_cosine=None)
                        gradient_rows.append(grow)
                        grad_bar.update(1)
                        glog.write(json.dumps(json_safe(dict(**grow, completed=grad_bar.n, total=grad_bar.total,
                            **_runtime_fields(start=progress_start, completed=grad_bar.n, total=grad_bar.total, device=device))), allow_nan=False)+"\n")
                    glog.flush()
                    update_progress(grad_bar, device=device, metrics={"已算回合": grad_bar.n})
                bar.update(1)
                log.write(json.dumps(json_safe(dict(**identity, split=split_name, completed=bar.n, total=bar.total,
                          **_runtime_fields(start=progress_start, completed=bar.n, total=bar.total, device=device))), allow_nan=False)+"\n")
                log.flush()
                update_progress(bar, device=device, metrics={"已评估": bar.n})
            after = _model_digest(model)
            untouched = all(p.grad is None for p in model.parameters())
            if digest != after or not untouched:
                raise RuntimeError("A9 evaluation modified model")
            integrity.append(dict(**identity, before=digest, after=after, parameters_unchanged=True, grad_fields_untouched=True))
    bar.close(); grad_bar.close()
    b = budget(s)
    if (len(endpoints), len(selected), len(gradient_rows)) != (b["endpoint_records"], b["selected_records"], b["gradient_records"]):
        raise RuntimeError("A9 evaluation record coverage mismatch")
    write_rows(output / "endpoint_metrics.csv", endpoints)
    write_rows(output / "selected_metrics.csv", selected)
    write_rows(output / "subgroup_metrics.csv", subgroups)
    write_rows(output / "episode_gradient_metrics.csv", gradient_rows)
    arrays = {}
    reference_keys = {}
    for split_name in ("training", "confirmation_id"):
        vectors = []
        for p in s["policy_seeds"]:
            rv = []
            for r in s["replicate_indices"]:
                av = []
                for arm in REWARD_ARMS:
                    tv = []
                    for t in (s["primary_early_update"], s["primary_late_update"]):
                        key = p, r, arm, t, split_name
                        if split_name not in reference_keys:
                            reference_keys[split_name] = keys[key]
                        if keys[key] != reference_keys[split_name]:
                            raise RuntimeError("A9 paired model episode keys differ")
                        tv.append(errors[key])
                    av.append(tv)
                rv.append(av)
            vectors.append(rv)
        arrays[split_name] = np.array(vectors)
    comparisons, decision = primary_comparisons(arrays["confirmation_id"], arrays["training"], reference_keys["confirmation_id"], s)
    write_rows(output / "primary_comparisons.csv", comparisons)
    return dict(endpoint_records=len(endpoints), selected_records=len(selected), gradient_records=len(gradient_rows),
                undefined_gradient_records=sum(r["status"] != "OK" for r in gradient_rows),
                parameter_integrity=integrity, comparisons=comparisons, interpretation=decision)
