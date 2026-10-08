"""A8-D2：只读诊断训练—验证梯度翻转与可观测分布漂移。"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from src.rl.s4_r3_h16_data_scaling import _read_json, _read_pairs, _verify_hash
from src.rl.s4_r3_h16_head_split import validate_coverage
from src.rl.s4_r3_h16_reward_pairwise_probe import RewardPairDataset
from src.rl.s4_training import (
    _file_sha256,
    _load_yaml,
    _project_path,
    _relative,
    _runtime_record,
    _source_manifest,
    _write_json,
    json_safe,
)
from src.rl.s4_r3_h16_head_split import _git_record_utf8
from src.runtime import resolve_device
from src.training_progress import counted_progress, progress_message, update_progress


ARMS = ("shared", "split")
SPLITS = ("training", "validation")
FAMILIES = ("boiling", "combined", "frozen")
TARGET_FIELDS = (
    "positive_fraction", "reward_mean", "reward_abs_mean", "reward_std",
    "power_mean", "power_abs_mean", "power_std",
)
NO_UPDATES = {
    "checkpoint_loaded": False,
    "gradient_recomputed": False,
    "optimizer_created": False,
    "optimizer_updates": 0,
    "original_critic_updates": 0,
    "actor_updates": 0,
    "alpha_updates": 0,
    "student_updates": 0,
}


def _walk_values(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for nested in value.values():
            yield from _walk_values(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _walk_values(nested)
    else:
        yield value


def _seed_item(mapping: dict[Any, Any], seed: int) -> Any:
    if seed in mapping:
        return mapping[seed]
    if str(seed) in mapping:
        return mapping[str(seed)]
    raise KeyError(seed)


def _condition_family(condition_id: str) -> str:
    matches = [family for family in FAMILIES if condition_id.endswith(family)]
    if len(matches) != 1:
        raise RuntimeError(f"A8-D2 unknown condition family: {condition_id}")
    return matches[0]


def _effective_settings(experiment: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    settings = {
        "quick": quick,
        "policy_seeds": [int(seed) for seed in experiment["design"]["policy_seeds"]],
        "episodes_per_condition": None,
        "bootstrap_replicates": int(experiment["comparison"]["bootstrap_replicates"]),
        "output_directory": experiment["outputs"]["quick_directory" if quick else "directory"],
    }
    if quick:
        settings["policy_seeds"] = [int(seed) for seed in experiment["quick"]["policy_seeds"]]
        settings["episodes_per_condition"] = int(experiment["quick"]["episodes_per_condition"])
        settings["bootstrap_replicates"] = int(experiment["quick"]["bootstrap_replicates"])
    return settings


def _validate_contract(experiment: dict[str, Any]) -> None:
    metadata = experiment["metadata"]
    if metadata["stage"] != "S4-D2-R3-D2-A8-D2" or not metadata["diagnostic_only"]:
        raise RuntimeError("A8-D2 stage or diagnostic contract changed")
    allowed_true = {"diagnostic_only", "post_hoc_mechanism_diagnostic",
                    "allow_saved_gradient_read", "allow_frozen_dataset_read"}
    if not all(bool(metadata[name]) for name in allowed_true):
        raise RuntimeError("A8-D2 required read-only permissions changed")
    for name, value in metadata.items():
        if name.startswith("allow_") and name not in allowed_true and bool(value):
            raise RuntimeError(f"A8-D2 safety flag must remain false: {name}")
    design = experiment["design"]
    if ([int(seed) for seed in design["policy_seeds"]] != [9301, 9302, 9303]
            or tuple(design["arms"]) != ARMS or tuple(design["splits"]) != SPLITS
            or tuple(design["condition_families"]) != FAMILIES
            or tuple(design["target_descriptors"]) != TARGET_FIELDS
            or int(design["feature_size"]) != 221 or int(design["pairs_per_episode"]) != 18):
        raise RuntimeError("A8-D2 frozen data design changed")
    matching = design["matching"]
    if (tuple(matching["methods"]) != ("target", "feature")
            or not matching["within_condition_family"] or not matching["replacement"]
            or float(matching["minimum_unique_match_fraction"]) != 0.5
            or int(matching["maximum_reuse_count"]) != 4
            or float(matching["minimum_gap_closure"]) != 0.5):
        raise RuntimeError("A8-D2 matching contract changed")
    comparison = experiment["comparison"]
    if (int(comparison["bootstrap_replicates"]) != 20000
            or int(comparison["family_size"]) != 24
            or int(comparison["distribution_permutation_family_size"]) != 6
            or float(comparison["familywise_alpha"]) != 0.05
            or float(comparison["meaningful_cosine_shift"]) != 0.2
            or float(comparison["meaningful_standardized_drift"]) != 0.25
            or int(comparison["minimum_supporting_seeds"]) != 2):
        raise RuntimeError("A8-D2 preregistered comparison changed")
    runtime = experiment["runtime"]
    if runtime["device"] != "cuda" or not runtime["require_cuda"] or not runtime["deterministic_algorithms"]:
        raise RuntimeError("A8-D2 CUDA/determinism contract changed")
    if any("independent_test" in str(value).lower() for value in _walk_values(experiment["datasets"])):
        raise RuntimeError("A8-D2 may not reference independent-test samples")


def _read_gradient_rows(path: str | Path) -> list[dict[str, Any]]:
    with _project_path(path).open("r", encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle))
    required = {"policy_seed", "arm", "split", "condition_id", "episode_seed", "pairs",
                "gradient_cosine", "gradient_conflict"}
    if not raw or not required.issubset(raw[0]):
        raise RuntimeError("A8-D2 upstream gradient schema changed")
    rows: list[dict[str, Any]] = []
    for row in raw:
        converted: dict[str, Any] = {}
        for name, value in row.items():
            if name in {"arm", "split", "condition_id"}:
                converted[name] = value
            elif name in {"policy_seed", "episode_seed", "pairs"}:
                converted[name] = int(float(value))
            else:
                converted[name] = float(value)
        converted["condition_family"] = _condition_family(converted["condition_id"])
        rows.append(converted)
    keys = [(row["policy_seed"], row["arm"], row["split"], row["episode_seed"]) for row in rows]
    if len(keys) != len(set(keys)) or any(not math.isfinite(float(row["gradient_cosine"])) for row in rows):
        raise RuntimeError("A8-D2 invalid or duplicated gradient rows")
    return rows


def _episode_groups(data: RewardPairDataset) -> list[tuple[str, int, list[int]]]:
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(data.rows):
        groups[_condition_family(str(row["condition_id"])), int(row["episode_seed"])].append(index)
    result = [(family, seed, indices) for (family, seed), indices in sorted(groups.items())]
    if any(len(indices) != 18 for _, _, indices in result):
        raise RuntimeError("A8-D2 requires 18 complete pairs per episode")
    return result


def _selected_keys(data: RewardPairDataset, episodes_per_condition: int | None) -> set[tuple[str, int]]:
    groups = _episode_groups(data)
    if episodes_per_condition is None:
        return {(family, seed) for family, seed, _ in groups}
    by_family: dict[str, list[int]] = defaultdict(list)
    for family, seed, _ in groups:
        by_family[family].append(seed)
    if any(len(values) < episodes_per_condition for values in by_family.values()):
        raise RuntimeError("A8-D2 quick subset lacks complete episodes")
    return {(family, seed) for family in FAMILIES for seed in sorted(by_family[family])[:episodes_per_condition]}


def _build_descriptors(
    data: RewardPairDataset,
    *,
    policy_seed: int,
    split: str,
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
    selected: set[tuple[str, int]],
    device: torch.device,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    rows: list[dict[str, Any]] = []
    centroids: list[torch.Tensor] = []
    features = data.features.to(device=device, dtype=torch.float64)
    rewards = data.reward_delta.to(device=device, dtype=torch.float64)
    powers = data.power_delta.to(device=device, dtype=torch.float64)
    for family, episode, indices_list in _episode_groups(data):
        if (family, episode) not in selected:
            continue
        indices = torch.tensor(indices_list, device=device, dtype=torch.long)
        reward, power = rewards[indices], powers[indices]
        centroid = ((features[indices] - feature_mean) / feature_scale).mean(dim=0)
        values = {
            "positive_fraction": float((reward > 0).double().mean()),
            "reward_mean": float(reward.mean()),
            "reward_abs_mean": float(reward.abs().mean()),
            "reward_std": float(reward.std(unbiased=False)),
            "power_mean": float(power.mean()),
            "power_abs_mean": float(power.abs().mean()),
            "power_std": float(power.std(unbiased=False)),
        }
        rows.append({"policy_seed": policy_seed, "split": split, "condition_family": family,
                     "episode_seed": episode, "pairs": len(indices_list), **values})
        centroids.append(centroid)
    if not rows:
        raise RuntimeError("A8-D2 descriptor table is empty")
    return rows, torch.stack(centroids)


def _index_by_episode(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, int], int]:
    result = {(str(row["condition_family"]), int(row["episode_seed"])): index for index, row in enumerate(rows)}
    if len(result) != len(rows):
        raise RuntimeError("A8-D2 duplicated descriptor episodes")
    return result


def _quantile_interval(
    distribution: torch.Tensor,
    estimate: float,
    *,
    familywise_alpha: float,
    family_size: int,
    familywise: bool = True,
) -> dict[str, Any]:
    distribution = distribution.detach()
    result = {"estimate": float(estimate),
              "ci95_low": float(torch.quantile(distribution, 0.025)),
              "ci95_high": float(torch.quantile(distribution, 0.975)),
              "replicates": int(distribution.numel())}
    if familywise:
        tail = familywise_alpha / (2 * family_size)
        result.update({"familywise_ci_low": float(torch.quantile(distribution, tail)),
                       "familywise_ci_high": float(torch.quantile(distribution, 1 - tail)),
                       "familywise_alpha": familywise_alpha, "family_size": family_size})
    return result


def _bootstrap_stratified_mean(
    rows: Sequence[dict[str, Any]],
    field: str,
    *,
    replicates: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    by_family: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_family[str(row["condition_family"])].append(float(row[field]))
    if set(by_family) != set(FAMILIES):
        raise RuntimeError("A8-D2 missing a condition family")
    generator = torch.Generator(device=device).manual_seed(seed)
    total_n = sum(len(values) for values in by_family.values())
    result = torch.zeros(replicates, dtype=torch.float64, device=device)
    for family in FAMILIES:
        values = torch.tensor(by_family[family], dtype=torch.float64, device=device)
        indices = torch.randint(len(values), (replicates, len(values)), generator=generator, device=device)
        result += values[indices].mean(dim=1) * (len(values) / total_n)
    return result


def two_sample_interval(
    training: Sequence[dict[str, Any]],
    validation: Sequence[dict[str, Any]],
    field: str,
    *,
    replicates: int,
    seed: int,
    device: torch.device,
    familywise_alpha: float,
    family_size: int,
) -> dict[str, Any]:
    """返回 validation-training 的分层两样本重采样区间。"""
    train_dist = _bootstrap_stratified_mean(training, field, replicates=replicates, seed=seed, device=device)
    val_dist = _bootstrap_stratified_mean(validation, field, replicates=replicates, seed=seed + 1, device=device)
    estimate = float(np.mean([float(row[field]) for row in validation])
                     - np.mean([float(row[field]) for row in training]))
    result = _quantile_interval(val_dist - train_dist, estimate,
                                familywise_alpha=familywise_alpha, family_size=family_size)
    result.update({"unit": "complete_episode_seed", "stratified_by": "condition_family",
                   "comparison": "validation_minus_training", "paired": False})
    return result


def _bootstrap_standardized_shift(
    training: torch.Tensor,
    validation: torch.Tensor,
    training_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
    device: torch.device,
    chunk_size: int = 200,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if training.ndim != 2 or validation.ndim != 2 or training.shape[1] != validation.shape[1]:
        raise ValueError("A8-D2 descriptor matrices must be aligned")
    scale = training.std(dim=0, unbiased=True).clamp_min(1e-12)
    point = (validation.mean(dim=0) - training.mean(dim=0)) / scale
    train_index = _index_by_episode(training_rows)
    val_index = _index_by_episode(validation_rows)
    train_by_family = {family: torch.tensor([index for (name, _), index in train_index.items() if name == family],
                                             dtype=torch.long, device=device) for family in FAMILIES}
    val_by_family = {family: torch.tensor([index for (name, _), index in val_index.items() if name == family],
                                           dtype=torch.long, device=device) for family in FAMILIES}
    generator = torch.Generator(device=device).manual_seed(seed)
    output = torch.empty((replicates, training.shape[1]), dtype=torch.float64, device=device)
    for start in range(0, replicates, chunk_size):
        size = min(chunk_size, replicates - start)
        train_mean = torch.zeros((size, training.shape[1]), dtype=torch.float64, device=device)
        val_mean = torch.zeros_like(train_mean)
        for family in FAMILIES:
            ti, vi = train_by_family[family], val_by_family[family]
            td = torch.randint(len(ti), (size, len(ti)), generator=generator, device=device)
            vd = torch.randint(len(vi), (size, len(vi)), generator=generator, device=device)
            train_mean += training[ti[td]].mean(dim=1) * (len(ti) / len(training))
            val_mean += validation[vi[vd]].mean(dim=1) * (len(vi) / len(validation))
        output[start:start + size] = (val_mean - train_mean) / scale
    score = output.square().mean(dim=1).sqrt()
    return point, output, score


def descriptor_drift(
    training: torch.Tensor,
    validation: torch.Tensor,
    training_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
    *,
    names: Sequence[str] | None,
    replicates: int,
    seed: int,
    device: torch.device,
    familywise_alpha: float,
    family_size: int,
    permutation_family_size: int,
) -> dict[str, Any]:
    point, distribution, scores = _bootstrap_standardized_shift(
        training, validation, training_rows, validation_rows,
        replicates=replicates, seed=seed, device=device,
    )
    score = float(point.square().mean().sqrt())
    permutation = permutation_drift_test(
        training, validation, training_rows, validation_rows,
        observed_score=score, replicates=replicates, seed=seed + 500000,
        device=device, familywise_alpha=familywise_alpha,
        permutation_family_size=permutation_family_size,
    )
    result: dict[str, Any] = {
        "rms_standardized_drift": _quantile_interval(
            scores, score, familywise_alpha=familywise_alpha, family_size=family_size, familywise=False,
        ),
        "dimensions": int(training.shape[1]),
        "max_abs_standardized_drift": float(point.abs().max()),
        "permutation_test": permutation,
    }
    if names is not None:
        if len(names) != training.shape[1]:
            raise ValueError("A8-D2 descriptor names do not align")
        result["descriptors"] = {
            name: _quantile_interval(distribution[:, index], float(point[index]),
                                     familywise_alpha=familywise_alpha, family_size=family_size)
            for index, name in enumerate(names)
        }
    else:
        top = torch.topk(point.abs(), k=min(10, point.numel())).indices.tolist()
        result["largest_feature_dimensions"] = [
            {"index": int(index), "standardized_drift": float(point[index])} for index in top
        ]
    return result


def permutation_drift_test(
    training: torch.Tensor,
    validation: torch.Tensor,
    training_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
    *,
    observed_score: float,
    replicates: int,
    seed: int,
    device: torch.device,
    familywise_alpha: float,
    permutation_family_size: int,
    chunk_size: int = 200,
) -> dict[str, Any]:
    """按条件交换训练/验证标签，校准高维范数固有的向上偏差。"""
    if permutation_family_size < 1 or replicates < 20:
        raise ValueError("A8-D2 invalid permutation settings")
    scale = training.std(dim=0, unbiased=True).clamp_min(1e-12)
    train_index = _index_by_episode(training_rows)
    val_index = _index_by_episode(validation_rows)
    generator = torch.Generator(device=device).manual_seed(seed)
    scores = torch.empty(replicates, dtype=torch.float64, device=device)
    for start in range(0, replicates, chunk_size):
        size = min(chunk_size, replicates - start)
        train_mean = torch.zeros((size, training.shape[1]), dtype=torch.float64, device=device)
        val_mean = torch.zeros_like(train_mean)
        for family in FAMILIES:
            ti = [index for (name, _), index in train_index.items() if name == family]
            vi = [index for (name, _), index in val_index.items() if name == family]
            combined = torch.cat([training[ti], validation[vi]], dim=0)
            order = torch.rand((size, len(combined)), generator=generator, device=device).argsort(dim=1)
            train_draw = combined[order[:, :len(ti)]].mean(dim=1)
            val_draw = combined[order[:, len(ti):]].mean(dim=1)
            train_mean += train_draw * (len(ti) / len(training))
            val_mean += val_draw * (len(vi) / len(validation))
        standardized = (val_mean - train_mean) / scale
        scores[start:start + size] = standardized.square().mean(dim=1).sqrt()
    exceedances = int((scores >= observed_score).sum())
    return {
        "p_value": (exceedances + 1) / (replicates + 1),
        "observed_score": observed_score,
        "null_median": float(torch.quantile(scores, 0.5)),
        "null_95": float(torch.quantile(scores, 0.95)),
        "replicates": replicates,
        "familywise_alpha": familywise_alpha / permutation_family_size,
        "family_size": permutation_family_size,
        "stratified_by": "condition_family",
    }


def nearest_matches(
    training: torch.Tensor,
    validation: torch.Tensor,
    training_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
) -> tuple[dict[tuple[str, int], tuple[str, int]], dict[str, Any]]:
    scale = training.std(dim=0, unbiased=True).clamp_min(1e-12)
    train = training / scale
    val = validation / scale
    mapping: dict[tuple[str, int], tuple[str, int]] = {}
    distances: list[float] = []
    selected_indices: list[int] = []
    for family in FAMILIES:
        ti = [index for index, row in enumerate(training_rows) if row["condition_family"] == family]
        vi = [index for index, row in enumerate(validation_rows) if row["condition_family"] == family]
        matrix = torch.cdist(val[vi], train[ti]) / math.sqrt(training.shape[1])
        nearest = matrix.argmin(dim=1)
        for local_v, local_t in enumerate(nearest.tolist()):
            vrow, trow = validation_rows[vi[local_v]], training_rows[ti[local_t]]
            mapping[(family, int(vrow["episode_seed"]))] = (family, int(trow["episode_seed"]))
            selected_indices.append(ti[local_t])
            distances.append(float(matrix[local_v, local_t]))
    reuse = Counter(selected_indices)
    return mapping, {
        "validation_episodes": len(validation_rows),
        "unique_training_matches": len(reuse),
        "unique_match_fraction": len(reuse) / len(validation_rows),
        "maximum_reuse_count": max(reuse.values()),
        "mean_standardized_distance": float(np.mean(distances)),
        "median_standardized_distance": float(np.median(distances)),
    }


def matched_gap_interval(
    gradient_training: Sequence[dict[str, Any]],
    gradient_validation: Sequence[dict[str, Any]],
    mapping: dict[tuple[str, int], tuple[str, int]],
    *,
    replicates: int,
    seed: int,
    device: torch.device,
    familywise_alpha: float,
    family_size: int,
) -> dict[str, Any]:
    training = {(str(row["condition_family"]), int(row["episode_seed"])): row for row in gradient_training}
    validation = {(str(row["condition_family"]), int(row["episode_seed"])): row for row in gradient_validation}
    if validation.keys() != mapping.keys() or any(key not in training for key in mapping.values()):
        raise RuntimeError("A8-D2 matched gradient episodes do not align")
    rows = [{"condition_family": key[0], "episode_seed": key[1],
             "difference": float(validation[key]["gradient_cosine"])
                           - float(training[mapping[key]]["gradient_cosine"])} for key in sorted(validation)]
    distribution = _bootstrap_stratified_mean(rows, "difference", replicates=replicates, seed=seed, device=device)
    estimate = float(np.mean([row["difference"] for row in rows]))
    result = _quantile_interval(distribution, estimate,
                                familywise_alpha=familywise_alpha, family_size=family_size)
    result.update({"unit": "matched_validation_episode", "stratified_by": "condition_family",
                   "comparison": "validation_minus_matched_training", "paired": "descriptor_matched_not_physical_pair"})
    return result


def interpret_gradient_shift(
    gradient_shifts: Sequence[dict[str, Any]],
    target_drifts: Sequence[dict[str, Any]],
    feature_drifts: Sequence[dict[str, Any]],
    matches: Sequence[dict[str, Any]],
    *,
    policy_seeds: Sequence[int],
    meaningful_cosine_shift: float,
    meaningful_drift: float,
    minimum_supporting_seeds: int,
    minimum_gap_closure: float,
    minimum_unique_fraction: float,
    maximum_reuse: int,
    distribution_permutation_alpha: float,
    quick: bool,
) -> dict[str, Any]:
    if quick:
        return {"status": "QUICK_SMOKE_ONLY", "shared_gradient_reversal_supported": False,
                "target_distribution_associated_candidate": False,
                "feature_distribution_associated_candidate": False,
                "new_model_training_authorized": False, "full_rl_authorized": False,
                "s4d3_authorized": False, "real_hardware_authorized": False}
    shift = {(int(row["policy_seed"]), row["arm"]): row for row in gradient_shifts}
    if shift.keys() != {(seed, arm) for seed in policy_seeds for arm in ARMS}:
        raise RuntimeError("A8-D2 gradient-shift coverage incomplete")
    reversal = all(shift[seed, "shared"]["gradient_cosine_shift"]["familywise_ci_low"] > meaningful_cosine_shift
                   for seed in policy_seeds)
    target = {int(row["policy_seed"]): row for row in target_drifts}
    feature = {int(row["policy_seed"]): row for row in feature_drifts}
    match = {(int(row["policy_seed"]), row["arm"], row["method"]): row for row in matches}

    def associated(kind: str, drifts: dict[int, dict[str, Any]]) -> tuple[bool, list[int]]:
        supporting = []
        for seed in policy_seeds:
            row = match[seed, "shared", kind]
            stable = (row["unique_match_fraction"] >= minimum_unique_fraction
                      and row["maximum_reuse_count"] <= maximum_reuse)
            if (drifts[seed]["rms_standardized_drift"]["estimate"] > meaningful_drift
                    and drifts[seed]["permutation_test"]["p_value"] <= distribution_permutation_alpha
                    and row["absolute_gap_closure"] >= minimum_gap_closure and stable):
                supporting.append(seed)
        return len(supporting) >= minimum_supporting_seeds, supporting

    target_candidate, target_seeds = associated("target", target)
    feature_candidate, feature_seeds = associated("feature", feature)
    if not reversal:
        status = "SHARED_GRADIENT_REVERSAL_NOT_CONFIRMED"
    elif target_candidate and feature_candidate:
        status = "REVERSAL_ASSOCIATED_WITH_TARGET_AND_FEATURE_DISTRIBUTIONS"
    elif target_candidate:
        status = "REVERSAL_ASSOCIATED_WITH_TARGET_DISTRIBUTION"
    elif feature_candidate:
        status = "REVERSAL_ASSOCIATED_WITH_FEATURE_DISTRIBUTION"
    else:
        status = "REVERSAL_NOT_EXPLAINED_BY_OBSERVED_DISTRIBUTIONS"
    return {
        "status": status,
        "shared_gradient_reversal_supported": reversal,
        "target_distribution_associated_candidate": target_candidate,
        "target_supporting_seeds": target_seeds,
        "feature_distribution_associated_candidate": feature_candidate,
        "feature_supporting_seeds": feature_seeds,
        "causal_explanation_authorized": False,
        "new_model_training_authorized": False,
        "full_rl_authorized": False,
        "s4d3_authorized": False,
        "real_hardware_authorized": False,
    }


def preflight_gradient_shift(
    config_path: str | Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    _validate_contract(experiment)
    if not torch.cuda.is_available():
        raise RuntimeError("A8-D2 requires CUDA; CPU fallback is forbidden")
    input_hashes: dict[str, str] = {}
    for section in ("design_contract", "upstream_d1"):
        for name, checksum in experiment[section].items():
            if name.endswith("_sha256"):
                path = experiment[section][name[:-7]]
                _verify_hash(path, str(checksum))
                input_hashes[path] = str(checksum)
    summary = _read_json(experiment["upstream_d1"]["summary"])
    success = _read_json(experiment["upstream_d1"]["success"])
    if (summary["experiment"]["quick"] or summary["interpretation"]["status"] != experiment["upstream_d1"]["required_status"]
            or int(summary["episode_records"]) != int(experiment["upstream_d1"]["required_episode_records"])
            or success["status"] != "DIAGNOSTIC_COMPLETED_PENDING_AUDIT"
            or success["summary_sha256"] != experiment["upstream_d1"]["summary_sha256"]):
        raise RuntimeError("A8-D2 upstream D1 evidence changed")
    if not all(row["parameters_unchanged"] and row["parameter_grad_fields_untouched"] for row in summary["parameter_integrity"]):
        raise RuntimeError("A8-D2 upstream parameter integrity failed")
    gradients = _read_gradient_rows(experiment["upstream_d1"]["episode_metrics"])
    if (len(gradients) != int(experiment["upstream_d1"]["required_episode_records"])
            or any(row["pairs"] != 18 or row["arm"] not in ARMS or row["split"] not in SPLITS
                   or row["policy_seed"] not in (9301, 9302, 9303)
                   or row["gradient_conflict"] != float(row["gradient_cosine"] < 0) for row in gradients)):
        raise RuntimeError("A8-D2 upstream gradient coverage changed")
    dataset_specs: dict[int, dict[str, Any]] = {}
    expected_gradient_keys: set[tuple[int, str, str, int]] = set()
    allowed_episode_keys: dict[tuple[int, str], set[tuple[str, int]]] = {}
    selected_counts: dict[int, dict[str, int]] = {}
    for seed in settings["policy_seeds"]:
        dataset_specs[seed], selected_counts[seed] = {}, {}
        for split in SPLITS:
            spec = dict(_seed_item(experiment["datasets"], seed)[split])
            _verify_hash(spec["path"], spec["sha256"])
            input_hashes[spec["path"]] = spec["sha256"]
            data = _read_pairs(_project_path(spec["path"]))
            validate_coverage(data, episodes=int(spec["episodes"]), pairs=int(spec["pairs"]))
            keys = _selected_keys(data, settings["episodes_per_condition"])
            allowed_episode_keys[seed, split] = keys
            selected_counts[seed][split] = len(keys)
            for arm in ARMS:
                expected_gradient_keys.update((seed, arm, split, episode) for _, episode in keys)
            dataset_specs[seed][split] = spec
    observed = {(row["policy_seed"], row["arm"], row["split"], row["episode_seed"])
                for row in gradients if row["policy_seed"] in settings["policy_seeds"]
                and (row["condition_family"], row["episode_seed"])
                in allowed_episode_keys[row["policy_seed"], row["split"]]}
    if observed != expected_gradient_keys:
        raise RuntimeError("A8-D2 gradient/data episode alignment failed")
    sources = _source_manifest(experiment["tracked_source_files"])
    sources[_relative(_project_path(config_path))] = _file_sha256(_project_path(config_path))
    output = _project_path(settings["output_directory"])
    if output.exists():
        raise FileExistsError(f"A8-D2 output already exists: {output}")
    return {
        "status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_DIAGNOSTIC",
        "quick": quick, "policy_seeds": settings["policy_seeds"],
        "selected_episodes": selected_counts,
        "expected_descriptor_records": sum(sum(values.values()) for values in selected_counts.values()),
        "expected_gradient_shift_comparisons": len(settings["policy_seeds"]) * len(ARMS),
        "expected_matching_comparisons": len(settings["policy_seeds"]) * len(ARMS) * 2,
        "dataset_specs": dataset_specs, "frozen_input_hashes": input_hashes,
        "frozen_source_hashes": sources, "output_directory": _relative(output),
        "independent_test_sample_paths": [], "new_data_generated": False,
        "training_run": False, "full_rl_training": False, "s4d3_access": False,
        "real_slm_actions": False, **NO_UPDATES,
    }


def _write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty rows: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _flatten(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        flat: dict[str, Any] = {}
        for name, value in row.items():
            if isinstance(value, dict) and all(not isinstance(item, dict) for item in value.values()):
                flat.update({f"{name}_{key}": item for key, item in value.items()})
            elif not isinstance(value, dict):
                flat[name] = value
        result.append(flat)
    return result


def _execute(
    experiment: dict[str, Any],
    settings: dict[str, Any],
    preflight: dict[str, Any],
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    gradients = _read_gradient_rows(experiment["upstream_d1"]["episode_metrics"])
    descriptor_rows: dict[int, dict[str, list[dict[str, Any]]]] = {}
    feature_centroids: dict[int, dict[str, torch.Tensor]] = {}
    progress_path = output / "progress.jsonl"
    bar = counted_progress(total=preflight["expected_descriptor_records"], description="A8-D2 分布诊断", unit="回合")
    completed = 0
    with progress_path.open("w", encoding="utf-8") as progress:
        for seed in settings["policy_seeds"]:
            datasets = {split: _read_pairs(_project_path(preflight["dataset_specs"][seed][split]["path"])) for split in SPLITS}
            training_features = datasets["training"].features.to(device=device, dtype=torch.float64)
            feature_mean = training_features.mean(dim=0)
            feature_scale = training_features.std(dim=0, unbiased=True).clamp_min(1e-12)
            descriptor_rows[seed], feature_centroids[seed] = {}, {}
            for split in SPLITS:
                selected = _selected_keys(datasets[split], settings["episodes_per_condition"])
                rows, centroids = _build_descriptors(
                    datasets[split], policy_seed=seed, split=split,
                    feature_mean=feature_mean, feature_scale=feature_scale,
                    selected=selected, device=device,
                )
                descriptor_rows[seed][split], feature_centroids[seed][split] = rows, centroids
                for row in rows:
                    completed += 1
                    elapsed = time.perf_counter() - started
                    record = {"completed_episode_records": completed,
                              "total_episode_records": preflight["expected_descriptor_records"],
                              "policy_seed": seed, "split": split,
                              "condition_family": row["condition_family"], "episode_seed": row["episode_seed"],
                              "estimated_remaining_seconds": elapsed / completed * (preflight["expected_descriptor_records"] - completed),
                              "cuda_allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
                              "cuda_reserved_gb": torch.cuda.memory_reserved(device) / 1024**3}
                    progress.write(json.dumps(record, ensure_ascii=False) + "\n")
                    progress.flush()
                    bar.update(1)
                    update_progress(bar, device=device, metrics={"已完成": completed / preflight["expected_descriptor_records"]})
    bar.close()
    selected_gradient: list[dict[str, Any]] = []
    for seed in settings["policy_seeds"]:
        valid_keys = {split: {(row["condition_family"], row["episode_seed"]) for row in descriptor_rows[seed][split]}
                      for split in SPLITS}
        selected_gradient.extend(row for row in gradients if row["policy_seed"] == seed
                                 and (row["condition_family"], row["episode_seed"]) in valid_keys[row["split"]])
    comparison = experiment["comparison"]
    stat = {"replicates": settings["bootstrap_replicates"], "device": device,
            "familywise_alpha": float(comparison["familywise_alpha"]),
            "family_size": int(comparison["family_size"])}
    drift_stat = {**stat, "permutation_family_size": int(comparison["distribution_permutation_family_size"])}
    gradient_shifts: list[dict[str, Any]] = []
    condition_shifts: list[dict[str, Any]] = []
    for seed in settings["policy_seeds"]:
        for arm in ARMS:
            by_split = {split: [row for row in selected_gradient if row["policy_seed"] == seed
                                and row["arm"] == arm and row["split"] == split] for split in SPLITS}
            interval = two_sample_interval(by_split["training"], by_split["validation"], "gradient_cosine",
                                           seed=int(comparison["bootstrap_seed"]) + seed * 10 + ARMS.index(arm), **stat)
            gradient_shifts.append({"policy_seed": seed, "arm": arm, "gradient_cosine_shift": interval,
                                    "training_mean": float(np.mean([row["gradient_cosine"] for row in by_split["training"]])),
                                    "validation_mean": float(np.mean([row["gradient_cosine"] for row in by_split["validation"]]))})
            for family in FAMILIES:
                train = [row["gradient_cosine"] for row in by_split["training"] if row["condition_family"] == family]
                val = [row["gradient_cosine"] for row in by_split["validation"] if row["condition_family"] == family]
                condition_shifts.append({"policy_seed": seed, "arm": arm, "condition_family": family,
                                         "training_mean": float(np.mean(train)), "validation_mean": float(np.mean(val)),
                                         "validation_minus_training": float(np.mean(val) - np.mean(train))})
    target_drifts: list[dict[str, Any]] = []
    feature_drifts: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    progress_message("A8-D2 阶段2/3：重采样与条件内置换检验")
    drift_bar = counted_progress(total=len(settings["policy_seeds"]) * 2,
                                 description="A8-D2 漂移检验", unit="复合指标")
    for seed in settings["policy_seeds"]:
        train_rows, val_rows = descriptor_rows[seed]["training"], descriptor_rows[seed]["validation"]
        target_train = torch.tensor([[float(row[name]) for name in TARGET_FIELDS] for row in train_rows],
                                    dtype=torch.float64, device=device)
        target_val = torch.tensor([[float(row[name]) for name in TARGET_FIELDS] for row in val_rows],
                                  dtype=torch.float64, device=device)
        target = descriptor_drift(target_train, target_val, train_rows, val_rows, names=TARGET_FIELDS,
                                  seed=int(comparison["bootstrap_seed"]) + 100000 + seed, **drift_stat)
        target_drifts.append({"policy_seed": seed, **target})
        drift_bar.update(1)
        update_progress(drift_bar, device=device,
                        metrics={"置换": drift_bar.n / (len(settings["policy_seeds"]) * 2)})
        feature = descriptor_drift(feature_centroids[seed]["training"], feature_centroids[seed]["validation"],
                                   train_rows, val_rows, names=None,
                                   seed=int(comparison["bootstrap_seed"]) + 200000 + seed, **drift_stat)
        feature_drifts.append({"policy_seed": seed, **feature})
        drift_bar.update(1)
        update_progress(drift_bar, device=device,
                        metrics={"置换": drift_bar.n / (len(settings["policy_seeds"]) * 2)})
        mappings = {
            "target": nearest_matches(target_train, target_val, train_rows, val_rows),
            "feature": nearest_matches(feature_centroids[seed]["training"], feature_centroids[seed]["validation"],
                                       train_rows, val_rows),
        }
        for arm in ARMS:
            gradient_train = [row for row in selected_gradient if row["policy_seed"] == seed
                              and row["arm"] == arm and row["split"] == "training"]
            gradient_val = [row for row in selected_gradient if row["policy_seed"] == seed
                            and row["arm"] == arm and row["split"] == "validation"]
            raw = next(row["gradient_cosine_shift"] for row in gradient_shifts
                       if row["policy_seed"] == seed and row["arm"] == arm)
            for method, (mapping, quality) in mappings.items():
                interval = matched_gap_interval(
                    gradient_train, gradient_val, mapping,
                    seed=int(comparison["bootstrap_seed"]) + 300000 + seed * 10
                         + ARMS.index(arm) * 2 + (method == "feature"), **stat,
                )
                raw_abs = abs(float(raw["estimate"]))
                closure = 1 - abs(float(interval["estimate"])) / raw_abs if raw_abs > 1e-12 else 0.0
                matches.append({"policy_seed": seed, "arm": arm, "method": method,
                                "matched_gradient_gap": interval, "raw_gradient_gap": float(raw["estimate"]),
                                "absolute_gap_closure": float(closure), **quality})
    drift_bar.close()
    progress_message("A8-D2 阶段3/3：保存分布漂移、匹配敏感性与预注册判定")
    interpretation = interpret_gradient_shift(
        gradient_shifts, target_drifts, feature_drifts, matches,
        policy_seeds=settings["policy_seeds"],
        meaningful_cosine_shift=float(comparison["meaningful_cosine_shift"]),
        meaningful_drift=float(comparison["meaningful_standardized_drift"]),
        minimum_supporting_seeds=int(comparison["minimum_supporting_seeds"]),
        minimum_gap_closure=float(experiment["design"]["matching"]["minimum_gap_closure"]),
        minimum_unique_fraction=float(experiment["design"]["matching"]["minimum_unique_match_fraction"]),
        maximum_reuse=int(experiment["design"]["matching"]["maximum_reuse_count"]),
        distribution_permutation_alpha=(float(comparison["familywise_alpha"])
                                        / int(comparison["distribution_permutation_family_size"])),
        quick=settings["quick"],
    )
    all_descriptors = [row for seed in settings["policy_seeds"] for split in SPLITS for row in descriptor_rows[seed][split]]
    _write_rows(output / "episode_descriptors.csv", all_descriptors)
    _write_rows(output / "gradient_shift.csv", _flatten(gradient_shifts))
    _write_rows(output / "condition_shift.csv", condition_shifts)
    _write_rows(output / "target_drift.csv", _flatten(target_drifts))
    _write_rows(output / "feature_drift.csv", _flatten(feature_drifts))
    _write_rows(output / "target_descriptor_drift.csv", [
        {"policy_seed": row["policy_seed"], "descriptor": name, **interval}
        for row in target_drifts for name, interval in row["descriptors"].items()
    ])
    _write_rows(output / "largest_feature_dimensions.csv", [
        {"policy_seed": row["policy_seed"], **item}
        for row in feature_drifts for item in row["largest_feature_dimensions"]
    ])
    _write_rows(output / "matched_sensitivity.csv", _flatten(matches))
    summary = {
        "material_passport": {"origin_skill": "academic-research-suite / experiment-agent",
                              "origin_mode": "run", "origin_date": datetime.now(timezone.utc).isoformat(),
                              "verification_status": "UNVERIFIED", "version_label": experiment["metadata"]["version_label"]},
        "experiment": {"id": experiment["metadata"]["experiment_id"], "quick": settings["quick"],
                       "status": "quick_smoke_only" if settings["quick"] else "completed_pending_audit",
                       "duration_seconds": time.perf_counter() - started, "device": str(device),
                       "gpu_name": torch.cuda.get_device_name(device)},
        "contract": {"diagnostic_only": True, "post_hoc": True,
                     "statistical_unit": "complete_episode_seed", "training_validation_paired": False,
                     "independent_test_samples_accessed": False},
        "descriptor_records": len(all_descriptors), "gradient_shifts": gradient_shifts,
        "condition_shifts": condition_shifts, "target_drifts": target_drifts,
        "feature_drifts": feature_drifts, "matched_sensitivity": matches,
        "interpretation": interpretation,
        "records": {_relative(path): _file_sha256(path) for path in output.iterdir() if path.is_file()},
        "evidence_boundary": {"saved_gradient_read_only": True, "frozen_dataset_read_only": True,
                              "training_run": False, "new_data_generated": False,
                              "independent_test_samples_accessed": False, "causal_explanation_authorized": False,
                              "full_rl_trained": False, "s4d3_accessed": False, "real_slm_actions": False,
                              **NO_UPDATES},
        "runtime": _runtime_record(), "git": _git_record_utf8(),
        "next_action": "停止并进行只读审计；不要自动训练新模型或强化学习。",
    }
    _write_json(output / "summary.json", summary)
    marker = output / ("QUICK_SUCCESS.json" if settings["quick"] else "SUCCESS.json")
    _write_json(marker, {"created_at": datetime.now(timezone.utc).isoformat(),
                         "status": "QUICK_SMOKE_ONLY" if settings["quick"] else "DIAGNOSTIC_COMPLETED_PENDING_AUDIT",
                         "summary_sha256": _file_sha256(output / "summary.json"), **NO_UPDATES})
    return json_safe(summary)


def run_s4_r3_h16_gradient_shift(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    config_path = _project_path(config_path)
    experiment = _load_yaml(config_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight = preflight_gradient_shift(config_path, experiment, settings, quick=quick)
    if preflight_only:
        return preflight
    device = resolve_device(experiment["runtime"]["device"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    output = _project_path(settings["output_directory"])
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("preflight", preflight), ("effective_config", {"experiment": experiment, "settings": settings}),
                        ("source_manifest", preflight["frozen_source_hashes"]),
                        ("input_manifest", preflight["frozen_input_hashes"])):
        _write_json(output / f"{name}.json", value)
    try:
        return _execute(experiment, settings, preflight, output, device)
    except Exception as error:
        import traceback
        _write_json(output / "failure.json", {"exception": type(error).__name__, "message": str(error),
                                               "traceback": traceback.format_exc(), "automatic_retry": False})
        raise
