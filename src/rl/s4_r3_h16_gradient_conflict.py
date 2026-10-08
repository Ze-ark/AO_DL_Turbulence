"""A8-D1：只读诊断收益回归与动作分类在共享主干上的梯度冲突。"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.rl.s4_r3_h16_data_scaling import _episode_seeds, _read_json, _read_pairs, _verify_hash
from src.rl.s4_r3_h16_head_split import HeadSplitProbe, _git_record_utf8, validate_coverage
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
from src.runtime import resolve_device
from src.training_progress import counted_progress, update_progress


ARMS = ("shared", "split")
SPLITS = ("training", "validation")
NO_UPDATES = {
    "optimizer_created": False,
    "optimizer_updates": 0,
    "original_critic_updates": 0,
    "actor_updates": 0,
    "alpha_updates": 0,
    "student_updates": 0,
}


def _seed_item(mapping: dict[Any, Any], seed: int) -> Any:
    if seed in mapping:
        return mapping[seed]
    if str(seed) in mapping:
        return mapping[str(seed)]
    raise KeyError(seed)


def _effective_settings(experiment: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    seeds = [int(experiment["upstream_a8"]["paired_reference_seed"]),
             *[int(v) for v in experiment["upstream_a8"]["paired_failure_seeds"]]]
    settings = {
        "quick": quick,
        "policy_seeds": seeds,
        "output_directory": experiment["outputs"]["quick_directory" if quick else "directory"],
        "bootstrap_replicates": int(experiment["comparison"]["bootstrap_replicates"]),
        "episodes_per_condition": None,
    }
    if quick:
        settings["policy_seeds"] = [int(v) for v in experiment["quick"]["policy_seeds"]]
        settings["bootstrap_replicates"] = int(experiment["quick"]["bootstrap_replicates"])
        settings["episodes_per_condition"] = int(experiment["quick"]["episodes_per_condition"])
    return settings


def _validate_contract(experiment: dict[str, Any]) -> None:
    metadata = experiment["metadata"]
    if metadata["stage"] != "S4-D2-R3-D2-A8-D1" or not metadata["diagnostic_only"]:
        raise RuntimeError("A8-D1 stage or diagnostic-only contract changed")
    allowed_true = {"diagnostic_only", "post_hoc_mechanism_diagnostic",
                    "allow_checkpoint_loading", "allow_checkpoint_gradient_read"}
    for name, value in metadata.items():
        if name.startswith("allow_") and name not in allowed_true and bool(value):
            raise RuntimeError(f"A8-D1 safety flag must remain false: {name}")
    if not all(bool(metadata[name]) for name in allowed_true):
        raise RuntimeError("A8-D1 required read-only flags changed")
    design = experiment["design"]
    if tuple(design["arms"]) != ARMS or tuple(design["splits"]) != SPLITS:
        raise RuntimeError("A8-D1 arm/split design changed")
    if (int(design["feature_size"]), int(design["hidden_size"]), int(design["target_horizon"])) != (221, 256, 16):
        raise RuntimeError("A8-D1 frozen model shape or horizon changed")
    if list(design["shared_parameter_prefixes"]) != ["network.0", "network.2"]:
        raise RuntimeError("A8-D1 shared trunk definition changed")
    loss = design["loss"]
    if (loss["regression"] != "huber" or float(loss["huber_delta"]) != 1.0
            or loss["classification"] != "weighted_binary_cross_entropy"
            or float(loss["balanced_sign_weight"]) != 0.25
            or loss["positive_class_weight_source"] != "complete_training_split"):
        raise RuntimeError("A8-D1 must reproduce the A8 loss definitions")
    if design["statistical_unit"] != "complete_episode_seed" or int(design["pairs_per_episode"]) != 18:
        raise RuntimeError("A8-D1 complete-episode statistical unit changed")
    comparison = experiment["comparison"]
    if (int(comparison["bootstrap_replicates"]) != 20000
            or float(comparison["familywise_alpha"]) != 0.05
            or int(comparison["family_size"]) != 20
            or float(comparison["meaningful_cosine_difference"]) != 0.05
            or float(comparison["conflict_cosine_threshold"]) != 0.0
            or float(comparison["conflict_fraction_threshold"]) != 0.5):
        raise RuntimeError("A8-D1 preregistered statistical contract changed")
    runtime = experiment["runtime"]
    if runtime["device"] != "cuda" or not runtime["require_cuda"] or not runtime["deterministic_algorithms"]:
        raise RuntimeError("A8-D1 formal CUDA/determinism contract changed")
    if any("independent_test" in str(value).lower() for value in _walk_values(experiment["datasets"])):
        raise RuntimeError("A8-D1 may not reference independent-test samples")


def _walk_values(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for nested in value.values():
            yield from _walk_values(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _walk_values(nested)
    else:
        yield value


def _episode_groups(data: RewardPairDataset) -> list[tuple[str, int, list[int]]]:
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    seed_condition: dict[int, str] = {}
    for index, row in enumerate(data.rows):
        condition, episode = str(row["condition_id"]), int(row["episode_seed"])
        if episode in seed_condition and seed_condition[episode] != condition:
            raise RuntimeError("A8-D1 one episode belongs to multiple conditions")
        seed_condition[episode] = condition
        groups[condition, episode].append(index)
    result = [(condition, episode, indices) for (condition, episode), indices in sorted(groups.items())]
    if any(len(indices) != 18 for _, _, indices in result):
        raise RuntimeError("A8-D1 requires 18 complete pairs per episode")
    return result


def _selected_groups(data: RewardPairDataset, episodes_per_condition: int | None) -> list[tuple[str, int, list[int]]]:
    groups = _episode_groups(data)
    if episodes_per_condition is None:
        return groups
    selected: list[tuple[str, int, list[int]]] = []
    by_condition: dict[str, list[tuple[str, int, list[int]]]] = defaultdict(list)
    for group in groups:
        by_condition[group[0]].append(group)
    for condition in sorted(by_condition):
        if len(by_condition[condition]) < episodes_per_condition:
            raise RuntimeError("A8-D1 quick selection lacks complete episodes")
        selected.extend(by_condition[condition][:episodes_per_condition])
    return selected


def _model_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def gradient_relation(
    regression_gradients: Sequence[torch.Tensor],
    classification_gradients: Sequence[torch.Tensor],
    *,
    classification_weight: float,
) -> dict[str, float]:
    """计算两个梯度向量的方向和相对大小，不修改参数。"""
    if len(regression_gradients) != len(classification_gradients) or not regression_gradients:
        raise ValueError("A8-D1 gradient lists must be aligned and non-empty")
    dot = torch.zeros((), dtype=torch.float64, device=regression_gradients[0].device)
    norm_r2 = torch.zeros_like(dot)
    norm_c2 = torch.zeros_like(dot)
    for regression, classification in zip(regression_gradients, classification_gradients, strict=True):
        if regression.shape != classification.shape:
            raise ValueError("A8-D1 gradient tensor shapes differ")
        r, c = regression.detach().double().reshape(-1), classification.detach().double().reshape(-1)
        if not bool(torch.isfinite(r).all() and torch.isfinite(c).all()):
            raise RuntimeError("A8-D1 non-finite gradient")
        dot += torch.dot(r, c)
        norm_r2 += torch.dot(r, r)
        norm_c2 += torch.dot(c, c)
    norm_r, norm_c = norm_r2.sqrt(), norm_c2.sqrt()
    if float(norm_r) <= 0 or float(norm_c) <= 0:
        raise RuntimeError("A8-D1 zero shared-trunk gradient")
    cosine = dot / (norm_r * norm_c)
    weighted_norm = abs(classification_weight) * norm_c
    return {
        "gradient_cosine": float(cosine),
        "gradient_dot": float(dot),
        "regression_gradient_norm": float(norm_r),
        "classification_gradient_norm": float(norm_c),
        "weighted_classification_gradient_norm": float(weighted_norm),
        "weighted_classification_to_regression_norm_ratio": float(weighted_norm / norm_r),
    }


def _episode_gradient(
    model: HeadSplitProbe,
    data: RewardPairDataset,
    indices: list[int],
    checkpoint: dict[str, Any],
    *,
    positive_weight: torch.Tensor,
    classification_weight: float,
    huber_delta: float,
    device: torch.device,
) -> dict[str, float]:
    named = [(name, parameter) for name, parameter in model.named_parameters()
             if name.startswith("network.0.") or name.startswith("network.2.")]
    if [name for name, _ in named] != ["network.0.weight", "network.0.bias", "network.2.weight", "network.2.bias"]:
        raise RuntimeError("A8-D1 shared trunk parameters changed")
    parameters = [parameter for _, parameter in named]
    norms = {name: value.to(device) for name, value in checkpoint["normalization"].items()}
    index = torch.tensor(indices, dtype=torch.long)
    features = data.features[index].to(device)
    target = data.reward_delta[index].to(device)
    normalized = (features - norms["feature_mean"]) / norms["feature_scale"]
    value, score = model(normalized)
    regression_loss = F.huber_loss(value, target / norms["target_scale"], delta=huber_delta)
    classification_loss = F.binary_cross_entropy_with_logits(
        score, (target > 0).float(), pos_weight=positive_weight,
    )
    regression_gradients = torch.autograd.grad(regression_loss, parameters, retain_graph=True)
    classification_gradients = torch.autograd.grad(classification_loss, parameters)
    relation = gradient_relation(regression_gradients, classification_gradients,
                                 classification_weight=classification_weight)
    for prefix in ("network.0", "network.2"):
        positions = [i for i, (name, _) in enumerate(named) if name.startswith(prefix + ".")]
        layer = gradient_relation([regression_gradients[i] for i in positions],
                                  [classification_gradients[i] for i in positions],
                                  classification_weight=classification_weight)
        relation[f"{prefix.replace('.', '_')}_gradient_cosine"] = layer["gradient_cosine"]
    relation.update({
        "regression_loss": float(regression_loss.detach()),
        "classification_loss": float(classification_loss.detach()),
        "weighted_classification_loss": float(classification_weight * classification_loss.detach()),
        "positive_fraction": float((target > 0).float().mean()),
        "gradient_conflict": float(relation["gradient_cosine"] < 0.0),
    })
    return relation


def stratified_interval(
    rows: Sequence[dict[str, Any]],
    value_field: str,
    *,
    replicates: int,
    seed: int,
    familywise_alpha: float,
    family_size: int,
) -> dict[str, Any]:
    if replicates < 20 or not rows or family_size < 1 or not 0 < familywise_alpha < 1:
        raise ValueError("A8-D1 invalid interval settings")
    strata: dict[str, list[float]] = defaultdict(list)
    keys: set[tuple[str, int]] = set()
    for row in rows:
        key = str(row["condition_id"]), int(row["episode_seed"])
        if key in keys:
            raise RuntimeError("A8-D1 interval received duplicate episodes")
        keys.add(key)
        value = float(row[value_field])
        if not np.isfinite(value):
            raise RuntimeError("A8-D1 interval received non-finite values")
        strata[key[0]].append(value)
    rng = np.random.default_rng(seed)
    totals = np.zeros(replicates, dtype=np.float64)
    count = 0
    for values in strata.values():
        array = np.asarray(values, dtype=np.float64)
        totals += array[rng.integers(len(array), size=(replicates, len(array)))].sum(axis=1)
        count += len(array)
    distribution = totals / count
    tail = familywise_alpha / (2 * family_size)
    return {
        "estimate": float(np.mean([float(row[value_field]) for row in rows])),
        "ci95_low": float(np.quantile(distribution, 0.025)),
        "ci95_high": float(np.quantile(distribution, 0.975)),
        "familywise_ci_low": float(np.quantile(distribution, tail)),
        "familywise_ci_high": float(np.quantile(distribution, 1 - tail)),
        "replicates": replicates,
        "clusters": len(rows),
        "unit": "complete_episode_seed",
        "stratified_by": "condition_id",
        "familywise_alpha": familywise_alpha,
        "family_size": family_size,
    }


def paired_interval(
    left: Sequence[dict[str, Any]],
    right: Sequence[dict[str, Any]],
    value_field: str,
    *,
    replicates: int,
    seed: int,
    familywise_alpha: float,
    family_size: int,
) -> dict[str, Any]:
    """返回 right-left 的配对均值区间。"""
    left_by_key = {(str(row["condition_id"]), int(row["episode_seed"])): row for row in left}
    right_by_key = {(str(row["condition_id"]), int(row["episode_seed"])): row for row in right}
    if len(left_by_key) != len(left) or len(right_by_key) != len(right) or left_by_key.keys() != right_by_key.keys():
        raise RuntimeError("A8-D1 paired episodes do not align")
    differences = [{"condition_id": key[0], "episode_seed": key[1],
                    "difference": float(right_by_key[key][value_field]) - float(left_by_key[key][value_field])}
                   for key in sorted(left_by_key)]
    return stratified_interval(differences, "difference", replicates=replicates, seed=seed,
                               familywise_alpha=familywise_alpha, family_size=family_size)


def _failed_mean_interval(
    reference: Sequence[dict[str, Any]],
    failed_groups: Sequence[Sequence[dict[str, Any]]],
    *,
    replicates: int,
    seed: int,
    familywise_alpha: float,
    family_size: int,
) -> dict[str, Any]:
    reference_by_key = {(str(row["condition_id"]), int(row["episode_seed"])): row for row in reference}
    failed_by_key = [{(str(row["condition_id"]), int(row["episode_seed"])): row for row in group}
                     for group in failed_groups]
    if any(group.keys() != reference_by_key.keys() for group in failed_by_key):
        raise RuntimeError("A8-D1 cross-seed episode pairing changed")
    differences = []
    for key in sorted(reference_by_key):
        failed_mean = float(np.mean([float(group[key]["gradient_cosine"]) for group in failed_by_key]))
        differences.append({"condition_id": key[0], "episode_seed": key[1],
                            "difference": failed_mean - float(reference_by_key[key]["gradient_cosine"])})
    return stratified_interval(differences, "difference", replicates=replicates, seed=seed,
                               familywise_alpha=familywise_alpha, family_size=family_size)


def interpret_gradient_conflict(
    summaries: Sequence[dict[str, Any]],
    head_comparisons: Sequence[dict[str, Any]],
    failure_comparisons: Sequence[dict[str, Any]],
    *,
    reference_seed: int,
    failure_seeds: Sequence[int],
    meaningful_difference: float,
    quick: bool,
) -> dict[str, Any]:
    if quick:
        return {"status": "QUICK_SMOKE_ONLY", "shared_trunk_conflict_supported": False,
                "failure_seed_pattern_supported": False, "last_layer_split_alleviates_conflict": False,
                "independent_trunk_experiment_design_authorized": False, "full_rl_authorized": False,
                "s4d3_authorized": False, "real_hardware_authorized": False}
    lookup = {(int(row["policy_seed"]), row["arm"], row["split"]): row for row in summaries}
    required = {(seed, arm, split) for seed in [reference_seed, *failure_seeds]
                for arm in ARMS for split in SPLITS}
    if lookup.keys() != required:
        raise RuntimeError("A8-D1 summary coverage incomplete")
    conflict = all(
        lookup[seed, "split", split]["gradient_cosine"]["familywise_ci_high"] < 0
        and lookup[seed, "split", split]["conflict_fraction"]["familywise_ci_low"] > 0.5
        for seed in failure_seeds for split in SPLITS
    )
    failure_lookup = {row["split"]: row for row in failure_comparisons}
    failure_pattern = set(failure_lookup) == set(SPLITS) and all(
        failure_lookup[split]["failed_mean_minus_reference_cosine"]["familywise_ci_high"] < -meaningful_difference
        for split in SPLITS
    )
    head_lookup = {(int(row["policy_seed"]), row["split"]): row for row in head_comparisons}
    alleviates = all(
        head_lookup[seed, "validation"]["split_minus_shared_cosine"]["familywise_ci_low"] > meaningful_difference
        for seed in [reference_seed, *failure_seeds]
    )
    if conflict and failure_pattern:
        status = "SHARED_TRUNK_CONFLICT_AND_FAILURE_PATTERN_SUPPORTED"
    elif conflict:
        status = "SHARED_TRUNK_CONFLICT_SUPPORTED_WITHOUT_FAILURE_PATTERN"
    else:
        status = "SHARED_TRUNK_CONFLICT_NOT_CONFIRMED"
    return {
        "status": status,
        "shared_trunk_conflict_supported": conflict,
        "failure_seed_pattern_supported": failure_pattern,
        "last_layer_split_alleviates_conflict": alleviates,
        "independent_trunk_experiment_design_authorized": bool(conflict and failure_pattern),
        "full_rl_authorized": False,
        "s4d3_authorized": False,
        "real_hardware_authorized": False,
        "post_hoc_causal_claim_authorized": False,
    }


def preflight_gradient_conflict(
    config_path: str | Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    _validate_contract(experiment)
    if not torch.cuda.is_available():
        raise RuntimeError("A8-D1 requires CUDA; CPU fallback is forbidden")
    input_hashes: dict[str, str] = {}
    for section in ("design_contract", "upstream_a8"):
        for name, checksum in experiment[section].items():
            if name.endswith("_sha256"):
                path = experiment[section][name[:-7]]
                _verify_hash(path, str(checksum))
                input_hashes[path] = str(checksum)
    summary = _read_json(experiment["upstream_a8"]["summary"])
    if summary["experiment"]["quick"] or summary["interpretation"]["status"] != experiment["upstream_a8"]["required_interpretation_status"]:
        raise RuntimeError("A8-D1 upstream A8 status changed")
    if summary["experiment"].get("completion_mode") != "metadata_only_recovery":
        raise RuntimeError("A8-D1 recovery provenance changed")
    recovered = _read_json(experiment["upstream_a8"]["recovered_success"])
    if recovered["status"] != "RECOVERED_METADATA_ONLY" or recovered["summary_sha256"] != experiment["upstream_a8"]["summary_sha256"]:
        raise RuntimeError("A8-D1 recovered summary marker changed")
    paired = {int(row["policy_seed"]): bool(row["passed"]) for row in summary["interpretation"]["paired_benefit"]}
    reference = int(experiment["upstream_a8"]["paired_reference_seed"])
    failures = [int(v) for v in experiment["upstream_a8"]["paired_failure_seeds"]]
    if paired != {reference: True, **{seed: False for seed in failures}}:
        raise RuntimeError("A8-D1 upstream paired outcome mapping changed")
    dataset_specs: dict[int, dict[str, Any]] = {}
    episode_sets: dict[str, list[set[int]]] = {split: [] for split in SPLITS}
    condition_maps: dict[str, list[dict[int, str]]] = {split: [] for split in SPLITS}
    for seed in settings["policy_seeds"]:
        dataset_specs[seed] = {}
        for split in SPLITS:
            spec = dict(_seed_item(experiment["datasets"], seed)[split])
            _verify_hash(spec["path"], spec["sha256"])
            input_hashes[spec["path"]] = spec["sha256"]
            data = _read_pairs(_project_path(spec["path"]))
            validate_coverage(data, episodes=int(spec["episodes"]), pairs=int(spec["pairs"]))
            groups = _episode_groups(data)
            episode_sets[split].append({episode for _, episode, _ in groups})
            condition_maps[split].append({episode: condition for condition, episode, _ in groups})
            dataset_specs[seed][split] = spec
        train_seeds = episode_sets["training"][-1]
        validation_seeds = episode_sets["validation"][-1]
        if train_seeds & validation_seeds:
            raise RuntimeError("A8-D1 training/validation episode leakage")
    for split in SPLITS:
        if any(values != episode_sets[split][0] for values in episode_sets[split][1:]) or any(
                values != condition_maps[split][0] for values in condition_maps[split][1:]):
            raise RuntimeError("A8-D1 policy seeds do not share paired complete episodes")
    checkpoint_specs: dict[int, dict[str, Any]] = {}
    for seed in settings["policy_seeds"]:
        checkpoint_specs[seed] = {}
        for arm in ARMS:
            spec = dict(_seed_item(experiment["checkpoints"], seed)[arm])
            _verify_hash(spec["path"], spec["sha256"])
            input_hashes[spec["path"]] = spec["sha256"]
            checkpoint = torch.load(_project_path(spec["path"]), map_location="cpu", weights_only=False)
            if (int(checkpoint["policy_seed"]) != seed or checkpoint["arm"] != arm
                    or checkpoint["target_horizon"] != 16 or checkpoint["independent_test_used_for_selection"]):
                raise RuntimeError("A8-D1 checkpoint provenance changed")
            if any(int(checkpoint[name]) != 0 for name in ("original_critic_updates", "actor_updates", "alpha_updates", "student_updates")):
                raise RuntimeError("A8-D1 checkpoint update boundary changed")
            checkpoint_specs[seed][arm] = spec
    sources = _source_manifest(experiment["tracked_source_files"])
    sources[_relative(_project_path(config_path))] = _file_sha256(_project_path(config_path))
    output = _project_path(settings["output_directory"])
    if output.exists():
        raise FileExistsError(f"A8-D1 output already exists: {output}")
    selected_per_split = {}
    first_seed = settings["policy_seeds"][0]
    for split in SPLITS:
        data = _read_pairs(_project_path(dataset_specs[first_seed][split]["path"]))
        selected_per_split[split] = len(_selected_groups(data, settings["episodes_per_condition"]))
    return {
        "status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_DIAGNOSTIC",
        "quick": quick,
        "policy_seeds": settings["policy_seeds"],
        "selected_episodes_per_split": selected_per_split,
        "expected_episode_records": len(settings["policy_seeds"]) * len(ARMS) * sum(selected_per_split.values()),
        "gradient_evaluations_per_record": 2,
        "dataset_specs": dataset_specs,
        "checkpoint_specs": checkpoint_specs,
        "frozen_input_hashes": input_hashes,
        "frozen_source_hashes": sources,
        "output_directory": _relative(output),
        "independent_test_sample_paths": [],
        "new_data_generated": False,
        "training_run": False,
        "full_rl_training": False,
        "s4d3_access": False,
        "real_slm_actions": False,
        **NO_UPDATES,
    }


def _write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty rows: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _flatten_intervals(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    flat = []
    for row in rows:
        record = {}
        for name, value in row.items():
            if isinstance(value, dict):
                record.update({f"{name}_{key}": item for key, item in value.items()})
            else:
                record[name] = value
        flat.append(record)
    return flat


def _execute(
    experiment: dict[str, Any],
    settings: dict[str, Any],
    preflight: dict[str, Any],
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    episode_rows: list[dict[str, Any]] = []
    unchanged: list[dict[str, Any]] = []
    progress_path = output / "progress.jsonl"
    bar = counted_progress(total=preflight["expected_episode_records"], description="A8-D1 梯度诊断", unit="回合")
    completed = 0
    with progress_path.open("w", encoding="utf-8") as progress:
        for seed in settings["policy_seeds"]:
            datasets = {split: _read_pairs(_project_path(preflight["dataset_specs"][seed][split]["path"]))
                        for split in SPLITS}
            training_target = datasets["training"].reward_delta.to(device)
            positives = int((training_target > 0).sum())
            if positives == 0 or positives == len(training_target):
                raise RuntimeError("A8-D1 complete training split must contain both classes")
            positive_weight = torch.tensor((len(training_target) - positives) / positives, device=device)
            for arm in ARMS:
                spec = preflight["checkpoint_specs"][seed][arm]
                checkpoint = torch.load(_project_path(spec["path"]), map_location=device, weights_only=False)
                model = HeadSplitProbe(**checkpoint["config"], arm=arm).to(device)
                model.load_state_dict(checkpoint["probe"])
                model.eval()
                before = _model_digest(model)
                if any(parameter.grad is not None for parameter in model.parameters()):
                    raise RuntimeError("A8-D1 loaded checkpoint already has parameter gradients")
                for split in SPLITS:
                    data = datasets[split]
                    groups = _selected_groups(data, settings["episodes_per_condition"])
                    for condition, episode, indices in groups:
                        metrics = _episode_gradient(
                            model, data, indices, checkpoint,
                            positive_weight=positive_weight,
                            classification_weight=float(experiment["design"]["loss"]["balanced_sign_weight"]),
                            huber_delta=float(experiment["design"]["loss"]["huber_delta"]),
                            device=device,
                        )
                        episode_rows.append({"policy_seed": seed, "arm": arm, "split": split,
                                             "condition_id": condition, "episode_seed": episode,
                                             "pairs": len(indices), **metrics})
                        completed += 1
                        elapsed = time.perf_counter() - started
                        record = {"completed_episode_records": completed,
                                  "total_episode_records": preflight["expected_episode_records"],
                                  "policy_seed": seed, "arm": arm, "split": split,
                                  "condition_id": condition, "episode_seed": episode,
                                  "latest_gradient_cosine": metrics["gradient_cosine"],
                                  "mean_gradient_cosine": float(np.mean([row["gradient_cosine"] for row in episode_rows])),
                                  "estimated_remaining_seconds": elapsed / completed * (preflight["expected_episode_records"] - completed),
                                  "cuda_allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
                                  "cuda_reserved_gb": torch.cuda.memory_reserved(device) / 1024**3}
                        progress.write(json.dumps(record, ensure_ascii=False) + "\n")
                        progress.flush()
                        bar.update(1)
                        update_progress(bar, device=device, metrics={"余弦": record["mean_gradient_cosine"]})
                after = _model_digest(model)
                gradients_untouched = all(parameter.grad is None for parameter in model.parameters())
                unchanged.append({"policy_seed": seed, "arm": arm, "before_sha256": before,
                                  "after_sha256": after, "parameters_unchanged": before == after,
                                  "parameter_grad_fields_untouched": gradients_untouched})
                if before != after or not gradients_untouched:
                    raise RuntimeError("A8-D1 changed a frozen model")
                del model, checkpoint
    bar.close()
    cfg = experiment["comparison"]
    kwargs = {"replicates": settings["bootstrap_replicates"],
              "familywise_alpha": float(cfg["familywise_alpha"]),
              "family_size": int(cfg["family_size"])}
    summaries = []
    for seed in settings["policy_seeds"]:
        for arm in ARMS:
            for split in SPLITS:
                rows = [row for row in episode_rows if row["policy_seed"] == seed and row["arm"] == arm and row["split"] == split]
                offset = seed * 100 + ARMS.index(arm) * 10 + SPLITS.index(split)
                summaries.append({"policy_seed": seed, "arm": arm, "split": split,
                                  "gradient_cosine": stratified_interval(rows, "gradient_cosine", seed=int(cfg["bootstrap_seed"]) + offset, **kwargs),
                                  "conflict_fraction": stratified_interval(rows, "gradient_conflict", seed=int(cfg["bootstrap_seed"]) + 100000 + offset, **kwargs),
                                  "weighted_norm_ratio_mean": float(np.mean([row["weighted_classification_to_regression_norm_ratio"] for row in rows])),
                                  "layer_1_cosine_mean": float(np.mean([row["network_0_gradient_cosine"] for row in rows])),
                                  "layer_2_cosine_mean": float(np.mean([row["network_2_gradient_cosine"] for row in rows]))})
    head_comparisons = []
    for seed in settings["policy_seeds"]:
        for split in SPLITS:
            shared = [row for row in episode_rows if row["policy_seed"] == seed and row["arm"] == "shared" and row["split"] == split]
            separated = [row for row in episode_rows if row["policy_seed"] == seed and row["arm"] == "split" and row["split"] == split]
            head_comparisons.append({"policy_seed": seed, "split": split,
                                     "split_minus_shared_cosine": paired_interval(shared, separated, "gradient_cosine",
                                         seed=int(cfg["bootstrap_seed"]) + 200000 + seed * 10 + SPLITS.index(split), **kwargs)})
    failure_comparisons = []
    reference_seed = int(experiment["upstream_a8"]["paired_reference_seed"])
    failure_seeds = [int(v) for v in experiment["upstream_a8"]["paired_failure_seeds"]]
    if set([reference_seed, *failure_seeds]).issubset(settings["policy_seeds"]):
        for split in SPLITS:
            reference = [row for row in episode_rows if row["policy_seed"] == reference_seed and row["arm"] == "split" and row["split"] == split]
            failed = [[row for row in episode_rows if row["policy_seed"] == seed and row["arm"] == "split" and row["split"] == split]
                      for seed in failure_seeds]
            failure_comparisons.append({"split": split,
                "failed_mean_minus_reference_cosine": _failed_mean_interval(reference, failed,
                    seed=int(cfg["bootstrap_seed"]) + 300000 + SPLITS.index(split), **kwargs)})
    interpretation = interpret_gradient_conflict(
        summaries, head_comparisons, failure_comparisons,
        reference_seed=reference_seed, failure_seeds=failure_seeds,
        meaningful_difference=float(cfg["meaningful_cosine_difference"]), quick=settings["quick"],
    )
    _write_rows(output / "episode_gradient_metrics.csv", episode_rows)
    _write_rows(output / "arm_summaries.csv", _flatten_intervals(summaries))
    _write_rows(output / "head_comparisons.csv", _flatten_intervals(head_comparisons))
    if failure_comparisons:
        _write_rows(output / "failure_seed_comparisons.csv", _flatten_intervals(failure_comparisons))
    summary = {
        "material_passport": {"origin_skill": "academic-research-suite / experiment-agent",
                              "origin_mode": "run", "origin_date": datetime.now(timezone.utc).isoformat(),
                              "verification_status": "UNVERIFIED", "version_label": experiment["metadata"]["version_label"]},
        "experiment": {"id": experiment["metadata"]["experiment_id"], "quick": settings["quick"],
                       "status": "quick_smoke_only" if settings["quick"] else "completed_pending_audit",
                       "duration_seconds": time.perf_counter() - started, "device": str(device),
                       "gpu_name": torch.cuda.get_device_name(device)},
        "contract": {"diagnostic_only": True, "post_hoc": True,
                     "losses_match_a8": True, "shared_trunk_only": ["network.0", "network.2"],
                     "statistical_unit": "complete_episode_seed", "independent_test_samples_accessed": False},
        "episode_records": len(episode_rows), "parameter_integrity": unchanged,
        "arm_summaries": summaries, "head_comparisons": head_comparisons,
        "failure_seed_comparisons": failure_comparisons, "interpretation": interpretation,
        "records": {_relative(path): _file_sha256(path) for path in output.iterdir() if path.is_file()},
        "evidence_boundary": {"checkpoint_gradient_read_only": True, "training_run": False,
                              "new_data_generated": False, "independent_test_samples_accessed": False,
                              "post_hoc_causal_claim_authorized": False, "full_rl_trained": False,
                              "s4d3_accessed": False, "real_slm_actions": False, **NO_UPDATES},
        "runtime": _runtime_record(), "git": _git_record_utf8(),
        "next_action": "停止并进行只读审计；不要自动训练完全独立主干或强化学习。",
    }
    _write_json(output / "summary.json", summary)
    marker = output / ("QUICK_SUCCESS.json" if settings["quick"] else "SUCCESS.json")
    _write_json(marker, {"created_at": datetime.now(timezone.utc).isoformat(),
                         "status": "QUICK_SMOKE_ONLY" if settings["quick"] else "DIAGNOSTIC_COMPLETED_PENDING_AUDIT",
                         "summary_sha256": _file_sha256(output / "summary.json"), **NO_UPDATES})
    return json_safe(summary)


def run_s4_r3_h16_gradient_conflict(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    config_path = _project_path(config_path)
    experiment = _load_yaml(config_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight = preflight_gradient_conflict(config_path, experiment, settings, quick=quick)
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
