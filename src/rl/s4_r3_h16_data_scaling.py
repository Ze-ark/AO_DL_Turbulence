"""A7：固定模型与预算，比较嵌套的96/384条独立训练回合。"""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from src.rl.s4_r3_critic_source_diagnostic import _load_extended_policy
from src.rl.s4_r3_failure_diagnostic import _load_student
from src.rl.s4_r3_h16_reward_pairwise_probe import (
    RewardPairDataset,
    _effective_settings as _a6_settings,
    _fit_csv_row,
    _initial_probe_state,
    _normalization_from_training,
    _pair_dataset_summary,
    _predict_dataset,
    _runtime_fields,
    evaluate_h16_prediction,
    independent_subgroup_metrics,
    interpret_h16_reward_fits,
    load_reward_pair_dataset,
)
from src.rl.s4_r3_multistep_critic_training import (
    _collect_policy_split,
    _effective_settings as _collector_settings,
    preflight_s4_r3_multistep_critic_training,
)
from src.rl.s4_r3_pairwise_rank_learnability import (
    PairwiseDeltaProbe,
    classification_metrics,
)
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import (
    _file_sha256, _git_record, _load_yaml, _project_path, _relative,
    _runtime_record, _source_manifest, _write_json, json_safe,
)
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.training_progress import counted_progress, progress_bar, progress_message, update_progress

SCALES = ("small", "large")


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_project_path(path).read_text(encoding="utf-8"))


def _verify_hash(path: str | Path, expected: str) -> None:
    resolved = _project_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if _file_sha256(resolved) != expected:
        raise RuntimeError(f"A7 hash mismatch: {resolved}")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("A7 cannot write empty records")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(json_safe(rows))


def _effective_settings(experiment: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    a6 = _load_yaml(_project_path(experiment["upstream_a6"]["experiment_config"]))
    source = _load_yaml(_project_path(experiment["frozen_collection"]["experiment_config"]))
    settings = _a6_settings(a6, quick=False)
    settings["quick"] = quick
    settings["output_directory"] = experiment["outputs"]["quick_directory" if quick else "directory"]
    settings["maximum_updates"] = int(experiment["training"]["maximum_updates"])
    settings["allow_critic_design_authorization"] = False
    settings["comparison"] = deepcopy(experiment["comparison"])
    settings["late_best_update_fraction"] = float(experiment["training"]["late_best_update_fraction"])
    data = experiment["data"]
    offsets = data["added_chunk_offsets"]
    chunk_size = int(data["added_episodes_per_condition_per_chunk"])
    test_count = int(data["test_episodes_per_condition"])
    test_offset = 0
    profiles = list(source["data"]["profile_ids"])
    probes = list(source["data"]["probe_steps"])
    if quick:
        reduced = experiment["quick"]
        settings["policy_seeds"] = list(reduced["policy_seeds"])
        for name in ("maximum_updates", "validation_interval_updates", "batch_size", "cluster_bootstrap_replicates"):
            settings[name] = int(reduced[name])
        settings["comparison"]["bootstrap_replicates"] = int(reduced["comparison_bootstrap_replicates"])
        settings["old_episodes_per_condition"] = int(reduced["old_episodes_per_condition"])
        offsets = reduced["added_chunk_offsets"]
        chunk_size = int(reduced["added_episodes_per_condition_per_chunk"])
        test_count = int(reduced["test_episodes_per_condition"])
        test_offset = int(reduced["test_seed_offset"])
        profiles = list(reduced["profile_ids"])
        probes = list(reduced["probe_steps"])
    settings["candidate_actions"] = deepcopy(source["candidate_actions"])
    common = {
        "profile_ids": profiles, "probe_steps": probes,
        "episode_length_steps": int(source["data"]["episode_length_steps"]),
    }
    settings["added_splits"] = []
    for offset in offsets:
        conditions = deepcopy(source["data"]["development"]["conditions"])
        for condition in conditions:
            condition["base_seed"] += int(offset)
        settings["added_splits"].append({
            **deepcopy(common), "conditions": conditions,
            "episodes_per_condition": chunk_size, "episode_index_offset": int(offset),
        })
    test_conditions = deepcopy(source["data"]["validation"]["conditions"])
    for condition, seed in zip(test_conditions, data["test_base_seeds"], strict=True):
        condition["base_seed"] = int(seed) + test_offset
        condition["id"] = "a7_test_" + str(condition["id"])
    settings["test_split"] = {
        **deepcopy(common), "conditions": test_conditions,
        "episodes_per_condition": test_count, "episode_index_offset": 0,
    }
    return settings


def _split_seeds(split: dict[str, Any]) -> set[int]:
    seeds: set[int] = set()
    count = int(split["episodes_per_condition"])
    if count <= 0:
        raise ValueError("A7 episode count must be positive")
    for condition in split["conditions"]:
        block = set(range(int(condition["base_seed"]), int(condition["base_seed"]) + count))
        if seeds & block:
            raise RuntimeError("A7 condition seed overlap")
        seeds.update(block)
    return seeds


def verify_new_seed_namespace(
    settings: dict[str, Any], *, protected: set[int], reserved: int
) -> dict[str, Any]:
    seen = set(protected)
    records = []
    for name, split in [
        *[(f"added_{index}", split) for index, split in enumerate(settings["added_splits"])],
        ("test", settings["test_split"]),
    ]:
        seeds = _split_seeds(split)
        if min(seeds) < 0 or max(seeds) >= reserved:
            raise RuntimeError("A7 entered a reserved or invalid seed namespace")
        if seen & seeds:
            raise RuntimeError(f"A7 episode seed overlap: {name}")
        seen.update(seeds)
        records.append({"split": name, "episodes": len(seeds), "minimum": min(seeds), "maximum": max(seeds)})
    return {"overlap": 0, "splits": records, "reserved_s4d3_seed_base": reserved}


def _load_pairs(spec: dict[str, Any]) -> RewardPairDataset:
    return load_reward_pair_dataset(
        spec, target_source="empirical_reward_returns", power_source="empirical_power_returns",
        horizon_index=15, state_size=210, action_size=11,
    )


def _episode_seeds(dataset: RewardPairDataset) -> set[int]:
    return {int(row["episode_seed"]) for row in dataset.rows}


def _select_old_episodes(dataset: RewardPairDataset, count: int) -> RewardPairDataset:
    by_condition: dict[str, set[int]] = defaultdict(set)
    for row in dataset.rows:
        by_condition[str(row["condition_id"])].add(int(row["episode_seed"]))
    wanted = {seed for seeds in by_condition.values() for seed in sorted(seeds)[:count]}
    indices = [i for i, row in enumerate(dataset.rows) if int(row["episode_seed"]) in wanted]
    result = RewardPairDataset(dataset.features[indices], dataset.reward_delta[indices],
                               dataset.power_delta[indices], [dataset.rows[i] for i in indices])
    result.validate(feature_size=221)
    return result


def concatenate_pairs(datasets: list[RewardPairDataset]) -> RewardPairDataset:
    seen: set[int] = set()
    for dataset in datasets:
        if seen & _episode_seeds(dataset):
            raise RuntimeError("A7 duplicated episodes across training chunks")
        seen.update(_episode_seeds(dataset))
    result = RewardPairDataset(
        torch.cat([x.features for x in datasets]), torch.cat([x.reward_delta for x in datasets]),
        torch.cat([x.power_delta for x in datasets]), [row for x in datasets for row in x.rows],
    )
    result.validate(feature_size=221)
    return result


def verify_dataset_separation(datasets: dict[str, RewardPairDataset]) -> None:
    names = list(datasets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if _episode_seeds(datasets[left]) & _episode_seeds(datasets[right]):
                raise RuntimeError(f"A7 episode leakage: {left}/{right}")


def _save_pairs(path: Path, dataset: RewardPairDataset) -> None:
    torch.save({"features": dataset.features, "reward_delta": dataset.reward_delta,
                "power_delta": dataset.power_delta, "rows": dataset.rows}, path)


def _read_pairs(path: Path) -> RewardPairDataset:
    data = torch.load(path, map_location="cpu", weights_only=False)
    result = RewardPairDataset(**data)
    result.validate(feature_size=221)
    return result


def preflight_s4_r3_h16_data_scaling(
    config_path: Path, experiment: dict[str, Any], settings: dict[str, Any], *, quick: bool
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """只读锁定证据和种子，不生成新回合，不载入旧测试作为样本。"""
    metadata = experiment["metadata"]
    if metadata["stage"] != "S4-D2-R3-D2-A7":
        raise ValueError("A7 stage changed")
    for field in ("diagnostic_only", "allow_new_episode_generation", "allow_supervised_probe_training"):
        if metadata[field] is not True:
            raise RuntimeError(f"A7 required flag changed: {field}")
    for field in ("allow_full_rl_training", "allow_original_critic_updates", "allow_actor_updates",
                  "allow_alpha_updates", "allow_student_updates", "allow_mechanism_audit_access",
                  "allow_s4d3_access", "allow_real_hardware_actions"):
        if metadata[field] is not False:
            raise RuntimeError(f"A7 safety guard changed: {field}")
    if not experiment["runtime"]["require_cuda"] or not experiment["runtime"]["deterministic_algorithms"]:
        raise RuntimeError("A7 requires deterministic CUDA")
    if resolve_device(experiment["runtime"]["device"]).type != "cuda":
        raise RuntimeError("A7 requires CUDA")
    if experiment["training"] != {
        "maximum_updates": 10000, "early_stopping": False, "late_best_update_fraction": 0.8
    }:
        raise RuntimeError("A7 equal training budget changed")
    if experiment["data"] != {
        "training_episode_counts": [96, 384], "added_chunk_offsets": [32, 64, 96],
        "added_episodes_per_condition_per_chunk": 32,
        "test_base_seeds": [3860000, 3870000, 3880000], "test_episodes_per_condition": 32,
        "reserved_s4d3_seed_base": 4000000, "test_generation_after_all_checkpoints_frozen": True,
    }:
        raise RuntimeError("A7 frozen episode scaling design changed")
    if experiment["comparison"] != {
        "bootstrap_replicates": 20000, "bootstrap_seed_offset": 540000,
        "familywise_alpha": 0.05, "family_size": 6, "minimum_balanced_accuracy_gain": 0.02,
    }:
        raise RuntimeError("A7 frozen statistical comparison changed")
    source_hashes: dict[str, str] = {}
    input_hashes: dict[str, str] = {}
    for section in ("design_contract", "upstream_a6", "frozen_collection"):
        for key, value in experiment[section].items():
            if key.endswith("_sha256"):
                _verify_hash(experiment[section][key[:-7]], str(value))
                input_hashes[experiment[section][key[:-7]]] = str(value)
    for section in ("upstream_a6", "frozen_collection"):
        for name, checksum in _read_json(experiment[section]["source_manifest"]).items():
            _verify_hash(name, checksum)
            source_hashes[name] = checksum
    a6_summary = _read_json(experiment["upstream_a6"]["summary"])
    if a6_summary["experiment"]["quick"] or a6_summary["interpretation"]["status"] != experiment["upstream_a6"]["required_status"]:
        raise RuntimeError("A7 upstream A6 evidence status changed")
    if len(a6_summary["fits"]) != 6:
        raise RuntimeError("A7 upstream A6 fit coverage changed")
    a6 = _load_yaml(_project_path(experiment["upstream_a6"]["experiment_config"]))
    source_path = _project_path(experiment["frozen_collection"]["experiment_config"])
    source = _load_yaml(source_path)
    # 逐级核对配置链，不能只锁最外层配置而放过内部物理参数。
    source_link = source
    for field in ("upstream_d1c", "upstream_d1b", "upstream_d1a", "upstream_d1"):
        contract = source_link[field]
        _verify_hash(contract["experiment_config"], contract["experiment_config_sha256"])
        input_hashes[contract["experiment_config"]] = contract["experiment_config_sha256"]
        source_link = _load_yaml(_project_path(contract["experiment_config"]))
    repair_settings = _collector_settings(source, quick=False)
    _, physical_experiment, checkpoints = preflight_s4_r3_multistep_critic_training(
        source_path, source, repair_settings, quick=False
    )
    checkpoints = [item for item in checkpoints if int(item["policy_seed"]) in settings["policy_seeds"]]
    if len(checkpoints) != len(settings["policy_seeds"]):
        raise RuntimeError("A7 missing frozen policies")
    for checkpoint in checkpoints:
        _verify_hash(checkpoint["path"], checkpoint["sha256"])
        _verify_hash(checkpoint["student_path"], checkpoint["student_sha256"])
        input_hashes[checkpoint["path"]] = checkpoint["sha256"]
        input_hashes[checkpoint["student_path"]] = checkpoint["student_sha256"]
    _verify_hash(physical_experiment["environment_config"], physical_experiment["environment_config_sha256"])
    input_hashes[physical_experiment["environment_config"]] = physical_experiment["environment_config_sha256"]
    protected: set[int] = set()
    for split in repair_settings["splits"].values():
        protected.update(_split_seeds(split))
    for split in _collector_settings(source, quick=True)["splits"].values():
        protected.update(_split_seeds(split))
    confirmation = _load_yaml(_project_path("configs/experiments/s4_r3_physical_target_confirmation_v1.yaml"))
    for data in (confirmation["confirmation_data"], confirmation["quick"]):
        protected.update(_split_seeds(data))
    # 同时验证正式和冒烟空间，避免冒烟提前使用新正式测试回合。
    formal_settings = _effective_settings(experiment, quick=False)
    formal_namespace = verify_new_seed_namespace(formal_settings, protected=protected, reserved=4000000)
    formal_seeds = set().union(*[_split_seeds(split) for split in [
        *formal_settings["added_splits"], formal_settings["test_split"]
    ]])
    quick_settings = _effective_settings(experiment, quick=True)
    quick_namespace = verify_new_seed_namespace(quick_settings, protected=protected | formal_seeds, reserved=4000000)
    old_data_records = []
    for seed in settings["policy_seeds"]:
        datasets = {}
        for split, episodes, pairs in (("development", 96, 1728), ("validation", 48, 864)):
            spec = a6["data"][split][seed]
            _verify_hash(spec["path"], spec["sha256"])
            input_hashes[spec["path"]] = spec["sha256"]
            datasets[split] = _load_pairs(spec)
            if len(_episode_seeds(datasets[split])) != episodes or len(datasets[split].rows) != pairs:
                raise RuntimeError("A7 upstream episode/sample counts changed")
            old_data_records.append(_pair_dataset_summary(seed, split, datasets[split]))
        verify_dataset_separation(datasets)
    expected_branches = len(checkpoints) * sum(
        len(split["profile_ids"]) * len(split["conditions"]) * len(split["probe_steps"]) * 2
        for split in [*settings["added_splits"], settings["test_split"]]
    )
    source_hashes.update(_source_manifest(experiment["tracked_source_files"]))
    source_hashes[_relative(config_path)] = _file_sha256(config_path)
    output = _project_path(settings["output_directory"])
    if output.exists():
        raise FileExistsError(f"A7 output already exists; do not overwrite or auto-retry: {output}")
    return {
        "status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_TRAINING",
        "quick": quick, "planned_probe_fits": len(checkpoints) * len(settings["objectives"]) * 2,
        "maximum_updates_per_fit": settings["maximum_updates"], "early_stopping": False,
        "formal_training_episode_counts": [96, 384], "formal_test_episodes": 96,
        "expected_branch_rollouts": expected_branches,
        "seed_namespace": quick_namespace if quick else formal_namespace,
        "upstream_data": old_data_records, "frozen_source_hashes": source_hashes,
        "frozen_input_hashes": input_hashes,
        "new_episodes_generated": False, "new_test_opened": False,
        "cuda_required": True, "full_rl_training": False, "s4d3_access": False,
        "real_slm_actions": False, "output_directory": _relative(output),
    }, physical_experiment, checkpoints


def fit_scaling_probe(
    *, development: RewardPairDataset, validation: RewardPairDataset,
    settings: dict[str, Any], policy_seed: int, scale: str, objective: dict[str, Any],
    initial_state: dict[str, torch.Tensor], output_directory: Path, device: torch.device,
) -> dict[str, Any]:
    """该函数没有测试集参数；固定预算，只用验证集选检查点。"""
    verify_dataset_separation({"development": development, "validation": validation})
    fit_dir = output_directory / f"seed_{policy_seed}" / scale / str(objective["id"])
    fit_dir.mkdir(parents=True, exist_ok=False)
    model = PairwiseDeltaProbe(settings["feature_size"], settings["hidden_size"]).to(device)
    model.load_state_dict(initial_state)
    normalization = _normalization_from_training(development, settings)
    norms = {key: value.to(device) for key, value in normalization.items()}
    x = (development.features.to(device) - norms["feature_mean"]) / norms["feature_scale"]
    y = development.reward_delta.to(device)
    positives = int((y > 0).sum())
    pos_weight = torch.tensor((len(y) - positives) / positives, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    generator = torch.Generator().manual_seed(settings["batch_order_seed_offset"] + policy_seed)
    seen = torch.zeros(len(y), dtype=torch.bool)
    best_score = (-float("inf"), float("inf"))
    best_update = 0
    recent: deque[float] = deque(maxlen=100)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    best_path, last_path = fit_dir / "checkpoint_best.pt", fit_dir / "checkpoint_last.pt"
    progress_path, loss_path = fit_dir / "progress.jsonl", fit_dir / "loss_history.csv"
    bar = progress_bar(range(1, settings["maximum_updates"] + 1),
                       description=f"A7 {policy_seed} {scale} {objective['id']}", unit="批")

    def payload(update: int, metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            "algorithm": "h16_episode_scaling_supervised_probe", "policy_seed": policy_seed,
            "scale": scale, "objective": deepcopy(objective), "update": update,
            "best_update": best_update, "validation_metrics": metrics,
            "config": {"feature_size": settings["feature_size"], "hidden_size": settings["hidden_size"]},
            "label_source": "empirical_reward_returns", "target_horizon": 16,
            "probe": model.state_dict(), "normalization": normalization,
            "training_reward_mean": float(development.reward_delta.mean()),
            "training_pairs": len(y), "training_episodes": len(_episode_seeds(development)),
            "independent_test_used_for_selection": False, "original_critic_updates": 0,
            "actor_updates": 0, "alpha_updates": 0, "student_updates": 0,
        }

    for update in bar:
        indices_cpu = torch.randint(len(y), (settings["batch_size"],), generator=generator)
        seen[indices_cpu] = True
        indices = indices_cpu.to(device)
        prediction = model(x[indices])
        regression = F.huber_loss(prediction, y[indices] / norms["target_scale"], delta=settings["huber_delta"])
        sign = F.binary_cross_entropy_with_logits(prediction, (y[indices] > 0).float(), pos_weight=pos_weight)
        loss = regression + float(objective["balanced_sign_weight"]) * sign
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("A7 non-finite training loss; retain logs and do not retry")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        recent.append(float(loss.detach()))
        if update % settings["validation_interval_updates"] != 0 and update != settings["maximum_updates"]:
            continue
        val_prediction = _predict_dataset(model, validation, **norms, device=device)
        metrics = classification_metrics(val_prediction, validation.reward_delta)
        score = (metrics["balanced_accuracy"], metrics["mae"])
        if score[0] > best_score[0] + 1e-12 or (abs(score[0] - best_score[0]) <= 1e-12 and score[1] < best_score[1]):
            best_score, best_update = score, update
            torch.save(payload(update, metrics), best_path)
        record = {
            "policy_seed": policy_seed, "scale": scale, "objective": objective["id"],
            "update": update, "total_updates": settings["maximum_updates"],
            "training_loss": float(loss.detach()), "mean_training_loss": sum(recent) / len(recent),
            "validation_balanced_accuracy": metrics["balanced_accuracy"],
            "validation_mae": metrics["mae"], "validation_mcc": metrics["matthews_correlation"],
            "best_update": best_update, "sample_presentations": update * settings["batch_size"],
            "equivalent_data_passes": update * settings["batch_size"] / len(y),
            "unique_training_pairs_seen": int(seen.sum()),
            **_runtime_fields(start=started, completed=update, total=settings["maximum_updates"], device=device),
        }
        records.append(record)
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_safe(record), ensure_ascii=False) + "\n")
        with loss_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record))
            if len(records) == 1:
                writer.writeheader()
            writer.writerow(record)
        update_progress(bar, device=device, metrics={"平均损失": record["mean_training_loss"],
                        "验证平衡准确率": metrics["balanced_accuracy"], "验证MCC": metrics["matthews_correlation"]})
    bar.close()
    torch.save(payload(settings["maximum_updates"], metrics), last_path)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["probe"])
    predictions = {name: _predict_dataset(model, data, **norms, device=device)
                   for name, data in (("training", development), ("validation", validation))}
    validation_metrics = evaluate_h16_prediction(
        predictions["validation"], validation, training_reward_mean=checkpoint["training_reward_mean"],
        magnitude_threshold=settings["high_magnitude_thresholds"][policy_seed],
        bootstrap_replicates=settings["cluster_bootstrap_replicates"],
        bootstrap_seed=settings["cluster_bootstrap_seed_offset"] + policy_seed,
        confidence_level=settings["confidence_level"],
    )
    return {
        "policy_seed": policy_seed, "scale": scale, "objective": objective["id"],
        "updates_completed": settings["maximum_updates"], "best_update": best_update,
        "best_near_budget_end": best_update >= settings["maximum_updates"] * settings["late_best_update_fraction"],
        "development_pairs": len(y), "training_episodes": len(_episode_seeds(development)),
        "validation_pairs": len(validation.rows), "training": classification_metrics(predictions["training"], development.reward_delta),
        "validation": validation_metrics, "duration_seconds": time.perf_counter() - started,
        "sample_presentations": settings["maximum_updates"] * settings["batch_size"],
        "equivalent_data_passes": settings["maximum_updates"] * settings["batch_size"] / len(y),
        "unique_training_pairs_seen": int(seen.sum()), "all_training_pairs_seen": bool(seen.all()),
        "checkpoint": _relative(best_path), "checkpoint_sha256": _file_sha256(best_path),
        "last_checkpoint": _relative(last_path), "last_checkpoint_sha256": _file_sha256(last_path),
        "progress_log": _relative(progress_path), "loss_history": _relative(loss_path),
        "progress_sha256": _file_sha256(progress_path), "loss_history_sha256": _file_sha256(loss_path),
        "independent_test_used_for_selection": False, "original_critic_updates": 0,
        "actor_updates": 0, "alpha_updates": 0, "student_updates": 0,
    }


def paired_episode_bootstrap(
    small_prediction: torch.Tensor, large_prediction: torch.Tensor, dataset: RewardPairDataset,
    *, replicates: int, seed: int, familywise_alpha: float, family_size: int,
) -> dict[str, Any]:
    """按条件分层、同回合配对抽样，重复档位/时刻不增加独立样本量。"""
    if small_prediction.shape != dataset.reward_delta.shape or large_prediction.shape != dataset.reward_delta.shape:
        raise ValueError("A7 paired prediction shapes do not align")
    if not 0 < familywise_alpha < 1 or family_size < 1 or replicates < 20:
        raise ValueError("A7 invalid bootstrap settings")
    labels = dataset.reward_delta.numpy() > 0
    predictions = np.stack([small_prediction.numpy() > 0, large_prediction.numpy() > 0])
    clusters: dict[tuple[str, int], list[int]] = defaultdict(list)
    seed_conditions: dict[int, str] = {}
    for i, row in enumerate(dataset.rows):
        condition, episode = str(row["condition_id"]), int(row["episode_seed"])
        if episode in seed_conditions and seed_conditions[episode] != condition:
            raise RuntimeError("A7 one episode assigned to multiple conditions")
        seed_conditions[episode] = condition
        clusters[condition, episode].append(i)
    counts: dict[str, list[np.ndarray]] = defaultdict(list)
    for (condition, _), indices in sorted(clusters.items()):
        truth, pred = labels[indices], predictions[:, indices]
        counts[condition].append(np.stack([
            (pred & truth).sum(axis=1), ((~pred) & truth).sum(axis=1),
            ((~pred) & (~truth)).sum(axis=1), (pred & (~truth)).sum(axis=1),
        ], axis=1))
    rng = np.random.default_rng(seed)
    totals = np.zeros((replicates, 2, 4), dtype=np.float64)
    for condition_counts in counts.values():
        matrix = np.stack(condition_counts)
        draws = rng.integers(len(matrix), size=(replicates, len(matrix)))
        totals += matrix[draws].sum(axis=1)
    positives, negatives = totals[:, :, 0] + totals[:, :, 1], totals[:, :, 2] + totals[:, :, 3]
    valid = (positives > 0).all(axis=1) & (negatives > 0).all(axis=1)
    if int(valid.sum()) < max(10, int(0.9 * replicates)):
        raise RuntimeError("A7 insufficient valid paired bootstrap draws")
    ba = 0.5 * (totals[valid, :, 0] / positives[valid] + totals[valid, :, 2] / negatives[valid])
    differences = ba[:, 1] - ba[:, 0]
    point = classification_metrics(large_prediction, dataset.reward_delta)["balanced_accuracy"] - classification_metrics(small_prediction, dataset.reward_delta)["balanced_accuracy"]
    tail = familywise_alpha / (2 * family_size)
    return {
        "balanced_accuracy_gain": float(point), "ci95_low": float(np.quantile(differences, 0.025)),
        "ci95_high": float(np.quantile(differences, 0.975)),
        "familywise_ci_low": float(np.quantile(differences, tail)),
        "familywise_ci_high": float(np.quantile(differences, 1 - tail)),
        "familywise_alpha": familywise_alpha, "family_size": family_size,
        "replicates": int(valid.sum()), "clusters": len(clusters),
        "unit": "complete_episode_seed", "stratified_by": "condition_id",
    }


def seal_training(fits: list[dict[str, Any]], *, settings: dict[str, Any], output: Path) -> Path:
    expected = {(seed, scale, objective["id"]) for seed in settings["policy_seeds"]
                for scale in SCALES for objective in settings["objectives"]}
    observed = {(fit["policy_seed"], fit["scale"], fit["objective"]) for fit in fits}
    if observed != expected or len(fits) != len(expected):
        raise RuntimeError("A7 cannot open test before all fits are complete")
    for fit in fits:
        if fit["updates_completed"] != settings["maximum_updates"]:
            raise RuntimeError("A7 training budget incomplete")
        _verify_hash(fit["checkpoint"], fit["checkpoint_sha256"])
    path = output / "TRAINING_FROZEN.json"
    if path.exists():
        raise FileExistsError(path)
    _write_json(path, {"created_at": datetime.now(timezone.utc).isoformat(),
                      "test_generated": False, "fits": fits})
    return path


def _collect_and_save(
    *, name: str, split: dict[str, Any], checkpoint: dict[str, Any],
    physical_experiment: dict[str, Any], base_config: Any, basis: torch.Tensor,
    settings: dict[str, Any], bar: Any, progress_path: Path, collection_started: float,
    dataset_directory: Path, device: torch.device,
) -> tuple[RewardPairDataset, dict[str, Any]]:
    _verify_hash(checkpoint["path"], checkpoint["sha256"])
    _verify_hash(checkpoint["student_path"], checkpoint["student_sha256"])
    policy = _load_extended_policy(checkpoint, device=device)
    student = _load_student(checkpoint, experiment=physical_experiment, device=device)
    raw = _collect_policy_split(
        split_name=name, split=split, experiment=physical_experiment, base_config=base_config,
        basis=basis, policy=policy, student=student, candidate_actions=settings["candidate_actions"],
        max_horizon=32, collection_bar=bar, collection_progress_path=progress_path,
        collection_started=collection_started, device=device,
    )
    # 原采集器的索引在每块从0开始；聚合后必须唯一，不能压缩独立回合数。
    for row in raw.rows:
        row["episode_index"] += int(split["episode_index_offset"])
    raw.validate(max_horizon=32)
    path = dataset_directory / f"seed_{policy.policy_seed}_{name}_raw.pt"
    torch.save(raw.payload(), path)
    dataset = _load_pairs({"path": _relative(path)})
    if _episode_seeds(dataset) != _split_seeds(split):
        raise RuntimeError("A7 generated seed metadata do not match the declared split")
    record = {**_pair_dataset_summary(policy.policy_seed, name, dataset),
              "path": _relative(path), "sha256": _file_sha256(path),
              "raw_horizons": 32, "training_target_horizon": 16}
    return dataset, record


def _checkpoint_predictions(fit: dict[str, Any], dataset: RewardPairDataset, *, device: torch.device) -> tuple[torch.Tensor, float]:
    _verify_hash(fit["checkpoint"], fit["checkpoint_sha256"])
    checkpoint = torch.load(_project_path(fit["checkpoint"]), map_location=device, weights_only=False)
    model = PairwiseDeltaProbe(checkpoint["config"]["feature_size"], checkpoint["config"]["hidden_size"]).to(device)
    model.load_state_dict(checkpoint["probe"])
    norms = {key: value.to(device) for key, value in checkpoint["normalization"].items()}
    return _predict_dataset(model, dataset, **norms, device=device), float(checkpoint["training_reward_mean"])


def scaling_decision(
    comparisons: list[dict[str, Any]], fits: list[dict[str, Any]],
    *, settings: dict[str, Any], quick: bool,
) -> dict[str, Any]:
    methods = []
    for objective in settings["objectives"]:
        selected = [row for row in comparisons if row["objective"] == objective["id"]]
        complete = len(selected) == len(settings["policy_seeds"]) and {
            row["policy_seed"] for row in selected
        } == set(settings["policy_seeds"])
        supports = complete and all(
            row["balanced_accuracy_gain"] >= settings["comparison"]["minimum_balanced_accuracy_gain"]
            and row["familywise_ci_low"] > 0 for row in selected
        )
        methods.append({"objective": objective["id"], "consistent_data_benefit": supports and not quick})
    benefit = any(item["consistent_data_benefit"] for item in methods)
    return {
        "status": "QUICK_SMOKE_ONLY" if quick else (
            "CONSISTENT_BENEFIT_FROM_MORE_EPISODES" if benefit else "DATA_BENEFIT_NOT_CONFIRMED_AT_THIS_BUDGET"
        ),
        "methods": methods, "budget_end_warning": any(fit["best_near_budget_end"] for fit in fits),
        "interpretation_limit": "Failure to confirm is not proof that more data cannot help; one nested training sample per policy seed.",
        "critic_design_authorized": False, "full_rl_authorized": False,
        "s4d3_authorized": False, "real_hardware_authorized": False,
    }


def run_s4_r3_h16_data_scaling(
    config_path: str | Path, *, quick: bool = False, preflight_only: bool = False
) -> dict[str, Any]:
    """正式采集与训练只能由用户在IDE启动；无覆盖、重试或续跑。"""
    config_path = _project_path(config_path)
    experiment = _load_yaml(config_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight, physical_experiment, checkpoints = preflight_s4_r3_h16_data_scaling(
        config_path, experiment, settings, quick=quick
    )
    if preflight_only:
        return preflight
    device = resolve_device(experiment["runtime"]["device"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    output = _project_path(settings["output_directory"])
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "preflight.json", preflight)
    _write_json(output / "effective_config.json", {"experiment": experiment, "settings": settings})
    _write_json(output / "source_manifest.json", preflight["frozen_source_hashes"])
    _write_json(output / "input_manifest.json", preflight["frozen_input_hashes"])
    try:
        return _execute_scaling(experiment, settings, preflight, physical_experiment, checkpoints,
                                output=output, device=device, quick=quick)
    except Exception as error:
        import traceback
        _write_json(output / "failure.json", {
            "exception": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(), "automatic_retry": False,
        })
        raise


def _execute_scaling(
    experiment: dict[str, Any], settings: dict[str, Any], preflight: dict[str, Any],
    physical_experiment: dict[str, Any], checkpoints: list[dict[str, Any]],
    *, output: Path, device: torch.device, quick: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    directory = output / "datasets"
    directory.mkdir()
    base_config, _ = load_s1_config(_project_path(physical_experiment["environment_config"]))
    representation = ActionRepresentation.from_mapping(physical_experiment["representation"])
    base_config = replace(base_config, num_modes=representation.num_modes)
    basis, _, basis_diagnostics = build_action_basis(base_config, representation, device)
    records: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    train_branch_count = len(checkpoints) * sum(
        len(split["profile_ids"]) * len(split["conditions"]) * len(split["probe_steps"]) * 2
        for split in settings["added_splits"]
    )
    collection_started = time.perf_counter()
    progress_message("A7 阶段1/5：新增训练回合采集（此阶段不是网络训练）")
    bar = counted_progress(total=train_branch_count, description="A7 新增训练回合", unit="批量分支")
    for checkpoint in checkpoints:
        seed = int(checkpoint["policy_seed"])
        small = _load_pairs(settings["datasets"]["development"][seed])
        validation = _load_pairs(settings["datasets"]["validation"][seed])
        if quick:
            small = _select_old_episodes(small, settings["old_episodes_per_condition"])
            validation = _select_old_episodes(validation, settings["old_episodes_per_condition"])
        parts = [small]
        for index, split in enumerate(settings["added_splits"]):
            addition, record = _collect_and_save(
                name=f"added_{index}", split=split, checkpoint=checkpoint,
                physical_experiment=physical_experiment, base_config=base_config, basis=basis,
                settings=settings, bar=bar, progress_path=output / "training_collection_progress.jsonl",
                collection_started=collection_started, dataset_directory=directory, device=device,
            )
            parts.append(addition)
            records.append(record)
            _write_json(output / "data_manifest.json", {"datasets": records})
        large = concatenate_pairs(parts)
        verify_dataset_separation({"training": large, "validation": validation})
        if not torch.equal(small.features, large.features[:len(small.rows)]) or not torch.equal(small.reward_delta, large.reward_delta[:len(small.rows)]):
            raise RuntimeError("A7 small dataset is not exactly nested in the large dataset")
        if not quick and (len(_episode_seeds(small)), len(_episode_seeds(large)), len(small.rows), len(large.rows)) != (96, 384, 1728, 6912):
            raise RuntimeError("A7 formal training coverage changed")
        for scale, dataset in (("small", small), ("large", large), ("validation", validation)):
            path = directory / f"seed_{seed}_{scale}_pairs.pt"
            _save_pairs(path, dataset)
            records.append({**_pair_dataset_summary(seed, scale, dataset),
                            "path": _relative(path), "sha256": _file_sha256(path)})
        _write_json(output / "data_manifest.json", {"datasets": records})
    bar.close()
    progress_message("A7 阶段2/5：两组等预算监督训练；新测试回合尚未生成")
    total_fits = len(checkpoints) * len(settings["objectives"]) * 2
    for checkpoint in checkpoints:
        seed = int(checkpoint["policy_seed"])
        validation = _read_pairs(directory / f"seed_{seed}_validation_pairs.pt")
        initial = _initial_probe_state(settings=settings, policy_seed=seed, device=device)
        for scale in SCALES:
            development = _read_pairs(directory / f"seed_{seed}_{scale}_pairs.pt")
            for objective in settings["objectives"]:
                progress_message(f"A7 模型 {len(fits) + 1}/{total_fits}，训练回合数={len(_episode_seeds(development))}")
                fits.append(fit_scaling_probe(
                    development=development, validation=validation, settings=settings,
                    policy_seed=seed, scale=scale, objective=objective, initial_state=initial,
                    output_directory=output, device=device,
                ))
                _write_json(output / "training_fits.json", {"fits": fits, "new_test_opened": False})
    progress_message("A7 阶段3/5：冻结全部最佳检查点")
    frozen_path = seal_training(fits, settings=settings, output=output)
    frozen_hash = _file_sha256(frozen_path)
    # 防止采集/训练中途代码或数据被更改后继续生成测试。
    for name, checksum in preflight["frozen_source_hashes"].items():
        _verify_hash(name, checksum)
    for name, checksum in preflight["frozen_input_hashes"].items():
        _verify_hash(name, checksum)
    for record in records:
        _verify_hash(record["path"], record["sha256"])
    _write_json(output / "NEW_TEST_OPENED.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "training_frozen_sha256": frozen_hash,
        "settings": settings["test_split"], "quick": quick,
    })
    progress_message("A7 阶段4/5：生成共同的新测试回合；不再训练或选择模型")
    test_started = time.perf_counter()
    bar = counted_progress(total=preflight["expected_branch_rollouts"] - train_branch_count,
                           description="A7 共同新测试回合", unit="批量分支")
    predictions: dict[tuple[int, str, str], torch.Tensor] = {}
    subgroup_rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        seed = int(checkpoint["policy_seed"])
        test, record = _collect_and_save(
            name="independent_test", split=settings["test_split"], checkpoint=checkpoint,
            physical_experiment=physical_experiment, base_config=base_config, basis=basis,
            settings=settings, bar=bar, progress_path=output / "test_collection_progress.jsonl",
            collection_started=test_started, dataset_directory=directory, device=device,
        )
        verify_dataset_separation({"training": _read_pairs(directory / f"seed_{seed}_large_pairs.pt"),
                                   "validation": _read_pairs(directory / f"seed_{seed}_validation_pairs.pt"), "test": test})
        if not quick and (len(_episode_seeds(test)), len(test.rows)) != (96, 1728):
            raise RuntimeError("A7 formal test coverage changed")
        records.append(record)
        _write_json(output / "data_manifest.json", {"datasets": records})
        for fit in [item for item in fits if item["policy_seed"] == seed]:
            prediction, training_mean = _checkpoint_predictions(fit, test, device=device)
            predictions[seed, fit["scale"], fit["objective"]] = prediction
            fit["independent_test"] = evaluate_h16_prediction(
                prediction, test, training_reward_mean=training_mean,
                magnitude_threshold=settings["high_magnitude_thresholds"][seed],
                bootstrap_replicates=settings["cluster_bootstrap_replicates"],
                bootstrap_seed=settings["cluster_bootstrap_seed_offset"] + 10000 + seed,
                confidence_level=settings["confidence_level"],
            )
            groups = independent_subgroup_metrics(prediction, test, policy_seed=seed, objective=fit["objective"])
            valid = [float(row["balanced_accuracy"]) for row in groups if row["balanced_accuracy"] is not None]
            fit["independent_test_pairs"] = len(test.rows)
            fit["independent_test_subgroup_minimum_balanced_accuracy"] = min(valid) if valid else None
            fit["independent_test_subgroups_all_two_class"] = len(valid) == len(groups)
            subgroup_rows.extend({"scale": fit["scale"], **row} for row in groups)
        for objective in settings["objectives"]:
            method = objective["id"]
            comparison = settings["comparison"]
            comparisons.append({"policy_seed": seed, "objective": method, **paired_episode_bootstrap(
                predictions[seed, "small", method], predictions[seed, "large", method], test,
                replicates=int(comparison["bootstrap_replicates"]),
                seed=int(comparison["bootstrap_seed_offset"]) + seed,
                familywise_alpha=float(comparison["familywise_alpha"]), family_size=int(comparison["family_size"]),
            )})
    bar.close()
    progress_message("A7 阶段5/5：写入两组配对比较和完整门槛；完成后等待只读审计")
    _verify_hash(frozen_path, frozen_hash)
    for fit in fits:
        _verify_hash(fit["checkpoint"], fit["checkpoint_sha256"])
    for name, checksum in preflight["frozen_source_hashes"].items():
        _verify_hash(name, checksum)
    for name, checksum in preflight["frozen_input_hashes"].items():
        _verify_hash(name, checksum)
    learnability = {}
    for scale in SCALES:
        result = interpret_h16_reward_fits([fit for fit in fits if fit["scale"] == scale], settings=settings, quick=quick)
        if not quick and not result["passed_objectives"]:
            result["status"] = "GATES_NOT_MET_IN_THIS_RUN"
        learnability[scale] = result
    _write_rows(output / "fit_summary.csv", [{"scale": fit["scale"], "training_episodes": fit["training_episodes"],
                "training_balanced_accuracy": fit["training"]["balanced_accuracy"],
                "equivalent_data_passes": fit["equivalent_data_passes"], **_fit_csv_row(fit)} for fit in fits])
    _write_rows(output / "scaling_comparison.csv", comparisons)
    _write_rows(output / "test_subgroups.csv", subgroup_rows)
    summary = {
        "material_passport": {"origin_skill": "academic-research-suite / experiment-agent", "origin_mode": "run",
            "origin_date": datetime.now(timezone.utc).isoformat(), "verification_status": "UNVERIFIED",
            "version_label": experiment["metadata"]["version_label"]},
        "experiment": {"id": experiment["metadata"]["experiment_id"], "quick": quick,
            "status": "quick_smoke_only" if quick else "completed_pending_audit",
            "duration_seconds": time.perf_counter() - started, "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device)},
        "contract": {"changed_factor_between_arms": "independent_training_episode_count",
            "fixed_updates_each": settings["maximum_updates"], "early_stopping": False,
            "target_horizon": 16, "target_source": "empirical_reward_returns",
            "test_generated_after_all_fits": True, "small_exactly_nested_in_large": True,
            "comparison_to_old_a6_scores_is_not_primary": True},
        "data": records, "fits": fits, "comparisons": comparisons,
        "interpretation": scaling_decision(comparisons, fits, settings=settings, quick=quick),
        "learnability_by_scale": learnability, "basis_diagnostics": basis_diagnostics,
        "training_frozen_sha256": frozen_hash,
        "records": {path.name: _file_sha256(path) for path in output.iterdir() if path.is_file()},
        "evidence_boundary": {"supervised_probe_training_only": True, "new_simulation_episodes_generated": True,
            "independent_test_used_for_training_or_selection": False, "mechanism_audit_accessed": False,
            "original_critic_updates": 0, "actor_updates": 0, "alpha_updates": 0, "student_updates": 0,
            "full_rl_trained": False, "s4d3_accessed": False, "real_slm_actions": False},
        "runtime": _runtime_record(), "git": _git_record(),
        "next_action": "Stop for read-only audit; do not train RL or rerun automatically.",
    }
    _write_json(output / "summary.json", summary)
    return json_safe(summary)
