"""A8：固定384回合，比较共享标量与分类/回归分离输出。"""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.rl.s4_r3_h16_data_scaling import (
    _collect_and_save, _effective_settings as _a7_settings, _episode_seeds,
    _read_json, _read_pairs, _select_old_episodes, _split_seeds, _verify_hash,
    _write_rows, preflight_s4_r3_h16_data_scaling, verify_dataset_separation,
)
from src.rl.s4_r3_h16_reward_pairwise_probe import (
    RewardPairDataset, _initial_probe_state, _normalization_from_training, _runtime_fields,
)
from src.rl.s4_r3_pairwise_rank_learnability import PairwiseDeltaProbe, classification_metrics
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import (
    PROJECT_ROOT, _file_sha256, _load_yaml, _project_path, _relative, _runtime_record,
    _source_manifest, _write_json, json_safe,
)
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.training_progress import counted_progress, progress_bar, progress_message, update_progress


ARMS = ("shared", "split")
NO_UPDATES = {"original_critic_updates": 0, "actor_updates": 0,
              "alpha_updates": 0, "student_updates": 0}


def _git_record_utf8() -> dict[str, Any]:
    """Git路径按UTF-8读取；中文文件名不得让科学汇总在末端失败。"""
    try:
        commit_output = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        ).stdout
        dirty_output = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        ).stdout
        return {
            "commit": (commit_output or "").strip() or None,
            "dirty": bool((dirty_output or "").strip()),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


class HeadSplitProbe(nn.Module):
    """两个隐藏层不变；分离组只增加一个257参数的末端线性映射。"""

    def __init__(self, feature_size: int, hidden_size: int, arm: str) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown A8 arm: {arm}")
        self.arm = arm
        self.network = PairwiseDeltaProbe(feature_size, hidden_size).network
        self.sign_head = deepcopy(self.network[4]) if arm == "split" else None

    def load_shared_initial(self, initial: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(initial, strict=False)
        expected = {"sign_head.weight", "sign_head.bias"} if self.arm == "split" else set()
        if set(missing) != expected or unexpected:
            raise RuntimeError("A8 initial state does not match the frozen A7 trunk")
        if self.sign_head is not None:
            self.sign_head.load_state_dict(self.network[4].state_dict())

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.network[:4](features)
        value = self.network[4](hidden).squeeze(-1)
        score = self.sign_head(hidden).squeeze(-1) if self.sign_head is not None else value
        return value, score


def _effective_settings(experiment: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    upstream = _load_yaml(_project_path(experiment["upstream_a7"]["experiment_config"]))
    settings = _a7_settings(upstream, quick=False)
    settings["quick"] = quick
    settings["output_directory"] = experiment["outputs"]["quick_directory" if quick else "directory"]
    settings["balanced_sign_weight"] = float(experiment["design"]["balanced_sign_weight"])
    settings["comparison"] = deepcopy(experiment["comparison"])
    settings["test_split"] = deepcopy(settings["test_split"])
    settings["test_split"]["episodes_per_condition"] = int(experiment["data"]["test_episodes_per_condition"])
    for condition, seed in zip(settings["test_split"]["conditions"], experiment["data"]["test_base_seeds"], strict=True):
        condition["base_seed"] = int(seed)
        condition["id"] = str(condition["id"]).replace("a7_test_", "a8_test_", 1)
    settings["added_splits"] = []
    if quick:
        reduced = experiment["quick"]
        settings["policy_seeds"] = list(reduced["policy_seeds"])
        for name in ("maximum_updates", "validation_interval_updates", "batch_size", "cluster_bootstrap_replicates"):
            settings[name] = int(reduced[name])
        settings["comparison"]["bootstrap_replicates"] = int(reduced["comparison_bootstrap_replicates"])
        settings["old_episodes_per_condition"] = int(reduced["old_episodes_per_condition"])
        split = settings["test_split"]
        split["profile_ids"] = list(reduced["profile_ids"])
        split["probe_steps"] = list(reduced["probe_steps"])
        split["episodes_per_condition"] = int(reduced["test_episodes_per_condition"])
        for condition in split["conditions"]:
            condition["base_seed"] += int(reduced["test_seed_offset"])
    return settings


def _validate_contract(experiment: dict[str, Any]) -> None:
    metadata = experiment["metadata"]
    if metadata["stage"] != "S4-D2-R3-D2-A8":
        raise RuntimeError("A8 stage changed")
    for field in ("diagnostic_only", "allow_new_test_generation", "allow_supervised_probe_training"):
        if metadata[field] is not True:
            raise RuntimeError(f"A8 required flag changed: {field}")
    for field in ("allow_new_training_data", "allow_full_rl_training", "allow_original_critic_updates",
                  "allow_actor_updates", "allow_alpha_updates", "allow_student_updates",
                  "allow_s4d3_access", "allow_real_hardware_actions"):
        if metadata[field] is not False:
            raise RuntimeError(f"A8 safety flag changed: {field}")
    if experiment["design"] != {
        "arms": ["shared", "split"], "training_episodes": 384, "maximum_updates": 10000,
        "early_stopping": False, "balanced_sign_weight": 0.25,
        "selection": "validation_ranking_ba_then_value_mae", "initial_heads_identical": True,
    }:
        raise RuntimeError("A8 single-factor design changed")
    if experiment["data"] != {
        "test_base_seeds": [3861000, 3871000, 3881000], "test_episodes_per_condition": 32,
        "test_after_all_fits_frozen": True, "reserved_s4d3_seed_base": 4000000,
    }:
        raise RuntimeError("A8 test isolation contract changed")
    if experiment["comparison"] != {
        "bootstrap_replicates": 20000, "bootstrap_seed_offset": 550000,
        "familywise_alpha": 0.05, "family_size": 6,
        "minimum_relative_mae_reduction": 0.05, "ranking_noninferiority_margin": 0.02,
    }:
        raise RuntimeError("A8 statistical contract changed")
    runtime = experiment["runtime"]
    if runtime["require_cuda"] is not True or runtime["deterministic_algorithms"] is not True:
        raise RuntimeError("A8 requires deterministic CUDA")
    if resolve_device(runtime["device"]).type != "cuda":
        raise RuntimeError("A8 requires CUDA")


def preflight_head_split(
    config_path: Path, experiment: dict[str, Any], settings: dict[str, Any], *, quick: bool
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """只核对已有证据和种子，不生成新测试、载入旧测试样本或训练。"""
    _validate_contract(experiment)
    pinned: dict[str, str] = {}
    for section in ("design_contract", "upstream_a7"):
        for key, checksum in experiment[section].items():
            if key.endswith("_sha256"):
                name = experiment[section][key[:-7]]
                _verify_hash(name, checksum)
                pinned[name] = checksum
    upstream = _load_yaml(_project_path(experiment["upstream_a7"]["experiment_config"]))
    old_settings = _a7_settings(upstream, quick=False)
    # 复用A7的只读物理/配置链核对；目标设为本次目录，绝不重启A7。
    old_settings["output_directory"] = settings["output_directory"]
    parent, physical, checkpoints = preflight_s4_r3_h16_data_scaling(
        _project_path(experiment["upstream_a7"]["experiment_config"]), upstream, old_settings, quick=False
    )
    summary = _read_json(experiment["upstream_a7"]["summary"])
    if summary["experiment"]["quick"] or summary["interpretation"]["status"] != experiment["upstream_a7"]["required_status"]:
        raise RuntimeError("A8 requires the audited formal A7 result")
    if len(summary["fits"]) != 12 or any(fit["updates_completed"] != 10000 for fit in summary["fits"]):
        raise RuntimeError("A8 upstream A7 budget/coverage incomplete")
    records = _read_json(experiment["upstream_a7"]["data_manifest"])["datasets"]
    selected = [record for record in records if record["split"] in ("large", "validation")]
    if len(selected) != 6 or len({(r["policy_seed"], r["split"]) for r in selected}) != 6:
        raise RuntimeError("A8 requires six distinct frozen train/validation datasets")
    protected: set[int] = set()
    for mode in (False, True):
        previous = _a7_settings(upstream, quick=mode)
        for split in [*previous["added_splits"], previous["test_split"]]:
            protected.update(_split_seeds(split))
    dataset_specs: dict[int, dict[str, Any]] = defaultdict(dict)
    for seed in (9301, 9302, 9303):
        data = {}
        for split, episodes, pairs in (("large", 384, 6912), ("validation", 48, 864)):
            record = next(r for r in selected if r["policy_seed"] == seed and r["split"] == split)
            _verify_hash(record["path"], record["sha256"])
            pinned[record["path"]] = record["sha256"]
            data[split] = _read_pairs(_project_path(record["path"]))
            validate_coverage(data[split], episodes=episodes, pairs=pairs)
            protected.update(_episode_seeds(data[split]))
            dataset_specs[seed][split] = record
        verify_dataset_separation(data)
    namespaces = []
    for mode in (False, True):
        split = _effective_settings(experiment, quick=mode)["test_split"]
        seeds = _split_seeds(split)
        if min(seeds) < 3861000 or max(seeds) >= 3890000 or protected & seeds:
            raise RuntimeError("A8 test seeds overlap old data or reserved namespaces")
        protected.update(seeds)
        namespaces.append({"quick": mode, "episodes": len(seeds), "minimum": min(seeds), "maximum": max(seeds)})
    references = [f for f in summary["fits"] if f["scale"] == "large" and f["objective"] == "paired_delta_plus_balanced_sign"]
    if {f["policy_seed"] for f in references} != {9301, 9302, 9303}:
        raise RuntimeError("A8 missing control references")
    for fit in references:
        _verify_hash(fit["checkpoint"], fit["checkpoint_sha256"])
        pinned[fit["checkpoint"]] = fit["checkpoint_sha256"]
    sources = {**parent["frozen_source_hashes"], **_source_manifest(experiment["tracked_source_files"])}
    sources[_relative(config_path)] = _file_sha256(config_path)
    output = _project_path(settings["output_directory"])
    if output.exists():
        raise FileExistsError(f"A8 output exists; do not overwrite or retry: {output}")
    checkpoints = [c for c in checkpoints if c["policy_seed"] in settings["policy_seeds"]]
    split = settings["test_split"]
    return {
        "status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_TRAINING", "quick": quick,
        "planned_fits": len(checkpoints) * len(ARMS), "maximum_updates_per_fit": settings["maximum_updates"],
        "formal_training_episodes": 384, "formal_validation_episodes": 48, "formal_test_episodes": 96,
        "parameter_counts": {arm: sum(p.numel() for p in HeadSplitProbe(221, 256, arm).parameters()) for arm in ARMS},
        "test_namespace": namespaces[1 if quick else 0], "data_specs": dict(dataset_specs),
        "control_references": references, "frozen_source_hashes": sources,
        "frozen_input_hashes": {**parent["frozen_input_hashes"], **pinned},
        "expected_test_branches": len(checkpoints) * len(split["conditions"]) * len(split["profile_ids"]) * len(split["probe_steps"]) * 2,
        "new_test_opened": False, "new_training_data_generated": False, "output_directory": _relative(output),
        "full_rl_training": False, "s4d3_access": False, "real_slm_actions": False,
    }, physical, checkpoints


def validate_coverage(dataset: RewardPairDataset, *, episodes: int, pairs: int) -> None:
    dataset.validate(feature_size=221)
    if len(_episode_seeds(dataset)) != episodes or len(dataset.rows) != pairs:
        raise RuntimeError("A8 dataset coverage changed")
    keys = {(r["episode_seed"], r["profile_id"], r["probe_step"]) for r in dataset.rows}
    if len(keys) != pairs:
        raise RuntimeError("A8 duplicated episode/profile/probe rows")
    if pairs // episodes == 18:
        for seed in _episode_seeds(dataset):
            rows = [r for r in dataset.rows if r["episode_seed"] == seed]
            if len(rows) != 18 or {r["probe_step"] for r in rows} != {0, 80, 160}:
                raise RuntimeError("A8 lost a full episode or startup samples")


def ranking_metrics(score: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    """分类分数没有奖励单位；绝不为它报告奖励MAE。"""
    positive = target > 0
    if not bool(positive.any()) or not bool((~positive).any()):
        return {"status": "SINGLE_CLASS", "balanced_accuracy": None,
                "matthews_correlation": None, "positive_class_fraction": float(positive.float().mean())}
    result: dict[str, Any] = classification_metrics(score, target)
    del result["mae"], result["rmse"]
    guessed = score > 0
    result["positive_precision"] = float(positive[guessed].float().mean()) if bool(guessed.any()) else None
    result["status"] = "OK"
    return result


def measure_outputs(
    value: torch.Tensor, score: torch.Tensor, data: RewardPairDataset, training_mean: float
) -> dict[str, Any]:
    if value.shape != data.reward_delta.shape or score.shape != value.shape:
        raise ValueError("A8 output/label shapes differ")
    if not bool(torch.isfinite(value).all() and torch.isfinite(score).all()):
        raise RuntimeError("A8 non-finite predictions")
    error = value.double() - data.reward_delta.double()
    constant_mae = float((data.reward_delta.double() - training_mean).abs().mean())
    return {
        "ranking": ranking_metrics(score, data.reward_delta),
        "value": {"mae": float(error.abs().mean()), "rmse": float(error.square().mean().sqrt()),
                  "bias": float(error.mean()), "constant_training_mean_mae": constant_mae,
                  "mae_better_than_constant": float(error.abs().mean()) < constant_mae},
        "value_sign_ranking": ranking_metrics(value, data.reward_delta),
        "head_sign_disagreement_fraction": float(((value > 0) != (score > 0)).float().mean()),
        "physical_power_ranking": ranking_metrics(score, data.power_delta),
    }


@torch.no_grad()
def predict_outputs(
    model: HeadSplitProbe, data: RewardPairDataset, norms: dict[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    value, score = model((data.features.to(device) - norms["feature_mean"]) / norms["feature_scale"])
    return (value * norms["target_scale"]).cpu(), score.cpu()


def fit_head_probe(
    *, training: RewardPairDataset, validation: RewardPairDataset, settings: dict[str, Any],
    policy_seed: int, arm: str, initial: dict[str, torch.Tensor], output: Path, device: torch.device,
) -> dict[str, Any]:
    """无测试参数；两组相同初始化、样本顺序、更新次数与选择规则。"""
    training.validate(feature_size=settings["feature_size"])
    validation.validate(feature_size=settings["feature_size"])
    verify_dataset_separation({"training": training, "validation": validation})
    directory = output / f"seed_{policy_seed}" / arm
    directory.mkdir(parents=True, exist_ok=False)
    model = HeadSplitProbe(settings["feature_size"], settings["hidden_size"], arm).to(device)
    model.load_shared_initial(initial)
    normalization = _normalization_from_training(training, settings)
    norms = {k: v.to(device) for k, v in normalization.items()}
    x = (training.features.to(device) - norms["feature_mean"]) / norms["feature_scale"]
    y = training.reward_delta.to(device)
    positives = int((y > 0).sum())
    weight = torch.tensor((len(y) - positives) / positives, device=device)
    training_mean = float(training.reward_delta.mean())
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    generator = torch.Generator().manual_seed(settings["batch_order_seed_offset"] + policy_seed)
    seen = torch.zeros(len(y), dtype=torch.bool)
    best_score, best_update = (-float("inf"), float("inf")), 0
    recent: deque[float] = deque(maxlen=100)
    started = time.perf_counter()
    best_path, last_path = directory / "checkpoint_best.pt", directory / "checkpoint_last.pt"
    history_path, progress_path = directory / "loss_history.csv", directory / "progress.jsonl"

    def payload(update: int) -> dict[str, Any]:
        return {"algorithm": "h16_shared_vs_split_output_probe", "arm": arm, "policy_seed": policy_seed,
                "update": update, "best_update": best_update, "probe": model.state_dict(),
                "normalization": normalization, "training_reward_mean": training_mean,
                "config": {"feature_size": settings["feature_size"], "hidden_size": settings["hidden_size"]},
                "label_source": "empirical_reward_returns", "target_horizon": 16,
                "ranking_score_is_reward": False, "independent_test_used_for_selection": False, **NO_UPDATES}

    bar = progress_bar(range(1, settings["maximum_updates"] + 1), description=f"A8 {policy_seed} {arm}", unit="批")
    with history_path.open("w", encoding="utf-8", newline="") as history, progress_path.open("w", encoding="utf-8") as progress:
        writer = None
        for update in bar:
            indices_cpu = torch.randint(len(y), (settings["batch_size"],), generator=generator)
            seen[indices_cpu] = True
            indices = indices_cpu.to(device)
            model.train()
            value, score = model(x[indices])
            regression = F.huber_loss(value, y[indices] / norms["target_scale"], delta=settings["huber_delta"])
            sign = F.binary_cross_entropy_with_logits(score, (y[indices] > 0).float(), pos_weight=weight)
            loss = regression + settings["balanced_sign_weight"] * sign
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("A8 non-finite loss; preserve outputs and do not retry")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            recent.append(float(loss.detach()))
            if update % settings["validation_interval_updates"] and update != settings["maximum_updates"]:
                continue
            prediction, logits = predict_outputs(model, validation, norms, device)
            metrics = measure_outputs(prediction, logits, validation, training_mean)
            current = (metrics["ranking"]["balanced_accuracy"], metrics["value"]["mae"])
            if current[0] > best_score[0] + 1e-12 or (abs(current[0] - best_score[0]) <= 1e-12 and current[1] < best_score[1]):
                best_score, best_update = current, update
                torch.save(payload(update), best_path)
            record = {"policy_seed": policy_seed, "arm": arm, "update": update,
                      "total_updates": settings["maximum_updates"], "mean_training_loss": sum(recent) / len(recent),
                      "regression_loss": float(regression.detach()), "classification_loss": float(sign.detach()),
                      "validation_balanced_accuracy": current[0], "validation_value_mae": current[1],
                      "validation_mcc": metrics["ranking"]["matthews_correlation"],
                      "validation_value_sign_ba": metrics["value_sign_ranking"]["balanced_accuracy"],
                      "validation_head_disagreement": metrics["head_sign_disagreement_fraction"],
                      "best_update": best_update, "unique_training_pairs_seen": int(seen.sum()),
                      "sample_presentations": update * settings["batch_size"],
                      "equivalent_data_passes": update * settings["batch_size"] / len(y),
                      **_runtime_fields(start=started, completed=update, total=settings["maximum_updates"], device=device)}
            if writer is None:
                writer = csv.DictWriter(history, fieldnames=list(record))
                writer.writeheader()
            writer.writerow(record)
            history.flush()
            progress.write(json.dumps(record) + "\n")
            progress.flush()
            update_progress(bar, device=device, metrics={"平均损失": record["mean_training_loss"],
                            "判断得分": current[0], "收益误差": current[1]})
    bar.close()
    torch.save(payload(settings["maximum_updates"]), last_path)
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["probe"])
    fit: dict[str, Any] = {
        "policy_seed": policy_seed, "arm": arm, "updates_completed": settings["maximum_updates"],
        "best_update": best_update, "parameter_count": sum(p.numel() for p in model.parameters()),
        "training_episodes": len(_episode_seeds(training)), "training_pairs": len(training.rows),
        "all_training_pairs_seen": bool(seen.all()), "unique_training_pairs_seen": int(seen.sum()),
        "checkpoint": _relative(best_path), "checkpoint_sha256": _file_sha256(best_path),
        "last_checkpoint": _relative(last_path), "last_checkpoint_sha256": _file_sha256(last_path),
        "progress_log": _relative(progress_path), "progress_sha256": _file_sha256(progress_path),
        "loss_history": _relative(history_path), "loss_history_sha256": _file_sha256(history_path),
        "duration_seconds": time.perf_counter() - started, "independent_test_used_for_selection": False,
        "predictions": {}, **NO_UPDATES,
    }
    for name, dataset in (("training", training), ("validation", validation)):
        value, score = predict_outputs(model, dataset, norms, device)
        fit[name] = evaluate_outputs(value, score, dataset, training_mean, settings, policy_seed)
        fit["predictions"][name] = save_predictions(directory, name, value, score, dataset)
    return fit


def _cluster_sums(
    arrays: np.ndarray, rows: list[dict[str, Any]], *, replicates: int, seed: int
) -> np.ndarray:
    """按条件分层，以全局回合种子配对重采样，保留所有档位和时刻。"""
    if len(arrays) != len(rows) or not np.isfinite(arrays).all():
        raise ValueError("A8 invalid cluster arrays")
    by_key: dict[tuple[str, int], list[int]] = defaultdict(list)
    conditions: dict[int, str] = {}
    for index, row in enumerate(rows):
        condition, episode = str(row["condition_id"]), int(row["episode_seed"])
        if episode in conditions and conditions[episode] != condition:
            raise RuntimeError("A8 one episode assigned to multiple conditions")
        conditions[episode] = condition
        by_key[condition, episode].append(index)
    strata: dict[str, list[np.ndarray]] = defaultdict(list)
    for (condition, _), indices in sorted(by_key.items()):
        strata[condition].append(arrays[indices].sum(axis=0))
    rng = np.random.default_rng(seed)
    total = np.zeros((replicates, arrays.shape[1]), dtype=np.float64)
    for values in strata.values():
        matrix = np.stack(values)
        total += matrix[rng.integers(len(matrix), size=(replicates, len(matrix)))].sum(axis=1)
    return total


def _confusion_array(score: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    prediction, positive = score.detach().cpu().numpy() > 0, truth.detach().cpu().numpy() > 0
    return np.stack([prediction & positive, ~prediction & positive,
                     ~prediction & ~positive, prediction & ~positive], axis=1).astype(float)


def _ba_from_counts(counts: np.ndarray) -> np.ndarray:
    return 0.5 * (counts[:, 0] / (counts[:, 0] + counts[:, 1]) + counts[:, 2] / (counts[:, 2] + counts[:, 3]))


def evaluate_outputs(
    value: torch.Tensor, score: torch.Tensor, data: RewardPairDataset, training_mean: float,
    settings: dict[str, Any], seed: int,
) -> dict[str, Any]:
    metrics = measure_outputs(value, score, data, training_mean)
    counts = _cluster_sums(_confusion_array(score, data.reward_delta), data.rows,
                          replicates=settings["cluster_bootstrap_replicates"],
                          seed=settings["cluster_bootstrap_seed_offset"] + seed)
    valid = (counts[:, :2].sum(axis=1) > 0) & (counts[:, 2:].sum(axis=1) > 0)
    if valid.sum() < max(10, 0.9 * len(counts)):
        raise RuntimeError("A8 insufficient two-class bootstrap draws")
    values = _ba_from_counts(counts[valid])
    tail = (1 - settings["confidence_level"]) / 2
    metrics["ranking_ci"] = {"low": float(np.quantile(values, tail)), "high": float(np.quantile(values, 1 - tail)),
                             "clusters": len(_episode_seeds(data)), "unit": "episode_seed", "stratified_by": "condition_id"}
    return metrics


def compare_heads(
    shared: tuple[torch.Tensor, torch.Tensor], split: tuple[torch.Tensor, torch.Tensor],
    data: RewardPairDataset, settings: dict[str, Any], seed: int,
) -> dict[str, Any]:
    if any(t.shape != data.reward_delta.shape or not bool(torch.isfinite(t).all()) for pair in (shared, split) for t in pair):
        raise ValueError("A8 paired predictions do not align")
    cfg = settings["comparison"]
    errors = np.stack([(pair[0].double() - data.reward_delta.double()).abs().numpy() for pair in (shared, split)], axis=1)
    counts = np.concatenate([_confusion_array(pair[1], data.reward_delta) for pair in (shared, split)], axis=1)
    totals = _cluster_sums(np.concatenate([counts, errors], axis=1), data.rows,
                           replicates=cfg["bootstrap_replicates"], seed=cfg["bootstrap_seed_offset"] + seed)
    valid = np.ones(len(totals), dtype=bool)
    for start in (0, 4):
        valid &= (totals[:, start:start + 2].sum(axis=1) > 0) & (totals[:, start + 2:start + 4].sum(axis=1) > 0)
    valid &= totals[:, 8] > 0
    if valid.sum() < max(10, 0.9 * len(totals)) or errors[:, 0].mean() <= 0:
        raise RuntimeError("A8 insufficient valid paired bootstrap draws")
    totals = totals[valid]
    distributions = {
        "ranking_ba_delta": _ba_from_counts(totals[:, 4:8]) - _ba_from_counts(totals[:, :4]),
        "relative_mae_reduction": 1 - totals[:, 9] / totals[:, 8],
    }
    point = {"ranking_ba_delta": ranking_metrics(split[1], data.reward_delta)["balanced_accuracy"] - ranking_metrics(shared[1], data.reward_delta)["balanced_accuracy"],
             "relative_mae_reduction": float(1 - errors[:, 1].mean() / errors[:, 0].mean())}
    tail = cfg["familywise_alpha"] / (2 * cfg["family_size"])
    return {"policy_seed": seed, "clusters": len(_episode_seeds(data)), "unit": "complete_episode_seed",
            "stratified_by": "condition_id", "replicates": len(totals), "family_size": cfg["family_size"],
            "familywise_alpha": cfg["familywise_alpha"],
            **{name: {"estimate": point[name], "ci95_low": float(np.quantile(values, .025)),
                      "ci95_high": float(np.quantile(values, .975)),
                      "familywise_ci_low": float(np.quantile(values, tail)),
                      "familywise_ci_high": float(np.quantile(values, 1 - tail))}
               for name, values in distributions.items()}}


def subgroup_outputs(
    value: torch.Tensor, score: torch.Tensor, data: RewardPairDataset, training_mean: float,
    *, policy_seed: int, arm: str,
) -> list[dict[str, Any]]:
    results = []
    for field in ("profile_id", "condition_id", "probe_step"):
        for group in sorted({r[field] for r in data.rows}, key=str):
            indices = [i for i, row in enumerate(data.rows) if row[field] == group]
            subset = RewardPairDataset(data.features[indices], data.reward_delta[indices], data.power_delta[indices], [data.rows[i] for i in indices])
            metrics = measure_outputs(value[indices], score[indices], subset, training_mean)
            results.append({"policy_seed": policy_seed, "arm": arm, "group_kind": field,
                            "group_id": group, "samples": len(indices), "episodes": len(_episode_seeds(subset)),
                            "balanced_accuracy": metrics["ranking"]["balanced_accuracy"],
                            "two_class": metrics["ranking"]["status"] == "OK", "value_mae": metrics["value"]["mae"],
                            "head_sign_disagreement_fraction": metrics["head_sign_disagreement_fraction"]})
    return results


def save_predictions(directory: Path, name: str, value: torch.Tensor, score: torch.Tensor, data: RewardPairDataset) -> dict[str, str]:
    path = directory / f"predictions_{name}.pt"
    if path.exists():
        raise FileExistsError(path)
    torch.save({"reward_prediction": value, "ranking_logit": score, "reward_delta": data.reward_delta,
                "power_delta": data.power_delta, "rows": data.rows}, path)
    return {"path": _relative(path), "sha256": _file_sha256(path)}


def checkpoint_outputs(fit: dict[str, Any], data: RewardPairDataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
    _verify_hash(fit["checkpoint"], fit["checkpoint_sha256"])
    checkpoint = torch.load(_project_path(fit["checkpoint"]), map_location=device, weights_only=False)
    model = HeadSplitProbe(**checkpoint["config"], arm=checkpoint["arm"]).to(device)
    model.load_state_dict(checkpoint["probe"])
    norms = {k: v.to(device) for k, v in checkpoint["normalization"].items()}
    value, score = predict_outputs(model, data, norms, device)
    return value, score, float(checkpoint["training_reward_mean"])


def verify_control_parity(fit: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """共享组须复现冻结A7大数据联合损失模型，不读取A7测试样本。"""
    _verify_hash(fit["checkpoint"], fit["checkpoint_sha256"])
    _verify_hash(reference["checkpoint"], reference["checkpoint_sha256"])
    current = torch.load(_project_path(fit["checkpoint"]), map_location="cpu", weights_only=False)["probe"]
    previous = torch.load(_project_path(reference["checkpoint"]), map_location="cpu", weights_only=False)["probe"]
    if current.keys() != previous.keys():
        raise RuntimeError("A8 shared control parameter keys changed")
    difference = max(float((current[k] - previous[k]).abs().max()) for k in current)
    if fit["best_update"] != reference["best_update"] or difference > 1e-7:
        raise RuntimeError("A8 shared control did not reproduce A7; test remains closed")
    return {"matched": True, "max_parameter_difference": difference, "reference_sha256": reference["checkpoint_sha256"]}


def seal_heads(fits: list[dict[str, Any]], settings: dict[str, Any], output: Path) -> Path:
    expected = {(seed, arm) for seed in settings["policy_seeds"] for arm in ARMS}
    if len(fits) != len(expected) or {(f["policy_seed"], f["arm"]) for f in fits} != expected:
        raise RuntimeError("A8 cannot open test before all fits are complete")
    for fit in fits:
        if fit["updates_completed"] != settings["maximum_updates"]:
            raise RuntimeError("A8 incomplete training budget")
        if not settings["quick"] and fit["arm"] == "shared" and not fit.get("control_parity", {}).get("matched"):
            raise RuntimeError("A8 shared control parity not verified")
        _verify_hash(fit["checkpoint"], fit["checkpoint_sha256"])
    path = output / "TRAINING_FROZEN.json"
    if path.exists():
        raise FileExistsError(path)
    _write_json(path, {"created_at": datetime.now(timezone.utc).isoformat(), "new_test_opened": False, "fits": fits})
    return path


def interpret_heads(fits: list[dict[str, Any]], comparisons: list[dict[str, Any]], settings: dict[str, Any]) -> dict[str, Any]:
    cfg = settings["comparison"]
    expected = {(seed, arm) for seed in settings["policy_seeds"] for arm in ARMS}
    if {(f["policy_seed"], f["arm"]) for f in fits} != expected or len(fits) != len(expected):
        raise RuntimeError("A8 incomplete fit interpretation")
    if {c["policy_seed"] for c in comparisons} != set(settings["policy_seeds"]) or len(comparisons) != len(settings["policy_seeds"]):
        raise RuntimeError("A8 incomplete paired comparisons")
    benefits = [{"policy_seed": row["policy_seed"], "passed": (
        row["relative_mae_reduction"]["estimate"] >= cfg["minimum_relative_mae_reduction"]
        and row["relative_mae_reduction"]["familywise_ci_low"] > 0
        and row["ranking_ba_delta"]["familywise_ci_low"] > -cfg["ranking_noninferiority_margin"]
    )} for row in comparisons]
    gates = []
    for fit in fits:
        checks = {}
        for split in ("validation", "independent_test"):
            metrics = fit[split]
            checks[split] = (metrics["ranking"]["balanced_accuracy"] >= settings["minimum_balanced_accuracy"]
                and metrics["ranking"]["matthews_correlation"] >= settings["minimum_matthews_correlation"]
                and metrics["ranking_ci"]["low"] > settings["minimum_balanced_accuracy_ci_low"]
                and metrics["value"]["mae_better_than_constant"])
        original_groups = [row for row in fit["test_subgroups"] if row["group_kind"] != "probe_step"]
        coverage = (len(original_groups) == 9
                    and sum(row["group_kind"] == "profile_id" for row in original_groups) == 6
                    and len({(row["group_kind"], row["group_id"]) for row in original_groups}) == 9)
        checks["original_profile_condition_groups"] = coverage and all(
            row["two_class"] and row["balanced_accuracy"] >= settings["minimum_subgroup_balanced_accuracy"]
            for row in original_groups)
        gates.append({"policy_seed": fit["policy_seed"], "arm": fit["arm"], "checks": checks,
                      "passed": all(checks.values())})
    benefit = all(r["passed"] for r in benefits)
    return {
        "status": "QUICK_SMOKE_ONLY" if settings["quick"] else (
            "CONSISTENT_HEAD_SPLIT_BENEFIT" if benefit else "HEAD_SPLIT_BENEFIT_NOT_CONFIRMED"),
        "paired_benefit": benefits, "multitask_probe_gates": gates,
        "all_split_probe_gates_pass": not settings["quick"] and all(g["passed"] for g in gates if g["arm"] == "split"),
        "scalar_q_gate_equivalent": False, "critic_design_authorized": False,
        "full_rl_authorized": False, "s4d3_authorized": False, "real_hardware_authorized": False,
    }


def run_s4_r3_h16_head_split(config_path: str | Path, *, quick: bool = False, preflight_only: bool = False) -> dict[str, Any]:
    config_path = _project_path(config_path)
    experiment = _load_yaml(config_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight, physical, policies = preflight_head_split(config_path, experiment, settings, quick=quick)
    if preflight_only:
        return preflight
    device = resolve_device(experiment["runtime"]["device"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    output = _project_path(settings["output_directory"])
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("preflight", preflight), ("effective_config", {"experiment": experiment, "settings": settings}),
                        ("source_manifest", preflight["frozen_source_hashes"]), ("input_manifest", preflight["frozen_input_hashes"])):
        _write_json(output / f"{name}.json", value)
    try:
        return _execute(experiment, settings, preflight, physical, policies, output, device)
    except Exception as error:
        import traceback
        _write_json(output / "failure.json", {"exception": type(error).__name__, "message": str(error),
                    "traceback": traceback.format_exc(), "automatic_retry": False})
        raise


def _execute(
    experiment: dict[str, Any], settings: dict[str, Any], preflight: dict[str, Any],
    physical: dict[str, Any], policies: list[dict[str, Any]], output: Path, device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    fits: list[dict[str, Any]] = []
    data: dict[int, dict[str, RewardPairDataset]] = {}
    progress_message("A8 阶段1/4：复用冻结训练/验证数据；新测试尚未生成")
    for seed in settings["policy_seeds"]:
        data[seed] = {name: _read_pairs(_project_path(spec["path"])) for name, spec in preflight["data_specs"][seed].items()}
        if settings["quick"]:
            data[seed] = {name: _select_old_episodes(dataset, settings["old_episodes_per_condition"]) for name, dataset in data[seed].items()}
        initial = _initial_probe_state(settings=settings, policy_seed=seed, device=device)
        for arm in ARMS:
            progress_message(f"A8 模型 {len(fits) + 1}/{preflight['planned_fits']}：{seed} / {arm}")
            fit = fit_head_probe(training=data[seed]["large"], validation=data[seed]["validation"], settings=settings,
                                 policy_seed=seed, arm=arm, initial=initial, output=output, device=device)
            if not settings["quick"] and arm == "shared":
                fit["control_parity"] = verify_control_parity(fit, next(f for f in preflight["control_references"] if f["policy_seed"] == seed))
            fits.append(fit)
            _write_json(output / "training_fits.json", {"fits": fits, "new_test_opened": False})
    progress_message("A8 阶段2/4：冻结全部最佳模型与源文件；之后禁止再训练或选择")
    frozen = seal_heads(fits, settings, output)
    frozen_hash = _file_sha256(frozen)
    for collection in (preflight["frozen_source_hashes"], preflight["frozen_input_hashes"]):
        for path, checksum in collection.items():
            _verify_hash(path, checksum)
    _write_json(output / "NEW_TEST_OPENED.json", {"created_at": datetime.now(timezone.utc).isoformat(),
                "training_frozen_sha256": frozen_hash, "settings": settings["test_split"], "quick": settings["quick"]})
    progress_message("A8 阶段3/4：采集共同的新测试回合；此阶段不是网络训练")
    directory = output / "datasets"
    directory.mkdir()
    config, _ = load_s1_config(_project_path(physical["environment_config"]))
    representation = ActionRepresentation.from_mapping(physical["representation"])
    config = replace(config, num_modes=representation.num_modes)
    basis, _, basis_diagnostics = build_action_basis(config, representation, device)
    records, groups, comparisons = [], [], []
    collection_started = time.perf_counter()
    bar = counted_progress(total=preflight["expected_test_branches"], description="A8 新测试", unit="批量分支")
    for policy in policies:
        seed = int(policy["policy_seed"])
        test, record = _collect_and_save(name="independent_test", split=settings["test_split"], checkpoint=policy,
            physical_experiment=physical, base_config=config, basis=basis, settings=settings, bar=bar,
            progress_path=output / "test_collection_progress.jsonl", collection_started=collection_started,
            dataset_directory=directory, device=device)
        verify_dataset_separation({**data[seed], "independent_test": test})
        if not settings["quick"]:
            validate_coverage(test, episodes=96, pairs=1728)
        records.append(record)
        _write_json(output / "data_manifest.json", {"reused": preflight["data_specs"], "generated_test": records})
        outputs = {}
        for fit in [f for f in fits if f["policy_seed"] == seed]:
            value, score, training_mean = checkpoint_outputs(fit, test, device)
            fit["independent_test"] = evaluate_outputs(value, score, test, training_mean, settings, seed)
            fit["test_subgroups"] = subgroup_outputs(value, score, test, training_mean, policy_seed=seed, arm=fit["arm"])
            fit["predictions"]["independent_test"] = save_predictions(output / f"seed_{seed}" / fit["arm"], "independent_test", value, score, test)
            groups.extend(fit["test_subgroups"])
            outputs[fit["arm"]] = (value, score)
        comparisons.append(compare_heads(outputs["shared"], outputs["split"], test, settings, seed))
    bar.close()
    progress_message("A8 阶段4/4：保存成对比较与分工输出门槛；完成后等待审计")
    _verify_hash(frozen, frozen_hash)
    for collection in (preflight["frozen_source_hashes"], preflight["frozen_input_hashes"]):
        for path, checksum in collection.items():
            _verify_hash(path, checksum)
    for fit in fits:
        _verify_hash(fit["checkpoint"], fit["checkpoint_sha256"])
    _write_rows(output / "test_subgroups.csv", groups)
    _write_rows(output / "head_comparison.csv", [{"policy_seed": c["policy_seed"], **{
        f"{metric}_{key}": value for metric in ("relative_mae_reduction", "ranking_ba_delta")
        for key, value in c[metric].items()}} for c in comparisons])
    _write_rows(output / "fit_summary.csv", [{"policy_seed": f["policy_seed"], "arm": f["arm"],
        "best_update": f["best_update"], "parameter_count": f["parameter_count"], **{
            f"{name}_{key}": value for name in ("training", "validation", "independent_test")
            for key, value in {"ranking_ba": f[name]["ranking"]["balanced_accuracy"],
                               "value_mae": f[name]["value"]["mae"],
                               "value_mae_better_than_constant": f[name]["value"]["mae_better_than_constant"],
                               "head_disagreement": f[name]["head_sign_disagreement_fraction"]}.items()}} for f in fits])
    summary = {
        "material_passport": {"origin_skill": "academic-research-suite / experiment-agent", "origin_mode": "run",
                              "origin_date": datetime.now(timezone.utc).isoformat(), "verification_status": "UNVERIFIED",
                              "version_label": experiment["metadata"]["version_label"]},
        "experiment": {"id": experiment["metadata"]["experiment_id"], "quick": settings["quick"],
                       "status": "quick_smoke_only" if settings["quick"] else "completed_pending_audit",
                       "duration_seconds": time.perf_counter() - started, "device": str(device)},
        "fits": fits, "comparisons": comparisons, "interpretation": interpret_heads(fits, comparisons, settings),
        "basis_diagnostics": basis_diagnostics, "training_frozen_sha256": frozen_hash,
        "records": {_relative(p): _file_sha256(p) for p in output.rglob("*") if p.is_file()},
        "evidence_boundary": {"supervised_only": True, "ranking_score_is_reward": False,
                              "old_test_used_for_training_or_selection": False, "new_training_data_generated": False,
                              "full_rl_trained": False, "s4d3_accessed": False, "real_slm_actions": False, **NO_UPDATES},
        "runtime": _runtime_record(), "git": _git_record_utf8(), "next_action": "Stop for read-only audit; do not retrain automatically.",
    }
    _write_json(output / "summary.json", summary)
    return json_safe(summary)
