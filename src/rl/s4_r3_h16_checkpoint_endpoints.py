"""A8-D3：冻结最佳/末次模型的无训练配对诊断。"""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from src.rl.s4_r3_h16_data_scaling import _read_json, _read_pairs
from src.rl.s4_r3_h16_gradient_conflict import (
    ARMS, SPLITS, NO_UPDATES, _episode_gradient, _episode_groups,
    _model_digest, _seed_item, _selected_groups, _write_rows,
)
from src.rl.s4_r3_h16_gradient_shift import (
    FAMILIES, _condition_family, _flatten, _read_gradient_rows, two_sample_interval,
)
from src.rl.s4_r3_h16_head_split import HeadSplitProbe, _git_record_utf8, measure_outputs, validate_coverage
from src.rl.s4_r3_h16_reward_pairwise_probe import RewardPairDataset
from src.rl.s4_training import (
    _file_sha256, _load_yaml, _project_path, _relative, _runtime_record,
    _source_manifest, _write_json, json_safe,
)
from src.runtime import resolve_device
from src.training_progress import counted_progress, progress_message, update_progress


ROLES = ("best", "last")
SEEDS = (9301, 9302, 9303)
BEST_UPDATES = {9301: {"shared": 2300, "split": 200},
                9302: {"shared": 1100, "split": 3900},
                9303: {"shared": 1500, "split": 4600}}
DESIGN = {"policy_seeds": list(SEEDS), "arms": list(ARMS), "splits": list(SPLITS),
          "checkpoint_roles": list(ROLES), "feature_size": 221, "hidden_size": 256,
          "target_horizon": 16, "pairs_per_episode": 18, "maximum_updates": 10000,
          "validation_interval_updates": 100, "huber_delta": 1.0,
          "balanced_sign_weight": 0.25, "statistical_unit": "complete_episode_seed"}
COMPARISON = {"bootstrap_replicates": 20000, "bootstrap_seed": 890000,
              "familywise_alpha": 0.05, "family_size": 6,
              "meaningful_gap_change": 0.20, "minimum_supporting_seeds": 2}
ALIGNMENT = {"atol": 1e-6, "rtol": 1e-5, "balanced_accuracy_atol": 1e-6}
BOUNDARY = {"training_run": False, "new_data_generated": False,
            "independent_test_samples_accessed": False, "full_rl_trained": False,
            "s4d3_accessed": False, "real_slm_actions": False, **NO_UPDATES}


def _validate_contract(experiment: dict[str, Any]) -> None:
    metadata = experiment["metadata"]
    if metadata["stage"] != "S4-D2-R3-D2-A8-D3":
        raise RuntimeError("A8-D3 stage changed")
    required = {"diagnostic_only", "post_hoc_mechanism_diagnostic",
                "allow_checkpoint_loading", "allow_checkpoint_gradient_read"}
    if any(metadata.get(name) is not True for name in required):
        raise RuntimeError("A8-D3 required read-only flags changed")
    forbidden = {"allow_training", "allow_optimizer_updates", "allow_new_data_generation",
                 "allow_independent_test_access", "allow_full_rl_training",
                 "allow_s4d3_access", "allow_real_hardware_actions"}
    if (any(metadata.get(name) is not False for name in forbidden)
            or any(name.startswith("allow_") and name not in required and value is not False
                   for name, value in metadata.items())):
        raise RuntimeError("A8-D3 safety flag changed")
    if experiment["design"] != DESIGN or experiment["comparison"] != COMPARISON:
        raise RuntimeError("A8-D3 frozen design/statistical contract changed")
    if experiment["alignment"] != ALIGNMENT:
        raise RuntimeError("A8-D3 alignment tolerance changed")
    if experiment["runtime"] != {"device": "cuda", "require_cuda": True, "deterministic_algorithms": True}:
        raise RuntimeError("A8-D3 requires CUDA and deterministic algorithms")
    if experiment["quick"] != {"policy_seeds": [9301], "episodes_per_condition": 2, "bootstrap_replicates": 100}:
        raise RuntimeError("A8-D3 quick coverage changed")


def _safe_input(value: str, *, expected: str | None = None) -> Path:
    root = _project_path(".").resolve()
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError("A8-D3 input must be repository relative")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise RuntimeError("A8-D3 input escapes repository")
    relative = resolved.relative_to(root).as_posix()
    if "independent_test" in relative.lower() or "s4d3" in relative.lower():
        raise RuntimeError("A8-D3 forbidden test sample path")
    if expected is not None and relative != expected:
        raise RuntimeError(f"A8-D3 unexpected input path: {relative}")
    return resolved


def _pin(path: str, checksum: str, manifest: dict[str, str], *, expected: str | None = None) -> Path:
    resolved = _safe_input(path, expected=expected)
    if len(checksum) != 64 or _file_sha256(resolved) != checksum:
        raise RuntimeError(f"A8-D3 frozen hash mismatch: {path}")
    manifest[path] = checksum
    return resolved


def _output_path(value: str) -> Path:
    root = _project_path("outputs").resolve()
    path = _project_path(value).resolve()
    if path.parent != root or not path.name.startswith("s4_r3_h16_checkpoint_endpoints_v1"):
        raise RuntimeError("A8-D3 output must be a dedicated endpoint directory")
    if path.exists():
        raise FileExistsError(f"A8-D3 output already exists; do not overwrite: {path}")
    return path


def _effective_settings(experiment: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    return {"quick": quick, "policy_seeds": experiment["quick"]["policy_seeds"] if quick else list(SEEDS),
            "episodes_per_condition": experiment["quick"]["episodes_per_condition"] if quick else None,
            "bootstrap_replicates": experiment["quick"]["bootstrap_replicates"] if quick else 20000,
            "output_directory": experiment["outputs"]["quick_directory" if quick else "directory"]}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_history(rows: Sequence[dict[str, Any]], *, seed: int, arm: str) -> dict[str, Any]:
    if [int(row["update"]) for row in rows] != list(range(100, 10001, 100)):
        raise RuntimeError("A8-D3 missing/duplicated history updates")
    best_score = (-math.inf, math.inf)
    best_update = 0
    for row in rows:
        if int(row["policy_seed"]) != seed or row["arm"] != arm or int(row["total_updates"]) != 10000:
            raise RuntimeError("A8-D3 history identity changed")
        score = float(row["validation_balanced_accuracy"]), float(row["validation_value_mae"])
        if not all(math.isfinite(v) for v in score):
            raise RuntimeError("A8-D3 non-finite history")
        if score[0] > best_score[0] + 1e-12 or (abs(score[0] - best_score[0]) <= 1e-12 and score[1] < best_score[1]):
            best_score, best_update = score, int(row["update"])
        if int(row["best_update"]) != best_update:
            raise RuntimeError("A8-D3 history selection rule mismatch")
    if best_update != BEST_UPDATES[seed][arm]:
        raise RuntimeError("A8-D3 best checkpoint update changed")
    return {"best_update": best_update, "last_update": 10000, "history_rows": len(rows),
            "last_validation_balanced_accuracy": float(rows[-1]["validation_balanced_accuracy"]),
            "last_validation_value_mae": float(rows[-1]["validation_value_mae"])}


def validate_checkpoint(checkpoint: dict[str, Any], *, seed: int, arm: str, role: str) -> None:
    if role not in ROLES or seed not in SEEDS or arm not in ARMS:
        raise RuntimeError("A8-D3 unknown checkpoint identity")
    expected_update = BEST_UPDATES[seed][arm] if role == "best" else 10000
    if (checkpoint["policy_seed"] != seed or checkpoint["arm"] != arm
            or checkpoint["update"] != expected_update or checkpoint["best_update"] != BEST_UPDATES[seed][arm]
            or checkpoint["config"] != {"feature_size": 221, "hidden_size": 256}
            or checkpoint["target_horizon"] != 16 or checkpoint["label_source"] != "empirical_reward_returns"
            or checkpoint["independent_test_used_for_selection"] is not False
            or checkpoint["ranking_score_is_reward"] is not False):
        raise RuntimeError("A8-D3 checkpoint identity/update contract changed")
    if any(checkpoint.get(name) != 0 for name in ("original_critic_updates", "actor_updates", "alpha_updates", "student_updates")):
        raise RuntimeError("A8-D3 checkpoint update boundary changed")
    norms = checkpoint["normalization"]
    for name in ("feature_mean", "feature_scale", "target_scale"):
        value = norms[name]
        if not bool(torch.isfinite(value).all()) or (name.endswith("scale") and not bool((value > 0).all())):
            raise RuntimeError("A8-D3 invalid checkpoint normalization")
    if (norms["feature_mean"].shape != (221,) or norms["feature_scale"].shape != (221,)
            or norms["target_scale"].numel() != 1
            or not math.isfinite(float(checkpoint["training_reward_mean"]))):
        raise RuntimeError("A8-D3 normalization shape/mean changed")
    if any(not bool(torch.isfinite(value).all()) for value in checkpoint["probe"].values()):
        raise RuntimeError("A8-D3 non-finite checkpoint parameters")
    shapes = {"network.0.weight": (256, 221), "network.0.bias": (256,),
              "network.2.weight": (256, 256), "network.2.bias": (256,),
              "network.4.weight": (1, 256), "network.4.bias": (1,)}
    if arm == "split":
        shapes.update({"sign_head.weight": (1, 256), "sign_head.bias": (1,)})
    if checkpoint["probe"].keys() != shapes.keys() or any(checkpoint["probe"][k].shape != v for k, v in shapes.items()):
        raise RuntimeError("A8-D3 checkpoint parameter shape changed")


def _gradient_key(row: dict[str, Any]) -> tuple[int, str, str, str, int]:
    return int(row["policy_seed"]), str(row["arm"]), str(row["split"]), str(row["condition_id"]), int(row["episode_seed"])


def check_gradient_alignment(actual: dict[str, Any], reference: dict[str, Any]) -> float:
    if _gradient_key(actual) != _gradient_key(reference) or int(actual["pairs"]) != int(reference["pairs"]):
        raise RuntimeError("ALIGNMENT_FAILED: gradient key/pairs")
    if actual["gradient_conflict"] != reference["gradient_conflict"]:
        raise RuntimeError("ALIGNMENT_FAILED: conflict sign")
    error = 0.0
    identifiers = {"policy_seed", "arm", "split", "condition_id", "condition_family", "episode_seed", "pairs"}
    for name in reference.keys() - identifiers:
        value, target = float(actual[name]), float(reference[name])
        if not math.isfinite(value) or not math.isclose(value, target, abs_tol=ALIGNMENT["atol"], rel_tol=ALIGNMENT["rtol"]):
            raise RuntimeError(f"ALIGNMENT_FAILED: {name} at {_gradient_key(actual)}: {value} vs {target}")
        error = max(error, abs(value - target))
    return error


def preflight_endpoints(config_path: str | Path, experiment: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    _validate_contract(experiment)
    output = _output_path(settings["output_directory"])
    device = resolve_device(experiment["runtime"]["device"])
    hashes: dict[str, str] = {}
    for name in ("frozen_files", "frozen_sources"):
        for path, checksum in experiment[name].items():
            _pin(path, checksum, hashes)
    inputs = experiment["inputs"]
    if any(path not in experiment["frozen_files"] for path in inputs.values()):
        raise RuntimeError("A8-D3 unpinned input reference")
    for name, expected in (("d1", "SHARED_TRUNK_CONFLICT_NOT_CONFIRMED"),
                           ("d2", "REVERSAL_NOT_EXPLAINED_BY_OBSERVED_DISTRIBUTIONS")):
        summary, marker = _read_json(inputs[f"{name}_summary"]), _read_json(inputs[f"{name}_success"])
        if (summary["experiment"]["quick"] or summary["interpretation"]["status"] != expected
                or marker["summary_sha256"] != hashes[inputs[f"{name}_summary"]]
                or marker["status"] != "DIAGNOSTIC_COMPLETED_PENDING_AUDIT"):
            raise RuntimeError(f"A8-D3 upstream {name} status/hash changed")
    d1 = _load_yaml(_safe_input(inputs["d1_config"]))
    fits_list = _read_json(inputs["training_fits"])["fits"]
    fits = {(int(f["policy_seed"]), f["arm"]): f for f in fits_list}
    if len(fits) != 6 or len(fits_list) != 6 or set(fits) != {(s, a) for s in SEEDS for a in ARMS}:
        raise RuntimeError("A8-D3 fit coverage mismatch")
    dataset_specs: dict[int, dict[str, Any]] = {}
    all_keys: set[tuple[int, str, str, str, int]] = set()
    episode_maps: dict[str, list[set[tuple[str, int]]]] = {s: [] for s in SPLITS}
    selected: dict[str, int] = {}
    for seed in SEEDS:
        dataset_specs[seed] = {}
        for split in SPLITS:
            spec = dict(_seed_item(d1["datasets"], seed)[split])
            suffix = "large" if split == "training" else "validation"
            path = _pin(spec["path"], spec["sha256"], hashes,
                        expected=f"outputs/s4_r3_h16_data_scaling_v1/datasets/seed_{seed}_{suffix}_pairs.pt")
            episodes = 384 if split == "training" else 48
            if (spec["episodes"], spec["pairs"]) != (episodes, episodes * 18):
                raise RuntimeError("A8-D3 data budget changed")
            data = _read_pairs(path)
            validate_coverage(data, episodes=episodes, pairs=episodes * 18)
            groups = _episode_groups(data)
            if Counter(_condition_family(c) for c, _, _ in groups) != {f: episodes // 3 for f in FAMILIES}:
                raise RuntimeError("A8-D3 condition coverage changed")
            episode_maps[split].append({(c, e) for c, e, _ in groups})
            all_keys.update((seed, a, split, c, e) for a in ARMS for c, e, _ in groups)
            selected[split] = len(_selected_groups(data, settings["episodes_per_condition"]))
            dataset_specs[seed][split] = spec
        if {e for _, e in episode_maps["training"][-1]} & {e for _, e in episode_maps["validation"][-1]}:
            raise RuntimeError("A8-D3 training/validation episode leakage")
    if any(any(keys != maps[0] for keys in maps[1:]) for maps in episode_maps.values()):
        raise RuntimeError("A8-D3 cross-policy episode alignment failed")
    reference = _read_gradient_rows(inputs["d1_gradients"])
    if len(reference) != 2592 or {_gradient_key(row) for row in reference} != all_keys:
        raise RuntimeError("A8-D3 upstream gradient episode coverage failed")
    for row in reference:
        if row["pairs"] != 18 or row["gradient_conflict"] != float(row["gradient_cosine"] < 0):
            raise RuntimeError("A8-D3 invalid upstream gradient record")
    inventory = []
    for seed in SEEDS:
        for arm in ARMS:
            fit = fits[seed, arm]
            prefix = f"outputs/s4_r3_h16_head_split_v1/seed_{seed}/{arm}"
            for key, checksum_key, filename in (("checkpoint", "checkpoint_sha256", "checkpoint_best.pt"),
                                                ("last_checkpoint", "last_checkpoint_sha256", "checkpoint_last.pt"),
                                                ("loss_history", "loss_history_sha256", "loss_history.csv"),
                                                ("progress_log", "progress_sha256", "progress.jsonl")):
                _pin(fit[key], fit[checksum_key], hashes, expected=f"{prefix}/{filename}")
            if (fit["best_update"] != BEST_UPDATES[seed][arm] or fit["updates_completed"] != 10000
                    or fit["independent_test_used_for_selection"] is not False):
                raise RuntimeError("A8-D3 frozen fit metadata changed")
            rows = _read_csv(_safe_input(fit["loss_history"]))
            history = validate_history(rows, seed=seed, arm=arm)
            logs = [json.loads(line) for line in _safe_input(fit["progress_log"]).read_text(encoding="utf-8").splitlines()]
            if len(logs) != len(rows) or any(
                    any((float(row[k]) != v if isinstance(v, (int, float)) else row[k] != v) for k, v in log.items())
                    for row, log in zip(rows, logs, strict=True)):
                raise RuntimeError("A8-D3 history/progress mismatch")
            checkpoints = {}
            for role, path_key, checksum_key in (("best", "checkpoint", "checkpoint_sha256"),
                                                  ("last", "last_checkpoint", "last_checkpoint_sha256")):
                checkpoint = torch.load(_safe_input(fit[path_key]), map_location=device, weights_only=True)
                validate_checkpoint(checkpoint, seed=seed, arm=arm, role=role)
                checkpoints[role] = checkpoint
                inventory.append({"policy_seed": seed, "arm": arm, "checkpoint_role": role,
                                  "update": checkpoint["update"], "path": fit[path_key], "sha256": fit[checksum_key], **history})
            b, last = checkpoints["best"], checkpoints["last"]
            if b["normalization"].keys() != last["normalization"].keys() or any(
                    not torch.equal(b["normalization"][k], last["normalization"][k]) for k in b["normalization"]):
                raise RuntimeError("A8-D3 checkpoint normalizations differ")
            if b["training_reward_mean"] != last["training_reward_mean"]:
                raise RuntimeError("A8-D3 checkpoint training means differ")
            if fit["checkpoint_sha256"] != _seed_item(d1["checkpoints"], seed)[arm]["sha256"]:
                raise RuntimeError("A8-D3 best checkpoint differs from D1")
    sources = _source_manifest(experiment["tracked_source_files"])
    sources[_relative(_project_path(config_path))] = _file_sha256(_project_path(config_path))
    return {"status": "READY_FOR_QUICK_SMOKE" if settings["quick"] else "READY_FOR_USER_DIAGNOSTIC",
            "quick": settings["quick"], "device": str(device), "policy_seeds": settings["policy_seeds"],
            "selected_episodes_per_split": selected, "expected_episode_records": len(settings["policy_seeds"]) * 4 * sum(selected.values()),
            "expected_endpoint_metrics": len(settings["policy_seeds"]) * 8,
            "expected_gap_comparisons": len(settings["policy_seeds"]) * 2,
            "dataset_specs": dataset_specs, "checkpoint_inventory": inventory,
            "frozen_input_hashes": hashes, "frozen_source_hashes": sources,
            "output_directory": _relative(output), "independent_test_sample_paths": [], **BOUNDARY}


def _paired_changes(best: Sequence[dict[str, Any]], last: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    left = {(r["condition_id"], int(r["episode_seed"])): r for r in best}
    right = {(r["condition_id"], int(r["episode_seed"])): r for r in last}
    if len(left) != len(best) or len(right) != len(last) or left.keys() != right.keys() or not left:
        raise RuntimeError("A8-D3 paired episode keys mismatch/duplicate")
    result = []
    for key in sorted(left):
        values = float(left[key]["gradient_cosine"]), float(right[key]["gradient_cosine"])
        if not all(math.isfinite(v) for v in values):
            raise RuntimeError("A8-D3 non-finite paired gradient")
        result.append({"condition_id": key[0], "condition_family": _condition_family(key[0]),
                       "episode_seed": key[1], "change": values[1] - values[0]})
    return result


def paired_gap_interval(best_training: Sequence[dict[str, Any]], last_training: Sequence[dict[str, Any]],
                        best_validation: Sequence[dict[str, Any]], last_validation: Sequence[dict[str, Any]],
                        *, replicates: int, seed: int, device: torch.device) -> dict[str, Any]:
    train = _paired_changes(best_training, last_training)
    val = _paired_changes(best_validation, last_validation)
    if {r["episode_seed"] for r in train} & {r["episode_seed"] for r in val}:
        raise RuntimeError("A8-D3 training/validation episode leakage")
    if {r["condition_family"] for r in train} != set(FAMILIES) or {r["condition_family"] for r in val} != set(FAMILIES):
        raise RuntimeError("A8-D3 missing condition family")
    interval = two_sample_interval(train, val, "change", replicates=replicates, seed=seed, device=device,
                                   familywise_alpha=0.05, family_size=6)
    interval.update({"comparison": "(validation_last-validation_best)-(training_last-training_best)",
                     "checkpoint_paired": True, "training_validation_paired": False,
                     "training_episodes": len(train), "validation_episodes": len(val)})
    return interval


def interpret_endpoints(comparisons: Sequence[dict[str, Any]], *, quick: bool) -> dict[str, Any]:
    boundary = {"causal_explanation_authorized": False, "overfitting_confirmed": False,
                "new_model_training_authorized": False, "full_rl_authorized": False,
                "s4d3_authorized": False, "real_hardware_authorized": False}
    if quick:
        return {"status": "QUICK_SMOKE_ONLY", "checkpoint_sensitivity_supported": False, **boundary}
    lookup = {(int(row["policy_seed"]), row["arm"]): row["delta_gap"] for row in comparisons}
    if len(comparisons) != 6 or lookup.keys() != {(s, a) for s in SEEDS for a in ARMS}:
        raise RuntimeError("A8-D3 comparison coverage mismatch")
    positive, negative = [], []
    for (seed, arm), interval in lookup.items():
        lo, hi = float(interval["familywise_ci_low"]), float(interval["familywise_ci_high"])
        if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
            raise RuntimeError("A8-D3 invalid comparison interval")
        if arm == "shared":
            if lo > 0.20:
                positive.append(seed)
            if hi < -0.20:
                negative.append(seed)
    supported = len(positive) >= 2 or len(negative) >= 2
    return {"status": "CHECKPOINT_SENSITIVE_GAP_CANDIDATE" if supported else "CHECKPOINT_SENSITIVITY_NOT_CONFIRMED",
            "checkpoint_sensitivity_supported": supported, "positive_supporting_seeds": positive,
            "negative_supporting_seeds": negative, **boundary}


@torch.no_grad()
def evaluate_endpoint(model: HeadSplitProbe, data: RewardPairDataset, checkpoint: dict[str, Any],
                      *, positive_weight: torch.Tensor, device: torch.device) -> dict[str, Any]:
    model.eval()
    norms = {k: v.to(device) for k, v in checkpoint["normalization"].items()}
    raw, score = model((data.features.to(device) - norms["feature_mean"]) / norms["feature_scale"])
    target = data.reward_delta.to(device)
    metrics = measure_outputs((raw * norms["target_scale"]).cpu(), score.cpu(), data, checkpoint["training_reward_mean"])
    reg = float(F.huber_loss(raw, target / norms["target_scale"], delta=1.0))
    sign = float(F.binary_cross_entropy_with_logits(score, (target > 0).float(), pos_weight=positive_weight))
    truth, guessed = target > 0, score > 0
    return {"pairs": len(data.rows), "episodes": len(_episode_groups(data)),
            "balanced_accuracy": metrics["ranking"]["balanced_accuracy"],
            "mcc": metrics["ranking"]["matthews_correlation"], "value_mae": metrics["value"]["mae"],
            "value_rmse": metrics["value"]["rmse"], "value_sign_ba": metrics["value_sign_ranking"]["balanced_accuracy"],
            "head_disagreement": metrics["head_sign_disagreement_fraction"],
            "positive_pairs": int(truth.sum()), "negative_pairs": int((~truth).sum()),
            "tp": int((truth & guessed).sum()), "fn": int((truth & ~guessed).sum()),
            "fp": int((~truth & guessed).sum()), "tn": int((~truth & ~guessed).sum()),
            "regression_loss": reg, "classification_loss": sign, "total_loss": reg + 0.25 * sign}


def _align_metrics(metrics: dict[str, Any], *, role: str, split: str,
                   fit: dict[str, Any], inventory: dict[str, Any]) -> None:
    expected: dict[str, float] = {}
    if role == "best":
        original = fit[split]
        expected = {"balanced_accuracy": original["ranking"]["balanced_accuracy"],
                    "value_mae": original["value"]["mae"], "mcc": original["ranking"]["matthews_correlation"],
                    "value_sign_ba": original["value_sign_ranking"]["balanced_accuracy"],
                    "head_disagreement": original["head_sign_disagreement_fraction"]}
    elif split == "validation":
        expected = {"balanced_accuracy": inventory["last_validation_balanced_accuracy"],
                    "value_mae": inventory["last_validation_value_mae"]}
    for name, value in expected.items():
        atol = ALIGNMENT["balanced_accuracy_atol"] if name == "balanced_accuracy" else ALIGNMENT["atol"]
        rtol = 0.0 if name == "balanced_accuracy" else ALIGNMENT["rtol"]
        if not math.isclose(float(metrics[name]), float(value), abs_tol=atol, rel_tol=rtol):
            raise RuntimeError(f"ALIGNMENT_FAILED: {role}/{split}/{name}: {metrics[name]} vs {value}")


def _verify_manifests(preflight: dict[str, Any]) -> None:
    for name in ("frozen_input_hashes", "frozen_source_hashes"):
        for path, expected in preflight[name].items():
            if _file_sha256(_project_path(path)) != expected:
                raise RuntimeError(f"A8-D3 input/source changed during execution: {path}")


def _execute(experiment: dict[str, Any], settings: dict[str, Any], preflight: dict[str, Any],
             output: Path, device: torch.device) -> dict[str, Any]:
    started = time.perf_counter()
    _verify_manifests(preflight)
    inputs = experiment["inputs"]
    references = {_gradient_key(r): r for r in _read_gradient_rows(inputs["d1_gradients"])}
    fits = {(int(f["policy_seed"]), f["arm"]): f for f in _read_json(inputs["training_fits"])["fits"]}
    inventory = [row for row in preflight["checkpoint_inventory"] if row["policy_seed"] in settings["policy_seeds"]]
    _write_rows(output / "checkpoint_inventory.csv", inventory)
    spec_by_key = {(row["policy_seed"], row["arm"], row["checkpoint_role"]): row for row in inventory}
    episode_rows, endpoints, integrity = [], [], []
    alignment: dict[str, Any] = {"status": "IN_PROGRESS", "best_gradient_records_checked": 0,
                                "maximum_absolute_gradient_difference": 0.0, "endpoint_metrics_checked": 0}
    _write_json(output / "alignment.json", alignment)
    total = preflight["expected_episode_records"]
    bar = counted_progress(total=total, description="A8-D3 最佳/末次梯度", unit="回合")
    try:
        with (output / "progress.jsonl").open("w", encoding="utf-8") as progress, \
                (output / "episode_gradient_metrics.csv").open("w", encoding="utf-8", newline="") as csv_handle:
            writer = None
            # 先完成全部最佳版本的对齐，再读取末次版本产生新诊断数据。
            for role in ROLES:
                progress_message(f"A8-D3 阶段：{'最佳模型对齐' if role == 'best' else '末次模型对照'}")
                for seed in settings["policy_seeds"]:
                    datasets = {split: _read_pairs(_safe_input(preflight["dataset_specs"][seed][split]["path"])) for split in SPLITS}
                    y = datasets["training"].reward_delta.to(device)
                    positives = int((y > 0).sum())
                    weight = torch.tensor((len(y) - positives) / positives, device=device)
                    for arm in ARMS:
                        spec = spec_by_key[seed, arm, role]
                        checkpoint = torch.load(_safe_input(spec["path"]), map_location=device, weights_only=True)
                        model = HeadSplitProbe(**checkpoint["config"], arm=arm).to(device)
                        model.load_state_dict(checkpoint["probe"])
                        model.eval()
                        before = _model_digest(model)
                        if any(p.grad is not None for p in model.parameters()):
                            raise RuntimeError("A8-D3 model has pre-existing gradients")
                        for split in SPLITS:
                            data = datasets[split]
                            for condition, episode, indices in _selected_groups(data, settings["episodes_per_condition"]):
                                metrics = _episode_gradient(model, data, indices, checkpoint, positive_weight=weight,
                                                            classification_weight=0.25, huber_delta=1.0, device=device)
                                row = {"policy_seed": seed, "arm": arm, "checkpoint_role": role, "update": spec["update"],
                                       "split": split, "condition_id": condition, "episode_seed": episode, "pairs": len(indices), **metrics}
                                if writer is None:
                                    writer = csv.DictWriter(csv_handle, fieldnames=list(row))
                                    writer.writeheader()
                                writer.writerow(row)
                                csv_handle.flush()
                                if role == "best":
                                    error = check_gradient_alignment(row, references[_gradient_key(row)])
                                    alignment["maximum_absolute_gradient_difference"] = max(error, alignment["maximum_absolute_gradient_difference"])
                                    alignment["best_gradient_records_checked"] += 1
                                episode_rows.append(row)
                                completed = len(episode_rows)
                                elapsed = time.perf_counter() - started
                                record = {"phase": role, "completed_episode_records": completed, "total_episode_records": total,
                                          "policy_seed": seed, "arm": arm, "split": split, "condition_id": condition,
                                          "episode_seed": episode, "elapsed_seconds": elapsed,
                                          "estimated_remaining_seconds": elapsed / completed * (total - completed),
                                          "cuda_allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
                                          "cuda_reserved_gb": torch.cuda.memory_reserved(device) / 1024**3,
                                          "gradient_cosine": metrics["gradient_cosine"]}
                                progress.write(json.dumps(record, ensure_ascii=False) + "\n")
                                progress.flush()
                                bar.update(1)
                                update_progress(bar, device=device, metrics={"余弦": metrics["gradient_cosine"]})
                            # 完整数据读数用于对齐；快速模式也不拿子集去匹配全量日志。
                            metrics = evaluate_endpoint(model, data, checkpoint, positive_weight=weight, device=device)
                            _align_metrics(metrics, role=role, split=split, fit=fits[seed, arm], inventory=spec)
                            alignment["endpoint_metrics_checked"] += int(role == "best" or split == "validation")
                            endpoints.append({"policy_seed": seed, "arm": arm, "checkpoint_role": role,
                                              "update": spec["update"], "split": split, **metrics})
                            _write_rows(output / "endpoint_metrics.csv", endpoints)
                        after = _model_digest(model)
                        untouched = all(p.grad is None for p in model.parameters())
                        integrity.append({"policy_seed": seed, "arm": arm, "checkpoint_role": role,
                                          "before_sha256": before, "after_sha256": after,
                                          "parameters_unchanged": before == after, "parameter_grad_fields_untouched": untouched})
                        if before != after or not untouched:
                            raise RuntimeError("A8-D3 changed frozen model parameters/grad fields")
                        del model, checkpoint
                _write_json(output / "alignment.json", alignment)
    except Exception as error:
        alignment.update({"status": "ALIGNMENT_FAILED" if "ALIGNMENT_FAILED" in str(error) else "INCOMPLETE",
                          "error": str(error)})
        _write_json(output / "alignment.json", alignment)
        raise
    finally:
        bar.close()
    if (len(episode_rows) != total or len(endpoints) != preflight["expected_endpoint_metrics"]
            or alignment["best_gradient_records_checked"] != total // 2):
        raise RuntimeError("A8-D3 final record coverage incomplete")
    alignment["status"] = "PASSED"
    _write_json(output / "alignment.json", alignment)
    progress_message("A8-D3 阶段：配对缺口重采样与保存")
    comparisons, condition_rows = [], []
    for seed in settings["policy_seeds"]:
        for arm in ARMS:
            cells = {(role, split): [r for r in episode_rows if r["policy_seed"] == seed and r["arm"] == arm
                                     and r["checkpoint_role"] == role and r["split"] == split]
                     for role in ROLES for split in SPLITS}
            interval = paired_gap_interval(cells["best", "training"], cells["last", "training"],
                                           cells["best", "validation"], cells["last", "validation"],
                                           replicates=settings["bootstrap_replicates"],
                                           seed=890000 + (SEEDS.index(seed) * 2 + ARMS.index(arm)) * 100, device=device)
            def mean(rows: Sequence[dict[str, Any]]) -> float:
                return sum(float(r["gradient_cosine"]) for r in rows) / len(rows)
            gaps = {role: mean(cells[role, "validation"]) - mean(cells[role, "training"]) for role in ROLES}
            comparisons.append({"policy_seed": seed, "arm": arm, "best_gap": gaps["best"], "last_gap": gaps["last"], "delta_gap": interval})
            for role in ROLES:
                for family in FAMILIES:
                    train = [r for r in cells[role, "training"] if _condition_family(r["condition_id"]) == family]
                    val = [r for r in cells[role, "validation"] if _condition_family(r["condition_id"]) == family]
                    condition_rows.append({"policy_seed": seed, "arm": arm, "checkpoint_role": role,
                                           "condition_family": family, "training_mean": mean(train), "validation_mean": mean(val),
                                           "gradient_gap": mean(val) - mean(train)})
    _write_rows(output / "gap_comparisons.csv", _flatten(comparisons))
    _write_rows(output / "condition_gaps.csv", condition_rows)
    _verify_manifests(preflight)
    interpretation = interpret_endpoints(comparisons, quick=settings["quick"])
    summary = {"material_passport": {"origin_skill": "academic-research-suite / experiment-agent", "origin_mode": "run",
                                       "origin_date": datetime.now(timezone.utc).isoformat(), "verification_status": "UNVERIFIED",
                                       "version_label": experiment["metadata"]["version_label"]},
               "experiment": {"id": experiment["metadata"]["experiment_id"], "quick": settings["quick"],
                              "status": "quick_smoke_only" if settings["quick"] else "completed_pending_audit",
                              "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
                              "duration_seconds": time.perf_counter() - started},
               "contract": {"post_hoc": True, "checkpoint_paired": True, "training_validation_paired": False,
                            "statistical_unit": "complete_episode_seed", "full_endpoint_metrics_are_descriptive": True},
               "episode_records": len(episode_rows), "alignment": alignment, "parameter_integrity": integrity,
               "endpoint_metrics": endpoints, "gap_comparisons": comparisons, "interpretation": interpretation,
               "evidence_boundary": {"checkpoint_loaded": True, "gradient_recomputed": True, **BOUNDARY},
               "records": {_relative(p): _file_sha256(p) for p in output.iterdir() if p.is_file()},
               "runtime": _runtime_record(), "git": _git_record_utf8(),
               "next_action": "停止并等待只读审计；不要自动训练、替换模型或进入强化学习。"}
    _write_json(output / "summary.json", summary)
    _write_json(output / ("QUICK_SUCCESS.json" if settings["quick"] else "SUCCESS.json"),
                {"status": "QUICK_SMOKE_ONLY" if settings["quick"] else "DIAGNOSTIC_COMPLETED_PENDING_AUDIT",
                 "summary_sha256": _file_sha256(output / "summary.json"), **BOUNDARY})
    return json_safe(summary)


def run_s4_r3_h16_checkpoint_endpoints(config_path: str | Path, *, quick: bool = False,
                                      preflight_only: bool = False) -> dict[str, Any]:
    config_path = _project_path(config_path)
    experiment = _load_yaml(config_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight = preflight_endpoints(config_path, experiment, settings)
    if preflight_only:
        return preflight
    device = resolve_device(experiment["runtime"]["device"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    output = _output_path(settings["output_directory"])
    output.mkdir(parents=True, exist_ok=False)
    try:
        for name, value in (("preflight", preflight), ("effective_config", {"experiment": experiment, "settings": settings}),
                            ("input_manifest", preflight["frozen_input_hashes"]), ("source_manifest", preflight["frozen_source_hashes"])):
            _write_json(output / f"{name}.json", value)
        return _execute(experiment, settings, preflight, output, device)
    except Exception as error:
        import traceback
        _write_json(output / "failure.json", {"exception": type(error).__name__, "message": str(error),
                                                "traceback": traceback.format_exc(), "automatic_retry": False})
        raise
