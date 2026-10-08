"""A9四组监督拟合、固定时点留档。训练函数不接收确认集。"""
from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from src.rl.s4_r3_h16_fixed_time_contract import ARMS, BOUNDARY
from src.rl.s4_r3_h16_data_scaling import verify_dataset_separation
from src.rl.s4_r3_h16_head_split import HeadSplitProbe, ranking_metrics
from src.rl.s4_r3_h16_reward_pairwise_probe import RewardPairDataset, _normalization_from_training, _runtime_fields
from src.rl.s4_training import _file_sha256, _project_path, _relative, json_safe
from src.training_progress import progress_bar, update_progress


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(json_safe(obj), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("A9 empty records")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(json_safe(rows))


class FixedTimeProbe(HeadSplitProbe):
    def __init__(self, feature_size: int, hidden_size: int, task: str) -> None:
        if task not in ARMS:
            raise ValueError("A9 unknown task")
        super().__init__(feature_size, hidden_size, "split" if task == "split" else "shared")
        self.task = task


def loss_terms(model: FixedTimeProbe, features: torch.Tensor, target: torch.Tensor,
               positive_weight: torch.Tensor, *, huber_delta: float) -> tuple[torch.Tensor, Any, Any]:
    value, score = model(features)
    reg = None if model.task == "classification_only" else F.huber_loss(value, target, delta=huber_delta)
    cls = None if model.task == "regression_only" else F.binary_cross_entropy_with_logits(
        score, (target > 0).float(), pos_weight=positive_weight)
    total = (reg if reg is not None else 0) + (0.25*cls if cls is not None else 0)
    if not bool(torch.isfinite(total)):
        raise RuntimeError("A9 non-finite loss")
    return total, reg, cls


def episode_groups(data: RewardPairDataset) -> list[tuple[str, int, list[int]]]:
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    seed_condition: dict[int, str] = {}
    unique = set()
    for i, row in enumerate(data.rows):
        condition, seed = str(row["condition_id"]), int(row["episode_seed"])
        if seed in seed_condition and seed_condition[seed] != condition:
            raise RuntimeError("A9 episode assigned to multiple conditions")
        seed_condition[seed] = condition
        key = seed, str(row["profile_id"]), int(row["probe_step"])
        if key in unique:
            raise RuntimeError("A9 duplicate action pair")
        unique.add(key)
        groups[condition, seed].append(i)
    return [(c, seed, ix) for (c, seed), ix in sorted(groups.items())]


def validate_data(data: RewardPairDataset, split: dict, policy_seed: int, *, collection_record: dict) -> None:
    data.validate(feature_size=221)
    groups = episode_groups(data)
    desired = {(str(c["id"]), int(c["base_seed"])+i) for c in split["conditions"] for i in range(split["episodes_per_condition"])}
    if {(c, seed) for c, seed, _ in groups} != desired:
        raise RuntimeError("A9 episode coverage mismatch")
    desired_pairs = {(p, t) for p in split["profile_ids"] for t in split["probe_steps"]}
    for _, _, ix in groups:
        if {(str(data.rows[i]["profile_id"]), int(data.rows[i]["probe_step"])) for i in ix} != desired_pairs:
            raise RuntimeError("A9 incomplete episode pairs")
    # 上游配对加载器仅保留回合/动作键；策略身份由实际加载策略生成的采集记录提供。
    # 不向配对行补写预期种子，否则会把需要验证的身份变成自我声明。
    if collection_record.get("policy_seed") != policy_seed:
        raise RuntimeError("A9 policy identity mismatch")


@torch.no_grad()
def predictions(model: FixedTimeProbe, data: RewardPairDataset, norms: dict, device: torch.device) -> tuple[Any, torch.Tensor]:
    model.eval()
    n = {k: v.to(device) for k, v in norms.items()}
    value, score = model((data.features.to(device)-n["feature_mean"])/n["feature_scale"])
    if not bool(torch.isfinite(value).all() and torch.isfinite(score).all()):
        raise RuntimeError("A9 non-finite predictions")
    return (None if model.task == "classification_only" else (value*n["target_scale"]).cpu()), score.cpu()


def metrics(value: Any, score: torch.Tensor, data: RewardPairDataset, training_mean: float) -> dict:
    y = data.reward_delta
    if score.shape != y.shape or (value is not None and value.shape != y.shape):
        raise ValueError("A9 prediction shape mismatch")
    if not bool(torch.isfinite(score).all()) or (value is not None and not bool(torch.isfinite(value).all())):
        raise RuntimeError("A9 non-finite metric input")
    ranking = ranking_metrics(score, y)
    pos, pred = y > 0, score > 0
    result = dict(pairs=len(y), episodes=len(episode_groups(data)),
                  positive_pairs=int(pos.sum()), negative_pairs=int((~pos).sum()),
                  tp=int((pos & pred).sum()), fn=int((pos & ~pred).sum()),
                  fp=int((~pos & pred).sum()), tn=int((~pos & ~pred).sum()),
                  ranking_status=ranking["status"], balanced_accuracy=ranking.get("balanced_accuracy"),
                  mcc=ranking.get("matthews_correlation"), value_mae=None, value_rmse=None,
                  value_bias=None, constant_mae=None, value_sign_ba=None, head_disagreement=None,
                  power_direction_ba=ranking_metrics(score, data.power_delta).get("balanced_accuracy"))
    if value is not None:
        error = value.double()-y.double()
        result.update(value_mae=float(error.abs().mean()), value_rmse=float(error.square().mean().sqrt()),
                      value_bias=float(error.mean()), constant_mae=float((y.double()-training_mean).abs().mean()),
                      value_sign_ba=ranking_metrics(value, y).get("balanced_accuracy"),
                      head_disagreement=float(((value > 0) != pred).float().mean()))
    return result


def selection_key(task: str, m: dict) -> tuple[float, ...]:
    if task == "regression_only":
        return (-m["value_mae"],)
    if m["balanced_accuracy"] is None:
        raise RuntimeError("A9 selection set lacks a class")
    return (m["balanced_accuracy"],) if task == "classification_only" else (m["balanced_accuracy"], -m["value_mae"])


def fit_probe(*, training: RewardPairDataset, selection: RewardPairDataset, s: dict,
              policy_seed: int, replicate: int, task: str, initial: dict, directory: Path,
              device: torch.device) -> dict:
    training.validate(feature_size=s["model"]["feature_size"])
    selection.validate(feature_size=s["model"]["feature_size"])
    verify_dataset_separation({"training": training, "selection": selection})
    directory.mkdir(parents=True, exist_ok=False)
    model = FixedTimeProbe(s["model"]["feature_size"], s["model"]["hidden_sizes"][0], task).to(device)
    model.load_shared_initial(initial)
    norms = _normalization_from_training(training, s["model"])
    n = {k: v.to(device) for k, v in norms.items()}
    x = (training.features.to(device)-n["feature_mean"])/n["feature_scale"]
    y = training.reward_delta.to(device)
    positive_weight = ((y <= 0).sum()/(y > 0).sum()).to(dtype=torch.float32)
    train_mean = float(training.reward_delta.mean())
    init_seed = s["initialization_seed_offset"] + policy_seed + s["replicate_seed_stride"]*replicate
    batch_seed = s["batch_order_seed_offset"] + policy_seed + s["replicate_seed_stride"]*replicate
    generator = torch.Generator().manual_seed(batch_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=s["learning_rate"])
    order_hash = hashlib.sha256()
    identity = dict(policy_seed=policy_seed, replicate=replicate, arm=task,
                    initialization_seed=init_seed, batch_order_seed=batch_seed)
    inventory = []
    seen = torch.zeros(len(y), dtype=torch.bool)
    best_key: tuple = (-math.inf,)
    best_update = 0
    selected_path = directory / "checkpoint_selected.pt"
    started = time.perf_counter()
    recent: deque[float] = deque(maxlen=s["log_interval_updates"])

    def save(path: Path, update: int, role: str) -> dict:
        payload = dict(**identity, update=update, role=role,
                       probe={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                       normalization=norms, training_reward_mean=train_mean,
                       positive_class_weight=float(positive_weight), config=s["model"],
                       target_horizon=16, label_source="empirical_reward_returns",
                       independent_confirmation_used_for_selection=False, **BOUNDARY)
        torch.save(payload, path)
        return dict(**identity, update=update, role=role, path=_relative(path), sha256=_file_sha256(path))

    inventory.append(save(directory / "checkpoint_00000.pt", 0, "fixed"))
    history = directory / "loss_history.csv"
    log_path = directory / "progress.jsonl"
    bar = progress_bar(range(1, s["maximum_updates"]+1), description=f"A9 {policy_seed}/{replicate}/{task}", unit="批")
    with history.open("w", encoding="utf-8", newline="") as hf, log_path.open("w", encoding="utf-8") as lf:
        writer = None
        for update in bar:
            model.train()
            ix_cpu = torch.randint(len(y), (s["batch_size"],), generator=generator)
            order_hash.update(ix_cpu.numpy().tobytes())
            seen[ix_cpu] = True
            ix = ix_cpu.to(device)
            loss, reg, cls = loss_terms(model, x[ix], y[ix]/n["target_scale"], positive_weight, huber_delta=s["huber_delta"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            recent.append(float(loss.detach()))
            if update in s["fixed_checkpoint_updates"]:
                inventory.append(save(directory / f"checkpoint_{update:05d}.pt", update, "fixed"))
            if update % s["selection_interval_updates"]:
                continue
            v, score = predictions(model, selection, norms, device)
            m = metrics(v, score, selection, train_mean)
            key = selection_key(task, m)
            if key > best_key:
                best_key, best_update = key, update
                selected_record = save(selected_path, update, "selected")
            record = dict(**identity, update=update, total_updates=s["maximum_updates"],
                          mean_training_loss=sum(recent)/len(recent),
                          regression_loss=None if reg is None else float(reg.detach()),
                          classification_loss=None if cls is None else float(cls.detach()),
                          selection_balanced_accuracy=m["balanced_accuracy"], selection_value_mae=m["value_mae"],
                          best_update=best_update, unique_training_pairs_seen=int(seen.sum()),
                          sample_presentations=update*s["batch_size"],
                          **_runtime_fields(start=started, completed=update, total=s["maximum_updates"], device=device))
            if writer is None:
                writer = csv.DictWriter(hf, fieldnames=list(record))
                writer.writeheader()
            writer.writerow(record); hf.flush()
            lf.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n"); lf.flush()
            update_progress(bar, device=device, metrics={"平均损失": record["mean_training_loss"], "判断得分": m["balanced_accuracy"]})
    bar.close()
    if best_update == 0:
        raise RuntimeError("A9 no selected checkpoint")
    inventory.append(selected_record)
    return dict(**identity, updates_completed=s["maximum_updates"], selected_update=best_update,
                inventory=inventory, batch_order_sha256=order_hash.hexdigest(),
                all_training_pairs_seen=bool(seen.all()), unique_training_pairs_seen=int(seen.sum()),
                parameter_count=sum(p.numel() for p in model.parameters()),
                duration_seconds=time.perf_counter()-started, **BOUNDARY)


def validate_inventory(fits: list[dict], s: dict) -> list[dict]:
    expected = {(p, r, a) for p in s["policy_seeds"] for r in s["replicate_indices"] for a in ARMS}
    keys = [(f["policy_seed"], f["replicate"], f["arm"]) for f in fits]
    if len(keys) != len(expected) or set(keys) != expected:
        raise RuntimeError("A9 cannot freeze incomplete fits")
    inventory = []
    for fit in fits:
        if fit["updates_completed"] != s["maximum_updates"]:
            raise RuntimeError("A9 incomplete update budget")
        records = fit["inventory"]
        fixed = [r["update"] for r in records if r["role"] == "fixed"]
        selected = [r for r in records if r["role"] == "selected"]
        if sorted(fixed) != s["fixed_checkpoint_updates"] or len(selected) != 1 or len(records) != len(fixed)+1:
            raise RuntimeError("A9 incomplete checkpoint schedule")
        for rec in records:
            if tuple(rec[k] for k in ("policy_seed", "replicate", "arm")) != tuple(fit[k] for k in ("policy_seed", "replicate", "arm")):
                raise RuntimeError("A9 inventory identity mismatch")
            if _file_sha256(_project_path(rec["path"])) != rec["sha256"]:
                raise RuntimeError("A9 checkpoint hash changed")
        inventory.extend(records)
    for p in s["policy_seeds"]:
        for r in s["replicate_indices"]:
            group = [f for f in fits if f["policy_seed"] == p and f["replicate"] == r]
            if len({f["batch_order_sha256"] for f in group}) != 1:
                raise RuntimeError("A9 paired batch order diverged")
    if len({x["path"] for x in inventory}) != len(inventory):
        raise RuntimeError("A9 reused checkpoint paths")
    return inventory


def seal_training(fits: list[dict], s: dict, output: Path) -> dict:
    marker = output / "TRAINING_FROZEN.json"
    if marker.exists():
        raise FileExistsError(marker)
    inventory = validate_inventory(fits, s)
    write_json(marker, dict(fits=fits, inventory=inventory, confirmation_opened=False, **BOUNDARY))
    return dict(path=_relative(marker), sha256=_file_sha256(marker))


def require_frozen(output: Path, frozen: dict, s: dict) -> list[dict]:
    marker = output / "TRAINING_FROZEN.json"
    if _relative(marker) != frozen["path"] or not marker.is_file() or _file_sha256(marker) != frozen["sha256"]:
        raise RuntimeError("A9 confirmation gate requires the unchanged training freeze")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    return validate_inventory(payload["fits"], s)
