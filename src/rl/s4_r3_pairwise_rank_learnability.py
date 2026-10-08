"""R3-D2-A3成对动作排序可学习性诊断。"""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.runtime import resolve_device
from src.rl.s4_training import _load_yaml, json_safe
from src.training_progress import progress_bar, update_progress


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OBJECTIVE_IDS = (
    "paired_delta_regression",
    "paired_delta_plus_balanced_sign",
)


@dataclass(frozen=True)
class PairDataset:
    """同一状态下演员动作与零动作的32步目标差。"""

    features: torch.Tensor
    target_delta: torch.Tensor
    rows: list[dict[str, Any]]

    def validate(self, *, feature_size: int) -> None:
        if self.features.ndim != 2 or self.features.shape[1] != feature_size:
            raise ValueError("pair feature shape mismatch")
        if self.target_delta.shape != (self.features.shape[0],):
            raise ValueError("pair target shape mismatch")
        if len(self.rows) != self.features.shape[0]:
            raise ValueError("pair row count mismatch")
        if not torch.isfinite(self.features).all() or not torch.isfinite(
            self.target_delta
        ).all():
            raise ValueError("pair dataset contains non-finite values")
        labels = self.target_delta > 0
        if not bool(labels.any()) or not bool((~labels).any()):
            raise ValueError("pair dataset must contain both ranking classes")


class PairwiseDeltaProbe(nn.Module):
    """直接预测演员动作相对零动作的长期目标差。"""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def run_s4_r3_pairwise_rank_learnability(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """预检或运行A3诊断；正式入口只由用户在IDE启动。"""

    config_path = _project_path(config_path)
    experiment = _load_yaml(config_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight = preflight_s4_r3_pairwise_rank_learnability(
        config_path, experiment, settings, quick=quick
    )
    if preflight_only:
        return preflight

    device = resolve_device(experiment["runtime"]["device"])
    if bool(experiment["runtime"]["deterministic_algorithms"]):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False

    output_directory = _project_path(settings["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=False)
    _write_json(output_directory / "preflight.json", preflight)
    _write_json(output_directory / "effective_config.json", experiment)
    source_manifest = _source_manifest(config_path, experiment)
    _write_json(output_directory / "source_manifest.json", source_manifest)

    start = time.perf_counter()
    fits: list[dict[str, Any]] = []
    data_summary: list[dict[str, Any]] = []
    for policy_seed in settings["policy_seeds"]:
        development = load_pair_dataset(
            settings["datasets"]["development"][policy_seed],
            horizon_index=settings["target_horizon_index"],
            state_size=settings["state_size"],
            action_size=settings["action_size"],
        )
        validation = load_pair_dataset(
            settings["datasets"]["validation"][policy_seed],
            horizon_index=settings["target_horizon_index"],
            state_size=settings["state_size"],
            action_size=settings["action_size"],
        )
        if quick:
            development = subsample_pair_dataset(
                development, settings["maximum_development_pairs"]
            )
            validation = subsample_pair_dataset(
                validation, settings["maximum_validation_pairs"]
            )
        _verify_pair_split_separation(development, validation)
        data_summary.extend(
            [
                _pair_dataset_summary(policy_seed, "development", development),
                _pair_dataset_summary(policy_seed, "validation", validation),
            ]
        )
        normalization = _normalization_from_training(development, settings)
        initial_state = _initial_probe_state(
            settings=settings,
            policy_seed=policy_seed,
            device=device,
        )
        for objective in settings["objectives"]:
            fits.append(
                fit_pairwise_probe(
                    development=development,
                    validation=validation,
                    objective=objective,
                    settings=settings,
                    policy_seed=policy_seed,
                    normalization=normalization,
                    initial_state=initial_state,
                    output_directory=output_directory,
                    device=device,
                )
            )

    interpretation = interpret_pairwise_fits(fits, settings=settings, quick=quick)
    fit_summary_path = output_directory / "fit_summary.csv"
    _write_rows(fit_summary_path, fits)
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": __import__("datetime")
            .datetime.now()
            .astimezone()
            .isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": experiment["metadata"]["version_label"],
        },
        "experiment": {
            "id": experiment["metadata"]["experiment_id"],
            "status": "quick_smoke_only" if quick else "completed_pending_audit",
            "quick": quick,
            "duration_seconds": time.perf_counter() - start,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "inputs": {
            "config": _relative(config_path),
            "config_sha256": _file_sha256(config_path),
            "upstream_summary": experiment["upstream_r3d2a"]["summary"],
            "upstream_summary_sha256": experiment["upstream_r3d2a"][
                "summary_sha256"
            ],
            "source_manifest": source_manifest,
        },
        "data": data_summary,
        "fits": fits,
        "interpretation": interpretation,
        "records": {
            "fit_summary": _relative(fit_summary_path),
            "fit_summary_sha256": _file_sha256(fit_summary_path),
            "fit_rows": len(fits),
        },
        "evidence_boundary": {
            "developmental_supervised_diagnostic_only": True,
            "mechanism_audit_accessed": False,
            "original_critic_updates": 0,
            "actor_updates": 0,
            "alpha_updates": 0,
            "student_updates": 0,
            "full_rl_trained": False,
            "s4d3_accessed": False,
            "real_slm_actions": False,
        },
        "next_action": (
            "Quick output is software evidence only."
            if quick
            else "Stop for read-only audit; do not start RL or access the revealed mechanism audit set."
        ),
    }
    _write_json(output_directory / "summary.json", summary)
    return summary


def preflight_s4_r3_pairwise_rank_learnability(
    config_path: Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    """锁定上游哈希、数据范围、规模和安全保护。"""

    metadata = experiment["metadata"]
    required_false = (
        "allow_full_rl_training",
        "allow_original_critic_updates",
        "allow_actor_updates",
        "allow_alpha_updates",
        "allow_student_updates",
        "allow_mechanism_audit_access",
        "allow_s4d3_access",
        "allow_real_hardware_actions",
    )
    if not bool(metadata["diagnostic_only"]):
        raise RuntimeError("A3 must remain diagnostic-only")
    if any(bool(metadata[name]) for name in required_false):
        raise RuntimeError("A3 safety guard was relaxed")
    if tuple(item["id"] for item in experiment["objectives"]) != OBJECTIVE_IDS:
        raise RuntimeError("A3 objective set changed")
    if not bool(experiment["runtime"]["require_cuda"]):
        raise RuntimeError("A3 formal diagnostic requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("A3 requires an available CUDA device")

    for key in ("audit_record", "plan"):
        path = _project_path(experiment["design_contract"][key])
        _verify_hash(path, experiment["design_contract"][f"{key}_sha256"])
    for key in (
        "summary",
        "preflight",
        "effective_config",
        "source_manifest",
        "fit_summary",
    ):
        path = _project_path(experiment["upstream_r3d2a"][key])
        _verify_hash(path, experiment["upstream_r3d2a"][f"{key}_sha256"])
    upstream_summary = json.loads(
        _project_path(experiment["upstream_r3d2a"]["summary"]).read_text(
            encoding="utf-8"
        )
    )
    if (
        upstream_summary["interpretation"]["status"]
        != experiment["upstream_r3d2a"]["required_status"]
    ):
        raise RuntimeError("A3 upstream failure status changed")
    boundary = upstream_summary["evidence_boundary"]
    if boundary["actor_updates"] != 0 or boundary["alpha_updates"] != 0:
        raise RuntimeError("A3 upstream freeze boundary failed")

    episode_seeds: dict[str, set[int]] = defaultdict(set)
    pair_counts: dict[str, int] = defaultdict(int)
    for policy_seed in settings["policy_seeds"]:
        for split in ("development", "validation"):
            spec = settings["datasets"][split][policy_seed]
            path = _project_path(spec["path"])
            _verify_hash(path, spec["sha256"])
            pairs = load_pair_dataset(
                spec,
                horizon_index=settings["target_horizon_index"],
                state_size=settings["state_size"],
                action_size=settings["action_size"],
            )
            expected = settings[f"expected_{split}_pairs_per_seed"]
            if pairs.features.shape[0] != expected:
                raise RuntimeError(f"A3 {split} pair count changed")
            episode_seeds[split].update(int(row["episode_seed"]) for row in pairs.rows)
            pair_counts[split] += int(pairs.features.shape[0])
    if episode_seeds["development"] & episode_seeds["validation"]:
        raise RuntimeError("A3 development and validation episode leakage")

    output_directory = _project_path(settings["output_directory"])
    if output_directory.exists():
        raise FileExistsError(f"A3 output already exists: {output_directory}")
    if "mechanism_audit" in json.dumps(experiment["data"], ensure_ascii=False):
        raise RuntimeError("A3 data configuration must not expose mechanism audit")

    formal_command = (
        ".\\.venv\\Scripts\\python.exe scripts\\train_s4_r3_pairwise_rank_probe.py "
        "--config configs\\experiments\\s4_r3_pairwise_rank_learnability_v1.yaml"
    )
    return {
        "status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_TRAINING",
        "quick": quick,
        "upstream_status": upstream_summary["interpretation"]["status"],
        "policy_seeds": settings["policy_seeds"],
        "objectives": [item["id"] for item in settings["objectives"]],
        "planned_probe_fits": len(settings["policy_seeds"])
        * len(settings["objectives"]),
        "development_pairs": (
            len(settings["policy_seeds"]) * settings["maximum_development_pairs"]
            if quick
            else pair_counts["development"]
        ),
        "validation_pairs": (
            len(settings["policy_seeds"]) * settings["maximum_validation_pairs"]
            if quick
            else pair_counts["validation"]
        ),
        "maximum_updates_per_fit": settings["maximum_updates"],
        "mechanism_audit_access": False,
        "original_critic_updates": 0,
        "actor_updates": 0,
        "alpha_updates": 0,
        "student_updates": 0,
        "cuda_required": True,
        "output_directory": settings["output_directory"],
        "s4d3_access": False,
        "real_slm_actions": False,
        "formal_command": formal_command,
    }


def load_pair_dataset(
    spec: dict[str, Any],
    *,
    horizon_index: int,
    state_size: int,
    action_size: int,
) -> PairDataset:
    """从A阶段冻结数据建立严格的一对一动作差值样本。"""

    path = _project_path(spec["path"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    states = payload["states"].float()
    actions = payload["actions"].float()
    targets = payload["n_step_targets"].double()
    rows = payload["rows"]
    if states.shape[1] != state_size or actions.shape[1] != action_size:
        raise ValueError("A3 upstream state/action size changed")
    groups: dict[tuple[Any, ...], dict[str, int]] = defaultdict(dict)
    for index, row in enumerate(rows):
        key = (
            row["profile_id"],
            row["condition_id"],
            int(row["probe_step"]),
            int(row["episode_index"]),
            int(row["episode_seed"]),
        )
        candidate = str(row["candidate"])
        if candidate in groups[key]:
            raise ValueError("A3 duplicate candidate in pair")
        groups[key][candidate] = index

    pair_features: list[torch.Tensor] = []
    pair_targets: list[torch.Tensor] = []
    pair_rows: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda value: tuple(map(str, value))):
        indices = groups[key]
        if set(indices) != {"zero", "actor"}:
            raise ValueError("A3 incomplete zero/actor pair")
        zero_index, actor_index = indices["zero"], indices["actor"]
        if not torch.equal(states[zero_index], states[actor_index]):
            raise ValueError("A3 paired candidates do not share the same state")
        if float(actions[zero_index].abs().max()) > 1e-8:
            raise ValueError("A3 zero candidate action is not zero")
        pair_features.append(torch.cat([states[actor_index], actions[actor_index]]))
        pair_targets.append(
            targets[actor_index, horizon_index] - targets[zero_index, horizon_index]
        )
        pair_rows.append(
            {
                "profile_id": key[0],
                "condition_id": key[1],
                "probe_step": int(key[2]),
                "episode_index": int(key[3]),
                "episode_seed": int(key[4]),
            }
        )
    dataset = PairDataset(
        features=torch.stack(pair_features).float(),
        target_delta=torch.stack(pair_targets).float(),
        rows=pair_rows,
    )
    dataset.validate(feature_size=state_size + action_size)
    return dataset


def subsample_pair_dataset(dataset: PairDataset, maximum_pairs: int) -> PairDataset:
    if maximum_pairs >= dataset.features.shape[0]:
        return dataset
    indices = torch.linspace(
        0, dataset.features.shape[0] - 1, steps=maximum_pairs
    ).round().long()
    return PairDataset(
        features=dataset.features[indices],
        target_delta=dataset.target_delta[indices],
        rows=[dataset.rows[int(index)] for index in indices],
    )


def fit_pairwise_probe(
    *,
    development: PairDataset,
    validation: PairDataset,
    objective: dict[str, Any],
    settings: dict[str, Any],
    policy_seed: int,
    normalization: dict[str, torch.Tensor],
    initial_state: dict[str, torch.Tensor],
    output_directory: Path,
    device: torch.device,
) -> dict[str, Any]:
    """用相同初始化和批次顺序训练单个成对诊断探针。"""

    method = str(objective["id"])
    fit_directory = output_directory / f"seed_{policy_seed}" / method
    fit_directory.mkdir(parents=True, exist_ok=False)
    progress_path = fit_directory / "progress.jsonl"
    loss_path = fit_directory / "loss_history.csv"
    checkpoint_path = fit_directory / "checkpoint_best.pt"

    model = PairwiseDeltaProbe(settings["feature_size"], settings["hidden_size"]).to(
        device
    )
    model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    feature_mean = normalization["feature_mean"].to(device)
    feature_scale = normalization["feature_scale"].to(device)
    target_scale = normalization["target_scale"].to(device)
    x_train = ((development.features.to(device) - feature_mean) / feature_scale)
    y_train = development.target_delta.to(device)
    x_validation = (
        validation.features.to(device) - feature_mean
    ) / feature_scale
    y_validation = validation.target_delta.to(device)
    positive = int((development.target_delta > 0).sum())
    negative = int(development.target_delta.numel() - positive)
    positive_weight = torch.tensor(negative / positive, device=device)
    generator = torch.Generator(device="cpu").manual_seed(
        settings["batch_order_seed_offset"] + policy_seed
    )

    best_score = (-math.inf, math.inf)
    best_update = 0
    recent_losses: deque[float] = deque(maxlen=100)
    records: list[dict[str, Any]] = []
    start = time.perf_counter()
    progress = progress_bar(
        range(1, settings["maximum_updates"] + 1),
        description=f"A3 种子{policy_seed} {method}",
        unit="批",
    )
    for update in progress:
        indices = torch.randint(
            0,
            x_train.shape[0],
            (settings["batch_size"],),
            generator=generator,
        ).to(device)
        prediction_scaled = model(x_train[indices])
        target_scaled = y_train[indices] / target_scale
        regression_loss = F.huber_loss(
            prediction_scaled,
            target_scaled,
            delta=settings["huber_delta"],
        )
        sign_loss = F.binary_cross_entropy_with_logits(
            prediction_scaled,
            (y_train[indices] > 0).float(),
            pos_weight=positive_weight,
        )
        loss = regression_loss + float(objective["balanced_sign_weight"]) * sign_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
        recent_losses.append(loss_value)

        should_validate = (
            update % settings["validation_interval_updates"] == 0
            or update == settings["maximum_updates"]
        )
        if should_validate:
            prediction = _predict_delta(
                model, x_validation, target_scale=target_scale
            )
            metrics = classification_metrics(
                prediction.cpu(), y_validation.cpu()
            )
            score = (metrics["balanced_accuracy"], metrics["mae"])
            improved = score[0] > best_score[0] + 1e-12 or (
                abs(score[0] - best_score[0]) <= 1e-12 and score[1] < best_score[1]
            )
            if improved:
                best_score = score
                best_update = update
                torch.save(
                    {
                        "algorithm": "pairwise_delta_supervised_probe",
                        "policy_seed": policy_seed,
                        "objective": deepcopy(objective),
                        "config": {
                            "feature_size": settings["feature_size"],
                            "hidden_size": settings["hidden_size"],
                        },
                        "best_update": best_update,
                        "best_metrics": metrics,
                        "probe": model.state_dict(),
                        "normalization": {
                            key: value.cpu() for key, value in normalization.items()
                        },
                        "original_critic_updates": 0,
                        "actor_updates": 0,
                        "alpha_updates": 0,
                        "student_updates": 0,
                    },
                    checkpoint_path,
                )
            runtime = _runtime_fields(
                start=start,
                completed=update,
                total=settings["maximum_updates"],
                device=device,
            )
            record = {
                "policy_seed": policy_seed,
                "objective": method,
                "update": update,
                "training_loss": loss_value,
                "mean_training_loss": sum(recent_losses) / len(recent_losses),
                **metrics,
                **runtime,
            }
            records.append(record)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(json_safe(record), ensure_ascii=False) + "\n")
            update_progress(
                progress,
                device=device,
                metrics={
                    "平均损失": record["mean_training_loss"],
                    "平衡准确率": metrics["balanced_accuracy"],
                    "MCC": metrics["matthews_correlation"],
                },
            )
            if update - best_update >= settings["early_stopping_patience_updates"]:
                break

    _write_rows(loss_path, records)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["probe"])
    final_prediction = _predict_delta(model, x_validation, target_scale=target_scale).cpu()
    final_metrics = classification_metrics(final_prediction, validation.target_delta)
    final_metrics["constant_mean_mae"] = float(
        (validation.target_delta - development.target_delta.mean()).abs().mean()
    )
    final_metrics["mae_better_than_constant"] = (
        final_metrics["mae"] < final_metrics["constant_mean_mae"]
    )
    final_metrics["balanced_accuracy_cluster_ci"] = cluster_bootstrap_balanced_accuracy(
        final_prediction,
        validation.target_delta,
        validation.rows,
        replicates=settings["cluster_bootstrap_replicates"],
        seed=settings["cluster_bootstrap_seed_offset"] + policy_seed,
        confidence_level=settings["confidence_level"],
    )
    return {
        "policy_seed": policy_seed,
        "objective": method,
        "updates_completed": int(records[-1]["update"]),
        "best_update": int(checkpoint["best_update"]),
        "development_pairs": int(development.features.shape[0]),
        "validation_pairs": int(validation.features.shape[0]),
        **final_metrics,
        "checkpoint": _relative(checkpoint_path),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "loss_history": _relative(loss_path),
        "progress_log": _relative(progress_path),
        "original_critic_updates": 0,
        "actor_updates": 0,
        "alpha_updates": 0,
        "student_updates": 0,
    }


@torch.no_grad()
def _predict_delta(
    model: nn.Module, features: torch.Tensor, *, target_scale: torch.Tensor
) -> torch.Tensor:
    model.eval()
    prediction = model(features) * target_scale
    model.train()
    return prediction


def classification_metrics(
    prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    """返回不受类别比例支配的排序指标。"""

    prediction = prediction.detach().double().flatten()
    target = target.detach().double().flatten()
    predicted_positive = prediction > 0
    actual_positive = target > 0
    tp = int((predicted_positive & actual_positive).sum())
    tn = int((~predicted_positive & ~actual_positive).sum())
    fp = int((predicted_positive & ~actual_positive).sum())
    fn = int((~predicted_positive & actual_positive).sum())
    positive_total, negative_total = tp + fn, tn + fp
    if positive_total == 0 or negative_total == 0:
        raise ValueError("balanced metrics require both target classes")
    sensitivity = tp / positive_total
    specificity = tn / negative_total
    denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    raw_accuracy = (tp + tn) / target.numel()
    majority = max(positive_total, negative_total) / target.numel()
    return {
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "matthews_correlation": mcc,
        "raw_accuracy": raw_accuracy,
        "majority_class_accuracy": majority,
        "positive_class_fraction": positive_total / target.numel(),
        "sensitivity_actor_better": sensitivity,
        "specificity_actor_worse": specificity,
        "mae": float((prediction - target).abs().mean()),
        "rmse": float(torch.sqrt(((prediction - target) ** 2).mean())),
    }


def cluster_bootstrap_balanced_accuracy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    rows: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """以完整动态条件×回合为单位重采样平衡准确率。"""

    clusters: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        clusters[f"{row['condition_id']}:{row['episode_index']}"].append(index)
    keys = sorted(clusters)
    generator = random.Random(seed)
    values: list[float] = []
    for _ in range(replicates):
        sampled = [generator.choice(keys) for _ in keys]
        indices = [index for key in sampled for index in clusters[key]]
        try:
            values.append(
                classification_metrics(prediction[indices], target[indices])[
                    "balanced_accuracy"
                ]
            )
        except ValueError:
            continue
    if len(values) < max(10, int(0.9 * replicates)):
        raise RuntimeError("insufficient valid cluster bootstrap replicates")
    tensor = torch.tensor(values, dtype=torch.float64)
    alpha = 1.0 - confidence_level
    return {
        "low": float(torch.quantile(tensor, alpha / 2)),
        "high": float(torch.quantile(tensor, 1.0 - alpha / 2)),
        "replicates": len(values),
        "clusters": len(keys),
        "unit": "condition_x_episode",
    }


def interpret_pairwise_fits(
    fits: list[dict[str, Any]], *, settings: dict[str, Any], quick: bool
) -> dict[str, Any]:
    if quick:
        return {
            "status": "QUICK_SMOKE_ONLY",
            "full_rl_authorized": False,
            "s4d3_authorized": False,
        }
    seed_results = []
    for fit in fits:
        checks = {
            "balanced_accuracy": float(fit["balanced_accuracy"])
            >= settings["minimum_balanced_accuracy"],
            "cluster_ci_above_chance": float(
                fit["balanced_accuracy_cluster_ci"]["low"]
            )
            > settings["minimum_balanced_accuracy_ci_low"],
            "matthews_correlation": float(fit["matthews_correlation"])
            >= settings["minimum_matthews_correlation"],
            "mae_better_than_constant": bool(fit["mae_better_than_constant"]),
        }
        seed_results.append(
            {
                "policy_seed": fit["policy_seed"],
                "objective": fit["objective"],
                "gate": "PASS" if all(checks.values()) else "FAIL",
                "checks": checks,
            }
        )
    method_results = []
    for objective in OBJECTIVE_IDS:
        selected = [x for x in seed_results if x["objective"] == objective]
        method_results.append(
            {
                "objective": objective,
                "all_seed_gate": (
                    "PASS"
                    if len(selected) == len(settings["policy_seeds"])
                    and all(x["gate"] == "PASS" for x in selected)
                    else "FAIL"
                ),
            }
        )
    passed = {
        item["objective"]
        for item in method_results
        if item["all_seed_gate"] == "PASS"
    }
    if set(OBJECTIVE_IDS) <= passed:
        status = "INDIRECT_ABSOLUTE_Q_DIFFERENCING_BOTTLENECK"
    elif "paired_delta_plus_balanced_sign" in passed:
        status = "RANKING_OBJECTIVE_BOTTLENECK"
    else:
        status = "PAIRWISE_SIGNAL_NOT_LEARNABLE_WITH_CURRENT_REPRESENTATION"
    return {
        "status": status,
        "seed_results": seed_results,
        "method_results": method_results,
        "independent_unseen_holdout_required": bool(passed),
        "full_rl_authorized": False,
        "s4d3_authorized": False,
        "real_hardware_authorized": False,
    }


def _effective_settings(experiment: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    data = experiment["data"]
    training = experiment["probe_training"]
    statistics = experiment["statistics"]
    gate = experiment["gate"]
    settings: dict[str, Any] = {
        "output_directory": experiment["outputs"]["directory"],
        "policy_seeds": [int(x) for x in data["policy_seeds"]],
        "datasets": {
            split: {int(seed): value for seed, value in data[split].items()}
            for split in ("development", "validation")
        },
        "target_horizon_index": int(data["target_horizon_index"]),
        "state_size": int(data["state_size"]),
        "action_size": int(data["action_size"]),
        "feature_size": int(data["state_size"]) + int(data["action_size"]),
        "expected_development_pairs_per_seed": int(
            data["expected_development_pairs_per_seed"]
        ),
        "expected_validation_pairs_per_seed": int(
            data["expected_validation_pairs_per_seed"]
        ),
        "objectives": deepcopy(experiment["objectives"]),
        "hidden_size": int(training["hidden_size"]),
        "learning_rate": float(training["learning_rate"]),
        "batch_size": int(training["batch_size"]),
        "maximum_updates": int(training["maximum_updates"]),
        "validation_interval_updates": int(training["validation_interval_updates"]),
        "log_interval_updates": int(training["log_interval_updates"]),
        "early_stopping_patience_updates": int(
            training["early_stopping_patience_updates"]
        ),
        "huber_delta": float(training["huber_delta"]),
        "initialization_seed_offset": int(training["initialization_seed_offset"]),
        "batch_order_seed_offset": int(training["batch_order_seed_offset"]),
        "target_scale_floor": float(training["target_scale_floor"]),
        "feature_scale_floor": float(training["feature_scale_floor"]),
        "cluster_bootstrap_replicates": int(
            statistics["cluster_bootstrap_replicates"]
        ),
        "cluster_bootstrap_seed_offset": int(
            statistics["cluster_bootstrap_seed_offset"]
        ),
        "confidence_level": float(statistics["confidence_level"]),
        "minimum_balanced_accuracy": float(gate["minimum_balanced_accuracy"]),
        "minimum_balanced_accuracy_ci_low": float(
            gate["minimum_balanced_accuracy_ci_low"]
        ),
        "minimum_matthews_correlation": float(
            gate["minimum_matthews_correlation"]
        ),
    }
    if quick:
        q = experiment["quick"]
        settings.update(
            {
                "output_directory": experiment["outputs"]["quick_directory"],
                "policy_seeds": [int(x) for x in q["policy_seeds"]],
                "maximum_development_pairs": int(q["maximum_development_pairs"]),
                "maximum_validation_pairs": int(q["maximum_validation_pairs"]),
                "maximum_updates": int(q["maximum_updates"]),
                "validation_interval_updates": int(
                    q["validation_interval_updates"]
                ),
                "log_interval_updates": int(q["log_interval_updates"]),
                "early_stopping_patience_updates": int(
                    q["early_stopping_patience_updates"]
                ),
                "batch_size": int(q["batch_size"]),
                "cluster_bootstrap_replicates": int(
                    q["cluster_bootstrap_replicates"]
                ),
            }
        )
    return settings


def _normalization_from_training(
    dataset: PairDataset, settings: dict[str, Any]
) -> dict[str, torch.Tensor]:
    return {
        "feature_mean": dataset.features.mean(dim=0),
        "feature_scale": dataset.features.std(dim=0, unbiased=False).clamp_min(
            settings["feature_scale_floor"]
        ),
        "target_scale": dataset.target_delta.std(unbiased=False).clamp_min(
            settings["target_scale_floor"]
        ),
    }


def _initial_probe_state(
    *, settings: dict[str, Any], policy_seed: int, device: torch.device
) -> dict[str, torch.Tensor]:
    torch.manual_seed(settings["initialization_seed_offset"] + policy_seed)
    torch.cuda.manual_seed_all(settings["initialization_seed_offset"] + policy_seed)
    model = PairwiseDeltaProbe(settings["feature_size"], settings["hidden_size"]).to(
        device
    )
    return deepcopy(model.state_dict())


def _verify_pair_split_separation(
    development: PairDataset, validation: PairDataset
) -> None:
    dev = {int(row["episode_seed"]) for row in development.rows}
    val = {int(row["episode_seed"]) for row in validation.rows}
    if dev & val:
        raise RuntimeError("A3 pair split episode leakage")


def _pair_dataset_summary(
    policy_seed: int, split: str, dataset: PairDataset
) -> dict[str, Any]:
    positive_fraction = float((dataset.target_delta > 0).float().mean())
    return {
        "policy_seed": policy_seed,
        "split": split,
        "pairs": int(dataset.features.shape[0]),
        "feature_size": int(dataset.features.shape[1]),
        "actor_better_fraction": positive_fraction,
        "actor_worse_fraction": 1.0 - positive_fraction,
        "target_delta_mean": float(dataset.target_delta.mean()),
        "independent_episode_seeds": len(
            {int(row["episode_seed"]) for row in dataset.rows}
        ),
    }


def _runtime_fields(
    *, start: float, completed: int, total: int, device: torch.device
) -> dict[str, float]:
    elapsed = time.perf_counter() - start
    remaining = elapsed / max(completed, 1) * max(total - completed, 0)
    allocated = 0.0
    reserved = 0.0
    if device.type == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
    return {
        "elapsed_seconds": elapsed,
        "estimated_remaining_seconds": remaining,
        "cuda_allocated_gb": allocated,
        "cuda_reserved_gb": reserved,
    }


def _source_manifest(
    config_path: Path, experiment: dict[str, Any]
) -> dict[str, str]:
    paths = [
        config_path,
        _project_path("scripts/train_s4_r3_pairwise_rank_probe.py"),
        _project_path("src/rl/s4_r3_pairwise_rank_learnability.py"),
        _project_path("src/runtime.py"),
        _project_path("src/training_progress.py"),
        _project_path(experiment["design_contract"]["audit_record"]),
        _project_path(experiment["design_contract"]["plan"]),
    ]
    return {_relative(path): _file_sha256(path) for path in paths}


def _verify_hash(path: Path, expected: str) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    actual = _file_sha256(path)
    if actual != expected:
        raise RuntimeError(f"A3 hash mismatch: {path}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("A3 row output must not be empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)
