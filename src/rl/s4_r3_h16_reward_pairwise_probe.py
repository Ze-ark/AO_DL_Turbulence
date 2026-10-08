"""R3-D2-A6：用16步实际累计奖励训练成对动作排序探针。"""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.runtime import resolve_device
from src.rl.s4_r3_pairwise_rank_learnability import (
    PairwiseDeltaProbe,
    classification_metrics,
    cluster_bootstrap_balanced_accuracy,
)
from src.rl.s4_training import _load_yaml, json_safe
from src.training_progress import progress_bar, update_progress


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OBJECTIVE_IDS = (
    "paired_delta_regression",
    "paired_delta_plus_balanced_sign",
)
SPLITS = ("development", "validation", "independent_test")


@dataclass(frozen=True)
class RewardPairDataset:
    """同一状态下演员动作相对零动作的16步奖励和功率差。"""

    features: torch.Tensor
    reward_delta: torch.Tensor
    power_delta: torch.Tensor
    rows: list[dict[str, Any]]

    @property
    def target_delta(self) -> torch.Tensor:
        """兼容A3的归一化和统计术语，A6目标始终是奖励差。"""

        return self.reward_delta

    def validate(self, *, feature_size: int) -> None:
        if self.features.ndim != 2 or self.features.shape[1] != feature_size:
            raise ValueError("A6 pair feature shape mismatch")
        expected = (self.features.shape[0],)
        if self.reward_delta.shape != expected or self.power_delta.shape != expected:
            raise ValueError("A6 pair target shape mismatch")
        if len(self.rows) != self.features.shape[0]:
            raise ValueError("A6 pair row count mismatch")
        if not all(
            bool(torch.isfinite(value).all())
            for value in (self.features, self.reward_delta, self.power_delta)
        ):
            raise ValueError("A6 pair dataset contains non-finite values")
        labels = self.reward_delta > 0
        if not bool(labels.any()) or not bool((~labels).any()):
            raise ValueError("A6 pair dataset must contain both reward ranking classes")


def run_s4_r3_h16_reward_pairwise_probe(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """预检或运行A6；正式训练只由用户在IDE手动启动。"""

    config_path = _project_path(config_path)
    experiment = _load_yaml(config_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight = preflight_s4_r3_h16_reward_pairwise_probe(
        config_path,
        experiment,
        settings,
        quick=quick,
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
    subgroup_rows: list[dict[str, Any]] = []
    data_summary: list[dict[str, Any]] = []
    for policy_seed in settings["policy_seeds"]:
        datasets = {
            split: load_reward_pair_dataset(
                settings["datasets"][split][policy_seed],
                target_source=settings["target_source"],
                power_source=settings["power_source"],
                horizon_index=settings["target_horizon_index"],
                state_size=settings["state_size"],
                action_size=settings["action_size"],
            )
            for split in SPLITS
        }
        if quick:
            datasets = {
                split: subsample_reward_pair_dataset(
                    dataset,
                    settings[f"maximum_{split}_pairs"],
                )
                for split, dataset in datasets.items()
            }
        _verify_split_separation(datasets)
        data_summary.extend(
            _pair_dataset_summary(policy_seed, split, datasets[split])
            for split in SPLITS
        )
        normalization = _normalization_from_training(
            datasets["development"], settings
        )
        initial_state = _initial_probe_state(
            settings=settings,
            policy_seed=policy_seed,
            device=device,
        )
        for objective in settings["objectives"]:
            fit, groups = fit_h16_reward_probe(
                development=datasets["development"],
                validation=datasets["validation"],
                independent_test=datasets["independent_test"],
                objective=objective,
                settings=settings,
                policy_seed=policy_seed,
                normalization=normalization,
                initial_state=initial_state,
                output_directory=output_directory,
                device=device,
            )
            fits.append(fit)
            subgroup_rows.extend(groups)

    interpretation = interpret_h16_reward_fits(
        fits,
        settings=settings,
        quick=quick,
    )
    fit_summary_path = output_directory / "fit_summary.csv"
    subgroup_path = output_directory / "independent_test_subgroups.csv"
    _write_rows(fit_summary_path, [_fit_csv_row(fit) for fit in fits])
    _write_rows(subgroup_path, subgroup_rows)
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
        "single_factor_contract": {
            "a3_model_and_training_budget_held_fixed": True,
            "changed_factor": "target label only",
            "a3_target": "32-step bootstrapped critic target delta",
            "a6_target": "16-step empirical cumulative reward delta",
            "model_selection_split": "validation",
            "independent_test_used_for_selection": False,
        },
        "inputs": {
            "config": _relative(config_path),
            "config_sha256": _file_sha256(config_path),
            "upstream_a5r1_summary": experiment["upstream_a5r1"]["summary"],
            "upstream_a5r1_summary_sha256": experiment["upstream_a5r1"][
                "summary_sha256"
            ],
            "upstream_a3_summary": experiment["upstream_a3"]["summary"],
            "upstream_a3_summary_sha256": experiment["upstream_a3"][
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
            "independent_test_subgroups": _relative(subgroup_path),
            "independent_test_subgroups_sha256": _file_sha256(subgroup_path),
            "subgroup_rows": len(subgroup_rows),
        },
        "evidence_boundary": {
            "supervised_probe_training_only": True,
            "new_simulation_episodes_generated": False,
            "independent_test_used_for_training_or_selection": False,
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
            else "Stop for read-only audit; do not start critic or RL training automatically."
        ),
    }
    _write_json(output_directory / "summary.json", summary)
    return summary


def preflight_s4_r3_h16_reward_pairwise_probe(
    config_path: Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    """封存上游证据、数据拆分、单因素对照与安全边界。"""

    metadata = experiment["metadata"]
    required_false = (
        "allow_new_episode_generation",
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
        raise RuntimeError("A6 must remain diagnostic-only")
    if not bool(metadata["allow_supervised_probe_training"]):
        raise RuntimeError("A6 supervised-probe authorization is missing")
    if any(bool(metadata[name]) for name in required_false):
        raise RuntimeError("A6 safety guard was relaxed")
    if tuple(item["id"] for item in experiment["objectives"]) != OBJECTIVE_IDS:
        raise RuntimeError("A6 objective set changed")
    if (
        settings["target_source"] != "empirical_reward_returns"
        or settings["power_source"] != "empirical_power_returns"
        or settings["target_horizon"] != 16
        or settings["target_horizon_index"] != 15
    ):
        raise RuntimeError("A6 fixed H16 empirical target changed")
    if not bool(experiment["runtime"]["require_cuda"]):
        raise RuntimeError("A6 formal probe requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("A6 requires an available CUDA device")

    for key in ("audit_record", "plan"):
        path = _project_path(experiment["design_contract"][key])
        _verify_hash(path, experiment["design_contract"][f"{key}_sha256"])
    for section, keys in (
        (
            "upstream_a5r1",
            (
                "summary",
                "preflight",
                "effective_config",
                "source_manifest",
                "confirmation_metrics",
                "decision",
                "collection_progress",
            ),
        ),
        (
            "upstream_a3",
            (
                "summary",
                "preflight",
                "effective_config",
                "source_manifest",
                "fit_summary",
                "audit_record",
                "experiment_config",
            ),
        ),
    ):
        for key in keys:
            path = _project_path(experiment[section][key])
            _verify_hash(path, experiment[section][f"{key}_sha256"])

    a5r1_summary = _read_json(experiment["upstream_a5r1"]["summary"])
    a5r1_decision = a5r1_summary["decision"]
    if a5r1_decision["status"] != experiment["upstream_a5r1"]["required_status"]:
        raise RuntimeError("A6 upstream A5-R1 decision changed")
    if not bool(a5r1_decision["supervised_probe_design_authorized"]):
        raise RuntimeError("A6 was not authorized by A5-R1")
    if bool(a5r1_decision["full_rl_authorized"]):
        raise RuntimeError("A5-R1 unexpectedly authorized full RL")

    a3_summary = _read_json(experiment["upstream_a3"]["summary"])
    if a3_summary["interpretation"]["status"] != experiment["upstream_a3"][
        "required_status"
    ]:
        raise RuntimeError("A6 upstream A3 negative result changed")
    _verify_a3_single_factor_contract(experiment, settings)

    episode_seeds: dict[str, set[int]] = defaultdict(set)
    pair_counts: dict[str, int] = defaultdict(int)
    for policy_seed in settings["policy_seeds"]:
        for split in SPLITS:
            spec = settings["datasets"][split][policy_seed]
            path = _project_path(spec["path"])
            _verify_hash(path, spec["sha256"])
            dataset = load_reward_pair_dataset(
                spec,
                target_source=settings["target_source"],
                power_source=settings["power_source"],
                horizon_index=settings["target_horizon_index"],
                state_size=settings["state_size"],
                action_size=settings["action_size"],
            )
            expected = settings[f"expected_{split}_pairs_per_seed"]
            if dataset.features.shape[0] != expected:
                raise RuntimeError(f"A6 {split} pair count changed")
            episode_seeds[split].update(
                int(row["episode_seed"]) for row in dataset.rows
            )
            pair_counts[split] += int(dataset.features.shape[0])
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            if episode_seeds[left] & episode_seeds[right]:
                raise RuntimeError(f"A6 episode leakage between {left} and {right}")

    if "mechanism_audit" in json.dumps(experiment["data"], ensure_ascii=False):
        raise RuntimeError("A6 data configuration exposes mechanism audit")
    output_directory = _project_path(settings["output_directory"])
    if output_directory.exists():
        raise FileExistsError(f"A6 output already exists: {output_directory}")

    formal_command = (
        ".\\.venv\\Scripts\\python.exe scripts\\train_s4_r3_h16_reward_probe.py "
        "--config configs\\experiments\\s4_r3_h16_reward_pairwise_probe_v1.yaml"
    )
    return {
        "status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_TRAINING",
        "quick": quick,
        "upstream_a5r1_status": a5r1_decision["status"],
        "upstream_a3_status": a3_summary["interpretation"]["status"],
        "single_factor_change": "H32 bootstrapped target -> H16 empirical reward",
        "policy_seeds": settings["policy_seeds"],
        "objectives": [item["id"] for item in settings["objectives"]],
        "planned_probe_fits": len(settings["policy_seeds"])
        * len(settings["objectives"]),
        "development_pairs": _preflight_pair_total(
            pair_counts, settings, "development", quick
        ),
        "validation_pairs": _preflight_pair_total(
            pair_counts, settings, "validation", quick
        ),
        "independent_test_pairs": _preflight_pair_total(
            pair_counts, settings, "independent_test", quick
        ),
        "maximum_updates_per_fit": settings["maximum_updates"],
        "independent_test_used_for_model_selection": False,
        "new_episode_generation": False,
        "mechanism_audit_access": False,
        "original_critic_updates": 0,
        "actor_updates": 0,
        "alpha_updates": 0,
        "student_updates": 0,
        "full_rl_training": False,
        "cuda_required": True,
        "output_directory": settings["output_directory"],
        "s4d3_access": False,
        "real_slm_actions": False,
        "formal_command": formal_command,
    }


def load_reward_pair_dataset(
    spec: dict[str, Any],
    *,
    target_source: str,
    power_source: str,
    horizon_index: int,
    state_size: int,
    action_size: int,
) -> RewardPairDataset:
    """从冻结张量建立零动作/演员动作一一对应的H16样本。"""

    payload = torch.load(
        _project_path(spec["path"]), map_location="cpu", weights_only=False
    )
    states = payload["states"].float()
    actions = payload["actions"].float()
    rewards = payload[target_source].double()
    powers = payload[power_source].double()
    rows = payload["rows"]
    if states.shape[1] != state_size or actions.shape[1] != action_size:
        raise ValueError("A6 upstream state/action size changed")
    if rewards.ndim != 2 or powers.shape != rewards.shape:
        raise ValueError("A6 empirical return tensor shape mismatch")
    if horizon_index < 0 or horizon_index >= rewards.shape[1]:
        raise ValueError("A6 H16 index is unavailable")

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
            raise ValueError("A6 duplicate candidate in pair")
        groups[key][candidate] = index

    pair_features: list[torch.Tensor] = []
    reward_deltas: list[torch.Tensor] = []
    power_deltas: list[torch.Tensor] = []
    pair_rows: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda value: tuple(map(str, value))):
        indices = groups[key]
        if set(indices) != {"zero", "actor"}:
            raise ValueError("A6 incomplete zero/actor pair")
        zero_index, actor_index = indices["zero"], indices["actor"]
        if not torch.equal(states[zero_index], states[actor_index]):
            raise ValueError("A6 paired candidates do not share the same state")
        if float(actions[zero_index].abs().max()) > 1e-8:
            raise ValueError("A6 zero candidate action is not zero")
        pair_features.append(torch.cat([states[actor_index], actions[actor_index]]))
        reward_deltas.append(
            rewards[actor_index, horizon_index] - rewards[zero_index, horizon_index]
        )
        power_deltas.append(
            powers[actor_index, horizon_index] - powers[zero_index, horizon_index]
        )
        pair_rows.append(
            {
                "profile_id": str(key[0]),
                "condition_id": str(key[1]),
                "probe_step": int(key[2]),
                "episode_index": int(key[3]),
                "episode_seed": int(key[4]),
            }
        )
    dataset = RewardPairDataset(
        features=torch.stack(pair_features).float(),
        reward_delta=torch.stack(reward_deltas).float(),
        power_delta=torch.stack(power_deltas).float(),
        rows=pair_rows,
    )
    dataset.validate(feature_size=state_size + action_size)
    return dataset


def subsample_reward_pair_dataset(
    dataset: RewardPairDataset, maximum_pairs: int
) -> RewardPairDataset:
    if maximum_pairs >= dataset.features.shape[0]:
        return dataset
    indices = torch.linspace(
        0, dataset.features.shape[0] - 1, steps=maximum_pairs
    ).round().long()
    result = RewardPairDataset(
        features=dataset.features[indices],
        reward_delta=dataset.reward_delta[indices],
        power_delta=dataset.power_delta[indices],
        rows=[dataset.rows[int(index)] for index in indices],
    )
    result.validate(feature_size=dataset.features.shape[1])
    return result


def fit_h16_reward_probe(
    *,
    development: RewardPairDataset,
    validation: RewardPairDataset,
    independent_test: RewardPairDataset,
    objective: dict[str, Any],
    settings: dict[str, Any],
    policy_seed: int,
    normalization: dict[str, torch.Tensor],
    initial_state: dict[str, torch.Tensor],
    output_directory: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """只用开发集训练、验证集选点，最后一次性评估独立测试集。"""

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
    x_train = (development.features.to(device) - feature_mean) / feature_scale
    y_train = development.reward_delta.to(device)
    x_validation = (
        validation.features.to(device) - feature_mean
    ) / feature_scale
    positive = int((development.reward_delta > 0).sum())
    negative = int(development.reward_delta.numel() - positive)
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
        description=f"A6 种子{policy_seed} {method}",
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
        if not should_validate:
            continue
        prediction = _predict_delta(
            model,
            x_validation,
            target_scale=target_scale,
        ).cpu()
        metrics = classification_metrics(prediction, validation.reward_delta)
        score = (metrics["balanced_accuracy"], metrics["mae"])
        improved = score[0] > best_score[0] + 1e-12 or (
            abs(score[0] - best_score[0]) <= 1e-12 and score[1] < best_score[1]
        )
        if improved:
            best_score = score
            best_update = update
            torch.save(
                {
                    "algorithm": "h16_empirical_reward_pairwise_supervised_probe",
                    "label_source": settings["target_source"],
                    "target_horizon": settings["target_horizon"],
                    "policy_seed": policy_seed,
                    "objective": deepcopy(objective),
                    "config": {
                        "feature_size": settings["feature_size"],
                        "hidden_size": settings["hidden_size"],
                    },
                    "best_update": best_update,
                    "best_validation_metrics": metrics,
                    "probe": model.state_dict(),
                    "normalization": {
                        key: value.cpu() for key, value in normalization.items()
                    },
                    "training_reward_mean": development.reward_delta.mean().cpu(),
                    "independent_test_used_for_selection": False,
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
                "验证平衡准确率": metrics["balanced_accuracy"],
                "验证MCC": metrics["matthews_correlation"],
            },
        )
        if update - best_update >= settings["early_stopping_patience_updates"]:
            break

    _write_rows(loss_path, records)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["probe"])
    training_mean = float(checkpoint["training_reward_mean"])
    validation_prediction = _predict_dataset(
        model,
        validation,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_scale=target_scale,
        device=device,
    )
    test_prediction = _predict_dataset(
        model,
        independent_test,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_scale=target_scale,
        device=device,
    )
    validation_metrics = evaluate_h16_prediction(
        validation_prediction,
        validation,
        training_reward_mean=training_mean,
        magnitude_threshold=settings["high_magnitude_thresholds"][policy_seed],
        bootstrap_replicates=settings["cluster_bootstrap_replicates"],
        bootstrap_seed=settings["cluster_bootstrap_seed_offset"] + policy_seed,
        confidence_level=settings["confidence_level"],
    )
    independent_test_metrics = evaluate_h16_prediction(
        test_prediction,
        independent_test,
        training_reward_mean=training_mean,
        magnitude_threshold=settings["high_magnitude_thresholds"][policy_seed],
        bootstrap_replicates=settings["cluster_bootstrap_replicates"],
        bootstrap_seed=settings["cluster_bootstrap_seed_offset"] + 10000 + policy_seed,
        confidence_level=settings["confidence_level"],
    )
    subgroup_rows = independent_subgroup_metrics(
        test_prediction,
        independent_test,
        policy_seed=policy_seed,
        objective=method,
    )
    valid_subgroups = [
        float(row["balanced_accuracy"])
        for row in subgroup_rows
        if row["balanced_accuracy"] is not None
    ]
    subgroup_minimum = min(valid_subgroups) if valid_subgroups else None
    subgroup_all_valid = len(valid_subgroups) == len(subgroup_rows)
    return (
        {
            "policy_seed": policy_seed,
            "objective": method,
            "updates_completed": int(records[-1]["update"]),
            "best_update": int(checkpoint["best_update"]),
            "development_pairs": int(development.features.shape[0]),
            "validation_pairs": int(validation.features.shape[0]),
            "independent_test_pairs": int(independent_test.features.shape[0]),
            "validation": validation_metrics,
            "independent_test": independent_test_metrics,
            "independent_test_subgroup_minimum_balanced_accuracy": subgroup_minimum,
            "independent_test_subgroups_all_two_class": subgroup_all_valid,
            "checkpoint": _relative(checkpoint_path),
            "checkpoint_sha256": _file_sha256(checkpoint_path),
            "loss_history": _relative(loss_path),
            "progress_log": _relative(progress_path),
            "independent_test_used_for_selection": False,
            "original_critic_updates": 0,
            "actor_updates": 0,
            "alpha_updates": 0,
            "student_updates": 0,
        },
        subgroup_rows,
    )


def evaluate_h16_prediction(
    prediction: torch.Tensor,
    dataset: RewardPairDataset,
    *,
    training_reward_mean: float,
    magnitude_threshold: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """同时报告训练目标、独立功率和冻结高幅值层。"""

    reward = classification_metrics(prediction, dataset.reward_delta)
    reward["constant_training_mean_mae"] = float(
        (dataset.reward_delta - training_reward_mean).abs().mean()
    )
    reward["mae_better_than_constant"] = (
        reward["mae"] < reward["constant_training_mean_mae"]
    )
    reward["balanced_accuracy_cluster_ci"] = (
        cluster_bootstrap_balanced_accuracy(
            prediction,
            dataset.reward_delta,
            dataset.rows,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            confidence_level=confidence_level,
        )
    )
    power_alignment = _safe_classification_metrics(prediction, dataset.power_delta)
    high_mask = dataset.reward_delta.abs() >= magnitude_threshold
    high_magnitude = _safe_classification_metrics(
        prediction[high_mask], dataset.reward_delta[high_mask]
    )
    high_magnitude["samples"] = int(high_mask.sum())
    high_magnitude["threshold"] = float(magnitude_threshold)
    return {
        "reward_ranking": reward,
        "physical_power_alignment": power_alignment,
        "high_magnitude_reward_ranking": high_magnitude,
    }


def independent_subgroup_metrics(
    prediction: torch.Tensor,
    dataset: RewardPairDataset,
    *,
    policy_seed: int,
    objective: str,
) -> list[dict[str, Any]]:
    """在固定测试集按硬件画像和湍流条件检查平均数掩盖的反转。"""

    rows: list[dict[str, Any]] = []
    for group_kind, field in (
        ("profile", "profile_id"),
        ("condition", "condition_id"),
    ):
        group_ids = sorted({str(row[field]) for row in dataset.rows})
        for group_id in group_ids:
            indices = torch.tensor(
                [
                    index
                    for index, row in enumerate(dataset.rows)
                    if str(row[field]) == group_id
                ],
                dtype=torch.long,
            )
            metrics = _safe_classification_metrics(
                prediction[indices], dataset.reward_delta[indices]
            )
            rows.append(
                {
                    "policy_seed": policy_seed,
                    "objective": objective,
                    "group_kind": group_kind,
                    "group_id": group_id,
                    "samples": int(indices.numel()),
                    "actor_better_fraction": float(
                        (dataset.reward_delta[indices] > 0).float().mean()
                    ),
                    "balanced_accuracy": metrics.get("balanced_accuracy"),
                    "matthews_correlation": metrics.get("matthews_correlation"),
                    "mae": metrics.get("mae"),
                    "two_class": metrics["status"] == "OK",
                }
            )
    return rows


def interpret_h16_reward_fits(
    fits: list[dict[str, Any]],
    *,
    settings: dict[str, Any],
    quick: bool,
) -> dict[str, Any]:
    if quick:
        return {
            "status": "QUICK_SMOKE_ONLY",
            "critic_design_authorized": False,
            "full_rl_authorized": False,
            "s4d3_authorized": False,
            "real_hardware_authorized": False,
        }

    seed_results: list[dict[str, Any]] = []
    for fit in fits:
        validation_checks = _gate_checks(fit["validation"], settings)
        test_checks = _gate_checks(fit["independent_test"], settings)
        subgroup_minimum = fit[
            "independent_test_subgroup_minimum_balanced_accuracy"
        ]
        subgroup_check = bool(fit["independent_test_subgroups_all_two_class"]) and (
            subgroup_minimum is not None
            and float(subgroup_minimum)
            >= settings["minimum_subgroup_balanced_accuracy"]
        )
        all_checks = (
            all(validation_checks.values())
            and all(test_checks.values())
            and subgroup_check
        )
        seed_results.append(
            {
                "policy_seed": fit["policy_seed"],
                "objective": fit["objective"],
                "validation_gate": (
                    "PASS" if all(validation_checks.values()) else "FAIL"
                ),
                "validation_checks": validation_checks,
                "independent_test_gate": (
                    "PASS" if all(test_checks.values()) else "FAIL"
                ),
                "independent_test_checks": test_checks,
                "subgroup_gate": "PASS" if subgroup_check else "FAIL",
                "all_gates": "PASS" if all_checks else "FAIL",
            }
        )

    method_results: list[dict[str, Any]] = []
    for objective in OBJECTIVE_IDS:
        selected = [row for row in seed_results if row["objective"] == objective]
        method_results.append(
            {
                "objective": objective,
                "all_seed_validation_gate": _all_seed_gate(
                    selected,
                    settings,
                    "validation_gate",
                ),
                "all_seed_independent_test_gate": _all_seed_gate(
                    selected,
                    settings,
                    "independent_test_gate",
                ),
                "all_seed_subgroup_gate": _all_seed_gate(
                    selected,
                    settings,
                    "subgroup_gate",
                ),
                "all_seed_final_gate": _all_seed_gate(
                    selected,
                    settings,
                    "all_gates",
                ),
            }
        )
    passed = {
        row["objective"]
        for row in method_results
        if row["all_seed_final_gate"] == "PASS"
    }
    if set(OBJECTIVE_IDS) <= passed:
        status = "STABLE_H16_LABEL_RESTORES_PAIRWISE_LEARNABILITY"
    elif "paired_delta_plus_balanced_sign" in passed:
        status = "STABLE_H16_LABEL_LEARNABLE_WITH_RANKING_LOSS"
    elif "paired_delta_regression" in passed:
        status = "STABLE_H16_LABEL_LEARNABLE_WITH_REGRESSION"
    elif any(
        row["all_seed_validation_gate"] == "PASS"
        and row["all_seed_independent_test_gate"] == "PASS"
        and row["all_seed_subgroup_gate"] == "FAIL"
        for row in method_results
    ):
        status = "AGGREGATE_LEARNABILITY_HAS_SUBGROUP_REVERSAL"
    elif any(
        row["all_seed_validation_gate"] == "PASS"
        and row["all_seed_independent_test_gate"] == "FAIL"
        for row in method_results
    ):
        status = "DEVELOPMENT_LEARNABILITY_NOT_REPLICATED"
    else:
        status = "STABLE_H16_LABEL_NOT_LEARNABLE_WITH_CURRENT_REPRESENTATION"
    critic_design_authorized = bool(passed) and settings[
        "allow_critic_design_authorization"
    ]
    return {
        "status": status,
        "seed_results": seed_results,
        "method_results": method_results,
        "passed_objectives": sorted(passed),
        "critic_design_authorized": critic_design_authorized,
        "authorization_scope": (
            "non_bootstrapped_critic_design_only" if critic_design_authorized else "none"
        ),
        "full_rl_authorized": False,
        "s4d3_authorized": False,
        "real_hardware_authorized": False,
    }


def _effective_settings(
    experiment: dict[str, Any], *, quick: bool
) -> dict[str, Any]:
    data = experiment["data"]
    training = experiment["probe_training"]
    statistics = experiment["statistics"]
    gate = experiment["gate"]
    settings: dict[str, Any] = {
        "output_directory": experiment["outputs"]["directory"],
        "policy_seeds": [int(value) for value in data["policy_seeds"]],
        "datasets": {
            split: {int(seed): value for seed, value in data[split].items()}
            for split in SPLITS
        },
        "target_source": str(data["target_source"]),
        "power_source": str(data["power_source"]),
        "target_horizon": int(data["target_horizon"]),
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
        "expected_independent_test_pairs_per_seed": int(
            data["expected_independent_test_pairs_per_seed"]
        ),
        "high_magnitude_thresholds": {
            int(seed): float(value)
            for seed, value in data["high_magnitude_thresholds"].items()
        },
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
        "minimum_subgroup_balanced_accuracy": float(
            gate["minimum_subgroup_balanced_accuracy"]
        ),
        "require_all_policy_seeds": bool(gate["require_all_policy_seeds"]),
        "allow_critic_design_authorization": bool(
            gate["allow_critic_design_authorization"]
        ),
    }
    if quick:
        quick_config = experiment["quick"]
        settings.update(
            {
                "output_directory": experiment["outputs"]["quick_directory"],
                "policy_seeds": [int(value) for value in quick_config["policy_seeds"]],
                "maximum_development_pairs": int(
                    quick_config["maximum_development_pairs"]
                ),
                "maximum_validation_pairs": int(
                    quick_config["maximum_validation_pairs"]
                ),
                "maximum_independent_test_pairs": int(
                    quick_config["maximum_independent_test_pairs"]
                ),
                "maximum_updates": int(quick_config["maximum_updates"]),
                "validation_interval_updates": int(
                    quick_config["validation_interval_updates"]
                ),
                "log_interval_updates": int(quick_config["log_interval_updates"]),
                "early_stopping_patience_updates": int(
                    quick_config["early_stopping_patience_updates"]
                ),
                "batch_size": int(quick_config["batch_size"]),
                "cluster_bootstrap_replicates": int(
                    quick_config["cluster_bootstrap_replicates"]
                ),
            }
        )
    return settings


def _normalization_from_training(
    dataset: RewardPairDataset, settings: dict[str, Any]
) -> dict[str, torch.Tensor]:
    return {
        "feature_mean": dataset.features.mean(dim=0),
        "feature_scale": dataset.features.std(dim=0, unbiased=False).clamp_min(
            settings["feature_scale_floor"]
        ),
        "target_scale": dataset.reward_delta.std(unbiased=False).clamp_min(
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


@torch.no_grad()
def _predict_delta(
    model: nn.Module,
    features: torch.Tensor,
    *,
    target_scale: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    prediction = model(features) * target_scale
    model.train()
    return prediction


def _predict_dataset(
    model: nn.Module,
    dataset: RewardPairDataset,
    *,
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
    target_scale: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    features = (dataset.features.to(device) - feature_mean) / feature_scale
    return _predict_delta(model, features, target_scale=target_scale).cpu()


def _safe_classification_metrics(
    prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, Any]:
    if target.numel() == 0:
        return {"status": "EMPTY"}
    labels = target > 0
    if not bool(labels.any()) or not bool((~labels).any()):
        return {
            "status": "ONE_CLASS",
            "positive_class_fraction": float(labels.float().mean()),
            "mae": float((prediction.double() - target.double()).abs().mean()),
        }
    return {"status": "OK", **classification_metrics(prediction, target)}


def _gate_checks(metrics: dict[str, Any], settings: dict[str, Any]) -> dict[str, bool]:
    reward = metrics["reward_ranking"]
    return {
        "balanced_accuracy": float(reward["balanced_accuracy"])
        >= settings["minimum_balanced_accuracy"],
        "cluster_ci_above_chance": float(
            reward["balanced_accuracy_cluster_ci"]["low"]
        )
        > settings["minimum_balanced_accuracy_ci_low"],
        "matthews_correlation": float(reward["matthews_correlation"])
        >= settings["minimum_matthews_correlation"],
        "mae_better_than_constant": bool(reward["mae_better_than_constant"]),
    }


def _all_seed_gate(
    selected: list[dict[str, Any]],
    settings: dict[str, Any],
    field: str,
) -> str:
    complete = len(selected) == len(settings["policy_seeds"])
    passed = all(row[field] == "PASS" for row in selected)
    return "PASS" if complete and passed else "FAIL"


def _verify_split_separation(datasets: dict[str, RewardPairDataset]) -> None:
    seed_sets = {
        split: {int(row["episode_seed"]) for row in dataset.rows}
        for split, dataset in datasets.items()
    }
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            if seed_sets[left] & seed_sets[right]:
                raise RuntimeError(f"A6 pair split leakage between {left} and {right}")


def _pair_dataset_summary(
    policy_seed: int, split: str, dataset: RewardPairDataset
) -> dict[str, Any]:
    reward_positive = dataset.reward_delta > 0
    power_positive = dataset.power_delta > 0
    return {
        "policy_seed": policy_seed,
        "split": split,
        "pairs": int(dataset.features.shape[0]),
        "feature_size": int(dataset.features.shape[1]),
        "actor_better_fraction": float(reward_positive.float().mean()),
        "reward_delta_mean": float(dataset.reward_delta.mean()),
        "power_delta_mean": float(dataset.power_delta.mean()),
        "reward_power_sign_agreement": float(
            (reward_positive == power_positive).float().mean()
        ),
        "independent_episode_seeds": len(
            {int(row["episode_seed"]) for row in dataset.rows}
        ),
    }


def _fit_csv_row(fit: dict[str, Any]) -> dict[str, Any]:
    validation = fit["validation"]["reward_ranking"]
    test = fit["independent_test"]["reward_ranking"]
    power = fit["independent_test"]["physical_power_alignment"]
    high = fit["independent_test"]["high_magnitude_reward_ranking"]
    return {
        "policy_seed": fit["policy_seed"],
        "objective": fit["objective"],
        "updates_completed": fit["updates_completed"],
        "best_update": fit["best_update"],
        "validation_balanced_accuracy": validation["balanced_accuracy"],
        "validation_ci_low": validation["balanced_accuracy_cluster_ci"]["low"],
        "validation_mcc": validation["matthews_correlation"],
        "validation_mae": validation["mae"],
        "validation_constant_mae": validation["constant_training_mean_mae"],
        "test_balanced_accuracy": test["balanced_accuracy"],
        "test_ci_low": test["balanced_accuracy_cluster_ci"]["low"],
        "test_mcc": test["matthews_correlation"],
        "test_mae": test["mae"],
        "test_constant_mae": test["constant_training_mean_mae"],
        "test_power_balanced_accuracy": power.get("balanced_accuracy"),
        "test_high_magnitude_balanced_accuracy": high.get("balanced_accuracy"),
        "test_subgroup_minimum_balanced_accuracy": fit[
            "independent_test_subgroup_minimum_balanced_accuracy"
        ],
        "checkpoint": fit["checkpoint"],
        "checkpoint_sha256": fit["checkpoint_sha256"],
    }


def _verify_a3_single_factor_contract(
    experiment: dict[str, Any], settings: dict[str, Any]
) -> None:
    a3_config = _read_json(experiment["upstream_a3"]["effective_config"])
    if a3_config["objectives"] != experiment["objectives"]:
        raise RuntimeError("A6 objectives are not identical to A3")
    if (
        int(a3_config["data"]["state_size"]) != settings["state_size"]
        or int(a3_config["data"]["action_size"]) != settings["action_size"]
    ):
        raise RuntimeError("A6 feature dimensions are not identical to A3")
    fixed_training_fields = (
        "hidden_size",
        "learning_rate",
        "batch_size",
        "maximum_updates",
        "validation_interval_updates",
        "log_interval_updates",
        "early_stopping_patience_updates",
        "huber_delta",
        "initialization_seed_offset",
        "batch_order_seed_offset",
        "target_scale_floor",
        "feature_scale_floor",
    )
    for field in fixed_training_fields:
        if a3_config["probe_training"][field] != experiment["probe_training"][field]:
            raise RuntimeError(f"A6 single-factor contract changed {field}")
    for split in ("development", "validation"):
        for seed in settings["policy_seeds"]:
            a3_spec = a3_config["data"][split][str(seed)]
            if experiment["data"][split][seed] != a3_spec:
                raise RuntimeError(f"A6 {split} data differs from A3")


def _preflight_pair_total(
    pair_counts: dict[str, int],
    settings: dict[str, Any],
    split: str,
    quick: bool,
) -> int:
    if not quick:
        return pair_counts[split]
    return len(settings["policy_seeds"]) * settings[f"maximum_{split}_pairs"]


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
    paths = [_project_path(path) for path in experiment["tracked_source_files"]]
    if config_path not in paths:
        paths.insert(0, config_path)
    return {_relative(path): _file_sha256(path) for path in paths}


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_project_path(path).read_text(encoding="utf-8"))


def _verify_hash(path: Path, expected: str) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if _file_sha256(path) != expected:
        raise RuntimeError(f"A6 hash mismatch: {path}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("A6 row output must not be empty")
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
