"""R3-D2-A4状态可观测性与标签一致性无梯度诊断。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from src.runtime import resolve_device
from src.rl.s4_r3_pairwise_rank_learnability import (
    classification_metrics,
    cluster_bootstrap_balanced_accuracy,
)
from src.rl.s4_training import _load_yaml, json_safe
from src.training_progress import progress_bar, update_progress


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIEW_IDS = (
    "action_only",
    "current_sensor_plus_action",
    "four_frame_history_plus_action",
    "full_controller_state_plus_action",
)


@dataclass(frozen=True)
class ObservabilityPairDataset:
    states: torch.Tensor
    actions: torch.Tensor
    target_deltas: dict[int, torch.Tensor]
    reward_h32_delta: torch.Tensor
    power_h32_delta: torch.Tensor
    rows: list[dict[str, Any]]

    def validate(self, *, state_size: int, action_size: int) -> None:
        count = self.states.shape[0]
        if self.states.shape != (count, state_size):
            raise ValueError("A4 state shape mismatch")
        if self.actions.shape != (count, action_size):
            raise ValueError("A4 action shape mismatch")
        if len(self.rows) != count:
            raise ValueError("A4 row count mismatch")
        tensors = [
            self.states,
            self.actions,
            self.reward_h32_delta,
            self.power_h32_delta,
            *self.target_deltas.values(),
        ]
        if any(not torch.isfinite(value).all() for value in tensors):
            raise ValueError("A4 dataset contains non-finite values")
        if any(value.shape != (count,) for value in self.target_deltas.values()):
            raise ValueError("A4 target delta shape mismatch")


def run_s4_r3_observability_label_consistency(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    config_path = _project_path(config_path)
    experiment = _load_yaml(config_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight = preflight_s4_r3_observability_label_consistency(
        config_path, experiment, settings, quick=quick
    )
    if preflight_only:
        return preflight

    device = resolve_device(experiment["runtime"]["device"])
    if bool(experiment["runtime"]["deterministic_algorithms"]):
        torch.use_deterministic_algorithms(True)

    output_directory = _project_path(settings["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=False)
    _write_json(output_directory / "preflight.json", preflight)
    _write_json(output_directory / "effective_config.json", experiment)
    source_manifest = _source_manifest(config_path, experiment)
    _write_json(output_directory / "source_manifest.json", source_manifest)

    start = time.perf_counter()
    view_records: list[dict[str, Any]] = []
    neighborhood_records: list[dict[str, Any]] = []
    label_records: list[dict[str, Any]] = []
    data_records: list[dict[str, Any]] = []
    tasks = [
        (seed, view["id"])
        for seed in settings["policy_seeds"]
        for view in settings["views"]
    ]
    progress = progress_bar(tasks, description="A4 局部可观测性", unit="视图")
    cached: dict[int, tuple[ObservabilityPairDataset, ObservabilityPairDataset]] = {}
    for policy_seed, view_id in progress:
        if policy_seed not in cached:
            development = load_observability_pair_dataset(
                settings["datasets"]["development"][policy_seed],
                target_horizons=settings["target_horizons"],
                state_size=settings["state_size"],
                action_size=settings["action_size"],
            )
            validation = load_observability_pair_dataset(
                settings["datasets"]["validation"][policy_seed],
                target_horizons=settings["target_horizons"],
                state_size=settings["state_size"],
                action_size=settings["action_size"],
            )
            if quick:
                development = subsample_observability_dataset(
                    development, settings["maximum_development_pairs"]
                )
                validation = subsample_observability_dataset(
                    validation, settings["maximum_validation_pairs"]
                )
            _verify_split_separation(development, validation)
            cached[policy_seed] = (development, validation)
            data_records.extend(
                [
                    _data_summary(policy_seed, "development", development),
                    _data_summary(policy_seed, "validation", validation),
                ]
            )
            label_records.extend(
                label_consistency_records(policy_seed, "development", development)
            )
            label_records.extend(
                label_consistency_records(policy_seed, "validation", validation)
            )
        development, validation = cached[policy_seed]
        train_features = select_feature_view(development, view_id, settings=settings)
        validation_features = select_feature_view(
            validation, view_id, settings=settings
        )
        predictions, neighbor_info = knn_target_predictions(
            train_features,
            development.target_deltas[32],
            validation_features,
            validation.target_deltas[32],
            k_values=settings["k_values"],
            chunk_size=settings["validation_chunk_size"],
            device=device,
        )
        for k in settings["k_values"]:
            metrics = classification_metrics(
                predictions[k], validation.target_deltas[32]
            )
            constant_mae = float(
                (
                    validation.target_deltas[32]
                    - development.target_deltas[32].mean()
                )
                .abs()
                .mean()
            )
            record: dict[str, Any] = {
                "policy_seed": policy_seed,
                "view": view_id,
                "split": "validation",
                "k": k,
                "primary": k == settings["primary_k"],
                **metrics,
                "constant_mean_mae": constant_mae,
                "mae_better_than_constant": metrics["mae"] < constant_mae,
                "mean_neighbor_distance": neighbor_info[k][
                    "mean_neighbor_distance"
                ],
                "neighbor_sign_agreement": neighbor_info[k][
                    "neighbor_sign_agreement"
                ],
                "mean_neighbor_target_std": neighbor_info[k][
                    "mean_neighbor_target_std"
                ],
            }
            if k == settings["primary_k"]:
                record["balanced_accuracy_cluster_ci"] = (
                    cluster_bootstrap_balanced_accuracy(
                        predictions[k],
                        validation.target_deltas[32],
                        validation.rows,
                        replicates=settings["cluster_bootstrap_replicates"],
                        seed=settings["cluster_bootstrap_seed_offset"]
                        + policy_seed
                        + VIEW_IDS.index(view_id) * 10_000,
                        confidence_level=settings["confidence_level"],
                    )
                )
            view_records.append(record)

        primary_record = next(
            record
            for record in reversed(view_records)
            if record["policy_seed"] == policy_seed
            and record["view"] == view_id
            and record["primary"]
        )

        development_prediction, development_info = leave_episode_out_knn(
            train_features,
            development.target_deltas[32],
            development.rows,
            k=settings["primary_k"],
            chunk_size=settings["validation_chunk_size"],
            device=device,
        )
        development_metrics = classification_metrics(
            development_prediction, development.target_deltas[32]
        )
        neighborhood_records.append(
            {
                "policy_seed": policy_seed,
                "view": view_id,
                "split": "development_leave_episode_out",
                "k": settings["primary_k"],
                **development_metrics,
                **development_info,
            }
        )
        update_progress(
            progress,
            device=device,
            metrics={
                "平衡准确率": primary_record["balanced_accuracy"],
                "MCC": primary_record["matthews_correlation"],
            },
        )

    interpretation = interpret_observability(
        view_records, label_records, settings=settings, quick=quick
    )
    paths = {
        "view_metrics": output_directory / "view_metrics.csv",
        "neighborhood_metrics": output_directory / "neighborhood_metrics.csv",
        "label_consistency": output_directory / "label_consistency.csv",
    }
    _write_rows(paths["view_metrics"], view_records)
    _write_rows(paths["neighborhood_metrics"], neighborhood_records)
    _write_rows(paths["label_consistency"], label_records)
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
            "gradient_updates": 0,
        },
        "state_contract": {
            "history_frames": settings["history_frames"],
            "sensor_features_per_frame": settings["sensor_features_per_frame"],
            "controller_state_features": settings["controller_state_features"],
            "state_size": settings["state_size"],
            "action_size": settings["action_size"],
            "views": settings["views"],
        },
        "data": data_records,
        "view_metrics": view_records,
        "neighborhood_metrics": neighborhood_records,
        "label_consistency": label_records,
        "interpretation": interpretation,
        "inputs": {
            "config": _relative(config_path),
            "config_sha256": _file_sha256(config_path),
            "upstream_summary": experiment["upstream_a3"]["summary"],
            "upstream_summary_sha256": experiment["upstream_a3"][
                "summary_sha256"
            ],
            "source_manifest": source_manifest,
        },
        "records": {
            key: _relative(path) for key, path in paths.items()
        }
        | {
            f"{key}_sha256": _file_sha256(path) for key, path in paths.items()
        },
        "evidence_boundary": {
            "nonparametric_local_diagnostic_only": True,
            "gradient_updates": 0,
            "model_training": False,
            "mechanism_audit_accessed": False,
            "new_physical_episodes_generated": False,
            "full_rl_trained": False,
            "s4d3_accessed": False,
            "real_slm_actions": False,
        },
        "next_action": (
            "Quick output is software evidence only."
            if quick
            else "Stop for read-only audit; do not train RL or expose the mechanism audit set."
        ),
    }
    _write_json(output_directory / "summary.json", summary)
    return summary


def preflight_s4_r3_observability_label_consistency(
    config_path: Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    metadata = experiment["metadata"]
    if not bool(metadata["diagnostic_only"]):
        raise RuntimeError("A4 must remain diagnostic-only")
    for guard in (
        "allow_gradient_updates",
        "allow_model_training",
        "allow_mechanism_audit_access",
        "allow_s4d3_access",
        "allow_real_hardware_actions",
    ):
        if bool(metadata[guard]):
            raise RuntimeError("A4 safety guard was relaxed")
    if tuple(item["id"] for item in experiment["views"]) != VIEW_IDS:
        raise RuntimeError("A4 state view set changed")
    expected_sizes = (11, 53, 179, 221)
    if tuple(int(item["expected_size"]) for item in experiment["views"]) != expected_sizes:
        raise RuntimeError("A4 state view size changed")
    if not bool(experiment["runtime"]["require_cuda"]):
        raise RuntimeError("A4 requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("A4 requires an available CUDA device")
    for key in ("audit_record", "plan"):
        _verify_hash(
            _project_path(experiment["design_contract"][key]),
            experiment["design_contract"][f"{key}_sha256"],
        )
    for key in (
        "summary",
        "preflight",
        "effective_config",
        "source_manifest",
        "fit_summary",
    ):
        _verify_hash(
            _project_path(experiment["upstream_a3"][key]),
            experiment["upstream_a3"][f"{key}_sha256"],
        )
    upstream = json.loads(
        _project_path(experiment["upstream_a3"]["summary"]).read_text(
            encoding="utf-8"
        )
    )
    if upstream["interpretation"]["status"] != experiment["upstream_a3"][
        "required_status"
    ]:
        raise RuntimeError("A4 upstream A3 status changed")
    if "mechanism_audit" in json.dumps(experiment["data"], ensure_ascii=False):
        raise RuntimeError("A4 data configuration exposes mechanism audit")

    split_seeds: dict[str, set[int]] = defaultdict(set)
    pair_counts: dict[str, int] = defaultdict(int)
    for policy_seed in settings["policy_seeds"]:
        for split in ("development", "validation"):
            spec = settings["datasets"][split][policy_seed]
            _verify_hash(_project_path(spec["path"]), spec["sha256"])
            dataset = load_observability_pair_dataset(
                spec,
                target_horizons=settings["target_horizons"],
                state_size=settings["state_size"],
                action_size=settings["action_size"],
            )
            expected = settings[f"expected_{split}_pairs_per_seed"]
            if len(dataset.rows) != expected:
                raise RuntimeError(f"A4 {split} pair count changed")
            pair_counts[split] += len(dataset.rows)
            split_seeds[split].update(
                int(row["episode_seed"]) for row in dataset.rows
            )
    if split_seeds["development"] & split_seeds["validation"]:
        raise RuntimeError("A4 development/validation episode leakage")
    output_directory = _project_path(settings["output_directory"])
    if output_directory.exists():
        raise FileExistsError(f"A4 output already exists: {output_directory}")
    command = (
        ".\\.venv\\Scripts\\python.exe scripts\\diagnose_s4_r3_observability.py "
        "--config configs\\experiments\\s4_r3_observability_label_consistency_v1.yaml"
    )
    return {
        "status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_DIAGNOSTIC",
        "quick": quick,
        "upstream_status": upstream["interpretation"]["status"],
        "policy_seeds": settings["policy_seeds"],
        "views": [item["id"] for item in settings["views"]],
        "primary_k": settings["primary_k"],
        "sensitivity_k": settings["sensitivity_k"],
        "planned_view_seed_analyses": len(settings["policy_seeds"])
        * len(settings["views"]),
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
        "gradient_updates": 0,
        "model_training": False,
        "mechanism_audit_access": False,
        "cuda_required": True,
        "output_directory": settings["output_directory"],
        "s4d3_access": False,
        "real_slm_actions": False,
        "formal_command": command,
    }


def load_observability_pair_dataset(
    spec: dict[str, Any],
    *,
    target_horizons: list[int],
    state_size: int,
    action_size: int,
) -> ObservabilityPairDataset:
    payload = torch.load(
        _project_path(spec["path"]), map_location="cpu", weights_only=False
    )
    states = payload["states"].float()
    actions = payload["actions"].float()
    targets = payload["n_step_targets"].double()
    rewards = payload["empirical_reward_returns"].double()
    powers = payload["empirical_power_returns"].double()
    groups: dict[tuple[Any, ...], dict[str, int]] = defaultdict(dict)
    for index, row in enumerate(payload["rows"]):
        key = (
            row["profile_id"],
            row["condition_id"],
            int(row["probe_step"]),
            int(row["episode_index"]),
            int(row["episode_seed"]),
        )
        groups[key][str(row["candidate"])] = index
    pair_states: list[torch.Tensor] = []
    pair_actions: list[torch.Tensor] = []
    pair_targets: dict[int, list[torch.Tensor]] = {
        horizon: [] for horizon in target_horizons
    }
    pair_rewards: list[torch.Tensor] = []
    pair_powers: list[torch.Tensor] = []
    pair_rows: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda value: tuple(map(str, value))):
        indices = groups[key]
        if set(indices) != {"zero", "actor"}:
            raise ValueError("A4 incomplete action pair")
        zero, actor = indices["zero"], indices["actor"]
        if not torch.equal(states[zero], states[actor]):
            raise ValueError("A4 candidates do not share the same state")
        if float(actions[zero].abs().max()) > 1e-8:
            raise ValueError("A4 zero action is not zero")
        pair_states.append(states[actor])
        pair_actions.append(actions[actor])
        for horizon in target_horizons:
            pair_targets[horizon].append(
                targets[actor, horizon - 1] - targets[zero, horizon - 1]
            )
        pair_rewards.append(rewards[actor, 31] - rewards[zero, 31])
        pair_powers.append(powers[actor, 31] - powers[zero, 31])
        pair_rows.append(
            {
                "profile_id": key[0],
                "condition_id": key[1],
                "probe_step": key[2],
                "episode_index": key[3],
                "episode_seed": key[4],
            }
        )
    dataset = ObservabilityPairDataset(
        states=torch.stack(pair_states).float(),
        actions=torch.stack(pair_actions).float(),
        target_deltas={
            key: torch.stack(value).float() for key, value in pair_targets.items()
        },
        reward_h32_delta=torch.stack(pair_rewards).float(),
        power_h32_delta=torch.stack(pair_powers).float(),
        rows=pair_rows,
    )
    dataset.validate(state_size=state_size, action_size=action_size)
    return dataset


def subsample_observability_dataset(
    dataset: ObservabilityPairDataset, maximum_pairs: int
) -> ObservabilityPairDataset:
    if maximum_pairs >= len(dataset.rows):
        return dataset
    indices = torch.linspace(0, len(dataset.rows) - 1, maximum_pairs).round().long()
    return ObservabilityPairDataset(
        states=dataset.states[indices],
        actions=dataset.actions[indices],
        target_deltas={key: value[indices] for key, value in dataset.target_deltas.items()},
        reward_h32_delta=dataset.reward_h32_delta[indices],
        power_h32_delta=dataset.power_h32_delta[indices],
        rows=[dataset.rows[int(index)] for index in indices],
    )


def select_feature_view(
    dataset: ObservabilityPairDataset,
    view_id: str,
    *,
    settings: dict[str, Any],
) -> torch.Tensor:
    history_size = settings["history_frames"] * settings["sensor_features_per_frame"]
    current_start = history_size - settings["sensor_features_per_frame"]
    if view_id == "action_only":
        result = dataset.actions
    elif view_id == "current_sensor_plus_action":
        result = torch.cat(
            [dataset.states[:, current_start:history_size], dataset.actions], dim=1
        )
    elif view_id == "four_frame_history_plus_action":
        result = torch.cat([dataset.states[:, :history_size], dataset.actions], dim=1)
    elif view_id == "full_controller_state_plus_action":
        result = torch.cat([dataset.states, dataset.actions], dim=1)
    else:
        raise ValueError(f"unknown A4 view: {view_id}")
    expected = next(
        int(item["expected_size"])
        for item in settings["views"]
        if item["id"] == view_id
    )
    if result.shape[1] != expected:
        raise RuntimeError("A4 feature view size mismatch")
    return result


def knn_target_predictions(
    training_features: torch.Tensor,
    training_target: torch.Tensor,
    query_features: torch.Tensor,
    query_target: torch.Tensor,
    *,
    k_values: list[int],
    chunk_size: int,
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], dict[int, dict[str, float]]]:
    mean = training_features.mean(dim=0)
    scale = training_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    training = ((training_features - mean) / scale).to(device)
    queries = ((query_features - mean) / scale).to(device)
    target = training_target.to(device)
    query_target_device = query_target.to(device)
    if queries.shape[0] != query_target_device.shape[0]:
        raise ValueError("A4 query features and targets must have equal length")
    maximum_k = max(k_values)
    predictions: dict[int, list[torch.Tensor]] = {k: [] for k in k_values}
    distances_by_k: dict[int, list[torch.Tensor]] = {k: [] for k in k_values}
    agreement_by_k: dict[int, list[torch.Tensor]] = {k: [] for k in k_values}
    std_by_k: dict[int, list[torch.Tensor]] = {k: [] for k in k_values}
    for start in range(0, queries.shape[0], chunk_size):
        stop = min(start + chunk_size, queries.shape[0])
        distance = torch.cdist(queries[start:stop], training)
        nearest_distance, nearest_index = torch.topk(
            distance, maximum_k, dim=1, largest=False
        )
        nearest_target = target[nearest_index]
        for k in k_values:
            subset = nearest_target[:, :k]
            predictions[k].append(subset.mean(dim=1).cpu())
            distances_by_k[k].append(nearest_distance[:, :k].mean(dim=1).cpu())
            query_sign = query_target_device[start:stop, None] > 0
            agreement_by_k[k].append(
                ((subset > 0) == query_sign).float().mean(dim=1).cpu()
            )
            std_by_k[k].append(subset.std(dim=1, unbiased=False).cpu())
    info = {
        k: {
            "mean_neighbor_distance": float(torch.cat(distances_by_k[k]).mean()),
            "neighbor_sign_agreement": float(torch.cat(agreement_by_k[k]).mean()),
            "mean_neighbor_target_std": float(torch.cat(std_by_k[k]).mean()),
        }
        for k in k_values
    }
    return {k: torch.cat(value) for k, value in predictions.items()}, info


def leave_episode_out_knn(
    features: torch.Tensor,
    target: torch.Tensor,
    rows: list[dict[str, Any]],
    *,
    k: int,
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = ((features - mean) / scale).to(device)
    target_device = target.to(device)
    episode_seed = torch.tensor(
        [int(row["episode_seed"]) for row in rows], device=device
    )
    predictions: list[torch.Tensor] = []
    distances: list[torch.Tensor] = []
    agreements: list[torch.Tensor] = []
    target_stds: list[torch.Tensor] = []
    for start in range(0, normalized.shape[0], chunk_size):
        stop = min(start + chunk_size, normalized.shape[0])
        distance = torch.cdist(normalized[start:stop], normalized)
        same_episode = episode_seed[start:stop, None] == episode_seed[None, :]
        distance.masked_fill_(same_episode, float("inf"))
        nearest_distance, nearest_index = torch.topk(
            distance, k, dim=1, largest=False
        )
        if not torch.isfinite(nearest_distance).all():
            raise RuntimeError("A4 cannot find enough leave-episode-out neighbors")
        neighbors = target_device[nearest_index]
        predictions.append(neighbors.mean(dim=1).cpu())
        distances.append(nearest_distance.mean(dim=1).cpu())
        query_sign = target_device[start:stop, None] > 0
        agreements.append(((neighbors > 0) == query_sign).float().mean(dim=1).cpu())
        target_stds.append(neighbors.std(dim=1, unbiased=False).cpu())
    return torch.cat(predictions), {
        "mean_neighbor_distance": float(torch.cat(distances).mean()),
        "neighbor_sign_agreement": float(torch.cat(agreements).mean()),
        "mean_neighbor_target_std": float(torch.cat(target_stds).mean()),
    }


def label_consistency_records(
    policy_seed: int,
    split: str,
    dataset: ObservabilityPairDataset,
) -> list[dict[str, Any]]:
    comparisons = {
        "target_h16_vs_target_h32": (
            dataset.target_deltas[16],
            dataset.target_deltas[32],
        ),
        "target_h32_vs_reward_h32": (
            dataset.target_deltas[32],
            dataset.reward_h32_delta,
        ),
        "reward_h32_vs_power_h32": (
            dataset.reward_h32_delta,
            dataset.power_h32_delta,
        ),
        "target_h32_vs_power_h32": (
            dataset.target_deltas[32],
            dataset.power_h32_delta,
        ),
    }
    magnitude = dataset.target_deltas[32].abs()
    boundaries = torch.quantile(
        magnitude.double(), torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
    )
    bins = [
        magnitude <= boundaries[0],
        (magnitude > boundaries[0]) & (magnitude <= boundaries[1]),
        (magnitude > boundaries[1]) & (magnitude <= boundaries[2]),
        magnitude > boundaries[2],
    ]
    records: list[dict[str, Any]] = []
    for comparison, (prediction, target) in comparisons.items():
        for margin_bin, mask in [("all", torch.ones_like(magnitude, dtype=torch.bool))] + [
            (f"q{index + 1}", value) for index, value in enumerate(bins)
        ]:
            try:
                metrics = classification_metrics(prediction[mask], target[mask])
                balanced_accuracy: float | None = metrics["balanced_accuracy"]
                matthews_correlation: float | None = metrics[
                    "matthews_correlation"
                ]
            except ValueError:
                balanced_accuracy = None
                matthews_correlation = None
            records.append(
                {
                    "policy_seed": policy_seed,
                    "split": split,
                    "comparison": comparison,
                    "margin_bin": margin_bin,
                    "samples": int(mask.sum()),
                    "sign_agreement": float(
                        ((prediction[mask] > 0) == (target[mask] > 0)).float().mean()
                    ),
                    "balanced_accuracy": balanced_accuracy,
                    "matthews_correlation": matthews_correlation,
                }
            )
    return records


def interpret_observability(
    view_records: list[dict[str, Any]],
    label_records: list[dict[str, Any]],
    *,
    settings: dict[str, Any],
    quick: bool,
) -> dict[str, Any]:
    if quick:
        return {
            "status": "QUICK_SMOKE_ONLY",
            "full_rl_authorized": False,
            "s4d3_authorized": False,
        }
    primary = [
        item
        for item in view_records
        if item["primary"] and item["split"] == "validation"
    ]
    seed_results = []
    for item in primary:
        checks = {
            "balanced_accuracy": item["balanced_accuracy"]
            >= settings["minimum_balanced_accuracy"],
            "cluster_ci_above_chance": item["balanced_accuracy_cluster_ci"]["low"]
            > settings["minimum_balanced_accuracy_ci_low"],
            "matthews_correlation": item["matthews_correlation"]
            >= settings["minimum_matthews_correlation"],
            "mae_better_than_constant": bool(item["mae_better_than_constant"]),
        }
        seed_results.append(
            {
                "policy_seed": item["policy_seed"],
                "view": item["view"],
                "gate": "PASS" if all(checks.values()) else "FAIL",
                "checks": checks,
            }
        )
    view_results = []
    passed_views = set()
    for view in VIEW_IDS:
        selected = [item for item in seed_results if item["view"] == view]
        passed = len(selected) == len(settings["policy_seeds"]) and all(
            item["gate"] == "PASS" for item in selected
        )
        if passed:
            passed_views.add(view)
        view_results.append(
            {"view": view, "all_seed_gate": "PASS" if passed else "FAIL"}
        )
    label_primary = [
        item
        for item in label_records
        if item["split"] == "validation"
        and item["margin_bin"] == "all"
        and item["comparison"]
        in {"target_h16_vs_target_h32", "target_h32_vs_reward_h32"}
    ]
    labels_stable = all(
        item["sign_agreement"] >= settings["minimum_label_sign_agreement"]
        and item["balanced_accuracy"] >= settings["minimum_label_balanced_accuracy"]
        for item in label_primary
    )
    if "action_only" in passed_views:
        status = "ACTION_MAGNITUDE_SHORTCUT_PRESENT"
    elif (
        "current_sensor_plus_action" not in passed_views
        and "four_frame_history_plus_action" in passed_views
    ):
        status = "TEMPORAL_HISTORY_NECESSARY"
    elif (
        "four_frame_history_plus_action" not in passed_views
        and "full_controller_state_plus_action" in passed_views
    ):
        status = "CONTROLLER_STATE_NECESSARY"
    elif not passed_views and not labels_stable:
        status = "HORIZON_OR_BOOTSTRAP_LABEL_INSTABILITY"
    elif not passed_views:
        status = "LOCAL_STATE_ALIASING_OR_MISSING_CONTEXT"
    else:
        status = "LOCAL_SIGNAL_PRESENT_IN_DEPLOYABLE_STATE"
    return {
        "status": status,
        "seed_results": seed_results,
        "view_results": view_results,
        "labels_stable": labels_stable,
        "gradient_updates": 0,
        "full_rl_authorized": False,
        "s4d3_authorized": False,
        "real_hardware_authorized": False,
    }


def _effective_settings(experiment: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    data = experiment["data"]
    neighbors = experiment["nearest_neighbors"]
    statistics = experiment["statistics"]
    thresholds = experiment["diagnostic_thresholds"]
    settings: dict[str, Any] = {
        "output_directory": experiment["outputs"]["directory"],
        "policy_seeds": [int(value) for value in data["policy_seeds"]],
        "datasets": {
            split: {int(seed): spec for seed, spec in data[split].items()}
            for split in ("development", "validation")
        },
        "state_size": int(data["state_size"]),
        "action_size": int(data["action_size"]),
        "history_frames": int(data["history_frames"]),
        "sensor_features_per_frame": int(data["sensor_features_per_frame"]),
        "controller_state_features": int(data["controller_state_features"]),
        "target_horizons": [int(value) for value in data["target_horizons"]],
        "expected_development_pairs_per_seed": int(
            data["expected_development_pairs_per_seed"]
        ),
        "expected_validation_pairs_per_seed": int(
            data["expected_validation_pairs_per_seed"]
        ),
        "views": experiment["views"],
        "primary_k": int(neighbors["primary_k"]),
        "sensitivity_k": [int(value) for value in neighbors["sensitivity_k"]],
        "validation_chunk_size": int(neighbors["validation_chunk_size"]),
        "cluster_bootstrap_replicates": int(
            statistics["cluster_bootstrap_replicates"]
        ),
        "cluster_bootstrap_seed_offset": int(
            statistics["cluster_bootstrap_seed_offset"]
        ),
        "confidence_level": float(statistics["confidence_level"]),
        "minimum_balanced_accuracy": float(
            thresholds["minimum_balanced_accuracy"]
        ),
        "minimum_balanced_accuracy_ci_low": float(
            thresholds["minimum_balanced_accuracy_ci_low"]
        ),
        "minimum_matthews_correlation": float(
            thresholds["minimum_matthews_correlation"]
        ),
        "minimum_label_sign_agreement": float(
            thresholds["minimum_label_sign_agreement"]
        ),
        "minimum_label_balanced_accuracy": float(
            thresholds["minimum_label_balanced_accuracy"]
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
                "cluster_bootstrap_replicates": int(
                    quick_config["cluster_bootstrap_replicates"]
                ),
                "sensitivity_k": [
                    int(value) for value in quick_config["sensitivity_k"]
                ],
            }
        )
    settings["k_values"] = sorted(
        {settings["primary_k"], *settings["sensitivity_k"]}
    )
    return settings


def _verify_split_separation(
    development: ObservabilityPairDataset,
    validation: ObservabilityPairDataset,
) -> None:
    development_seeds = {int(row["episode_seed"]) for row in development.rows}
    validation_seeds = {int(row["episode_seed"]) for row in validation.rows}
    if development_seeds & validation_seeds:
        raise RuntimeError("A4 episode leakage")


def _data_summary(
    policy_seed: int,
    split: str,
    dataset: ObservabilityPairDataset,
) -> dict[str, Any]:
    return {
        "policy_seed": policy_seed,
        "split": split,
        "pairs": len(dataset.rows),
        "episode_seeds": len({int(row["episode_seed"]) for row in dataset.rows}),
        "actor_better_fraction_h32": float(
            (dataset.target_deltas[32] > 0).float().mean()
        ),
        "target_h32_mean": float(dataset.target_deltas[32].mean()),
    }


def _source_manifest(
    config_path: Path, experiment: dict[str, Any]
) -> dict[str, str]:
    paths = [
        config_path,
        _project_path("scripts/diagnose_s4_r3_observability.py"),
        _project_path("src/rl/s4_r3_observability_label_consistency.py"),
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
    if _file_sha256(path) != expected:
        raise RuntimeError(f"A4 hash mismatch: {path}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("A4 output rows must not be empty")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)
