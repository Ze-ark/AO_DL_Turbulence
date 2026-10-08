"""R3-D2-A5纯物理目标时域稳定性无梯度扫描。"""

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
from src.rl.s4_r3_pairwise_rank_learnability import classification_metrics
from src.rl.s4_training import _load_yaml, json_safe
from src.training_progress import progress_bar, update_progress


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_CANDIDATES = {"zero", "actor"}


@dataclass(frozen=True)
class PhysicalTargetPairDataset:
    reward_deltas: torch.Tensor
    power_deltas: torch.Tensor
    target_deltas: torch.Tensor
    rows: list[dict[str, Any]]

    def validate(self, *, maximum_horizon: int) -> None:
        count = len(self.rows)
        expected = (count, maximum_horizon)
        for name, value in (
            ("reward", self.reward_deltas),
            ("power", self.power_deltas),
            ("target", self.target_deltas),
        ):
            if value.shape != expected:
                raise ValueError(f"A5 {name} curve shape mismatch")
            if not torch.isfinite(value).all():
                raise ValueError(f"A5 {name} curve contains non-finite values")


def run_s4_r3_physical_target_stability(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    config_path = _project_path(config_path)
    experiment = _load_yaml(config_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight = preflight_s4_r3_physical_target_stability(
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

    started = time.perf_counter()
    cached: dict[int, dict[str, PhysicalTargetPairDataset]] = {}
    magnitude_thresholds: dict[int, dict[int, float]] = {}
    data_records: list[dict[str, Any]] = []
    for policy_seed in settings["policy_seeds"]:
        cached[policy_seed] = {}
        for split in ("development", "validation"):
            dataset = load_physical_target_pairs(
                settings["datasets"][split][policy_seed],
                maximum_horizon=settings["maximum_horizon"],
                state_size=settings["state_size"],
                action_size=settings["action_size"],
            )
            if quick:
                dataset = subsample_physical_target_pairs(
                    dataset, settings[f"maximum_{split}_pairs"]
                )
            dataset = move_physical_target_pairs(dataset, device)
            cached[policy_seed][split] = dataset
            data_records.append(_data_summary(policy_seed, split, dataset))
        _verify_split_separation(
            cached[policy_seed]["development"],
            cached[policy_seed]["validation"],
        )
        development_rewards = cached[policy_seed]["development"].reward_deltas
        magnitude_thresholds[policy_seed] = {
            horizon: float(
                torch.quantile(
                    development_rewards[:, horizon - 1].abs().double(),
                    settings["high_magnitude_quantile"],
                )
            )
            for horizon in settings["candidate_horizons"]
        }

    tasks = [
        (seed, split, horizon, stratum)
        for seed in settings["policy_seeds"]
        for split in ("development", "validation")
        for horizon in settings["candidate_horizons"]
        for stratum in settings["strata"]
    ]
    progress = progress_bar(tasks, description="A5 纯物理目标稳定性", unit="指标")
    records: list[dict[str, Any]] = []
    for policy_seed, split, horizon, stratum in progress:
        dataset = cached[policy_seed][split]
        record = horizon_stability_record(
            dataset,
            policy_seed=policy_seed,
            split=split,
            horizon=horizon,
            stratum=stratum,
            magnitude_threshold=magnitude_thresholds[policy_seed][horizon],
            terminal_horizon=settings["terminal_horizon"],
            local_offsets=settings["local_offsets"],
            thresholds=settings,
            device=device,
        )
        records.append(record)
        update_progress(
            progress,
            device=device,
            metrics={
                "步长": float(horizon),
                "奖功同号": float(record["reward_power_sign_agreement"]),
                "局部稳定": float(record["reward_local_min_sign_agreement"]),
            },
        )

    selection = select_stable_horizon(records, settings=settings, quick=quick)
    metrics_path = output_directory / "horizon_metrics.csv"
    selection_path = output_directory / "selection.json"
    _write_rows(metrics_path, records)
    _write_json(selection_path, selection)
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
            "duration_seconds": time.perf_counter() - started,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "gradient_updates": 0,
        },
        "data": data_records,
        "scan_contract": {
            "candidate_horizons": settings["candidate_horizons"],
            "local_offsets": settings["local_offsets"],
            "terminal_horizon": settings["terminal_horizon"],
            "strata": settings["strata"],
            "high_magnitude_quantile": settings["high_magnitude_quantile"],
            "magnitude_thresholds_from_development": magnitude_thresholds,
        },
        "selection": selection,
        "inputs": {
            "config": _relative(config_path),
            "config_sha256": _file_sha256(config_path),
            "upstream_summary": experiment["upstream_a4"]["summary"],
            "upstream_summary_sha256": experiment["upstream_a4"][
                "summary_sha256"
            ],
            "source_manifest": source_manifest,
        },
        "records": {
            "horizon_metrics": _relative(metrics_path),
            "selection": _relative(selection_path),
            "horizon_metrics_sha256": _file_sha256(metrics_path),
            "selection_sha256": _file_sha256(selection_path),
        },
        "evidence_boundary": {
            "existing_paired_curves_only": True,
            "gradient_updates": 0,
            "model_training": False,
            "new_physical_episodes_generated": False,
            "mechanism_audit_accessed": False,
            "full_rl_trained": False,
            "s4d3_accessed": False,
            "real_slm_actions": False,
        },
        "next_action": (
            "Quick output is software evidence only."
            if quick
            else "Stop for read-only audit; do not train RL."
        ),
    }
    _write_json(output_directory / "summary.json", summary)
    return summary


def preflight_s4_r3_physical_target_stability(
    config_path: Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    metadata = experiment["metadata"]
    required_false = (
        "allow_gradient_updates",
        "allow_model_training",
        "allow_new_episode_generation",
        "allow_mechanism_audit_access",
        "allow_s4d3_access",
        "allow_real_hardware_actions",
    )
    if not bool(metadata["diagnostic_only"]):
        raise RuntimeError("A5 must remain diagnostic-only")
    if any(bool(metadata[name]) for name in required_false):
        raise RuntimeError("A5 safety guard was relaxed")
    if settings["candidate_horizons"] != list(range(8, 33)) and not quick:
        raise RuntimeError("A5 formal candidate horizon set changed")
    if settings["local_offsets"] != [1, 2, 4]:
        raise RuntimeError("A5 local horizon offsets changed")
    if settings["terminal_horizon"] != settings["maximum_horizon"]:
        raise RuntimeError("A5 terminal horizon changed")
    if settings["strata"] != ["all", "high_magnitude"]:
        raise RuntimeError("A5 strata changed")
    if not bool(experiment["runtime"]["require_cuda"]):
        raise RuntimeError("A5 formal diagnostic requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("A5 requires an available CUDA device")

    for key in ("audit_record", "plan"):
        path = _project_path(experiment["design_contract"][key])
        _verify_hash(path, experiment["design_contract"][f"{key}_sha256"])
    for key in (
        "summary",
        "preflight",
        "effective_config",
        "source_manifest",
        "view_metrics",
        "neighborhood_metrics",
        "label_consistency",
    ):
        path = _project_path(experiment["upstream_a4"][key])
        _verify_hash(path, experiment["upstream_a4"][f"{key}_sha256"])
    upstream = json.loads(
        _project_path(experiment["upstream_a4"]["summary"]).read_text(
            encoding="utf-8"
        )
    )
    if upstream["interpretation"]["status"] != experiment["upstream_a4"][
        "required_status"
    ]:
        raise RuntimeError("A5 upstream status changed")
    boundary = upstream["evidence_boundary"]
    if boundary["gradient_updates"] != 0 or boundary["model_training"]:
        raise RuntimeError("A5 upstream evidence boundary changed")

    pair_counts: dict[str, int] = defaultdict(int)
    episode_seeds: dict[str, set[int]] = defaultdict(set)
    for policy_seed in settings["policy_seeds"]:
        for split in ("development", "validation"):
            spec = settings["datasets"][split][policy_seed]
            _verify_hash(_project_path(spec["path"]), spec["sha256"])
            dataset = load_physical_target_pairs(
                spec,
                maximum_horizon=settings["maximum_horizon"],
                state_size=settings["state_size"],
                action_size=settings["action_size"],
            )
            expected = settings[f"expected_{split}_pairs_per_seed"]
            if len(dataset.rows) != expected:
                raise RuntimeError(f"A5 {split} pair count changed")
            pair_counts[split] += len(dataset.rows)
            episode_seeds[split].update(
                int(row["episode_seed"]) for row in dataset.rows
            )
    if episode_seeds["development"] & episode_seeds["validation"]:
        raise RuntimeError("A5 development and validation episode leakage")

    output_directory = _project_path(settings["output_directory"])
    if output_directory.exists():
        raise FileExistsError(f"A5 output already exists: {output_directory}")
    command = (
        ".\\.venv\\Scripts\\python.exe "
        "scripts\\diagnose_s4_r3_physical_target_stability.py --config "
        f"{str(_relative(config_path)).replace('/', chr(92))}"
    )
    return {
        "status": "READY_FOR_USER_DIAGNOSTIC",
        "quick": quick,
        "upstream_status": upstream["interpretation"]["status"],
        "policy_seeds": settings["policy_seeds"],
        "candidate_horizons": settings["candidate_horizons"],
        "local_offsets": settings["local_offsets"],
        "planned_metric_rows": (
            len(settings["policy_seeds"])
            * 2
            * len(settings["candidate_horizons"])
            * len(settings["strata"])
        ),
        "development_pairs": pair_counts["development"],
        "validation_pairs": pair_counts["validation"],
        "gradient_updates": 0,
        "model_training": False,
        "new_episode_generation": False,
        "mechanism_audit_access": False,
        "cuda_required": True,
        "output_directory": settings["output_directory"],
        "s4d3_access": False,
        "real_slm_actions": False,
        "formal_command": command,
    }


def load_physical_target_pairs(
    spec: dict[str, Any],
    *,
    maximum_horizon: int,
    state_size: int,
    action_size: int,
) -> PhysicalTargetPairDataset:
    payload = torch.load(
        _project_path(spec["path"]), map_location="cpu", weights_only=False
    )
    states = payload["states"].float()
    actions = payload["actions"].float()
    if states.shape[1] != state_size or actions.shape[1] != action_size:
        raise ValueError("A5 state or action size mismatch")
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

    rewards: list[torch.Tensor] = []
    powers: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda value: tuple(map(str, value))):
        indices = groups[key]
        if set(indices) != REQUIRED_CANDIDATES:
            raise ValueError("A5 incomplete action pair")
        zero, actor = indices["zero"], indices["actor"]
        if not torch.equal(states[zero], states[actor]):
            raise ValueError("A5 candidates do not share the same state")
        if float(actions[zero].abs().max()) > 1e-8:
            raise ValueError("A5 zero action is not zero")
        rewards.append(
            payload["empirical_reward_returns"][actor, :maximum_horizon]
            - payload["empirical_reward_returns"][zero, :maximum_horizon]
        )
        powers.append(
            payload["empirical_power_returns"][actor, :maximum_horizon]
            - payload["empirical_power_returns"][zero, :maximum_horizon]
        )
        targets.append(
            payload["n_step_targets"][actor, :maximum_horizon]
            - payload["n_step_targets"][zero, :maximum_horizon]
        )
        rows.append(
            {
                "profile_id": key[0],
                "condition_id": key[1],
                "probe_step": key[2],
                "episode_index": key[3],
                "episode_seed": key[4],
            }
        )
    dataset = PhysicalTargetPairDataset(
        reward_deltas=torch.stack(rewards).double(),
        power_deltas=torch.stack(powers).double(),
        target_deltas=torch.stack(targets).double(),
        rows=rows,
    )
    dataset.validate(maximum_horizon=maximum_horizon)
    return dataset


def subsample_physical_target_pairs(
    dataset: PhysicalTargetPairDataset, maximum_pairs: int
) -> PhysicalTargetPairDataset:
    if maximum_pairs >= len(dataset.rows):
        return dataset
    indices = torch.linspace(0, len(dataset.rows) - 1, maximum_pairs).round().long()
    return PhysicalTargetPairDataset(
        reward_deltas=dataset.reward_deltas[indices],
        power_deltas=dataset.power_deltas[indices],
        target_deltas=dataset.target_deltas[indices],
        rows=[dataset.rows[int(index)] for index in indices],
    )


def move_physical_target_pairs(
    dataset: PhysicalTargetPairDataset, device: torch.device
) -> PhysicalTargetPairDataset:
    return PhysicalTargetPairDataset(
        reward_deltas=dataset.reward_deltas.to(device),
        power_deltas=dataset.power_deltas.to(device),
        target_deltas=dataset.target_deltas.to(device),
        rows=dataset.rows,
    )


def horizon_stability_record(
    dataset: PhysicalTargetPairDataset,
    *,
    policy_seed: int,
    split: str,
    horizon: int,
    stratum: str,
    magnitude_threshold: float,
    terminal_horizon: int,
    local_offsets: list[int],
    thresholds: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    rewards = dataset.reward_deltas.to(device)
    powers = dataset.power_deltas.to(device)
    targets = dataset.target_deltas.to(device)
    reward = rewards[:, horizon - 1]
    power = powers[:, horizon - 1]
    target = targets[:, horizon - 1]
    if stratum == "all":
        mask = torch.ones(reward.shape[0], dtype=torch.bool, device=device)
    elif stratum == "high_magnitude":
        mask = reward.abs() >= magnitude_threshold
    else:
        raise ValueError(f"unknown A5 stratum: {stratum}")
    if int(mask.sum()) < 2:
        raise RuntimeError("A5 stratum contains too few samples")

    reward_power = _comparison_metrics(reward, power, mask)
    reward_terminal = _comparison_metrics(
        reward, rewards[:, terminal_horizon - 1], mask
    )
    power_terminal = _comparison_metrics(
        power, powers[:, terminal_horizon - 1], mask
    )
    target_reward = _comparison_metrics(target, reward, mask)
    local_reward = {
        offset: _comparison_metrics(reward, rewards[:, horizon - offset - 1], mask)
        for offset in local_offsets
    }
    local_power = {
        offset: _comparison_metrics(power, powers[:, horizon - offset - 1], mask)
        for offset in local_offsets
    }
    record: dict[str, Any] = {
        "policy_seed": policy_seed,
        "split": split,
        "horizon": horizon,
        "stratum": stratum,
        "samples": int(mask.sum()),
        "magnitude_threshold_from_development": magnitude_threshold,
        "reward_actor_better_fraction": float((reward[mask] > 0).float().mean()),
        "power_actor_better_fraction": float((power[mask] > 0).float().mean()),
        "reward_power_sign_agreement": reward_power["sign_agreement"],
        "reward_power_balanced_accuracy": reward_power["balanced_accuracy"],
        "reward_power_matthews_correlation": reward_power[
            "matthews_correlation"
        ],
        "reward_terminal_sign_agreement": reward_terminal["sign_agreement"],
        "reward_terminal_balanced_accuracy": reward_terminal[
            "balanced_accuracy"
        ],
        "power_terminal_sign_agreement": power_terminal["sign_agreement"],
        "power_terminal_balanced_accuracy": power_terminal[
            "balanced_accuracy"
        ],
        "target_reward_sign_agreement": target_reward["sign_agreement"],
        "target_reward_balanced_accuracy": target_reward["balanced_accuracy"],
    }
    for offset in local_offsets:
        record[f"reward_local_offset_{offset}_sign_agreement"] = local_reward[
            offset
        ]["sign_agreement"]
        record[f"reward_local_offset_{offset}_balanced_accuracy"] = local_reward[
            offset
        ]["balanced_accuracy"]
        record[f"power_local_offset_{offset}_sign_agreement"] = local_power[
            offset
        ]["sign_agreement"]
        record[f"power_local_offset_{offset}_balanced_accuracy"] = local_power[
            offset
        ]["balanced_accuracy"]
    record["reward_local_min_sign_agreement"] = min(
        value["sign_agreement"] for value in local_reward.values()
    )
    record["reward_local_min_balanced_accuracy"] = _optional_min(
        value["balanced_accuracy"] for value in local_reward.values()
    )
    record["power_local_min_sign_agreement"] = min(
        value["sign_agreement"] for value in local_power.values()
    )
    record["power_local_min_balanced_accuracy"] = _optional_min(
        value["balanced_accuracy"] for value in local_power.values()
    )
    checks = stability_gate_checks(record, thresholds=thresholds)
    record["gate"] = "PASS" if all(checks.values()) else "FAIL"
    record["gate_checks"] = json.dumps(checks, ensure_ascii=False, sort_keys=True)
    return record


def stability_gate_checks(
    record: dict[str, Any], *, thresholds: dict[str, Any]
) -> dict[str, bool]:
    minimum_sign = thresholds["minimum_sign_agreement"]
    minimum_balanced = thresholds["minimum_balanced_accuracy"]
    lower = thresholds["minimum_actor_better_fraction"]
    upper = thresholds["maximum_actor_better_fraction"]
    balanced_fields = (
        "reward_power_balanced_accuracy",
        "reward_terminal_balanced_accuracy",
        "power_terminal_balanced_accuracy",
        "reward_local_min_balanced_accuracy",
        "power_local_min_balanced_accuracy",
    )
    return {
        "reward_power_sign": record["reward_power_sign_agreement"]
        >= minimum_sign,
        "reward_power_balanced": _at_least_all(
            [record["reward_power_balanced_accuracy"]], minimum_balanced
        ),
        "terminal_sign": min(
            record["reward_terminal_sign_agreement"],
            record["power_terminal_sign_agreement"],
        )
        >= minimum_sign,
        "terminal_balanced": _at_least_all(
            [
                record["reward_terminal_balanced_accuracy"],
                record["power_terminal_balanced_accuracy"],
            ],
            minimum_balanced,
        ),
        "local_sign": min(
            record["reward_local_min_sign_agreement"],
            record["power_local_min_sign_agreement"],
        )
        >= minimum_sign,
        "local_balanced": _at_least_all(
            [record[name] for name in balanced_fields[-2:]], minimum_balanced
        ),
        "class_balance": lower
        <= record["reward_actor_better_fraction"]
        <= upper,
    }


def select_stable_horizon(
    records: list[dict[str, Any]],
    *,
    settings: dict[str, Any],
    quick: bool,
) -> dict[str, Any]:
    if quick:
        return {
            "status": "QUICK_SMOKE_ONLY",
            "development_candidates": [],
            "selected_horizon": None,
            "validation_gate": "NOT_EVALUATED",
            "full_rl_authorized": False,
            "s4d3_authorized": False,
        }
    required_per_horizon = len(settings["policy_seeds"]) * len(settings["strata"])
    development_candidates: list[int] = []
    for horizon in settings["candidate_horizons"]:
        selected = [
            row
            for row in records
            if row["split"] == "development" and row["horizon"] == horizon
        ]
        if len(selected) == required_per_horizon and all(
            row["gate"] == "PASS" for row in selected
        ):
            development_candidates.append(horizon)
    selected_horizon = (
        min(development_candidates) if development_candidates else None
    )
    if selected_horizon is None:
        status = "NO_STABLE_HORIZON_IN_CURRENT_DATA"
        validation_gate = "NOT_EVALUATED"
    else:
        validation = [
            row
            for row in records
            if row["split"] == "validation"
            and row["horizon"] == selected_horizon
        ]
        passed = len(validation) == required_per_horizon and all(
            row["gate"] == "PASS" for row in validation
        )
        validation_gate = "PASS" if passed else "FAIL"
        status = (
            "PHYSICAL_TARGET_HORIZON_STABLE"
            if passed
            else "DEVELOPMENT_SELECTED_HORIZON_NOT_VALIDATED"
        )
    return {
        "status": status,
        "development_candidates": development_candidates,
        "selected_horizon": selected_horizon,
        "validation_gate": validation_gate,
        "selection_rule": "shortest_common_development_candidate_then_fixed_validation",
        "full_rl_authorized": False,
        "s4d3_authorized": False,
        "real_hardware_authorized": False,
    }


def _comparison_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float | None]:
    prediction = prediction[mask]
    target = target[mask]
    sign_agreement = float(((prediction > 0) == (target > 0)).float().mean())
    try:
        metrics = classification_metrics(prediction, target)
        balanced_accuracy: float | None = metrics["balanced_accuracy"]
        mcc: float | None = metrics["matthews_correlation"]
    except ValueError:
        balanced_accuracy = None
        mcc = None
    return {
        "sign_agreement": sign_agreement,
        "balanced_accuracy": balanced_accuracy,
        "matthews_correlation": mcc,
    }


def _optional_min(values: Any) -> float | None:
    selected = [float(value) for value in values if value is not None]
    return min(selected) if selected else None


def _at_least_all(values: list[float | None], threshold: float) -> bool:
    return all(value is not None and value >= threshold for value in values)


def _effective_settings(
    experiment: dict[str, Any], *, quick: bool
) -> dict[str, Any]:
    data = experiment["data"]
    scan = experiment["scan"]
    thresholds = experiment["diagnostic_thresholds"]
    settings: dict[str, Any] = {
        "output_directory": experiment["outputs"]["directory"],
        "policy_seeds": [int(value) for value in data["policy_seeds"]],
        "datasets": {
            split: {int(seed): spec for seed, spec in data[split].items()}
            for split in ("development", "validation")
        },
        "maximum_horizon": int(data["maximum_horizon"]),
        "state_size": int(data["state_size"]),
        "action_size": int(data["action_size"]),
        "expected_development_pairs_per_seed": int(
            data["expected_development_pairs_per_seed"]
        ),
        "expected_validation_pairs_per_seed": int(
            data["expected_validation_pairs_per_seed"]
        ),
        "candidate_horizons": [int(value) for value in scan["candidate_horizons"]],
        "local_offsets": [int(value) for value in scan["local_offsets"]],
        "terminal_horizon": int(scan["terminal_horizon"]),
        "strata": [str(value) for value in scan["strata"]],
        "high_magnitude_quantile": float(scan["high_magnitude_quantile"]),
        "minimum_sign_agreement": float(
            thresholds["minimum_sign_agreement"]
        ),
        "minimum_balanced_accuracy": float(
            thresholds["minimum_balanced_accuracy"]
        ),
        "minimum_actor_better_fraction": float(
            thresholds["minimum_actor_better_fraction"]
        ),
        "maximum_actor_better_fraction": float(
            thresholds["maximum_actor_better_fraction"]
        ),
    }
    if quick:
        quick_config = experiment["quick"]
        settings.update(
            {
                "output_directory": experiment["outputs"]["quick_directory"],
                "policy_seeds": [int(value) for value in quick_config["policy_seeds"]],
                "candidate_horizons": [
                    int(value) for value in quick_config["candidate_horizons"]
                ],
                "maximum_development_pairs": int(
                    quick_config["maximum_development_pairs"]
                ),
                "maximum_validation_pairs": int(
                    quick_config["maximum_validation_pairs"]
                ),
            }
        )
    return settings


def _verify_split_separation(
    development: PhysicalTargetPairDataset,
    validation: PhysicalTargetPairDataset,
) -> None:
    development_seeds = {int(row["episode_seed"]) for row in development.rows}
    validation_seeds = {int(row["episode_seed"]) for row in validation.rows}
    if development_seeds & validation_seeds:
        raise RuntimeError("A5 episode leakage")


def _data_summary(
    policy_seed: int,
    split: str,
    dataset: PhysicalTargetPairDataset,
) -> dict[str, Any]:
    return {
        "policy_seed": policy_seed,
        "split": split,
        "pairs": len(dataset.rows),
        "episode_seeds": len({int(row["episode_seed"]) for row in dataset.rows}),
        "reward_actor_better_fraction_h32": float(
            (dataset.reward_deltas[:, -1] > 0).float().mean()
        ),
        "power_actor_better_fraction_h32": float(
            (dataset.power_deltas[:, -1] > 0).float().mean()
        ),
    }


def _source_manifest(
    config_path: Path, experiment: dict[str, Any]
) -> dict[str, str]:
    paths = [
        config_path,
        _project_path("scripts/diagnose_s4_r3_physical_target_stability.py"),
        _project_path("src/rl/s4_r3_physical_target_stability.py"),
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
        raise RuntimeError(f"A5 hash mismatch: {path}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("A5 output rows must not be empty")
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
