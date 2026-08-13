"""S4-D2-R2策略前10维与新增高阶维的只读配对消融。"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import torch

from src.rl.residual_sac import SacConfig, SquashedGaussianActor
from src.rl.s4_r2_diagnostic import (
    ACTION_METRICS,
    SCIENCE_METRICS,
    _collect_pair,
    _episode_rows,
    _group_variant_summaries,
    _load_actor,
    _policy_variant_summaries,
    _rollout_anchored_baseline,
    _rollout_variant,
    _scenario_record,
    _write_csv,
)
from src.rl.s4_training import (
    _distribution,
    _file_sha256,
    _git_record,
    _load_yaml,
    _profiles,
    _project_path,
    _relative,
    _runtime_record,
    _source_manifest,
    _write_json,
)
from src.runtime import resolve_device
from src.simulation.config import S1EnvConfig, load_s1_config
from src.simulation.hardware_effects import HardwareProfile
from src.simulation.robust_control import RobustnessCondition
from src.training_progress import advance_to, counted_progress, update_progress


class ModalMaskActor:
    """在不修改原检查点的前提下遮住指定动作子空间。"""

    def __init__(
        self,
        actor: SquashedGaussianActor,
        *,
        mask: str,
        num_modes: int,
        anchor_modes: int,
    ) -> None:
        if not 0 < anchor_modes < num_modes:
            raise ValueError("modal ablation requires anchor_modes < num_modes")
        if mask not in {"all", "anchor_only", "added_only"}:
            raise ValueError(f"unknown modal mask: {mask}")
        self.actor = actor
        self.mask = mask
        self.num_modes = num_modes
        self.anchor_modes = anchor_modes

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        action = self.actor.deterministic(state)
        expected = (state.shape[0], self.num_modes)
        if action.shape != expected:
            raise RuntimeError(f"masked actor expected action shape {expected}")
        if self.mask == "all":
            return action
        masked = action.clone()
        if self.mask == "anchor_only":
            masked[:, self.anchor_modes :] = 0
        else:
            masked[:, : self.anchor_modes] = 0
        return masked


def run_s4_r2_modal_ablation(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """运行25%动作的模态分解；不训练、不修改模型。"""
    experiment_path = _project_path(config_path)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight = preflight_s4_r2_modal_ablation(
        experiment_path,
        experiment,
        settings,
        quick=quick,
    )
    if preflight_only:
        return preflight

    device = resolve_device("cuda")
    output_directory = _project_path(settings["output_directory"])
    if output_directory.exists():
        raise FileExistsError(
            "modal-ablation output already exists; preserve it for audit: "
            f"{output_directory}"
        )
    output_directory.mkdir(parents=True)
    _write_json(output_directory / "preflight.json", preflight)
    _write_json(output_directory / "effective_config.json", settings)
    _write_json(
        output_directory / "source_manifest.json",
        _source_manifest(experiment["tracked_source_files"]),
    )

    base_config, _ = load_s1_config(_project_path(experiment["environment_config"]))
    profiles = _profiles(experiment, settings["profile_ids"])
    conditions = [
        RobustnessCondition.from_mapping(item)
        for item in settings["physical_conditions"]
    ]
    anchor_modes = int(experiment["frozen_controller"]["active_anchor_modes"])
    scenario_count = len(profiles) * len(conditions)
    checkpoint_counts = {
        str(rep["id"]): sum(
            str(checkpoint["representation_id"]) == str(rep["id"])
            for checkpoint in settings["checkpoints"]
        )
        for rep in settings["representations"]
    }
    total_rollouts = sum(
        scenario_count
        * (2 + checkpoint_counts[str(rep["id"])] * len(settings["variants"]))
        for rep in settings["representations"]
    )
    progress = counted_progress(
        total=total_rollouts,
        description="S4-D2-R2模态分解",
        unit="轨迹组",
    )
    completed = 0
    started = time.perf_counter()
    candidates: dict[
        tuple[str, str, int | None], dict[str, list[torch.Tensor]]
    ] = defaultdict(lambda: defaultdict(list))
    baselines: dict[
        tuple[str, str, int | None], dict[str, list[torch.Tensor]]
    ] = defaultdict(lambda: defaultdict(list))
    scenario_records: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    zero_max_abs_by_representation: dict[str, float] = {}
    zero_variant = {"id": "zero_residual", "scale": 0.0, "mask": "zero"}

    for representation in settings["representations"]:
        representation_id = str(representation["id"])
        num_modes = int(representation["num_modes"])
        representation_experiment = deepcopy(experiment)
        representation_experiment["active_representation"] = deepcopy(representation)
        representation_experiment["action"]["num_modes"] = num_modes
        representation_base = replace(
            base_config,
            num_modes=num_modes,
            batch_size=int(settings["batch_size"]),
            episode_length=max(base_config.episode_length, int(settings["steps"])),
        )
        configured_scenarios: list[
            tuple[HardwareProfile, RobustnessCondition, S1EnvConfig]
        ] = []
        for profile in profiles:
            for condition in conditions:
                configured_scenarios.append(
                    (
                        profile,
                        condition,
                        profile.environment_config(
                            condition.environment_config(representation_base)
                        ),
                    )
                )

        baseline_cache: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
        zero_max_abs = 0.0
        for profile, condition, config in configured_scenarios:
            scenario_key = (profile.identifier, condition.identifier)
            baseline = _rollout_anchored_baseline(
                representation_experiment,
                config,
                condition,
                profile,
                int(settings["steps"]),
                device,
            )
            baseline_cache[scenario_key] = baseline
            completed += 1
            advance_to(progress, completed)
            update_progress(
                progress,
                device=device,
                metrics={
                    "表示维数": float(num_modes),
                    "基线功率": float(baseline["power_in_bucket"].mean()),
                },
            )

            zero = _rollout_variant(
                actor=None,
                scale=0.0,
                experiment=representation_experiment,
                config=config,
                condition=condition,
                profile=profile,
                steps=int(settings["steps"]),
                device=device,
            )
            zero_diff = max(
                float((zero[metric] - baseline[metric]).abs().max())
                for metric in SCIENCE_METRICS
            )
            zero_max_abs = max(zero_max_abs, zero_diff)
            key = (representation_id, str(zero_variant["id"]), None)
            _collect_pair(candidates, baselines, key, zero, baseline)
            scenario_records.append(
                _scenario_record(
                    representation_id=representation_id,
                    num_modes=num_modes,
                    policy_seed=None,
                    variant=zero_variant,
                    profile=profile,
                    condition=condition,
                    candidate=zero,
                    baseline=baseline,
                )
            )
            episode_records.extend(
                _episode_rows(
                    representation_id=representation_id,
                    num_modes=num_modes,
                    policy_seed=None,
                    variant=zero_variant,
                    profile=profile,
                    condition=condition,
                    candidate=zero,
                    baseline=baseline,
                )
            )
            completed += 1
            advance_to(progress, completed)
            update_progress(
                progress,
                device=device,
                metrics={"零残差误差": zero_max_abs},
            )
        zero_max_abs_by_representation[representation_id] = zero_max_abs

        representation_checkpoints = [
            item
            for item in settings["checkpoints"]
            if str(item["representation_id"]) == representation_id
        ]
        for checkpoint in representation_checkpoints:
            policy_seed = int(checkpoint["policy_seed"])
            actor = _load_actor(checkpoint, device)
            for variant in settings["variants"]:
                masked_actor = ModalMaskActor(
                    actor,
                    mask=str(variant["mask"]),
                    num_modes=num_modes,
                    anchor_modes=anchor_modes,
                )
                key = (representation_id, str(variant["id"]), policy_seed)
                for profile, condition, config in configured_scenarios:
                    baseline = baseline_cache[
                        (profile.identifier, condition.identifier)
                    ]
                    candidate = _rollout_variant(
                        actor=masked_actor,  # type: ignore[arg-type]
                        scale=float(variant["scale"]),
                        experiment=representation_experiment,
                        config=config,
                        condition=condition,
                        profile=profile,
                        steps=int(settings["steps"]),
                        device=device,
                    )
                    _collect_pair(
                        candidates,
                        baselines,
                        key,
                        candidate,
                        baseline,
                    )
                    scenario_records.append(
                        _scenario_record(
                            representation_id=representation_id,
                            num_modes=num_modes,
                            policy_seed=policy_seed,
                            variant=variant,
                            profile=profile,
                            condition=condition,
                            candidate=candidate,
                            baseline=baseline,
                        )
                    )
                    episode_records.extend(
                        _episode_rows(
                            representation_id=representation_id,
                            num_modes=num_modes,
                            policy_seed=policy_seed,
                            variant=variant,
                            profile=profile,
                            condition=condition,
                            candidate=candidate,
                            baseline=baseline,
                        )
                    )
                    completed += 1
                    advance_to(progress, completed)
                    update_progress(
                        progress,
                        device=device,
                        metrics={
                            "表示维数": float(num_modes),
                            "策略种子": float(policy_seed),
                            "掩码编号": float(
                                ["all", "anchor_only", "added_only"].index(
                                    str(variant["mask"])
                                )
                            ),
                            "功率差": float(
                                (
                                    candidate["power_in_bucket"]
                                    - baseline["power_in_bucket"]
                                ).mean()
                            ),
                        },
                    )
            del actor
    progress.close()

    summary_variants = [zero_variant, *deepcopy(settings["variants"])]
    policy_variant_summaries = _policy_variant_summaries(
        candidates,
        baselines,
        summary_variants,
    )
    grouped_variant_summaries = _group_variant_summaries(
        candidates,
        baselines,
        summary_variants,
    )
    block_comparisons = _block_comparisons(candidates)
    upstream_reproduction = _upstream_quarter_reproduction(
        experiment,
        grouped_variant_summaries,
        quick=quick,
        tolerance=float(settings["upstream_quarter_reproduction_tolerance"]),
    )
    interpretation = interpret_modal_ablation(
        grouped_variant_summaries,
        zero_max_abs_by_representation=zero_max_abs_by_representation,
        zero_tolerance=float(settings["zero_equivalence_tolerance"]),
        violation_limit=float(settings["violation_limit"]),
    )
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run-diagnostic",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "s4d2_r2_modal_ablation_v1",
        },
        "experiment": {
            "id": "AO-S4-D2-R2-MODAL-ABLATION",
            "type": "software_only_cuda_paired_post_hoc_ablation",
            "status": "completed_pending_audit",
            "quick": quick,
            "duration_seconds": time.perf_counter() - started,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "evidence_boundary": {
            "training_performed": False,
            "optimizer_updates": 0,
            "checkpoints_modified": False,
            "upstream_trajectory_files_reused": False,
            "upstream_diagnostic_seeds_reused_for_pairing": not quick,
            "s4d3_accessed": False,
            "real_slm_actions": False,
            "post_hoc_diagnostic_only": True,
        },
        "inputs": {
            "experiment_config": _relative(experiment_path),
            "experiment_config_sha256": _file_sha256(experiment_path),
            "upstream_diagnostic_summary_sha256": preflight[
                "upstream_diagnostic_summary_sha256"
            ],
            "checkpoint_count": len(settings["checkpoints"]),
        },
        "design": {
            "paired_episode_seeds": True,
            "scale": 0.25,
            "anchor_modes": anchor_modes,
            "representations": deepcopy(settings["representations"]),
            "profiles": list(settings["profile_ids"]),
            "physical_conditions": deepcopy(settings["physical_conditions"]),
            "variants": deepcopy(settings["variants"]),
            "non_additivity_note": (
                "Masked variants generate different closed-loop states; anchor-only and "
                "added-only effects must not be algebraically summed."
            ),
        },
        "zero_residual_equivalence": {
            "max_absolute_science_metric_difference_by_representation": (
                zero_max_abs_by_representation
            ),
            "tolerance": float(settings["zero_equivalence_tolerance"]),
            "status": (
                "PASS"
                if all(
                    value <= float(settings["zero_equivalence_tolerance"])
                    for value in zero_max_abs_by_representation.values()
                )
                else "FAIL"
            ),
        },
        "upstream_all_modes_quarter_reproduction": upstream_reproduction,
        "policy_variant_summaries": policy_variant_summaries,
        "grouped_variant_summaries": grouped_variant_summaries,
        "paired_block_comparisons": block_comparisons,
        "interpretation": interpretation,
        "record_counts": {
            "scenario_records": len(scenario_records),
            "episode_records": len(episode_records),
        },
        "runtime": _runtime_record(),
        "git": _git_record(),
        "next_action": (
            "Stop and ask the assistant to audit the modal ablation. Do not retrain, "
            "change algorithms, open S4-D3, or operate real SLM hardware."
        ),
    }
    _write_json(output_directory / "summary.json", summary)
    _write_csv(output_directory / "scenario_records.csv", scenario_records)
    _write_csv(output_directory / "episode_records.csv", episode_records)
    return summary


def preflight_s4_r2_modal_ablation(
    experiment_path: Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    """核对上游正式诊断、掩码契约与只读执行边界。"""
    metadata = experiment["metadata"]
    if str(metadata["stage"]) != "S4-D2-R2-MODAL-ABLATION":
        raise ValueError("metadata must identify S4-D2-R2-MODAL-ABLATION")
    if not bool(metadata.get("reuse_upstream_diagnostic_seeds_for_paired_ablation")):
        raise RuntimeError("formal modal ablation requires paired upstream diagnostic seeds")
    forbidden_flags = (
        "allow_training",
        "allow_checkpoint_updates",
        "allow_prior_trajectory_file_reuse",
        "allow_s4d3_access",
        "allow_real_hardware_actions",
        "allow_algorithm_change",
        "allow_reward_change",
        "allow_gate_relaxation",
    )
    if any(bool(metadata.get(field, True)) for field in forbidden_flags):
        raise RuntimeError("all modal-ablation mutation and scope flags must be false")

    upstream = experiment["upstream_diagnostic"]
    upstream_fields = (
        "summary",
        "episode_records",
        "scenario_records",
        "preflight",
        "effective_config",
        "source_manifest",
        "experiment_config",
        "audit_record",
    )
    upstream_paths = {
        field: _project_path(upstream[field]) for field in upstream_fields
    }
    for field, path in upstream_paths.items():
        if _file_sha256(path) != str(upstream[f"{field}_sha256"]):
            raise RuntimeError(f"upstream diagnostic evidence hash mismatch: {field}")
    audit_text = upstream_paths["audit_record"].read_text(encoding="utf-8")
    if (
        "Verification Status: `ANALYZED`" not in audit_text
        or "25%动作模态分解诊断：`AUTHORIZED`" not in audit_text
    ):
        raise RuntimeError("upstream diagnostic audit does not authorize modal ablation")

    upstream_summary = json.loads(
        upstream_paths["summary"].read_text(encoding="utf-8")
    )
    if (
        bool(upstream_summary["experiment"]["quick"])
        or upstream_summary["experiment"]["status"] != "completed_pending_audit"
        or bool(upstream_summary["evidence_boundary"]["training_performed"])
        or bool(upstream_summary["evidence_boundary"]["s4d3_accessed"])
        or bool(upstream_summary["diagnosis"]["s4d3_authorized"])
    ):
        raise RuntimeError("upstream formal diagnostic state or S4-D3 lock changed")
    if upstream_summary["zero_residual_equivalence"]["status"] != "PASS":
        raise RuntimeError("upstream zero-residual wiring check did not pass")
    diagnoses = upstream_summary["diagnosis"]["by_representation"]
    if set(diagnoses) != {"zernike_21", "zernike_36"}:
        raise RuntimeError("upstream diagnostic representation set changed")
    for representation_id, diagnosis in diagnoses.items():
        amplitude = diagnosis["amplitude_check"]
        if float(amplitude["relative_power_gain_by_scale"]["0.25"]) >= 0:
            raise RuntimeError(
                f"upstream quarter-scale failure changed: {representation_id}"
            )

    recorded_source = json.loads(
        upstream_paths["source_manifest"].read_text(encoding="utf-8")
    )
    for relative, digest in recorded_source.items():
        path = _project_path(relative)
        if not path.is_file() or _file_sha256(path) != digest:
            raise RuntimeError(f"upstream diagnostic source changed: {relative}")

    with upstream_paths["episode_records"].open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        if sum(1 for _ in csv.DictReader(handle)) != 7488:
            raise RuntimeError("upstream diagnostic episode row count changed")
    with upstream_paths["scenario_records"].open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        if sum(1 for _ in csv.DictReader(handle)) != 468:
            raise RuntimeError("upstream diagnostic scenario row count changed")

    upstream_config = _load_yaml(upstream_paths["experiment_config"])
    if upstream_summary["inputs"]["diagnostic_config_sha256"] != _file_sha256(
        upstream_paths["experiment_config"]
    ):
        raise RuntimeError("upstream summary and diagnostic configuration disagree")
    for field in (
        "environment_config",
        "environment_config_sha256",
        "hardware_profile_source",
        "hardware_profile_source_sha256",
        "representations",
        "frozen_controller",
        "policy_observation",
        "action",
        "reward",
        "checkpoints",
        "diagnostic_profile_ids",
        "diagnostic_physical_conditions",
    ):
        if experiment[field] != upstream_config[field]:
            raise RuntimeError(f"modal ablation changed frozen upstream field: {field}")
    upstream_effective = json.loads(
        upstream_paths["effective_config"].read_text(encoding="utf-8")
    )
    formal_effective_checks = {
        "representations": experiment["representations"],
        "checkpoints": experiment["checkpoints"],
        "profile_ids": experiment["diagnostic_profile_ids"],
        "physical_conditions": experiment["diagnostic_physical_conditions"],
        "batch_size": int(experiment["evaluation"]["episodes_per_physical_condition"]),
        "steps": int(experiment["evaluation"]["steps"]),
    }
    for field, expected in formal_effective_checks.items():
        if upstream_effective[field] != expected:
            raise RuntimeError(f"formal paired field differs from upstream: {field}")

    variants = list(experiment["variants"])
    expected_variants = {
        ("all_modes_quarter", "all"),
        ("anchor_only_quarter", "anchor_only"),
        ("added_modes_only_quarter", "added_only"),
    }
    if {
        (str(item["id"]), str(item["mask"])) for item in variants
    } != expected_variants or any(float(item["scale"]) != 0.25 for item in variants):
        raise RuntimeError("modal ablation must use the three declared masks at 25% scale")
    active_variant_ids = [str(item["id"]) for item in settings["variants"]]
    if len(active_variant_ids) != len(set(active_variant_ids)):
        raise ValueError("active modal-ablation variants must be unique")

    checkpoint_contracts: list[dict[str, Any]] = []
    for checkpoint in settings["checkpoints"]:
        path = _project_path(checkpoint["path"])
        if _file_sha256(path) != str(checkpoint["sha256"]):
            raise RuntimeError("R2 checkpoint hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("algorithm") != "residual_sac":
            raise RuntimeError("modal ablation checkpoint is not residual SAC")
        actor_config = SacConfig(**payload["config"])
        num_modes = int(checkpoint["num_modes"])
        if (
            actor_config.state_size != 10 * num_modes
            or actor_config.action_size != num_modes
            or not all(torch.isfinite(value).all() for value in payload["actor"].values())
        ):
            raise RuntimeError("checkpoint dimensions or finite-value contract changed")
        checkpoint_contracts.append(
            {
                "representation_id": str(checkpoint["representation_id"]),
                "policy_seed": int(checkpoint["policy_seed"]),
                "state_size": actor_config.state_size,
                "action_size": actor_config.action_size,
            }
        )
    active_keys = {
        (str(item["representation_id"]), int(item["policy_seed"]))
        for item in settings["checkpoints"]
    }
    if not quick:
        expected_keys = {
            (representation_id, seed)
            for representation_id in ("zernike_21", "zernike_36")
            for seed in (7101, 7102, 7103)
        }
        if active_keys != expected_keys:
            raise RuntimeError("formal modal ablation must use all six R2 best actors")

    environment_path = _project_path(experiment["environment_config"])
    hardware_path = _project_path(experiment["hardware_profile_source"])
    if _file_sha256(environment_path) != str(experiment["environment_config_sha256"]):
        raise RuntimeError("environment configuration hash mismatch")
    if _file_sha256(hardware_path) != str(experiment["hardware_profile_source_sha256"]):
        raise RuntimeError("hardware profile source hash mismatch")
    if not all(_project_path(path).is_file() for path in experiment["tracked_source_files"]):
        raise FileNotFoundError("one or more modal-ablation tracked source files are missing")
    output_directory = _project_path(settings["output_directory"])
    if output_directory.exists():
        raise FileExistsError(f"modal-ablation output already exists: {output_directory}")

    scenario_count = len(settings["profile_ids"]) * len(settings["physical_conditions"])
    rollout_count = sum(
        scenario_count
        * (
            2
            + sum(
                str(checkpoint["representation_id"]) == str(rep["id"])
                for checkpoint in settings["checkpoints"]
            )
            * len(settings["variants"])
        )
        for rep in settings["representations"]
    )
    return {
        "status": "READY_FOR_QUICK_DIAGNOSTIC" if quick else "READY_FOR_USER_DIAGNOSTIC",
        "quick": quick,
        "experiment_config": _relative(experiment_path),
        "experiment_config_sha256": _file_sha256(experiment_path),
        "output_directory": _relative(output_directory),
        "upstream_diagnostic_status": "ANALYZED",
        "upstream_diagnostic_summary_sha256": _file_sha256(
            upstream_paths["summary"]
        ),
        "representations": [str(item["id"]) for item in settings["representations"]],
        "checkpoint_contracts": checkpoint_contracts,
        "variant_ids": active_variant_ids,
        "scenario_count_per_representation": scenario_count,
        "rollout_count": rollout_count,
        "episode_count_per_rollout": int(settings["batch_size"]),
        "steps_per_episode": int(settings["steps"]),
        "formal_seed_pairing_with_upstream": not quick,
        "cuda_required": True,
        "training_allowed": False,
        "optimizer_updates": 0,
        "checkpoints_are_read_only": True,
        "upstream_trajectory_files_reused": False,
        "s4d3_accessed": False,
        "real_slm_actions": False,
        "formal_command": (
            ".\\.venv\\Scripts\\python.exe scripts\\diagnose_s4_r2_modal_ablation.py "
            "--config configs\\experiments\\s4_residual_sac_r2_modal_ablation_v1.yaml"
        ),
    }


def _block_comparisons(
    candidates: dict[
        tuple[str, str, int | None], dict[str, list[torch.Tensor]]
    ],
) -> list[dict[str, Any]]:
    comparisons = (
        ("anchor_only_minus_all", "anchor_only_quarter", "all_modes_quarter"),
        ("added_only_minus_all", "added_modes_only_quarter", "all_modes_quarter"),
        (
            "added_only_minus_anchor_only",
            "added_modes_only_quarter",
            "anchor_only_quarter",
        ),
    )
    records: list[dict[str, Any]] = []
    representation_ids = sorted({key[0] for key in candidates})
    for representation_id in representation_ids:
        seeds = sorted(
            {
                int(key[2])
                for key in candidates
                if key[0] == representation_id and key[2] is not None
            }
        )
        for comparison_id, left_variant, right_variant in comparisons:
            parts: dict[str, list[torch.Tensor]] = {
                metric: [] for metric in SCIENCE_METRICS
            }
            for policy_seed in seeds:
                left_key = (representation_id, left_variant, policy_seed)
                right_key = (representation_id, right_variant, policy_seed)
                for metric in SCIENCE_METRICS:
                    left = torch.cat(candidates[left_key][metric])
                    right = torch.cat(candidates[right_key][metric])
                    if left.shape != right.shape:
                        raise RuntimeError("modal ablation lost paired episode alignment")
                    parts[metric].append(left - right)
            records.append(
                {
                    "representation_id": representation_id,
                    "comparison": comparison_id,
                    "left_variant": left_variant,
                    "right_variant": right_variant,
                    "policy_seeds": seeds,
                    "episodes": int(
                        sum(item.numel() for item in parts["power_in_bucket"])
                    ),
                    "paired_delta_left_minus_right": {
                        metric: _distribution(torch.cat(values))
                        for metric, values in parts.items()
                    },
                }
            )
    return records


def _upstream_quarter_reproduction(
    experiment: dict[str, Any],
    grouped: list[dict[str, Any]],
    *,
    quick: bool,
    tolerance: float,
) -> dict[str, Any]:
    if quick:
        return {
            "status": "NOT_APPLICABLE_QUICK",
            "tolerance": tolerance,
            "note": "Quick smoke uses a different seed and cannot reproduce formal upstream values.",
        }
    upstream_summary = json.loads(
        _project_path(experiment["upstream_diagnostic"]["summary"]).read_text(
            encoding="utf-8"
        )
    )
    upstream_by_representation = {
        str(item["representation_id"]): item
        for item in upstream_summary["grouped_variant_summaries"]
        if str(item["variant"]) == "quarter_residual"
    }
    current_by_representation = {
        str(item["representation_id"]): item
        for item in grouped
        if str(item["variant"]) == "all_modes_quarter"
    }
    records: dict[str, Any] = {}
    overall_max = 0.0
    for representation_id in ("zernike_21", "zernike_36"):
        upstream = upstream_by_representation[representation_id]
        current = current_by_representation[representation_id]
        differences = {
            "relative_power_gain": abs(
                float(current["relative_power_gain"])
                - float(upstream["relative_power_gain"])
            )
        }
        for metric in SCIENCE_METRICS:
            differences[f"candidate_{metric}_mean"] = abs(
                float(current["candidate"][metric]["mean"])
                - float(upstream["candidate"][metric]["mean"])
            )
            differences[f"baseline_{metric}_mean"] = abs(
                float(current["baseline"][metric]["mean"])
                - float(upstream["baseline"][metric]["mean"])
            )
        for metric in ACTION_METRICS:
            differences[f"action_{metric}_mean"] = abs(
                float(current["action_diagnostics"][metric]["mean"])
                - float(upstream["action_diagnostics"][metric]["mean"])
            )
        max_difference = max(differences.values())
        overall_max = max(overall_max, max_difference)
        records[representation_id] = {
            "max_absolute_summary_difference": max_difference,
            "differences": differences,
        }
    return {
        "status": "PASS" if overall_max <= tolerance else "FAIL",
        "max_absolute_summary_difference": overall_max,
        "tolerance": tolerance,
        "by_representation": records,
    }


def interpret_modal_ablation(
    grouped: list[dict[str, Any]],
    *,
    zero_max_abs_by_representation: dict[str, float],
    zero_tolerance: float,
    violation_limit: float,
) -> dict[str, Any]:
    """给出保守的模态归因标签，不自动授权训练或算法修改。"""
    results: dict[str, Any] = {}
    for representation_id in sorted({str(item["representation_id"]) for item in grouped}):
        by_variant = {
            str(item["variant"]): item
            for item in grouped
            if str(item["representation_id"]) == representation_id
        }
        required = {
            "zero_residual",
            "all_modes_quarter",
            "anchor_only_quarter",
            "added_modes_only_quarter",
        }
        if not required.issubset(by_variant):
            raise RuntimeError("modal-ablation interpretation is missing a required variant")
        gains = {
            variant: float(by_variant[variant]["relative_power_gain"])
            for variant in required
        }
        anchor_gain = gains["anchor_only_quarter"]
        added_gain = gains["added_modes_only_quarter"]
        full_gain = gains["all_modes_quarter"]
        wiring_failed = (
            zero_max_abs_by_representation[representation_id] > zero_tolerance
        )
        if wiring_failed:
            label = "WIRING_OR_BASELINE_MISMATCH_SUSPECTED"
        elif anchor_gain >= 0 and added_gain < 0:
            label = "ADDED_MODE_ACTIONS_DESTRUCTIVE"
        elif added_gain >= 0 and anchor_gain < 0:
            label = "ANCHOR_MODE_INTERFERENCE"
        elif anchor_gain >= 0 and added_gain >= 0 and full_gain < 0:
            label = "CLOSED_LOOP_SUBSPACE_INTERACTION_FAILURE"
        elif anchor_gain < 0 and added_gain < 0:
            label = "BOTH_ACTION_SUBSPACES_DESTRUCTIVE"
        elif full_gain >= 0:
            label = "ALL_MODE_QUARTER_ACTION_RECOVERED"
        else:
            label = "CAUSE_NOT_ISOLATED"
        variant_details: dict[str, Any] = {}
        for variant in (
            "all_modes_quarter",
            "anchor_only_quarter",
            "added_modes_only_quarter",
        ):
            record = by_variant[variant]
            violation = float(record["candidate"]["violation_fraction"]["mean"])
            variant_details[variant] = {
                "relative_power_gain": float(record["relative_power_gain"]),
                "power_delta_mean": float(
                    record["paired_delta_candidate_minus_baseline"][
                        "power_in_bucket"
                    ]["mean"]
                ),
                "strehl_delta_mean": float(
                    record["paired_delta_candidate_minus_baseline"]["strehl"][
                        "mean"
                    ]
                ),
                "phase_rmse_delta_mean": float(
                    record["paired_delta_candidate_minus_baseline"]["phase_rmse"][
                        "mean"
                    ]
                ),
                "violation_fraction": violation,
                "violation_pass": violation <= violation_limit,
                "residual_l2_projection_fraction": float(
                    record["action_diagnostics"][
                        "residual_l2_projection_fraction"
                    ]["mean"]
                ),
                "final_projection_fraction": float(
                    record["action_diagnostics"]["final_projection_fraction"][
                        "mean"
                    ]
                ),
            }
        results[representation_id] = {
            "primary_label": label,
            "wiring_status": "FAIL" if wiring_failed else "PASS",
            "zero_max_absolute_difference": zero_max_abs_by_representation[
                representation_id
            ],
            "variant_details": variant_details,
            "anchor_only_has_positive_gain": anchor_gain > 0,
            "added_only_has_positive_gain": added_gain > 0,
            "all_modes_has_positive_gain": full_gain > 0,
        }
    return {
        "by_representation": results,
        "training_authorized": False,
        "algorithm_change_authorized": False,
        "s4d3_authorized": False,
        "real_hardware_authorized": False,
        "note": (
            "A formal audit must review per-policy and per-profile consistency before "
            "using these post-hoc labels to choose the next experiment."
        ),
    }


def _effective_settings(
    experiment: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    evaluation = experiment["evaluation"]
    settings = {
        "quick": quick,
        "output_directory": experiment["outputs"]["directory"],
        "representations": deepcopy(experiment["representations"]),
        "checkpoints": deepcopy(experiment["checkpoints"]),
        "variants": deepcopy(experiment["variants"]),
        "profile_ids": list(experiment["diagnostic_profile_ids"]),
        "physical_conditions": deepcopy(experiment["diagnostic_physical_conditions"]),
        "batch_size": int(evaluation["episodes_per_physical_condition"]),
        "steps": int(evaluation["steps"]),
        "zero_equivalence_tolerance": float(
            evaluation["zero_equivalence_tolerance"]
        ),
        "upstream_quarter_reproduction_tolerance": float(
            evaluation["upstream_quarter_reproduction_tolerance"]
        ),
        "violation_limit": float(evaluation["violation_limit"]),
    }
    if quick:
        quick_settings = experiment["quick"]
        representation_ids = set(map(str, quick_settings["representation_ids"]))
        policy_seeds = set(map(int, quick_settings["policy_seeds"]))
        variant_ids = set(map(str, quick_settings["variant_ids"]))
        settings.update(
            {
                "output_directory": experiment["outputs"]["quick_directory"],
                "representations": [
                    item
                    for item in experiment["representations"]
                    if str(item["id"]) in representation_ids
                ],
                "checkpoints": [
                    item
                    for item in experiment["checkpoints"]
                    if str(item["representation_id"]) in representation_ids
                    and int(item["policy_seed"]) in policy_seeds
                ],
                "variants": [
                    item
                    for item in experiment["variants"]
                    if str(item["id"]) in variant_ids
                ],
                "profile_ids": list(quick_settings["profile_ids"]),
                "physical_conditions": deepcopy(
                    quick_settings["physical_conditions"]
                ),
                "batch_size": int(quick_settings["batch_size"]),
                "steps": int(quick_settings["steps"]),
            }
        )
    return settings
