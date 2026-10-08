"""R3-D2-A冻结演员的多步评论家训练与一次性机制审计。"""

from __future__ import annotations

import csv
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any

from scipy.stats import t as student_t
import torch
from torch.nn import functional

from src.rl.residual_sac import QNetwork, SacConfig
from src.rl.s4_r3_critic_source_diagnostic import (
    ExtendedPolicy,
    _load_extended_policy,
)
from src.rl.s4_r3_failure_diagnostic import (
    _load_student,
    _scenario_config,
)
from src.rl.s4_r3_multistep_critic_repair import (
    CriticTargetSpec,
    preflight_s4_r3_multistep_critic_repair_design,
    select_critic_target,
)
from src.rl.s4_r3_multistep_target_diagnostic import (
    _effective_settings as _d1c_effective_settings,
)
from src.rl.s4_r3_student_anchored_sac import _make_controller, _normalize_state
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import (
    _append_jsonl,
    _file_sha256,
    _git_record,
    _load_yaml,
    _noisy_observation,
    _profiles,
    _project_path,
    _relative,
    _residual_reward,
    _runtime_record,
    _source_manifest,
    _write_json,
)
from src.rl.student_anchored_control import FrozenStudentPolicy
from src.runtime import resolve_device
from src.simulation.config import S1EnvConfig, load_s1_config
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.hardware_effects import HardwareProfile
from src.simulation.robust_control import RobustnessCondition
from src.training_progress import counted_progress, update_progress


REQUIRED_CANDIDATES = ("zero", "actor")
REPAIR_METHODS = ("n_step_16", "n_step_32", "td_lambda_095")


@dataclass(frozen=True)
class ProbeDataset:
    """一个策略种子、一个完整数据拆分的配对评论家样本。"""

    states: torch.Tensor
    actions: torch.Tensor
    n_step_targets: torch.Tensor
    empirical_reward_returns: torch.Tensor
    empirical_power_returns: torch.Tensor
    original_q: torch.Tensor
    rows: list[dict[str, Any]]

    def validate(self, *, max_horizon: int) -> None:
        count = int(self.states.shape[0])
        if self.states.ndim != 2 or self.actions.ndim != 2:
            raise ValueError("critic states and actions must be matrices")
        if self.n_step_targets.shape != (count, max_horizon):
            raise ValueError("critic target curve has the wrong shape")
        if self.empirical_reward_returns.shape != (count, max_horizon):
            raise ValueError("empirical reward curve has the wrong shape")
        if self.empirical_power_returns.shape != (count, max_horizon):
            raise ValueError("empirical power curve has the wrong shape")
        if self.actions.shape[0] != count or self.original_q.shape != (count,):
            raise ValueError("critic dataset tensors do not align")
        if len(self.rows) != count:
            raise ValueError("critic dataset metadata do not align")
        tensors = (
            self.states,
            self.actions,
            self.n_step_targets,
            self.empirical_reward_returns,
            self.empirical_power_returns,
            self.original_q,
        )
        if any(not bool(torch.isfinite(item).all()) for item in tensors):
            raise RuntimeError("critic dataset contains non-finite values")
        keys = [
            (
                row["profile_id"],
                row["condition_id"],
                row["probe_step"],
                row["episode_index"],
                row["candidate"],
            )
            for row in self.rows
        ]
        if len(keys) != len(set(keys)):
            raise RuntimeError("critic dataset contains duplicate sample keys")

    def payload(self) -> dict[str, Any]:
        return {
            "states": self.states,
            "actions": self.actions,
            "n_step_targets": self.n_step_targets,
            "empirical_reward_returns": self.empirical_reward_returns,
            "empirical_power_returns": self.empirical_power_returns,
            "original_q": self.original_q,
            "rows": self.rows,
        }


def run_s4_r3_multistep_critic_training(
    config_path: str | Path,
    *,
    quick: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """运行冻结演员评论家训练；正式入口只允许CUDA。"""
    experiment_path = _project_path(config_path)
    experiment = _load_yaml(experiment_path)
    settings = _effective_settings(experiment, quick=quick)
    preflight, d1_experiment, checkpoints = preflight_s4_r3_multistep_critic_training(
        experiment_path, experiment, settings, quick=quick
    )
    if preflight_only:
        return preflight

    device = resolve_device("cuda")
    output_directory = _project_path(settings["output_directory"])
    if output_directory.exists():
        raise FileExistsError(
            f"R3-D2-A output already exists; preserve it for audit: {output_directory}"
        )
    output_directory.mkdir(parents=True)
    (output_directory / "datasets").mkdir()
    _write_json(output_directory / "preflight.json", preflight)
    _write_json(output_directory / "effective_config.json", settings)
    source_manifest = _source_manifest(experiment["tracked_source_files"])
    _write_json(output_directory / "source_manifest.json", source_manifest)

    base_config, _ = load_s1_config(_project_path(d1_experiment["environment_config"]))
    representation = ActionRepresentation.from_mapping(d1_experiment["representation"])
    base_config = replace(base_config, num_modes=representation.num_modes)
    basis, _, basis_diagnostics = build_action_basis(
        base_config, representation, device
    )
    target_specs = [CriticTargetSpec.from_mapping(item) for item in settings["targets"]]

    started = time.perf_counter()
    fits: list[dict[str, Any]] = []
    datasets: dict[tuple[int, str], ProbeDataset] = {}
    dataset_records: list[dict[str, Any]] = []
    total_branches = int(preflight["expected_branch_rollouts"])
    collection_bar = counted_progress(
        total=total_branches,
        description="R3-D2-A采集配对目标",
        unit="分支",
    )
    collection_started = time.perf_counter()
    collection_progress_path = output_directory / "collection_progress.jsonl"
    for checkpoint in checkpoints:
        policy = _load_extended_policy(checkpoint, device=device)
        student = _load_student(checkpoint, experiment=d1_experiment, device=device)
        for split_name in ("development", "validation", "mechanism_audit"):
            dataset = _collect_policy_split(
                split_name=split_name,
                split=settings["splits"][split_name],
                experiment=d1_experiment,
                base_config=base_config,
                basis=basis,
                policy=policy,
                student=student,
                candidate_actions=settings["candidate_actions"],
                max_horizon=int(settings["max_target_horizon"]),
                collection_bar=collection_bar,
                collection_progress_path=collection_progress_path,
                collection_started=collection_started,
                device=device,
            )
            dataset.validate(max_horizon=int(settings["max_target_horizon"]))
            datasets[(policy.policy_seed, split_name)] = dataset
            dataset_path = (
                output_directory
                / "datasets"
                / f"seed_{policy.policy_seed}_{split_name}.pt"
            )
            torch.save(dataset.payload(), dataset_path)
            dataset_records.append(
                {
                    "policy_seed": policy.policy_seed,
                    "split": split_name,
                    "path": _relative(dataset_path),
                    "sha256": _file_sha256(dataset_path),
                    "samples": int(dataset.states.shape[0]),
                    "state_size": int(dataset.states.shape[1]),
                    "action_size": int(dataset.actions.shape[1]),
                    "target_horizons": int(dataset.n_step_targets.shape[1]),
                }
            )
    collection_bar.close()

    d1c_alignment = _verify_d1c_audit_alignment(
        datasets,
        experiment=experiment,
        quick=quick,
        tolerance=float(experiment["alignment"]["tolerance"]),
    )
    for checkpoint in checkpoints:
        policy_seed = int(checkpoint["policy_seed"])
        initial_states = _initial_critic_states(
            settings=settings,
            policy_seed=policy_seed,
            device=device,
        )
        for spec in target_specs:
            run_directory = output_directory / f"seed_{policy_seed}" / spec.identifier
            run_directory.mkdir(parents=True)
            result = fit_frozen_critic_pair(
                training=datasets[(policy_seed, "development")],
                validation=datasets[(policy_seed, "validation")],
                spec=spec,
                settings=settings,
                policy_seed=policy_seed,
                initial_states=initial_states,
                output_directory=run_directory,
                device=device,
            )
            result["checkpoint_sha256"] = _file_sha256(
                _project_path(result["checkpoint"])
            )
            fits.append(result)

    audit, audit_rows = _audit_fitted_critics(
        fits=fits,
        datasets=datasets,
        target_specs=target_specs,
        settings=settings,
        output_directory=output_directory,
        quick=quick,
        device=device,
    )
    audit_path = output_directory / "mechanism_audit_records.csv"
    _write_rows(audit_path, audit_rows)
    fit_path = output_directory / "fit_summary.csv"
    _write_rows(fit_path, [_flat_fit_record(item) for item in fits])

    interpretation = _interpret_audit(
        audit,
        fits=fits,
        settings=settings,
        quick=quick,
    )
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": str(experiment["metadata"]["version_label"]),
        },
        "experiment": {
            "id": str(experiment["metadata"]["experiment_id"]),
            "status": "completed_pending_independent_audit",
            "quick": quick,
            "duration_seconds": time.perf_counter() - started,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "evidence_boundary": {
            "frozen_actor_critic_only_training": True,
            "actor_updates": 0,
            "alpha_updates": 0,
            "student_updates": 0,
            "reward_changed": False,
            "action_budget_changed": False,
            "full_rl_retrained": False,
            "sealed_s4d3_accessed": False,
            "real_slm_actions": False,
        },
        "design": {
            "policy_seeds": settings["policy_seeds"],
            "target_methods": settings["targets"],
            "candidate_actions": settings["candidate_actions"],
            "split_unit": "complete_turbulence_episode",
            "checkpoint_selection_uses_mechanism_audit": False,
            "common_reference": "deterministic_32_step_bootstrap_target",
        },
        "basis_diagnostics": basis_diagnostics,
        "d1c_alignment": d1c_alignment,
        "datasets": dataset_records,
        "fits": fits,
        "audit": audit,
        "interpretation": interpretation,
        "records": {
            "collection_progress": _relative(collection_progress_path),
            "collection_progress_sha256": _file_sha256(collection_progress_path),
            "mechanism_audit_records": _relative(audit_path),
            "mechanism_audit_records_sha256": _file_sha256(audit_path),
            "mechanism_audit_rows": len(audit_rows),
            "fit_summary": _relative(fit_path),
            "fit_summary_sha256": _file_sha256(fit_path),
            "fit_rows": len(fits),
        },
        "inputs": {
            "experiment_config": _relative(experiment_path),
            "experiment_config_sha256": _file_sha256(experiment_path),
            "upstream_d1c_summary": experiment["upstream_d1c"]["summary"],
            "upstream_d1c_summary_sha256": experiment["upstream_d1c"][
                "summary_sha256"
            ],
            "source_manifest": source_manifest,
        },
        "runtime": _runtime_record(),
        "git": _git_record(),
        "next_action": (
            "Stop for read-only audit. Do not start full RL, open S4-D3, or "
            "select a repair from quick-smoke output."
        ),
    }
    _write_json(output_directory / "summary.json", summary)
    return summary


def preflight_s4_r3_multistep_critic_training(
    experiment_path: Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    *,
    quick: bool,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """锁定上游证据、训练规模和所有禁止修改项。"""
    metadata = experiment["metadata"]
    if str(metadata["stage"]) != "S4-D2-R3-D2-A":
        raise ValueError("critic training metadata must identify S4-D2-R3-D2-A")
    if not bool(metadata.get("user_authorized_training_preparation", False)):
        raise RuntimeError("R3-D2-A trainer preparation requires authorization")
    if not bool(metadata.get("allow_formal_critic_training", False)):
        raise RuntimeError("R3-D2-A critic-only training is not enabled")
    for field in (
        "allow_actor_updates",
        "allow_alpha_updates",
        "allow_student_finetuning",
        "allow_reward_change",
        "allow_action_budget_change",
        "allow_gate_relaxation",
        "allow_s4d3_access",
        "allow_real_hardware_actions",
    ):
        if bool(metadata.get(field, True)):
            raise RuntimeError(f"R3-D2-A protection flag must remain false: {field}")

    design = experiment["design_contract"]
    for field in ("config", "plan"):
        if _file_sha256(_project_path(design[field])) != str(
            design[f"{field}_sha256"]
        ):
            raise RuntimeError(f"R3-D2-A design contract hash mismatch: {field}")
    design_preflight = preflight_s4_r3_multistep_critic_repair_design(
        design["config"]
    )
    if design_preflight["status"] != "READY_FOR_TRAINER_IMPLEMENTATION":
        raise RuntimeError("R3-D2-A design is not ready for implementation")

    upstream = experiment["upstream_d1c"]
    for field in (
        "summary",
        "preflight",
        "effective_config",
        "source_manifest",
        "probe_records",
        "episode_records",
        "experiment_config",
        "audit_record",
    ):
        if _file_sha256(_project_path(upstream[field])) != str(
            upstream[f"{field}_sha256"]
        ):
            raise RuntimeError(f"R3-D2-A upstream hash mismatch: {field}")
    upstream_summary = json.loads(
        _project_path(upstream["summary"]).read_text(encoding="utf-8")
    )
    if bool(upstream_summary["experiment"]["quick"]):
        raise RuntimeError("R3-D2-A cannot consume quick D1-C evidence")
    if str(upstream_summary["interpretation"]["status"]) != str(
        upstream["required_interpretation_status"]
    ):
        raise RuntimeError("R3-D2-A upstream D1-C interpretation changed")
    if not bool(upstream_summary["upstream_alignment"]["pass"]):
        raise RuntimeError("R3-D2-A requires passing D1-C alignment")

    d1c_experiment = _load_yaml(_project_path(upstream["experiment_config"]))
    d1b = _load_yaml(_project_path(d1c_experiment["upstream_d1b"]["experiment_config"]))
    d1a = _load_yaml(_project_path(d1b["upstream_d1a"]["experiment_config"]))
    d1_experiment = _load_yaml(_project_path(d1a["upstream_d1"]["experiment_config"]))
    d1c_settings = _d1c_effective_settings(d1c_experiment, quick=False)
    checkpoints = [
        item
        for item in d1c_settings["checkpoints"]
        if str(item["arm"]) == "student_backbone"
        and int(item["policy_seed"]) in set(settings["policy_seeds"])
    ]
    if len(checkpoints) != len(settings["policy_seeds"]):
        raise RuntimeError("R3-D2-A requires one student checkpoint per policy seed")
    if tuple(item["id"] for item in settings["candidate_actions"]) != REQUIRED_CANDIDATES:
        raise RuntimeError("R3-D2-A must compare only zero and actor")
    specs = [CriticTargetSpec.from_mapping(item) for item in settings["targets"]]
    if [item.identifier for item in specs] != [
        "one_step_control",
        "n_step_16",
        "n_step_32",
        "td_lambda_095",
    ]:
        raise RuntimeError("R3-D2-A target methods changed")
    if max(item.horizon for item in specs) != int(settings["max_target_horizon"]):
        raise RuntimeError("R3-D2-A target horizon mismatch")
    for split in settings["splits"].values():
        if max(split["probe_steps"]) + int(settings["max_target_horizon"]) > int(
            settings["episode_length_steps"]
        ):
            raise RuntimeError("R3-D2-A probe exceeds the locked episode")
    _verify_split_seeds(settings, reserved=int(experiment["data"]["reserved_s4d3_seed_base"]))

    missing = [
        item
        for item in experiment["tracked_source_files"]
        if not _project_path(item).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"R3-D2-A tracked source files missing: {missing}")
    expected_branches = sum(
        len(checkpoints)
        * len(split["profile_ids"])
        * len(split["conditions"])
        * len(split["probe_steps"])
        * len(settings["candidate_actions"])
        for split in settings["splits"].values()
    )
    expected_samples = sum(
        len(checkpoints)
        * len(split["profile_ids"])
        * len(split["conditions"])
        * len(split["probe_steps"])
        * len(settings["candidate_actions"])
        * int(split["episodes_per_condition"])
        for split in settings["splits"].values()
    )
    output = _project_path(settings["output_directory"])
    preflight = {
        "status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_TRAINING",
        "quick": quick,
        "upstream_interpretation": upstream_summary["interpretation"]["status"],
        "policy_seeds": settings["policy_seeds"],
        "target_ids": [item.identifier for item in specs],
        "planned_critic_fits": len(checkpoints) * len(specs),
        "expected_branch_rollouts": expected_branches,
        "expected_samples": expected_samples,
        "expected_horizon_labels": expected_samples
        * int(settings["max_target_horizon"]),
        "maximum_updates_per_fit": int(settings["maximum_updates"]),
        "actor_updates": 0,
        "alpha_updates": 0,
        "student_updates": 0,
        "cuda_required": True,
        "output_directory": _relative(output),
        "output_exists": output.exists(),
        "sealed_s4d3_access": False,
        "real_slm_actions": False,
        "formal_command": (
            ".\\.venv\\Scripts\\python.exe "
            "scripts\\train_s4_r3_multistep_critics.py --config "
            f"{str(_relative(experiment_path)).replace('/', chr(92))}"
        ),
    }
    return preflight, d1_experiment, checkpoints


@torch.no_grad()
def _collect_policy_split(
    *,
    split_name: str,
    split: dict[str, Any],
    experiment: dict[str, Any],
    base_config: S1EnvConfig,
    basis: torch.Tensor,
    policy: ExtendedPolicy,
    student: FrozenStudentPolicy,
    candidate_actions: list[dict[str, Any]],
    max_horizon: int,
    collection_bar: Any,
    collection_progress_path: Path,
    collection_started: float,
    device: torch.device,
) -> ProbeDataset:
    states: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    powers: list[torch.Tensor] = []
    q_values: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    profiles = _profiles(experiment, split["profile_ids"])
    conditions = [RobustnessCondition.from_mapping(item) for item in split["conditions"]]
    preview = int(experiment["teacher"]["preview_horizon_frames"])
    for profile in profiles:
        for condition in conditions:
            config = _scenario_config(
                base_config,
                condition,
                profile,
                episodes=int(split["episodes_per_condition"]),
                steps=int(split["episode_length_steps"]),
                preview=preview,
            )
            for probe_step in split["probe_steps"]:
                for candidate in candidate_actions:
                    result = _collect_target_branch(
                        experiment=experiment,
                        config=config,
                        condition=condition,
                        profile=profile,
                        basis=basis,
                        policy=policy,
                        student=student,
                        candidate=candidate,
                        probe_step=int(probe_step),
                        max_horizon=max_horizon,
                        device=device,
                    )
                    count = int(result["states"].shape[0])
                    states.append(result["states"])
                    actions.append(result["actions"])
                    targets.append(result["n_step_targets"])
                    rewards.append(result["empirical_reward_returns"])
                    powers.append(result["empirical_power_returns"])
                    q_values.append(result["original_q"])
                    for episode_index in range(count):
                        rows.append(
                            {
                                "split": split_name,
                                "arm": policy.arm,
                                "policy_seed": policy.policy_seed,
                                "student_id": policy.student_id,
                                "profile_id": profile.identifier,
                                "condition_id": condition.identifier,
                                "probe_step": int(probe_step),
                                "episode_index": episode_index,
                                "episode_seed": condition.base_seed + episode_index,
                                "candidate": str(candidate["id"]),
                            }
                        )
                    collection_bar.update(1)
                    _append_jsonl(
                        collection_progress_path,
                        {
                            "completed_branches": int(collection_bar.n),
                            "total_branches": int(collection_bar.total),
                            "split": split_name,
                            "policy_seed": policy.policy_seed,
                            "profile_id": profile.identifier,
                            "condition_id": condition.identifier,
                            "probe_step": int(probe_step),
                            "candidate": str(candidate["id"]),
                            **_progress_runtime_fields(
                                completed=int(collection_bar.n),
                                total=int(collection_bar.total),
                                started=collection_started,
                                device=device,
                            ),
                        },
                    )
                    update_progress(
                        collection_bar,
                        device=device,
                        metrics={
                            "种子": float(policy.policy_seed),
                            "探针": float(probe_step),
                            "Q": float(result["original_q"].mean()),
                            "32步": float(result["n_step_targets"][:, -1].mean()),
                        },
                    )
    return ProbeDataset(
        states=torch.cat(states),
        actions=torch.cat(actions),
        n_step_targets=torch.cat(targets),
        empirical_reward_returns=torch.cat(rewards),
        empirical_power_returns=torch.cat(powers),
        original_q=torch.cat(q_values),
        rows=rows,
    )


@torch.no_grad()
def _collect_target_branch(
    *,
    experiment: dict[str, Any],
    config: S1EnvConfig,
    condition: RobustnessCondition,
    profile: HardwareProfile,
    basis: torch.Tensor,
    policy: ExtendedPolicy,
    student: FrozenStudentPolicy,
    candidate: dict[str, Any],
    probe_step: int,
    max_horizon: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if str(candidate["id"]) not in REQUIRED_CANDIDATES:
        raise ValueError(f"unsupported critic candidate: {candidate['id']}")
    environment = AdaptiveOpticsEnv(
        config, device, profile.effects_config(), basis_override=basis
    )
    observation, _ = environment.reset(seed=condition.base_seed)
    observation_generator = torch.Generator(device=device).manual_seed(
        condition.base_seed + 40_000_000
    )
    noisy = _noisy_observation(
        observation,
        config.num_modes,
        profile.observation_noise_std_rad,
        observation_generator,
    )
    controller = _make_controller(experiment, config, student)
    state = controller.reset(noisy)
    zeros = torch.zeros(config.batch_size, 11, device=device)
    for _ in range(probe_step):
        prefix = controller.compose_action(state, zeros)
        observation, _, terminated, truncated, _ = environment.step(
            prefix.composed.final_delta_rad
        )
        if bool((terminated | truncated).any()):
            raise RuntimeError("R3-D2-A prefix terminated before the probe")
        noisy = _noisy_observation(
            observation,
            config.num_modes,
            profile.observation_noise_std_rad,
            observation_generator,
        )
        state = controller.advance_observation(noisy)

    normalized_state = _normalize_state(state, student)
    actor_action = policy.actor.deterministic(normalized_state)
    first_action = float(candidate["scale"]) * actor_action
    original_q = torch.minimum(
        policy.q1(normalized_state, first_action),
        policy.q2(normalized_state, first_action),
    ).squeeze(-1)
    reward_return = torch.zeros(config.batch_size, device=device, dtype=torch.float64)
    power_return = torch.zeros_like(reward_return)
    target_curves: list[torch.Tensor] = []
    reward_curves: list[torch.Tensor] = []
    power_curves: list[torch.Tensor] = []
    for local_step in range(1, max_horizon + 1):
        correction = (
            first_action
            if local_step == 1
            else policy.actor.deterministic(_normalize_state(state, student))
        )
        action = controller.compose_action(state, correction)
        observation, _, terminated, truncated, info = environment.step(
            action.composed.final_delta_rad
        )
        if bool((terminated | truncated).any()):
            raise RuntimeError("R3-D2-A target branch terminated within 32 steps")
        reward = _residual_reward(
            measured_power=info["measured_power_in_bucket"],
            normalized_residual=action.correction_normalized,
            violation=info["violation_fraction"],
            reward_config=experiment["reward"],
        )
        discount = policy.gamma ** (local_step - 1)
        reward_return += discount * reward.double()
        power_return += discount * info["reward_power_in_bucket"].double()
        next_noisy = _noisy_observation(
            observation,
            config.num_modes,
            profile.observation_noise_std_rad,
            observation_generator,
        )
        next_state = controller.advance_observation(next_noisy)
        normalized_next = _normalize_state(next_state, student)
        bootstrap_action = policy.actor.deterministic(normalized_next)
        bootstrap_q = torch.minimum(
            policy.target_q1(normalized_next, bootstrap_action),
            policy.target_q2(normalized_next, bootstrap_action),
        ).squeeze(-1)
        target_curves.append(
            reward_return.clone() + (policy.gamma**local_step) * bootstrap_q.double()
        )
        reward_curves.append(reward_return.clone())
        power_curves.append(power_return.clone())
        state = next_state
    return {
        "states": normalized_state.detach().cpu().float(),
        "actions": first_action.detach().cpu().float(),
        "n_step_targets": torch.stack(target_curves, dim=1).cpu(),
        "empirical_reward_returns": torch.stack(reward_curves, dim=1).cpu(),
        "empirical_power_returns": torch.stack(power_curves, dim=1).cpu(),
        "original_q": original_q.detach().cpu().float(),
    }


def fit_frozen_critic_pair(
    *,
    training: ProbeDataset,
    validation: ProbeDataset,
    spec: CriticTargetSpec,
    settings: dict[str, Any],
    policy_seed: int,
    initial_states: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
    output_directory: Path,
    device: torch.device,
) -> dict[str, Any]:
    """在冻结标签上拟合双评论家；不创建或更新演员。"""
    config = SacConfig(
        state_size=int(settings["state_size"]),
        action_size=int(settings["action_size"]),
        hidden_size=int(settings["hidden_size"]),
        learning_rate=float(settings["learning_rate"]),
    )
    q1 = QNetwork(config).to(device)
    q2 = QNetwork(config).to(device)
    q1.load_state_dict(initial_states[0])
    q2.load_state_dict(initial_states[1])
    optimizer = torch.optim.Adam(
        (*q1.parameters(), *q2.parameters()), lr=float(settings["learning_rate"])
    )
    train_target = select_critic_target(training.n_step_targets, spec).float()
    validation_target = select_critic_target(validation.n_step_targets, spec).float()
    batch_size = int(settings["batch_size"])
    if int(training.states.shape[0]) < batch_size:
        raise ValueError("R3-D2-A training dataset is smaller than one batch")
    generator = torch.Generator().manual_seed(
        int(settings["batch_order_seed_offset"]) + policy_seed
    )
    loss_path = output_directory / "loss_history.csv"
    progress_path = output_directory / "progress.jsonl"
    checkpoint_path = output_directory / "checkpoint_best.pt"
    fields = [
        "policy_seed",
        "target_method",
        "update",
        "training_loss",
        "mean_training_loss",
        "own_target_validation_mse",
        "common_reference_validation_mse",
        "validation_ranking_accuracy",
        "elapsed_seconds",
        "estimated_remaining_seconds",
        "cuda_allocated_gb",
        "cuda_reserved_gb",
    ]
    best_mse = float("inf")
    best_update = 0
    best_metrics: dict[str, float] | None = None
    last_loss = float("nan")
    recent_losses: deque[float] = deque(maxlen=100)
    fit_started = time.perf_counter()
    maximum_updates = int(settings["maximum_updates"])
    validation_interval = int(settings["validation_interval_updates"])
    log_interval = int(settings["log_interval_updates"])
    patience = int(settings["early_stopping_patience_updates"])
    bar = counted_progress(
        total=maximum_updates,
        description=f"D2-A 种子{policy_seed} {spec.identifier}",
        unit="更新",
    )
    with loss_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for update in range(1, maximum_updates + 1):
            indices = torch.randint(
                int(training.states.shape[0]),
                (batch_size,),
                generator=generator,
            )
            states = training.states[indices].to(device)
            actions = training.actions[indices].to(device)
            labels = train_target[indices].to(device)
            p1 = q1(states, actions).squeeze(-1)
            p2 = q2(states, actions).squeeze(-1)
            loss = functional.mse_loss(p1, labels) + functional.mse_loss(p2, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach())
            recent_losses.append(last_loss)
            should_validate = (
                update % validation_interval == 0
                or update % log_interval == 0
                or update == maximum_updates
            )
            if should_validate:
                metrics = _critic_validation_metrics(
                    q1,
                    q2,
                    validation,
                    validation_target=validation_target,
                    device=device,
                )
                own_mse = metrics["own_target_validation_mse"]
                if own_mse < best_mse:
                    best_mse = own_mse
                    best_update = update
                    best_metrics = metrics
                    torch.save(
                        {
                            "algorithm": "frozen_actor_multistep_critic_fit",
                            "policy_seed": policy_seed,
                            "target_spec": {
                                "id": spec.identifier,
                                "kind": spec.kind,
                                "horizon": spec.horizon,
                                "trace_lambda": spec.trace_lambda,
                            },
                            "config": {
                                "state_size": config.state_size,
                                "action_size": config.action_size,
                                "hidden_size": config.hidden_size,
                            },
                            "q1": {k: v.detach().cpu() for k, v in q1.state_dict().items()},
                            "q2": {k: v.detach().cpu() for k, v in q2.state_dict().items()},
                            "best_update": best_update,
                            "best_metrics": best_metrics,
                            "actor_updates": 0,
                            "alpha_updates": 0,
                        },
                        checkpoint_path,
                    )
                record = {
                    "policy_seed": policy_seed,
                    "target_method": spec.identifier,
                    "update": update,
                    "training_loss": last_loss,
                    "mean_training_loss": sum(recent_losses) / len(recent_losses),
                    **metrics,
                    **_progress_runtime_fields(
                        completed=update,
                        total=maximum_updates,
                        started=fit_started,
                        device=device,
                    ),
                }
                writer.writerow(record)
                handle.flush()
                _append_jsonl(progress_path, record)
                update_progress(
                    bar,
                    device=device,
                    metrics={
                        "平均损失": sum(recent_losses) / len(recent_losses),
                        "验证误差": metrics["own_target_validation_mse"],
                        "32步误差": metrics["common_reference_validation_mse"],
                        "排序率": metrics["validation_ranking_accuracy"],
                    },
                )
            bar.update(1)
            if best_update and update - best_update >= patience:
                break
    bar.close()
    if best_metrics is None or not checkpoint_path.is_file():
        raise RuntimeError("R3-D2-A critic fit finished without a best checkpoint")
    return {
        "policy_seed": policy_seed,
        "target_method": spec.identifier,
        "updates_completed": int(bar.n),
        "best_update": best_update,
        "training_samples": int(training.states.shape[0]),
        "validation_samples": int(validation.states.shape[0]),
        **best_metrics,
        "checkpoint": _relative(checkpoint_path),
        "loss_history": _relative(loss_path),
        "progress_log": _relative(progress_path),
        "actor_updates": 0,
        "alpha_updates": 0,
    }


@torch.no_grad()
def _critic_validation_metrics(
    q1: QNetwork,
    q2: QNetwork,
    dataset: ProbeDataset,
    *,
    validation_target: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    prediction = _predict_min_q(q1, q2, dataset.states, dataset.actions, device=device)
    common = dataset.n_step_targets[:, -1].float()
    return {
        "own_target_validation_mse": float(
            functional.mse_loss(prediction, validation_target)
        ),
        "common_reference_validation_mse": float(functional.mse_loss(prediction, common)),
        "validation_ranking_accuracy": _pairwise_ranking_accuracy(
            dataset, prediction, common
        ),
    }


@torch.no_grad()
def _predict_min_q(
    q1: QNetwork,
    q2: QNetwork,
    states: torch.Tensor,
    actions: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 2048,
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    q1.eval()
    q2.eval()
    for start in range(0, int(states.shape[0]), batch_size):
        stop = min(start + batch_size, int(states.shape[0]))
        s = states[start:stop].to(device)
        a = actions[start:stop].to(device)
        values.append(torch.minimum(q1(s, a), q2(s, a)).squeeze(-1).cpu())
    q1.train()
    q2.train()
    return torch.cat(values)


def _initial_critic_states(
    *, settings: dict[str, Any], policy_seed: int, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    torch.manual_seed(int(settings["initialization_seed_offset"]) + policy_seed)
    torch.cuda.manual_seed_all(int(settings["initialization_seed_offset"]) + policy_seed)
    config = SacConfig(
        state_size=int(settings["state_size"]),
        action_size=int(settings["action_size"]),
        hidden_size=int(settings["hidden_size"]),
    )
    q1 = QNetwork(config).to(device)
    q2 = QNetwork(config).to(device)
    return deepcopy(q1.state_dict()), deepcopy(q2.state_dict())


def _audit_fitted_critics(
    *,
    fits: list[dict[str, Any]],
    datasets: dict[tuple[int, str], ProbeDataset],
    target_specs: list[CriticTargetSpec],
    settings: dict[str, Any],
    output_directory: Path,
    quick: bool,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    del output_directory
    audits: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    fit_by_key = {
        (int(item["policy_seed"]), str(item["target_method"])): item for item in fits
    }
    for policy_seed in settings["policy_seeds"]:
        dataset = datasets[(int(policy_seed), "mechanism_audit")]
        for spec in target_specs:
            fit = fit_by_key[(int(policy_seed), spec.identifier)]
            checkpoint = torch.load(
                _project_path(fit["checkpoint"]), map_location=device, weights_only=False
            )
            config = SacConfig(
                state_size=int(checkpoint["config"]["state_size"]),
                action_size=int(checkpoint["config"]["action_size"]),
                hidden_size=int(checkpoint["config"]["hidden_size"]),
            )
            q1 = QNetwork(config).to(device)
            q2 = QNetwork(config).to(device)
            q1.load_state_dict(checkpoint["q1"])
            q2.load_state_dict(checkpoint["q2"])
            prediction = _predict_min_q(
                q1, q2, dataset.states, dataset.actions, device=device
            )
            pairs = _paired_values(
                dataset,
                {
                    "repaired_q": prediction,
                    "original_q": dataset.original_q,
                    "target_h32": dataset.n_step_targets[:, -1].float(),
                    "reward_h32": dataset.empirical_reward_returns[:, -1].float(),
                    "power_h32": dataset.empirical_power_returns[:, -1].float(),
                },
            )
            cluster_ids = [
                f"{item['condition_id']}:{item['episode_index']}" for item in pairs
            ]
            repaired = torch.tensor(
                [item["repaired_q_delta"] for item in pairs], dtype=torch.float64
            )
            original = torch.tensor(
                [item["original_q_delta"] for item in pairs], dtype=torch.float64
            )
            target = torch.tensor(
                [item["target_h32_delta"] for item in pairs], dtype=torch.float64
            )
            reward = torch.tensor(
                [item["reward_h32_delta"] for item in pairs], dtype=torch.float64
            )
            original_excess = float((original - target).mean())
            repaired_excess = float((repaired - target).mean())
            reduction = 1.0 - abs(repaired_excess) / max(abs(original_excess), 1e-12)
            distribution = _clustered_t_distribution(repaired, cluster_ids)
            audit = {
                "policy_seed": int(policy_seed),
                "target_method": spec.identifier,
                "pairs": len(pairs),
                "independent_trajectory_clusters": distribution["clusters"],
                "repaired_q_advantage_vs_zero": distribution,
                "original_q_advantage_mean": float(original.mean()),
                "target_h32_advantage_mean": float(target.mean()),
                "reward_h32_advantage_mean": float(reward.mean()),
                "original_q_excess_vs_target": original_excess,
                "repaired_q_excess_vs_target": repaired_excess,
                "online_q_excess_reduction_fraction": reduction,
                "pairwise_ranking_accuracy": float(
                    ((repaired * target) > 0).double().mean()
                ),
                "profiles_reported": len({item["profile_id"] for item in pairs}),
                "two_sided_p_value": distribution["two_sided_p_value"],
                "quick": quick,
            }
            audits.append(audit)
            for item in pairs:
                audit_rows.append(
                    {
                        "policy_seed": int(policy_seed),
                        "target_method": spec.identifier,
                        **item,
                    }
                )
    _attach_holm_adjustment(audits)
    return audits, audit_rows


def _interpret_audit(
    audits: list[dict[str, Any]],
    *,
    fits: list[dict[str, Any]],
    settings: dict[str, Any],
    quick: bool,
) -> dict[str, Any]:
    fit_lookup = {
        (int(item["policy_seed"]), str(item["target_method"])): item for item in fits
    }
    control_mse = {
        seed: float(
            fit_lookup[(seed, "one_step_control")][
                "common_reference_validation_mse"
            ]
        )
        for seed in settings["policy_seeds"]
    }
    alpha = float(settings["multiple_comparison_alpha"])
    seed_results: list[dict[str, Any]] = []
    for item in audits:
        method = str(item["target_method"])
        seed = int(item["policy_seed"])
        if method == "one_step_control":
            item["gate"] = "REFERENCE_ONLY"
            continue
        checks = {
            "negative_q_advantage": float(
                item["repaired_q_advantage_vs_zero"]["ci95_high"]
            )
            < float(settings["actor_vs_zero_q_advantage_ci95_high"]),
            "holm_corrected": float(item["holm_adjusted_p_value"]) < alpha,
            "excess_reduction": float(item["online_q_excess_reduction_fraction"])
            >= float(settings["min_online_q_excess_reduction_fraction"]),
            "ranking_accuracy": float(item["pairwise_ranking_accuracy"])
            >= float(settings["min_pairwise_ranking_accuracy"]),
            "common_mse_better_than_control": float(
                fit_lookup[(seed, method)]["common_reference_validation_mse"]
            )
            < control_mse[seed],
            "all_profiles_reported": int(item["profiles_reported"])
            == int(settings["expected_profile_count"]),
        }
        passed = all(checks.values())
        item["checks"] = checks
        item["gate"] = "PASS" if passed else "FAIL"
        seed_results.append(
            {
                "policy_seed": seed,
                "target_method": method,
                "gate": item["gate"],
                "checks": checks,
            }
        )
    method_results: list[dict[str, Any]] = []
    for method in REPAIR_METHODS:
        selected = [item for item in seed_results if item["target_method"] == method]
        passed = len(selected) == len(settings["policy_seeds"]) and all(
            item["gate"] == "PASS" for item in selected
        )
        common_mses = torch.tensor(
            [
                fit_lookup[(int(seed), method)]["common_reference_validation_mse"]
                for seed in settings["policy_seeds"]
            ],
            dtype=torch.float64,
        )
        method_results.append(
            {
                "target_method": method,
                "all_seed_gate": "PASS" if passed else "FAIL",
                "mean_common_reference_validation_mse": float(common_mses.mean()),
                "standard_error_common_reference_validation_mse": (
                    float(common_mses.std(unbiased=True) / math.sqrt(common_mses.numel()))
                    if common_mses.numel() > 1
                    else 0.0
                ),
            }
        )
    if quick:
        status = "QUICK_SMOKE_ONLY"
        chosen = None
    else:
        eligible = [item for item in method_results if item["all_seed_gate"] == "PASS"]
        chosen = _select_method(eligible)
        status = (
            "CRITIC_REPAIR_CANDIDATE_IDENTIFIED"
            if chosen is not None
            else "NO_CRITIC_REPAIR_PASSED"
        )
    return {
        "status": status,
        "seed_results": seed_results,
        "method_results": method_results,
        "selected_method": chosen,
        "full_rl_retraining_authorized": False,
        "s4d3_authorized": False,
        "real_hardware_authorized": False,
        "next_rule": (
            "Independent audit is mandatory. A passing frozen critic only authorizes "
            "design of off-policy-corrected R3-D2-B; it does not authorize full RL."
        ),
    }


def _select_method(eligible: list[dict[str, Any]]) -> str | None:
    if not eligible:
        return None
    best = min(eligible, key=lambda item: item["mean_common_reference_validation_mse"])
    n16 = next((item for item in eligible if item["target_method"] == "n_step_16"), None)
    if n16 is not None and float(n16["mean_common_reference_validation_mse"]) <= float(
        best["mean_common_reference_validation_mse"]
    ) + float(best["standard_error_common_reference_validation_mse"]):
        return "n_step_16"
    return str(best["target_method"])


def _paired_values(
    dataset: ProbeDataset,
    values: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, int]] = {}
    for index, row in enumerate(dataset.rows):
        key = (
            row["profile_id"],
            row["condition_id"],
            row["probe_step"],
            row["episode_index"],
            row["episode_seed"],
        )
        grouped.setdefault(key, {})[str(row["candidate"])] = index
    pairs: list[dict[str, Any]] = []
    for key, candidates in grouped.items():
        if tuple(sorted(candidates)) != ("actor", "zero"):
            raise RuntimeError(f"paired critic candidates are incomplete: {key}")
        actor = candidates["actor"]
        zero = candidates["zero"]
        item = {
            "profile_id": key[0],
            "condition_id": key[1],
            "probe_step": key[2],
            "episode_index": key[3],
            "episode_seed": key[4],
        }
        for name, tensor in values.items():
            item[f"{name}_delta"] = float(tensor[actor] - tensor[zero])
        pairs.append(item)
    return pairs


def _pairwise_ranking_accuracy(
    dataset: ProbeDataset,
    prediction: torch.Tensor,
    reference: torch.Tensor,
) -> float:
    pairs = _paired_values(dataset, {"prediction": prediction, "reference": reference})
    return sum(
        item["prediction_delta"] * item["reference_delta"] > 0 for item in pairs
    ) / len(pairs)


def _clustered_t_distribution(
    values: torch.Tensor, cluster_ids: list[str]
) -> dict[str, float | int]:
    if values.ndim != 1 or values.numel() != len(cluster_ids):
        raise ValueError("clustered values and identifiers must align")
    grouped: dict[str, list[float]] = {}
    for value, identifier in zip(values.tolist(), cluster_ids, strict=True):
        grouped.setdefault(identifier, []).append(float(value))
    cluster_means = torch.tensor(
        [sum(items) / len(items) for items in grouped.values()], dtype=torch.float64
    )
    count = int(cluster_means.numel())
    mean = float(cluster_means.mean())
    if count <= 1:
        margin = 0.0
        statistic = 0.0
        p_value = 1.0
    else:
        standard_error = float(cluster_means.std(unbiased=True) / math.sqrt(count))
        critical = float(student_t.ppf(0.975, count - 1))
        margin = critical * standard_error
        statistic = mean / standard_error if standard_error > 0 else 0.0
        p_value = (
            float(2 * student_t.sf(abs(statistic), count - 1))
            if standard_error > 0
            else (0.0 if mean != 0 else 1.0)
        )
    return {
        "mean": mean,
        "median": float(cluster_means.median()),
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "t_statistic": statistic,
        "two_sided_p_value": p_value,
        "clusters": count,
        "raw_observations": int(values.numel()),
    }


def _attach_holm_adjustment(audits: list[dict[str, Any]]) -> None:
    seeds = sorted({int(item["policy_seed"]) for item in audits})
    for seed in seeds:
        selected = [
            item
            for item in audits
            if int(item["policy_seed"]) == seed
            and str(item["target_method"]) in REPAIR_METHODS
        ]
        ordered = sorted(selected, key=lambda item: float(item["two_sided_p_value"]))
        running = 0.0
        for rank, item in enumerate(ordered):
            adjusted = min(
                1.0,
                float(item["two_sided_p_value"]) * (len(ordered) - rank),
            )
            running = max(running, adjusted)
            item["holm_adjusted_p_value"] = running
    for item in audits:
        if str(item["target_method"]) == "one_step_control":
            item["holm_adjusted_p_value"] = None


def _verify_d1c_audit_alignment(
    datasets: dict[tuple[int, str], ProbeDataset],
    *,
    experiment: dict[str, Any],
    quick: bool,
    tolerance: float,
) -> dict[str, Any]:
    if quick:
        return {"checked": False, "reason": "quick smoke uses separate seeds"}
    path = _project_path(experiment["upstream_d1c"]["episode_records"])
    with path.open("r", encoding="utf-8", newline="") as handle:
        upstream = {
            (
                int(row["policy_seed"]),
                row["profile_id"],
                row["condition_id"],
                int(row["probe_step"]),
                int(row["episode_index"]),
                row["candidate"],
            ): row
            for row in csv.DictReader(handle)
            if row["arm"] == "student_backbone"
        }
    errors = {"q": [], "target_h1": [], "target_h32": [], "reward_h32": [], "power_h32": []}
    rows = 0
    for (seed, split), dataset in datasets.items():
        if split != "mechanism_audit":
            continue
        for index, row in enumerate(dataset.rows):
            key = (
                seed,
                row["profile_id"],
                row["condition_id"],
                int(row["probe_step"]),
                int(row["episode_index"]),
                row["candidate"],
            )
            if key not in upstream:
                raise RuntimeError(f"R3-D2-A missing D1-C alignment key: {key}")
            source = upstream[key]
            errors["q"].append(abs(float(dataset.original_q[index]) - float(source["q_min"])))
            errors["target_h1"].append(
                abs(
                    float(dataset.n_step_targets[index, 0])
                    - float(source["deterministic_bootstrap_target_h1"])
                )
            )
            errors["target_h32"].append(
                abs(
                    float(dataset.n_step_targets[index, 31])
                    - float(source["deterministic_bootstrap_target_h32"])
                )
            )
            errors["reward_h32"].append(
                abs(
                    float(dataset.empirical_reward_returns[index, 31])
                    - float(source["empirical_reward_return_h32"])
                )
            )
            errors["power_h32"].append(
                abs(
                    float(dataset.empirical_power_returns[index, 31])
                    - float(source["empirical_true_power_return_h32"])
                )
            )
            rows += 1
    maxima = {name: max(items, default=0.0) for name, items in errors.items()}
    if rows != len(upstream) or max(maxima.values()) > tolerance:
        raise RuntimeError("R3-D2-A truth regeneration does not align with D1-C")
    return {
        "checked": True,
        "rows": rows,
        "max_abs_errors": maxima,
        "tolerance": tolerance,
        "pass": True,
    }


def _verify_split_seeds(settings: dict[str, Any], *, reserved: int) -> None:
    seen: dict[int, str] = {}
    for split_name, split in settings["splits"].items():
        for item in split["conditions"]:
            base = int(item["base_seed"])
            count = int(split["episodes_per_condition"])
            if base + count > reserved:
                raise RuntimeError("R3-D2-A data seed overlaps S4-D3 reservation")
            for seed in range(base, base + count):
                previous = seen.get(seed)
                if previous is not None and previous != split_name:
                    raise RuntimeError(f"R3-D2-A seed leakage: {previous} vs {split_name}")
                seen[seed] = split_name


def _effective_settings(experiment: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    d1c = _load_yaml(_project_path(experiment["upstream_d1c"]["experiment_config"]))
    d1c_settings = _d1c_effective_settings(d1c, quick=False)
    training = experiment["critic_training"]
    data = experiment["data"]
    settings: dict[str, Any] = {
        "quick": quick,
        "output_directory": experiment["outputs"]["directory"],
        "policy_seeds": [int(value) for value in experiment["frozen_policy"]["policy_seeds"]],
        "candidate_actions": deepcopy(experiment["candidate_actions"]),
        "targets": deepcopy(experiment["targets"]),
        "max_target_horizon": int(data["max_target_horizon"]),
        "episode_length_steps": int(data["episode_length_steps"]),
        "splits": {
            "development": {
                "profile_ids": list(data["profile_ids"]),
                "conditions": deepcopy(data["development"]["conditions"]),
                "probe_steps": [int(value) for value in data["probe_steps"]],
                "episodes_per_condition": int(data["development"]["episodes_per_condition"]),
                "episode_length_steps": int(data["episode_length_steps"]),
            },
            "validation": {
                "profile_ids": list(data["profile_ids"]),
                "conditions": deepcopy(data["validation"]["conditions"]),
                "probe_steps": [int(value) for value in data["probe_steps"]],
                "episodes_per_condition": int(data["validation"]["episodes_per_condition"]),
                "episode_length_steps": int(data["episode_length_steps"]),
            },
            "mechanism_audit": {
                "profile_ids": list(d1c_settings["profile_ids"]),
                "conditions": deepcopy(d1c_settings["diagnostic_conditions"]),
                "probe_steps": list(d1c_settings["probe_steps"]),
                "episodes_per_condition": int(d1c_settings["episodes_per_condition"]),
                "episode_length_steps": int(d1c_settings["episode_length_steps"]),
            },
        },
        "state_size": int(training["state_size"]),
        "action_size": int(training["action_size"]),
        "hidden_size": int(training["hidden_size"]),
        "learning_rate": float(training["learning_rate"]),
        "batch_size": int(training["batch_size"]),
        "maximum_updates": int(training["maximum_updates"]),
        "validation_interval_updates": int(training["validation_interval_updates"]),
        "log_interval_updates": int(training["log_interval_updates"]),
        "early_stopping_patience_updates": int(training["early_stopping_patience_updates"]),
        "initialization_seed_offset": int(training["initialization_seed_offset"]),
        "batch_order_seed_offset": int(training["batch_order_seed_offset"]),
        "actor_vs_zero_q_advantage_ci95_high": float(experiment["gate"]["actor_vs_zero_q_advantage_ci95_high"]),
        "min_online_q_excess_reduction_fraction": float(experiment["gate"]["min_online_q_excess_reduction_fraction"]),
        "min_pairwise_ranking_accuracy": float(experiment["gate"]["min_pairwise_ranking_accuracy"]),
        "multiple_comparison_alpha": float(experiment["gate"]["multiple_comparison_alpha"]),
        "expected_profile_count": len(data["profile_ids"]),
    }
    if quick:
        quick_config = experiment["quick"]
        settings["output_directory"] = experiment["outputs"]["quick_directory"]
        settings["policy_seeds"] = [int(value) for value in quick_config["policy_seeds"]]
        for split_name in ("development", "validation", "mechanism_audit"):
            split = quick_config[split_name]
            settings["splits"][split_name] = {
                "profile_ids": list(quick_config["profile_ids"]),
                "conditions": deepcopy(split["conditions"]),
                "probe_steps": [int(value) for value in quick_config["probe_steps"]],
                "episodes_per_condition": int(split["episodes_per_condition"]),
                "episode_length_steps": max(40, max(quick_config["probe_steps"]) + int(data["max_target_horizon"])),
            }
        for name in (
            "maximum_updates",
            "validation_interval_updates",
            "log_interval_updates",
            "early_stopping_patience_updates",
            "batch_size",
        ):
            settings[name] = int(quick_config[name])
        settings["expected_profile_count"] = len(quick_config["profile_ids"])
    return settings


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("R3-D2-A records must not be empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _flat_fit_record(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in item.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }


def _progress_runtime_fields(
    *,
    completed: int,
    total: int,
    started: float,
    device: torch.device,
) -> dict[str, float]:
    elapsed = max(time.perf_counter() - started, 1e-9)
    rate = completed / elapsed if completed > 0 else 0.0
    remaining = (total - completed) / rate if rate > 0 else float("nan")
    if device.type == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
    else:
        allocated = reserved = 0.0
    return {
        "elapsed_seconds": elapsed,
        "estimated_remaining_seconds": remaining,
        "cuda_allocated_gb": allocated,
        "cuda_reserved_gb": reserved,
    }
