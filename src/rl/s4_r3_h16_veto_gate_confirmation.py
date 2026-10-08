"""A10冻结收益模型的有害动作否决开关独立确认。"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np
import torch

from src.rl.s4_r3_h16_data_scaling import _collect_and_save, verify_dataset_separation
from src.rl.s4_r3_h16_fixed_time_analysis import _load_model
from src.rl.s4_r3_h16_fixed_time_training import episode_groups, predictions, validate_data, write_json, write_rows
from src.rl.s4_r3_h16_gradient_conflict import _model_digest
from src.rl.s4_r3_h16_head_split import _git_record_utf8
from src.rl.s4_r3_h16_veto_gate_contract import BOUNDARY, SPLITS, budget, load_contract, preflight, safe_path, verify_manifests
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _project_path, _relative, _runtime_record
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.training_progress import counted_progress, progress_message, update_progress


def gate_decision(model_predictions: torch.Tensor, *, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    if model_predictions.ndim != 2 or model_predictions.shape[0] < 1:
        raise ValueError("A10 predictions must be model by action-pair")
    if not bool(torch.isfinite(model_predictions).all()):
        raise RuntimeError("A10 non-finite gate prediction")
    mean_prediction = model_predictions.double().mean(dim=0)
    return mean_prediction, mean_prediction > threshold


def episode_decisions(*, policy_seed: int, split: str, data: Any,
                      mean_prediction: torch.Tensor, release: torch.Tensor,
                      actor_normalized_bound_violation: torch.Tensor) -> list[dict]:
    if mean_prediction.shape != data.reward_delta.shape or release.shape != data.reward_delta.shape:
        raise ValueError("A10 gate/data shape mismatch")
    if (actor_normalized_bound_violation.shape != release.shape
            or actor_normalized_bound_violation.dtype != torch.bool):
        raise ValueError("A10 action-bound audit shape mismatch")
    rows = []
    for condition, episode_seed, indices in episode_groups(data):
        ix = torch.tensor(indices, dtype=torch.long)
        reward = data.reward_delta[ix].double()
        power = data.power_delta[ix].double()
        choose = release[ix]
        rows.append(dict(
            policy_seed=policy_seed, split=split, condition_id=condition,
            condition_family=condition.split("_")[-1], episode_seed=episode_seed,
            pairs=len(indices), actor_release_fraction=float(choose.double().mean()),
            mean_predicted_reward_delta=float(mean_prediction[ix].mean()),
            gate_reward_vs_zero=float((reward * choose).mean()),
            gate_power_vs_zero=float((power * choose).mean()),
            always_actor_reward_vs_zero=float(reward.mean()),
            always_actor_power_vs_zero=float(power.mean()),
            gate_reward_vs_always_actor=float((reward * choose).mean() - reward.mean()),
            always_actor_normalized_bound_violation_rate=float(
                actor_normalized_bound_violation[ix].double().mean()),
            gate_normalized_bound_violation_rate=float(
                (actor_normalized_bound_violation[ix] & choose).double().mean()),
        ))
    return rows


def comparison_rows(rows: list[dict], s: dict, *, split: str) -> tuple[list[dict], dict]:
    metrics = tuple(s["statistics"]["comparisons_per_policy"])
    family_size = int(s["statistics"]["family_size"])
    alpha = float(s["statistics"]["familywise_alpha"])
    replicates = int(s["statistics"]["bootstrap_replicates"])
    result = []
    policy_pass: list[int] = []
    condition_safe: list[int] = []
    for policy in s["policy_seeds"]:
        selected = [r for r in rows if r["policy_seed"] == policy and r["split"] == split]
        expected = len(next(iter(s["splits"].values()))["conditions"]) * s["splits"][split]["episodes_per_condition"]
        keys = [(r["condition_id"], r["episode_seed"]) for r in selected]
        if len(selected) != expected or len(set(keys)) != expected:
            raise RuntimeError("A10 incomplete or duplicate episode decisions")
        strata: dict[str, list[int]] = {}
        for index, row in enumerate(selected):
            strata.setdefault(row["condition_family"], []).append(index)
        rng = np.random.default_rng(812000 + policy + (0 if split == "confirmation_id" else 10000))
        weights = np.zeros((replicates, len(selected)), dtype=np.float64)
        for indices in strata.values():
            weights[:, indices] = rng.multinomial(
                len(indices), np.full(len(indices), 1 / len(indices)), size=replicates)
        weights /= len(selected)
        passed_metrics = []
        for metric in metrics:
            values = np.asarray([r[metric] for r in selected], dtype=np.float64)
            if not np.isfinite(values).all():
                raise RuntimeError("A10 non-finite episode outcome")
            draws = weights @ values
            tail = alpha / (2 * family_size)
            q = np.quantile(draws, [0.025, 0.975, tail, 1 - tail])
            passed = bool(q[2] > 0)
            passed_metrics.append(passed)
            result.append(dict(policy_seed=policy, split=split, metric=metric,
                               estimate=float(values.mean()), ci95_low=float(q[0]), ci95_high=float(q[1]),
                               familywise_ci_low=float(q[2]), familywise_ci_high=float(q[3]),
                               episodes=len(values), replicates=replicates, family_size=family_size,
                               familywise_alpha=alpha, unit="complete_episode_seed",
                               stratified_by="condition_family", passes_positive_gate=passed))
        conditions_ok = all(
            np.mean([r[metric] for r in selected if r["condition_family"] == family]) >= 0
            for family in strata for metric in ("gate_reward_vs_zero", "gate_power_vs_zero"))
        if conditions_ok:
            condition_safe.append(policy)
        if all(passed_metrics) and conditions_ok:
            policy_pass.append(policy)
    minimum = int(s["statistics"]["minimum_supporting_policy_seeds"])
    if s["quick"]:
        status = "QUICK_SMOKE_ONLY"
    elif split == s["statistics"]["primary_split"]:
        status = (s["statistics"]["pass_status"] if len(policy_pass) >= minimum
                  else s["statistics"]["fail_status"])
    else:
        status = "SHIFTED_ROBUSTNESS_PASS" if len(policy_pass) >= minimum else "SHIFTED_ROBUSTNESS_FAIL"
    return result, dict(status=status, passing_policy_seeds=policy_pass,
                        condition_safe_policy_seeds=condition_safe,
                        required_policy_seeds=minimum,
                        requires_all_metrics_positive_and_each_condition_nonnegative=True)


def actor_normalized_action_bound_violations(
    raw_path: Path,
    data: Any,
    *,
    normalized_limit: float = 1.0,
) -> torch.Tensor:
    """检查保存的演员修正是否越过归一化边界，而不是误当作弧度。"""
    if normalized_limit <= 0:
        raise ValueError("A10 normalized action limit must be positive")
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    actor = {}
    for index, row in enumerate(raw["rows"]):
        if row["candidate"] != "actor":
            continue
        key = (str(row["profile_id"]), str(row["condition_id"]), int(row["probe_step"]),
               int(row["episode_index"]), int(row["episode_seed"]))
        if key in actor:
            raise RuntimeError("A10 duplicate actor action")
        actor[key] = raw["actions"][index]
    result = []
    for row in data.rows:
        key = (str(row["profile_id"]), str(row["condition_id"]), int(row["probe_step"]),
               int(row["episode_index"]), int(row["episode_seed"]))
        if key not in actor:
            raise RuntimeError("A10 missing actor action for bound audit")
        result.append(bool((actor[key].abs() > normalized_limit + 1e-7).any()))
    return torch.tensor(result, dtype=torch.bool)


def evaluate_gate(models: dict[int, list[dict]], data: dict[int, dict], violations: dict[int, dict], s: dict,
                  output: Path, device: torch.device) -> dict:
    decision_rows, integrity = [], []
    total = len(s["policy_seeds"]) * len(SPLITS)
    bar = counted_progress(total=total, description="A10 冻结开关评估", unit="组")
    start = time.perf_counter()
    with (output / "evaluation_progress.jsonl").open("w", encoding="utf-8") as log:
        for policy in s["policy_seeds"]:
            for split in SPLITS:
                dataset = data[policy][split]
                values = []
                for rec in models[policy]:
                    model, checkpoint = _load_model(rec, device)
                    before = _model_digest(model)
                    value, _ = predictions(model, dataset, checkpoint["normalization"], device)
                    if value is None:
                        raise RuntimeError("A10 selected model has no reward prediction")
                    values.append(value)
                    after = _model_digest(model)
                    untouched = all(p.grad is None for p in model.parameters())
                    if before != after or not untouched:
                        raise RuntimeError("A10 evaluation modified a frozen model")
                    integrity.append(dict(policy_seed=policy, replicate=rec["replicate"], split=split,
                                          before=before, after=after,
                                          parameters_unchanged=True, grad_fields_untouched=True))
                model_values = torch.stack(values)
                mean_value, release = gate_decision(model_values, threshold=s["gate"]["threshold"])
                decision_rows.extend(episode_decisions(policy_seed=policy, split=split, data=dataset,
                                                       mean_prediction=mean_value, release=release,
                                                       actor_normalized_bound_violation=violations[policy][split]))
                torch.save(dict(policy_seed=policy, split=split, model_reward_predictions=model_values,
                                ensemble_reward_prediction=mean_value, actor_released=release,
                                reward_delta=dataset.reward_delta, power_delta=dataset.power_delta,
                                rows=dataset.rows, **BOUNDARY),
                           output / f"policy_{policy}_{split}_gate_predictions.pt")
                bar.update(1)
                record = dict(policy_seed=policy, split=split, completed=bar.n, total=bar.total,
                              elapsed_seconds=time.perf_counter() - start,
                              estimated_remaining_seconds=(time.perf_counter() - start) / bar.n * (bar.total - bar.n),
                              cuda_allocated_gb=torch.cuda.memory_allocated(device) / 1024 ** 3,
                              cuda_reserved_gb=torch.cuda.memory_reserved(device) / 1024 ** 3)
                log.write(json.dumps(record, ensure_ascii=False) + "\n"); log.flush()
                update_progress(bar, device=device, metrics={"放行比例": float(release.float().mean())})
    bar.close()
    primary, primary_decision = comparison_rows(decision_rows, s, split="confirmation_id")
    shifted, shifted_decision = comparison_rows(decision_rows, s, split="confirmation_shift")
    write_rows(output / "episode_gate_outcomes.csv", decision_rows)
    write_rows(output / "primary_comparisons.csv", primary)
    write_rows(output / "shifted_comparisons.csv", shifted)
    return dict(episode_decision_records=len(decision_rows), model_evaluations=len(integrity),
                parameter_integrity=integrity, primary_comparisons=primary,
                shifted_comparisons=shifted, interpretation=primary_decision,
                shifted_interpretation=shifted_decision)


def run_s4_r3_h16_veto_gate_confirmation(config_path: str | Path, *, quick: bool = False,
                                          preflight_only: bool = False,
                                          quick_run_tag: str | None = None) -> dict:
    cfg, settings = load_contract(config_path, quick=quick, quick_run_tag=quick_run_tag)
    report, physical, policies, models = preflight(config_path, cfg, settings)
    if preflight_only:
        return report
    device = resolve_device(cfg["runtime"]["device"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    output = safe_path(settings["output_directory"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "preflight.json", report)
    write_json(output / "effective_config.json", settings)
    write_json(output / "input_manifest.json", report["frozen_input_hashes"])
    write_json(output / "source_manifest.json", report["frozen_source_hashes"])
    try:
        return _execute(settings, report, physical, policies, models, output, device)
    except Exception as exc:
        write_json(output / "failure.json", dict(exception=type(exc).__name__, message=str(exc),
                   traceback=traceback.format_exc(), automatic_retry=False, **BOUNDARY))
        raise


def _execute(s: dict, report: dict, physical: dict, policies: list[dict], models: dict[int, list[dict]],
             output: Path, device: torch.device) -> dict:
    started = time.perf_counter()
    base, _ = load_s1_config(_project_path(physical["environment_config"]))
    representation = ActionRepresentation.from_mapping(physical["representation"])
    base = replace(base, num_modes=representation.num_modes)
    basis, _, diagnostics = build_action_basis(base, representation, device)
    dataset_dir = output / "datasets"; dataset_dir.mkdir()
    data = {p: {} for p in s["policy_seeds"]}
    violations = {p: {} for p in s["policy_seeds"]}
    records = []
    collection = counted_progress(total=budget(s)["collection_branches"], description="A10 全新确认回合", unit="分支")
    collection_start = time.perf_counter()
    for policy in policies:
        seed = int(policy["policy_seed"])
        for split in SPLITS:
            progress_message(f"A10 采集策略{seed}/{split}")
            pairs, record = _collect_and_save(
                name=split, split=s["splits"][split], checkpoint=policy,
                physical_experiment=physical, base_config=base, basis=basis, settings=s,
                bar=collection, progress_path=output / "collection_progress.jsonl",
                collection_started=collection_start, dataset_directory=dataset_dir, device=device)
            validate_data(pairs, s["splits"][split], seed, collection_record=record)
            data[seed][split] = pairs
            violations[seed][split] = actor_normalized_action_bound_violations(
                _project_path(record["path"]), pairs)
            record["actor_normalized_action_limit"] = 1.0
            record["actor_normalized_bound_violation_count"] = int(
                violations[seed][split].sum())
            record["actor_normalized_bound_violation_rate"] = float(
                violations[seed][split].float().mean())
            record["residual_action_limit_rad"] = float(
                physical["action"]["residual_action_limit_rad"])
            record["correction_component_limit_rad"] = float(
                physical["action"]["correction_component_limit_rad"])
            verify_dataset_separation(data[seed])
            record["physical_split_config"] = s["splits"][split]
            records.append(record)
            write_json(output / "data_manifest.json", dict(records=records))
    collection.close()
    if len(records) != budget(s)["dataset_files"] or collection.n != collection.total:
        raise RuntimeError("A10 collection incomplete")
    write_json(output / "DATA_FROZEN.json", dict(records=records, created_at=datetime.now(timezone.utc).isoformat(), **BOUNDARY))
    verify_manifests(report)
    evaluation = evaluate_gate(models, data, violations, s, output, device)
    verify_manifests(report)
    for record in records:
        if _file_sha256(_project_path(record["path"])) != record["sha256"]:
            raise RuntimeError("A10 collected dataset changed")
    expected = budget(s)
    if evaluation["episode_decision_records"] != expected["episode_decision_records"]:
        raise RuntimeError("A10 decision record budget mismatch")
    file_hashes = {_relative(p): _file_sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    result = dict(
        material_passport=dict(origin_skill="academic-research-suite / experiment-agent",
                               origin_mode="run", origin_date=datetime.now(timezone.utc).isoformat(),
                               verification_status="UNVERIFIED", version_label="s4d2_r3_d2a10_veto_gate_v1"),
        experiment=dict(id="AO-S4-D2-R3-D2-A10-VETO-GATE-CONFIRMATION", quick=s["quick"],
                        status="completed_pending_audit", device=str(device),
                        gpu_name=torch.cuda.get_device_name(device),
                        duration_seconds=time.perf_counter() - started),
        budget=expected, evaluation={k: v for k, v in evaluation.items() if k not in ("interpretation", "shifted_interpretation")},
        interpretation=evaluation["interpretation"], shifted_interpretation=evaluation["shifted_interpretation"],
        basis_diagnostics=diagnostics, evidence_boundary=dict(new_data_generated=True,
            a9_samples_loaded=False, frozen_models_evaluated=expected["frozen_models"], **BOUNDARY),
        runtime=_runtime_record(), git=_git_record_utf8(), records=file_hashes,
        next_action="停止并等待用户通知与只读审计；不要自动重跑、改阈值或进入完整RL。")
    write_json(output / "summary.json", result)
    write_json(output / "SUCCESS.json", dict(status="COMPLETED_PENDING_AUDIT", quick=s["quick"],
               summary_sha256=_file_sha256(output / "summary.json"), **BOUNDARY))
    return result
