"""用全新天气独立确认冻结的候选动作重选规则；不训练、不操作硬件。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import traceback

import torch

from src.rl.r4_action_response import physical_step
from src.rl.r4_baseline_selection import read
from src.rl.r4_baselines import guarded_correction, range_eligible
from src.rl.r4_candidate_reselection import tune_robust_margin_gate
from src.rl.r4_closed_loop import model_ensembles
from src.rl.r4_closed_loop_ranking import (
    AUDIT_METRICS,
    CANDIDATES,
    candidate_sequences,
    discounted_objective,
)
from src.rl.r4_control import NominalCalibration
from src.rl.r4_dynamics_experiment import Progress, safe_git_record, verify_hashes
from src.rl.r4_interface_smoke import write_json
from src.rl.r4_mpc import SearchConfig, sequence_scores
from src.rl.r4_observation import R4Interface, simulation_residual_proxy
from src.rl.r4_selected_anchor import anchor_parameters, load_selected, selected_plan
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _profiles, _project_path, _relative
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.robust_control import RobustnessCondition


SOURCES = "configs/experiments/s4_r4_candidate_confirmation_v1_sources.json"
BRANCHES = ("zero", "original_selected", "frozen_reselection")
FROZEN_RULE = "ensemble_mean_safe_gate"


def budget(quick: bool) -> dict[str, int]:
    """上限按完整批搜索计数；物理转移是精确值。"""
    batches, batch, probes, prefix = (1, 2, 1, 8) if quick else (18, 16, 4, 180)
    search = SearchConfig()
    planning = batches * (prefix + 1) * batch * search.population * search.iterations * search.horizon * 3
    panel = batches * probes * batch * len(CANDIDATES) * search.horizon * 3
    return {
        "physical_transitions": batches * batch * (prefix + probes * len(BRANCHES) * search.horizon),
        "max_model_forward_samples": planning + panel,
    }


def frozen_gate_choices(member_scores: torch.Tensor, threshold: float,
                        eligible: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """三成员等权均值选优；优势必须严格大于开发集冻结阈值。"""
    if (member_scores.ndim != 3 or member_scores.shape[-2:] != (len(CANDIDATES), 3)
            or not bool(torch.isfinite(member_scores).all()) or threshold < 0):
        raise ValueError("invalid frozen gate inputs")
    ensemble = member_scores.double().mean(-1)
    best = ensemble.argmax(-1)
    margin = ensemble.max(-1).values - ensemble[:, 0]
    choices = torch.where(margin > threshold, best, torch.zeros_like(best))
    if eligible is not None:
        if eligible.shape != choices.shape or eligible.dtype != torch.bool:
            raise ValueError("invalid eligibility mask")
        choices = torch.where(eligible, choices, torch.zeros_like(choices))
    return choices, margin


def _interval(values: torch.Tensor, *, seed: int, repeats: int) -> dict:
    """按三个湍流家族分层，对完整天气做配对自助法。"""
    if (values.ndim != 2 or values.shape[0] != 3 or values.shape[1] < 2
            or repeats < 1 or not bool(torch.isfinite(values).all())):
        raise ValueError("invalid weather-level confirmation statistic")
    generator = torch.Generator(device=values.device).manual_seed(seed)
    samples = values.new_zeros(repeats, dtype=torch.float64)
    for family in range(3):
        draw = torch.randint(values.shape[1], (repeats, values.shape[1]),
                             generator=generator, device=values.device)
        samples += values[family][draw].double().mean(1) / 3
    ci = torch.quantile(samples, values.new_tensor([.025, .975], dtype=torch.float64))
    return {
        "mean": float(values.double().mean()),
        "ci95": ci.tolist(),
        "family_means": [float(values[index].double().mean()) for index in range(3)],
    }


def summarize(audit_objective: torch.Tensor, measured_objective: torch.Tensor,
              metrics: torch.Tensor, choices: torch.Tensor, *, seed: int,
              repeats: int) -> dict:
    """[分支,家族,档位,天气,时刻]；档位和时刻先在天气内平均。"""
    expected = (len(BRANCHES), 3, 6, 16, 4)
    if (audit_objective.shape != expected or measured_objective.shape != expected
            or metrics.shape != expected + (len(AUDIT_METRICS),)
            or choices.shape != expected[1:]
            or any(not bool(torch.isfinite(item).all())
                   for item in (audit_objective, measured_objective, metrics))):
        raise ValueError("requires complete independent confirmation tensors")

    def weather(delta: torch.Tensor) -> torch.Tensor:
        return delta.double().mean((1, 3))

    comparisons = {}
    for number, (name, reference) in enumerate((
            ("frozen_reselection_vs_zero", 0),
            ("frozen_reselection_vs_original", 1))):
        audit_weather = weather(audit_objective[2] - audit_objective[reference])
        measured_weather = weather(measured_objective[2] - measured_objective[reference])
        row = {
            "audit_objective_gain": _interval(
                audit_weather, seed=seed + number * 100, repeats=repeats),
            "measured_objective_gain": _interval(
                measured_weather, seed=seed + number * 100 + 1, repeats=repeats),
            "audit_metric_delta": {},
        }
        for metric_index, metric_name in enumerate(AUDIT_METRICS):
            delta = weather(metrics[2, ..., metric_index] - metrics[reference, ..., metric_index])
            row["audit_metric_delta"][metric_name] = _interval(
                delta, seed=seed + number * 100 + 10 + metric_index, repeats=repeats)
        comparisons[name] = row

    primary = comparisons["frozen_reselection_vs_zero"]
    objective = primary["audit_objective_gain"]
    metric = primary["audit_metric_delta"]
    action_rate = float((choices != 0).double().mean())
    gates = {
        "audit_objective_ci_lower_positive": objective["ci95"][0] > 0,
        "all_family_audit_objective_means_positive": min(objective["family_means"]) > 0,
        "audit_power_mean_positive": metric["audit_power"]["mean"] > 0,
        "audit_strehl_mean_nonnegative": metric["audit_strehl"]["mean"] >= 0,
        "audit_violation_increase_at_most_point001": metric["audit_violation"]["mean"] <= .001,
        "audit_phase_rmse_mean_nonpositive": metric["audit_phase_rmse"]["mean"] <= 0,
        "nontrivial_action_rate": action_rate >= .05,
    }
    zero_power = float(metrics[0, ..., AUDIT_METRICS.index("audit_power")].double().mean())
    relative_power = (metric["audit_power"]["mean"] / zero_power) if zero_power > 0 else float("nan")
    passed = all(gates.values())
    return {
        "status": ("INDEPENDENT_LOCAL_RULE_CONFIRMATION_PASS_REQUIRES_AUDIT"
                   if passed else "INDEPENDENT_LOCAL_RULE_CONFIRMATION_FAIL_REQUIRES_AUDIT"),
        "frozen_rule": FROZEN_RULE,
        "branch_order": list(BRANCHES),
        "candidate_order": list(CANDIDATES),
        "independent_weather": 48,
        "weather_per_family": 16,
        "profiles_per_weather": 6,
        "probes_per_weather": 4,
        "primary_comparison": "frozen_reselection_vs_zero",
        "secondary_comparison_is_descriptive": "frozen_reselection_vs_original",
        "comparisons": comparisons,
        "action_rate": action_rate,
        "selection_counts": {name: int((choices == index).sum())
                             for index, name in enumerate(CANDIDATES)},
        "local_relative_audit_power_gain": relative_power,
        "confirmation_gates": gates,
        "all_confirmation_gates_pass": passed,
        "full_closed_loop_design_authorized": passed,
        "full_closed_loop_performance_established": False,
        "rl_authorized": False,
        "bootstrap_seed": seed,
        "bootstrap_repeats": repeats,
        "statistical_unit": "complete_weather; profiles_and_probes_averaged_within_weather",
        "practical_effect_warning": "局部八步分支通过也不等于完整闭环达到1%实用增益门槛。",
    }


def _validate_sources(cfg: dict) -> tuple[float, dict]:
    verify_hashes(cfg["source_files"])
    directory = _project_path(cfg["reselection_output"])
    summary = read(directory / "summary.json")
    if (summary["status"] != "CANDIDATE_RESELECTION_COMPLETE_REQUIRES_AUDIT"
            or read(directory / "SUCCESS.json")["summary_sha256"] != _file_sha256(directory / "summary.json")
            or _file_sha256(directory / "artifact_manifest.json") != summary["artifact_manifest_sha256"]
            or summary["analysis"]["status"] != "RESELECTION_RULE_WORTH_INDEPENDENT_CONFIRMATION"
            or summary["analysis"]["nominated_rules"] != [FROZEN_RULE]
            or summary["analysis"]["rl_authorized"]):
        raise RuntimeError("candidate reselection prerequisite is not the audited nomination")
    source = torch.load(_project_path(cfg["development_metrics"]), map_location="cpu", weights_only=True)
    tuned = tune_robust_margin_gate(
        source["model_member_scores"].double().mean(-1),
        source["audit_objective"].double(),
        torch.ones(3, 32, dtype=torch.bool),
    )
    if (tuned["threshold"] != cfg["frozen_threshold"]
            or min(tuned["train_family_gain"]) <= 0):
        raise RuntimeError("frozen full-development threshold cannot be reproduced")
    return float(tuned["threshold"]), tuned


def preflight(path: str | Path, quick: bool) -> tuple:
    own = read(_project_path(SOURCES))
    verify_hashes(own)
    config_path = _project_path(path)
    if _relative(config_path) not in own:
        raise ValueError("candidate-confirmation configuration is not frozen")
    cfg = _load_yaml(config_path)
    threshold, tuned = _validate_sources(cfg)
    spec, frozen = load_selected()
    frozen.update(cfg["source_files"])
    frozen.update(own)
    frozen[SOURCES] = _file_sha256(_project_path(SOURCES))
    parent = _load_yaml(_project_path(cfg["parent"]))
    expected_starts = [3780000, 3780256, 3780512]
    expected_budget = budget(False)
    if (cfg["formal_starts"] != expected_starts
            or cfg["quick_start"] != 3781000
            or cfg["weather_per_family"] != 16
            or cfg["batch_episodes"] != 16
            or cfg["probe_steps"] != [8, 40, 100, 180]
            or cfg["candidate_order"] != list(CANDIDATES)
            or cfg["branch_order"] != list(BRANCHES)
            or cfg["frozen_rule"] != FROZEN_RULE
            or cfg["frozen_threshold"] != threshold
            or cfg["training_updates"] != 0
            or cfg["automatic_retry"] is not False
            or any(cfg[key] != value for key, value in expected_budget.items())):
        raise ValueError("predeclared independent-confirmation design or budget changed")
    old_ranges = [(3762048 + offset, 3762048 + offset + 32)
                  for offset in parent["data"]["family_offsets"]]
    if any(any(low <= start < high or low < start + 16 <= high for low, high in old_ranges)
           for start in cfg["formal_starts"]):
        raise ValueError("confirmation weather overlaps development weather")
    if not quick:
        quick_output = _project_path(cfg["output_directory"] + "_quick")
        quick_summary = read(quick_output / "summary.json")
        if (quick_summary["status"] != "QUICK_COMPLETE_NO_CONCLUSION"
                or read(quick_output / "SUCCESS.json")["summary_sha256"] != _file_sha256(quick_output / "summary.json")
                or _file_sha256(quick_output / "artifact_manifest.json") != quick_summary["artifact_manifest_sha256"]
                or read(quick_output / "preflight.json")["frozen_files"].get(SOURCES)
                != _file_sha256(_project_path(SOURCES))):
            raise RuntimeError("same-version quick prerequisite missing")
        frozen.update(read(quick_output / "artifact_manifest.json"))
        for name in ("summary.json", "SUCCESS.json", "artifact_manifest.json"):
            frozen[_relative(quick_output / name)] = _file_sha256(quick_output / name)
    verify_hashes(frozen)
    output = _project_path(cfg["output_directory"] + ("_quick" if quick else ""))
    if output.exists():
        raise FileExistsError(f"preserve candidate-confirmation output: {output}")
    if shutil.disk_usage(_project_path(".")).free < (1024 if not quick else 128) * 1024**2:
        raise RuntimeError("insufficient output disk space")
    device = resolve_device("cuda")
    report = {
        "status": "READY_FOR_DIAGNOSTIC" if quick else "READY_FOR_USER_IDE",
        "quick": quick,
        "device": str(device),
        "frozen_files": frozen,
        "frozen_threshold": threshold,
        "threshold_reproduction": tuned,
        **budget(quick),
        "training_updates": 0,
        "real_slm_actions": False,
        "automatic_retry": False,
    }
    return cfg, parent, spec, output, report


@torch.no_grad()
def execute(cfg: dict, parent: dict, spec, output: Path, report: dict) -> dict:
    device = resolve_device("cuda")
    quick = report["quick"]
    batch, episode_length = (2, 24) if quick else (16, 200)
    families = parent["families"][:1] if quick else parent["families"]
    profile_ids = parent["profile_ids"][:1] if quick else parent["profile_ids"]
    profiles = _profiles(parent, profile_ids)
    starts = [cfg["quick_start"]] if quick else cfg["formal_starts"]
    probes = [8] if quick else cfg["probe_steps"]
    weather = batch
    base, _ = load_s1_config(_project_path(parent["environment_config"]))
    base = replace(base, num_modes=21, batch_size=batch, episode_length=episode_length)
    basis, _, _ = build_action_basis(
        base, ActionRepresentation("r4_zernike21", "zernike", 21), device)
    bounds = read(_project_path("outputs/s4_r4_baseline_safety_v1/calibration.json"))
    calibration = NominalCalibration(**parent["nominal_calibration"])
    models = model_ensembles(device)["gru_mpc"]
    search = SearchConfig()
    anchor = anchor_parameters(spec)
    shape = (len(BRANCHES), len(families), len(profiles), weather, len(probes))
    audit_objective = torch.zeros(shape, dtype=torch.float64, device=device)
    measured_objective = torch.zeros_like(audit_objective)
    metrics = torch.zeros(shape + (len(AUDIT_METRICS),), dtype=torch.float64, device=device)
    choices = torch.zeros(shape[1:], dtype=torch.long, device=device)
    progress = Progress(output, device)
    progress.phase("独立天气局部闭环确认" if not quick else "独立确认快速冒烟（无结论）",
                   report["physical_transitions"] // batch)
    physical = calls = 0
    active: dict = {}
    manifest = []
    (output / "branches").mkdir()

    def tick() -> None:
        nonlocal physical
        physical += batch
        progress.tick({"物理转移": physical, "模型前向": calls})

    try:
        for family_index, family in enumerate(families):
            for profile_index, profile in enumerate(profiles):
                seed = starts[family_index]
                active = {"family": family_index, "profile": profile.identifier,
                          "seed": seed, "step": 0}
                condition = RobustnessCondition.from_mapping(dict(family, base_seed=seed))
                env = AdaptiveOpticsEnv(
                    profile.environment_config(condition.environment_config(base)), device,
                    profile.effects_config(), basis_override=basis)
                raw, _ = env.reset(seed=seed)
                sensor = torch.Generator(device=device).manual_seed(
                    seed + parent["data"]["sensor_seed_offset"])
                interface = R4Interface(calibration=calibration)
                interface.reset(simulation_residual_proxy(
                    raw, generator=sensor, noise_std_rad=profile.observation_noise_std_rad),
                    episode_id=str(seed))
                probe_records = []
                for step in range(max(probes) + 1):
                    active["step"] = step
                    snapshot = interface.snapshot()
                    h, valid = snapshot.features, snapshot.valid
                    eligible = range_eligible(h, valid, bounds)
                    sequence = h.new_zeros(batch, search.horizon, 11)
                    plan_calls = 0
                    plan_seed = (cfg["search_seed"] + family_index * 100000
                                 + profile_index * 10000 + step)
                    if bool(eligible.any()):
                        first = int(torch.nonzero(eligible)[0, 0])
                        plan_h = torch.where(eligible[:, None, None], h, h[first:first + 1])
                        plan_v = torch.where(eligible[:, None], valid, valid[first:first + 1])
                        planned = selected_plan(models, plan_h, plan_v, spec, calibration,
                                                search, seed=plan_seed)
                        sequence[eligible] = planned["sequence"][eligible]
                        plan_calls = int(planned["model_forward_samples"])
                    correction, reasons = guarded_correction(h, valid, sequence[:, 0], bounds)
                    accepted = torch.tensor([reason == "accepted" for reason in reasons],
                                            device=device, dtype=torch.bool)
                    sequence = torch.where(accepted[:, None, None], sequence,
                                           torch.zeros_like(sequence))
                    if not torch.equal(sequence[:, 0], correction):
                        raise RuntimeError("guarded first action and branch sequence differ")
                    calls += plan_calls

                    if step in probes:
                        panel = candidate_sequences(sequence)
                        member_scores = torch.stack([
                            sequence_scores([model], h, valid, panel, anchor, calibration, search)
                            for model in models
                        ], dim=-1)
                        calls += batch * len(CANDIDATES) * search.horizon * len(models)
                        selected, margin = frozen_gate_choices(
                            member_scores, cfg["frozen_threshold"], eligible=accepted)
                        selected_sequence = panel.gather(
                            1, selected[:, None, None, None].expand(-1, 1, search.horizon, 11)
                        ).squeeze(1)
                        branch_sequences = torch.stack(
                            (panel[:, 0], panel[:, 3], selected_sequence), dim=1)
                        branch_results = []
                        for branch_index, branch_name in enumerate(BRANCHES):
                            branch_env, branch_interface, branch_sensor = deepcopy(
                                (env, interface, sensor))
                            rows = []
                            for horizon_step in range(search.horizon):
                                rows.append(physical_step(
                                    branch_env, branch_interface, branch_sensor, profile,
                                    branch_sequences[:, branch_index, horizon_step], anchor))
                                tick()
                            stacked = {key: torch.stack([row[key] for row in rows], 1)
                                       for key in rows[0]}
                            if any(not bool(value.isfinite().all()) for value in stacked.values()):
                                raise RuntimeError("nonfinite confirmation branch")
                            branch_results.append({"branch": branch_name, **stacked})
                        measured = torch.stack([row["power"] for row in branch_results], 1)
                        audit = torch.stack([row["audit_power"] for row in branch_results], 1)
                        measured_score = discounted_objective(measured, branch_sequences, search)
                        audit_score = discounted_objective(audit, branch_sequences, search)
                        probe_index = probes.index(step)
                        audit_objective[:, family_index, profile_index, :, probe_index] = audit_score.T.double()
                        measured_objective[:, family_index, profile_index, :, probe_index] = measured_score.T.double()
                        choices[family_index, profile_index, :, probe_index] = selected
                        for branch_index, row in enumerate(branch_results):
                            metrics[branch_index, family_index, profile_index, :, probe_index] = torch.stack([
                                row[name].double().mean(1) for name in AUDIT_METRICS], -1)
                        probe_records.append({
                            "step": step,
                            "plan_seed": plan_seed,
                            "eligible": eligible.cpu(),
                            "guard_reasons": reasons,
                            "candidate_sequences": panel.cpu(),
                            "model_member_scores": member_scores.cpu(),
                            "choice": selected.cpu(),
                            "margin": margin.cpu(),
                            "branch_sequences": branch_sequences.cpu(),
                            "measured_objective": measured_score.cpu(),
                            "audit_objective": audit_score.cpu(),
                            "branches": [{key: value.cpu() if torch.is_tensor(value) else value
                                          for key, value in row.items()} for row in branch_results],
                        })
                        if not torch.equal(interface.snapshot().features, snapshot.features):
                            raise RuntimeError("physical branches mutated base state")

                    if step < max(probes):
                        physical_step(env, interface, sensor, profile, correction, anchor)
                        tick()
                target = output / "branches" / f"family_{family_index}_{profile.identifier}_{seed}.pt"
                torch.save({"family": family_index, "profile": profile.identifier,
                            "seed": seed, "weather_seeds": env.episode_seeds.cpu(),
                            "probes": probe_records}, target)
                manifest.append({"file": _relative(target), "sha256": _file_sha256(target)})
        expected = budget(quick)
        if physical != expected["physical_transitions"] or calls > expected["max_model_forward_samples"]:
            raise RuntimeError("independent-confirmation budget mismatch")
        write_json(output / "branch_manifest.json", manifest)
        if not quick:
            torch.save({
                "audit_objective": audit_objective.cpu(),
                "measured_objective": measured_objective.cpu(),
                "metrics": metrics.cpu(),
                "choices": choices.cpu(),
                "branch_order": BRANCHES,
                "candidate_order": CANDIDATES,
                "audit_metric_order": AUDIT_METRICS,
                "frozen_threshold": cfg["frozen_threshold"],
            }, output / "metrics.pt")
        analysis = {} if quick else summarize(
            audit_objective, measured_objective, metrics, choices,
            seed=cfg["bootstrap_seed"], repeats=cfg["bootstrap_repeats"])
        return {
            "status": ("QUICK_COMPLETE_NO_CONCLUSION" if quick
                       else "CANDIDATE_CONFIRMATION_COMPLETE_REQUIRES_AUDIT"),
            "analysis": analysis,
            "completed_batches": len(manifest),
            "physical_transitions": physical,
            "model_forward_samples": calls,
            "max_model_forward_samples": expected["max_model_forward_samples"],
        }
    except Exception:
        write_json(output / "interrupted_context.json", {
            "active": active,
            "physical_transitions": physical,
            "model_forward_samples": calls,
        })
        raise
    finally:
        progress.close()


def run(path: str | Path, *, quick: bool = False, preflight_only: bool = False) -> dict:
    cfg, parent, spec, output, report = preflight(path, quick)
    if preflight_only:
        return {key: value for key, value in report.items() if key != "frozen_files"}
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
        raise ValueError("unsupported CUDA determinism setting")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "preflight.json", report)
    write_json(output / "config.json", cfg)
    write_json(output / "runtime.json", {
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(),
        "git": safe_git_record(),
    })
    try:
        result = execute(cfg, parent, spec, output, report)
        verify_hashes(report["frozen_files"])
        write_json(output / "source_manifest.json", report["frozen_files"])
        write_json(output / "artifact_manifest.json", {
            _relative(item): _file_sha256(item)
            for item in output.rglob("*") if item.is_file()
        })
        result.update({
            "training_updates": 0,
            "independent_weather": 0 if quick else 48,
            "real_slm_actions": False,
            "automatic_retry": False,
            "rl_authorized": False,
            "artifact_manifest_sha256": _file_sha256(output / "artifact_manifest.json"),
            "material_passport": {
                "origin_skill": "academic-research-suite",
                "origin_mode": "run",
                "origin_date": datetime.now(timezone.utc).isoformat(),
                "verification_status": "REQUIRES_AUDIT",
                "version_label": "r4_candidate_confirmation_v1",
            },
            "next_action": ("停止；快速冒烟不产生科学结论。" if quick else
                            "停止并通知助手只读审计；不自动进入完整闭环、训练或RL。"),
        })
        write_json(output / "summary.json", result)
        write_json(output / "SUCCESS.json", {
            "summary_sha256": _file_sha256(output / "summary.json")})
        return result
    except Exception:
        write_json(output / "failure.json", {
            "traceback": traceback.format_exc(), "automatic_retry": False})
        raise
