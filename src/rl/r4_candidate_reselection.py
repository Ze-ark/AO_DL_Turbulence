"""在既有五候选表上交叉拟合安全重选规则；不训练、不重跑仿真。"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import traceback

import torch

from src.rl.r4_baseline_selection import read
from src.rl.r4_dynamics_experiment import safe_git_record, verify_hashes
from src.rl.r4_interface_smoke import write_json
from src.rl.s4_training import _file_sha256, _load_yaml, _project_path, _relative
from src.runtime import resolve_device


SOURCES = "configs/experiments/s4_r4_candidate_reselection_v1_sources.json"
CANDIDATES = ("zero", "negative", "half", "selected", "scaled_1p5")
AUDIT_METRICS = ("audit_power", "audit_strehl", "audit_violation", "audit_phase_rmse")
PRIMARY_RULES = ("conservative_safe_gate", "ensemble_mean_safe_gate")


def stratified_fold_assignment(*, seed: int, families: int = 3,
                               weather: int = 32, folds: int = 4) -> torch.Tensor:
    """每个湍流家族独立随机分层，每折得到相同数量的完整天气。"""
    if families < 1 or weather < folds or folds < 2 or weather % folds:
        raise ValueError("weather must divide evenly into at least two folds")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    assignment = torch.empty(families, weather, dtype=torch.long)
    labels = torch.arange(weather) % folds
    for family in range(families):
        assignment[family, torch.randperm(weather, generator=generator)] = labels
    return assignment


def _weather_mean(values: torch.Tensor) -> torch.Tensor:
    """把档位和探测时刻留在完整天气内平均。"""
    result = values.double().mean((1, 3))
    if result.shape != (3, 32) or not bool(torch.isfinite(result).all()):
        raise ValueError("invalid weather-level values")
    return result


def _gather(values: torch.Tensor, choices: torch.Tensor) -> torch.Tensor:
    if values.shape[:-1] != choices.shape:
        raise ValueError("choice and candidate arrays are not aligned")
    return values.gather(-1, choices[..., None]).squeeze(-1)


def tune_robust_margin_gate(score: torch.Tensor, actual: torch.Tensor,
                            train_weather: torch.Tensor) -> dict:
    """最大化训练天气中最弱家族收益；完全相同时偏向更严格的回退阈值。"""
    if (score.shape != (3, 6, 32, 4, 5) or actual.shape != score.shape
            or train_weather.shape != (3, 32) or train_weather.dtype != torch.bool
            or bool((train_weather.sum(1) == 0).any())
            or any(not bool(torch.isfinite(x).all()) for x in (score, actual))):
        raise ValueError("invalid margin-gate inputs")
    best = score.argmax(-1)
    margin = score.max(-1).values - score[..., 0]
    # 大于最大优势时所有状态都回退零动作，因此零动作也被显式纳入选择。
    train_state = train_weather[:, None, :, None].expand_as(margin)
    thresholds = torch.unique(torch.cat((
        margin[train_state], margin.new_zeros(1)))).sort().values
    thresholds = torch.cat((thresholds, torch.nextafter(
        thresholds[-1:], torch.full_like(thresholds[-1:], float("inf")))))
    threshold_view = thresholds.reshape(-1, 1, 1, 1, 1)
    candidate = torch.where(margin[None] > threshold_view, best[None],
                            torch.zeros_like(best)[None])
    expanded_actual = actual[None].expand(thresholds.numel(), *actual.shape)
    selected_actual = expanded_actual.gather(-1, candidate[..., None]).squeeze(-1)
    weather_gain = (selected_actual - actual[..., 0][None]).double().mean((2, 4))
    counts = train_weather.sum(1).to(weather_gain.dtype)
    family_gain = (weather_gain * train_weather[None]).sum(2) / counts[None]
    worst = family_gain.min(1).values
    eligible = worst == worst.max()
    overall = family_gain.mean(1)
    eligible &= overall == overall[eligible].max()
    # thresholds已升序；最终相同时选最后一个，即最严格的安全回退。
    selected_index = int(torch.where(eligible)[0][-1])
    selected_choice = candidate[selected_index]
    return {"threshold": float(thresholds[selected_index]),
            "train_family_gain": family_gain[selected_index].tolist(),
            "train_mean_gain": float(overall[selected_index]),
            "train_action_rate": float((selected_choice[train_state] != 0).double().mean())}


def crossfit_margin_gate(score: torch.Tensor, actual: torch.Tensor,
                         assignment: torch.Tensor) -> tuple[torch.Tensor, list[dict]]:
    """每折阈值仅用其余天气拟合，返回完整的折外动作选择。"""
    if assignment.shape != (3, 32):
        raise ValueError("invalid fold assignment")
    folds = int(assignment.max()) + 1
    if set(assignment.unique().tolist()) != set(range(folds)):
        raise ValueError("fold identifiers must be contiguous")
    best = score.argmax(-1)
    margin = score.max(-1).values - score[..., 0]
    choices = torch.full(best.shape, -1, dtype=torch.long, device=best.device)
    records = []
    for fold in range(folds):
        test_weather = assignment == fold
        train_weather = ~test_weather
        tuned = tune_robust_margin_gate(score, actual, train_weather.to(score.device))
        candidate = torch.where(margin > tuned["threshold"], best, torch.zeros_like(best))
        mask = test_weather[:, None, :, None].to(score.device).expand_as(best)
        choices[mask] = candidate[mask]
        records.append({"fold": fold, **tuned,
                        "train_weather": int(train_weather.sum()),
                        "test_weather": int(test_weather.sum())})
    if bool((choices < 0).any()):
        raise RuntimeError("cross-fitting left states without a decision")
    return choices, records


def _interval(values: torch.Tensor, *, seed: int, repeats: int,
              confidence: float) -> dict:
    if (values.shape != (3, 32) or repeats < 1 or not .5 < confidence < 1
            or not bool(torch.isfinite(values).all())):
        raise ValueError("invalid bootstrap values")
    generator = torch.Generator(device=values.device).manual_seed(seed)
    samples = values.new_zeros(repeats, dtype=torch.float64)
    for family in range(3):
        draw = torch.randint(32, (repeats, 32), device=values.device, generator=generator)
        samples += values[family][draw].double().mean(1) / 3
    alpha = (1 - confidence) / 2
    quantiles = torch.tensor([alpha, 1 - alpha], device=values.device, dtype=torch.float64)
    ci = torch.quantile(samples, quantiles)
    return {"mean": float(values.mean()), "ci": ci.tolist(), "confidence": confidence,
            "family_means": [float(values[index].mean()) for index in range(3)]}


def _rule_statistics(name: str, choices: torch.Tensor, measured: torch.Tensor,
                     audit: torch.Tensor, metrics: torch.Tensor, *, seed: int,
                     repeats: int) -> dict:
    zero_measured = measured[..., 0]
    zero_audit = audit[..., 0]
    result = {
        "measured_objective_gain": _interval(
            _weather_mean(_gather(measured, choices) - zero_measured),
            seed=seed, repeats=repeats, confidence=.95),
        "audit_objective_gain": _interval(
            _weather_mean(_gather(audit, choices) - zero_audit),
            seed=seed + 1, repeats=repeats, confidence=.95),
        "audit_objective_gain_familywise": _interval(
            _weather_mean(_gather(audit, choices) - zero_audit),
            seed=seed + 2, repeats=repeats, confidence=.975),
        "action_rate": float((choices != 0).double().mean()),
        "selection_counts": {candidate: int((choices == index).sum())
                             for index, candidate in enumerate(CANDIDATES)},
    }
    metric_results = {}
    for metric_index, metric_name in enumerate(AUDIT_METRICS):
        candidate_values = metrics[..., metric_index]
        delta = _gather(candidate_values, choices) - candidate_values[..., 0]
        metric_results[metric_name] = _interval(
            _weather_mean(delta), seed=seed + 10 + metric_index,
            repeats=repeats, confidence=.95)
    result["audit_metric_delta"] = metric_results
    if name in PRIMARY_RULES:
        objective = result["audit_objective_gain_familywise"]
        metric = result["audit_metric_delta"]
        checks = {
            "objective_ci_lower_positive": objective["ci"][0] > 0,
            "all_family_objective_means_positive": min(objective["family_means"]) > 0,
            "audit_power_mean_positive": metric["audit_power"]["mean"] > 0,
            "audit_strehl_mean_nonnegative": metric["audit_strehl"]["mean"] >= 0,
            "audit_violation_increase_at_most_point001": metric["audit_violation"]["mean"] <= .001,
            "audit_phase_rmse_mean_nonpositive": metric["audit_phase_rmse"]["mean"] <= 0,
            "nontrivial_action_rate": result["action_rate"] >= .05,
        }
        result["nomination_checks"] = checks
        result["nominated_for_independent_confirmation"] = all(checks.values())
    return result


def analyze_reselection(model_members: torch.Tensor, measured: torch.Tensor,
                        audit: torch.Tensor, metrics: torch.Tensor, *, disagreement: float,
                        folds: int, split_seed: int, bootstrap_seed: int,
                        bootstrap_repeats: int) -> tuple[dict, dict]:
    expected = (3, 6, 32, 4, 5)
    if (model_members.shape != expected + (3,) or measured.shape != expected
            or audit.shape != expected or metrics.shape != expected + (4,)
            or any(not bool(torch.isfinite(x).all()) for x in
                   (model_members, measured, audit, metrics))):
        raise ValueError("requires complete finite five-candidate tensors")
    ensemble = model_members.double().mean(-1)
    conservative = ensemble - disagreement * model_members.double().std(-1, correction=0)
    assignment = stratified_fold_assignment(seed=split_seed, folds=folds).to(model_members.device)
    conservative_gate, conservative_folds = crossfit_margin_gate(conservative, audit.double(), assignment)
    mean_gate, mean_folds = crossfit_margin_gate(ensemble, audit.double(), assignment)
    shape = measured.shape[:-1]
    choices = {
        "zero": torch.zeros(shape, dtype=torch.long, device=model_members.device),
        "original_selected": torch.full(shape, 3, dtype=torch.long, device=model_members.device),
        "conservative_argmax": conservative.argmax(-1),
        "ensemble_mean_argmax": ensemble.argmax(-1),
        "conservative_safe_gate": conservative_gate,
        "ensemble_mean_safe_gate": mean_gate,
        "oracle_upper_bound": audit.argmax(-1),
    }
    statistics = {
        name: _rule_statistics(name, choice, measured.double(), audit.double(), metrics.double(),
                               seed=bootstrap_seed + index * 100,
                               repeats=bootstrap_repeats)
        for index, (name, choice) in enumerate(choices.items())
    }
    nominated = [name for name in PRIMARY_RULES
                 if statistics[name]["nominated_for_independent_confirmation"]]
    analysis = {
        "candidate_order": list(CANDIDATES),
        "audit_metric_order": list(AUDIT_METRICS),
        "independent_weather": 96,
        "folds": folds,
        "weather_per_test_fold": 96 // folds,
        "threshold_training_target": "audit_objective_gain_over_zero",
        "threshold_selection_rule": "maximize worst-family train gain, then overall gain, then stricter threshold",
        "fold_records": {"conservative_safe_gate": conservative_folds,
                         "ensemble_mean_safe_gate": mean_folds},
        "statistics": statistics,
        "primary_rules": list(PRIMARY_RULES),
        "nominated_rules": nominated,
        "status": ("RESELECTION_RULE_WORTH_INDEPENDENT_CONFIRMATION"
                   if nominated else "NO_RESELECTION_RULE_NOMINATED"),
        "development_cross_validation_only": True,
        "independent_confirmation_access": False,
        "rl_authorized": False,
        "split_seed": split_seed,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_repeats": bootstrap_repeats,
    }
    saved = {"fold_assignment": assignment.cpu(),
             "choices": {name: choice.cpu() for name, choice in choices.items()}}
    return analysis, saved


def preflight(path: str | Path) -> tuple[dict, Path, Path, dict]:
    own = read(_project_path(SOURCES))
    verify_hashes(own)
    config_path = _project_path(path)
    if _relative(config_path) not in own:
        raise ValueError("candidate-reselection config is not frozen")
    cfg = _load_yaml(config_path)
    source = _project_path(cfg["source_metrics"])
    output = _project_path(cfg["output_directory"])
    if output.exists():
        raise FileExistsError(f"preserve candidate-reselection output: {output}")
    verify_hashes(cfg["source_files"])
    finalized = read(_project_path(cfg["finalized_summary"]))
    success = read(_project_path(cfg["finalized_success"]))
    if (success["summary_sha256"] != _file_sha256(_project_path(cfg["finalized_summary"]))
            or finalized["source_metrics_sha256"] != cfg["source_files"][_relative(source)]
            or finalized["new_physical_transitions"] != 0
            or finalized["new_model_forward_samples"] != 0):
        raise RuntimeError("finalized ranking source is not aligned")
    device = resolve_device("cuda")
    data = torch.load(source, map_location=device, weights_only=True)
    if (tuple(data["model_member_scores"].shape) != (3, 6, 32, 4, 5, 3)
            or tuple(data["measured_objective"].shape) != (3, 6, 32, 4, 5)
            or tuple(data["audit_objective"].shape) != (3, 6, 32, 4, 5)
            or tuple(data["metrics"].shape) != (3, 6, 32, 4, 5, 4)
            or tuple(data["candidate_order"]) != CANDIDATES
            or tuple(data["audit_metric_order"]) != AUDIT_METRICS):
        raise RuntimeError("source ranking tensors do not match the frozen schema")
    frozen = dict(own)
    frozen.update(cfg["source_files"])
    frozen[SOURCES] = _file_sha256(_project_path(SOURCES))
    verify_hashes(frozen)
    report = {"status": "READY_FOR_USER_IDE", "device": str(device),
              "frozen_files": frozen, "input_weather": 96, "folds": cfg["folds"],
              "new_physical_transitions": 0, "new_model_forward_samples": 0,
              "training_updates": 0, "real_slm_actions": False,
              "automatic_retry": False}
    return cfg, source, output, report


@torch.no_grad()
def run(path: str | Path, *, preflight_only: bool = False) -> dict:
    cfg, source, output, report = preflight(path)
    if preflight_only:
        return {key: value for key, value in report.items() if key != "frozen_files"}
    device = resolve_device("cuda")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "preflight.json", report)
    write_json(output / "config.json", cfg)
    write_json(output / "runtime.json", {"torch": str(torch.__version__),
               "gpu": torch.cuda.get_device_name(device), "git": safe_git_record()})
    try:
        data = torch.load(source, map_location=device, weights_only=True)
        analysis, saved = analyze_reselection(
            data["model_member_scores"], data["measured_objective"],
            data["audit_objective"], data["metrics"], disagreement=cfg["disagreement"],
            folds=cfg["folds"], split_seed=cfg["split_seed"],
            bootstrap_seed=cfg["bootstrap_seed"],
            bootstrap_repeats=cfg["bootstrap_repeats"])
        torch.save(saved, output / "decisions.pt")
        verify_hashes(report["frozen_files"])
        write_json(output / "source_manifest.json", report["frozen_files"])
        write_json(output / "artifact_manifest.json",
                   {_relative(item): _file_sha256(item)
                    for item in output.rglob("*") if item.is_file()})
        result = {
            "status": "CANDIDATE_RESELECTION_COMPLETE_REQUIRES_AUDIT",
            "analysis": analysis,
            "completion_mode": "existing_candidate_table_cross_validation",
            "source_metrics": _relative(source),
            "source_metrics_sha256": cfg["source_files"][_relative(source)],
            "new_physical_transitions": 0,
            "new_model_forward_samples": 0,
            "training_updates": 0,
            "confirmation_access": False,
            "real_slm_actions": False,
            "automatic_retry": False,
            "artifact_manifest_sha256": _file_sha256(output / "artifact_manifest.json"),
            "material_passport": {"origin_skill": "academic-research-suite",
                "origin_mode": "validate", "origin_date": datetime.now(timezone.utc).isoformat(),
                "verification_status": "REQUIRES_AUDIT",
                "version_label": "r4_candidate_reselection_crossfit_v1"},
            "next_action": "停止并通知助手只读审计；不自动生成确认集、训练或放行RL。",
        }
        write_json(output / "summary.json", result)
        write_json(output / "SUCCESS.json", {"summary_sha256": _file_sha256(output / "summary.json")})
        return result
    except Exception:
        write_json(output / "failure.json", {"traceback": traceback.format_exc(),
                                              "automatic_retry": False})
        raise
