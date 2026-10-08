"""从已完整落盘的R4排序诊断生成并列感知摘要；不重跑仿真。"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import os
import traceback

import torch

from src.rl.r4_baseline_selection import read
from src.rl.r4_dynamics_experiment import safe_git_record, verify_hashes
from src.rl.r4_interface_smoke import write_json
from src.rl.s4_training import _file_sha256, _load_yaml, _project_path, _relative
from src.runtime import resolve_device


SOURCES = "configs/experiments/s4_r4_closed_loop_ranking_finalize_v1_sources.json"
CANDIDATES = ("zero", "negative", "half", "selected", "scaled_1p5")


def tie_aware_metrics(model_score: torch.Tensor, actual_score: torch.Tensor) -> dict[str, torch.Tensor]:
    """并列不算对也不算错；所有覆盖量都显式返回。"""
    if (model_score.shape != actual_score.shape or model_score.ndim < 2
            or model_score.shape[-1] != len(CANDIDATES)
            or not bool(torch.isfinite(model_score).all())
            or not bool(torch.isfinite(actual_score).all())):
        raise ValueError("invalid ranking arrays")
    shape = model_score.shape[:-1]
    concordant = model_score.new_zeros(shape)
    comparable = model_score.new_zeros(shape)
    for left in range(len(CANDIDATES)):
        for right in range(left + 1, len(CANDIDATES)):
            predicted = model_score[..., left] - model_score[..., right]
            observed = actual_score[..., left] - actual_score[..., right]
            valid = (predicted != 0) & (observed != 0)
            comparable += valid
            concordant += valid & (predicted.sign() == observed.sign())
    model_max = model_score.max(-1).values
    actual_max = actual_score.max(-1).values
    model_unique = (model_score == model_max[..., None]).sum(-1) == 1
    actual_unique = (actual_score == actual_max[..., None]).sum(-1) == 1
    model_best = model_score.argmax(-1)
    actual_best = actual_score.argmax(-1)
    chosen = actual_score.gather(-1, model_best[..., None]).squeeze(-1)
    selected_model = model_score[..., 3] - model_score[..., 0]
    selected_actual = actual_score[..., 3] - actual_score[..., 0]
    sign_defined = (selected_model != 0) & (selected_actual != 0)
    return {
        "pairwise_concordant": concordant,
        "pairwise_comparable": comparable,
        "top1_correct": (model_best == actual_best).to(model_score.dtype),
        "top1_defined": model_unique & actual_unique,
        "regret": actual_max - chosen,
        "regret_defined": model_unique,
        "selected_zero_sign_correct": (selected_model.sign() == selected_actual.sign()).to(model_score.dtype),
        "selected_zero_sign_defined": sign_defined,
        "selected_minus_zero": selected_actual,
        "oracle_minus_zero": actual_max - actual_score[..., 0],
    }


def _weather_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    num = numerator.double().sum((1, 3))
    den = denominator.double().sum((1, 3))
    if num.shape != (3, 32) or bool((den <= 0).any()):
        raise ValueError("at least one complete weather lacks defined ranking information")
    return num / den


def _weather_mean(values: torch.Tensor, defined: torch.Tensor | None = None) -> torch.Tensor:
    if defined is None:
        result = values.double().mean((1, 3))
    else:
        numerator = torch.where(defined, values, torch.zeros_like(values)).double().sum((1, 3))
        denominator = defined.double().sum((1, 3))
        if bool((denominator <= 0).any()):
            raise ValueError("at least one complete weather lacks defined values")
        result = numerator / denominator
    if result.shape != (3, 32) or not bool(torch.isfinite(result).all()):
        raise ValueError("invalid weather collapse")
    return result


def _interval(values: torch.Tensor, *, seed: int, repeats: int) -> dict:
    if values.shape != (3, 32) or repeats < 1 or not bool(torch.isfinite(values).all()):
        raise ValueError("invalid weather statistic")
    generator = torch.Generator(device=values.device).manual_seed(seed)
    bootstrap = values.new_zeros(repeats, dtype=torch.float64)
    for family in range(3):
        draw = torch.randint(32, (repeats, 32), device=values.device, generator=generator)
        bootstrap += values[family][draw].double().mean(1) / 3
    ci = torch.quantile(bootstrap, values.new_tensor([.025, .975], dtype=torch.float64))
    return {"mean": float(values.double().mean()), "ci95": ci.tolist()}


def summarize_tie_aware(model_members: torch.Tensor, measured: torch.Tensor, audit: torch.Tensor,
                        *, disagreement: float, seed: int, repeats: int) -> dict:
    expected = (3, 6, 32, 4, 5)
    if (model_members.shape != expected + (3,) or measured.shape != expected or audit.shape != expected
            or any(not bool(torch.isfinite(x).all()) for x in (model_members, measured, audit))):
        raise ValueError("requires complete finite ranking tensors")
    ensemble = model_members.double().mean(-1)
    conservative = ensemble - disagreement * model_members.double().std(-1, correction=0)
    inputs = {
        "conservative_vs_measured": (conservative, measured.double()),
        "ensemble_mean_vs_measured": (ensemble, measured.double()),
        "conservative_vs_audit": (conservative, audit.double()),
    }
    total_states = 3 * 6 * 32 * 4
    total_pairs = total_states * 10
    statistics = {}
    for comparison_index, (comparison, (model_score, actual_score)) in enumerate(inputs.items()):
        metric = tie_aware_metrics(model_score, actual_score)
        pair_defined = metric["pairwise_comparable"] > 0
        top_defined = metric["top1_defined"]
        regret_defined = metric["regret_defined"]
        sign_defined = metric["selected_zero_sign_defined"]
        weather_values = {
            "pairwise_accuracy": _weather_ratio(metric["pairwise_concordant"], metric["pairwise_comparable"]),
            "top1_match": _weather_ratio(metric["top1_correct"] * top_defined, top_defined),
            "regret": _weather_mean(metric["regret"], regret_defined),
            "selected_zero_sign_match": _weather_ratio(
                metric["selected_zero_sign_correct"] * sign_defined, sign_defined),
            "selected_minus_zero": _weather_mean(metric["selected_minus_zero"]),
            "oracle_minus_zero": _weather_mean(metric["oracle_minus_zero"]),
        }
        coverage = {
            "pairwise_accuracy": {"defined_states": int(pair_defined.sum()), "total_states": total_states,
                "comparable_pairs": int(metric["pairwise_comparable"].sum()), "total_pairs": total_pairs},
            "top1_match": {"defined_states": int(top_defined.sum()), "total_states": total_states},
            "regret": {"defined_states": int(regret_defined.sum()), "total_states": total_states},
            "selected_zero_sign_match": {"defined_states": int(sign_defined.sum()), "total_states": total_states},
            "selected_minus_zero": {"defined_states": total_states, "total_states": total_states},
            "oracle_minus_zero": {"defined_states": total_states, "total_states": total_states},
        }
        statistics[comparison] = {
            name: {**_interval(values, seed=seed + comparison_index * 100 + metric_index,
                               repeats=repeats), "coverage": coverage[name]}
            for metric_index, (name, values) in enumerate(weather_values.items())
        }
    audit_stats = statistics["conservative_vs_audit"]
    tied = total_states - audit_stats["pairwise_accuracy"]["coverage"]["defined_states"]
    return {
        "candidate_order": list(CANDIDATES),
        "independent_weather": 96,
        "state_comparisons": total_states,
        "all_tied_uninformative_states": tied,
        "informative_pairwise_states": total_states - tied,
        "statistics": statistics,
        "interpretation_flags": {
            "conservative_pairwise_ci_above_chance": audit_stats["pairwise_accuracy"]["ci95"][0] > .5,
            "selected_zero_direction_ci_above_chance": audit_stats["selected_zero_sign_match"]["ci95"][0] > .5,
            "selected_audit_gain_positive": audit_stats["selected_minus_zero"]["mean"] > 0,
            "oracle_audit_gain_positive": audit_stats["oracle_minus_zero"]["mean"] > 0,
        },
        "bootstrap_seed": seed,
        "bootstrap_repeats": repeats,
        "development_diagnostic_only": True,
        "rl_authorized": False,
    }


def preflight(path: str | Path) -> tuple[dict, Path, Path, dict]:
    own = read(_project_path(SOURCES)); verify_hashes(own)
    config_path = _project_path(path)
    if _relative(config_path) not in own:
        raise ValueError("finalization config is not frozen")
    cfg = _load_yaml(config_path)
    source = _project_path(cfg["source_directory"])
    output = _project_path(cfg["output_directory"])
    if output.exists():
        raise FileExistsError(f"preserve finalized ranking output: {output}")
    verify_hashes(cfg["source_files"])
    if (source / "SUCCESS.json").exists() or (source / "summary.json").exists():
        raise RuntimeError("source unexpectedly completed; metadata recovery is not applicable")
    failure = read(source / "failure.json")
    if "nonfinite collapsed statistic: conservative_vs_measured/pairwise_accuracy" not in failure["traceback"]:
        raise RuntimeError("source failure cause changed")
    source_preflight = read(source / "preflight.json")
    verify_hashes(source_preflight["frozen_files"])
    manifest = read(source / "branch_manifest.json")
    if len(manifest) != 36:
        raise RuntimeError("incomplete branch manifest")
    verify_hashes({record["file"]: record["sha256"] for record in manifest})
    progress = [json.loads(line) for line in (source / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    if (len(progress) != 12240 or progress[-1]["completed"] != 12240
            or progress[-1]["total"] != 12240 or progress[-1]["物理转移"] != 195840
            or progress[-1]["模型前向"] != 28588032):
        raise RuntimeError("source execution did not finish its frozen budget")
    device = resolve_device("cuda")
    metrics = torch.load(source / "metrics.pt", map_location=device, weights_only=True)
    if (tuple(metrics["model_member_scores"].shape) != (3, 6, 32, 4, 5, 3)
            or tuple(metrics["measured_objective"].shape) != (3, 6, 32, 4, 5)
            or tuple(metrics["audit_objective"].shape) != (3, 6, 32, 4, 5)
            or any(not bool(torch.isfinite(metrics[key]).all()) for key in
                   ("model_member_scores", "measured_objective", "audit_objective", "metrics"))):
        raise RuntimeError("source metric tensors are incomplete or nonfinite")
    frozen = dict(own); frozen.update(cfg["source_files"])
    frozen.update(source_preflight["frozen_files"])
    frozen.update({record["file"]: record["sha256"] for record in manifest})
    frozen[SOURCES] = _file_sha256(_project_path(SOURCES))
    verify_hashes(frozen)
    report = {"status": "READY_FOR_METADATA_ONLY_FINALIZATION", "device": str(device),
              "frozen_files": frozen, "new_physical_transitions": 0,
              "new_model_forward_samples": 0, "training_updates": 0,
              "real_slm_actions": False, "automatic_retry": False}
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
        metrics = torch.load(source / "metrics.pt", map_location=device, weights_only=True)
        analysis = summarize_tie_aware(metrics["model_member_scores"], metrics["measured_objective"],
            metrics["audit_objective"], disagreement=cfg["disagreement"],
            seed=cfg["bootstrap_seed"], repeats=cfg["bootstrap_repeats"])
        verify_hashes(report["frozen_files"])
        source_manifest = dict(cfg["source_files"])
        source_manifest.update({record["file"]: record["sha256"]
                                for record in read(source / "branch_manifest.json")})
        write_json(output / "source_manifest.json", source_manifest)
        write_json(output / "artifact_manifest.json",
                   {_relative(p): _file_sha256(p) for p in output.rglob("*") if p.is_file()})
        result = {
            "status": "RANKING_DIAGNOSTIC_COMPLETE_REQUIRES_AUDIT",
            "analysis": analysis,
            "completion_mode": "metadata_only_tie_aware_finalization",
            "source_directory": _relative(source),
            "source_metrics_sha256": cfg["source_files"][_relative(source / "metrics.pt")],
            "logical_physical_transitions": 195840,
            "logical_model_forward_samples": 28588032,
            "new_physical_transitions": 0,
            "new_model_forward_samples": 0,
            "completed_batches": 36,
            "training_updates": 0,
            "confirmation_access": False,
            "real_slm_actions": False,
            "automatic_retry": False,
            "artifact_manifest_sha256": _file_sha256(output / "artifact_manifest.json"),
            "material_passport": {"origin_skill": "academic-research-suite",
                "origin_mode": "validate", "origin_date": datetime.now(timezone.utc).isoformat(),
                "verification_status": "REQUIRES_AUDIT",
                "version_label": "r4_closed_loop_ranking_tie_aware_finalize_v1"},
            "next_action": "停止，等待只读审计；不重跑仿真、不训练或放行RL。",
        }
        write_json(output / "summary.json", result)
        write_json(output / "SUCCESS.json", {"summary_sha256": _file_sha256(output / "summary.json")})
        return result
    except Exception:
        write_json(output / "failure.json", {"traceback": traceback.format_exc(),
                                              "automatic_retry": False})
        raise
