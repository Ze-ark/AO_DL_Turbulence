"""R4闭环候选排序诊断；固定模型与轨迹，不训练、不操作硬件。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import traceback

import torch

from src.rl.r4_action_response import physical_step
from src.rl.r4_amplitude_probe import require_close
from src.rl.r4_baseline_selection import read
from src.rl.r4_baselines import range_eligible
from src.rl.r4_closed_loop import model_ensembles
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


SOURCES = "configs/experiments/s4_r4_closed_loop_ranking_v1_sources.json"
CANDIDATES = ("zero", "negative", "half", "selected", "scaled_1p5")
PROFILES = ("nominal", "delay_3", "settling_050", "registration_moderate",
            "registration_severe", "combined_moderate")
STARTS = (3762048, 3762304, 3762560)
AUDIT_METRICS = ("audit_power", "audit_strehl", "audit_violation", "audit_phase_rmse")
SCORE_REPLAY_ATOL = 5e-6
SCORE_REPLAY_RTOL = 1e-6


def budget(quick: bool) -> dict[str, int]:
    batches, probes, prefix = (1, 2, 40) if quick else (36, 4, 180)
    batch, candidates, horizon, members = 16, len(CANDIDATES), 8, 3
    planning = batches * probes * batch * 128 * 4 * horizon * members
    panel_scoring = batches * probes * batch * candidates * horizon * members
    return {
        "physical_transitions": batches * batch * (prefix + probes * candidates * horizon),
        "model_forward_samples": planning + panel_scoring,
    }


def candidate_sequences(selected: torch.Tensor) -> torch.Tensor:
    """从规划器选中序列构造预声明五候选，不改变原张量。"""
    if (selected.ndim != 3 or selected.shape[1:] != (8, 11)
            or not bool(torch.isfinite(selected).all())
            or bool((selected.abs() > 1).any())):
        raise ValueError("invalid selected sequence")
    return torch.stack((torch.zeros_like(selected), -selected, selected * .5, selected,
                        (selected * 1.5).clamp(-1, 1)), dim=1)


def require_score_replay_close(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """候选批量改变会改变float32累加顺序；仅为该重评分设置专用容差。"""
    if (actual.shape != expected.shape or not bool(torch.isfinite(actual).all())
            or not bool(torch.isfinite(expected).all())):
        raise RuntimeError("invalid selected candidate score replay")
    error = float((actual - expected).abs().max())
    if not torch.allclose(actual, expected, atol=SCORE_REPLAY_ATOL, rtol=SCORE_REPLAY_RTOL):
        raise RuntimeError(f"selected candidate score replay drift: max_abs_error={error:.9g}")
    return error


def discounted_objective(power: torch.Tensor, sequences: torch.Tensor,
                         config: SearchConfig) -> torch.Tensor:
    """按规划器同一折扣和动作代价计算物理分支分数。"""
    if (power.ndim != 3 or sequences.ndim != 4
            or power.shape != sequences.shape[:3]
            or sequences.shape[-2:] != (config.horizon, 11)
            or power.device != sequences.device
            or not bool(torch.isfinite(power).all())
            or not bool(torch.isfinite(sequences).all())):
        raise ValueError("unaligned physical objective")
    discount = power.new_tensor([config.discount ** i for i in range(config.horizon)])
    return ((power - config.action_cost * sequences.square().mean(-1)) * discount).sum(-1)


def ranking_metrics(model_score: torch.Tensor, actual_score: torch.Tensor) -> dict[str, torch.Tensor]:
    """逐状态比较五候选排序；平分对不冒充正确或错误。"""
    if (model_score.shape != actual_score.shape or model_score.ndim < 2
            or model_score.shape[-1] != len(CANDIDATES)
            or not bool(torch.isfinite(model_score).all())
            or not bool(torch.isfinite(actual_score).all())):
        raise ValueError("invalid ranking arrays")
    concordant = model_score.new_zeros(model_score.shape[:-1])
    comparable = model_score.new_zeros(model_score.shape[:-1])
    for left in range(len(CANDIDATES)):
        for right in range(left + 1, len(CANDIDATES)):
            predicted = model_score[..., left] - model_score[..., right]
            observed = actual_score[..., left] - actual_score[..., right]
            valid = (predicted != 0) & (observed != 0)
            comparable += valid
            concordant += valid & (predicted.sign() == observed.sign())
    pairwise = torch.where(comparable > 0, concordant / comparable,
                           torch.full_like(comparable, float("nan")))
    model_best = model_score.argmax(-1)
    actual_best = actual_score.argmax(-1)
    chosen = actual_score.gather(-1, model_best[..., None]).squeeze(-1)
    selected_model = model_score[..., 3] - model_score[..., 0]
    selected_actual = actual_score[..., 3] - actual_score[..., 0]
    selected_valid = (selected_model != 0) & (selected_actual != 0)
    return {
        "pairwise_accuracy": pairwise,
        "comparable_pairs": comparable,
        "top1_match": (model_best == actual_best).to(model_score.dtype),
        "regret": actual_score.max(-1).values - chosen,
        "selected_zero_sign_match": torch.where(
            selected_valid,
            (selected_model.sign() == selected_actual.sign()).to(model_score.dtype),
            torch.full_like(selected_model, float("nan"))),
        "selected_minus_zero": selected_actual,
        "oracle_minus_zero": actual_score.max(-1).values - actual_score[..., 0],
    }


def _stratified_interval(values: torch.Tensor, *, seed: int, repeats: int) -> dict:
    """输入[家族,天气]，按三个家族分别重采样完整天气。"""
    if values.shape != (3, 32) or repeats < 1 or not bool(torch.isfinite(values).all()):
        raise ValueError("invalid weather-level statistic")
    generator = torch.Generator(device=values.device).manual_seed(seed)
    boot = values.new_zeros(repeats, dtype=torch.float64)
    for family in range(3):
        draws = torch.randint(32, (repeats, 32), device=values.device, generator=generator)
        boot += values[family][draws].double().mean(1) / 3
    ci = torch.quantile(boot, values.new_tensor([.025, .975], dtype=torch.float64))
    return {"mean": float(values.double().mean()), "ci95": ci.tolist()}


def summarize(model_members: torch.Tensor, measured: torch.Tensor, audit: torch.Tensor,
              *, disagreement: float, seed: int, repeats: int) -> dict:
    """不把档位或探测时刻当作独立样本。"""
    expected_scores = (3, 6, 32, 4, 5)
    if (model_members.shape != expected_scores + (3,)
            or measured.shape != expected_scores or audit.shape != expected_scores
            or any(not bool(torch.isfinite(x).all()) for x in (model_members, measured, audit))):
        raise ValueError("requires all frozen weather, profiles, probes and candidates")
    ensemble_mean = model_members.double().mean(-1)
    conservative = ensemble_mean - disagreement * model_members.double().std(-1, correction=0)
    analyses = {
        "conservative_vs_measured": ranking_metrics(conservative, measured.double()),
        "ensemble_mean_vs_measured": ranking_metrics(ensemble_mean, measured.double()),
        "conservative_vs_audit": ranking_metrics(conservative, audit.double()),
    }
    reported = {}
    for comparison, metrics in analyses.items():
        reported[comparison] = {}
        for name in ("pairwise_accuracy", "top1_match", "regret",
                     "selected_zero_sign_match", "selected_minus_zero", "oracle_minus_zero"):
            weather = metrics[name].mean((1, 3))  # 档位、探测时刻先在天气内平均
            if not bool(torch.isfinite(weather).all()):
                raise ValueError(f"nonfinite collapsed statistic: {comparison}/{name}")
            reported[comparison][name] = _stratified_interval(
                weather, seed=seed + len(reported) * 100 + len(reported[comparison]), repeats=repeats)
    conservative_audit = reported["conservative_vs_audit"]
    return {
        "candidate_order": list(CANDIDATES),
        "independent_weather": 96,
        "state_comparisons": 96 * 6 * 4,
        "statistics": reported,
        "interpretation_flags": {
            "conservative_pairwise_ci_above_chance": conservative_audit["pairwise_accuracy"]["ci95"][0] > .5,
            "selected_zero_direction_ci_above_chance": conservative_audit["selected_zero_sign_match"]["ci95"][0] > .5,
            "selected_audit_gain_positive": conservative_audit["selected_minus_zero"]["mean"] > 0,
            "oracle_audit_gain_positive": conservative_audit["oracle_minus_zero"]["mean"] > 0,
        },
        "bootstrap_seed": seed,
        "bootstrap_repeats": repeats,
        "development_diagnostic_only": True,
        "rl_authorized": False,
    }


def preflight(path: str | Path, quick: bool) -> tuple:
    own = read(_project_path(SOURCES)); verify_hashes(own)
    config_path = _project_path(path)
    if _relative(config_path) not in own:
        raise ValueError("configuration not frozen")
    cfg = _load_yaml(config_path)
    upstream = _project_path(cfg["upstream"])
    upstream_summary = read(upstream / "summary.json")
    if (_file_sha256(upstream / "summary.json") != cfg["upstream_sha256"]
            or read(upstream / "SUCCESS.json")["summary_sha256"] != cfg["upstream_sha256"]
            or _file_sha256(upstream / "artifact_manifest.json") != upstream_summary["artifact_manifest_sha256"]
            or upstream_summary["status"] != "AMPLITUDE_CLOSED_LOOP_COMPLETE_REQUIRES_AUDIT"
            or upstream_summary["analysis"]["all_development_gates_pass"]
            or upstream_summary["analysis"]["rl_authorized"]):
        raise RuntimeError("requires frozen negative amplitude recovery result")
    spec, frozen = load_selected()
    frozen.update(read(upstream / "preflight.json")["frozen_files"])
    frozen.update(read(upstream / "artifact_manifest.json")); frozen.update(own)
    for item in (upstream / "summary.json", upstream / "SUCCESS.json",
                 upstream / "artifact_manifest.json", _project_path(SOURCES)):
        frozen[_relative(item)] = _file_sha256(item)
    if not quick:
        quick_output = _project_path(cfg["output_directory"] + "_quick")
        quick_summary = read(quick_output / "summary.json")
        if (quick_summary["status"] != "QUICK_COMPLETE_NO_CONCLUSION"
                or read(quick_output / "SUCCESS.json")["summary_sha256"] != _file_sha256(quick_output / "summary.json")
                or _file_sha256(quick_output / "artifact_manifest.json") != quick_summary["artifact_manifest_sha256"]
                or read(quick_output / "preflight.json")["frozen_files"].get(SOURCES) != _file_sha256(_project_path(SOURCES))):
            raise RuntimeError("same-version quick prerequisite missing")
        frozen.update(read(quick_output / "artifact_manifest.json"))
        for name in ("summary.json", "SUCCESS.json", "artifact_manifest.json"):
            frozen[_relative(quick_output / name)] = _file_sha256(quick_output / name)
    verify_hashes(frozen)
    if (cfg["probe_steps"] != [8, 40, 100, 180]
            or cfg["candidate_order"] != list(CANDIDATES)
            or cfg["horizon"] != 8
            or cfg["training_updates"] != 0
            or cfg["automatic_retry"] is not False
            or any(cfg[key] != value for key, value in budget(False).items())):
        raise ValueError("predeclared ranking design or budget changed")
    records = [record for record in read(upstream / "trajectory_manifest.json")
               if record["controller"] == "gru_mpc"]
    parent = _load_yaml(_project_path(cfg["parent"]))
    expected = {(family, profile, STARTS[family] + offset)
                for family in range(3) for profile in parent["profile_ids"] for offset in (0, 16)}
    if len(records) != 36 or {(r["family"], r["profile"], r["seed"]) for r in records} != expected:
        raise RuntimeError("incomplete original-strength trajectories")
    if quick:
        # v3快速检查直接覆盖v2正式运行首次触发浮点重评分边界的批次。
        records = [next(r for r in records
                        if (r["family"], r["profile"], r["seed"]) == (0, "nominal", STARTS[0] + 16))]
    output = _project_path(cfg["output_directory"] + ("_quick" if quick else ""))
    if output.exists():
        raise FileExistsError(f"preserve ranking output: {output}")
    device = resolve_device("cuda")
    report = {"status": "READY_FOR_DIAGNOSTIC" if quick else "READY_FOR_USER_IDE",
              "quick": quick, "device": str(device), "frozen_files": frozen, **budget(quick)}
    return cfg, parent, spec, records, output, report


@torch.no_grad()
def execute(cfg: dict, parent: dict, spec, records: list, output: Path, report: dict) -> dict:
    device = resolve_device("cuda")
    calibration = NominalCalibration(**parent["nominal_calibration"])
    models = model_ensembles(device)["gru_mpc"]
    search = SearchConfig()
    base, _ = load_s1_config(_project_path(parent["environment_config"]))
    base = replace(base, num_modes=21, batch_size=16, episode_length=200)
    basis, _, _ = build_action_basis(base, ActionRepresentation("r4_zernike21", "zernike", 21), device)
    profiles = {profile.identifier: profile for profile in _profiles(parent, parent["profile_ids"])}
    bounds = read(_project_path("outputs/s4_r4_baseline_safety_v1/calibration.json"))
    probes = cfg["probe_steps"][:2] if report["quick"] else cfg["probe_steps"]
    member_scores = torch.zeros(3, 6, 32, 4, 5, 3, dtype=torch.float64, device=device)
    measured_scores = torch.zeros(3, 6, 32, 4, 5, dtype=torch.float64, device=device)
    audit_scores = torch.zeros_like(measured_scores)
    metrics = torch.zeros(3, 6, 32, 4, 5, 4, dtype=torch.float64, device=device)
    progress = Progress(output, device)
    progress.phase("闭环候选排序诊断", report["physical_transitions"] // 16)
    physical = calls = 0; active = {}; manifest = []; score_replay_max_abs_error = 0.
    (output / "branches").mkdir()

    def tick() -> None:
        nonlocal physical
        physical += 16
        progress.tick({"物理转移": physical, "模型前向": calls})

    try:
        for record in records:
            active = dict(record)
            saved = torch.load(_project_path(record["file"]), map_location=device, weights_only=True)
            family, seed = record["family"], record["seed"]
            profile = profiles[record["profile"]]
            condition = RobustnessCondition.from_mapping(dict(parent["families"][family], base_seed=seed))
            env = AdaptiveOpticsEnv(profile.environment_config(condition.environment_config(base)), device,
                                    profile.effects_config(), basis_override=basis)
            raw, _ = env.reset(seed=seed)
            sensor = torch.Generator(device=device).manual_seed(seed + parent["data"]["sensor_seed_offset"])
            interface = R4Interface(calibration=calibration)
            interface.reset(simulation_residual_proxy(raw, generator=sensor,
                            noise_std_rad=profile.observation_noise_std_rad), episode_id=str(seed))
            require_close(interface.snapshot().features[:, -1], saved["frames"][:, 0], "ranking reset")
            batch_results = []
            for step in range(max(probes) + 1):
                active["replay_step"] = step
                require_close(interface.snapshot().features[:, -1], saved["frames"][:, step], f"ranking frame {step}")
                original = saved["rows"][step]["correction"]
                if step in probes:
                    snapshot = interface.snapshot()
                    if not bool(range_eligible(snapshot.features, snapshot.valid, bounds).all()):
                        raise RuntimeError("probe state left frozen planning range")
                    offset = seed - STARTS[family]
                    plan_seed = cfg["search_seed"] + family * 100000 + parent["profile_ids"].index(profile.identifier) * 10000 + offset * 200 + step
                    planned = selected_plan(models, snapshot.features, snapshot.valid, spec,
                                            calibration, search, seed=plan_seed)
                    require_close(planned["correction"], original, "replanned correction")
                    require_close(planned["score"], saved["rows"][step]["unscaled_plan_score"], "replanned score")
                    require_close(planned["zero_score"], saved["rows"][step]["zero_score"], "replanned zero score")
                    panel = candidate_sequences(planned["sequence"])
                    per_member = torch.stack([
                        sequence_scores([model], snapshot.features, snapshot.valid, panel,
                                        anchor_parameters(spec), calibration, search)
                        for model in models
                    ], dim=-1)
                    calls += planned["model_forward_samples"] + 16 * len(CANDIDATES) * search.horizon * len(models)
                    conservative = per_member.mean(-1) - search.disagreement * per_member.std(-1, correction=0)
                    score_replay_max_abs_error = max(score_replay_max_abs_error,
                        require_score_replay_close(conservative[:, 3], planned["score"]))
                    branch_results = []
                    for candidate_index, name in enumerate(CANDIDATES):
                        branch_env, branch_interface, branch_sensor = deepcopy((env, interface, sensor))
                        branch_rows = []
                        for horizon_step in range(search.horizon):
                            row = physical_step(branch_env, branch_interface, branch_sensor, profile,
                                                panel[:, candidate_index, horizon_step], anchor_parameters(spec))
                            if name == "selected" and horizon_step == 0:
                                require_close(row["audit_power"], saved["rows"][step]["reward_power_in_bucket"], "selected first power")
                                require_close(row["applied_modal"], saved["rows"][step]["applied_modal"], "selected first actuator")
                            branch_rows.append(row); tick()
                        stacked = {key: torch.stack([row[key] for row in branch_rows], 1)
                                   for key in branch_rows[0]}
                        if any(not bool(value.isfinite().all()) for value in stacked.values()):
                            raise RuntimeError("nonfinite ranking branch")
                        branch_results.append({"candidate": name, **{k: v.cpu() for k, v in stacked.items()}})
                    measured = torch.stack([branch["power"] for branch in branch_results], 1)
                    audit = torch.stack([branch["audit_power"] for branch in branch_results], 1)
                    # 分支张量已转到CPU以便保存；目标计算显式使用同设备候选，
                    # 避免隐式跨设备复制掩盖审计错误。
                    panel_cpu = panel.cpu()
                    measured_objective = discounted_objective(measured, panel_cpu, search)
                    audit_objective = discounted_objective(audit, panel_cpu, search)
                    profile_index = parent["profile_ids"].index(profile.identifier)
                    probe_index = probes.index(step)
                    sl = (family, profile_index, slice(offset, offset + 16), probe_index)
                    member_scores[sl] = per_member.double()
                    measured_scores[sl] = measured_objective.to(device).double()
                    audit_scores[sl] = audit_objective.to(device).double()
                    for candidate_index, branch in enumerate(branch_results):
                        metrics[sl + (candidate_index,)] = torch.stack([
                            branch[name].double().mean(1) for name in AUDIT_METRICS], -1).to(device)
                    batch_results.append({"step": step, "plan_seed": plan_seed,
                        "candidate_sequences": panel.cpu(), "model_member_scores": per_member.cpu(),
                        "score_replay_max_abs_error": float((conservative[:, 3] - planned["score"]).abs().max()),
                        "measured_objective": measured_objective.cpu(), "audit_objective": audit_objective.cpu(),
                        "branches": branch_results})
                    require_close(interface.snapshot().features, snapshot.features, "ranking branch isolation")
                if step < max(probes):
                    row = physical_step(env, interface, sensor, profile, original, anchor_parameters(spec)); tick()
                    require_close(row["audit_power"], saved["rows"][step]["reward_power_in_bucket"], f"ranking replay power {step}")
                    require_close(row["requested_modal"], saved["rows"][step]["requested_modal"], f"ranking replay request {step}")
                    require_close(row["applied_modal"], saved["rows"][step]["applied_modal"], f"ranking replay actuator {step}")
            target = output / "branches" / Path(record["file"]).name
            torch.save({"source": record, "seeds": saved["seeds"], "probes": batch_results}, target)
            manifest.append({"file": _relative(target), "sha256": _file_sha256(target)})
        if {"physical_transitions": physical, "model_forward_samples": calls} != budget(report["quick"]):
            raise RuntimeError("ranking diagnostic budget mismatch")
        write_json(output / "branch_manifest.json", manifest)
        if not report["quick"]:
            torch.save({"model_member_scores": member_scores.cpu(),
                        "measured_objective": measured_scores.cpu(), "audit_objective": audit_scores.cpu(),
                        "metrics": metrics.cpu(), "candidate_order": CANDIDATES,
                        "audit_metric_order": AUDIT_METRICS}, output / "metrics.pt")
        analysis = {} if report["quick"] else summarize(
            member_scores, measured_scores, audit_scores, disagreement=search.disagreement,
            seed=cfg["bootstrap_seed"], repeats=cfg["bootstrap_repeats"])
        return {"status": "QUICK_COMPLETE_NO_CONCLUSION" if report["quick"] else "RANKING_DIAGNOSTIC_COMPLETE_REQUIRES_AUDIT",
                **budget(report["quick"]), "completed_batches": len(records),
                "score_replay_max_abs_error": score_replay_max_abs_error,
                "score_replay_atol": SCORE_REPLAY_ATOL, "score_replay_rtol": SCORE_REPLAY_RTOL,
                "analysis": analysis}
    except Exception:
        write_json(output / "interrupted_context.json",
                   {"active": active, "physical_transitions": physical, "model_forward_samples": calls})
        raise
    finally:
        progress.close()


def run(path: str | Path, *, quick: bool = False, preflight_only: bool = False) -> dict:
    cfg, parent, spec, records, output, report = preflight(path, quick)
    if preflight_only:
        return {key: value for key, value in report.items() if key != "frozen_files"}
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
        raise ValueError("unsupported CUDA determinism setting")
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "preflight.json", report); write_json(output / "config.json", cfg)
    write_json(output / "runtime.json", {"torch": str(torch.__version__),
               "gpu": torch.cuda.get_device_name(), "git": safe_git_record()})
    try:
        result = execute(cfg, parent, spec, records, output, report)
        verify_hashes(report["frozen_files"])
        write_json(output / "artifact_manifest.json",
                   {_relative(p): _file_sha256(p) for p in output.rglob("*") if p.is_file()})
        result.update(training_updates=0, confirmation_access=False, real_slm_actions=False,
            automatic_retry=False, artifact_manifest_sha256=_file_sha256(output / "artifact_manifest.json"),
            material_passport={"origin_skill": "academic-research-suite", "origin_mode": "run",
                "origin_date": datetime.now(timezone.utc).isoformat(),
                "verification_status": "REQUIRES_AUDIT", "version_label": "r4_closed_loop_ranking_v3"},
            next_action="停止，等待只读审计；不自动改模型、训练或放行RL。")
        write_json(output / "summary.json", result)
        write_json(output / "SUCCESS.json", {"summary_sha256": _file_sha256(output / "summary.json")})
        return result
    except Exception:
        write_json(output / "failure.json", {"traceback": traceback.format_exc(), "automatic_retry": False})
        raise
