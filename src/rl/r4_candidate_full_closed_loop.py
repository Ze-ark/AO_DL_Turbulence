"""冻结候选重选规则的全新天气完整闭环确认；不训练、不操作硬件。"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import time
import traceback

import torch

from src.rl.r4_baseline_selection import METRICS, read
from src.rl.r4_baselines import baseline_delta, guarded_correction, range_eligible
from src.rl.r4_candidate_confirmation import frozen_gate_choices
from src.rl.r4_closed_loop import model_ensembles
from src.rl.r4_closed_loop_ranking import CANDIDATES, candidate_sequences
from src.rl.r4_control import NominalCalibration
from src.rl.r4_dynamics_experiment import Progress, safe_git_record, verify_hashes
from src.rl.r4_interface_smoke import write_json
from src.rl.r4_mpc import SearchConfig, sequence_scores
from src.rl.r4_observation import PowerMeasurement, R4Interface, simulation_residual_proxy
from src.rl.r4_selected_anchor import anchor_parameters, load_selected, selected_plan, selected_request
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _profiles, _project_path, _relative
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.robust_control import RobustnessCondition


SOURCES = "configs/experiments/s4_r4_candidate_full_closed_loop_v1_sources.json"
CONTROLLERS = ("integrator", "original_gru_mpc", "frozen_reselection")
FROZEN_THRESHOLD = 0.0032300949096679688


def budget(quick: bool) -> dict[str, int]:
    """模型数按三成员、CEM全批搜索上限计数。"""
    weather, steps, families, profiles = (2, 12, 1, 1) if quick else (32, 200, 3, 6)
    search = SearchConfig()
    states = weather * steps * families * profiles
    planning = states * search.population * search.iterations * search.horizon * 3
    panel = states * len(CANDIDATES) * search.horizon * 3
    return {
        "physical_transitions": len(CONTROLLERS) * states,
        "max_model_forward_samples": planning * 2 + panel,
    }


def _interval(values: torch.Tensor, *, seed: int, repeats: int) -> dict:
    if (values.shape != (3, 32) or repeats < 1
            or not bool(torch.isfinite(values).all())):
        raise ValueError("requires three complete families of 32 weather episodes")
    generator = torch.Generator(device=values.device).manual_seed(seed)
    boot = values.new_zeros(repeats, dtype=torch.float64)
    for family in range(3):
        draw = torch.randint(32, (repeats, 32), generator=generator, device=values.device)
        boot += values[family][draw].double().mean(1) / 3
    ci = torch.quantile(boot, values.new_tensor([.025, .975], dtype=torch.float64))
    return {
        "mean": float(values.double().mean()),
        "ci95": ci.tolist(),
        "family_means": [float(values[index].double().mean()) for index in range(3)],
    }


def summarize(means: torch.Tensor, selection_counts: torch.Tensor, *,
              seed: int, repeats: int) -> dict:
    """先在天气内平均六档，再按完整天气分层配对。"""
    if (means.shape != (3, 3, 6, 32, len(METRICS))
            or selection_counts.shape != (len(CANDIDATES),)
            or int(selection_counts.sum()) != 3 * 6 * 32 * 200
            or not bool(torch.isfinite(means).all())):
        raise ValueError("complete finite full-loop confirmation is required")
    values = means.double().mean(2)  # 控制器、家族、天气、指标
    controller_means = values.mean((1, 2))

    def comparison(left: int, right: int, offset: int) -> dict:
        delta = values[left] - values[right]
        return {
            "power_delta": _interval(delta[..., 0], seed=seed + offset, repeats=repeats),
            "metric_mean_delta": {
                name: float(delta[..., index].mean())
                for index, name in enumerate(METRICS)
            },
        }

    primary = comparison(2, 0, 0)
    original = comparison(1, 0, 100)
    gated_vs_original = comparison(2, 1, 200)
    if float(controller_means[0, 0]) <= 0:
        raise ValueError("nonpositive integrator reference power")
    relative = float(controller_means[2, 0] / controller_means[0, 0] - 1)
    action_rate = float(selection_counts[1:].sum().double() / selection_counts.sum())
    delta = primary["metric_mean_delta"]
    gates = {
        "power_gain_at_least_one_percent": relative >= .01,
        "paired_power_ci_lower_positive": primary["power_delta"]["ci95"][0] > 0,
        "all_family_power_means_positive": min(primary["power_delta"]["family_means"]) > 0,
        "strehl_not_lower": delta["reward_strehl"] >= 0,
        "violation_increase_at_most_point001": delta["violation_fraction"] <= .001,
        "phase_rmse_not_higher": delta["reward_phase_rmse"] <= 0,
        "saturation_increase_at_most_point001": delta["saturated_fraction"] <= .001,
        "slew_limit_increase_at_most_point001": delta["slew_limited_fraction"] <= .001,
        "nontrivial_reselection_action_rate": action_rate >= .05,
    }
    passed = all(gates.values())
    return {
        "status": ("FULL_CLOSED_LOOP_CONFIRMATION_PASS_REQUIRES_AUDIT"
                   if passed else "FULL_CLOSED_LOOP_CONFIRMATION_FAIL_REQUIRES_AUDIT"),
        "controller_order": list(CONTROLLERS),
        "metric_order": list(METRICS),
        "primary_comparison": "frozen_reselection_vs_integrator",
        "secondary_comparisons_are_descriptive": [
            "original_gru_mpc_vs_integrator", "frozen_reselection_vs_original_gru_mpc"],
        "comparisons": {
            "frozen_reselection_vs_integrator": primary,
            "original_gru_mpc_vs_integrator": original,
            "frozen_reselection_vs_original_gru_mpc": gated_vs_original,
        },
        "controller_means": controller_means.tolist(),
        "relative_power_gain": relative,
        "selection_counts": {
            name: int(selection_counts[index]) for index, name in enumerate(CANDIDATES)},
        "reselection_action_rate": action_rate,
        "confirmation_gates": gates,
        "all_confirmation_gates_pass": passed,
        "independent_weather": 96,
        "weather_per_family": 32,
        "statistical_unit": "complete_weather; six_profiles_averaged_within_weather",
        "full_closed_loop_performance_established": passed,
        "rl_authorized": False,
        "bootstrap_seed": seed,
        "bootstrap_repeats": repeats,
    }


def preflight(path: str | Path, quick: bool) -> tuple:
    own = read(_project_path(SOURCES))
    verify_hashes(own)
    config_path = _project_path(path)
    if _relative(config_path) not in own:
        raise ValueError("full-loop confirmation config is not frozen")
    cfg = _load_yaml(config_path)
    verify_hashes(cfg["source_files"])
    upstream = _project_path(cfg["candidate_confirmation_output"])
    upstream_summary = read(upstream / "summary.json")
    analysis = upstream_summary["analysis"]
    if (_file_sha256(upstream / "summary.json") != cfg["candidate_confirmation_sha256"]
            or read(upstream / "SUCCESS.json")["summary_sha256"] != cfg["candidate_confirmation_sha256"]
            or _file_sha256(upstream / "artifact_manifest.json") != upstream_summary["artifact_manifest_sha256"]
            or upstream_summary["status"] != "CANDIDATE_CONFIRMATION_COMPLETE_REQUIRES_AUDIT"
            or analysis["status"] != "INDEPENDENT_LOCAL_RULE_CONFIRMATION_PASS_REQUIRES_AUDIT"
            or not analysis["all_confirmation_gates_pass"]
            or not analysis["full_closed_loop_design_authorized"]
            or analysis["rl_authorized"]):
        raise RuntimeError("audited local candidate confirmation prerequisite is not aligned")
    spec, frozen = load_selected()
    frozen.update(read(upstream / "preflight.json")["frozen_files"])
    frozen.update(read(upstream / "artifact_manifest.json"))
    frozen.update(cfg["source_files"])
    frozen.update(own)
    frozen[SOURCES] = _file_sha256(_project_path(SOURCES))
    for item in (upstream / "summary.json", upstream / "SUCCESS.json",
                 upstream / "artifact_manifest.json"):
        frozen[_relative(item)] = _file_sha256(item)
    parent = _load_yaml(_project_path(cfg["parent"]))
    expected_starts = [3790000, 3790256, 3790512]
    expected = budget(False)
    if (cfg["controllers"] != list(CONTROLLERS)
            or cfg["formal_starts"] != expected_starts
            or cfg["quick_start"] != 3791000
            or cfg["weather_per_family"] != 32
            or cfg["batch_episodes"] != 16
            or cfg["steps"] != 200
            or cfg["frozen_threshold"] != FROZEN_THRESHOLD
            or cfg["candidate_order"] != list(CANDIDATES)
            or cfg["training_updates"] != 0
            or cfg["automatic_retry"] is not False
            or any(cfg[key] != value for key, value in expected.items())):
        raise ValueError("predeclared full-loop design or budget changed")
    used_ranges = [
        (3762048 + offset, 3762048 + offset + 32)
        for offset in parent["data"]["family_offsets"]
    ] + [
        (3780000 + offset, 3780000 + offset + 16)
        for offset in (0, 256, 512)
    ]
    if any(any(start < high and start + 32 > low for low, high in used_ranges)
           for start in expected_starts):
        raise ValueError("full-loop confirmation weather overlaps prior evidence")
    if not quick:
        quick_output = _project_path(cfg["output_directory"] + "_quick")
        quick_summary = read(quick_output / "summary.json")
        if (quick_summary["status"] != "QUICK_COMPLETE_NO_RANKING"
                or read(quick_output / "SUCCESS.json")["summary_sha256"]
                != _file_sha256(quick_output / "summary.json")
                or _file_sha256(quick_output / "artifact_manifest.json")
                != quick_summary["artifact_manifest_sha256"]
                or read(quick_output / "preflight.json")["frozen_files"].get(SOURCES)
                != _file_sha256(_project_path(SOURCES))):
            raise RuntimeError("same-version full-loop quick prerequisite missing")
        frozen.update(read(quick_output / "artifact_manifest.json"))
        for name in ("summary.json", "SUCCESS.json", "artifact_manifest.json"):
            frozen[_relative(quick_output / name)] = _file_sha256(quick_output / name)
    verify_hashes(frozen)
    output = _project_path(cfg["output_directory"] + ("_quick" if quick else ""))
    if output.exists():
        raise FileExistsError(f"preserve full-loop confirmation output: {output}")
    if shutil.disk_usage(_project_path(".")).free < (4096 if not quick else 128) * 1024**2:
        raise RuntimeError("insufficient output disk space")
    device = resolve_device("cuda")
    report = {
        "status": "READY_FOR_DIAGNOSTIC" if quick else "READY_FOR_USER_IDE",
        "quick": quick,
        "device": str(device),
        "frozen_files": frozen,
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
    n, batch, steps = (2, 2, 12) if quick else (32, 16, 200)
    families = parent["families"][:1] if quick else parent["families"]
    profile_ids = parent["profile_ids"][:1] if quick else parent["profile_ids"]
    profiles = _profiles(parent, profile_ids)
    starts = [cfg["quick_start"]] if quick else cfg["formal_starts"]
    base, _ = load_s1_config(_project_path(parent["environment_config"]))
    base = replace(base, num_modes=21, batch_size=batch, episode_length=steps)
    basis, _, _ = build_action_basis(
        base, ActionRepresentation("r4_zernike21", "zernike", 21), device)
    bounds = read(_project_path("outputs/s4_r4_baseline_safety_v1/calibration.json"))
    calibration = NominalCalibration(**parent["nominal_calibration"])
    models = model_ensembles(device)["gru_mpc"]
    search = SearchConfig()
    anchor = anchor_parameters(spec)
    means = torch.zeros(len(CONTROLLERS), len(families), len(profiles), n,
                        len(METRICS), dtype=torch.float64, device=device)
    selection_counts = torch.zeros(len(CANDIDATES), dtype=torch.long, device=device)
    progress = Progress(output, device)
    progress.phase("冻结重选完整闭环确认" if not quick else "完整闭环快速冒烟（不排名）",
                   report["physical_transitions"] // batch)
    physical = calls = planned_total = 0
    active: dict = {}
    manifest = []
    fallback_counts: dict[str, int] = {}
    (output / "trajectories").mkdir()
    try:
        for controller_index, controller in enumerate(CONTROLLERS):
            for family_index, family in enumerate(families):
                for profile_index, profile in enumerate(profiles):
                    for offset in range(0, n, batch):
                        seed = starts[family_index] + offset
                        active = {"controller": controller, "family": family_index,
                                  "profile": profile.identifier, "seed": seed, "step": 0}
                        condition = RobustnessCondition.from_mapping(dict(family, base_seed=seed))
                        env = AdaptiveOpticsEnv(
                            profile.environment_config(condition.environment_config(base)), device,
                            profile.effects_config(), basis_override=basis)
                        raw, _ = env.reset(seed=seed)
                        sensor = torch.Generator(device=device).manual_seed(
                            seed + parent["data"]["sensor_seed_offset"])
                        proxy = lambda value: simulation_residual_proxy(
                            value, generator=sensor,
                            noise_std_rad=profile.observation_noise_std_rad)
                        interface = R4Interface(calibration=calibration)
                        interface.reset(proxy(raw), episode_id=str(seed))
                        frames = [interface.snapshot().features[:, -1].cpu()]
                        rows = []
                        for step in range(steps):
                            active["step"] = step
                            snapshot = interface.snapshot()
                            history, valid_history = snapshot.features, snapshot.valid
                            correction = history.new_zeros(batch, 11)
                            predicted = history.new_full((batch,), float("nan"))
                            zero_score = predicted.clone()
                            margin = predicted.clone()
                            choices = torch.zeros(batch, dtype=torch.long, device=device)
                            member_scores = history.new_full((batch, len(CANDIDATES), 3), float("nan"))
                            eligible = torch.zeros(batch, dtype=torch.bool, device=device)
                            reasons = ["integrator"] * batch
                            call_count = 0
                            torch.cuda.synchronize(device)
                            begin = time.perf_counter()
                            if controller != "integrator":
                                in_range = range_eligible(history, valid_history, bounds)
                                sequence = history.new_zeros(batch, search.horizon, 11)
                                if bool(in_range.any()):
                                    first = int(torch.nonzero(in_range)[0, 0])
                                    plan_history = torch.where(
                                        in_range[:, None, None], history, history[first:first + 1])
                                    plan_valid = torch.where(
                                        in_range[:, None], valid_history, valid_history[first:first + 1])
                                    plan_seed = (cfg["search_seed"] + family_index * 100000
                                                 + profile_index * 10000 + offset * 200 + step)
                                    planned = selected_plan(
                                        models, plan_history, plan_valid, spec, calibration,
                                        search, seed=plan_seed)
                                    sequence[in_range] = planned["sequence"][in_range]
                                    predicted[in_range] = planned["score"][in_range]
                                    zero_score[in_range] = planned["zero_score"][in_range]
                                    call_count += int(planned["model_forward_samples"])
                                first_action, reasons = guarded_correction(
                                    history, valid_history, sequence[:, 0], bounds)
                                eligible = torch.tensor(
                                    [reason == "accepted" for reason in reasons],
                                    device=device, dtype=torch.bool)
                                sequence = torch.where(
                                    eligible[:, None, None], sequence, torch.zeros_like(sequence))
                                if not torch.equal(sequence[:, 0], first_action):
                                    raise RuntimeError("planned sequence and guarded action differ")
                                correction = first_action
                                if controller == "frozen_reselection" and bool(eligible.any()):
                                    panel = candidate_sequences(sequence)
                                    first = int(torch.nonzero(eligible)[0, 0])
                                    score_history = torch.where(
                                        eligible[:, None, None], history, history[first:first + 1])
                                    score_valid = torch.where(
                                        eligible[:, None], valid_history, valid_history[first:first + 1])
                                    member_scores = torch.stack([
                                        sequence_scores([model], score_history, score_valid, panel,
                                                        anchor, calibration, search)
                                        for model in models
                                    ], dim=-1)
                                    call_count += batch * len(CANDIDATES) * search.horizon * len(models)
                                    choices, margin = frozen_gate_choices(
                                        member_scores, cfg["frozen_threshold"], eligible=eligible)
                                    selected_sequence = panel.gather(
                                        1, choices[:, None, None, None].expand(
                                            -1, 1, search.horizon, 11)).squeeze(1)
                                    correction, reasons = guarded_correction(
                                        history, valid_history, selected_sequence[:, 0], bounds)
                                    ensemble = member_scores.double().mean(-1)
                                    predicted = ensemble.gather(1, choices[:, None]).squeeze(1).to(history.dtype)
                                    zero_score = ensemble[:, 0].to(history.dtype)
                                if controller == "frozen_reselection":
                                    selection_counts += torch.bincount(
                                        choices, minlength=len(CANDIDATES))
                            calls += call_count
                            planned_total += int(eligible.sum())
                            expected = selected_request(spec, history, valid_history, correction)
                            delta, _ = baseline_delta(spec, history, valid_history, [], calibration)
                            action = interface.issue(delta, correction, step=step)
                            if not torch.equal(expected.requested_delta_rad, action.requested_delta_rad):
                                raise RuntimeError("full-loop request mismatch")
                            torch.cuda.synchronize(device)
                            latency = time.perf_counter() - begin
                            raw, _, terminated, truncated, info = env.step(action.requested_delta_rad)
                            physical += batch
                            interface.observe_next(
                                proxy(raw), step=step + 1,
                                power=PowerMeasurement(info["measured_power_in_bucket"], step, step + 1))
                            if bool(truncated.any()) or not bool((terminated == (step == steps - 1)).all()):
                                raise RuntimeError("incomplete full-loop episode")
                            if not torch.allclose(action.requested_modal_rad, info["requested_modal"],
                                                  atol=1e-6, rtol=0):
                                raise RuntimeError("environment action mismatch")
                            audit = {key: info[key].cpu() for key in (*METRICS, "applied_modal")}
                            if any(not bool(value.isfinite().all()) for value in audit.values()):
                                raise RuntimeError("nonfinite full-loop metric")
                            for reason in reasons:
                                key = f"{controller}:{reason}"
                                fallback_counts[key] = fallback_counts.get(key, 0) + 1
                            rows.append({
                                **audit,
                                "requested_delta": action.requested_delta_rad.cpu(),
                                "requested_modal": action.requested_modal_rad.cpu(),
                                "correction": correction.cpu(),
                                "predicted_score": predicted.cpu(),
                                "zero_score": zero_score.cpu(),
                                "margin": margin.cpu(),
                                "choice": choices.cpu(),
                                "candidate_member_scores": member_scores.cpu(),
                                "predicted_valid": eligible.cpu(),
                                "reasons": reasons,
                                "model_forward_samples": call_count,
                                "latency_seconds": latency,
                                "terminated": terminated.cpu(),
                                "measured_power": info["measured_power_in_bucket"].cpu(),
                            })
                            frames.append(interface.snapshot().features[:, -1].cpu())
                            progress.tick({"控制器": controller_index + 1, "回合步": step + 1,
                                           "物理转移": physical, "模型前向": calls})
                        episode = torch.stack([
                            torch.stack([row[key] for row in rows], 1).to(device).double().mean(1)
                            for key in METRICS], -1)
                        means[controller_index, family_index, profile_index,
                              offset:offset + batch] = episode
                        target = (output / "trajectories"
                                  / f"{controller}_{family['id']}_{profile.identifier}_{seed}.pt")
                        torch.save({
                            **active,
                            "split": "quick_diagnostic" if quick else "independent_confirmation",
                            "source": "simulation_residual_proxy_not_holography",
                            "seeds": list(range(seed, seed + batch)),
                            "frames": torch.stack(frames, 1),
                            "rows": rows,
                        }, target)
                        manifest.append({"file": _relative(target), "sha256": _file_sha256(target),
                                         **active})
                        with (output / "episode_metrics.jsonl").open("a", encoding="utf-8") as stream:
                            for index, values in enumerate(episode.cpu().tolist()):
                                stream.write(json.dumps({
                                    "controller": controller, "family": family_index,
                                    "profile": profile.identifier, "seed": seed + index,
                                    "metrics": values}, ensure_ascii=False) + "\n")
        expected_budget = budget(quick)
        if (physical != expected_budget["physical_transitions"]
                or calls > expected_budget["max_model_forward_samples"]):
            raise RuntimeError("full-loop confirmation budget mismatch")
        torch.save({"means": means.cpu(), "metric_order": METRICS,
                    "controller_order": CONTROLLERS,
                    "selection_counts": selection_counts.cpu()}, output / "metrics.pt")
        write_json(output / "trajectory_manifest.json", manifest)
        analysis = {} if quick else summarize(
            means, selection_counts, seed=cfg["bootstrap_seed"],
            repeats=cfg["bootstrap_repeats"])
        return {
            "status": "QUICK_COMPLETE_NO_RANKING" if quick else analysis["status"],
            "analysis": analysis,
            "physical_transitions": physical,
            "model_forward_samples": calls,
            "max_model_forward_samples": expected_budget["max_model_forward_samples"],
            "planned_actions": planned_total,
            "fallback_counts": fallback_counts,
            "selection_counts": {
                name: int(selection_counts[index]) for index, name in enumerate(CANDIDATES)},
            "completed_batches": len(manifest),
        }
    except Exception:
        write_json(output / "interrupted_context.json", {
            **active, "physical_transitions": physical, "model_forward_samples": calls})
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
    device = resolve_device("cuda")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "preflight.json", report)
    write_json(output / "config.json", cfg)
    write_json(output / "runtime.json", {
        "gpu": torch.cuda.get_device_name(device), "torch": str(torch.__version__),
        "git": safe_git_record()})
    try:
        result = execute(cfg, parent, spec, output, report)
        verify_hashes(report["frozen_files"])
        write_json(output / "source_manifest.json", report["frozen_files"])
        write_json(output / "artifact_manifest.json", {
            _relative(item): _file_sha256(item)
            for item in output.rglob("*") if item.is_file()})
        result.update({
            "training_updates": 0,
            "real_slm_actions": False,
            "automatic_retry": False,
            "rl_authorized": False,
            "artifact_manifest_sha256": _file_sha256(output / "artifact_manifest.json"),
            "material_passport": {
                "origin_skill": "academic-research-suite", "origin_mode": "run",
                "origin_date": datetime.now(timezone.utc).isoformat(),
                "verification_status": "REQUIRES_AUDIT",
                "version_label": "r4_candidate_full_closed_loop_v1"},
            "next_action": ("停止；快速冒烟不产生科学结论。" if quick else
                            "停止并通知助手只读审计；不自动训练、进入RL或连接SLM。"),
        })
        write_json(output / "summary.json", result)
        write_json(output / "SUCCESS.json", {
            "summary_sha256": _file_sha256(output / "summary.json")})
        return result
    except Exception:
        write_json(output / "failure.json", {
            "traceback": traceback.format_exc(), "automatic_retry": False})
        raise
