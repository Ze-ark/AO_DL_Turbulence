"""R4-0显式诊断运行；物理真值仅用于输出审计，不传入R4Interface。"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
from typing import Any

import torch

from src.rl.r4_control import NominalCalibration, R4Limits
from src.rl.r4_observation import FEATURE_FIELDS, SCHEMA, PowerMeasurement, R4Interface, simulation_residual_proxy
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _project_path, _relative
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.hardware_effects import HardwareEffectsConfig
from src.training_progress import counted_progress, update_progress


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def seed_blocks(obj: Any) -> set[int]:
    """保守扫描全部含seed字段，而不仅是旧接口的base_seed。"""
    if isinstance(obj, dict):
        result: set[int] = set()
        for key, value in obj.items():
            if "seed" in str(key).lower():
                vals = value if isinstance(value, list) else [value]
                result |= {x // 10000 for x in vals if type(x) is int and x >= 0}
            result |= seed_blocks(value)
        return result
    if isinstance(obj, list):
        return set().union(*(seed_blocks(x) for x in obj))
    return set()


def preflight(config_path: Path) -> tuple[dict, dict]:
    cfg = _load_yaml(config_path)
    if (cfg.get("stage") != "S4-D2-R4-0" or cfg.get("purpose") != "diagnostic_interface_smoke_only"
            or cfg.get("runtime") != {"device": "cuda", "require_cuda": True}
            or cfg.get("boundary") != dict(training_updates=0, scientific_ranking=False,
                real_slm_actions=False, s4d3_access=False, automatic_retry=False)):
        raise ValueError("R4-0 requires CUDA diagnostic configuration")
    if (cfg["history_frames"] != 8 or cfg["steps"] != 16 or cfg["episodes_per_case"] != 4
            or cfg["pulse_step"] != 2 or cfg["pulse_mode"] != 10 or len(cfg["cases"]) != 3):
        raise ValueError("R4-0 diagnostic budget changed")
    device = resolve_device(cfg["runtime"]["device"])
    if device.type != "cuda":
        raise RuntimeError("R4-0 CUDA required")
    output = _project_path(cfg["output_directory"]).resolve()
    if output != _project_path("outputs/s4_r4_interface_smoke_v1").resolve():
        raise ValueError("R4-0 output must use dedicated directory")
    if output.exists():
        raise FileExistsError(f"R4-0 output exists; preserve it: {output}")
    design_path = _project_path(cfg["design"])
    if _file_sha256(design_path) != cfg["design_sha256"]:
        raise RuntimeError("R4 design hash mismatch")
    design = _load_yaml(design_path)
    upstream = design["upstream"]
    source_summary = _project_path(upstream["summary"])
    if _file_sha256(source_summary) != upstream["summary_sha256"]:
        raise RuntimeError("R4 upstream hash mismatch")
    if json.loads(source_summary.read_text(encoding="utf-8"))["interpretation"]["status"] != upstream["required_status"]:
        raise RuntimeError("R4 upstream status mismatch")
    nominal = NominalCalibration(**cfg["nominal_calibration"])
    nominal.validate()
    cases = [(c["delay_frames"], c["settling_fraction"]) for c in cfg["cases"]]
    if cases != [(0, 1.0), (3, 1.0), (3, 0.5)]:
        raise ValueError("R4-0 impulse cases changed")
    seeds = [c["base_seed"] + i for c in cfg["cases"] for i in range(cfg["episodes_per_case"])]
    if len(set(seeds)) != 12 or any(x < 0 or x >= 4000000 or x // 10000 == 378 for x in seeds):
        raise ValueError("R4-0 reserved/duplicate seeds")
    scan_files = list(_project_path("configs").rglob("*.yaml"))
    for filename in ("effective_config.json", "data_manifest.json", "preflight.json"):
        scan_files += list(_project_path("outputs").glob("*/" + filename))
    historical: set[int] = set()
    hashes: dict[str, str] = {}
    for path in scan_files:
        if path.resolve() == config_path.resolve():
            continue
        obj = _load_yaml(path) if path.suffix == ".yaml" else json.loads(path.read_text(encoding="utf-8"))
        historical |= seed_blocks(obj)
        hashes[_relative(path)] = _file_sha256(path)
    if {x // 10000 for x in seeds} & historical:
        raise RuntimeError("R4-0 historical seed namespace collision")
    sources = list(_project_path("src/simulation").glob("*.py")) + [
        config_path, design_path, source_summary, _project_path(cfg["environment_config"]),
        _project_path("src/rl/r4_control.py"), _project_path("src/rl/r4_observation.py"),
        _project_path("src/rl/r4_interface_smoke.py"), _project_path("src/rl/residual_control.py"),
        _project_path("src/rl/s4_representation_capacity.py"), _project_path("src/runtime.py"),
        _project_path("scripts/run_s4_r4_interface_smoke.py"),
    ]
    hashes.update({_relative(p): _file_sha256(p) for p in sources})
    return cfg, dict(status="READY_FOR_CUDA_INTERFACE_SMOKE", device=str(device),
                     episode_seeds=seeds, historical_files_scanned=len(scan_files)-1,
                     frozen_files=hashes, weather_episodes=12, candidate_episodes=36,
                     expected_transitions=576, expected_batch_steps=144)


def run_interface_smoke(config_path: str | Path, *, preflight_only: bool = False) -> dict:
    cfg, report = preflight(_project_path(config_path))
    if preflight_only:
        return {k: v for k, v in report.items() if k != "frozen_files"}
    output = _project_path(cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "preflight.json", report)
    write_json(output / "effective_config.json", cfg)
    try:
        result = execute(cfg, output, report)
        write_json(output / "summary.json", result)
        write_json(output / "SUCCESS.json", dict(status=result["status"],
                   summary_sha256=_file_sha256(output / "summary.json")))
        return result
    except Exception as exc:
        write_json(output / "failure.json", dict(error=str(exc), traceback=traceback.format_exc(),
                                                 automatic_retry=False))
        raise


def execute(cfg: dict, output: Path, report: dict) -> dict:
    started = time.perf_counter()
    device = resolve_device("cuda")
    torch.use_deterministic_algorithms(True)
    base, _ = load_s1_config(_project_path(cfg["environment_config"]))
    base = replace(base, num_modes=21, batch_size=4, episode_length=cfg["steps"])
    basis, _, basis_diagnostics = build_action_basis(
        base, ActionRepresentation("r4_zernike21", "zernike", 21), device)
    limits = R4Limits(modal_rad=base.modal_limit_rad)
    bar = counted_progress(total=144, description="R4-0 因果接口检查", unit="批次")
    summaries, response_checks = [], []
    with (output / "progress.jsonl").open("w", encoding="utf-8") as log:
        for case in cfg["cases"]:
            paired_applied: dict[str, torch.Tensor] = {}
            for branch, sign in (("zero", 0.0), ("positive", 1.0), ("negative", -1.0)):
                env_cfg = replace(base, slm_delay_frames=case["delay_frames"])
                env = AdaptiveOpticsEnv(env_cfg, device,
                    HardwareEffectsConfig(settling_fraction=case["settling_fraction"]), basis_override=basis)
                raw, _ = env.reset(seed=case["base_seed"])
                generator = torch.Generator(device=device).manual_seed(case["base_seed"] + 50000000)
                adapter = R4Interface(limits=limits, calibration=NominalCalibration(**cfg["nominal_calibration"]))
                residual = simulation_residual_proxy(raw, generator=generator,
                    noise_std_rad=cfg["observation_noise_std_rad"])
                adapter.reset(residual, episode_id=f"{case['id']}/{branch}")
                records: dict[str, list[torch.Tensor]] = {k: [] for k in (
                    "history", "history_valid", "requested_delta_rad", "requested_modal_rad",
                    "correction_normalized", "next_residual", "action_power", "action_power_valid",
                    "audit_applied_modal_rad", "audit_reward_power", "audit_next_power", "audit_violation",
                    "audit_reward_strehl", "audit_reward_phase_rmse", "estimated_applied_modal_rad")}
                alignment_error = 0.0
                for t in range(cfg["steps"]):
                    correction = torch.zeros(4, 11, device=device)
                    if t == cfg["pulse_step"]:
                        correction[:, cfg["pulse_mode"] - 10] = sign
                    if t == cfg["pulse_step"] + 1:
                        correction[:, cfg["pulse_mode"] - 10] = -sign
                    action = adapter.issue(torch.zeros(4, 21, device=device), correction, step=t)
                    raw, _, terminated, truncated, info = env.step(action.requested_delta_rad)
                    residual = simulation_residual_proxy(raw, generator=generator,
                        noise_std_rad=cfg["observation_noise_std_rad"])
                    transition = adapter.observe_next(residual, step=t+1, power=PowerMeasurement(
                        info["measured_power_in_bucket"], action_step=t, arrival_observation_step=t+1))
                    # 真值只能在此处审计；策略接口从未接收info或env。
                    alignment_error = max(alignment_error,
                        float((action.requested_modal_rad - info["requested_modal"]).abs().max()))
                    assert transition.action_step == t and transition.next_observation_step == t + 1
                    assert torch.equal(transition.action_power, info["measured_power_in_bucket"])
                    assert bool(terminated.all()) == (t == cfg["steps"] - 1) and not bool(truncated.any())
                    vals = dict(history=transition.history.features, history_valid=transition.history.valid,
                        requested_delta_rad=action.requested_delta_rad, requested_modal_rad=action.requested_modal_rad,
                        correction_normalized=action.normalized_correction, next_residual=transition.next_residual,
                        action_power=transition.action_power, action_power_valid=transition.action_power_valid,
                        audit_applied_modal_rad=info["applied_modal"], audit_reward_power=info["reward_power_in_bucket"],
                        audit_next_power=raw[:, -1], audit_violation=info["violation_fraction"],
                        audit_reward_strehl=info["reward_strehl"], audit_reward_phase_rmse=info["reward_phase_rmse"],
                        estimated_applied_modal_rad=adapter.estimator.current)
                    for key, value in vals.items():
                        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                            raise RuntimeError(f"R4-0 non-finite output: {key}")
                        records[key].append(value.detach().cpu())
                    bar.update(1)
                    elapsed = time.perf_counter() - started
                    log.write(json.dumps(dict(completed=bar.n, total=144, case=case["id"], branch=branch,
                        step=t, elapsed_seconds=elapsed, eta_seconds=elapsed/bar.n*(144-bar.n),
                        cuda_allocated_gb=torch.cuda.memory_allocated(device)/1024**3), ensure_ascii=False)+"\n")
                    log.flush()
                    update_progress(bar, device=device, metrics={"对齐误差": alignment_error})
                if alignment_error > 1e-6:
                    raise RuntimeError("R4-0 requested command alignment failed")
                tensors = {k: torch.stack(v) for k, v in records.items()}
                paired_applied[branch] = tensors["audit_applied_modal_rad"]
                artifact = output / f"{case['id']}_{branch}.pt"
                torch.save(dict(schema=SCHEMA, data_source="simulation_residual_proxy_not_holography",
                    feature_fields=FEATURE_FIELDS, tensors=tensors, dt_s=base.dt_s,
                    action_steps=list(range(cfg["steps"])), next_observation_steps=list(range(1,cfg["steps"]+1)),
                    episode_seeds=[case["base_seed"]+i for i in range(4)], case=case, branch=branch,
                    nominal_calibration=cfg["nominal_calibration"], **cfg["boundary"]), artifact)
                summaries.append(dict(case=case["id"], branch=branch, transitions=64,
                    requested_alignment_max_error=alignment_error, file=_relative(artifact),
                    sha256=_file_sha256(artifact), actual_violation_mean=float(tensors["audit_violation"].mean()),
                    nominal_estimate_max_error=float((tensors["estimated_applied_modal_rad"]-
                                                     tensors["audit_applied_modal_rad"]).abs().max())))
            for branch in ("positive", "negative"):
                delta = paired_applied[branch] - paired_applied["zero"]
                expected_step = cfg["pulse_step"] + case["delay_frames"]
                for episode in range(4):
                    responsive = torch.nonzero(delta[:, episode].abs().amax(-1) > cfg["response_tolerance_rad"]).flatten()
                    onset = int(responsive[0]) if len(responsive) else None
                    passed = onset == expected_step and onset - cfg["pulse_step"] < 8
                    response_checks.append(dict(case=case["id"], branch=branch,
                        episode_seed=case["base_seed"]+episode, expected_action_step=expected_step,
                        observed_action_step=onset, passed=passed))
    bar.close()
    write_json(output / "response_checks.json", response_checks)
    if not all(r["passed"] for r in response_checks):
        raise RuntimeError("R4-0 delayed pulse onset failed; inspect response_checks.json")
    for path, digest in report["frozen_files"].items():
        if _file_sha256(_project_path(path)) != digest:
            raise RuntimeError(f"R4-0 frozen source/input changed: {path}")
    return dict(material_passport=dict(origin_skill="academic-research-suite / experiment-agent",
        origin_mode="run", origin_date=datetime.now(timezone.utc).isoformat(),
        verification_status="UNVERIFIED", version_label="r4_0_interface_smoke_v1"),
        status="R4_0_INTERFACE_SMOKE_PASS", scientific_status="DIAGNOSTIC_ONLY",
        device=str(device), gpu_name=torch.cuda.get_device_name(device),
        duration_seconds=time.perf_counter()-started, weather_episodes=12, candidate_episodes=36,
        transitions=sum(r["transitions"] for r in summaries), pulse_checks=len(response_checks),
        basis_diagnostics=basis_diagnostics, limits=asdict(limits), datasets=summaries,
        schema=SCHEMA, history_shape=[4, 8, 79], **cfg["boundary"],
        next_action="只读验收后进入R4-1连续轨迹采集与模型实现；本次没有训练模型。")
