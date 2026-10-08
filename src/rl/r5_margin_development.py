"""冻结 R5 策略的动作力度开发诊断，不训练、不读取确认结果。"""
from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import time
import traceback

import torch

from src.rl.r4_dynamics_experiment import Progress, safe_git_record
from src.rl.r4_interface_smoke import write_json
from src.rl.r4_observation import PowerMeasurement, R4Interface
from src.rl.r4_trajectory import anchor_delta
from src.rl.r5_batched_environment import R5BatchedEnvironment
from src.rl.r5_policy_training import ResidualGRUPolicy
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _profiles, _project_path
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.robust_control import RobustnessCondition


def effective_stream_seeds(base: int, weather_count: int, stride: int,
                           profile_count: int, family_count: int) -> list[int]:
    """按批量环境的实际公式展开每条湍流随机流。"""
    if min(weather_count, stride, profile_count, family_count) < 1:
        raise ValueError("invalid seed schedule")
    streams = [base + weather * stride + profile * 1000 + family
               for weather in range(weather_count)
               for profile in range(profile_count)
               for family in range(family_count)]
    if len(set(streams)) != len(streams):
        raise ValueError("reused turbulence stream across weather or conditions")
    return streams


def preflight(path: str | Path, quick: bool) -> tuple[dict, dict, Path, torch.device]:
    cfg = _load_yaml(_project_path(path))
    if (cfg["stage"] != "S4-D2-R5-D1"
            or cfg["runtime"] != {"device": "cuda", "formal_owner": "user_ide", "automatic_retry": False}
            or cfg["boundary"] != {"confirmation_access": False, "training_updates": 0,
                                    "real_slm_actions": False, "automatic_retry": False}
            or cfg["scales"] != [0.75, 1.0, 1.25]):
        raise ValueError("R5 development contract changed")
    if _file_sha256(_project_path(cfg["parent"])) != cfg["parent_sha256"]:
        raise RuntimeError("R5 training configuration changed")
    parent = _load_yaml(_project_path(cfg["parent"]))
    training = _project_path(cfg["training_output"])
    summary_path = training / "summary.json"
    if (_file_sha256(summary_path) != cfg["training_summary_sha256"]
            or json.loads(summary_path.read_text(encoding="utf-8"))["status"]
            != "R5_2_TRAINING_COMPLETE_REQUIRES_AUDIT"):
        raise RuntimeError("R5 training summary changed")
    if set(cfg["checkpoints"]) != {f"policy_{i}_02000.pt" for i in range(3)}:
        raise ValueError("three final checkpoints required")
    for name, digest in cfg["checkpoints"].items():
        if _file_sha256(training / "checkpoints" / name) != digest:
            raise RuntimeError(f"checkpoint changed: {name}")
    spec = cfg["quick"] if quick else cfg["data"]
    families = parent["families"][:1] if quick else parent["families"]
    profiles = _profiles(parent, spec["profile_ids"] if quick else parent["profile_ids"])
    count, steps = int(spec["weather_count"]), int(spec["episode_length"])
    stride, seed_base = int(cfg["data"]["seed_stride"]), int(spec["seed_base"])
    streams = effective_stream_seeds(seed_base, count, stride, len(profiles), len(families))
    viewed_streams = {5600000 + weather + profile * 1000 + family
                      for weather in range(32) for profile in range(6) for family in range(3)}
    if (set(streams) & viewed_streams
            or any(5500000 <= seed < 5510000 for seed in streams)):
        raise ValueError("development weather overlaps training or viewed confirmation")
    output = _project_path(cfg["quick_directory"] if quick else cfg["output_directory"])
    if output.exists():
        raise FileExistsError(f"preserve existing development output: {output}")
    device = resolve_device("cuda")
    branches = 1 + len(cfg["checkpoints"]) * len(cfg["scales"])
    report = {"status": "READY_FOR_DIAGNOSTIC" if quick else "READY_FOR_USER_IDE",
              "quick": quick, "device": str(device), "weather_count": count,
              "families": len(families), "profiles": len(profiles), "branches": branches,
              "episode_length": steps, "unique_turbulence_streams": len(streams),
              "physical_transitions": branches * count * len(families) * len(profiles) * steps,
              **cfg["boundary"]}
    return cfg, report, output, device


@torch.no_grad()
def _rollout(seed: int, label: str, scale: float, policy: ResidualGRUPolicy | None,
             steps: int, basis: torch.Tensor, base, families: list[dict], profiles: list,
             sensor_offset: int, progress: Progress) -> list[dict]:
    device = basis.device
    condition = RobustnessCondition.from_mapping(dict(families[0], base_seed=seed))
    env_cfg = replace(condition.environment_config(base), batch_size=len(families) * len(profiles),
                      episode_length=steps)
    env = R5BatchedEnvironment(env_cfg, device, basis, families, profiles, sensor_offset)
    raw, _ = env.reset(seed=seed)
    interface = R4Interface()
    interface.reset(env.proxy(raw), episode_id=f"r5-dev-{label}-{seed}")
    names = ("power", "strehl", "phase_rmse", "violation", "saturation",
             "slew_limited", "correction_abs", "requested_step_abs",
             "requested_modal_abs", "applied_modal_abs", "requested_applied_gap_abs")
    sums = {name: torch.zeros(len(families) * len(profiles), device=device, dtype=torch.float64)
            for name in names}
    forward_times: list[float] = []
    for step in range(steps):
        view = interface.snapshot()
        baseline = anchor_delta(view.features[:, -1], {"gain": .15, "leak": .10, "tracking_gain": .50})
        torch.cuda.synchronize(device)
        tick = time.perf_counter()
        correction = (torch.zeros((len(families) * len(profiles), 11), device=device)
                      if policy is None else policy(view.features, view.valid) * scale)
        torch.cuda.synchronize(device)
        forward_times.append(time.perf_counter() - tick)
        action = interface.issue(baseline, correction, step=step)
        raw, _, terminated, truncated, info = env.step(action.requested_delta_rad)
        interface.observe_next(env.proxy(raw), step=step + 1,
                               power=PowerMeasurement(info["measured_power_in_bucket"], step, step + 1))
        values = {"power": info["reward_power_in_bucket"], "strehl": info["reward_strehl"],
                  "phase_rmse": info["reward_phase_rmse"], "violation": info["violation_fraction"],
                  "saturation": info["saturated_fraction"],
                  "slew_limited": info["slew_limited_fraction"],
                  "correction_abs": action.normalized_correction.abs().mean(-1),
                  "requested_step_abs": action.requested_delta_rad.abs().mean(-1),
                  "requested_modal_abs": info["requested_modal"].abs().mean(-1),
                  "applied_modal_abs": info["applied_modal"].abs().mean(-1),
                  "requested_applied_gap_abs": (info["requested_modal"] - info["applied_modal"]).abs().mean(-1)}
        for name, value in values.items():
            sums[name] += value.double()
        if bool(truncated.any()) or bool(terminated.all()) != (step == steps - 1):
            raise RuntimeError("incomplete development episode")
        progress.tick({"天气种子": float(seed), "步": float(step + 1), "策略倍率": scale})
    ordered_times = sorted(forward_times)
    rows = []
    for profile_index, profile in enumerate(profiles):
        for family_index, family in enumerate(families):
            index = profile_index * len(families) + family_index
            row = {"controller": label, "scale": scale, "family": family["id"],
                   "profile": profile.identifier, "weather_seed": seed,
                   "turbulence_stream_seed": seed + profile_index * 1000 + family_index,
                   "policy_forward_seconds_per_step": sum(forward_times) / steps,
                   "policy_forward_p95_seconds": ordered_times[min(steps - 1, math.ceil(.95 * steps) - 1)]}
            row.update({name: float(value[index] / steps) for name, value in sums.items()})
            if any(not math.isfinite(value) for value in row.values() if isinstance(value, float)):
                raise RuntimeError("nonfinite development metric")
            rows.append(row)
    return rows


def run(path: str | Path = "configs/experiments/s4_r5_margin_development_v1.yaml",
        *, quick: bool = False, preflight_only: bool = False) -> dict:
    cfg, report, output, device = preflight(path, quick)
    if preflight_only:
        return report
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "preflight.json", report)
    write_json(output / "config.json", cfg)
    write_json(output / "runtime.json", {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                                         "gpu": torch.cuda.get_device_name(), "git": safe_git_record()})
    progress = Progress(output, device)
    try:
        parent = _load_yaml(_project_path(cfg["parent"]))
        spec = cfg["quick"] if quick else cfg["data"]
        families = parent["families"][:1] if quick else parent["families"]
        profiles = _profiles(parent, spec["profile_ids"] if quick else parent["profile_ids"])
        steps = report["episode_length"]
        base, _ = load_s1_config(_project_path(parent["environment_config"]))
        base = replace(base, num_modes=21, batch_size=1, episode_length=steps)
        basis, _, _ = build_action_basis(base, ActionRepresentation("r5_zernike21", "zernike", 21), device)
        training = _project_path(cfg["training_output"])
        policies = []
        for index in range(3):
            checkpoint = torch.load(training / "checkpoints" / f"policy_{index}_02000.pt",
                                    map_location=device, weights_only=True)
            if checkpoint["init"] != index or checkpoint["update"] != 2000:
                raise RuntimeError("R5 checkpoint identity mismatch")
            policy = ResidualGRUPolicy(parent["policy"]["hidden_size"],
                                       parent["policy"]["output_size"]).to(device)
            policy.load_state_dict(checkpoint["state_dict"])
            policy.eval()
            policies.append(policy)
        branches = [("integrator", 0.0, None)] + [
            (f"policy_{index}_scale_{scale}", float(scale), policy)
            for index, policy in enumerate(policies) for scale in cfg["scales"]]
        progress.phase("R5动作力度开发快速冒烟" if quick else "R5动作力度开发诊断",
                       report["branches"] * report["weather_count"] * steps)
        count = 0
        with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
            for label, scale, policy in branches:
                for weather in range(report["weather_count"]):
                    seed = int(spec["seed_base"]) + weather * int(cfg["data"]["seed_stride"])
                    rows = _rollout(seed, label, scale, policy, steps, basis, base, families,
                                    profiles, parent["data"]["sensor_seed_offset"], progress)
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += len(rows)
                    handle.flush()
        expected = report["branches"] * report["weather_count"] * report["families"] * report["profiles"]
        if count != expected:
            raise RuntimeError("development record count mismatch")
        result = {"status": "QUICK_SMOKE_NO_CONCLUSION" if quick else "DEVELOPMENT_DIAGNOSTIC_REQUIRES_AUDIT",
                  "records": count, "branches": [branch[0] for branch in branches],
                  "weather_count": report["weather_count"],
                  "physical_transitions": report["physical_transitions"],
                  "material_passport": {"origin_skill": "academic-research-suite / experiment-agent",
                                        "origin_mode": "run", "verification_status": "UNVERIFIED"},
                  **cfg["boundary"], "next_action": "停止并等待只读审计，不打开新确认集"}
        write_json(output / "summary.json", result)
        write_json(output / "SUCCESS.json", {"summary_sha256": _file_sha256(output / "summary.json")})
        return result
    except Exception:
        write_json(output / "failure.json", {"traceback": traceback.format_exc(), "automatic_retry": False})
        raise
    finally:
        progress.close()
