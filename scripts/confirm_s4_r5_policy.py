from __future__ import annotations

import argparse, json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from dataclasses import replace

from src.rl.r4_baselines import baseline_delta
from src.rl.r4_dynamics_experiment import Progress, safe_git_record
from src.rl.r4_interface_smoke import write_json
from src.rl.r4_observation import PowerMeasurement, R4Interface
from src.rl.r4_trajectory import anchor_delta
from src.rl.r5_batched_environment import R5BatchedEnvironment
from src.rl.r5_physics_adapter import causal_policy_features
from src.rl.r5_policy_training import ResidualGRUPolicy
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _profiles, _project_path
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.robust_control import RobustnessCondition


def _interval(values: torch.Tensor, seed: int, repeats: int) -> dict:
    # values: family x weather; weather is the statistical unit.
    g = torch.Generator(device=values.device).manual_seed(seed)
    draws = torch.randint(values.shape[1], (repeats, values.shape[1]), generator=g, device=values.device)
    samples = torch.stack([values[f][draws].mean(1) for f in range(values.shape[0])], 1).mean(1)
    q = torch.quantile(samples.double(), values.new_tensor([.025, .975], dtype=torch.float64))
    return {"mean": float(values.double().mean()), "ci95": [float(q[0]), float(q[1])],
            "family_means": [float(values[f].double().mean()) for f in range(values.shape[0])]}


def preflight(path: str, quick: bool) -> tuple[dict, Path, torch.device]:
    cfg = _load_yaml(_project_path(path))
    if cfg["stage"] not in ("S4-D2-R5-3", "S4-D2-R5-3-R1") or cfg["runtime"] != {"device": "cuda", "formal_owner": "user_ide", "automatic_retry": False}:
        raise ValueError("R5-3 frozen CUDA/user-owned configuration required")
    train_cfg = _load_yaml(_project_path(cfg["parent"]))
    training = _project_path(cfg["training_output"])
    summary = json.loads((training / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "R5_2_TRAINING_COMPLETE_REQUIRES_AUDIT" or not summary["confirmation_access"] is False:
        raise RuntimeError("R5-2 training prerequisite is not frozen")
    if len(cfg["checkpoint_initializations"]) != 3:
        raise ValueError("three frozen policy members are required")
    for init in cfg["checkpoint_initializations"]:
        p = training / "checkpoints" / f"policy_{init}_{cfg['checkpoint_update']:05d}.pt"
        if not p.exists(): raise FileNotFoundError(p)
        torch.load(p, map_location="cpu", weights_only=True)
    out = _project_path(cfg["quick_directory"] if quick else cfg["output_directory"])
    if out.exists(): raise FileExistsError(f"preserve existing confirmation output: {out}")
    seed_base = cfg["data"]["seed_base"]
    if seed_base // 10000 in {550, 540, 520, 528}:
        raise ValueError("confirmation seed namespace overlaps prior experiments")
    device = resolve_device("cuda")
    return cfg, out, device


@torch.no_grad()
def rollout(cfg: dict, quick: bool, seed: int, controller: str, model: torch.nn.Module | None,
            device: torch.device, basis: torch.Tensor, base, profiles: list, families: list[dict], progress: Progress) -> dict:
    steps = cfg["quick"]["episode_length"] if quick else cfg["data"]["episode_length"]
    env_cfg = replace(RobustnessCondition.from_mapping(dict(families[0], base_seed=seed)).environment_config(base),
                      batch_size=len(families) * len(profiles), episode_length=steps)
    env = R5BatchedEnvironment(env_cfg, device, basis, families, profiles, cfg["data"]["sensor_seed_offset"])
    raw, _ = env.reset(seed=seed)
    interface = R4Interface(); interface.reset(env.proxy(raw), episode_id=f"r5-confirm-{controller}-{seed}")
    previous = torch.zeros((len(families) * len(profiles), 11), device=device)
    sums = {k: torch.zeros(len(families) * len(profiles), device=device) for k in ("power", "measured_power", "strehl", "phase_rmse", "violation")}
    for t in range(steps):
        view = interface.snapshot()
        base_delta = anchor_delta(view.features[:, -1], {"gain": .15, "leak": .10, "tracking_gain": .50})
        correction = torch.zeros_like(previous) if model is None else model(view.features, view.valid)
        action = interface.issue(base_delta, correction, step=t)
        raw, _, terminated, truncated, info = env.step(action.requested_delta_rad)
        interface.observe_next(env.proxy(raw), step=t + 1,
                               power=PowerMeasurement(info["measured_power_in_bucket"], t, t + 1))
        sums["power"] += info["reward_power_in_bucket"]
        sums["measured_power"] += info["measured_power_in_bucket"]
        sums["strehl"] += info["reward_strehl"]
        sums["phase_rmse"] += info["reward_phase_rmse"]
        sums["violation"] += info["violation_fraction"]
        previous = action.normalized_correction
        if bool(truncated.any()) or bool(terminated.all()) != (t == steps - 1): raise RuntimeError("invalid confirmation episode")
        # 进度条格式化器只接受数值；控制器名称和种子保存在逐回合记录中。
        progress.tick({"天气种子": float(seed), "步": float(t + 1)})
    n = float(steps)
    rows = []
    for i, (family, profile) in enumerate((f, p) for p in profiles for f in families):
        rows.append({"controller": controller, "family": family["id"], "profile": profile.identifier,
                     "weather_seed": seed, **{k: float(v[i] / n) for k, v in sums.items()}})
    return rows


def run(path: str, quick: bool = False, preflight_only: bool = False) -> dict:
    cfg, out, device = preflight(path, quick)
    if preflight_only: return {"status": "READY_FOR_DIAGNOSTIC" if quick else "READY_FOR_USER_IDE", "device": str(device), "quick": quick}
    out.mkdir(parents=True); write_json(out / "preflight.json", {"quick": quick, "device": str(device), "git": safe_git_record()})
    base, _ = load_s1_config(_project_path(_load_yaml(_project_path(cfg["parent"]))["environment_config"]))
    base = replace(base, num_modes=21, batch_size=1, episode_length=cfg["data"]["episode_length"])
    basis, _, _ = build_action_basis(base, ActionRepresentation("r5_zernike21", "zernike", 21), device)
    parent = _load_yaml(_project_path(cfg["parent"]))
    profiles = _profiles(parent, cfg["quick"]["profile_ids"] if quick else cfg["profile_ids"])
    families = cfg["families"][:1] if quick else cfg["families"]
    count = cfg["quick"]["weather_per_family"] if quick else cfg["data"]["weather_per_family"]
    # 修订版：一个天气种子只运行一次三家族批次，避免外层重复运行整批家族。
    seeds = [cfg["data"]["seed_base"] + j for j in range(count)]
    checkpoints = []
    for init in cfg["checkpoint_initializations"]:
        p = _project_path(cfg["training_output"]) / "checkpoints" / f"policy_{init}_{cfg['checkpoint_update']:05d}.pt"
        ck = torch.load(p, map_location=device, weights_only=True); model = ResidualGRUPolicy(64, 11).to(device); model.load_state_dict(ck["state_dict"]); model.eval(); checkpoints.append((f"policy_{init}", model))
    controllers = [("integrator", None), *checkpoints]
    total = len(controllers) * count * cfg["quick" if quick else "data"]["episode_length"]
    progress = Progress(out, device); progress.phase("R5-3独立确认快速冒烟" if quick else "R5-3独立确认", total)
    rows = []
    try:
        for name, model in controllers:
            for seed in seeds:
                rows.extend(rollout(cfg, quick, seed, name, model, device, basis, base, profiles, families, progress))
        expected_rows = len(controllers) * count * len(families) * len(profiles)
        if len(rows) != expected_rows or len({r["weather_seed"] for r in rows if r["controller"] == "integrator"}) != count:
            raise RuntimeError("R5-3-R1 record or seed-layout contract failed")
        (out / "records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        keys = ["power", "measured_power", "strehl", "phase_rmse", "violation"]
        grouped = {}
        for controller in [x[0] for x in controllers]:
            grouped[controller] = {k: [[r[k] for r in rows if r["controller"] == controller and r["family"] == f["id"]] for f in families] for k in keys}
        result = {"status": "QUICK_COMPLETE_NO_CONCLUSION" if quick else "R5_3_CONFIRMATION_COMPLETE_REQUIRES_AUDIT", "controllers": [x[0] for x in controllers], "records": len(rows), "grouped": grouped, "confirmation_access": False if quick else True, "real_slm_actions": False, "training_updates": 0, "next_action": "停止并等待只读统计审计"}
        write_json(out / "summary.json", result); write_json(out / "SUCCESS.json", {"summary_sha256": _file_sha256(out / "summary.json")}); return result
    finally: progress.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/experiments/s4_r5_independent_confirmation_v1.yaml"); ap.add_argument("--quick", action="store_true"); ap.add_argument("--preflight-only", action="store_true")
    print(json.dumps(run(ap.parse_args().config, ap.parse_args().quick, ap.parse_args().preflight_only), ensure_ascii=False, indent=2))
