"""R5-D2 新开发天气动作力度诊断；正式运行仅由用户在 IDE 启动。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rl.r4_dynamics_experiment import Progress, safe_git_record
from src.rl.r4_interface_smoke import write_json
from src.rl.r5_independent_confirmation_r2 import source_bundle_sha256
from src.rl.r5_margin_development import _rollout, effective_stream_seeds
from src.rl.r5_policy_training import ResidualGRUPolicy
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _profiles, _project_path
from src.runtime import resolve_device
from src.simulation.config import load_s1_config


SCALES = (1.25, 1.5, 1.75, 2.0)
FAMILIES = ("frozen", "boiling", "varying")
PROFILES = ("nominal", "delay_3", "settling_050", "registration_moderate",
            "registration_severe", "combined_moderate")
SAFETY_METRICS = ("violation", "saturation", "slew_limited")
FROZEN_PARENT_SHA256 = "3f8f2cd5a8ffacd08e91f1606889f97a94b175b574180a725a0f706d2e08498e"
FROZEN_TRAINING_SUMMARY_SHA256 = "2bcd9a87290cd7d37eeaafcee5c7f2b08f5a42ec58fe88d4e5e0fc99ac8efa5d"
FROZEN_CHECKPOINTS = {
    "policy_0_02000.pt": "236f4ee3b83885181555b064e37391cb9fc8b93c260bcd076effa753620c78d0",
    "policy_1_02000.pt": "653600beb3829ec62ac901d2447322fc4dafaf25e32dafde8f51fe65e9a3929c",
    "policy_2_02000.pt": "2b4589eb6fdc93f6dd042f65b632b982f96bfb3042db98280d97ab8a9249ca49",
}
FROZEN_SOURCE_BUNDLE_SHA256 = "ccdc31faaa155361d8bd3e19ffc5bb82705f425a0ef4c7f58dd7f70831efd956"


class CorrectionClampTelemetry(nn.Module):
    """只旁路统计倍率后超出[-1,1]的比例；前向数值保持原策略不变。"""

    def __init__(self, policy: ResidualGRUPolicy, scale: float):
        super().__init__()
        self.policy = policy
        self.scale = scale
        self.clipped: torch.Tensor | None = None
        self.steps = 0

    def forward(self, history: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        raw = self.policy(history, valid)
        clipped = (raw * self.scale).abs().gt(1).to(torch.float64).mean(-1)
        self.clipped = clipped if self.clipped is None else self.clipped + clipped
        self.steps += 1
        return raw

    def rates(self, expected_steps: int, batch: int) -> list[float]:
        if self.steps != expected_steps or self.clipped is None or self.clipped.shape != (batch,):
            raise RuntimeError("incomplete correction clamp telemetry")
        return (self.clipped / expected_steps).detach().cpu().tolist()


def _contract(cfg: dict) -> None:
    if (cfg.get("stage") != "S4-D2-R5-D2"
            or cfg.get("runtime") != {"device": "cuda", "formal_owner": "user_ide",
                                      "automatic_retry": False}
            or cfg.get("boundary") != {"confirmation_access": False, "training_updates": 0,
                                       "real_slm_actions": False, "automatic_retry": False}
            or cfg.get("scales") != list(SCALES)
            or cfg.get("data") != {"seed_base": 6200000, "seed_stride": 10,
                                   "weather_count": 32, "episode_length": 200}
            or cfg.get("quick") != {"seed_base": 6210000, "weather_count": 1,
                                    "episode_length": 16,
                                    "profile_ids": ["nominal", "combined_moderate"]}
            or cfg.get("selection") != {
                "rule": "highest_three_member_mean_paired_power_gain",
                "tie_break": "smaller_scale",
                "require_each_member_positive": True,
                "require_each_family_positive": True,
                "require_each_profile_positive": True,
                "require_strehl_non_decrease": True,
                "require_phase_rmse_non_increase": True,
                "maximum_safety_increase": 0.001,
                "minimum_development_relative_gain_for_new_confirmation_design": 0.0105,
            }):
        raise ValueError("R5-D2 frozen development contract changed")
    if cfg.get("output_directory") != "outputs/s4_r5_margin_development_d2_v1" or cfg.get(
            "quick_directory") != "outputs/s4_r5_margin_development_d2_v1_quick_r1":
        raise ValueError("R5-D2 output contract changed")
    if cfg.get("training_output") != "outputs/s4_r5_policy_training_v4" or cfg.get(
            "parent") != "configs/experiments/s4_r5_policy_training_v1.yaml":
        raise ValueError("R5-D2 parent contract changed")
    if (cfg.get("parent_sha256") != FROZEN_PARENT_SHA256
            or cfg.get("training_summary_sha256") != FROZEN_TRAINING_SUMMARY_SHA256
            or cfg.get("checkpoints") != FROZEN_CHECKPOINTS
            or cfg.get("old_source_bundle_sha256") != FROZEN_SOURCE_BUNDLE_SHA256):
        raise ValueError("R5-D2 frozen source or policy identities changed")


def _stream_manifest(quick: bool, family_count: int,
                     profile_count: int) -> dict[str, list[int]]:
    formal = effective_stream_seeds(6200000, 32, 10, len(PROFILES), len(FAMILIES))
    smoke = effective_stream_seeds(6210000, 1, 10, 2, 1)
    if len(formal) != 576 or len(smoke) != 2 or set(formal) & set(smoke):
        raise RuntimeError("D2 formal/quick turbulence streams overlap")
    # R5历史实际使用的湍流种子位于5.2M--5.71M；本轮单独保留6.2M--6.22M。
    # 传感器和功率流分别加50M、60M，同样保持独立。
    if not (min(formal) >= 6200000 and max(formal) < 6210000
            and min(smoke) >= 6210000 and max(smoke) < 6220000
            and min(formal + smoke) > 5800000):
        raise RuntimeError("D2 weather namespace intersects historical R5 blocks")
    selected = smoke if quick else formal
    weather_bases = [6210000] if quick else [6200000 + 10 * i for i in range(32)]
    sensor = [seed + profile * 1000 + 50000000 for seed in weather_bases
              for profile in range(profile_count)]
    power = [seed + profile * 1000 + 60000000 for seed in weather_bases
             for profile in range(profile_count)]
    if (len(selected) != len(weather_bases) * family_count * profile_count
            or len(set(sensor)) != len(sensor) or len(set(power)) != len(power)):
        raise RuntimeError("D2 stream layout mismatch")
    return {"weather_bases": weather_bases, "turbulence": selected,
            "sensor": sensor, "power": power}


def preflight(path: str | Path = "configs/experiments/s4_r5_margin_development_d2_v1.yaml",
              *, quick: bool = False) -> tuple[dict, dict, Path, torch.device]:
    cfg = _load_yaml(_project_path(path))
    _contract(cfg)
    if source_bundle_sha256() != cfg["old_source_bundle_sha256"]:
        raise RuntimeError("frozen R5-R2 source bundle changed")
    parent_path = _project_path(cfg["parent"])
    if _file_sha256(parent_path) != cfg["parent_sha256"]:
        raise RuntimeError("R5 training configuration changed")
    parent = _load_yaml(parent_path)
    training = _project_path(cfg["training_output"])
    summary_path = training / "summary.json"
    if (_file_sha256(summary_path) != cfg["training_summary_sha256"]
            or json.loads(summary_path.read_text(encoding="utf-8"))["status"]
            != "R5_2_TRAINING_COMPLETE_REQUIRES_AUDIT"):
        raise RuntimeError("R5 training summary changed")
    if set(cfg["checkpoints"]) != {f"policy_{i}_02000.pt" for i in range(3)}:
        raise ValueError("three frozen final checkpoints required")
    for name, expected in cfg["checkpoints"].items():
        if _file_sha256(training / "checkpoints" / name) != expected:
            raise RuntimeError(f"R5 checkpoint changed: {name}")
    spec = cfg["quick"] if quick else cfg["data"]
    families = parent["families"][:1] if quick else parent["families"]
    profiles = _profiles(parent, spec["profile_ids"] if quick else parent["profile_ids"])
    if ([family["id"] for family in parent["families"]] != list(FAMILIES)
            or parent["profile_ids"] != list(PROFILES)
            or len(families) != (1 if quick else 3)
            or len(profiles) != (2 if quick else 6)):
        raise ValueError("R5 family/profile layout changed")
    streams = _stream_manifest(quick, len(families), len(profiles))
    output = _project_path(cfg["quick_directory"] if quick else cfg["output_directory"])
    if output.exists():
        raise FileExistsError(f"preserve existing R5-D2 output: {output}")
    device = resolve_device("cuda")
    count, steps = int(spec["weather_count"]), int(spec["episode_length"])
    branches = 1 + len(cfg["checkpoints"]) * len(cfg["scales"])
    report = {"status": "READY_FOR_DIAGNOSTIC" if quick else "READY_FOR_USER_IDE",
              "quick": quick, "device": str(device), "weather_count": count,
              "families": len(families), "profiles": len(profiles),
              "branches": branches, "episode_length": steps,
              "physical_transitions": branches * count * len(families) * len(profiles) * steps,
              "config_sha256": _file_sha256(_project_path(path)),
              "source_bundle_sha256": cfg["old_source_bundle_sha256"],
              "entry_sha256": _file_sha256(Path(__file__)),
              "unique_turbulence_streams": len(streams["turbulence"]),
              "unique_sensor_streams": len(streams["sensor"]),
              "unique_power_streams": len(streams["power"]),
              "historical_r5_stream_upper_exclusive": 5800000,
              **cfg["boundary"]}
    return cfg, report, output, device


def summarize_development(rows: list[dict], cfg: dict, *, quick: bool) -> dict:
    """按配对完整天气汇总；冒烟可以验算，但永远不给正式选型结论。"""
    spec = cfg["quick"] if quick else cfg["data"]
    families = FAMILIES[:1] if quick else FAMILIES
    profiles = ("nominal", "combined_moderate") if quick else PROFILES
    weather = [spec["seed_base"] + i * cfg["data"]["seed_stride"]
               for i in range(spec["weather_count"])]
    branches = ("integrator",) + tuple(
        f"policy_{member}_scale_{scale}" for member in range(3) for scale in SCALES)
    expected = len(branches) * len(weather) * len(families) * len(profiles)
    by_key = {(r["controller"], r["weather_seed"], r["family"], r["profile"]): r
              for r in rows}
    if len(rows) != expected or len(by_key) != expected:
        raise RuntimeError("D2 rows incomplete or duplicated")
    keys = [(seed, family, profile) for seed in weather for family in families
            for profile in profiles]
    if set(by_key) != {(branch, *key) for branch in branches for key in keys}:
        raise RuntimeError("D2 paired controller conditions misaligned")
    for row in rows:
        if (row["scale"] not in (0.0, *SCALES)
                or not math.isfinite(row["normalized_correction_clipped_fraction"])
                or not 0 <= row["normalized_correction_clipped_fraction"] <= 1):
            raise RuntimeError("D2 invalid correction clipping telemetry")
    if quick:
        return {"status": "QUICK_SMOKE_NO_CONCLUSION",
                "record_count": len(rows), "paired_conditions": len(keys),
                "clipping_telemetry_present": True, "selected_scale": None,
                "development_go_for_new_confirmation_design": False,
                "confirmation_not_run": True}
    mean = lambda values: math.fsum(values) / len(values)
    base = {key: by_key[("integrator", *key)] for key in keys}
    baseline_power = mean([base[key]["power"] for key in keys])
    if baseline_power <= 0:
        raise RuntimeError("D2 nonpositive baseline power")
    comparisons = []
    safety_limit = cfg["selection"]["maximum_safety_increase"]
    for scale in SCALES:
        selected = {(member, *key): by_key[(f"policy_{member}_scale_{scale}", *key)]
                    for member in range(3) for key in keys}
        def delta(metric: str, subset: list[tuple[int, str, str]] = keys) -> float:
            return mean([selected[(member, *key)][metric] - base[key][metric]
                         for member in range(3) for key in subset])
        absolute = delta("power")
        member = [mean([selected[(m, *key)]["power"] - base[key]["power"]
                        for key in keys]) for m in range(3)]
        family = {name: delta("power", [key for key in keys if key[1] == name])
                  for name in families}
        profile = {name: delta("power", [key for key in keys if key[2] == name])
                   for name in profiles}
        other = {name: delta(name) for name in ("strehl", "phase_rmse", *SAFETY_METRICS)}
        eligible = (all(value > 0 for value in member)
                    and all(value > 0 for value in family.values())
                    and all(value > 0 for value in profile.values())
                    and other["strehl"] >= 0 and other["phase_rmse"] <= 0
                    and all(other[name] <= safety_limit for name in SAFETY_METRICS))
        comparisons.append({"scale": scale, "absolute_power_gain": absolute,
                            "relative_power_gain": absolute / baseline_power,
                            "member_absolute_gains": member,
                            "family_absolute_gains": family,
                            "profile_absolute_gains": profile,
                            "other_deltas": other,
                            "normalized_correction_clipped_fraction": mean([
                                selected[(m, *key)]["normalized_correction_clipped_fraction"]
                                for m in range(3) for key in keys]),
                            "eligible": eligible})
    eligible = [item for item in comparisons if item["eligible"]]
    selected = min(eligible, key=lambda item: (-item["absolute_power_gain"], item["scale"])) \
        if eligible else None
    threshold = cfg["selection"]["minimum_development_relative_gain_for_new_confirmation_design"]
    return {"status": "DEVELOPMENT_ONLY_REQUIRES_AUDIT",
            "selection_rule": cfg["selection"], "baseline_power": baseline_power,
            "candidate_comparisons": comparisons,
            "selected_scale": None if selected is None else selected["scale"],
            "development_go_for_new_confirmation_design": bool(
                not quick and selected is not None and selected["relative_power_gain"] >= threshold),
            "confirmation_not_run": True}


def run(path: str | Path = "configs/experiments/s4_r5_margin_development_d2_v1.yaml",
        *, quick: bool = False, preflight_only: bool = False) -> dict:
    cfg, report, output, device = preflight(path, quick=quick)
    if preflight_only:
        return report
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "preflight.json", report)
    write_json(output / "config.json", cfg)
    stream_manifest = _stream_manifest(quick, report["families"], report["profiles"])
    write_json(output / "stream_manifest.json", stream_manifest)
    write_json(output / "runtime.json", {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                                         "gpu": torch.cuda.get_device_name(), "git": safe_git_record()})
    progress = Progress(output, device)
    started = time.perf_counter()
    try:
        parent = _load_yaml(_project_path(cfg["parent"]))
        spec = cfg["quick"] if quick else cfg["data"]
        families = parent["families"][:1] if quick else parent["families"]
        profiles = _profiles(parent, spec["profile_ids"] if quick else parent["profile_ids"])
        steps = report["episode_length"]
        base, _ = load_s1_config(_project_path(parent["environment_config"]))
        base = replace(base, num_modes=21, batch_size=1, episode_length=steps)
        basis, _, _ = build_action_basis(
            base, ActionRepresentation("r5_zernike21", "zernike", 21), device)
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
            (f"policy_{index}_scale_{scale}", scale, policy)
            for index, policy in enumerate(policies) for scale in SCALES]
        progress.phase("R5-D2动作力度快速冒烟" if quick else "R5-D2动作力度开发诊断",
                       report["branches"] * report["weather_count"] * steps)
        rows: list[dict] = []
        with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
            for label, scale, policy in branches:
                for weather in range(report["weather_count"]):
                    seed = int(spec["seed_base"]) + weather * int(cfg["data"]["seed_stride"])
                    meter = None if policy is None else CorrectionClampTelemetry(policy, scale)
                    episode = _rollout(seed, label, scale, meter, steps, basis, base, families,
                                       profiles, parent["data"]["sensor_seed_offset"], progress)
                    rates = ([0.0] * len(episode) if meter is None
                             else meter.rates(steps, len(episode)))
                    for row, rate in zip(episode, rates, strict=True):
                        row["normalized_correction_clipped_fraction"] = rate
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        rows.append(row)
                    handle.flush()
        analysis = summarize_development(rows, cfg, quick=quick)
        result = {"status": analysis["status"], "records": len(rows),
                  "branches": [branch[0] for branch in branches],
                  "weather_count": report["weather_count"],
                  "physical_transitions": report["physical_transitions"],
                  "elapsed_seconds": time.perf_counter() - started,
                  "records_sha256": _file_sha256(output / "records.jsonl"),
                  "stream_manifest_sha256": _file_sha256(output / "stream_manifest.json"),
                  "source_bundle_sha256": report["source_bundle_sha256"],
                  "entry_sha256": report["entry_sha256"],
                  "development_analysis": analysis,
                  "material_passport": {"origin_skill": "academic-research-suite / experiment-agent",
                                        "origin_mode": "run", "verification_status": "UNVERIFIED"},
                  **cfg["boundary"],
                  "next_action": "停止并等待只读审计；不得自动启动独立确认、训练或真实SLM"}
        write_json(output / "summary.json", result)
        write_json(output / "SUCCESS.json", {"summary_sha256": _file_sha256(output / "summary.json")})
        return result
    except Exception:
        write_json(output / "failure.json", {"traceback": traceback.format_exc(),
                                             "automatic_retry": False})
        raise
    finally:
        progress.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/s4_r5_margin_development_d2_v1.yaml")
    parser.add_argument("--quick", action="store_true", help="16帧诊断冒烟，不得作算法排名")
    parser.add_argument("--preflight-only", action="store_true", help="只读检查，不生成输出")
    args = parser.parse_args()
    print(json.dumps(run(args.config, quick=args.quick,
                         preflight_only=args.preflight_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
