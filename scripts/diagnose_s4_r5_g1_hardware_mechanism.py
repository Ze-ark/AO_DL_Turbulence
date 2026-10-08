"""G1 后续新天气开发诊断：同槽名义硬件与新误差档位配对，不作独立确认。"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_s4_r5_g1_cross_condition as g1
from scripts.diagnose_s4_r5_margin_d2 import CorrectionClampTelemetry
from src.rl.r4_dynamics_experiment import Progress, safe_git_record
from src.rl.r4_interface_smoke import write_json
from src.rl.r5_independent_confirmation_r2 import source_bundle_sha256
from src.rl.r5_margin_development import _rollout, effective_stream_seeds
from src.rl.r5_policy_training import ResidualGRUPolicy
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _profiles, _project_path
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.hardware_effects import HardwareProfile
from src.simulation.robust_control import RobustnessCondition


CONFIG = "configs/experiments/s4_r5_g1_hardware_mechanism_v1.yaml"
CONDITIONS = ("nominal_clone", "hardware_shift")
FAMILIES = g1.FAMILIES
PROFILES = g1.HARDWARE_PROFILES
SCALE = g1.SCALE
FORMAL_BASE = 6_600_000
QUICK_BASE = 6_610_000
FROZEN_SOURCE = g1.FROZEN_SOURCE
G1_CONFIG_SHA256 = "7158f2f7a62c42da9ef763a310208f1a300cda219a1ba56177e65e9e6ff26cde"
G1_ENTRY_SHA256 = "001c456b6e9e705b0e964fd68c1fef08c3170d105d57b1e6b89eeaf28fa91ad7"
G1_OUTPUT_HASHES = {
    "wind": {
        "summary.json": "fe2c7b7e0e3cc4b0e9ba0b2fc2b98ca9bd2f563446e1582fa56b6ab1799da7c5",
        "records.jsonl": "0f44120f854379bf5ffb10ce709af8142709cef035140c319b4e1458c845b136",
        "progress.jsonl": "19e0cfc0ff2bb4ccb6df41af051d7249dd656952085ca9db10e4b390c881f662",
        "stream_manifest.json": "9a4acafcec2fbcc1394693ce152bba60e7a88985a26b91c9ab507d03d1164504",
        "SUCCESS.json": "f146eb97bbd0d755ac6f94fb7dec85ff664da28374f43802a5872254b6f96c6c",
    },
    "hardware": {
        "summary.json": "708a08921f716d9669a9c6e54f45f809f967319a02ee11478d997b639f12a27f",
        "records.jsonl": "2bc53d1703136091d7291b8deb8e2a29042dfd714e0ce256c72ec893876eb876",
        "progress.jsonl": "6a2ca134d311186c457315e6e463c4376fde3c2e89154832c8c1c2d424965576",
        "stream_manifest.json": "279bf791408d9c193208d1a391b8d071c6889f6c31c5f8ba4c6287c1ea7fe015",
        "SUCCESS.json": "d3e231396159d93eaef3cc43d709116e7a618e3e496714271f45b9de04a3dd4c",
    },
}
METRICS = (
    "power", "strehl", "phase_rmse", "violation", "saturation", "slew_limited",
    "correction_abs", "requested_step_abs", "requested_modal_abs", "applied_modal_abs",
    "requested_applied_gap_abs", "normalized_correction_clipped_fraction",
    "policy_forward_seconds_per_step", "policy_forward_p95_seconds",
)


def _contract(cfg: dict) -> None:
    if (cfg.get("stage") != "S4-D2-R5-G1-D1"
            or cfg.get("purpose") != "development_only_paired_hardware_shift_mechanism"
            or cfg.get("runtime") != {"device": "cuda", "formal_owner": "user_ide", "automatic_retry": False}
            or cfg.get("g1_config") != g1.CONFIG
            or cfg.get("g1_config_sha256") != G1_CONFIG_SHA256
            or cfg.get("g1_entry_sha256") != G1_ENTRY_SHA256
            or cfg.get("g1_output_hashes") != G1_OUTPUT_HASHES
            or cfg.get("selected_scale") != SCALE
            or cfg.get("hardware_shift_profiles") != list(PROFILES)
            or cfg.get("data") != {"seed_base": FORMAL_BASE, "seed_stride": 10,
                                   "weather_count": 24, "episode_length": 200}
            or cfg.get("quick") != {"seed_base": QUICK_BASE, "weather_count": 1,
                                    "episode_length": 16}
            or cfg.get("statistics") != {"bootstrap_seed": 6_623_456,
                                         "bootstrap_repeats": 5_000}
            or cfg.get("output_directory") != "outputs/s4_r5_g1_hardware_mechanism_v1"
            or cfg.get("quick_directory") != "outputs/s4_r5_g1_hardware_mechanism_v1_quick"
            or cfg.get("boundary") != {"confirmation_access": False,
                                       "training_updates": 0, "real_slm_actions": False,
                                       "automatic_retry": False, "g1_results_read_only": True}):
        raise ValueError("G1-D1 新天气开发诊断合同被改变")


def _verify_lineage(cfg: dict) -> tuple[dict, dict]:
    if _file_sha256(_project_path(cfg["g1_config"])) != G1_CONFIG_SHA256:
        raise RuntimeError("G1 冻结配置已改变")
    if _file_sha256(_project_path("scripts/run_s4_r5_g1_cross_condition.py")) != G1_ENTRY_SHA256:
        raise RuntimeError("G1 冻结入口已改变")
    g1_cfg = _load_yaml(_project_path(cfg["g1_config"]))
    g1._contract(g1_cfg)
    r3_cfg = g1._verify_r3_source(g1_cfg)
    if source_bundle_sha256() != FROZEN_SOURCE:
        raise RuntimeError("R5 冻结仿真与控制源码已改变")
    for arm, expected in G1_OUTPUT_HASHES.items():
        root = _project_path(g1_cfg["outputs"][arm])
        if (root / "failure.json").exists():
            raise RuntimeError(f"G1 {arm} 臂存在失败标记")
        for filename, digest in expected.items():
            if _file_sha256(root / filename) != digest:
                raise RuntimeError(f"G1 {arm} 证据已改变: {filename}")
        success = json.loads((root / "SUCCESS.json").read_text(encoding="utf-8"))
        for key, filename in (("summary_sha256", "summary.json"),
                              ("records_sha256", "records.jsonl"),
                              ("progress_sha256", "progress.jsonl"),
                              ("stream_manifest_sha256", "stream_manifest.json")):
            if success.get(key) != expected[filename]:
                raise RuntimeError(f"G1 {arm} 成功标记与证据不符")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        if (summary.get("status") != "R5_G1_ARM_COMPLETE_REQUIRES_AUDIT"
                or summary.get("arm") != arm or summary.get("records") != 4608
                or summary.get("training_updates") != 0
                or summary.get("real_slm_actions") is not False
                or summary.get("confirmation_access") is not True
                or summary.get("config_sha256") != G1_CONFIG_SHA256
                or summary.get("entry_sha256") != G1_ENTRY_SHA256
                or summary.get("source_bundle_sha256") != FROZEN_SOURCE
                or summary.get("analysis", {}).get("preliminary_all_gates") is not (arm == "wind")):
            raise RuntimeError(f"G1 {arm} 冻结状态与已审计结论不符")
    return g1_cfg, r3_cfg


def stream_manifest(quick: bool) -> dict[str, list[int]]:
    base, count = (QUICK_BASE, 1) if quick else (FORMAL_BASE, 24)
    weather = [base + 10 * index for index in range(count)]
    turbulence = effective_stream_seeds(base, count, 10, 6, 3)
    sensor = [seed + 1_000 * slot + 50_000_000 for seed in weather for slot in range(6)]
    power = [seed + 1_000 * slot + 60_000_000 for seed in weather for slot in range(6)]
    if (len(turbulence) != count * 18 or len(set(sensor)) != count * 6
            or len(set(power)) != count * 6
            or min(turbulence) < base or max(turbulence) >= base + 10_000):
        raise RuntimeError("G1-D1 随机流冲突")
    return {"weather_bases": weather, "turbulence": turbulence,
            "sensor": sensor, "power": power}


def _profile_pairs(parent: dict) -> tuple[list[HardwareProfile], list[HardwareProfile]]:
    shifts = _profiles(parent, PROFILES)
    nominal = _profiles(parent, ["nominal"])[0]
    clones = [replace(nominal, identifier=f"nominal_for_{profile.identifier}",
                      label=f"名义对照：{profile.identifier}") for profile in shifts]
    physical_fields = [item.name for item in fields(HardwareProfile)
                       if item.name not in {"identifier", "label"}]
    if (len(set(profile.identifier for profile in clones + shifts)) != 12
            or any(any(getattr(clone, name) != getattr(nominal, name)
                       for name in physical_fields) for clone in clones)
            or [profile.identifier for profile in shifts] != list(PROFILES)):
        raise RuntimeError("名义六槽位克隆或新误差档位不一致")
    return clones, shifts


def preflight(path: str | Path = CONFIG, *, quick: bool = False
              ) -> tuple[dict, dict, dict, dict, Path, torch.device]:
    cfg = _load_yaml(_project_path(path))
    _contract(cfg)
    _, r3_cfg = _verify_lineage(cfg)
    parent = _load_yaml(_project_path(r3_cfg["parent"]))
    if ([family["id"] for family in parent["families"]] != list(FAMILIES)
            or parent["data"]["sensor_seed_offset"] != 50_000_000):
        raise RuntimeError("R5 湍流/观测分组来源已改变")
    clones, shifts = _profile_pairs(parent)
    selected = stream_manifest(quick)
    formal, smoke = stream_manifest(False), stream_manifest(True)
    history = set(g1.r3_stream_manifest(False)["turbulence"])
    history.update(g1.r3_stream_manifest(True)["turbulence"])
    for arm in ("wind", "hardware"):
        for old_quick in (False, True):
            history.update(g1.stream_manifest(arm, old_quick)["turbulence"])
    if (set(formal["turbulence"]) & set(smoke["turbulence"])
            or history & (set(formal["turbulence"]) | set(smoke["turbulence"]))):
        raise RuntimeError("新开发天气与 R3/G1 或快速冒烟重叠")
    spec = cfg["quick" if quick else "data"]
    base, _ = load_s1_config(_project_path(parent["environment_config"]))
    displacements = []
    for family in parent["families"]:
        condition = RobustnessCondition.from_mapping(dict(family, base_seed=spec["seed_base"]))
        simulation = condition.environment_config(replace(base, episode_length=spec["episode_length"]))
        displacement = (simulation.wind_speed_mps * (1 + simulation.wind_speed_modulation_fraction)
                        * simulation.dt_s * spec["episode_length"] / simulation.sample_pitch_m)
        if displacement >= simulation.turbulence_grid_size:
            raise RuntimeError("相位屏在完整回合内重复")
        displacements.append(displacement)
    output = _project_path(cfg["quick_directory"] if quick else cfg["output_directory"])
    if output.exists():
        raise FileExistsError(f"保留已有 G1-D1 开发输出，不覆盖/重跑: {output}")
    device = resolve_device("cuda")
    report = {
        "status": "READY_FOR_DIAGNOSTIC" if quick else "READY_FOR_USER_IDE",
        "quick": quick, "device": str(device), "weather_count": spec["weather_count"],
        "episode_length": spec["episode_length"], "families": 3, "slots": 6,
        "conditions": list(CONDITIONS), "controllers_per_condition": 4,
        "physical_transitions": 2 * 4 * spec["weather_count"] * 3 * 6 * spec["episode_length"],
        "unique_turbulence_streams_per_condition": len(selected["turbulence"]),
        "shared_streams_between_conditions": True,
        "maximum_displacement_pixels": displacements,
        "turbulence_grid_pixels": simulation.turbulence_grid_size,
        "shift_profile_ids": [profile.identifier for profile in shifts],
        "nominal_clone_ids": [profile.identifier for profile in clones],
        "nominal_physical_parameters": {name: getattr(clones[0], name)
                                        for name in ("slm_delay_frames", "slm_quantization_levels",
                                                     "slm_max_delta_rad", "observation_noise_std_rad",
                                                     "phase_scale", "settling_fraction", "shift_x_pixels",
                                                     "shift_y_pixels", "rotation_deg",
                                                     "power_noise_relative_std")},
        "selected_scale": SCALE, "config_sha256": _file_sha256(_project_path(path)),
        "entry_sha256": _file_sha256(Path(__file__)),
        "frozen_source_bundle_sha256": FROZEN_SOURCE,
        "g1_config_sha256": G1_CONFIG_SHA256,
        "g1_entry_sha256": G1_ENTRY_SHA256,
        **cfg["boundary"],
    }
    return cfg, r3_cfg, parent, report, output, device


def summarize(rows: list[dict], cfg: dict, *, quick: bool, device: torch.device) -> dict:
    spec = cfg["quick" if quick else "data"]
    weather = [spec["seed_base"] + 10 * index for index in range(spec["weather_count"])]
    controllers = ("integrator",) + tuple(f"policy_{index}_scale_{SCALE}" for index in range(3))
    by_key = {(row["condition"], row["controller"], row["family"],
               row["slot"], row["weather_seed"]): row for row in rows}
    expected = {(condition, controller, family, slot, seed)
                for condition in CONDITIONS for controller in controllers
                for family in FAMILIES for slot in range(6) for seed in weather}
    if len(rows) != len(expected) or set(by_key) != expected:
        raise RuntimeError("G1-D1 配对回合缺失、重复或条件错位")
    for row in rows:
        target = PROFILES[row["slot"]]
        expected_name = (f"nominal_for_{target}" if row["condition"] == "nominal_clone" else target)
        if (row["profile"] != expected_name or row["target_profile"] != target
                or row["scale"] != (0.0 if row["controller"] == "integrator" else SCALE)
                or row["turbulence_stream_seed"] != (row["weather_seed"]
                    + row["slot"] * 1_000 + FAMILIES.index(row["family"]))
                or any(not math.isfinite(row[name]) for name in METRICS)
                or not 0 <= row["normalized_correction_clipped_fraction"] <= 1):
            raise RuntimeError("G1-D1 档位、种子或指标错位")
    values = {name: torch.tensor(
        [[[[[by_key[(condition, controller, family, slot, seed)][name]
              for slot in range(6)] for seed in weather] for family in FAMILIES]
          for controller in controllers] for condition in CONDITIONS],
        device=device, dtype=torch.float64) for name in METRICS}
    power = values["power"]
    gains = power[:, 1:].mean(dim=1) - power[:, 0]
    difference_in_differences = gains[1] - gains[0]  # family × weather × slot
    repeats = cfg["statistics"]["bootstrap_repeats"]
    generator = torch.Generator(device=device).manual_seed(cfg["statistics"]["bootstrap_seed"])
    draw = torch.randint(len(weather), (repeats, 3, len(weather)), device=device,
                         generator=generator)
    sample = torch.gather(
        difference_in_differences[None].expand(repeats, -1, -1, -1), 2,
        draw[..., None].expand(-1, -1, -1, 6),
    ).mean(dim=(1, 2))
    quantiles = torch.tensor([.025, .975], device=device, dtype=torch.float64)
    profile_ci = torch.quantile(sample, quantiles, dim=0)
    overall_ci = torch.quantile(sample.mean(dim=-1), quantiles)
    profiles = {}
    for slot, name in enumerate(PROFILES):
        conditions = {}
        for condition_index, condition in enumerate(CONDITIONS):
            base = float(power[condition_index, 0, :, :, slot].mean())
            policy = float(power[condition_index, 1:, :, :, slot].mean())
            if base <= 0:
                raise RuntimeError("G1-D1 积分器桶内功率非正")
            conditions[condition] = {"integrator_power": base, "policy_power": policy,
                                     "policy_minus_integrator_power": policy - base,
                                     "relative_gain": (policy - base) / base}
        other = {}
        for metric in METRICS:
            if metric == "power":
                continue
            tensor = values[metric]
            nominal_integrator = float(tensor[0, 0, :, :, slot].mean())
            nominal_policy = float(tensor[0, 1:, :, :, slot].mean())
            shift_integrator = float(tensor[1, 0, :, :, slot].mean())
            shift_policy = float(tensor[1, 1:, :, :, slot].mean())
            other[metric] = {
                "nominal_integrator": nominal_integrator, "nominal_policy": nominal_policy,
                "shift_integrator": shift_integrator, "shift_policy": shift_policy,
                "integrator_shift_minus_nominal": shift_integrator - nominal_integrator,
                "policy_shift_minus_nominal": shift_policy - nominal_policy,
            }
        profiles[name] = {
            "slot": slot, "nominal_clone_id": f"nominal_for_{name}",
            "conditions": conditions,
            "policy_increment_shift_minus_nominal": float(difference_in_differences[:, :, slot].mean()),
            "exploratory_paired_ci95": [float(profile_ci[0, slot]), float(profile_ci[1, slot])],
            "policy_power_shift_minus_nominal": (conditions["hardware_shift"]["policy_power"]
                                                  - conditions["nominal_clone"]["policy_power"]),
            "integrator_power_shift_minus_nominal": (conditions["hardware_shift"]["integrator_power"]
                                                      - conditions["nominal_clone"]["integrator_power"]),
            "family_policy_increment_shift_minus_nominal": {
                family: float(difference_in_differences[family_index, :, slot].mean())
                for family_index, family in enumerate(FAMILIES)},
            "member_policy_increment_shift_minus_nominal": [
                float(((power[1, member + 1, :, :, slot] - power[1, 0, :, :, slot])
                       - (power[0, member + 1, :, :, slot] - power[0, 0, :, :, slot])).mean())
                for member in range(3)],
            "other_metrics": other,
        }
    return {
        "status": "EXPLORATORY_DEVELOPMENT_NO_CONFIRMATION_GATE",
        "paired_unit": "same_weather_family_slot_and_member; bootstrap_weather_within_family",
        "paired_random_streams_between_conditions": True,
        "profile_results": profiles,
        "overall_policy_increment_shift_minus_nominal": float(difference_in_differences.mean()),
        "overall_exploratory_paired_ci95": [float(item) for item in overall_ci],
        "multiple_comparisons_adjusted": False,
        "latency_scope": "CUDA policy forward only, not real SLM end-to-end latency",
        "independent_confirmation": False,
    }


def run(path: str | Path = CONFIG, *, quick: bool = False,
        preflight_only: bool = False) -> dict:
    cfg, r3_cfg, parent, report, output, device = preflight(path, quick=quick)
    if preflight_only:
        return report
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "preflight.json", report)
    write_json(output / "config.json", cfg)
    streams = stream_manifest(quick)
    write_json(output / "stream_manifest.json", streams)
    write_json(output / "runtime.json", {
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "git": safe_git_record(),
        "frozen_source_bundle_sha256": FROZEN_SOURCE,
    })
    progress = Progress(output, device)
    started = time.perf_counter()
    try:
        spec = cfg["quick" if quick else "data"]
        base, _ = load_s1_config(_project_path(parent["environment_config"]))
        base = replace(base, num_modes=21, batch_size=1, episode_length=spec["episode_length"])
        basis, _, _ = build_action_basis(base, ActionRepresentation("r5_zernike21", "zernike", 21), device)
        policies = []
        for member in range(3):
            checkpoint = torch.load(_project_path(r3_cfg["training_output"]) / "checkpoints"
                                    / f"policy_{member}_02000.pt", map_location=device, weights_only=True)
            if checkpoint["init"] != member or checkpoint["update"] != 2000:
                raise RuntimeError("冻结 R5 权重身份错误")
            policy = ResidualGRUPolicy(parent["policy"]["hidden_size"],
                                       parent["policy"]["output_size"]).to(device)
            policy.load_state_dict(checkpoint["state_dict"])
            policy.eval()
            policies.append(policy)
        branches = [("integrator", 0.0, None)] + [
            (f"policy_{member}_scale_{SCALE}", SCALE, policy)
            for member, policy in enumerate(policies)]
        clone_profiles, shifted_profiles = _profile_pairs(parent)
        progress.phase("G1-D1 快速冒烟" if quick else "G1-D1 新天气开发诊断",
                       2 * len(branches) * spec["weather_count"] * spec["episode_length"])
        rows: list[dict] = []
        with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
            for condition, profiles in zip(CONDITIONS, (clone_profiles, shifted_profiles), strict=True):
                for label, scale, policy in branches:
                    for seed in streams["weather_bases"]:
                        meter = None if policy is None else CorrectionClampTelemetry(policy, scale)
                        episode = _rollout(seed, label, scale, meter, spec["episode_length"],
                                           basis, base, parent["families"], profiles,
                                           parent["data"]["sensor_seed_offset"], progress)
                        rates = ([0.0] * len(episode) if meter is None
                                 else meter.rates(spec["episode_length"], len(episode)))
                        for row, rate in zip(episode, rates, strict=True):
                            slot = next(index for index, profile in enumerate(profiles)
                                        if profile.identifier == row["profile"])
                            row.update({"condition": condition, "slot": slot,
                                        "target_profile": PROFILES[slot],
                                        "normalized_correction_clipped_fraction": rate})
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                            rows.append(row)
                        handle.flush()
        expected = 2 * len(branches) * spec["weather_count"] * 3 * 6
        if len(rows) != expected:
            raise RuntimeError("G1-D1 完整分组回合不足")
        analysis = {} if quick else summarize(rows, cfg, quick=False, device=device)
        result = {
            "status": "QUICK_SMOKE_NO_CONCLUSION" if quick
                      else "DEVELOPMENT_DIAGNOSTIC_REQUIRES_AUDIT",
            "records": len(rows), "weather_count": spec["weather_count"],
            "episode_length": spec["episode_length"],
            "physical_transitions": report["physical_transitions"],
            "conditions": list(CONDITIONS), "controllers": [label for label, _, _ in branches],
            "elapsed_seconds": time.perf_counter() - started,
            "analysis": analysis,
            "config_sha256": report["config_sha256"],
            "entry_sha256": report["entry_sha256"],
            "frozen_source_bundle_sha256": FROZEN_SOURCE,
            "g1_config_sha256": G1_CONFIG_SHA256,
            "g1_entry_sha256": G1_ENTRY_SHA256,
            "material_passport": {"origin_skill": "academic-research-suite / experiment-agent",
                                  "origin_mode": "run", "verification_status": "UNVERIFIED"},
            **cfg["boundary"],
            "next_action": "停止，等待只读审计；开发结果不能改判 G1 或当独立确认",
        }
        write_json(output / "summary.json", result)
        write_json(output / "SUCCESS.json", {
            "summary_sha256": _file_sha256(output / "summary.json"),
            "records_sha256": _file_sha256(output / "records.jsonl"),
            "progress_sha256": _file_sha256(output / "progress.jsonl"),
            "stream_manifest_sha256": _file_sha256(output / "stream_manifest.json"),
        })
        return result
    except Exception:
        write_json(output / "failure.json", {"traceback": traceback.format_exc(),
                                              "automatic_retry": False})
        raise
    finally:
        progress.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--quick", action="store_true", help="16 帧 CUDA 冒烟，不产生性能结论")
    parser.add_argument("--preflight-only", action="store_true", help="只读预检，不生成输出")
    args = parser.parse_args()
    print(json.dumps(run(args.config, quick=args.quick,
                         preflight_only=args.preflight_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
