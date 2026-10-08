"""R5-G2-D1：冻结策略的硬裁剪与平滑有界动作映射配对开发诊断。"""
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

from scripts import diagnose_s4_r5_g1_d2_factorial as d2
from scripts import diagnose_s4_r5_g1_d3_nominal_bridge as d3
from scripts import diagnose_s4_r5_g1_hardware_mechanism as d1
from scripts import run_s4_r5_g1_cross_condition as g1
from src.rl.r4_dynamics_experiment import Progress, safe_git_record
from src.rl.r4_interface_smoke import write_json
from src.rl.r5_independent_confirmation_r2 import source_bundle_sha256
from src.rl.r5_margin_development import _rollout, effective_stream_seeds
from src.rl.r5_policy_training import ResidualGRUPolicy
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _project_path
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.robust_control import RobustnessCondition


CONFIG = "configs/experiments/s4_r5_g2_d1_action_mapping_v1.yaml"
CONDITIONS = ("nominal_clone", "hardware_shift")
MAPPINGS = ("hard", "smooth")
FAMILIES = d1.FAMILIES
PROFILES = d1.PROFILES
SCALE = 1.75
EPSILON = 1e-6
FORMAL_BASE = 6_800_000
QUICK_BASE = 6_810_000
FROZEN_SOURCE = d1.FROZEN_SOURCE
D3_CONFIG_SHA256 = "ff99a9e64398a6b982309158fc786fae7fc69fc3ab165f0863bafaac49766b95"
D3_ENTRY_SHA256 = "63ce7be078f0dfd7fec094d1bd2fbd5190195109d69f476d6b7ec73bc97b26e9"
D3_OUTPUT_HASHES = {
    "summary.json": "9bcf3e23f318aeee3a379f48bd72b37bc6a2eb1e6b302bb97b94dccd624025fb",
    "records.jsonl": "d520d1ce4e5cc4a214644961b36d34065d8592688f6c3460aa0d025ddd42fcc8",
    "progress.jsonl": "65ae254d3e45731edf1d81ff6cdf65834d10b7b9b4b377e9bd4ad9d8dee4cdc6",
    "stream_manifest.json": "09fb204e7fdcab9c24426fe96bc0324b21dd378c7803550bf1edb8919e69a80c",
    "SUCCESS.json": "0d212907fd9efcfbecf698c088a4faa7a674418bcbeb77378c4fb04856a4dc7f",
}
METRICS = d1.METRICS + ("raw_hard_clipped_fraction",)


def _map_action(raw: torch.Tensor, mapping: str, scale: float = SCALE,
                epsilon: float = EPSILON) -> torch.Tensor:
    """Return the pre-projection correction; the common R4 interface remains the safety gate."""
    if (mapping not in MAPPINGS or scale != SCALE or epsilon != EPSILON
            or raw.ndim != 2 or raw.shape[-1] != 11 or not bool(torch.isfinite(raw).all())
            or bool((raw.abs() > 1 + 1e-6).any())):
        raise ValueError("R5-G2-D1 动作映射输入或冻结参数无效")
    if mapping == "hard":
        return raw * scale
    return torch.tanh(scale * torch.atanh(raw.clamp(-1 + epsilon, 1 - epsilon)))


class ActionMappingTelemetry(nn.Module):
    """Adapter for the frozen _rollout: it multiplies this output by SCALE once."""

    def __init__(self, policy: ResidualGRUPolicy, mapping: str):
        super().__init__()
        if mapping not in MAPPINGS:
            raise ValueError("未知动作映射")
        self.policy = policy
        self.mapping = mapping
        self._raw_clipped: torch.Tensor | None = None
        self._mapped_clipped: torch.Tensor | None = None
        self.steps = 0

    def forward(self, history: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        raw = self.policy(history, valid)
        mapped = _map_action(raw, self.mapping)
        raw_clipped = (raw * SCALE).abs().gt(1).to(torch.float64).mean(-1)
        mapped_clipped = mapped.abs().gt(1).to(torch.float64).mean(-1)
        self._raw_clipped = raw_clipped if self._raw_clipped is None else self._raw_clipped + raw_clipped
        self._mapped_clipped = (mapped_clipped if self._mapped_clipped is None
                                else self._mapped_clipped + mapped_clipped)
        self.steps += 1
        # Preserve the historical hard branch bit-for-bit: _rollout multiplies by SCALE.
        return raw if self.mapping == "hard" else mapped / SCALE

    def rates(self, expected_steps: int, batch: int) -> tuple[list[float], list[float]]:
        if (self.steps != expected_steps or self._raw_clipped is None
                or self._mapped_clipped is None or self._raw_clipped.shape != (batch,)
                or self._mapped_clipped.shape != (batch,)):
            raise RuntimeError("动作裁剪遥测回合数或形状错误")
        return ((self._raw_clipped / expected_steps).detach().cpu().tolist(),
                (self._mapped_clipped / expected_steps).detach().cpu().tolist())


def _contract(cfg: dict) -> None:
    if (cfg.get("stage") != "S4-D2-R5-G2-D1"
            or cfg.get("purpose") != "development_only_frozen_policy_action_mapping"
            or cfg.get("runtime") != {"device": "cuda", "formal_owner": "user_ide", "automatic_retry": False}
            or cfg.get("g1_d3_config") != d3.CONFIG
            or cfg.get("g1_d3_config_sha256") != D3_CONFIG_SHA256
            or cfg.get("g1_d3_entry_sha256") != D3_ENTRY_SHA256
            or cfg.get("g1_d3_output") != "outputs/s4_r5_g1_d3_nominal_bridge_v1"
            or cfg.get("g1_d3_output_hashes") != D3_OUTPUT_HASHES
            or cfg.get("selected_scale") != SCALE
            or cfg.get("smooth_epsilon") != EPSILON
            or cfg.get("mapping_ids") != list(MAPPINGS)
            or cfg.get("hardware_shift_profiles") != list(PROFILES)
            or cfg.get("data") != {"seed_base": FORMAL_BASE, "seed_stride": 10,
                                   "weather_count": 24, "episode_length": 200}
            or cfg.get("quick") != {"seed_base": QUICK_BASE, "weather_count": 1,
                                    "episode_length": 16}
            or cfg.get("statistics") != {"bootstrap_seed": 6_823_456, "bootstrap_repeats": 5_000}
            or cfg.get("output_directory") != "outputs/s4_r5_g2_d1_action_mapping_v1"
            or cfg.get("quick_directory") != "outputs/s4_r5_g2_d1_action_mapping_v1_quick_r2"
            or cfg.get("boundary") != {"confirmation_access": False, "training_updates": 0,
                                       "real_slm_actions": False, "automatic_retry": False,
                                       "historical_results_read_only": True}):
        raise ValueError("R5-G2-D1 冻结动作映射开发合同被改变")


def _verify_lineage(cfg: dict) -> tuple[dict, dict]:
    if _file_sha256(_project_path(cfg["g1_d3_config"])) != D3_CONFIG_SHA256:
        raise RuntimeError("G1-D3 冻结配置已改变")
    if _file_sha256(_project_path("scripts/diagnose_s4_r5_g1_d3_nominal_bridge.py")) != D3_ENTRY_SHA256:
        raise RuntimeError("G1-D3 冻结入口已改变")
    d3_cfg = _load_yaml(_project_path(cfg["g1_d3_config"]))
    d3._contract(d3_cfg)
    _, r3_cfg, parent = d3._verify_lineage(d3_cfg)  # 验证 R3 权重及 G1/D1/D2 冻结证据。
    if source_bundle_sha256() != FROZEN_SOURCE:
        raise RuntimeError("R5 冻结仿真和控制源码已改变")
    root = _project_path(cfg["g1_d3_output"])
    if (root / "failure.json").exists():
        raise RuntimeError("G1-D3 历史输出有失败标记")
    for name, digest in D3_OUTPUT_HASHES.items():
        if _file_sha256(root / name) != digest:
            raise RuntimeError(f"G1-D3 历史证据已改变: {name}")
    success = json.loads((root / "SUCCESS.json").read_text(encoding="utf-8"))
    for key, name in (("summary_sha256", "summary.json"), ("records_sha256", "records.jsonl"),
                      ("progress_sha256", "progress.jsonl"),
                      ("stream_manifest_sha256", "stream_manifest.json")):
        if success.get(key) != D3_OUTPUT_HASHES[name]:
            raise RuntimeError("G1-D3 成功标记与冻结证据不符")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("status") != "DEVELOPMENT_BRIDGE_REQUIRES_AUDIT"
            or summary.get("records") != 3456 or summary.get("failed_group_episodes") != 0
            or summary.get("training_updates") != 0 or summary.get("real_slm_actions") is not False
            or summary.get("confirmation_access") is not False
            or summary.get("config_sha256") != D3_CONFIG_SHA256
            or summary.get("entry_sha256") != D3_ENTRY_SHA256
            or summary.get("frozen_source_bundle_sha256") != FROZEN_SOURCE):
        raise RuntimeError("G1-D3 已审计状态与冻结证据不符")
    return r3_cfg, parent


def stream_manifest(quick: bool) -> dict[str, list[int]]:
    base, count = (QUICK_BASE, 1) if quick else (FORMAL_BASE, 24)
    weather = [base + 10 * index for index in range(count)]
    turbulence = effective_stream_seeds(base, count, 10, 6, 3)
    sensor = [seed + 1_000 * slot + 50_000_000 for seed in weather for slot in range(6)]
    power = [seed + 1_000 * slot + 60_000_000 for seed in weather for slot in range(6)]
    if (len(set(turbulence)) != 18 * count or len(set(sensor)) != 6 * count
            or len(set(power)) != 6 * count):
        raise RuntimeError("R5-G2-D1 天气随机流冲突")
    return {"weather_bases": weather, "turbulence": turbulence,
            "sensor": sensor, "power": power}


def preflight(path: str | Path = CONFIG, *, quick: bool = False
              ) -> tuple[dict, dict, dict, dict, Path, torch.device]:
    cfg = _load_yaml(_project_path(path))
    _contract(cfg)
    r3_cfg, parent = _verify_lineage(cfg)
    if (parent["profile_ids"] != list(d2.ORIGINAL_PROFILES)
            or [item["id"] for item in parent["families"]] != list(FAMILIES)
            or parent["data"]["sensor_seed_offset"] != 50_000_000):
        raise RuntimeError("R5 冻结湍流、观测流或硬件合同已改变")
    nominal, shift = d1._profile_pairs(parent)
    if ([p.identifier for p in shift] != list(PROFILES)
            or [p.identifier for p in nominal] != [f"nominal_for_{name}" for name in PROFILES]):
        raise RuntimeError("六槽标称与新硬件档位无法严格配对")
    formal, smoke = stream_manifest(False), stream_manifest(True)
    historical = [d1.stream_manifest(False), d1.stream_manifest(True),
                  d2.stream_manifest(False), d2.stream_manifest(True), d3._manifest(True),
                  g1.r3_stream_manifest(False), g1.r3_stream_manifest(True)]
    for arm in ("wind", "hardware"):
        historical.extend((g1.stream_manifest(arm, False), g1.stream_manifest(arm, True)))
    for name in ("turbulence", "sensor", "power"):
        earlier = set().union(*(set(item[name]) for item in historical))
        if (set(formal[name]) & set(smoke[name])
                or earlier & (set(formal[name]) | set(smoke[name]))):
            raise RuntimeError(f"R5-G2-D1 {name} 随机流与已看开发/确认天气重叠")
    spec = cfg["quick" if quick else "data"]
    base, _ = load_s1_config(_project_path(parent["environment_config"]))
    displacements = []
    for family in parent["families"]:
        condition = RobustnessCondition.from_mapping(dict(family, base_seed=spec["seed_base"]))
        simulation = condition.environment_config(replace(base, episode_length=spec["episode_length"]))
        displacement = (simulation.wind_speed_mps * (1 + simulation.wind_speed_modulation_fraction)
                        * simulation.dt_s * spec["episode_length"] / simulation.sample_pitch_m)
        if displacement >= simulation.turbulence_grid_size:
            raise RuntimeError("R5-G2-D1 相位屏在回合内重复")
        displacements.append(displacement)
    output = _project_path(cfg["quick_directory" if quick else "output_directory"])
    if output.exists():
        raise FileExistsError(f"保留已有 R5-G2-D1 输出，不覆盖/重跑: {output}")
    device = resolve_device("cuda")
    report = {
        "status": "READY_FOR_DIAGNOSTIC" if quick else "READY_FOR_USER_IDE",
        "quick": quick, "device": str(device), "weather_count": spec["weather_count"],
        "episode_length": spec["episode_length"], "families": 3, "profile_slots_per_condition": 6,
        "hardware_conditions": list(CONDITIONS), "mapping_ids": list(MAPPINGS),
        "controllers_per_condition": 7,
        "physical_transitions": 2 * 7 * spec["weather_count"] * 3 * 6 * spec["episode_length"],
        "unique_turbulence_streams_per_condition": len(stream_manifest(quick)["turbulence"]),
        "shared_streams_between_conditions_and_controllers": True,
        "maximum_displacement_pixels": displacements,
        "turbulence_grid_pixels": simulation.turbulence_grid_size,
        "selected_scale": SCALE, "smooth_epsilon": EPSILON,
        "integrator_gain": .15, "integrator_leak": .10,
        "config_sha256": _file_sha256(_project_path(path)),
        "entry_sha256": _file_sha256(Path(__file__)),
        "frozen_source_bundle_sha256": FROZEN_SOURCE,
        "g1_d3_config_sha256": D3_CONFIG_SHA256,
        "g1_d3_entry_sha256": D3_ENTRY_SHA256,
        **cfg["boundary"],
    }
    return cfg, r3_cfg, parent, report, output, device


def _paired_ci(weather_values: torch.Tensor, draws: torch.Tensor) -> list[float]:
    sample_means = weather_values[draws].mean(dim=-1)
    return [float(x) for x in torch.quantile(sample_means, torch.tensor(
        [.025, .975], device=weather_values.device, dtype=weather_values.dtype))]


def summarize(rows: list[dict], cfg: dict, *, quick: bool, device: torch.device) -> dict:
    spec = cfg["quick" if quick else "data"]
    weather = [spec["seed_base"] + 10 * index for index in range(spec["weather_count"])]
    controllers = ("integrator",) + tuple(f"policy_{member}_{mapping}"
                                       for mapping in MAPPINGS for member in range(3))
    by_key = {(r["hardware_condition"], r["controller"], r["family"], r["slot"],
               r["weather_seed"]): r for r in rows}
    expected = {(condition, controller, family, slot, seed)
                for condition in CONDITIONS for controller in controllers
                for family in FAMILIES for slot in range(6) for seed in weather}
    if len(rows) != len(expected) or set(by_key) != expected:
        raise RuntimeError("R5-G2-D1 完整配对回合缺失、重复或条件错位")
    for row in rows:
        expected_profile = (f"nominal_for_{PROFILES[row['slot']]}"
                            if row["hardware_condition"] == "nominal_clone" else PROFILES[row["slot"]])
        if (row["profile"] != expected_profile
                or row["scale"] != (0.0 if row["controller"] == "integrator" else SCALE)
                or row["turbulence_stream_seed"] != row["weather_seed"] + 1_000 * row["slot"] + FAMILIES.index(row["family"])
                or row["mapping"] != ("integrator" if row["controller"] == "integrator"
                                      else row["controller"].split("_")[-1])
                or any(not math.isfinite(row[name]) for name in METRICS)
                or any(not 0 <= row[name] <= 1 for name in
                       ("normalized_correction_clipped_fraction", "raw_hard_clipped_fraction"))):
            raise RuntimeError("R5-G2-D1 档位、映射、随机流或指标错位")
    values = {name: torch.tensor(
        [[[[[by_key[(condition, controller, family, slot, seed)][name]
              for slot in range(6)] for seed in weather] for family in FAMILIES]
          for controller in controllers] for condition in CONDITIONS],
        device=device, dtype=torch.float64) for name in METRICS}
    # condition × controller × family × weather × slot
    power = values["power"]
    if bool((power[:, 0] <= 0).any()):
        raise RuntimeError("R5-G2-D1 积分器桶内功率非正")
    generator = torch.Generator(device=device).manual_seed(cfg["statistics"]["bootstrap_seed"])
    draws = torch.randint(len(weather), (cfg["statistics"]["bootstrap_repeats"], len(weather)),
                          generator=generator, device=device)
    cells = {}
    mapping_effects = []
    for ci, condition in enumerate(CONDITIONS):
        baseline = power[ci, 0]
        hard = power[ci, 1:4]
        smooth = power[ci, 4:7]
        effect = smooth - hard  # member × family × weather × slot; paired physical streams.
        mapping_effects.append(effect)
        baseline_mean = float(baseline.mean())
        paired_ci = _paired_ci(effect.mean(dim=(0, 1, 3)), draws)
        member_effects = [float(effect[member].mean()) for member in range(3)]
        family_effects = {family: float(effect[:, index].mean())
                          for index, family in enumerate(FAMILIES)}
        metric_deltas = {name: float((value[ci, 4:7] - value[ci, 1:4]).mean())
                         for name, value in values.items() if name != "power"}
        smooth_relative_gain = float((smooth - baseline).mean()) / baseline_mean
        criteria = {
            "smooth_minus_hard_mean_positive": float(effect.mean()) > 0,
            "paired_weather_ci95_lower_positive": paired_ci[0] > 0,
            "strehl_non_decrease": metric_deltas["strehl"] >= 0,
            "phase_rmse_non_increase": metric_deltas["phase_rmse"] <= 0,
            "violation_increase_at_most_0_001": metric_deltas["violation"] <= .001,
            "saturation_increase_at_most_0_001": metric_deltas["saturation"] <= .001,
            "slew_limited_increase_at_most_0_001": metric_deltas["slew_limited"] <= .001,
            "each_member_mapping_effect_positive": all(value > 0 for value in member_effects),
            "each_family_mapping_effect_positive": all(value > 0 for value in family_effects.values()),
            "smooth_relative_gain_at_least_1_05_percent": smooth_relative_gain >= .0105,
        }
        criteria["all"] = all(criteria.values())
        slot_powers = []
        for slot in range(6):
            baseline_slot = float(baseline[..., slot].mean())
            hard_slot = float(hard[..., slot].mean())
            smooth_slot = float(smooth[..., slot].mean())
            slot_powers.append({
                "slot": slot,
                "nominal_profile_or_shift_profile": (f"nominal_for_{PROFILES[slot]}"
                                                     if condition == "nominal_clone" else PROFILES[slot]),
                "integrator_power": baseline_slot,
                "hard_policy_power": hard_slot,
                "smooth_policy_power": smooth_slot,
                "hard_policy_minus_integrator_power": hard_slot - baseline_slot,
                "smooth_policy_minus_integrator_power": smooth_slot - baseline_slot,
                "hard_relative_gain": (hard_slot - baseline_slot) / baseline_slot,
                "smooth_relative_gain": (smooth_slot - baseline_slot) / baseline_slot,
            })
        cells[condition] = {
            "integrator_power": baseline_mean,
            "hard_policy_power": float(hard.mean()),
            "smooth_policy_power": float(smooth.mean()),
            "hard_policy_minus_integrator_power": float((hard - baseline).mean()),
            "smooth_policy_minus_integrator_power": float((smooth - baseline).mean()),
            "hard_relative_gain": float((hard - baseline).mean()) / baseline_mean,
            "smooth_relative_gain": smooth_relative_gain,
            "smooth_minus_hard_absolute_power": float(effect.mean()),
            "smooth_minus_hard_exploratory_paired_ci95": paired_ci,
            "member_mapping_effects": member_effects,
            "family_mapping_effects": family_effects,
            "slot_mapping_effects": [float(effect[..., slot].mean()) for slot in range(6)],
            "slot_absolute_powers_and_gains": slot_powers,
            "mapping_metric_deltas": metric_deltas,
            "development_continue_criteria": criteria,
            "hard_raw_hard_clipped_fraction": float(values["raw_hard_clipped_fraction"][ci, 1:4].mean()),
            "smooth_raw_hard_clipped_fraction": float(values["raw_hard_clipped_fraction"][ci, 4:7].mean()),
            "hard_actual_clipped_fraction": float(values["normalized_correction_clipped_fraction"][ci, 1:4].mean()),
            "smooth_actual_clipped_fraction": float(values["normalized_correction_clipped_fraction"][ci, 4:7].mean()),
        }
    overall = torch.stack(mapping_effects).mean(dim=0)
    return {
        "status": "EXPLORATORY_DEVELOPMENT_NO_CONFIRMATION_GATE",
        "cells": cells,
        "overall_smooth_minus_hard_absolute_power": float(overall.mean()),
        "overall_exploratory_paired_ci95": _paired_ci(overall.mean(dim=(0, 1, 3)), draws),
        "development_continue_criteria": {
            "nominal_clone": cells["nominal_clone"]["development_continue_criteria"],
            "hardware_shift": cells["hardware_shift"]["development_continue_criteria"],
            "all": all(cells[condition]["development_continue_criteria"]["all"]
                       for condition in CONDITIONS),
            "meaning": "仅允许另行设计独立确认；本开发诊断本身不产生确认通过结论。",
        },
        "paired_unit": "same_complete_weather_across_both_conditions_all_families_slots_members",
        "same_random_streams_between_nominal_and_shift": True,
        "profile_slot_warning": "标称克隆与新六档同槽配对；槽位是随机流索引，不是硬件严重程度等级。",
        "independent_confirmation": False,
        "multiple_comparisons_adjusted": False,
        "latency_scope": "_rollout CUDA-synchronized forward includes policy, mapping checks and telemetry; excludes R4 projection, environment step, actual SLM and optical I/O",
    }


def run(path: str | Path = CONFIG, *, quick: bool = False,
        preflight_only: bool = False) -> dict:
    cfg, r3_cfg, parent, report, output, device = preflight(path, quick=quick)
    if preflight_only:
        return report
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    created = False
    progress: Progress | None = None
    started = time.perf_counter()
    try:
        output.mkdir(parents=True, exist_ok=False)
        created = True
        write_json(output / "preflight.json", report)
        write_json(output / "config.json", cfg)
        write_json(output / "stream_manifest.json", stream_manifest(quick))
        write_json(output / "runtime.json", {
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "git": safe_git_record(),
            "frozen_source_bundle_sha256": FROZEN_SOURCE,
        })
        progress = Progress(output, device)
        spec = cfg["quick" if quick else "data"]
        base, _ = load_s1_config(_project_path(parent["environment_config"]))
        base = replace(base, num_modes=21, batch_size=1, episode_length=spec["episode_length"])
        basis, _, _ = build_action_basis(base, ActionRepresentation("r5_zernike21", "zernike", 21), device)
        policies = []
        for member in range(3):
            checkpoint = torch.load(_project_path(r3_cfg["training_output"]) / "checkpoints"
                                    / f"policy_{member}_02000.pt", map_location=device, weights_only=True)
            if checkpoint["init"] != member or checkpoint["update"] != 2000:
                raise RuntimeError("R5 冻结权重身份错误")
            policy = ResidualGRUPolicy(parent["policy"]["hidden_size"],
                                       parent["policy"]["output_size"]).to(device)
            policy.load_state_dict(checkpoint["state_dict"])
            policy.eval()
            policies.append(policy)
        branches = [("integrator", "integrator", None)] + [
            (f"policy_{member}_{mapping}", mapping, policies[member])
            for mapping in MAPPINGS for member in range(3)]
        progress.phase("R5-G2-D1 快速冒烟" if quick else "R5-G2-D1 动作映射配对开发诊断",
                       2 * len(branches) * spec["weather_count"] * spec["episode_length"])
        nominal, shift = d1._profile_pairs(parent)
        rows: list[dict] = []
        with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
            for condition, profiles in zip(CONDITIONS, (nominal, shift), strict=True):
                # Separate six-slot batches preserve profile_index 0..5 in every RNG stream.
                for label, mapping, policy in branches:
                    for seed in stream_manifest(quick)["weather_bases"]:
                        meter = None if policy is None else ActionMappingTelemetry(policy, mapping)
                        episode = _rollout(seed, label, 0.0 if meter is None else SCALE, meter,
                                           spec["episode_length"], basis, base, parent["families"],
                                           profiles, parent["data"]["sensor_seed_offset"], progress)
                        if meter is None:
                            raw_rates = mapped_rates = [0.0] * len(episode)
                        else:
                            raw_rates, mapped_rates = meter.rates(spec["episode_length"], len(episode))
                        slots = {item.identifier: index for index, item in enumerate(profiles)}
                        for row, raw_rate, mapped_rate in zip(episode, raw_rates, mapped_rates, strict=True):
                            row.update({"hardware_condition": condition, "mapping": mapping,
                                        "slot": slots[row["profile"]],
                                        "normalized_correction_clipped_fraction": mapped_rate,
                                        "raw_hard_clipped_fraction": raw_rate})
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                            rows.append(row)
                        handle.flush()
        expected = 2 * len(branches) * spec["weather_count"] * len(FAMILIES) * 6
        if len(rows) != expected:
            raise RuntimeError("R5-G2-D1 完整分组回合不足")
        analysis = {} if quick else summarize(rows, cfg, quick=False, device=device)
        result = {
            "status": "QUICK_SMOKE_NO_CONCLUSION" if quick else "DEVELOPMENT_DIAGNOSTIC_REQUIRES_AUDIT",
            "records": len(rows), "completed_group_episodes": len(rows), "failed_group_episodes": 0,
            "weather_count": spec["weather_count"], "episode_length": spec["episode_length"],
            "physical_transitions": report["physical_transitions"],
            "hardware_conditions": list(CONDITIONS), "mapping_ids": list(MAPPINGS),
            "controllers": [label for label, _, _ in branches],
            "elapsed_seconds": time.perf_counter() - started, "analysis": analysis,
            "config_sha256": report["config_sha256"], "entry_sha256": report["entry_sha256"],
            "frozen_source_bundle_sha256": FROZEN_SOURCE,
            "g1_d3_config_sha256": D3_CONFIG_SHA256,
            "g1_d3_entry_sha256": D3_ENTRY_SHA256,
            "material_passport": {"origin_skill": "academic-research-suite / experiment-agent",
                                  "origin_mode": "run", "verification_status": "UNVERIFIED"},
            **cfg["boundary"],
            "next_action": "停止并等待只读审计；开发诊断不改判 G1，也不是独立确认",
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
        if created:
            try:
                write_json(output / "failure.json", {"traceback": traceback.format_exc(),
                                                      "automatic_retry": False})
            except Exception:
                pass  # Preserve the original exception if the filesystem is unwritable.
        raise
    finally:
        if progress is not None:
            progress.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--quick", action="store_true", help="16 帧 CUDA 冒烟，不产生科学性能结论")
    parser.add_argument("--preflight-only", action="store_true", help="只读预检，不生成输出")
    args = parser.parse_args()
    print(json.dumps(run(args.config, quick=args.quick,
                         preflight_only=args.preflight_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
