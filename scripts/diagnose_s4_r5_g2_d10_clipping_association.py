"""G2-D10：复用已封存回合定位裁剪关联；不新增仿真、模型推理或训练。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import diagnose_s4_r5_g2_d9_action_penalty_development as d9
from src.rl.r4_control import R4Limits, project_request
from src.rl.r4_dynamics_experiment import Progress, safe_git_record
from src.rl.r4_interface_smoke import write_json
from src.rl.s4_training import _file_sha256, _load_yaml, _project_path
from src.runtime import resolve_device

CONFIG = "configs/experiments/s4_r5_g2_d10_clipping_association_v1.yaml"
D9_CONFIG_SHA256 = "8586fc54f059429b3563a11a884e92294a4df3162b14ea96dfc2418daf65b553"
D9_ENTRY_SHA256 = "0170da10cdc3e24ef783fbab5f3e41e565ce178a7b0fba36f4d9916aa5e7d73e"
D9_HASHES = {
    "summary.json": "a68b178f58f438c0336ede47945876879062cc89085ddc4b06edd4bd110add22",
    "records.jsonl": "0d0b28356ca10afbc8831565e89bef14f1767a25589a98e0b47ef31c04e2d9af",
    "progress.jsonl": "2de2389ffadb34643d4b8f5287b1f5622dcaad43d224c77248b84c0e6ce9c92c",
    "stream_manifest.json": "36ba154d4698ab44a9e39cea4cd1a29b4f47ae8b748fd3dfdc684b1599ffd3a6",
    "SUCCESS.json": "c073d8244e22fbc4e768e8eebb68a2ae745e33ba4fc336e307937ed6cd829c8e",
    "config.json": "ba36c75f1aec2eff173d0122434205ccef87f32f842a6b0aa83ed912a8938837",
    "preflight.json": "d4f8492362468b0f86b2d69dbc71da1aeb911e16045b3ec12b31d47f37a88874",
    "runtime.json": "ac9cebb5d511cdb779b60cd63783df9c6fe9127e042311425e3321448459939f",
}


def _contract(cfg: dict) -> None:
    expected = {
        "stage": "S4-D2-R5-G2-D10",
        "purpose": "cached_episode_clipping_association_and_controlled_projection_jacobian",
        "runtime": {"device": "cuda", "automatic_retry": False},
        "d9_config": d9.CONFIG, "d9_config_sha256": D9_CONFIG_SHA256,
        "d9_entry_sha256": D9_ENTRY_SHA256,
        "d9_output": "outputs/s4_r5_g2_d9_action_penalty_development_v1",
        "d9_hashes": D9_HASHES,
        "analysis": {
            "association_x": "new_normalized_clipping_fraction",
            "association_y": "new_minus_old_absolute_power",
            "center_within": "member_family_slot_across_weather",
            "bootstrap_unit": "complete_weather_shared_across_all_cells",
            "bootstrap_seed": 7_543_456, "bootstrap_repeats": 5000,
            "descriptive_interval": .95,
            "group_table": "all_family_slot_cells_without_selection",
        },
        "output_directory": "outputs/s4_r5_g2_d10_clipping_association_v1",
        "boundary": {"new_physical_transitions": 0, "model_forward_calls": 0,
                     "training_updates": 0, "confirmation_access": False,
                     "real_slm_actions": False, "causal_performance_claim": False,
                     "gate_reclassification": False},
    }
    if cfg != expected:
        raise ValueError("G2-D10 只读来源与诊断合同变化")


def preflight(path: str | Path = CONFIG) -> tuple[dict, dict, list[dict], Path, torch.device]:
    cfg = _load_yaml(_project_path(path))
    _contract(cfg)
    if (_file_sha256(_project_path(d9.CONFIG)) != D9_CONFIG_SHA256
            or _file_sha256(Path(d9.__file__)) != D9_ENTRY_SHA256):
        raise RuntimeError("G2-D9 冻结入口或配置变化")
    source_cfg = _load_yaml(_project_path(d9.CONFIG))
    d9._contract(source_cfg)
    d9._verify_training(source_cfg)
    root = _project_path(cfg["d9_output"])
    if (root / "failure.json").exists():
        raise RuntimeError("G2-D9 有失败标记")
    for name, digest in D9_HASHES.items():
        if _file_sha256(root / name) != digest:
            raise RuntimeError(f"G2-D9 输入证据变化: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if (summary["records"] != 8064 or summary["failed_group_episodes"] != 0
            or summary["physical_transitions"] != 1_612_800
            or summary["analysis"]["continue_criteria"]["all"] is not False
            or summary["training_updates"] != 0 or summary["confirmation_access"] is not False):
        raise RuntimeError("G2-D9 完成性或未过门槛结论不符")
    rows = [json.loads(line) for line in (root / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    output = _project_path(cfg["output_directory"])
    if output.exists():
        raise FileExistsError(f"保留已有 G2-D10 输出，不覆盖: {output}")
    device = resolve_device("cuda")
    report = {"status": "READY_FOR_CACHED_ANALYSIS", "device": str(device),
              "source_records": len(rows), "d9_hashes": D9_HASHES,
              "config_sha256": _file_sha256(_project_path(path)),
              "entry_sha256": _file_sha256(Path(__file__)), **cfg["boundary"]}
    return cfg, report, rows, output, device


def _correlation(x: torch.Tensor, y: torch.Tensor, *, within: bool) -> float | None:
    """W×G；G为初始化/湍流/槽位，不把重复分组当独立天气。"""
    if (x.shape != y.shape or x.ndim != 2 or min(x.shape) < 1
            or x.device != y.device or not x.is_floating_point() or not y.is_floating_point()
            or not bool(torch.isfinite(x).all() & torch.isfinite(y).all())):
        raise ValueError("关联输入必须为同形有限 W×G 矩阵")
    a = x - (x.mean(0, keepdim=True) if within else x.mean())
    b = y - (y.mean(0, keepdim=True) if within else y.mean())
    aa, bb = a.square().sum(), b.square().sum()
    if float(aa) <= 1e-24 or float(bb) <= 1e-24:
        return None
    return float(((a * b).sum() / torch.sqrt(aa * bb)).clamp(-1, 1))


def association(x: torch.Tensor, y: torch.Tensor, *, seed: int, repeats: int) -> dict:
    if type(repeats) is not int or repeats < 1 or type(seed) is not int or seed < 0:
        raise ValueError("重采样次数须为正整数，种子须为非负整数")
    point = _correlation(x, y, within=True)
    if point is None:
        return {"pooled_correlation": _correlation(x, y, within=False),
                "within_cell_correlation": None, "weather_cluster_ci95": None,
                "valid_bootstrap_draws": 0, "requested_bootstrap_draws": repeats,
                "undefined_reason": "no_within_cell_variance"}
    generator = torch.Generator(device=x.device).manual_seed(seed)
    draws = torch.randint(x.shape[0], (repeats, x.shape[0]), generator=generator, device=x.device)
    # 所有G分组共享天气索引；每次重采样重新在分组内去均值。
    a, b = x[draws], y[draws]
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    aa, bb = a.square().sum((1, 2)), b.square().sum((1, 2))
    valid = (aa > 1e-24) & (bb > 1e-24)
    estimates = ((a * b).sum((1, 2))[valid] / torch.sqrt(aa[valid] * bb[valid])).clamp(-1, 1)
    interval = (None if len(estimates) == 0 else torch.quantile(
        estimates, estimates.new_tensor((.025, .975))).tolist())
    return {"pooled_correlation": _correlation(x, y, within=False),
            "within_cell_correlation": point, "weather_cluster_ci95": interval,
            "valid_bootstrap_draws": int(valid.sum()), "requested_bootstrap_draws": repeats,
            "undefined_reason": None if interval is not None else "no_valid_bootstrap_draw"}


def controlled_jacobian(device: torch.device) -> dict:
    """解析受控请求投影的直接梯度，未调用仿真环境或策略模型。"""
    limits = R4Limits()
    command = torch.tensor([[-1.4, -1.2, -.8, -.4, -.1, 0., .1, .4, .8, 1.2, 1.4]],
                           dtype=torch.float64, device=device, requires_grad=True)
    prior = torch.zeros((1, 21), dtype=command.dtype, device=device)
    action = project_request(prior, torch.zeros_like(prior), command, limits)
    normalized_gradient = torch.autograd.grad(action.normalized_correction.sum(), command, retain_graph=True)[0]
    request_gradient = torch.autograd.grad(action.requested_delta_rad[:, 10:].sum(), command)[0]
    inside = command.detach().abs() < 1
    expected = inside.to(command.dtype)
    if (not torch.allclose(normalized_gradient, expected, atol=1e-12, rtol=0)
            or not torch.allclose(request_gradient, limits.correction_rad * expected, atol=1e-12, rtol=0)):
        raise RuntimeError("受控动作投影雅可比不符合冻结限制")
    return {"scaled_normalized_command": command.detach().flatten().tolist(),
            "normalized_output": action.normalized_correction.flatten().tolist(),
            "normalized_output_gradient": normalized_gradient.flatten().tolist(),
            "requested_high_order_delta_gradient": request_gradient.flatten().tolist(),
            "outside_direct_gradient_zero": bool((request_gradient[~inside] == 0).all()),
            "inside_direct_gradient": limits.correction_rad,
            "scope": "zero_prior_zero_baseline_direct_current_action_path_only",
            "not_observed_training_gradient_fraction": True}


def analyze(rows: list[dict], cfg: dict, *, device: torch.device) -> dict:
    # 先完整验证上游每一条回合及其原门槛；不会产生新轨迹。
    source_cfg = _load_yaml(_project_path(d9.CONFIG))
    prior = d9.summarize(rows, source_cfg, device=device)
    if prior["continue_criteria"]["all"]:
        raise RuntimeError("G2-D9 原未过门槛结论改变")
    weather = d9.stream_manifest(False)["weather_bases"]
    keys = {(r["hardware_condition"], r["controller"], r["family"], r["weather_seed"], r["slot"]): r for r in rows}
    cells, table = {}, []
    for condition in d9.CONDITIONS:
        def tensor(metric: str, arm: str) -> torch.Tensor:
            return torch.tensor([[[[keys[(condition, f"{arm}_{member}", family, seed, slot)][metric]
                                    for slot in range(6)] for seed in weather]
                                  for family in d9.FAMILIES] for member in range(3)],
                                dtype=torch.float64, device=device)
        new_power, old_power = tensor("power", d9.d8.ARM), tensor("power", d9.d8.COMPARATOR)
        new_clip = tensor("normalized_correction_clipped_fraction", d9.d8.ARM)
        old_clip = tensor("normalized_correction_clipped_fraction", d9.d8.COMPARATOR)
        safety = {metric: tensor(metric, d9.d8.ARM)
                  for metric in ("saturation", "slew_limited", "violation", "requested_applied_gap_abs")}
        baseline = torch.tensor([[[keys[(condition, "integrator", family, seed, slot)]["power"]
                                  for slot in range(6)] for seed in weather] for family in d9.FAMILIES],
                                dtype=torch.float64, device=device)
        effect = new_power - old_power
        x, y = new_clip.permute(2, 0, 1, 3).reshape(32, -1), effect.permute(2, 0, 1, 3).reshape(32, -1)
        linked = association(x, y, seed=cfg["analysis"]["bootstrap_seed"], repeats=cfg["analysis"]["bootstrap_repeats"])
        source = prior["cells"][condition]
        cells[condition] = {
            "new_clip_fraction": float(new_clip.mean()), "old_clip_fraction": float(old_clip.mean()),
            "new_relative_gain": source["new_relative_gain"], "old_relative_gain": source["old_relative_gain"],
            "power_shortfall_to_development_margin": max(0., .0105 * float(baseline.mean())
                                                        - float((new_power - baseline).mean())),
            "clip_vs_new_minus_old_power_association": linked,
        }
        for family_index, family in enumerate(d9.FAMILIES):
            for slot, profile in enumerate(d9.PROFILES):
                b = float(baseline[family_index, :, slot].mean())
                n, o = float(new_power[:, family_index, :, slot].mean()), float(old_power[:, family_index, :, slot].mean())
                row = {"hardware_condition": condition, "family": family, "slot": slot, "profile": profile,
                       "weather_count": 32, "initializations": 3,
                       "new_power": n, "old_power": o, "integrator_power": b,
                       "new_relative_gain": (n-b)/b, "new_minus_old_power": n-o,
                       "new_clip_fraction": float(new_clip[:, family_index, :, slot].mean()),
                       "old_clip_fraction": float(old_clip[:, family_index, :, slot].mean())}
                for metric, values in safety.items():
                    row[f"new_{metric}"] = float(values[:, family_index, :, slot].mean())
                table.append(row)
    return {"status": "CACHED_DESCRIPTIVE_DIAGNOSTIC_NOT_CAUSAL_PERFORMANCE_TEST",
            "cells": cells, "group_table": table,
            "prior_continue_criteria": prior["continue_criteria"],
            "unit": "32_complete_weather_clusters_per_condition; 54_repeated_member_family_slot_cells",
            "interval_scope": "descriptive_95_percent_not_a_new_gate_or_confirmatory_test",
            "limitations": ["episode_average_not_frame_or_action_component_trace",
                            "within_cell_weather_strength_can_confound_both_clipping_and_power",
                            "correlation_does_not_estimate_power_lost_to_clipping",
                            "post_failure_development_diagnostic_no_gate_reclassification"]}


def run(path: str | Path = CONFIG, *, preflight_only: bool = False) -> dict:
    cfg, report, rows, output, device = preflight(path)
    if preflight_only:
        return report
    created = False
    progress: Progress | None = None
    try:
        output.mkdir(parents=True, exist_ok=False)
        created = True
        write_json(output / "preflight.json", report)
        write_json(output / "config.json", cfg)
        write_json(output / "runtime.json", {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                                             "gpu": torch.cuda.get_device_name(device), "git": safe_git_record()})
        progress = Progress(output, device)
        progress.phase("G2-D10 已有回合分组诊断", 2)
        with torch.no_grad():
            analysis = analyze(rows, cfg, device=device)
        progress.tick({"已核对回合": len(rows)})
        jacobian = controlled_jacobian(device)
        progress.tick({"新仿真转移": 0})
        write_json(output / "group_table.json", analysis.pop("group_table"))
        result = {"status": "CACHED_DIAGNOSTIC_COMPLETE_REQUIRES_READ_ONLY_AUDIT",
                  "source_records": len(rows), "analysis": analysis,
                  "controlled_projection_jacobian": jacobian,
                  "d9_hashes": D9_HASHES, "config_sha256": report["config_sha256"],
                  "entry_sha256": report["entry_sha256"], **cfg["boundary"],
                  "material_passport": {"origin_skill": "academic-research-suite / experiment-agent",
                                        "origin_mode": "validate", "verification_status": "ANALYZED",
                                        "origin_date": datetime.now(timezone.utc).date().isoformat(),
                                        "version_label": "g2_d10_cached_diagnostic_v1"},
                  "next_action": "根据记录定位后再设计新天气同状态机制探针；不调惩罚权重、不扩大动作限制、不打开确认集"}
        write_json(output / "summary.json", result)
        write_json(output / "SUCCESS.json", {"summary_sha256": _file_sha256(output / "summary.json"),
                                             "group_table_sha256": _file_sha256(output / "group_table.json"),
                                             "progress_sha256": _file_sha256(output / "progress.jsonl")})
        return result
    except Exception:
        if created:
            try:
                write_json(output / "failure.json", {"traceback": traceback.format_exc(), "automatic_retry": False})
            except Exception:
                pass
        raise
    finally:
        if progress is not None:
            progress.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, preflight_only=args.preflight_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
