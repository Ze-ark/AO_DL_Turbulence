"""G2-D12-A 补存动作前因果观测，严格配对已有标签；正式采集由用户启动。"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import diagnose_s4_r5_g2_d11_same_state_r1 as source
from src.rl.r4_control import R4Limits, project_request
from src.rl.r5_physics_adapter import causal_policy_features

CONFIG = "configs/experiments/s4_r5_g2_d12_causal_dataset_v1.yaml"
SOURCE_ENTRY_SHA256 = "fd3eb33aa8dcb9e6e4ca9f4f252400c0ab3ffde3a4ba639d494ac4ac0919e935"
SOURCE_CONFIG_SHA256 = "3db9748dae850ff6ed617a5b2b8df37788d63265fb404312339e1e0c857dc88c"
SOURCE_HASHES = {
    False: {"summary": "3d29242289dbef4c4ad05b907d902bc32eab7eaf5a5bf5a77b0f36cbc273a217",
            "records": "bbb9e4394d82d5c2d048aef23d46b02888db9451b970c5671f5452edfe0d3dcb",
            "progress": "0484b12fa57e943b0209e4fffa367e053a937653611471b70b36975abecfef17",
            "stream_manifest": "33015f7dc5f9d286ee2b22adea1a53c111a538da86e75dc1d7f3453c854a7bb9"},
    True: {"summary": "4f26861748a4d1f41e5f67c792e96fdc028ce2aeb15a0285d1ee33010536f4c8",
           "records": "709e3006c167d8b31c0176b6e2b7dbe36a0f69375d4ebaa395b9dfab9c537051",
           "progress": "361f939c8019f38cfd2c4f21880e1797c78a3e7e3c676b34b9f3259344a46833",
           "stream_manifest": "39ec36ab5a304209d142c5535f3dc91868b1b2718a87f5e7ec081e635f9e2d22"},
}
KEY_FIELDS = ("hardware_condition", "weather_seed", "probe_step", "member", "family", "slot")
LABEL_FIELDS = ("power", "measured_power", "violation", "saturation", "slew")


def contract(cfg: dict) -> None:
    expected = {"stage": "S4-D2-R5-G2-D12-A", "purpose": "causal_input_pairing_with_frozen_d11_candidate_labels",
        "runtime": {"device": "cuda", "formal_owner": "user_ide", "automatic_retry": False},
        "source_config": source.CONFIG, "input_fields": ["history", "valid", "command"],
        "split": {"unit": "complete_weather", "folds": 4, "held_out_weather_per_fold": 2},
        "output_directory": "outputs/s4_r5_g2_d12_causal_dataset_v1",
        "quick_directory": "outputs/s4_r5_g2_d12_causal_dataset_v1_quick",
        "boundary": {"training_updates": 0, "confirmation_access": False, "real_slm_actions": False,
                     "candidate_resimulation": False, "automatic_retry": False,
                     "independent_confirmation": False, "gate_reclassification": False}}
    if cfg != expected:
        raise ValueError("G2-D12-A 冻结采集合同变化")


def budget(spec: dict) -> dict:
    batches = 2 * spec["weather_count"] * spec["initializations"]
    steps = batches * spec["episode_length"]
    return {"samples": batches * len(spec["probe_steps"]) * 18,
            "complete_source_episodes": batches * 18, "batched_environment_steps": steps,
            "physical_transitions": steps * 18, "policy_forward_calls": steps,
            "policy_forward_sample_steps": steps * 18, "new_candidate_probes": 0}


def preflight(path: str | Path = CONFIG, *, quick: bool = False) -> tuple:
    cfg = source._load_yaml(source._project_path(path)); contract(cfg)
    if (source._file_sha256(Path(source.__file__)) != SOURCE_ENTRY_SHA256
            or source._file_sha256(source._project_path(source.CONFIG)) != SOURCE_CONFIG_SHA256):
        raise RuntimeError("G2-D12-A 冻结来源代码或配置变化")
    upstream = source._load_yaml(source._project_path(source.CONFIG)); source._contract(upstream)
    parent = source.verify_sources(); source.verify_streams()
    root = source._project_path(upstream["quick_directory" if quick else "output_directory"])
    if (root / "failure.json").exists():
        raise RuntimeError("G2-D12-A 来源存在失败标记")
    success = json.loads((root / "SUCCESS.json").read_text(encoding="utf-8"))
    for key, digest in SOURCE_HASHES[quick].items():
        filename = f"{key}.jsonl" if key in ("records", "progress") else f"{key}.json"
        if source._file_sha256(root / filename) != digest or success.get(f"{key}_sha256") != digest:
            raise RuntimeError(f"G2-D12-A 来源证据变化: {filename}")
    if json.loads((root / "config.json").read_text(encoding="utf-8")) != upstream:
        raise RuntimeError("G2-D12-A 来源运行配置不符")
    if json.loads((root / "stream_manifest.json").read_text(encoding="utf-8")) != source.stream_manifest(quick):
        raise RuntimeError("G2-D12-A 来源随机流不符")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    spec = upstream["quick" if quick else "data"]
    if (summary["quick"] != quick or summary["device"] != "cuda"
            or summary["completed_source_episodes"] != budget(spec)["complete_source_episodes"]
            or summary["confirmation_access"] or summary["training_updates"] or summary["real_slm_actions"]):
        raise RuntimeError("G2-D12-A 来源边界或预算不符")
    output = source._project_path(cfg["quick_directory" if quick else "output_directory"])
    if output.exists():
        raise FileExistsError(f"保留已有 G2-D12-A 输出，禁止覆盖: {output}")
    device = source.resolve_device("cuda")
    report = {"status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_IDE",
        "quick": quick, "device": str(device), **budget(spec), "source_hashes": SOURCE_HASHES[quick],
        "source_entry_sha256": SOURCE_ENTRY_SHA256, "source_config_sha256": SOURCE_CONFIG_SHA256,
        "entry_sha256": source._file_sha256(Path(__file__)), "config_sha256": source._file_sha256(source._project_path(path)),
        "frozen_source_bundle_sha256": source.d10.d9.d8.SOURCE_BUNDLE_SHA256, **cfg["boundary"]}
    return cfg, upstream, parent, spec, root, output, device, report


def load_groups(root: Path, spec: dict, *, quick: bool) -> dict:
    expected = {(c, seed, step, member, family, slot) for c in source.CONDITIONS
        for seed in source.stream_manifest(quick)["weather_bases"] for step in spec["probe_steps"]
        for member in range(spec["initializations"]) for family in source.FAMILIES for slot in range(6)}
    groups = {}
    retained = (*KEY_FIELDS, "candidate", "profile", "source_state_sha256", "slm_delay_frames", "hold_steps",
        "command", "normalized", "requested_delta", "requested_modal", "post_arrival_power",
        "violation_max", "saturation_max", "slew_max", *LABEL_FIELDS)
    with (root / "records.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line); key = tuple(row[k] for k in KEY_FIELDS); name = row["candidate"]
            if key not in expected or name not in source.CANDIDATES or name in groups.setdefault(key, {}):
                raise RuntimeError("G2-D12-A 来源状态未知或候选重复")
            groups[key][name] = {k: row[k] for k in retained}
    if set(groups) != expected or any(set(g) != set(source.CANDIDATES) for g in groups.values()):
        raise RuntimeError("G2-D12-A 来源完整网格缺失")
    return groups


def causal_inputs(history: torch.Tensor, valid: torch.Tensor, command: torch.Tensor, *, step: int) -> dict:
    # 只接收白名单张量，不允许将候选结果或任意 info 字典传入本函数。
    causal_policy_features(history, valid)
    n = len(history)
    if history.shape != (n, 8, 79) or valid.shape != (n, 8) or command.shape != (n, 11):
        raise RuntimeError("G2-D12-A 输入形状不符")
    if not bool(torch.isfinite(history).all() & torch.isfinite(command).all()):
        raise RuntimeError("G2-D12-A 输入非有限")
    times = history[..., 75]; command_times = history[..., 76]; power_times = history[..., 77]
    power_valid = history[..., 78].bool()
    expected_times = torch.arange(step - 7, step + 1, device=history.device)
    expected_valid = expected_times >= 0
    if (not torch.equal(valid, expected_valid.expand(n, -1))
            or not torch.equal(times[valid], expected_times.expand(n, -1)[valid].to(history.dtype))
            or bool((command_times[valid] != times[valid] - 1).any())
            or bool((power_times[valid & power_valid] >= times[valid & power_valid]).any())):
        raise RuntimeError("G2-D12-A 未来观测或时间戳错位")
    return {"history": history.detach().clone(), "valid": valid.detach().clone(), "command": command.detach().clone()}


def aligned_targets(variants: dict, command: torch.Tensor, action, *, state_hash: str,
                    profile, device: torch.device, tolerance: float) -> dict:
    original = variants["original"]
    for row in variants.values():
        if (row["source_state_sha256"] != state_hash or row["profile"] != profile.identifier
                or row["slm_delay_frames"] != profile.slm_delay_frames):
            raise RuntimeError("G2-D12-A 来源状态哈希或档位不匹配")
    for key, value in (("command", command), ("normalized", action.normalized_correction),
                       ("requested_delta", action.requested_delta_rad), ("requested_modal", action.requested_modal_rad)):
        saved = torch.tensor(original[key], dtype=value.dtype, device=device)
        if not torch.equal(value, saved):
            raise RuntimeError(f"G2-D12-A 重放原动作不匹配: {key}")
    ordered = [variants[name] for name in source.CANDIDATES]
    traces = torch.tensor([[row[k] for k in LABEL_FIELDS] for row in ordered], dtype=torch.float64, device=device)
    delay = profile.slm_delay_frames; hold = original["hold_steps"]
    if traces.shape != (25, 5, hold) or not bool(torch.isfinite(traces).all()) or not 0 <= delay < hold:
        raise RuntimeError("G2-D12-A 标签窗口或有限值不符")
    means = traces[:, :2, delay:].mean(-1)
    saved_means = torch.tensor([row["post_arrival_power"] for row in ordered], dtype=torch.float64, device=device)
    safety = traces[:, 2:].amax(-1)
    saved_safety = torch.tensor([[row[f"{k}_max"] for k in ("violation", "saturation", "slew")]
                                for row in ordered], dtype=torch.float64, device=device)
    if (not torch.allclose(means[:, 0], saved_means, atol=tolerance, rtol=0)
            or not torch.allclose(safety, saved_safety, atol=tolerance, rtol=0)
            or bool(((traces[:, 2:] < 0) | (traces[:, 2:] > 1)).any())
            or not torch.equal(traces[0], traces[1])):
        raise RuntimeError("G2-D12-A 标签均值、安全或等价对照不符")
    gains = means - means[0]
    return {"power_delta": gains[:, 0], "measured_power_delta": gains[:, 1],
            "safety_maxima": safety, "safety_pass": (safety <= safety[0] + .001).all(-1)}


@torch.no_grad()
def execute(cfg: dict, upstream: dict, parent: dict, spec: dict, root: Path, output: Path,
            device: torch.device, report: dict) -> dict:
    groups = load_groups(root, spec, quick=report["quick"])
    base, _ = source.load_s1_config(source._project_path(parent["environment_config"]))
    base = replace(base, num_modes=21, batch_size=1, episode_length=spec["episode_length"])
    basis, _, _ = source.build_action_basis(base, source.ActionRepresentation("r5_zernike21", "zernike", 21), device)
    policies = []
    for member in range(spec["initializations"]):
        name = f"action_penalty_0_policy_{member}_00512.pt"
        checkpoint = source._project_path("outputs/s4_r5_g2_d8_action_penalty_zero_v1/checkpoints") / name
        if source._file_sha256(checkpoint) != source.d10.d9.D8_FINAL[name]:
            raise RuntimeError("G2-D12-A 冻结权重变化")
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        policy = source.ResidualGRUPolicy(parent["policy"]["hidden_size"], parent["policy"]["output_size"]).to(device)
        policy.load_state_dict(saved["state_dict"]); policies.append(policy.eval().requires_grad_(False))
    profiles_pair = source.d10.d9.d8.d2.g2.d1._profile_pairs(parent)
    input_parts = {k: [] for k in cfg["input_fields"]}
    target_parts = {k: [] for k in ("power_delta", "measured_power_delta", "safety_maxima", "safety_pass")}
    seen = set(); completed = 0; started = time.perf_counter()
    progress = source.d10.d9.d3.SparseProgress(output, device, 8 if report["quick"] else 200)
    progress.phase("G2-D12-A CUDA 技术冒烟" if report["quick"] else "G2-D12-A 来源回合观测配对", report["batched_environment_steps"])
    try:
        with (output / "index.jsonl").open("w", encoding="utf-8") as handle:
            seeds = source.stream_manifest(report["quick"])["weather_bases"]
            for condition, profiles in zip(source.CONDITIONS, profiles_pair, strict=True):
                for wi, seed in enumerate(seeds):
                    for member, policy in enumerate(policies):
                        first = source.RobustnessCondition.from_mapping(dict(parent["families"][0], base_seed=seed))
                        env = source.R5BatchedEnvironment(replace(first.environment_config(base), episode_length=spec["episode_length"]),
                            device, basis, parent["families"], profiles, parent["data"]["sensor_seed_offset"])
                        raw, _ = env.reset(seed=seed); interface = source.R4Interface()
                        interface.reset(env.proxy(raw), episode_id=f"g2-d11-{condition}-{seed}-{member}")
                        for step in range(spec["episode_length"]):
                            view = interface.snapshot(); command = policy(view.features, view.valid) * 1.75
                            baseline = source.anchor_delta(view.features[:, -1], {"gain": .15, "leak": .10, "tracking_gain": .50})
                            if step in spec["probe_steps"]:
                                state_hash = source.require_same_state((env, interface), (env, interface))
                                inputs = causal_inputs(view.features, view.valid, command, step=step)
                                projected = project_request(interface.requested, baseline, command, R4Limits())
                                for slot, profile in enumerate(profiles):
                                    for fi, family in enumerate(source.FAMILIES):
                                        i = slot * 3 + fi; key = (condition, seed, step, member, family, slot)
                                        action = type(projected)(*(getattr(projected, k)[i] for k in projected.__dataclass_fields__))
                                        target = aligned_targets(groups[key], command[i], action, state_hash=state_hash,
                                            profile=profile, device=device, tolerance=upstream["tolerances"]["replay"])
                                        if key in seen:
                                            raise RuntimeError("G2-D12-A 采样状态重复")
                                        for k in input_parts: input_parts[k].append(inputs[k][i].cpu())
                                        for k in target_parts: target_parts[k].append(target[k].cpu())
                                        row = dict(zip(KEY_FIELDS, key, strict=True))
                                        row.update(sample_index=len(seen), weather_fold=None if report["quick"] else wi // 2,
                                            source_state_sha256=state_hash, profile=profile.identifier,
                                            candidate_order=list(source.CANDIDATES), metadata_not_model_input=True)
                                        handle.write(json.dumps(row, ensure_ascii=False) + "\n"); seen.add(key)
                                handle.flush()
                            action = interface.issue(baseline, command, step=step)
                            raw, _, term, trunc, info = env.step(action.requested_delta_rad)
                            interface.observe_next(env.proxy(raw), step=step + 1,
                                power=source.PowerMeasurement(info["measured_power_in_bucket"], step, step + 1))
                            if bool(trunc.any()) or bool(term.all()) != (step + 1 == spec["episode_length"]):
                                raise RuntimeError("G2-D12-A 来源回合不完整")
                            progress.tick({"当前回合帧": float(step + 1), "已配对样本": float(len(seen))})
                        completed += 18
        if set(groups) != seen or len(seen) != report["samples"] or completed != report["complete_source_episodes"] or progress.bar.n != report["batched_environment_steps"]:
            raise RuntimeError("G2-D12-A 配对网格或执行预算不符")
        torch.save({k: torch.stack(v) for k, v in input_parts.items()}, output / "inputs.pt")
        torch.save({k: torch.stack(v) for k, v in target_parts.items()}, output / "targets.pt")
        result = {**report, "status": "QUICK_CAUSAL_PAIRING_PASS_NO_SCIENTIFIC_CONCLUSION" if report["quick"] else "CAUSAL_DATASET_PAIRED_REQUIRES_READ_ONLY_AUDIT",
            "paired_samples": len(seen), "completed_source_episodes": completed, "all_source_hashes_and_actions_exact": True,
            "input_schema": {"history": [len(seen), 8, 79], "valid": [len(seen), 8], "command": [len(seen), 11]},
            "target_schema": {"power_delta": [len(seen), 25], "measured_power_delta": [len(seen), 25],
                              "safety_maxima": [len(seen), 25, 3], "safety_pass": [len(seen), 25]},
            "analysis": {}, "elapsed_seconds": time.perf_counter() - started,
            "material_passport": {"origin_skill": "academic-research-suite / experiment-agent", "origin_mode": "run",
                "origin_date": datetime.now(timezone.utc).date().isoformat(), "verification_status": "UNVERIFIED", "version_label": "g2_d12_causal_dataset_v1"},
            "next_action": "停止等待只读配对审计；不自动训练方向选择器或打开独立确认集"}
        source.write_json(output / "summary.json", result)
        source.write_json(output / "SUCCESS.json", {f"{k}_sha256": source._file_sha256(output / filename)
            for k, filename in (("summary", "summary.json"), ("inputs", "inputs.pt"), ("targets", "targets.pt"),
                                ("index", "index.jsonl"), ("progress", "progress.jsonl"))})
        return result
    finally:
        progress.close()


def run(path: str | Path = CONFIG, *, quick: bool = False, preflight_only: bool = False) -> dict:
    cfg, upstream, parent, spec, root, output, device, report = preflight(path, quick=quick)
    if preflight_only: return report
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    source.write_json(output / "config.json", cfg); source.write_json(output / "preflight.json", report)
    source.write_json(output / "runtime.json", {"torch": str(torch.__version__), "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device), "git": source.safe_git_record(), "deterministic_algorithms": True, "allow_tf32": False})
    source.write_json(output / "stream_manifest.json", source.stream_manifest(quick))
    try:
        return execute(cfg, upstream, parent, spec, root, output, device, report)
    except Exception:
        source.write_json(output / "failure.json", {"traceback": traceback.format_exc(), "automatic_retry": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG); parser.add_argument("--quick", action="store_true")
    parser.add_argument("--preflight-only", action="store_true"); args = parser.parse_args()
    print(json.dumps(run(args.config, quick=args.quick, preflight_only=args.preflight_only), ensure_ascii=False, indent=2))
