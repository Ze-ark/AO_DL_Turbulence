"""G2-D11-R1 CUDA 请求安全时序回归；复用失败开发种子，不是正式实验。"""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import diagnose_s4_r5_g2_d11_same_state as legacy
from scripts import diagnose_s4_r5_g2_d11_same_state_r1 as repair

OUTPUT = "outputs/s4_r5_g2_d11_request_safety_regression_r1"


@torch.no_grad()
def execute(cfg: dict, parent: dict, output: Path, device: torch.device) -> dict:
    seed, probe_step, hold = repair.FORMAL_BASE, 75, 12
    base, _ = repair.load_s1_config(repair._project_path(parent["environment_config"]))
    base = replace(base, num_modes=21, batch_size=1, episode_length=200)
    basis, _, _ = repair.build_action_basis(base, repair.ActionRepresentation("r5_zernike21", "zernike", 21), device)
    name = "action_penalty_0_policy_0_00512.pt"
    checkpoint = repair._project_path("outputs/s4_r5_g2_d8_action_penalty_zero_v1/checkpoints") / name
    if repair._file_sha256(checkpoint) != repair.d10.d9.D8_FINAL[name]:
        raise RuntimeError("冻结权重变化")
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    policy = repair.ResidualGRUPolicy(parent["policy"]["hidden_size"], parent["policy"]["output_size"]).to(device)
    policy.load_state_dict(saved["state_dict"])
    policy.eval().requires_grad_(False)
    pair = repair.d10.d9.d8.d2.g2.d1._profile_pairs(parent)
    progress = repair.d10.d9.d3.SparseProgress(output, device, 100)
    progress.phase("G2-D11-R1 CUDA 失败状态技术重放（非正式实验）", 750)
    started = time.perf_counter()
    checks, count = [], 0
    try:
        with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
            for condition, profiles in zip(repair.CONDITIONS, pair, strict=True):
                first = repair.RobustnessCondition.from_mapping(dict(parent["families"][0], base_seed=seed))
                env = repair.R5BatchedEnvironment(first.environment_config(base), device, basis,
                                                 parent["families"], profiles, parent["data"]["sensor_seed_offset"])
                raw, _ = env.reset(seed=seed)
                interface = repair.R4Interface()
                interface.reset(env.proxy(raw), episode_id=f"g2-d11-r1-technical-{condition}")
                for step in range(probe_step):
                    view = interface.snapshot()
                    command = policy(view.features, view.valid) * cfg["deployment_scale"]
                    action = interface.issue(repair.anchor_delta(view.features[:, -1],
                                             {"gain": .15, "leak": .10, "tracking_gain": .50}), command, step=step)
                    raw, _, term, trunc, info = env.step(action.requested_delta_rad)
                    if bool(term.any()) or bool(trunc.any()):
                        raise RuntimeError("技术重放前缀意外结束")
                    interface.observe_next(env.proxy(raw), step=step + 1,
                                           power=repair.PowerMeasurement(info["measured_power_in_bucket"], step, step + 1))
                    progress.tick({"来源前缀步": float(step + 1)})
                view = interface.snapshot()
                command = policy(view.features, view.valid) * cfg["deployment_scale"]
                state_hash = repair.require_same_state((env, interface), (env, interface))
                results = {}
                for name, candidate in repair.candidate_commands(command, cfg["candidate_epsilon"]).items():
                    branch = repair.fork_state((env, interface))
                    results[name] = repair.probe_branch(branch, candidate, step=probe_step,
                                                        hold_steps=hold, progress=progress)
                    del branch
                kwargs = dict(condition=condition, seed=seed, step=probe_step, member=0,
                              profiles=profiles, state_hash=state_hash, replay_tolerance=cfg["tolerances"]["replay"])
                old_error = None
                try:
                    legacy.probe_rows(results, **kwargs)
                except RuntimeError as exc:
                    old_error = str(exc)
                    if not any(old_error == f"G2-D11 延迟到达前候选已影响 {key}"
                               for key in ("violation", "saturation")):
                        raise
                if condition == "nominal_clone" and old_error is None:
                    raise RuntimeError("未复现原始请求安全时序错误，停止回归")
                rows = repair.probe_rows(results, **kwargs)
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                count += len(rows)
                delays = torch.tensor([p.slm_delay_frames for p in profiles], device=device).repeat_interleave(3)
                mask = torch.arange(hold, device=device)[None, :] < delays[:, None]
                changes = {}
                for key in repair.TRACE_METRICS:
                    changes[key] = max(float(torch.where(mask,
                        (value[key] - results["original"][key]).abs(), 0).max()) for value in results.values())
                if any(changes[key] > cfg["tolerances"]["replay"] for key in repair.DELAYED_METRICS):
                    raise RuntimeError("延迟前物理量变化，不能视为请求安全时序修复")
                checks.append({"condition": condition, "legacy_error": old_error,
                               "pre_arrival_max_abs_differences": changes, "repair_rows": len(rows)})
                del results, env, interface
        if progress.bar.n != 750 or count != 900:
            raise RuntimeError("技术重放预算不符")
        result = {"status": "CUDA_REQUEST_SAFETY_TIMING_REGRESSION_PASS_NO_SCIENTIFIC_CONCLUSION",
                  "device": str(device), "weather_seed": seed, "probe_step": probe_step, "member": 0,
                  "complete_source_episodes": 0, "source_prefixes": 36, "source_prefix_frames": 75,
                  "records": count, "batched_environment_steps": 750, "physical_transitions": 13500,
                  "policy_forward_calls": 152, "policy_forward_sample_steps": 2736,
                  "training_updates": 0, "confirmation_access": False, "real_slm_actions": False,
                  "automatic_retry": False, "analysis": {}, "checks": checks,
                  "elapsed_seconds": time.perf_counter() - started,
                  "scope": "failed_development_weather_debug_replay_only_do_not_merge_into_formal_dataset"}
        repair.write_json(output / "summary.json", result)
        repair.write_json(output / "SUCCESS.json", {f"{key}_sha256": repair._file_sha256(output / name)
            for key, name in (("summary", "summary.json"), ("records", "records.jsonl"), ("progress", "progress.jsonl"))})
        return result
    finally:
        progress.close()


def run() -> dict:
    cfg, parent, report, _, device = repair.preflight()
    output = repair._project_path(OUTPUT)
    if output.exists():
        raise FileExistsError(f"保留已有技术重放输出，禁止自动重试: {output}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    repair.write_json(output / "runtime.json", {"torch": str(torch.__version__), "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device), "git": repair.safe_git_record(),
        "entry_sha256": repair._file_sha256(Path(__file__)), "repair_preflight": report,
        "deterministic_algorithms": True, "allow_tf32": False})
    try:
        return execute(cfg, parent, output, device)
    except Exception:
        repair.write_json(output / "failure.json", {"traceback": traceback.format_exc(), "automatic_retry": False})
        raise


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
