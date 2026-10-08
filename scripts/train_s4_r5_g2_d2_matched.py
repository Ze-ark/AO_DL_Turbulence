"""R5-G2-D2 等预算微调：训练动作映射 1.0 与 1.75 对照；正式训练仅由用户启动。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import diagnose_s4_r5_g2_d1_action_mapping as g2
from src.rl.r4_control import NominalCalibration
from src.rl.r4_dynamics_experiment import Progress, safe_git_record
from src.rl.r4_interface_smoke import write_json
from src.rl.r5_independent_confirmation_r2 import source_bundle_sha256
from src.rl.r5_policy_training import ResidualGRUPolicy, _rollout_batch
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _project_path
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.robust_control import RobustnessCondition


CONFIG = "configs/experiments/s4_r5_g2_d2_matched_training_v1.yaml"
ARMS = (("train_scale_1", 1.0), ("train_scale_1_75", 1.75))
FORMAL_BASE = 7_000_000
QUICK_BASE = 7_010_000
WEATHER_COUNT = 256
UPDATES = 512
FROZEN_SOURCE = "ccdc31faaa155361d8bd3e19ffc5bb82705f425a0ef4c7f58dd7f70831efd956"
G2_CONFIG_SHA256 = "c2bf5f34d251cb5e52793952cde4e604a595f8e6073228d9b8b6750d4d0cf81c"
G2_ENTRY_SHA256 = "de400aec9ec1c826a630a2fa2f3693f285288880aabfe2716a26227da3f26055"
G2_OUTPUT_HASHES = {
    "summary.json": "f73a09d1c03464dd306bfd52b7e4708a85bec523ab8f1a2b2df0f07600b0a9f7",
    "records.jsonl": "b3d309c846b1eee6d31d2ca9340a64094a75a55da7afa2e058fea94bc495a066",
    "progress.jsonl": "0026a709df75df0eec5b7ccec2e3dbcb82e3a3d8f193b775435ae31c45889d11",
    "stream_manifest.json": "227679b3cd62765938a503cb907c0e71b451c2d30ae45e017638579a8bfa9f80",
    "SUCCESS.json": "72c2f36131a14a7dae6db298a774fa626f9fd250c9d57ca8e5612c6a9a7222f8",
}


class TrainingScale(nn.Module):
    """只改训练阶段发往同一安全接口的修正幅度；基础策略权重仍单独保存。"""

    def __init__(self, policy: ResidualGRUPolicy, scale: float):
        super().__init__()
        if scale not in (1.0, 1.75):
            raise ValueError("未预注册的训练动作倍率")
        self.policy = policy
        self.scale = scale

    def forward(self, history: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        raw = self.policy(history, valid)
        return raw if self.scale == 1.0 else raw * self.scale


def _contract(cfg: dict) -> None:
    expected = {
        "stage": "S4-D2-R5-G2-D2",
        "purpose": "equal_budget_warm_start_train_deploy_action_mapping_comparison",
        "runtime": {"device": "cuda", "formal_owner": "user_ide", "automatic_retry": False},
        "g2_d1_config": g2.CONFIG,
        "g2_d1_config_sha256": G2_CONFIG_SHA256,
        "g2_d1_entry_sha256": G2_ENTRY_SHA256,
        "g2_d1_output": "outputs/s4_r5_g2_d1_action_mapping_v1",
        "g2_d1_output_hashes": G2_OUTPUT_HASHES,
        "source_bundle_sha256": FROZEN_SOURCE,
        "arms": [{"id": arm, "training_scale": scale, "deployment_scale": 1.75}
                 for arm, scale in ARMS],
        "data": {"train_seed_base": FORMAL_BASE, "seed_stride": 3,
                 "paired_weather_count": WEATHER_COUNT, "episode_length": 200,
                 "sensor_seed_offset": 50_000_000},
        "training": {"initializations": 3, "updates_per_arm_initialization": UPDATES,
                     "learning_rate": 0.00003, "gradient_norm_limit": 1.0,
                     "checkpoint_interval_updates": 64, "gradient_checkpointing": False},
        "objective": {"action_weight": 0.01, "smooth_weight": 0.001, "discount": 1.0},
        "quick": {"train_seed_base": QUICK_BASE, "initializations": 1,
                  "updates_per_arm_initialization": 2, "episode_length": 16},
        "reserved_not_accessed": {"development_seed_base": 7_100_000,
                                  "confirmation_seed_base": 7_200_000},
        "output_directory": "outputs/s4_r5_g2_d2_matched_training_v1",
        "quick_directory": "outputs/s4_r5_g2_d2_matched_training_v1_quick_r1",
        "boundary": {"confirmation_access": False, "development_evaluation_access": False,
                     "real_slm_actions": False, "automatic_retry": False,
                     "historical_results_read_only": True},
    }
    if cfg != expected:
        raise ValueError("R5-G2-D2 等预算训练合同被改变")


def _verify_lineage(cfg: dict) -> tuple[dict, dict]:
    if _file_sha256(_project_path(cfg["g2_d1_config"])) != G2_CONFIG_SHA256:
        raise RuntimeError("G2-D1 配置哈希不符")
    if _file_sha256(_project_path("scripts/diagnose_s4_r5_g2_d1_action_mapping.py")) != G2_ENTRY_SHA256:
        raise RuntimeError("G2-D1 入口哈希不符")
    g2_cfg = _load_yaml(_project_path(cfg["g2_d1_config"]))
    g2._contract(g2_cfg)
    r3_cfg, parent = g2._verify_lineage(g2_cfg)
    if source_bundle_sha256() != FROZEN_SOURCE:
        raise RuntimeError("R5 冻结物理/控制源码变化")
    root = _project_path(cfg["g2_d1_output"])
    if (root / "failure.json").exists():
        raise RuntimeError("G2-D1 历史输出有失败标记")
    for name, digest in G2_OUTPUT_HASHES.items():
        if _file_sha256(root / name) != digest:
            raise RuntimeError(f"G2-D1 历史证据变化: {name}")
    success = json.loads((root / "SUCCESS.json").read_text(encoding="utf-8"))
    for key, name in (("summary_sha256", "summary.json"), ("records_sha256", "records.jsonl"),
                      ("progress_sha256", "progress.jsonl"),
                      ("stream_manifest_sha256", "stream_manifest.json")):
        if success.get(key) != G2_OUTPUT_HASHES[name]:
            raise RuntimeError("G2-D1 成功标记不匹配")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("status") != "DEVELOPMENT_DIAGNOSTIC_REQUIRES_AUDIT"
            or summary.get("records") != 6048 or summary.get("failed_group_episodes") != 0
            or summary.get("training_updates") != 0 or summary.get("real_slm_actions") is not False
            or summary.get("confirmation_access") is not False
            or summary.get("analysis", {}).get("development_continue_criteria", {}).get("all") is not False
            or summary.get("config_sha256") != G2_CONFIG_SHA256
            or summary.get("entry_sha256") != G2_ENTRY_SHA256
            or summary.get("frozen_source_bundle_sha256") != FROZEN_SOURCE):
        raise RuntimeError("G2-D1 状态与只读审计结论不符")
    return r3_cfg, parent


def stream_manifest(*, quick: bool) -> dict:
    count = 1 if quick else WEATHER_COUNT
    base = QUICK_BASE if quick else FORMAL_BASE
    weather = [base + 3 * index for index in range(count)]
    turbulence = [seed + 1000 * slot + family for seed in weather
                  for slot in range(6) for family in range(3)]
    sensor = [seed + 1000 * slot + 50_000_000 for seed in weather for slot in range(6)]
    power = [seed + 1000 * slot + 60_000_000 for seed in weather for slot in range(6)]
    if (len(set(turbulence)) != 18 * count or len(set(sensor)) != 6 * count
            or len(set(power)) != 6 * count or max(turbulence) >= base + 10_000):
        raise RuntimeError("G2-D2 新天气随机流复用")
    return {"weather_bases": weather, "turbulence": turbulence,
            "sensor": sensor, "power": power,
            "condition_order": ["nominal_clone", "hardware_shift"],
            "same_streams_for_both_arms_and_three_initializations": True}


def schedule(update: int, *, quick: bool = False) -> tuple[str, int]:
    maximum = 2 if quick else UPDATES
    if not 1 <= update <= maximum:
        raise ValueError("训练更新序号超出预注册范围")
    condition = "nominal_clone" if update % 2 else "hardware_shift"
    base = QUICK_BASE if quick else FORMAL_BASE
    seed = base + 3 * ((update - 1) // 2)
    return condition, seed


def preflight(path: str | Path = CONFIG, *, quick: bool = False
              ) -> tuple[dict, dict, dict, dict, Path, torch.device]:
    cfg = _load_yaml(_project_path(path))
    _contract(cfg)
    r3_cfg, parent = _verify_lineage(cfg)
    if (parent["data"]["sensor_seed_offset"] != 50_000_000
            or [family["id"] for family in parent["families"]] != list(g2.FAMILIES)):
        raise RuntimeError("R5 因果观测或三类湍流合同变化")
    nominal, shift = g2.d1._profile_pairs(parent)
    if (len(nominal) != 6 or len(shift) != 6
            or [item.identifier for item in shift] != list(g2.PROFILES)):
        raise RuntimeError("六槽标称/新硬件档位合同变化")
    formal = stream_manifest(quick=False)
    smoke = stream_manifest(quick=True)
    for name in ("turbulence", "sensor", "power"):
        history = set(g2.stream_manifest(False)[name]) | set(g2.stream_manifest(True)[name])
        if (set(formal[name]) & set(smoke[name])
                or history & (set(formal[name]) | set(smoke[name]))):
            raise RuntimeError(f"训练/冒烟/G2-D1 {name} 随机流重叠")
    spec = cfg["quick"] if quick else cfg["data"]
    base, _ = load_s1_config(_project_path(parent["environment_config"]))
    displacement = []
    for family in parent["families"]:
        condition = RobustnessCondition.from_mapping(dict(family, base_seed=spec["train_seed_base"]))
        simulation = condition.environment_config(replace(base, episode_length=spec["episode_length"]))
        pixels = (simulation.wind_speed_mps * (1 + simulation.wind_speed_modulation_fraction)
                  * simulation.dt_s * spec["episode_length"] / simulation.sample_pitch_m)
        if pixels >= simulation.turbulence_grid_size:
            raise RuntimeError("相位屏在完整训练回合内重复")
        displacement.append(pixels)
    output = _project_path(cfg["quick_directory" if quick else "output_directory"])
    if output.exists():
        raise FileExistsError(f"保留已有 G2-D2 训练输出，不覆盖: {output}")
    device = resolve_device("cuda")
    initializations = cfg["quick"]["initializations"] if quick else cfg["training"]["initializations"]
    updates = cfg["quick"]["updates_per_arm_initialization"] if quick else UPDATES
    transitions = len(ARMS) * initializations * updates * 18 * spec["episode_length"]
    report = {
        "status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_IDE",
        "quick": quick, "device": str(device), "initializations": initializations,
        "arms": [arm for arm, _ in ARMS], "updates_per_arm_initialization": updates,
        "total_updates": len(ARMS) * initializations * updates,
        "episode_length": spec["episode_length"], "episodes_per_update": 18,
        "physical_transitions": transitions,
        "unique_weather_pairs": 1 if quick else WEATHER_COUNT,
        "maximum_displacement_pixels": displacement,
        "turbulence_grid_pixels": simulation.turbulence_grid_size,
        "config_sha256": _file_sha256(_project_path(path)),
        "entry_sha256": _file_sha256(Path(__file__)),
        "frozen_source_bundle_sha256": FROZEN_SOURCE,
        "g2_d1_config_sha256": G2_CONFIG_SHA256,
        "g2_d1_entry_sha256": G2_ENTRY_SHA256,
        **cfg["boundary"],
    }
    return cfg, r3_cfg, parent, report, output, device


def run(path: str | Path = CONFIG, *, quick: bool = False, preflight_only: bool = False) -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cfg, r3_cfg, parent, report, output, device = preflight(path, quick=quick)
    if preflight_only:
        return report
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    created = False
    progress: Progress | None = None
    loss_file = None
    started = time.perf_counter()
    try:
        output.mkdir(parents=True, exist_ok=False)
        created = True
        (output / "checkpoints").mkdir(exist_ok=False)
        write_json(output / "preflight.json", report)
        write_json(output / "config.json", cfg)
        write_json(output / "stream_manifest.json", stream_manifest(quick=quick))
        write_json(output / "runtime.json", {
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "git": safe_git_record(),
            "frozen_source_bundle_sha256": FROZEN_SOURCE,
        })
        base, _ = load_s1_config(_project_path(parent["environment_config"]))
        steps = report["episode_length"]
        base = replace(base, num_modes=21, batch_size=1, episode_length=steps)
        basis, _, _ = build_action_basis(base, ActionRepresentation("r5_zernike21", "zernike", 21), device)
        nominal, shift = g2.d1._profile_pairs(parent)
        profiles = {"nominal_clone": nominal, "hardware_shift": shift}
        calibration = NominalCalibration()
        progress = Progress(output, device)
        progress.phase("R5-G2-D2 等预算动作映射训练" if not quick else "R5-G2-D2 CUDA 快速冒烟",
                       report["total_updates"])
        loss_file = (output / "losses.jsonl").open("w", encoding="utf-8")
        results = []
        checkpoint_hashes = {}
        for arm_index, (arm, training_scale) in enumerate(ARMS):
            for init in range(report["initializations"]):
                torch.manual_seed(5_299_001 + init)
                torch.cuda.manual_seed_all(5_299_001 + init)
                source_path = _project_path(r3_cfg["training_output"]) / "checkpoints" / f"policy_{init}_02000.pt"
                source = torch.load(source_path, map_location=device, weights_only=True)
                if source["init"] != init or source["update"] != 2000:
                    raise RuntimeError("R5-2 冻结权重身份变化")
                policy = ResidualGRUPolicy(parent["policy"]["hidden_size"],
                                           parent["policy"]["output_size"]).to(device)
                policy.load_state_dict(source["state_dict"])
                policy.train()
                scaled = TrainingScale(policy, training_scale)
                optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["training"]["learning_rate"])
                running_loss = 0.0
                for update in range(1, report["updates_per_arm_initialization"] + 1):
                    condition, seed = schedule(update, quick=quick)
                    optimizer.zero_grad(set_to_none=True)
                    loss_score, timing = _rollout_batch(
                        scaled, cfg, {"steps": steps}, device, seed, parent["families"],
                        profiles[condition], basis, base, calibration,
                    )
                    loss = -loss_score
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("G2-D2 非有限训练损失")
                    backward_start = time.perf_counter()
                    loss.backward()
                    backward_seconds = time.perf_counter() - backward_start
                    torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                                   cfg["training"]["gradient_norm_limit"],
                                                   error_if_nonfinite=True)
                    optimizer.step()
                    loss_value = float(loss.detach())
                    running_loss += loss_value
                    record = {"arm": arm, "training_scale": training_scale,
                              "deployment_scale": 1.75, "initialization": init,
                              "update": update, "condition": condition, "weather_seed": seed,
                              "loss": loss_value, "average_loss": running_loss / update,
                              "policy_forward_seconds": timing["policy_forward"],
                              "env_step_seconds": timing["env_step"],
                              "interface_seconds": timing["interface"],
                              "objective_seconds": timing["objective"],
                              "backward_seconds": backward_seconds}
                    loss_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    loss_file.flush()
                    progress.tick({"实验臂序号": arm_index + 1, "初始化": init + 1,
                                   "初始化总数": report["initializations"],
                                   "当前批次": update, "平均损失": running_loss / update,
                                   "物理条件序号": 1 if condition == "nominal_clone" else 2})
                    if (update % cfg["training"]["checkpoint_interval_updates"] == 0
                            or update == report["updates_per_arm_initialization"]):
                        name = f"{arm}_policy_{init}_{update:05d}.pt"
                        checkpoint = output / "checkpoints" / name
                        torch.save({"state_dict": policy.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "arm": arm, "training_scale": training_scale,
                                    "deployment_scale": 1.75, "init": init, "update": update,
                                    "source_checkpoint_sha256": _file_sha256(source_path),
                                    "config_sha256": report["config_sha256"]}, checkpoint)
                        checkpoint_hashes[name] = _file_sha256(checkpoint)
                results.append({"arm": arm, "initialization": init,
                                "updates": report["updates_per_arm_initialization"],
                                "mean_training_loss": running_loss / report["updates_per_arm_initialization"]})
        write_json(output / "checkpoint_manifest.json", checkpoint_hashes)
        result = {
            "status": "QUICK_SMOKE_NO_SCIENTIFIC_CONCLUSION" if quick
                      else "DEVELOPMENT_TRAINING_COMPLETE_REQUIRES_AUDIT",
            "arms": [arm for arm, _ in ARMS], "results": results,
            "initializations": report["initializations"],
            "training_updates": report["total_updates"],
            "physical_transitions": report["physical_transitions"],
            "checkpoints": len(checkpoint_hashes),
            "elapsed_seconds": time.perf_counter() - started,
            "config_sha256": report["config_sha256"],
            "entry_sha256": report["entry_sha256"],
            "frozen_source_bundle_sha256": FROZEN_SOURCE,
            "g2_d1_output_hashes": G2_OUTPUT_HASHES,
            "material_passport": {"origin_skill": "academic-research-suite / experiment-agent",
                                  "origin_mode": "run", "verification_status": "UNVERIFIED"},
            **cfg["boundary"],
            "next_action": "停止并等待只读训练审计；训练损失不是独立科学性能，不自动运行评价或确认",
        }
        write_json(output / "summary.json", result)
        write_json(output / "SUCCESS.json", {
            "summary_sha256": _file_sha256(output / "summary.json"),
            "losses_sha256": _file_sha256(output / "losses.jsonl"),
            "progress_sha256": _file_sha256(output / "progress.jsonl"),
            "stream_manifest_sha256": _file_sha256(output / "stream_manifest.json"),
            "checkpoint_manifest_sha256": _file_sha256(output / "checkpoint_manifest.json"),
        })
        return result
    except Exception:
        if created:
            try:
                write_json(output / "failure.json", {"traceback": traceback.format_exc(),
                                                      "automatic_retry": False})
            except Exception:
                pass
        raise
    finally:
        if loss_file is not None:
            loss_file.close()
        if progress is not None:
            progress.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--quick", action="store_true", help="小型 CUDA 技术冒烟，不作模型排名")
    parser.add_argument("--preflight-only", action="store_true", help="只读预检，不创建输出")
    args = parser.parse_args()
    print(json.dumps(run(args.config, quick=args.quick,
                         preflight_only=args.preflight_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
