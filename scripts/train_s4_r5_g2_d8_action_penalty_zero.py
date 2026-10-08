"""G2-D8：只去掉动作幅度惩罚的等预算训练；正式运行须由用户在 IDE 启动。"""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import train_s4_r5_g2_d2_matched as d2
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


CONFIG = "configs/experiments/s4_r5_g2_d8_action_penalty_zero_v1.yaml"
D2_CONFIG_SHA256 = "297535002d1df67e4b2ce29e8a4d749c361bcda0c94e6e199195cdb2748dc63a"
D2_ENTRY_SHA256 = "f747952ae9d4d736e78dba7e4168ae41e49d876fc26a86ca8a602304a96d0706"
D2_RUNTIME_SHA256 = "7d8bf7af0f7e95eb79b12d3d0ba69f5a051948a670494bbc35a4703f23b03bea"
D2_HASHES = {
    "summary.json": "68471420177832f2f883448a8a3ab01b87c5481773e515570aca308201d22f8e",
    "losses.jsonl": "a447df87d3bbcefb94726a71f8ac0126275f03cfd7c8d36f6c9c90948445ff02",
    "progress.jsonl": "86d83bfd42f711b5f1c9e230a8ba7d0ff078b374f4c94cee348118765f4b33b9",
    "stream_manifest.json": "6312e5ec18ba93685dd4383c3ddc3689f4297ea60c28bb3a819a97adac026f58",
    "checkpoint_manifest.json": "a1d5c713dd2c35599e0c397659fc07c4e2c1d6f01acce3e529d1c4517a710e5f",
    "SUCCESS.json": "db66743be8529b70a81dd85e9d7a0acaf27a03d90d2037e8c9a4ef62f13d9287",
}
ARM = "action_penalty_0"
COMPARATOR = "train_scale_1_75"
TRAIN_SCALE = 1.75
QUICK_BASE = 7_510_000
DEVELOPMENT_BASE = 7_500_000
CONFIRMATION_BASE = 7_600_000
SOURCE_BUNDLE_SHA256 = d2.FROZEN_SOURCE
REFERENCE_RUNTIME = {
    "torch": "2.11.0+cu128",
    "cuda": "12.8",
    "gpu": "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
}


def _contract(cfg: dict) -> None:
    """固定单因素差异，防止悄悄更改预算、天气或安全条件。"""
    d2_cfg = _load_yaml(_project_path(d2.CONFIG))
    d2._contract(d2_cfg)
    expected = {
        "stage": "S4-D2-R5-G2-D8",
        "purpose": "equal_budget_action_penalty_single_factor_ablation",
        "runtime": {"device": "cuda", "formal_owner": "user_ide", "automatic_retry": False},
        "d2_config": d2.CONFIG,
        "d2_config_sha256": D2_CONFIG_SHA256,
        "d2_entry_sha256": D2_ENTRY_SHA256,
        "d2_output": "outputs/s4_r5_g2_d2_matched_training_v1",
        "d2_output_hashes": D2_HASHES,
        "d2_runtime_sha256": D2_RUNTIME_SHA256,
        "reference_runtime": REFERENCE_RUNTIME,
        "comparator": {"id": COMPARATOR, "training_scale": TRAIN_SCALE,
                       "deployment_scale": TRAIN_SCALE, "action_weight": 0.01},
        "new_arm": {"id": ARM, "training_scale": TRAIN_SCALE,
                    "deployment_scale": TRAIN_SCALE},
        "source_bundle_sha256": SOURCE_BUNDLE_SHA256,
        "data": d2_cfg["data"],
        "training": d2_cfg["training"],
        "objective": {**d2_cfg["objective"], "action_weight": 0.0},
        "quick": {"train_seed_base": QUICK_BASE, "initializations": 1,
                  "updates_per_arm_initialization": 2, "episode_length": 16},
        "reserved_not_accessed": {"development_seed_base": DEVELOPMENT_BASE,
                                  "development_weather_count": 32,
                                  "confirmation_seed_base": CONFIRMATION_BASE},
        "output_directory": "outputs/s4_r5_g2_d8_action_penalty_zero_v1",
        "quick_directory": "outputs/s4_r5_g2_d8_action_penalty_zero_v1_quick",
        "boundary": {"confirmation_access": False, "development_evaluation_access": False,
                     "real_slm_actions": False, "automatic_retry": False,
                     "historical_results_read_only": True},
    }
    if cfg != expected:
        raise ValueError("G2-D8 单因素训练合同变化")


def _verify_d2(cfg: dict) -> tuple[dict, dict, dict, dict[int, str], dict[int, str]]:
    if (_file_sha256(_project_path(cfg["d2_config"])) != D2_CONFIG_SHA256
            or _file_sha256(Path(d2.__file__)) != D2_ENTRY_SHA256):
        raise RuntimeError("G2-D2 冻结配置或训练入口变化")
    d2_cfg = _load_yaml(_project_path(cfg["d2_config"]))
    d2._contract(d2_cfg)
    r3_cfg, parent = d2._verify_lineage(d2_cfg)
    if (source_bundle_sha256() != SOURCE_BUNDLE_SHA256
            or d2_cfg["objective"] != {"action_weight": 0.01,
                                        "smooth_weight": 0.001, "discount": 1.0}):
        raise RuntimeError("G2-D2 冻结源码或原目标变化")
    root = _project_path(cfg["d2_output"])
    if (root / "failure.json").exists() or _file_sha256(root / "runtime.json") != D2_RUNTIME_SHA256:
        raise RuntimeError("G2-D2 历史训练失败或运行环境记录变化")
    for name, digest in D2_HASHES.items():
        if _file_sha256(root / name) != digest:
            raise RuntimeError(f"G2-D2 历史证据变化: {name}")
    success = json.loads((root / "SUCCESS.json").read_text(encoding="utf-8"))
    for key, name in (("summary_sha256", "summary.json"),
                      ("losses_sha256", "losses.jsonl"),
                      ("progress_sha256", "progress.jsonl"),
                      ("stream_manifest_sha256", "stream_manifest.json"),
                      ("checkpoint_manifest_sha256", "checkpoint_manifest.json")):
        if success.get(key) != D2_HASHES[name]:
            raise RuntimeError(f"G2-D2 成功标记不匹配: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("status") != "DEVELOPMENT_TRAINING_COMPLETE_REQUIRES_AUDIT"
            or summary.get("arms") != [arm for arm, _ in d2.ARMS]
            or summary.get("training_updates") != 3072
            or summary.get("physical_transitions") != 11_059_200
            or summary.get("checkpoints") != 48
            or summary.get("config_sha256") != D2_CONFIG_SHA256
            or summary.get("entry_sha256") != D2_ENTRY_SHA256
            or summary.get("frozen_source_bundle_sha256") != SOURCE_BUNDLE_SHA256
            or summary.get("confirmation_access") is not False
            or summary.get("real_slm_actions") is not False):
        raise RuntimeError("G2-D2 原训练完成性或边界不符")
    old_stream = json.loads((root / "stream_manifest.json").read_text(encoding="utf-8"))
    if old_stream != d2.stream_manifest(quick=False):
        raise RuntimeError("G2-D2 原天气流不符")
    manifest = json.loads((root / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    expected_names = {
        f"{arm}_policy_{member}_{update:05d}.pt"
        for arm, _ in d2.ARMS for member in range(3)
        for update in range(64, d2.UPDATES + 1, 64)
    }
    if set(manifest) != expected_names:
        raise RuntimeError("G2-D2 检查点网格不完整")
    for name, digest in manifest.items():
        if _file_sha256(root / "checkpoints" / name) != digest:
            raise RuntimeError(f"G2-D2 检查点变化: {name}")
    source_hashes: dict[int, str] = {}
    comparator_hashes: dict[int, str] = {}
    for member in range(3):
        name = f"{COMPARATOR}_policy_{member}_00512.pt"
        saved = torch.load(root / "checkpoints" / name, map_location="cpu", weights_only=True)
        source = (_project_path(r3_cfg["training_output"]) / "checkpoints"
                  / f"policy_{member}_02000.pt")
        source_hash = _file_sha256(source)
        if (saved.get("arm") != COMPARATOR or saved.get("init") != member
                or saved.get("update") != d2.UPDATES
                or saved.get("training_scale") != TRAIN_SCALE
                or saved.get("deployment_scale") != TRAIN_SCALE
                or saved.get("source_checkpoint_sha256") != source_hash
                or saved.get("config_sha256") != D2_CONFIG_SHA256):
            raise RuntimeError("G2-D2 历史对照检查点身份不符")
        source_hashes[member] = source_hash
        comparator_hashes[member] = manifest[name]
    count, comparator_count = 0, 0
    comparator_grid: set[tuple[int, int]] = set()
    with (root / "losses.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            count += 1
            if row["arm"] == COMPARATOR:
                comparator_count += 1
                condition, seed = d2.schedule(row["update"], quick=False)
                key = (row["initialization"], row["update"])
                if key in comparator_grid:
                    raise RuntimeError("G2-D2 对照训练日志有重复更新")
                comparator_grid.add(key)
                if (row["condition"] != condition or row["weather_seed"] != seed
                        or row["training_scale"] != TRAIN_SCALE
                        or row["deployment_scale"] != TRAIN_SCALE
                        or not math.isfinite(row["loss"])):
                    raise RuntimeError("G2-D2 对照训练日志调度或损失不符")
    expected_grid = {(member, update) for member in range(3)
                     for update in range(1, d2.UPDATES + 1)}
    if count != 3072 or comparator_count != 1536 or comparator_grid != expected_grid:
        raise RuntimeError("G2-D2 历史对照训练日志数量不足")
    return d2_cfg, r3_cfg, parent, source_hashes, comparator_hashes


def schedule(update: int, *, quick: bool) -> tuple[str, int]:
    if not quick:
        return d2.schedule(update, quick=False)
    if not 1 <= update <= 2:
        raise ValueError("G2-D8 快速冒烟更新号超出范围")
    return ("nominal_clone" if update % 2 else "hardware_shift",
            QUICK_BASE + 3 * ((update - 1) // 2))


def stream_manifest(*, quick: bool) -> dict:
    if not quick:
        return d2.stream_manifest(quick=False)
    weather = [QUICK_BASE]
    return {
        "weather_bases": weather,
        "turbulence": [QUICK_BASE + 1000 * slot + family
                       for slot in range(6) for family in range(3)],
        "sensor": [QUICK_BASE + 1000 * slot + 50_000_000 for slot in range(6)],
        "power": [QUICK_BASE + 1000 * slot + 60_000_000 for slot in range(6)],
        "condition_order": ["nominal_clone", "hardware_shift"],
        "same_streams_for_both_conditions": True,
    }


def preflight(path: str | Path = CONFIG, *, quick: bool = False
              ) -> tuple[dict, dict, dict, dict, Path, torch.device, dict[int, str], dict[int, str]]:
    cfg = _load_yaml(_project_path(path))
    _contract(cfg)
    d2_cfg, r3_cfg, parent, source_hashes, comparator_hashes = _verify_d2(cfg)
    if (cfg["data"] != d2_cfg["data"] or cfg["training"] != d2_cfg["training"]
            or cfg["objective"] != {**d2_cfg["objective"], "action_weight": 0.0}
            or [family["id"] for family in parent["families"]] != list(d2.g2.FAMILIES)
            or parent["data"]["sensor_seed_offset"] != 50_000_000):
        raise RuntimeError("G2-D8 训练变量不是单因素变化")
    nominal, shifted = d2.g2.d1._profile_pairs(parent)
    if (len(nominal) != 6 or len(shifted) != 6
            or [item.identifier for item in shifted] != list(d2.g2.PROFILES)):
        raise RuntimeError("G2-D8 六槽硬件配对不符")
    formal, smoke = stream_manifest(quick=False), stream_manifest(quick=True)
    if formal != d2.stream_manifest(quick=False):
        raise RuntimeError("G2-D8 正式训练流未与旧对照匹配")
    for name in ("turbulence", "sensor", "power"):
        if set(smoke[name]) & set(formal[name]):
            raise RuntimeError("G2-D8 技术冒烟与正式训练天气重叠")
    spec = cfg["quick"] if quick else cfg["data"]
    base, _ = load_s1_config(_project_path(parent["environment_config"]))
    displacement = []
    for family in parent["families"]:
        condition = RobustnessCondition.from_mapping(dict(family, base_seed=spec["train_seed_base"]))
        simulation = condition.environment_config(replace(base, episode_length=spec["episode_length"]))
        pixels = (simulation.wind_speed_mps
                  * (1 + simulation.wind_speed_modulation_fraction)
                  * simulation.dt_s * spec["episode_length"] / simulation.sample_pitch_m)
        if pixels >= simulation.turbulence_grid_size:
            raise RuntimeError("G2-D8 相位屏在完整训练回合内重复")
        displacement.append(pixels)
    output = _project_path(cfg["quick_directory" if quick else "output_directory"])
    if output.exists():
        raise FileExistsError(f"保留已有 G2-D8 训练输出，不覆盖: {output}")
    device = resolve_device("cuda")
    old_runtime = json.loads((_project_path(cfg["d2_output"]) / "runtime.json").read_text(
        encoding="utf-8"))
    actual_runtime = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                      "gpu": torch.cuda.get_device_name(device)}
    if (actual_runtime != REFERENCE_RUNTIME
            or {key: old_runtime.get(key) for key in REFERENCE_RUNTIME} != REFERENCE_RUNTIME):
        raise RuntimeError("G2-D8 运行时与 G2-D2 历史对照不符；须重新设计公平对照")
    initializations = cfg["quick"]["initializations"] if quick else cfg["training"]["initializations"]
    updates = (cfg["quick"]["updates_per_arm_initialization"] if quick
               else cfg["training"]["updates_per_arm_initialization"])
    transitions = initializations * updates * 18 * spec["episode_length"]
    report = {
        "status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_IDE",
        "quick": quick, "device": str(device), "arm": ARM,
        "comparator": COMPARATOR, "initializations": initializations,
        "updates_per_initialization": updates,
        "total_updates": initializations * updates,
        "episode_length": spec["episode_length"], "episodes_per_update": 18,
        "physical_transitions": transitions,
        "unique_weather_pairs": 1 if quick else d2.WEATHER_COUNT,
        "maximum_displacement_pixels": displacement,
        "turbulence_grid_pixels": simulation.turbulence_grid_size,
        "config_sha256": _file_sha256(_project_path(path)),
        "entry_sha256": _file_sha256(Path(__file__)),
        "d2_output_hashes": D2_HASHES,
        "d2_runtime_sha256": D2_RUNTIME_SHA256,
        "source_checkpoint_sha256": {str(key): value for key, value in source_hashes.items()},
        "comparator_final_checkpoint_sha256": {str(key): value
                                               for key, value in comparator_hashes.items()},
        "frozen_source_bundle_sha256": SOURCE_BUNDLE_SHA256,
        **cfg["boundary"],
    }
    return cfg, r3_cfg, parent, report, output, device, source_hashes, comparator_hashes


def run(path: str | Path = CONFIG, *, quick: bool = False,
        preflight_only: bool = False) -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cfg, r3_cfg, parent, report, output, device, sources, comparators = preflight(path, quick=quick)
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
            "gpu": torch.cuda.get_device_name(device), "git": safe_git_record(),
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "frozen_source_bundle_sha256": SOURCE_BUNDLE_SHA256,
        })
        base, _ = load_s1_config(_project_path(parent["environment_config"]))
        steps = report["episode_length"]
        base = replace(base, num_modes=21, batch_size=1, episode_length=steps)
        basis, _, _ = build_action_basis(
            base, ActionRepresentation("r5_zernike21", "zernike", 21), device)
        nominal, shifted = d2.g2.d1._profile_pairs(parent)
        profiles = {"nominal_clone": nominal, "hardware_shift": shifted}
        calibration = NominalCalibration()
        progress = Progress(output, device)
        progress.phase("G2-D8 CUDA 技术冒烟" if quick else "G2-D8 动作惩罚单因素训练",
                       report["total_updates"])
        loss_file = (output / "losses.jsonl").open("w", encoding="utf-8")
        checkpoint_hashes: dict[str, str] = {}
        results = []
        for member in range(report["initializations"]):
            torch.manual_seed(5_299_001 + member)
            torch.cuda.manual_seed_all(5_299_001 + member)
            source_path = (_project_path(r3_cfg["training_output"]) / "checkpoints"
                           / f"policy_{member}_02000.pt")
            if _file_sha256(source_path) != sources[member]:
                raise RuntimeError("G2-D8 暖启动检查点在预检后变化")
            source = torch.load(source_path, map_location=device, weights_only=True)
            if source["init"] != member or source["update"] != 2000:
                raise RuntimeError("G2-D8 暖启动身份不符")
            policy = ResidualGRUPolicy(parent["policy"]["hidden_size"],
                                       parent["policy"]["output_size"]).to(device)
            policy.load_state_dict(source["state_dict"])
            policy.train()
            scaled = d2.TrainingScale(policy, TRAIN_SCALE)
            optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["training"]["learning_rate"])
            running_loss = 0.0
            for update in range(1, report["updates_per_initialization"] + 1):
                condition, seed = schedule(update, quick=quick)
                optimizer.zero_grad(set_to_none=True)
                score, timing = _rollout_batch(
                    scaled, cfg, {"steps": steps}, device, seed, parent["families"],
                    profiles[condition], basis, base, calibration)
                loss = -score
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("G2-D8 非有限训练损失")
                backward_start = time.perf_counter()
                loss.backward()
                backward_seconds = time.perf_counter() - backward_start
                torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                               cfg["training"]["gradient_norm_limit"],
                                               error_if_nonfinite=True)
                optimizer.step()
                loss_value = float(loss.detach())
                running_loss += loss_value
                record = {
                    "arm": ARM, "training_scale": TRAIN_SCALE,
                    "deployment_scale": TRAIN_SCALE,
                    "action_weight": cfg["objective"]["action_weight"],
                    "initialization": member, "update": update,
                    "condition": condition, "weather_seed": seed,
                    "loss": loss_value, "average_loss": running_loss / update,
                    "policy_forward_seconds": timing["policy_forward"],
                    "env_step_seconds": timing["env_step"],
                    "interface_seconds": timing["interface"],
                    "objective_seconds": timing["objective"],
                    "backward_seconds": backward_seconds,
                }
                loss_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                loss_file.flush()
                progress.tick({"初始化": member + 1,
                               "初始化总数": report["initializations"],
                               "当前批次": update,
                               "平均损失": running_loss / update,
                               "物理条件序号": 1 if condition == "nominal_clone" else 2})
                interval = cfg["training"]["checkpoint_interval_updates"]
                if update % interval == 0 or update == report["updates_per_initialization"]:
                    name = f"{ARM}_policy_{member}_{update:05d}.pt"
                    checkpoint = output / "checkpoints" / name
                    torch.save({
                        "state_dict": policy.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "arm": ARM, "training_scale": TRAIN_SCALE,
                        "deployment_scale": TRAIN_SCALE,
                        "action_weight": cfg["objective"]["action_weight"],
                        "init": member, "update": update,
                        "source_checkpoint_sha256": sources[member],
                        "comparator_final_checkpoint_sha256": comparators[member],
                        "config_sha256": report["config_sha256"],
                    }, checkpoint)
                    checkpoint_hashes[name] = _file_sha256(checkpoint)
            results.append({"arm": ARM, "initialization": member,
                            "updates": report["updates_per_initialization"],
                            "mean_training_loss": running_loss / report["updates_per_initialization"]})
        write_json(output / "checkpoint_manifest.json", checkpoint_hashes)
        result = {
            "status": "QUICK_SMOKE_NO_SCIENTIFIC_CONCLUSION" if quick
                      else "DEVELOPMENT_TRAINING_COMPLETE_REQUIRES_AUDIT",
            "arm": ARM, "comparator": COMPARATOR,
            "results": results, "initializations": report["initializations"],
            "training_updates": report["total_updates"],
            "physical_transitions": report["physical_transitions"],
            "checkpoints": len(checkpoint_hashes),
            "elapsed_seconds": time.perf_counter() - started,
            "config_sha256": report["config_sha256"],
            "entry_sha256": report["entry_sha256"],
            "source_checkpoint_sha256": report["source_checkpoint_sha256"],
            "comparator_final_checkpoint_sha256": report["comparator_final_checkpoint_sha256"],
            "frozen_source_bundle_sha256": SOURCE_BUNDLE_SHA256,
            "d2_output_hashes": D2_HASHES,
            "material_passport": {"origin_skill": "academic-research-suite / experiment-agent",
                                  "origin_mode": "run", "verification_status": "UNVERIFIED"},
            **cfg["boundary"],
            "next_action": "停止并等待只读训练审计；训练损失不是科学性能，不能自动开发评价或独立确认",
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
                write_json(output / "failure.json", {
                    "traceback": traceback.format_exc(), "automatic_retry": False})
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
    parser.add_argument("--quick", action="store_true", help="16 帧 CUDA 技术冒烟，不形成科学结论")
    parser.add_argument("--preflight-only", action="store_true", help="只读预检，不创建输出")
    args = parser.parse_args()
    print(json.dumps(run(args.config, quick=args.quick,
                         preflight_only=args.preflight_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
