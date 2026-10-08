"""G2-D12-B：完整天气分折、因果输入的离线监督训练；不读取折外成绩选模型。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import prepare_s4_r5_g2_d12_causal_dataset as pairing
from src.rl.r5_physics_adapter import causal_policy_features
from src.runtime import resolve_device

CONFIG = "configs/experiments/s4_r5_g2_d12_candidate_scorer_v1.yaml"
PAIRING_ENTRY_SHA256 = "568d75df42ad963bcc2f465af50e3f783091ecd0d7effbab8a44cbc9cf3ccda5"
PAIRING_CONFIG_SHA256 = "3ae8f919ed5da0b69b0041ba90096378452ddc0c1ec35120be2428fa1e384c8f"
DATA_HASHES = {
    False: {"summary": "eb0de7553394d111fcdce8a653e8a87b2b45a431b9f78c514388d5f119eef66a",
        "inputs": "86f449de9f273120601e53b9594c5b54936256040e1a824cdd0f0071902670bd",
        "targets": "ef97f62b7fd8e9d1c5b48f9668143b5377309c1886142f37cb6518629bd3d651",
        "index": "de64c715c66d50fb94f2f01d7b9bbda8ffd070f3861ba69d2bdb02e00ca546f4",
        "progress": "7fc9c6244eb199c16db67bc3c2f24148959467cc8f8859e27b171baf22e16bd2"},
    True: {"summary": "9981ae41e9b2bd4e7dc7dbd072646f4ec6fd2aa2a2d77ac98a593fdfc1ca48a3",
        "inputs": "72542a1c945adb472778cfe93df780da53c77f987ecaaf02fdf88de4f42b0db5",
        "targets": "f6d237e7d9ddb5b22598e0907bf805c8576f75e2ccff7bc43922179f0d7c34dc",
        "index": "12e031b33cae0f84ffa6d2afbd1fd8026a14f8f13f015a450f4b18253658197c",
        "progress": "cf56655965bfecef9b796805b6d40e04e135bfda07de53026c40b9c9fbde0294"},
}
FEATURE_COLUMNS = (*range(75), 78)  # 去掉三个绝对时间字段，保留功率有效标志。


def contract(cfg: dict) -> None:
    expected = {"stage": "S4-D2-R5-G2-D12-B",
        "purpose": "offline_causal_candidate_scorer_training_only",
        "runtime": {"device": "cuda", "formal_owner": "user_ide", "automatic_retry": False},
        "dataset_config": pairing.CONFIG,
        "split": {"unit": "complete_weather", "folds": 4, "held_out_weather_per_fold": 2},
        "models": ["current", "history"],
        "training": {"seeds": [7564000, 7564001, 7564002], "updates": 1000, "batch_size": 128,
            "learning_rate": .001, "target_scale": 10000., "gradient_clip": 1.,
            "checkpoint_interval": 250, "display_interval": 25, "early_stopping": False,
            "checkpoint_selection": "last"},
        "quick": {"seeds": [7564999], "updates": 2, "batch_size": 8,
                  "checkpoint_interval": 2, "display_interval": 1},
        "output_directory": "outputs/s4_r5_g2_d12_candidate_scorer_v1",
        "quick_directory": "outputs/s4_r5_g2_d12_candidate_scorer_v1_quick",
        "boundary": {"confirmation_access": False, "real_slm_actions": False,
            "new_environment_transitions": 0, "held_out_evaluation": False,
            "automatic_retry": False, "gate_reclassification": False}}
    if cfg != expected:
        raise ValueError("G2-D12-B 冻结训练合同变化")


def verify_dataset(quick: bool) -> Path:
    src = pairing.source
    if (src._file_sha256(Path(pairing.__file__)) != PAIRING_ENTRY_SHA256
            or src._file_sha256(src._project_path(pairing.CONFIG)) != PAIRING_CONFIG_SHA256):
        raise RuntimeError("G2-D12-A 冻结代码或配置变化")
    cfg = src._load_yaml(src._project_path(pairing.CONFIG)); pairing.contract(cfg)
    src.verify_sources(); src.verify_streams()
    root = src._project_path(cfg["quick_directory" if quick else "output_directory"])
    if (root / "failure.json").exists():
        raise RuntimeError("配对数据有失败标记")
    success = json.loads((root / "SUCCESS.json").read_text(encoding="utf-8"))
    for key, digest in DATA_HASHES[quick].items():
        suffix = ".pt" if key in ("inputs", "targets") else ".jsonl" if key in ("index", "progress") else ".json"
        if src._file_sha256(root / (key + suffix)) != digest or success.get(key + "_sha256") != digest:
            raise RuntimeError(f"配对数据校验失败: {key}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if (summary["quick"] != quick or summary["device"] != "cuda"
            or not summary["all_source_hashes_and_actions_exact"] or summary["training_updates"] != 0
            or summary["confirmation_access"] or summary["real_slm_actions"]
            or summary["entry_sha256"] != PAIRING_ENTRY_SHA256
            or summary["config_sha256"] != PAIRING_CONFIG_SHA256):
        raise RuntimeError("配对数据审计边界不符")
    return root


def weather_splits(rows: list[dict], *, quick: bool) -> list[dict]:
    """元数据仅用于拆分和记录；模型不接收此列表。"""
    if not rows or [r["sample_index"] for r in rows] != list(range(len(rows))):
        raise ValueError("样本索引不连续")
    expected_seeds = pairing.source.stream_manifest(quick)["weather_bases"]
    expected_keys = {(c, seed, step, member, family, slot)
        for c in pairing.source.CONDITIONS for seed in expected_seeds
        for step in ([5] if quick else [25, 75, 150])
        for member in range(1 if quick else 3)
        for family in pairing.source.FAMILIES for slot in range(6)}
    seen = set()
    for row in rows:
        key = tuple(row[k] for k in pairing.KEY_FIELDS)
        expected_fold = None if quick else expected_seeds.index(row["weather_seed"]) // 2
        if (key not in expected_keys or key in seen or row["weather_fold"] != expected_fold
                or row["candidate_order"] != list(pairing.source.CANDIDATES)
                or row["metadata_not_model_input"] is not True):
            raise ValueError("完整天气网格、候选顺序或折分不符")
        seen.add(key)
    if seen != expected_keys:
        raise ValueError("完整天气网格缺失")
    if quick:
        return [{"fold": -1, "train_indices": list(range(len(rows))), "held_out_indices": [],
                 "train_weather": expected_seeds, "held_out_weather": [], "technical_only": True}]
    result = []
    for fold in range(4):
        train = [i for i, r in enumerate(rows) if r["weather_fold"] != fold]
        held = [i for i, r in enumerate(rows) if r["weather_fold"] == fold]
        train_weather = sorted({rows[i]["weather_seed"] for i in train})
        held_weather = sorted({rows[i]["weather_seed"] for i in held})
        if (len(train) != 1944 or len(held) != 648 or len(train_weather) != 6 or len(held_weather) != 2
                or set(train_weather) & set(held_weather)):
            raise ValueError("天气泄漏或折预算错误")
        result.append({"fold": fold, "train_indices": train, "held_out_indices": held,
                       "train_weather": train_weather, "held_out_weather": held_weather,
                       "technical_only": False})
    return result


def features(inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if set(inputs) != {"history", "valid", "command"}:
        raise ValueError("模型输入只允许 history/valid/command")
    h, valid, command = (inputs[k] for k in ("history", "valid", "command"))
    if (command.shape != (len(h), 11) or command.dtype != h.dtype or command.device != h.device
            or not bool(torch.isfinite(command).all())):
        raise ValueError("命令输入不符")
    return causal_policy_features(h, valid)[..., list(FEATURE_COLUMNS)], valid, command


def fit_normalizer(x: torch.Tensor, valid: torch.Tensor, command: torch.Tensor,
                   train_indices: torch.Tensor) -> dict[str, torch.Tensor]:
    # 每折只看训练天气，两个模型共享同一份标准化参数；无标签参与。
    frames = x[train_indices][valid[train_indices]]
    cmd = command[train_indices]
    return {"mean": frames.mean(0), "std": frames.std(0, correction=0).clamp_min(1e-6),
            "command_mean": cmd.mean(0), "command_std": cmd.std(0, correction=0).clamp_min(1e-6)}


def normalize(x: torch.Tensor, valid: torch.Tensor, command: torch.Tensor,
              normalizer: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = (x - normalizer["mean"]) / normalizer["std"]
    return normalized * valid[..., None], (command - normalizer["command_mean"]) / normalizer["command_std"]


class CandidateScorer(nn.Module):
    """输出23个非原动作候选的缩放功率差；原动作分数固定为0。"""

    def __init__(self, kind: str):
        super().__init__()
        if kind not in ("current", "history"):
            raise ValueError("未知模型")
        self.kind = kind
        self.recurrent = nn.GRU(76, 32, batch_first=True) if kind == "history" else None
        self.head = nn.Sequential(nn.Linear(43 if kind == "history" else 87, 64), nn.Tanh(),
                                  nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 23))

    def forward(self, x: torch.Tensor, valid: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        # x 只含白名单特征。有效掩码不可让未来帧进入序列。
        encoded = x[:, -1] if self.recurrent is None else self.recurrent(x * valid[..., None])[0][:, -1]
        return self.head(torch.cat((encoded, command), -1))


def candidate_choice(predicted: torch.Tensor) -> torch.Tensor:
    """固定选择规则，不能接收未来收益/安全标签；相同分数优先原动作。"""
    if predicted.ndim != 2 or predicted.shape[1] != 23 or not bool(torch.isfinite(predicted).all()):
        raise ValueError("候选预测不符")
    scores = torch.cat((predicted.new_zeros(len(predicted), 2), predicted), -1)
    return scores.argmax(-1)


def training_subset(x: torch.Tensor, valid: torch.Tensor, command: torch.Tensor,
                    labels: torch.Tensor, train: torch.Tensor, scale: float) -> tuple:
    # 将训练天气张量独立切出；训练循环没有折外索引或折外标签。
    return x[train], valid[train], command[train], (labels[train, 2:] * scale).float()


def load_dataset(root: Path, device: torch.device, *, quick: bool) -> tuple:
    inputs = torch.load(root / "inputs.pt", map_location=device, weights_only=True)
    targets = torch.load(root / "targets.pt", map_location=device, weights_only=True)
    rows = [json.loads(line) for line in (root / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = weather_splits(rows, quick=quick)
    n = 36 if quick else 2592
    if (inputs["history"].shape != (n, 8, 79) or inputs["history"].dtype != torch.float32
            or set(targets) != {"power_delta", "measured_power_delta", "safety_maxima", "safety_pass"}):
        raise ValueError("配对张量格式错误")
    for key, shape, dtype in (("power_delta", (n, 25), torch.float64),
            ("measured_power_delta", (n, 25), torch.float64),
            ("safety_maxima", (n, 25, 3), torch.float64), ("safety_pass", (n, 25), torch.bool)):
        if targets[key].shape != shape or targets[key].dtype != dtype or not bool(torch.isfinite(targets[key]).all()):
            raise ValueError(f"配对标签格式错误: {key}")
    for step in sorted({r["probe_step"] for r in rows}):
        ids = torch.tensor([i for i, r in enumerate(rows) if r["probe_step"] == step], device=device)
        pairing.causal_inputs(*(inputs[k][ids] for k in ("history", "valid", "command")), step=step)
    x, valid, command = features(inputs)
    if bool(targets["power_delta"][:, :2].any() | targets["measured_power_delta"][:, :2].any()):
        raise ValueError("原动作或等价动作的差值不为零")
    return x, valid, command, targets["power_delta"], splits


def preflight(path: str | Path = CONFIG, *, quick: bool = False) -> tuple:
    src = pairing.source
    cfg = src._load_yaml(src._project_path(path)); contract(cfg)
    root = verify_dataset(quick)
    output = src._project_path(cfg["quick_directory" if quick else "output_directory"])
    if output.exists():
        raise FileExistsError(f"保留已有 G2-D12-B 输出，禁止覆盖: {output}")
    device = resolve_device("cuda")
    data = load_dataset(root, device, quick=quick)
    settings = dict(cfg["training"])
    if quick:
        settings.update(cfg["quick"])
    count = len(data[-1]) * len(cfg["models"]) * len(settings["seeds"])
    report = {"status": "READY_FOR_QUICK_SMOKE" if quick else "READY_FOR_USER_IDE",
        "quick": quick, "device": str(device), "model_fits": count,
        "total_updates": count * settings["updates"], "updates_per_model": settings["updates"],
        "batch_size": settings["batch_size"], "dataset_hashes": DATA_HASHES[quick],
        "entry_sha256": src._file_sha256(Path(__file__)),
        "config_sha256": src._file_sha256(src._project_path(path)),
        "samples": len(data[0]), "held_out_samples_per_fold": 0 if quick else 648,
        "train_samples_per_fold": 36 if quick else 1944, **cfg["boundary"]}
    return cfg, settings, output, device, data, report


def execute(cfg: dict, settings: dict, output: Path, device: torch.device, data: tuple, report: dict) -> dict:
    src = pairing.source
    x, valid, command, labels, splits = data
    src.write_json(output / "splits.json", splits)
    started = time.perf_counter(); completed = 0; fits = []; checkpoint_hashes = {}
    with (output / "progress.jsonl").open("x", encoding="utf-8") as log:
        for split in splits:
            train = torch.tensor(split["train_indices"], device=device)
            # 固定方向基线只能由本折训练天气选择，供下一阶段配对评价。
            constant_choice = int(labels[train].mean(0).argmax())
            normalizer = fit_normalizer(x, valid, command, train)
            # 只标准化训练子集；训练结束不计算折外预测或科学指标。
            tx, tv, tc, ty = training_subset(x, valid, command, labels, train, settings["target_scale"])
            tx, tc = normalize(tx, tv, tc, normalizer)
            for seed in settings["seeds"]:
                for kind in cfg["models"]:
                    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                    generator = torch.Generator(device=device).manual_seed(seed + 100 * (split["fold"] + 1))
                    model = CandidateScorer(kind).to(device).train()
                    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
                    prefix = f"fold_{split['fold']}_{kind}_seed_{seed}"
                    running_loss = 0.; fit_start = time.perf_counter()
                    for update in range(1, settings["updates"] + 1):
                        batch = torch.randint(len(tx), (settings["batch_size"],), generator=generator, device=device)
                        optimizer.zero_grad(set_to_none=True)
                        loss = (model(tx[batch], tv[batch], tc[batch]) - ty[batch]).square().mean()
                        if not bool(torch.isfinite(loss)):
                            raise RuntimeError("训练损失非有限，保留日志并停止")
                        loss.backward()
                        norm = nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
                        optimizer.step(); completed += 1
                        value = float(loss.detach()); running_loss += value
                        elapsed = time.perf_counter() - started
                        row = {"completed_updates": completed, "total_updates": report["total_updates"],
                            "fold": split["fold"], "kind": kind, "seed": seed, "update": update,
                            "updates_per_model": settings["updates"], "batch_size": settings["batch_size"],
                            "loss": value, "average_loss": running_loss / update,
                            "gradient_norm_before_clip": float(norm), "elapsed_seconds": elapsed,
                            "seconds_per_update": elapsed / completed,
                            "eta_seconds": elapsed / completed * (report["total_updates"] - completed),
                            "cuda_allocated_gb": torch.cuda.memory_allocated(device) / 1e9}
                        log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"); log.flush()
                        if update % settings["display_interval"] == 0 or update == 1:
                            print(f"G2-D12-B {completed}/{report['total_updates']} | {kind} 折{split['fold']} "
                                  f"种子{seed} {update}/{settings['updates']} 批次{settings['batch_size']} "
                                  f"平均损失{row['average_loss']:.6g} "
                                  f"速度{row['seconds_per_update']:.3f}秒/更新 "
                                  f"剩余{row['eta_seconds']/60:.1f}分 显存{row['cuda_allocated_gb']:.3f}GB", flush=True)
                        if update % settings["checkpoint_interval"] == 0 or update == settings["updates"]:
                            name = f"{prefix}_{update:05d}.pt"
                            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                "normalizer": {k: v.detach().cpu() for k, v in normalizer.items()},
                                "kind": kind, "fold": split["fold"], "seed": seed, "update": update,
                                "target_scale": settings["target_scale"], "feature_columns": list(FEATURE_COLUMNS),
                                "candidate_order": list(src.CANDIDATES), "dataset_hashes": DATA_HASHES[report["quick"]],
                                "train_weather": split["train_weather"], "held_out_weather": split["held_out_weather"],
                                "training_only_constant_candidate": constant_choice,
                                "parameter_count": sum(p.numel() for p in model.parameters()),
                                "last_loss": value, "held_out_evaluation": False}, output / "checkpoints" / name)
                            checkpoint_hashes[name] = src._file_sha256(output / "checkpoints" / name)
                    fits.append({"kind": kind, "fold": split["fold"], "seed": seed,
                        "updates": settings["updates"], "last_loss": value, "mean_loss": running_loss / update,
                        "elapsed_seconds": time.perf_counter() - fit_start,
                        "parameter_count": sum(p.numel() for p in model.parameters()),
                        "training_only_constant_candidate": constant_choice,
                        "final_checkpoint": name})
    result = {**report, "status": "QUICK_TECHNICAL_SMOKE_ONLY" if report["quick"] else "TRAINING_COMPLETE_REQUIRES_READ_ONLY_AUDIT",
        "completed_updates": completed, "fits": fits, "checkpoint_hashes": checkpoint_hashes,
        "elapsed_seconds": time.perf_counter() - started, "analysis": {},
        "material_passport": {"origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run", "origin_date": time.strftime("%Y-%m-%d"),
            "verification_status": "UNVERIFIED", "version_label": "g2_d12_scorer_training_v1"},
        "next_action": "停止等待只读训练审计，之后另行准备完整天气折外评价；不自动训练或确认"}
    src.write_json(output / "summary.json", result)
    src.write_json(output / "SUCCESS.json", {f"{key}_sha256": src._file_sha256(output / file)
        for key, file in (("summary", "summary.json"), ("progress", "progress.jsonl"),
                          ("splits", "splits.json"), ("config", "config.json"))})
    return result


def run(path: str | Path = CONFIG, *, quick: bool = False, preflight_only: bool = False) -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cfg, settings, output, device, data, report = preflight(path, quick=quick)
    if preflight_only:
        return report
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    output.mkdir(parents=True, exist_ok=False); (output / "checkpoints").mkdir()
    src = pairing.source
    src.write_json(output / "config.json", cfg); src.write_json(output / "preflight.json", report)
    src.write_json(output / "runtime.json", {"python": platform.python_version(), "torch": str(torch.__version__),
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
        "platform": platform.platform(), "deterministic_algorithms": True, "tf32": False,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]})
    try:
        return execute(cfg, settings, output, device, data, report)
    except BaseException:
        src.write_json(output / "failure.json", {"error": traceback.format_exc(), "automatic_retry": False})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--quick", action="store_true", help="仅4次更新技术冒烟，无折外指标")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, quick=args.quick, preflight_only=args.preflight_only),
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
