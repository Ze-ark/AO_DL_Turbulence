"""R4-1C断电恢复：重演最后一个成员，核对原日志后补齐正式产物。"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback
from uuid import uuid4

import torch

from src.rl.r4_baseline_selection import read
from src.rl.r4_closed_loop import model_ensembles
from src.rl.r4_closed_loop_repair import (
    ARMS, SOURCES, budget, load_training_stores, mix_batches, prediction_gate,
    settings, split_seeds, validate_config,
)
from src.rl.r4_control import NominalCalibration
from src.rl.r4_delta_learning import paired_losses, pulse_rollout
from src.rl.r4_dynamics import normalized_errors, rollout
from src.rl.r4_dynamics_experiment import Progress, evaluate, safe_git_record, verify_hashes
from src.rl.r4_interface_smoke import write_json
from src.rl.s4_training import _file_sha256, _load_yaml, _project_path, _relative
from src.runtime import resolve_device


RECOVERY_SOURCES = "configs/experiments/s4_r4_closed_loop_repair_resume_v2_sources.json"
LAST_ARM = "mixed_data"
LAST_MEMBER = 2


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        lines = stream.readlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError(f"empty or malformed JSONL: {path}")
    return [json.loads(line) for line in lines]


def validate_prefix(losses: list[dict], developments: list[dict], updates: int,
                    interval: int) -> int:
    """只接受五个完整成员以及第六成员的连续前缀。"""
    if len(losses) <= 5 * updates or len(losses) >= 6 * updates:
        raise ValueError("expected five complete members and one incomplete member")
    prefix = len(losses) - 5 * updates
    if prefix < interval or prefix % interval == 0:
        # 断点必须位于一个可独立核对的检查点之后。
        raise ValueError("interruption must follow a completed checkpoint")
    for index, row in enumerate(losses):
        arm = ARMS[index // (3 * updates)]
        member = (index // updates) % 3
        update = index % updates + 1
        if (row["arm"], row["member"], row["update"]) != (arm, member, update):
            raise ValueError(f"noncontiguous training log at row {index + 1}")
        if not all(torch.isfinite(torch.tensor(float(row[key]))) for key in
                   ("loss", "average_loss", "trajectory", "absolute", "delta", "gradient_norm")):
            raise ValueError(f"nonfinite training log at row {index + 1}")
    expected = [(ARMS[i // 3], i % 3, step) for i in range(6)
                for step in range(interval, (updates if i < 5 else prefix) + 1, interval)]
    actual = [(r["arm"], r["member"], r["update"]) for r in developments]
    if actual != expected:
        raise ValueError("development history does not match saved checkpoints")
    return prefix


def require_same_state(actual: dict, reference: dict, label: str) -> None:
    if actual.keys() != reference.keys() or any(
            not torch.equal(actual[key].cpu(), reference[key].cpu()) for key in actual):
        raise RuntimeError(f"deterministic replay state differs: {label}")


def require_same_generator_state(actual: torch.Tensor, reference: torch.Tensor,
                                 label: str) -> None:
    """检查点可加载到CUDA；生成器状态比较前统一转回CPU。"""
    if not torch.equal(actual.cpu(), reference.cpu()):
        raise RuntimeError(f"deterministic replay sampler differs: {label}")


def require_same_record(actual: dict, reference: dict, label: str) -> None:
    if actual.keys() != reference.keys() or actual != reference:
        raise RuntimeError(f"deterministic replay log differs: {label}")


def existing_file_hashes(output: Path) -> dict[str, str]:
    return {_relative(path): _file_sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}


def prior_attempt_updates(output: Path, current: Path) -> dict[str, int]:
    """从保留的进度日志统计先前尝试已记录的优化器更新。"""
    result = {}
    directory = output / "recovery_attempts"
    if not directory.is_dir():
        return result
    for attempt in sorted(directory.iterdir()):
        if attempt == current or not attempt.is_dir() or not (attempt / "progress.jsonl").is_file():
            continue
        rows = read_jsonl(attempt / "progress.jsonl")
        updates = [int(row["completed"]) for row in rows
                   if str(row.get("phase", "")).startswith("R4-1C恢复：重演")]
        result[_relative(attempt)] = max(updates, default=0)
    return result


def preflight(path: str | Path) -> tuple:
    config_path = _project_path(path)
    cfg = _load_yaml(config_path)
    validate_config(cfg)
    c = settings(cfg, False)
    output = _project_path(cfg["output_directory"])
    if not output.is_dir():
        raise FileNotFoundError(f"interrupted output missing: {output}")
    if any((output / name).exists() for name in (
            "summary.json", "SUCCESS.json", "artifact_manifest.json", "prediction_metrics.json",
            "recovery_provenance.json", "checkpoints/mixed_data_2_02000.pt")):
        raise FileExistsError("repair already finalized or partially committed; preserve output for audit")
    verify_hashes(read(_project_path(SOURCES)))
    verify_hashes(read(_project_path(RECOVERY_SOURCES)))
    recorded = read(output / "preflight.json")
    if (recorded["status"] != "READY_FOR_USER_IDE" or recorded["phase"] != "train"
            or recorded["quick"] or recorded["budget"] != budget(c, "train")
            or read(output / "config.json") != cfg):
        raise ValueError("interrupted run does not match the frozen formal configuration")
    verify_hashes(recorded["frozen_files"])
    runtime = read(output / "runtime.json")
    device = resolve_device("cuda")
    if (runtime["torch"] != str(torch.__version__) or runtime["cuda"] != torch.version.cuda
            or runtime["gpu"] != torch.cuda.get_device_name(device)):
        raise RuntimeError("CUDA environment changed; exact optimizer replay cannot be asserted")
    manifest = read(output / "data_manifest.json")
    records = manifest["records"]
    if manifest["status"] != "COLLECTION_COMPLETE" or len(records) != 72:
        raise ValueError("formal collection is incomplete")
    if records != read_jsonl(output / "collection_manifest.jsonl"):
        raise ValueError("collected trajectory manifest differs from append-only record")
    expected_seeds = split_seeds(cfg)
    for split in ("train", "development"):
        group = [r for r in records if r["split"] == split]
        seeds = [seed for r in group for seed in r["weather_seeds"]]
        if (len(group) != 36 or len(seeds) != 576 or set(seeds) != expected_seeds[split]
                or any(seeds.count(seed) != 6 for seed in expected_seeds[split])
                or any(r["controller"] != "original_gru_mpc" for r in group)):
            raise ValueError(f"incomplete or overlapping {split} weather")
    for row in records:
        verify_hashes({row["file"]: row["sha256"], row["audit_file"]: row["audit_sha256"]})
    losses = read_jsonl(output / "loss_history.jsonl")
    developments = read_jsonl(output / "development_history.jsonl")
    prefix = validate_prefix(losses, developments, c["updates"], c["interval"])
    dev_lookup = {(r["arm"], r["member"], r["update"]): r for r in developments}
    parent = _load_yaml(_project_path(cfg["parent"]))
    for i in range(6):
        arm, member = ARMS[i // 3], i % 3
        initial = output / "checkpoints" / f"{arm}_{member}_initial.pt"
        if not initial.is_file():
            raise FileNotFoundError(initial)
        initial_data = torch.load(initial, map_location="cpu", weights_only=True)
        if initial_data["seed"] != parent["model"]["member_seeds"][member]:
            raise ValueError(f"initial model seed mismatch: {arm}/{member}")
        last = c["updates"] if i < 5 else prefix
        for update in range(c["interval"], last + 1, c["interval"]):
            checkpoint = output / "checkpoints" / f"{arm}_{member}_{update:05d}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if ((saved["arm"], saved["member"], saved["seed"], saved["update"], saved["selection"])
                    != (arm, member, parent["model"]["member_seeds"][member], update, "final_only")
                    or saved["metrics"]["windows"] != 5760
                    or any(not bool(torch.isfinite(value).all()) for value in saved["state_dict"].values())):
                raise ValueError(f"checkpoint metadata or weights invalid: {checkpoint}")
            require_same_record(dict(arm=arm, member=member, update=update, **saved["metrics"]),
                                dev_lookup[(arm, member, update)], f"saved development {arm}/{member}/{update}")
    if len(list((output / "checkpoints").glob("*.pt"))) != 6 + 5 * 4 + prefix // c["interval"]:
        raise ValueError("unexpected checkpoint count")
    for member in range(3):
        old = torch.load(output / "checkpoints" / f"old_data_{member}_initial.pt",
                         map_location="cpu", weights_only=True)
        mixed = torch.load(output / "checkpoints" / f"mixed_data_{member}_initial.pt",
                           map_location="cpu", weights_only=True)
        require_same_state(old["state_dict"], mixed["state_dict"], f"paired initial {member}")
    last_progress = read_jsonl(output / "progress.jsonl")[-1]
    if (last_progress.get("总完成批次") != len(losses)
            or last_progress.get("当前批次") != prefix):
        raise ValueError("progress log disagrees with training loss prefix")
    snapshots = existing_file_hashes(output)
    report = dict(status="READY_FOR_USER_IDE_RECOVERY", interruption_update=prefix,
                  replay_updates=c["updates"], missing_updates=c["updates"] - prefix,
                  preserved_files=len(snapshots), collected_batches=len(records),
                  existing_training_updates=len(losses), device=str(device),
                  original_output=_relative(output), rl_authorized=False,
                  automatic_retry=False)
    return cfg, c, parent, output, records, losses, developments, snapshots, report


def replay_member(cfg: dict, c: dict, parent: dict, output: Path, records: list[dict],
                  losses: list[dict], developments: list[dict], prefix: int,
                  device: torch.device, progress: Progress | None, limit: int | None = None) -> tuple:
    """从原始初始化重建Adam；先逐批核对日志，再计算缺失的批次。"""
    stores, pairs = load_training_stores(records, False)
    originals = model_ensembles(device)["gru_mpc"]
    seed = parent["model"]["member_seeds"][LAST_MEMBER]
    model = deepcopy(originals[LAST_MEMBER])
    model.gru.requires_grad_(True)
    model.head.requires_grad_(True)
    model.linear.requires_grad_(False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=c["learning_rate"])
    initial = torch.load(output / "checkpoints" / "mixed_data_2_initial.pt",
                         map_location="cpu", weights_only=True)
    if initial["seed"] != seed:
        raise RuntimeError("initial checkpoint seed mismatch")
    require_same_state(model.state_dict(), initial["state_dict"], "initial mixed member 2")
    pools = {key: stores[key].bootstrap_pool(seed) for key in ("old_train", "train")}
    pp = pairs.pool(seed)
    og, ng, pg = [torch.Generator().manual_seed(seed + offset) for offset in (100, 300, 200)]
    scale = torch.tensor(read(_project_path("outputs/s4_r4_delta_supervision_v1/delta_scale.json"))["value"],
                         device=device)
    cal = NominalCalibration(**parent["nominal_calibration"])
    evalcfg = deepcopy(parent)
    evalcfg["training"]["evaluation_starts"] = c["evaluation_starts"]
    end = limit if limit is not None else c["updates"]
    if progress is not None:
        progress.phase("R4-1C恢复：重演并核对模型3/3", end)
    prefix_rows = losses[5 * c["updates"]:]
    known_developments = {(r["arm"], r["member"], r["update"]): r for r in developments}
    tail = []
    running = 0.0
    final_checkpoint = None
    for update in range(1, end + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        old = stores["old_train"].sample(pools["old_train"], c["trajectory_batch"], 8, og, device)
        new = stores["train"].sample(pools["train"], c["trajectory_batch"] // 2, 8, ng, device)
        batch = mix_batches(old, new, LAST_ARM)
        pair = pairs.sample(pp, c["pair_batch"], pg, device)
        pred = rollout(model, batch["history"], batch["valid"], batch["commands"], batch["corrections"], cal)
        trajectory = normalized_errors(pred, batch, model.linear.y_scale, parent["model"]["horizons"]).mean()
        pred_pair = pulse_rollout(model, pair["history"], pair["valid"], pair["correction"],
                                  parent["collector_anchor"], cal)
        pred_zero = pulse_rollout(model, pair["history"], pair["valid"],
                                  torch.zeros_like(pair["correction"]), parent["collector_anchor"], cal)
        absolute, difference = paired_losses(pred_pair, pred_zero, pair["power"], pair["zero_power"],
                                             model.linear.y_scale[21], scale)
        loss = trajectory + absolute + difference
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite replay training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, c["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        value = float(loss.detach())
        running += value
        row = dict(arm=LAST_ARM, member=LAST_MEMBER, seed=seed, update=update,
                   loss=value, average_loss=running / update, trajectory=float(trajectory.detach()),
                   absolute=float(absolute.detach()), delta=float(difference.detach()),
                   gradient_norm=float(norm))
        if update <= prefix:
            require_same_record(row, prefix_rows[update - 1], f"batch {update}")
        else:
            tail.append(row)
        if progress is not None:
            progress.tick({"平均损失": running / update, "当前批次": update,
                           "总完成批次": 5 * c["updates"] + update,
                           "总批次": 6 * c["updates"], "已校验重演": min(update, prefix),
                           "新增批次": max(0, update - prefix)})
        if update % c["interval"] == 0:
            metrics = evaluate(model, stores["old_development"], evalcfg, device,
                               model.linear.y_scale)
            ck = dict(state_dict=model.state_dict(), arm=LAST_ARM, member=LAST_MEMBER,
                      seed=seed, update=update, metrics=metrics, selection="final_only",
                      delta_scale=float(scale), old_generator=og.get_state(),
                      new_generator=ng.get_state(), pair_generator=pg.get_state())
            if update <= prefix:
                saved = torch.load(output / "checkpoints" / f"mixed_data_2_{update:05d}.pt",
                                   map_location="cpu", weights_only=True)
                require_same_state(ck["state_dict"], saved["state_dict"], f"checkpoint {update}")
                for key in ("old_generator", "new_generator", "pair_generator"):
                    require_same_generator_state(ck[key], saved[key], f"checkpoint {update}: {key}")
                require_same_record(metrics, saved["metrics"], f"old development {update}")
                require_same_record(dict(arm=LAST_ARM, member=LAST_MEMBER, update=update, **metrics),
                                    known_developments[(LAST_ARM, LAST_MEMBER, update)],
                                    f"development log {update}")
            else:
                final_checkpoint = ck
    if limit is not None:
        return None, tail, None, None, min(end, prefix)
    if final_checkpoint is None or len(tail) != c["updates"] - prefix:
        raise RuntimeError("replay did not complete the missing update range")
    require_same_state(model.linear.state_dict(), originals[LAST_MEMBER].linear.state_dict(),
                       "frozen normalization")
    paired = torch.load(output / "checkpoints" / "old_data_2_02000.pt",
                        map_location="cpu", weights_only=True)
    for key in ("old_generator", "new_generator", "pair_generator"):
        require_same_generator_state(final_checkpoint[key], paired[key], f"equal-update {key}")
    return model, tail, final_checkpoint, stores, prefix


def finalize(cfg: dict, c: dict, parent: dict, output: Path, records: list[dict],
             snapshots: dict[str, str], attempt: Path, model: torch.nn.Module,
             tail: list[dict], checkpoint: dict, stores: dict) -> dict:
    device = resolve_device("cuda")
    evalcfg = deepcopy(parent)
    evalcfg["training"]["evaluation_starts"] = c["evaluation_starts"]
    original = model_ensembles(device)["gru_mpc"]
    metrics: dict[str, list] = {arm: [] for arm in ARMS}
    progress = Progress(attempt, device)
    progress.phase("R4-1C恢复：六个最终模型独立预测", 6)
    try:
        for arm in ARMS:
            for member in range(3):
                current = model if (arm, member) == (LAST_ARM, LAST_MEMBER) else deepcopy(original[member])
                if current is not model:
                    saved = torch.load(output / "checkpoints" / f"{arm}_{member}_02000.pt",
                                       map_location=device, weights_only=True)
                    current.load_state_dict(saved["state_dict"])
                current.eval().requires_grad_(False)
                result = evaluate(current, stores["development"], evalcfg, device,
                                  current.linear.y_scale)
                if result["windows"] != 2880:
                    raise RuntimeError("incomplete new development evaluation")
                metrics[arm].append(result["per_horizon"])
                progress.tick({"当前批次": member + 1, "总批次": 6,
                               "平均损失": result["score"], "总完成批次": 12000})
    finally:
        progress.close()
    gate = prediction_gate(metrics)
    verify_hashes(snapshots)
    if any((output / name).exists() for name in (
            "summary.json", "SUCCESS.json", "artifact_manifest.json", "prediction_metrics.json",
            "recovery_provenance.json", "checkpoints/mixed_data_2_02000.pt")):
        raise FileExistsError("recovery target appeared during replay")
    planning = 0
    for record in records:
        audit = torch.load(_project_path(record["audit_file"]), map_location="cpu", weights_only=True)
        if len(audit["rows"]) != c["steps"]:
            raise ValueError("incomplete recorded physical episode")
        planning += sum(row["model_forward_samples"] for row in audit["rows"])
    counters = dict(
        physical_transitions=len(records) * c["batch"] * c["steps"],
        planning_forward_samples=planning,
        training_updates=6 * c["updates"],
        training_forward_samples=6 * c["updates"] * 8 *
            (c["trajectory_batch"] + 2 * c["pair_batch"]),
        prediction_forward_samples=6 * 8 * len(c["evaluation_starts"]) *
            (4 * stores["old_development"].count + stores["development"].count),
    )
    planned = budget(c, "train")
    if (planning > planned["max_planning_forward_samples"]
            or any(counters[key] != planned[key] for key in (
                "physical_transitions", "training_updates", "training_forward_samples",
                "prediction_forward_samples"))):
        raise RuntimeError("restored budget mismatch")
    with (attempt / "tail_loss_history.jsonl").open("x", encoding="utf-8") as stream:
        for row in tail:
            stream.write(json.dumps(row) + "\n")
    write_json(attempt / "final_old_development.json", dict(
        arm=LAST_ARM, member=LAST_MEMBER, update=2000, **checkpoint["metrics"]))
    write_json(attempt / "replay_verification.json", dict(
        status="LOG_AND_CHECKPOINT_REPLAY_VERIFIED", compared_training_updates=2000 - len(tail),
        missing_updates=len(tail), exact_logged_losses=True,
        exact_checkpoints=list(range(c["interval"], 2000 - len(tail) + 1, c["interval"])),
        exact_sampler_state=True,
        frozen_normalization=True, equal_update_sampler_state=True,
        original_file_hashes=snapshots,
        recovery_source_manifest_sha256=_file_sha256(_project_path(RECOVERY_SOURCES)),
        original_source_manifest_sha256=_file_sha256(_project_path(SOURCES))))
    # 到这里才写入原正式目录；原始日志、轨迹、检查点逐字节保留。
    torch.save(checkpoint, output / "checkpoints" / "mixed_data_2_02000.pt")
    write_json(output / "prediction_metrics.json", metrics)
    previous = prior_attempt_updates(output, attempt)
    earlier_updates = sum(previous.values())
    replayed_updates = 2000 - len(tail)
    samples_per_update = 8 * (c["trajectory_batch"] + 2 * c["pair_batch"])
    write_json(output / "recovery_provenance.json", dict(
        reason="user_reported_shutdown", method="replay_from_initial_then_verify_and_complete",
        original_progress_batches=12000 - len(tail), recovered_batches=len(tail),
        effective_training_updates=counters["training_updates"],
        replayed_training_updates=replayed_updates,
        prior_attempt_recorded_updates=previous,
        recorded_training_updates_including_replay=counters["training_updates"] +
            replayed_updates + earlier_updates,
        replayed_training_forward_samples=replayed_updates * samples_per_update,
        prior_attempt_recorded_training_forward_samples=earlier_updates * samples_per_update,
        diagnostic_smoke_updates_not_included=True,
        repeated_prediction_forward_samples=(2000 - len(tail)) // c["interval"] *
            stores["old_development"].count * len(c["evaluation_starts"]) * 8 +
            5 * stores["development"].count * len(c["evaluation_starts"]) * 8,
        original_loss_history_unchanged=True, attempt=_relative(attempt),
        replay_verification=_relative(attempt / "replay_verification.json"),
        tail_loss_history=_relative(attempt / "tail_loss_history.jsonl"),
        final_old_development=_relative(attempt / "final_old_development.json"),
        recovery_source_manifest=RECOVERY_SOURCES,
        recovery_source_manifest_sha256=_file_sha256(_project_path(RECOVERY_SOURCES))))
    artifacts = existing_file_hashes(output)
    artifacts.update(read(_project_path(RECOVERY_SOURCES)))
    artifacts[RECOVERY_SOURCES] = _file_sha256(_project_path(RECOVERY_SOURCES))
    write_json(output / "artifact_manifest.json", artifacts)
    result = dict(status="REPAIR_PREDICTION_PASS_REQUIRES_AUDIT" if gate["passed"]
                  else "REPAIR_PREDICTION_FAIL_REQUIRES_AUDIT",
                  counters=counters, analysis={}, prediction_gate=gate,
                  quick_prediction_diagnostic=None, completed_batches=len(records),
                  frozen_normalization_verified=True, equal_update_sampling_verified=True,
                  primary_checkpoint="final_only", **cfg["boundary"], rl_authorized=False,
                  recovery=read(output / "recovery_provenance.json"),
                  artifact_manifest_sha256=_file_sha256(output / "artifact_manifest.json"),
                  material_passport=dict(origin_skill="academic-research-suite",
                      origin_mode="run", origin_date=datetime.now(timezone.utc).isoformat(),
                      verification_status="REQUIRES_AUDIT", version_label="r4_closed_loop_repair_v1_resume"),
                  next_action="停止并通知助手只读验收；不自动运行下一阶段。")
    write_json(output / "summary.json", result)
    write_json(output / "SUCCESS.json", dict(summary_sha256=_file_sha256(output / "summary.json")))
    return result


def run(path: str | Path, *, preflight_only: bool = False, smoke: bool = False) -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
        raise RuntimeError("unsupported deterministic CUDA setting")
    cfg, c, parent, output, records, losses, developments, snapshots, report = preflight(path)
    if preflight_only:
        return report
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = resolve_device("cuda")
    if smoke:
        replay_member(cfg, c, parent, output, records, losses, developments,
                      report["interruption_update"], device, None, limit=2)
        return dict(status="REPLAY_SMOKE_PASS_NO_RANKING", compared_updates=2,
                    original_output_unchanged=True, rl_authorized=False)
    attempt = output / "recovery_attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:8])
    attempt.mkdir(parents=True, exist_ok=False)
    write_json(attempt / "preflight.json", report)
    progress = Progress(attempt, device)
    try:
        model, tail, checkpoint, stores, _ = replay_member(
            cfg, c, parent, output, records, losses, developments,
            report["interruption_update"], device, progress)
        progress.close()
        return finalize(cfg, c, parent, output, records, snapshots, attempt,
                        model, tail, checkpoint, stores)
    except BaseException:
        progress.close()
        write_json(attempt / "failure.json", dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise
