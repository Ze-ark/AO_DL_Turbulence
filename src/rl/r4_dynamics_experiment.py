"""R4-1A：固定预算采集与监督动力学训练；不自动进入MPC或RL。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

import torch

from src.rl.r4_control import NominalCalibration
from src.rl.r4_dynamics import (ResidualGRUDynamics, normalized_errors, ridge_sufficient_statistics,
    rollout, solve_ridge, training_statistics)
from src.rl.r4_interface_smoke import seed_blocks, write_json
from src.rl.r4_trajectory import EpisodeStore, collect_episode_batch
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _profiles, _project_path, _relative
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.robust_control import RobustnessCondition
from src.training_progress import counted_progress, update_progress, progress_message

CONTRACT = "configs/experiments/s4_r4_dynamics_v1_sources.json"


def settings(cfg: dict, quick: bool) -> dict:
    c = deepcopy(cfg)
    c["quick_mode"] = quick
    c["output_directory"] = cfg["outputs"]["quick_directory" if quick else "directory"]
    if quick:
        q = c["quick"]
        c["data"].update(steps=q["steps"], batch_episodes=q["batch_episodes"],
            train_per_family=q["episodes_per_family"], development_per_family=q["episodes_per_family"])
        for k in ("updates_per_member", "batch_sequences", "evaluation_starts", "evaluation_interval"):
            c["training"][k] = q[k]
    return c


def weather_seeds(c: dict, split: str, family: int) -> list[int]:
    d = c["data"]
    offset = d["train_offset"] if split == "train" else d["development_offset"]
    start = d["namespace_seed"] + offset + d["family_offsets"][family]
    if c["quick_mode"]: start += d["quick_offset"]
    count = d["train_per_family"] if split == "train" else d["development_per_family"]
    return list(range(start, start+count))


def verify_hashes(manifest: dict[str, str]) -> None:
    for path, digest in manifest.items():
        if _file_sha256(_project_path(path)) != digest:
            raise RuntimeError(f"R4 frozen file changed: {path}")


def preflight(config_path: str | Path, quick: bool) -> tuple[dict, dict]:
    config_path = _project_path(config_path).resolve()
    cfg = _load_yaml(config_path)
    if cfg["stage"] != "S4-D2-R4-1A" or cfg["runtime"] != dict(
        device="cuda", require_cuda=True, formal_owner="user_ide", automatic_retry=False):
        raise ValueError("R4-1A requires user-owned CUDA execution")
    if cfg["boundary"] != dict(rl_updates=0, mpc_evaluation=False, confirmation_access=False,
                               real_slm_actions=False, s4d3_access=False):
        raise ValueError("R4-1A boundary changed")
    source_manifest = json.loads(_project_path(CONTRACT).read_text(encoding="utf-8"))
    verify_hashes(source_manifest)
    if _relative(config_path) not in source_manifest:
        raise ValueError("R4 config is not frozen")
    verify_hashes({cfg["design"]: cfg["design_sha256"], cfg["upstream_summary"]: cfg["upstream_sha256"]})
    up = _project_path(cfg["upstream_summary"])
    summary = json.loads(up.read_text(encoding="utf-8"))
    success = json.loads((up.parent/"SUCCESS.json").read_text(encoding="utf-8"))
    if summary["status"] != "R4_0_INTERFACE_SMOKE_PASS" or success["summary_sha256"] != cfg["upstream_sha256"]:
        raise RuntimeError("R4-0 acceptance missing")
    old_manifest = json.loads((up.parent/"preflight.json").read_text(encoding="utf-8"))["frozen_files"]
    verify_hashes(old_manifest)
    for item in summary["datasets"]:
        verify_hashes({item["file"]: item["sha256"]})
    c = settings(cfg, quick)
    output = _project_path(c["output_directory"]).resolve()
    expected = _project_path("outputs/s4_r4_dynamics_v1" + ("_quick" if quick else "")).resolve()
    if output != expected or output.exists():
        raise FileExistsError(f"R4 dedicated output must be new: {output}")
    # 新块只分配给本实验；不打开旧结果轨迹或未来独立确认集。
    historical = set(); scanned = 0
    paths = list(_project_path("configs").rglob("*.yaml"))
    for name in ("effective_config.json", "data_manifest.json", "preflight.json"):
        paths += list(_project_path("outputs").glob("*/"+name))
    own_outputs = {_project_path(x).resolve() for x in cfg["outputs"].values()}
    for p in paths:
        if p.resolve() == config_path or p.parent.resolve() in own_outputs: continue
        obj = _load_yaml(p) if p.suffix == ".yaml" else json.loads(p.read_text(encoding="utf-8"))
        historical |= seed_blocks(obj); scanned += 1
    if c["data"]["namespace_seed"]//10000 in historical:
        raise RuntimeError("R4 historical seed namespace collision")
    all_sets = []
    for mode in (False, True):
        s = settings(cfg, mode)
        for split in ("train", "development"):
            seeds = {x for f in range(3) for x in weather_seeds(s, split, f)}
            if any(x >= 4000000 or x < 0 or x//10000 == 378 for x in seeds):
                raise ValueError("R4 reserved seed")
            if any(seeds & prior for prior in all_sets): raise ValueError("R4 split overlap")
            all_sets.append(seeds)
    device = resolve_device("cuda")
    base, _ = load_s1_config(_project_path(c["environment_config"]))
    base = replace(base, num_modes=21, episode_length=c["data"]["steps"], batch_size=c["data"]["batch_episodes"])
    profiles = _profiles(c, c["profile_ids"])
    for f, family in enumerate(c["families"]):
        condition = RobustnessCondition.from_mapping(dict(family, base_seed=weather_seeds(c, "train", f)[0]))
        for profile in profiles: profile.environment_config(condition.environment_config(base)).validate()
    train_weather = c["data"]["train_per_family"]*3
    dev_weather = c["data"]["development_per_family"]*3
    manifest = dict(source_manifest)
    manifest[CONTRACT] = _file_sha256(_project_path(CONTRACT))
    report = dict(status="READY_FOR_USER_IDE" if not quick else "READY_FOR_DIAGNOSTIC_SMOKE",
        scientific_status="SUPERVISED_PREDICTION_ONLY", device=str(device), quick=quick,
        train_weather=train_weather, development_weather=dev_weather,
        training_transitions=train_weather*12*c["data"]["steps"],
        development_transitions=dev_weather*12*c["data"]["steps"],
        supervised_updates=3*c["training"]["updates_per_member"],
        historical_files_scanned=scanned, frozen_files=manifest, **c["boundary"])
    return c, report


class Progress:
    def __init__(self, output: Path, device: torch.device):
        self.path = output/"progress.jsonl"; self.device = device
        self.start = time.perf_counter(); self.bar = None

    def phase(self, name: str, total: int) -> None:
        if self.bar is not None: self.bar.close()
        self.name = name; self.phase_start = time.perf_counter()
        self.bar = counted_progress(total=total, description=name, unit="批次")

    def tick(self, metrics: dict | None = None) -> None:
        self.bar.update(1)
        metrics = metrics or {}
        update_progress(self.bar, device=self.device, metrics=metrics)
        elapsed = time.perf_counter()-self.phase_start
        record = dict(phase=self.name, completed=self.bar.n, total=self.bar.total,
            elapsed_seconds=time.perf_counter()-self.start,
            eta_seconds=elapsed/self.bar.n*(self.bar.total-self.bar.n),
            cuda_allocated_gb=torch.cuda.memory_allocated(self.device)/1024**3, **metrics)
        with self.path.open("a", encoding="utf-8") as f: f.write(json.dumps(record, ensure_ascii=False)+"\n")

    def close(self) -> None:
        if self.bar is not None: self.bar.close()


def collect(c: dict, output: Path, device: torch.device, progress: Progress) -> dict:
    base, _ = load_s1_config(_project_path(c["environment_config"]))
    base = replace(base, num_modes=21, episode_length=c["data"]["steps"], batch_size=c["data"]["batch_episodes"])
    basis, _, _ = build_action_basis(base, ActionRepresentation("r4_zernike21", "zernike", 21), device)
    profiles = _profiles(c, c["profile_ids"])
    records = []
    (output/"trajectories").mkdir(); (output/"audit").mkdir()
    for split in ("train", "development"):
        per_family = c["data"]["train_per_family"] if split == "train" else c["data"]["development_per_family"]
        progress.phase(f"采集{split}完整回合", 3*len(profiles)*2*math.ceil(per_family/base.batch_size)*base.episode_length)
        for f, family in enumerate(c["families"]):
            seeds = weather_seeds(c, split, f)
            for profile in profiles:
                for collector in c["data"]["collectors"]:
                    for start in range(0, len(seeds), base.batch_size):
                        batch_seeds = seeds[start:start+base.batch_size]
                        condition = RobustnessCondition.from_mapping(dict(family, base_seed=batch_seeds[0]))
                        env_cfg = replace(condition.environment_config(base), batch_size=len(batch_seeds))
                        data, audit = collect_episode_batch(env_cfg, profile, basis, batch_seeds[0], collector, c, progress.tick)
                        name = f"{split}_{family['id']}_{profile.identifier}_{collector}_{start:04d}.pt"
                        path = output/"trajectories"/name; audit_path = output/"audit"/name
                        torch.save(data, path); torch.save(audit, audit_path)
                        records.append(dict(split=split, family=family["id"], profile=profile.identifier,
                            collector=collector, weather_seeds=batch_seeds, file=_relative(path), sha256=_file_sha256(path),
                            audit_file=_relative(audit_path), audit_sha256=_file_sha256(audit_path)))
                        # 每个完整批回合完成即保存清单，崩溃也能定位已完成部分。
                        write_json(output/"data_manifest.json", dict(status="COLLECTING", records=records))
    manifest = dict(status="COLLECTION_COMPLETE", source="simulation_residual_proxy_not_holography", records=records)
    write_json(output/"data_manifest.json", manifest)
    return manifest


@torch.no_grad()
def evaluate(model, store: EpisodeStore, c: dict, device: torch.device, scale: torch.Tensor,
             initial_power_mean: float | None = None) -> dict:
    starts = torch.tensor(c["training"]["evaluation_starts"])
    episodes = torch.arange(store.count).repeat_interleave(len(starts))
    starts = starts.repeat(store.count)
    errors = []; horizon = max(c["model"]["horizons"])
    calibration = NominalCalibration(**c["nominal_calibration"])
    if model is not None: model.eval()
    for k in range(0, len(starts), c["training"]["evaluation_batch"]):
        batch = store.windows(episodes[k:k+c["training"]["evaluation_batch"]],
            starts[k:k+c["training"]["evaluation_batch"]], horizon, device)
        if model is None:
            if initial_power_mean is None:
                raise ValueError("persistence requires training-only initial power mean")
            last = batch["history"][:, -1]
            # 起点尚无功率时不能把补零当作保持预测，否则会人为削弱对照。
            power = torch.where(last[:, 78:79].bool(), last[:, 74:75],
                                torch.full_like(last[:, 74:75], initial_power_mean))
            pred = torch.cat((last[:, :21], power), dim=-1)[:, None].expand(-1, horizon, -1)
        else:
            pred = rollout(model, batch["history"], batch["valid"], batch["commands"], batch["corrections"], calibration)
        error = normalized_errors(pred, batch, scale, c["model"]["horizons"])
        if not bool(torch.isfinite(error).all()): raise RuntimeError("non-finite R4 development prediction")
        errors.append(error.cpu())
    values = torch.cat(errors)
    return dict(score=float(values.mean()), per_horizon=values.mean(0).tolist(),
                windows=len(values), interpretation="development_open_loop_prediction_not_control_gain")


def safe_git_record() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_project_path("."), check=True,
                                capture_output=True).stdout.decode("utf-8", errors="replace").strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=_project_path("."), check=True,
                                capture_output=True).stdout
        return dict(commit=commit, dirty=bool(status.strip()))
    except (OSError, subprocess.CalledProcessError) as exc:
        return dict(commit=None, dirty=None, unavailable=str(exc))


def execute(c: dict, report: dict, output: Path) -> dict:
    device = resolve_device("cuda")
    torch.use_deterministic_algorithms(True)
    # GRU需要对模型内历史反向传播；关闭cuDNN使train/eval路径和高阶梯度边界明确。
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    progress = Progress(output, device)
    try:
        manifest = collect(c, output, device, progress)
        stores = {split: EpisodeStore.from_files([_project_path(r["file"]) for r in manifest["records"] if r["split"] == split])
                  for split in ("train", "development")}
        train, dev = stores["train"], stores["development"]
        if set(train.data["weather_seeds"].tolist()) & set(dev.data["weather_seeds"].tolist()):
            raise RuntimeError("train/development weather overlap")
        chunk = c["model"]["ridge_chunk"]
        progress.phase("训练集归一化统计", math.ceil(train.count*train.steps/chunk))
        linear = training_statistics(train, c["model"], device, progress.tick)
        torch.save(linear.state_dict(), output/"training_normalization.pt")
        progress.phase("线性模型充分统计", math.ceil(train.count*train.steps/chunk))
        stats = ridge_sufficient_statistics(linear, train, torch.arange(train.count), chunk, device, progress.tick)
        selections = []
        progress.phase("线性正则系数开发选择", len(c["model"]["ridge_alphas"]))
        for alpha in c["model"]["ridge_alphas"]:
            candidate = solve_ridge(linear, stats, alpha)
            metrics = evaluate(candidate, dev, c, device, linear.y_scale)
            selections.append(dict(alpha=alpha, **metrics)); progress.tick({"开发误差": metrics["score"]})
        alpha = min(selections, key=lambda r:r["score"])["alpha"]
        selected_linear = solve_ridge(linear, stats, alpha)
        torch.save(selected_linear.state_dict(), output/"linear_full_train.pt")
        del stats
        persistence = evaluate(None, dev, c, device, linear.y_scale, initial_power_mean=float(linear.y_mean[21]))
        write_json(output/"linear_selection.json", dict(candidates=selections, selected_alpha=alpha, persistence=persistence))
        results = []
        (output/"checkpoints").mkdir()
        calibration = NominalCalibration(**c["nominal_calibration"])
        for member, seed in enumerate(c["model"]["member_seeds"]):
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            pool = train.bootstrap_pool(seed)
            progress.phase(f"模型{member+1}/3天气重采样线性拟合", math.ceil(len(pool)*train.steps/chunk))
            stats = ridge_sufficient_statistics(linear, train, pool, chunk, device, progress.tick)
            member_linear = solve_ridge(linear, stats, alpha); del stats
            linear_metrics = evaluate(member_linear, dev, c, device, linear.y_scale)
            torch.save(dict(state_dict=member_linear.state_dict(), pool=pool, seed=seed, alpha=alpha),
                       output/"checkpoints"/f"linear_{member}.pt")
            model = ResidualGRUDynamics(member_linear, c["model"]["hidden_size"]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=c["training"]["learning_rate"])
            generator = torch.Generator().manual_seed(seed+100)
            total = c["training"]["updates_per_member"]
            best_score = float("inf"); best_update = None
            progress.phase(f"时序预测训练 模型{member+1}/3", total)
            running_loss = 0.0
            for update in range(1, total+1):
                model.train()
                batch = train.sample(pool, c["training"]["batch_sequences"], 8, generator, device)
                optimizer.zero_grad(set_to_none=True)
                prediction = rollout(model, batch["history"], batch["valid"], batch["commands"], batch["corrections"], calibration)
                loss = normalized_errors(prediction, batch, linear.y_scale, c["model"]["horizons"]).mean()
                if not bool(torch.isfinite(loss)): raise RuntimeError("non-finite R4 training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), c["training"]["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                value = float(loss.detach()); running_loss += value
                progress.tick({"平均损失": running_loss/update, "当前损失": value})
                with (output/"loss_history.csv").open("a", encoding="utf-8") as log:
                    if member == 0 and update == 1: log.write("member,seed,update,loss,running_mean\n")
                    log.write(f"{member},{seed},{update},{value},{running_loss/update}\n")
                if update % c["training"]["evaluation_interval"] == 0 or update == total:
                    metrics = evaluate(model, dev, c, device, linear.y_scale)
                    checkpoint = dict(state_dict=model.state_dict(), seed=seed, update=update,
                        model_config=c["model"], metrics=metrics, normalization_source="train_only",
                        data_manifest_sha256=_file_sha256(output/"data_manifest.json"))
                    path = output/"checkpoints"/f"gru_{member}_{update:05d}.pt"
                    torch.save(checkpoint, path)
                    if metrics["score"] < best_score:
                        best_score=metrics["score"]; best_update=update
                        torch.save(checkpoint, output/"checkpoints"/f"gru_{member}_best.pt")
                    with (output/"development_history.jsonl").open("a", encoding="utf-8") as log:
                        log.write(json.dumps(dict(member=member, update=update, **metrics))+"\n")
                    progress_message(f"模型{member+1}/3 第{update}/{total}批：开发预测误差={metrics['score']:.6f}")
            results.append(dict(member=member, seed=seed, linear=linear_metrics,
                                best_gru_score=best_score, best_update=best_update))
        verify_hashes(report["frozen_files"])
        artifacts = { _relative(p): _file_sha256(p) for p in output.rglob("*") if p.is_file() }
        write_json(output/"artifact_manifest.json", artifacts)
        return dict(material_passport=dict(origin_skill="academic-research-suite / experiment-agent",
            origin_mode="run", origin_date=datetime.now(timezone.utc).isoformat(), verification_status="UNVERIFIED",
            version_label="s4d2_r4_1a_dynamics_v1"),
            status="R4_1A_QUICK_SMOKE_COMPLETE" if c["quick_mode"] else "R4_1A_SUPERVISED_TRAINING_COMPLETE_REQUIRES_AUDIT",
            scientific_status="DIAGNOSTIC_ONLY" if c["quick_mode"] else "DEVELOPMENT_PREDICTION_ONLY",
            budget={k:v for k,v in report.items() if k not in ("frozen_files",)}, members=results,
            selected_ridge_alpha=alpha, persistence=persistence,
            artifact_manifest_sha256=_file_sha256(output/"artifact_manifest.json"),
            elapsed_seconds=time.perf_counter()-progress.start,
            r4_1_gate="NOT_EVALUATED_ACTION_RESPONSE_AND_MPC_PENDING", **c["boundary"],
            next_action="停止并等待只读审计；不自动训练RL、不访问独立确认。")
    finally:
        progress.close()


def run(config_path: str | Path, *, quick: bool = False, preflight_only: bool = False) -> dict:
    # 显式运行时配置，在第一次CUDA矩阵运算前设置；模块导入无副作用。
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
        raise RuntimeError("R4 deterministic CUDA requires a supported CUBLAS_WORKSPACE_CONFIG")
    c, report = preflight(config_path, quick)
    if preflight_only: return {k:v for k,v in report.items() if k != "frozen_files"}
    output = _project_path(c["output_directory"]); output.mkdir(parents=True, exist_ok=False)
    write_json(output/"preflight.json", report)
    write_json(output/"effective_config.json", c)
    write_json(output/"runtime.json", dict(python=platform.python_version(), torch=str(torch.__version__),
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), git=safe_git_record()))
    try:
        result = execute(c, report, output)
        write_json(output/"summary.json", result)
        write_json(output/"SUCCESS.json", dict(status=result["status"], summary_sha256=_file_sha256(output/"summary.json")))
        return result
    except Exception as exc:
        write_json(output/"failure.json", dict(error=str(exc), traceback=traceback.format_exc(), automatic_retry=False))
        raise
