"""一次闭环状态动力学增补：采集/训练与完整闭环分阶段，禁止自动重试。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import time
import traceback

import torch

from src.rl.r4_baseline_selection import METRICS, read
from src.rl.r4_baselines import baseline_delta, guarded_correction, range_eligible
from src.rl.r4_closed_loop import model_ensembles
from src.rl.r4_control import NominalCalibration
from src.rl.r4_delta_learning import PairStore, paired_losses, pulse_rollout
from src.rl.r4_dynamics import normalized_errors, rollout
from src.rl.r4_dynamics_experiment import Progress, evaluate, safe_git_record, verify_hashes
from src.rl.r4_interface_smoke import write_json
from src.rl.r4_mpc import SearchConfig
from src.rl.r4_observation import PowerMeasurement, R4Interface, simulation_residual_proxy
from src.rl.r4_selected_anchor import load_selected, selected_plan, selected_request
from src.rl.r4_trajectory import EPISODE_SCHEMA, EpisodeStore
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _load_yaml, _profiles, _project_path, _relative
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.robust_control import RobustnessCondition

SOURCES = "configs/experiments/s4_r4_closed_loop_repair_v1_sources.json"
ARMS = ("old_data", "mixed_data")
CONTROLLERS = ("integrator", "old_data_mpc", "mixed_data_mpc")


def settings(cfg: dict, quick: bool) -> dict:
    t = dict(cfg["training"])
    if quick:
        t.update({k: v for k, v in cfg["quick"].items() if k not in ("steps", "per_family")})
    return dict(t, quick=quick, n=cfg["quick"]["per_family"] if quick else cfg["data"]["per_family"],
                batch=2 if quick else cfg["data"]["batch"],
                steps=cfg["quick"]["steps"] if quick else cfg["data"]["steps"],
                families=1 if quick else 3, profiles=1 if quick else 6)


def budget(c: dict, phase: str) -> dict[str, int]:
    states = c["families"] * c["profiles"] * c["n"] * c["steps"]
    if phase == "evaluate":
        return dict(physical_transitions=3*states, max_planning_forward_samples=2*states*12288,
                    training_updates=0, training_forward_samples=0, prediction_forward_samples=0)
    if phase != "train":
        raise ValueError("unknown phase")
    old_episodes = 16 if c["quick"] else 1152
    new_episodes = c["families"] * c["profiles"] * c["n"]
    events = c["updates"] // c["interval"]
    return dict(physical_transitions=2*states, max_planning_forward_samples=2*states*12288,
                training_updates=6*c["updates"],
                training_forward_samples=6*c["updates"]*8*(c["trajectory_batch"]+2*c["pair_batch"]),
                prediction_forward_samples=6*8*len(c["evaluation_starts"])*(events*old_episodes+new_episodes))


def split_seeds(cfg: dict) -> dict[str, set[int]]:
    data = cfg["data"]
    splits = {key: {s+i for s in data[key+"_starts"] for i in range(data["per_family"])}
              for key in ("train", "development", "reserved_confirmation")}
    for key in ("train", "development", "evaluation"):
        splits["quick_"+key] = set(range(data["quick_"+key+"_start"], data["quick_"+key+"_start"]+2))
    seen: set[int] = set()
    for values in splits.values():
        if not values or values & seen or min(values) < 3792000 or max(values) >= 3798200:
            raise ValueError("repair weather split overlap or reserved range violation")
        seen |= values
    return splits


def validate_config(cfg: dict) -> None:
    if (cfg["stage"] != "S4-D2-R4-1C" or cfg["runtime"] != dict(device="cuda", formal_owner="user_ide", automatic_retry=False)
            or cfg["boundary"] != dict(rl_updates=0, real_slm_actions=False, confirmation_access=False, automatic_retry=False)):
        raise ValueError("repair requires user-owned CUDA supervised experiment")
    if (cfg["output_directory"] != "outputs/s4_r4_closed_loop_repair_v1"
            or cfg["parent"] != "configs/experiments/s4_r4_dynamics_v1.yaml"
            or cfg["upstream"] != "outputs/s4_r4_candidate_full_closed_loop_v1"):
        raise ValueError("repair input/output scope changed")
    t = cfg["training"]
    if (t["arms"] != list(ARMS) or t["updates"] != 2000 or t["trajectory_batch"] != 64
            or t["pair_batch"] != 16 or t["learning_rate"] != .0001 or t["gradient_clip"] != 1.
            or t["interval"] != 500 or t["primary_checkpoint"] != "final_only"
            or t["pair_loss"] != "absolute_plus_delta" or t["mixed_new_fraction"] != .5
            or t["evaluation_starts"] != [0, 40, 80, 120, 160]
            or cfg["data"]["per_family"] != 32 or cfg["data"]["steps"] != 200 or cfg["data"]["batch"] != 16):
        raise ValueError("frozen repair training contract changed")
    split_seeds(cfg)
    for key, start in (("train", 3792000), ("development", 3794000), ("reserved_confirmation", 3796000)):
        if cfg["data"][key+"_starts"] != [start, start+256, start+512]:
            raise ValueError("frozen weather ranges changed")


def require_disjoint(stores: dict[str, EpisodeStore], pairs: PairStore) -> None:
    old = set(stores["old_train"].data["weather_seeds"].tolist())
    pair_seeds = set(pairs.data["weather"].tolist())
    if not pair_seeds <= old:
        raise ValueError("paired training labels outside original training weather")
    seen = old
    for key in ("old_development", "train", "development"):
        values = set(stores[key].data["weather_seeds"].tolist())
        if values & seen:
            raise ValueError("weather leakage across train/development stores")
        seen |= values


def mix_batches(old: dict, new: dict, arm: str) -> dict:
    if arm not in ARMS or old.keys() != new.keys():
        raise ValueError("invalid mixed batch")
    n = len(old["history"])
    if n < 2 or n % 2 or any(len(v) != n for v in old.values()) or any(len(v) != n//2 for v in new.values()):
        raise ValueError("exact half-batch required")
    return old if arm == "old_data" else {k: torch.cat((v[:n//2], new[k]), 0) for k, v in old.items()}


def prediction_gate(metrics: dict) -> dict:
    values = torch.tensor([metrics[arm] for arm in ARMS], dtype=torch.float64)
    if values.shape != (2, 3, 4, 2) or not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("six finite nonnegative prediction metrics required")
    # 比较三个模型的平均误差；模态与功率两头分别过门槛，不能互相掩盖。
    errors = values[:, :, -1].mean(1)
    good_reference = bool((errors[0] > 0).all())
    passed = good_reference and bool((errors[1] <= .95*errors[0]).all())
    return dict(passed=passed, h8_mean_errors=errors.tolist(),
                reduction=(1-errors[1]/errors[0]).tolist() if good_reference else None,
                components=["residual", "power"], comparison="mixed_data_vs_equal_update_old_data",
                threshold=.05, rl_authorized=False)


def summarize_closed_loop(means: torch.Tensor, seed: int, repeats: int) -> dict:
    if means.shape != (3, 3, 6, 32, 6) or not bool(torch.isfinite(means).all()) or repeats < 1:
        raise ValueError("complete finite weather metrics required")
    values = means.double().mean(2)
    avg = values.mean((1, 2))
    if float(avg[0, 0]) <= 0:
        raise ValueError("nonpositive integrator power")
    delta = values[2]-values[0]
    g = torch.Generator(device=means.device).manual_seed(seed)
    boot = means.new_zeros(repeats, dtype=torch.float64)
    for family in range(3):
        idx = torch.randint(32, (repeats, 32), generator=g, device=means.device)
        boot += delta[family, :, 0][idx].mean(1)/3
    ci = torch.quantile(boot, means.new_tensor([.025, .975], dtype=torch.float64))
    d = delta.mean((0, 1)); rel = float(avg[2, 0]/avg[0, 0]-1)
    gates = dict(power_at_least_one_percent=rel >= .01, paired_ci_positive=float(ci[0]) > 0,
                 all_families_positive=bool((delta[..., 0].mean(1) > 0).all()),
                 strehl_not_lower=float(d[1]) >= 0, phase_rmse_not_higher=float(d[3]) <= 0,
                 violations_safe=float(d[2]) <= .001, saturation_safe=float(d[4]) <= .001,
                 slew_safe=float(d[5]) <= .001, beats_equal_update_control=float(avg[2, 0]) > float(avg[1, 0]))
    return dict(status="REPAIR_DEVELOPMENT_PASS_REQUIRES_AUDIT" if all(gates.values()) else "REPAIR_DEVELOPMENT_FAIL_REQUIRES_AUDIT",
                gates=gates, controller_order=list(CONTROLLERS), metric_order=list(METRICS),
                controller_means=avg.tolist(), relative_power_gain=rel, paired_power_ci95=ci.tolist(),
                family_deltas=delta.mean(1).tolist(), independent_weather=96,
                statistical_unit="complete_weather_six_profiles_averaged", rl_authorized=False)


def completed(directory: Path) -> tuple[dict, dict]:
    s = read(directory/"summary.json")
    if (read(directory/"SUCCESS.json")["summary_sha256"] != _file_sha256(directory/"summary.json")
            or _file_sha256(directory/"artifact_manifest.json") != s["artifact_manifest_sha256"]):
        raise RuntimeError(f"completion hash mismatch: {directory}")
    manifest = read(directory/"artifact_manifest.json")
    verify_hashes(manifest)
    for name in ("summary.json", "SUCCESS.json", "artifact_manifest.json"):
        manifest[_relative(directory/name)] = _file_sha256(directory/name)
    return s, manifest


def preflight(path: str | Path, phase: str, quick: bool) -> tuple:
    cfg = _load_yaml(_project_path(path)); validate_config(cfg)
    c = settings(cfg, quick); planned = budget(c, phase)
    own = read(_project_path(SOURCES)); verify_hashes(own)
    if _relative(_project_path(path)) not in own:
        raise ValueError("unfrozen repair config")
    up = _project_path(cfg["upstream"])
    if _file_sha256(up/"summary.json") != cfg["upstream_sha256"]:
        raise RuntimeError("D3 evidence changed")
    # 确认数据只核对总结与来源哈希，不加载其轨迹或指标张量。
    us = read(up/"summary.json")
    if (us["status"] != "FULL_CLOSED_LOOP_CONFIRMATION_FAIL_REQUIRES_AUDIT"
            or read(up/"SUCCESS.json")["summary_sha256"] != cfg["upstream_sha256"]):
        raise RuntimeError("D3 audited failure prerequisite missing")
    frozen = read(up/"source_manifest.json")
    verify_hashes(frozen)
    frozen.update(own)
    for p in (up/"summary.json", up/"SUCCESS.json", up/"source_manifest.json", _project_path(SOURCES)):
        frozen[_relative(p)] = _file_sha256(p)
    # 扫描配置及运行元数据中明确记载的种子；历史隔离同时由输入清单逐回合校验。
    wanted = set().union(*split_seeds(cfg).values())
    scanned = 0
    paths = list(_project_path("configs").rglob("*.yaml"))
    for name in ("config.json", "effective_config.json", "data_manifest.json", "trajectory_manifest.json"):
        paths += list(_project_path("outputs").glob("*/"+name))
    for p in paths:
        if p.resolve() == _project_path(path).resolve() or p.parent.name.startswith("s4_r4_closed_loop_repair_v1"):
            continue
        nums = {int(x) for x in re.findall(r"(?<![\w.])\d{7}(?![\w.])", p.read_text(encoding="utf-8"))}
        if nums & wanted:
            raise ValueError(f"historical weather collision: {p}")
        scanned += 1
    suffix = ("_quick" if quick else "") + ("_evaluation" if phase == "evaluate" else "")
    output = _project_path(cfg["output_directory"]+suffix)
    if output.exists():
        raise FileExistsError(f"preserve repair output: {output}")
    if phase == "evaluate":
        training = _project_path(cfg["output_directory"]+("_quick" if quick else ""))
        prior, artifacts = completed(training)
        expected = "REPAIR_QUICK_COMPLETE_NO_RANKING" if quick else "REPAIR_PREDICTION_PASS_REQUIRES_AUDIT"
        if prior["status"] != expected or (not quick and not prior["prediction_gate"]["passed"]):
            raise RuntimeError("prediction gate failed; closed-loop evaluation is blocked")
        if read(training/"preflight.json")["frozen_files"].get(SOURCES) != _file_sha256(_project_path(SOURCES)):
            raise RuntimeError("training version mismatch")
        frozen.update(artifacts)
    if not quick:
        qdir = _project_path(cfg["output_directory"]+"_quick"+("_evaluation" if phase == "evaluate" else ""))
        qs, qm = completed(qdir)
        if qs["status"] != "REPAIR_QUICK_COMPLETE_NO_RANKING" or read(qdir/"preflight.json")["frozen_files"].get(SOURCES) != _file_sha256(_project_path(SOURCES)):
            raise RuntimeError("same-version CUDA quick prerequisite missing")
        frozen.update(qm)
    parent = _load_yaml(_project_path(cfg["parent"]))
    spec, selected = load_selected(); frozen.update(selected)
    if shutil.disk_usage(_project_path(".")).free < (256 if quick else 4096)*1024**2:
        raise RuntimeError("insufficient output disk space")
    device = resolve_device("cuda")
    return cfg, c, parent, spec, output, dict(status="READY_FOR_DIAGNOSTIC" if quick else "READY_FOR_USER_IDE",
        phase=phase, quick=quick, device=str(device), budget=planned, frozen_files=frozen,
        scanned_seed_files=scanned, **cfg["boundary"])


@torch.no_grad()
def collect(cfg: dict, c: dict, parent: dict, spec, models: list, split: str,
            output: Path, device: torch.device, progress: Progress, counters: dict,
            controller: str = "original_gru_mpc") -> tuple[list[dict], torch.Tensor]:
    base, _ = load_s1_config(_project_path(parent["environment_config"]))
    base = replace(base, num_modes=21, batch_size=c["batch"], episode_length=c["steps"])
    basis, _, _ = build_action_basis(base, ActionRepresentation("r4_zernike21", "zernike", 21), device)
    profiles = _profiles(parent, parent["profile_ids"][:c["profiles"]])
    cal = NominalCalibration(**parent["nominal_calibration"])
    bounds = read(_project_path("outputs/s4_r4_baseline_safety_v1/calibration.json"))
    starts = ([cfg["data"]["quick_"+split+"_start"]] if c["quick"] else
              cfg["data"]["train_starts" if split == "train" else "development_starts"])
    results = torch.zeros(c["families"], c["profiles"], c["n"], len(METRICS), device=device, dtype=torch.float64)
    records = []
    progress.phase(f"{split} {controller} 完整回合", c["families"]*c["profiles"]*c["n"]//c["batch"]*c["steps"])
    for fi, family in enumerate(parent["families"][:c["families"]]):
        for pi, profile in enumerate(profiles):
            for offset in range(0, c["n"], c["batch"]):
                seed = starts[fi]+offset
                condition = RobustnessCondition.from_mapping(dict(family, base_seed=seed))
                env = AdaptiveOpticsEnv(profile.environment_config(condition.environment_config(base)), device,
                                        profile.effects_config(), basis_override=basis)
                raw, _ = env.reset(seed=seed)
                sensor = torch.Generator(device=device).manual_seed(seed+parent["data"]["sensor_seed_offset"])
                proxy = lambda value: simulation_residual_proxy(value, generator=sensor, noise_std_rad=profile.observation_noise_std_rad)
                interface = R4Interface(calibration=cal); interface.reset(proxy(raw), episode_id=str(seed))
                frames = [interface.snapshot().features[:, -1].cpu()]
                rows = []; commands = []; corrections = []; powers = []; done = []
                for step in range(c["steps"]):
                    snap = interface.snapshot(); h, v = snap.features, snap.valid
                    correction = h.new_zeros(c["batch"], 11); calls = 0
                    reasons = ["integrator"]*c["batch"]
                    torch.cuda.synchronize(device); begin = time.perf_counter()
                    if models:
                        eligible = range_eligible(h, v, bounds)
                        if bool(eligible.any()):
                            first = int(torch.nonzero(eligible)[0, 0])
                            ph = torch.where(eligible[:, None, None], h, h[first:first+1])
                            pv = torch.where(eligible[:, None], v, v[first:first+1])
                            planned = selected_plan(models, ph, pv, spec, cal, SearchConfig(),
                                seed=cfg["search_seed"]+fi*100000+pi*10000+offset*200+step)
                            correction[eligible] = planned["correction"][eligible]
                            calls = int(planned["model_forward_samples"])
                        correction, reasons = guarded_correction(h, v, correction, bounds)
                    expected = selected_request(spec, h, v, correction)
                    delta, _ = baseline_delta(spec, h, v, [], cal)
                    action = interface.issue(delta, correction, step=step)
                    if not torch.equal(action.requested_delta_rad, expected.requested_delta_rad):
                        raise RuntimeError("repair request alignment failed")
                    torch.cuda.synchronize(device); latency = time.perf_counter()-begin
                    raw, _, term, trunc, info = env.step(action.requested_delta_rad)
                    transition = interface.observe_next(proxy(raw), step=step+1,
                        power=PowerMeasurement(info["measured_power_in_bucket"], step, step+1))
                    if bool(trunc.any()) or not bool((term == (step == c["steps"]-1)).all()):
                        raise RuntimeError("incomplete repair episode")
                    if not torch.allclose(action.requested_modal_rad, info["requested_modal"], atol=1e-6, rtol=0):
                        raise RuntimeError("repair environment request mismatch")
                    audit = {k: info[k].cpu() for k in (*METRICS, "applied_modal")}
                    if any(not bool(torch.isfinite(x).all()) for x in audit.values()):
                        raise RuntimeError("nonfinite physical metric")
                    frames.append(transition.next_history.features[:, -1].cpu())
                    commands.append(action.requested_delta_rad.cpu()); corrections.append(action.normalized_correction.cpu())
                    powers.append(transition.action_power.cpu()); done.append(term.cpu())
                    rows.append(dict(**audit, requested_modal=action.requested_modal_rad.cpu(),
                                     latency_seconds=latency, reasons=reasons, model_forward_samples=calls))
                    counters["physical_transitions"] += c["batch"]; counters["planning_forward_samples"] += calls
                    progress.tick(dict(step=step+1, **counters))
                stack = lambda seq: torch.stack(seq, 1)
                data = dict(schema=EPISODE_SCHEMA, source="simulation_residual_proxy_not_holography",
                    collector="old_gru_mpc_closed_loop_development" if controller == "original_gru_mpc" else controller,
                    split=split, frames=stack(frames), commands=stack(commands), corrections=stack(corrections),
                    powers=stack(powers), terminated=stack(done),
                    truncated=torch.zeros(c["batch"], c["steps"], dtype=torch.bool),
                    power_valid=torch.ones(c["batch"], c["steps"], dtype=torch.bool),
                    weather_seeds=torch.arange(seed, seed+c["batch"]), dt_s=base.dt_s)
                name = f"{split}_{controller}_{fi}_{profile.identifier}_{seed}.pt"
                path = output/"trajectories"/name; audit_path = output/"audit"/name
                torch.save(data, path); torch.save(dict(rows=rows), audit_path)
                record = dict(split=split, controller=controller, family=fi, profile=profile.identifier,
                    weather_seeds=data["weather_seeds"].tolist(), file=_relative(path), sha256=_file_sha256(path),
                    audit_file=_relative(audit_path), audit_sha256=_file_sha256(audit_path))
                records.append(record)
                with (output/"collection_manifest.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record)+"\n")
                episode = torch.stack([stack([row[k] for row in rows]).to(device).double().mean(1) for k in METRICS], -1)
                results[fi, pi, offset:offset+c["batch"]] = episode
    return records, results


def load_training_stores(records: list[dict], quick: bool) -> tuple[dict, PairStore]:
    original = read(_project_path("outputs/s4_r4_dynamics_v1/data_manifest.json"))["records"]
    stores = {}
    for split in ("train", "development"):
        paths = [_project_path(r["file"]) for r in original if r["split"] == split]
        stores["old_"+split] = EpisodeStore.from_files(paths[:1] if quick else paths)
        fresh = [r for r in records if r["split"] == split]
        if any(r["controller"] != "original_gru_mpc" for r in fresh):
            raise ValueError("repair data must come from frozen original planner")
        verify_hashes({r["file"]: r["sha256"] for r in fresh})
        stores[split] = EpisodeStore.from_files([_project_path(r["file"]) for r in fresh])
        expected = [s for r in fresh for s in r["weather_seeds"]]
        if stores[split].data["weather_seeds"].tolist() != expected:
            raise ValueError("trajectory weather differs from collection manifest")
    pr = read(_project_path("outputs/s4_r4_delta_supervision_v1/training_pair_manifest.json"))["records"]
    if quick:
        # 保证快速配对标签的天气属于快速读取的第一个旧训练文件。
        old_weather = set(stores["old_train"].data["weather_seeds"].tolist())
        selected = []
        for r in pr:
            item = torch.load(_project_path(r["file"]), map_location="cpu", weights_only=True)
            if set(item["weather"].tolist()) <= old_weather:
                selected = [item]; break
    else:
        selected = [torch.load(_project_path(r["file"]), map_location="cpu", weights_only=True) for r in pr]
    pairs = PairStore(selected, "train")
    require_disjoint(stores, pairs)
    return stores, pairs


def train_models(cfg: dict, c: dict, parent: dict, originals: list, records: list,
                 output: Path, device: torch.device, progress: Progress, counters: dict) -> dict:
    stores, pairs = load_training_stores(records, c["quick"])
    scale = torch.tensor(read(_project_path("outputs/s4_r4_delta_supervision_v1/delta_scale.json"))["value"], device=device)
    cal = NominalCalibration(**parent["nominal_calibration"])
    evalcfg = deepcopy(parent); evalcfg["training"]["evaluation_starts"] = c["evaluation_starts"]
    final: dict[str, list] = {arm: [] for arm in ARMS}
    sampling = {}
    for arm in ARMS:
        for member, seed in enumerate(parent["model"]["member_seeds"]):
            model = deepcopy(originals[member])
            model.gru.requires_grad_(True); model.head.requires_grad_(True); model.linear.requires_grad_(False)
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(parameters, lr=c["learning_rate"])
            pools = {key: stores[key].bootstrap_pool(seed) for key in ("old_train", "train")}
            pp = pairs.pool(seed)
            og, ng, pg = [torch.Generator().manual_seed(seed+offset) for offset in (100, 300, 200)]
            torch.save(dict(state_dict=model.state_dict(), seed=seed), output/"checkpoints"/f"{arm}_{member}_initial.pt")
            progress.phase(f"{arm} 模型{member+1}/3 总训练6组", c["updates"])
            running = 0.
            for update in range(1, c["updates"]+1):
                model.train(); optimizer.zero_grad(set_to_none=True)
                old = stores["old_train"].sample(pools["old_train"], c["trajectory_batch"], 8, og, device)
                # 两组都推进同样的独立采样器；对照组不使用新数据计算损失。
                new = stores["train"].sample(pools["train"], c["trajectory_batch"]//2, 8, ng, device)
                batch = mix_batches(old, new, arm)
                pair = pairs.sample(pp, c["pair_batch"], pg, device)
                pred = rollout(model, batch["history"], batch["valid"], batch["commands"], batch["corrections"], cal)
                trajectory = normalized_errors(pred, batch, model.linear.y_scale, parent["model"]["horizons"]).mean()
                pred_pair = pulse_rollout(model, pair["history"], pair["valid"], pair["correction"], parent["collector_anchor"], cal)
                pred_zero = pulse_rollout(model, pair["history"], pair["valid"], torch.zeros_like(pair["correction"]), parent["collector_anchor"], cal)
                absolute, difference = paired_losses(pred_pair, pred_zero, pair["power"], pair["zero_power"], model.linear.y_scale[21], scale)
                loss = trajectory+absolute+difference
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("nonfinite repair training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, c["gradient_clip"], error_if_nonfinite=True)
                optimizer.step(); counters["training_updates"] += 1
                counters["training_forward_samples"] += 8*(c["trajectory_batch"]+2*c["pair_batch"])
                value = float(loss.detach()); running += value
                record = dict(arm=arm, member=member, seed=seed, update=update, loss=value,
                    average_loss=running/update, trajectory=float(trajectory.detach()), absolute=float(absolute.detach()),
                    delta=float(difference.detach()), gradient_norm=float(norm))
                with (output/"loss_history.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record)+"\n")
                progress.tick({"平均损失": running/update, "当前批次": update, "总完成批次": counters["training_updates"], "总批次": 6*c["updates"]})
                if update % c["interval"] == 0:
                    metrics = evaluate(model, stores["old_development"], evalcfg, device, model.linear.y_scale)
                    counters["prediction_forward_samples"] += metrics["windows"]*8
                    ck = dict(state_dict=model.state_dict(), arm=arm, member=member, seed=seed, update=update,
                              metrics=metrics, selection="final_only", delta_scale=float(scale),
                              old_generator=og.get_state(), new_generator=ng.get_state(), pair_generator=pg.get_state())
                    torch.save(ck, output/"checkpoints"/f"{arm}_{member}_{update:05d}.pt")
                    with (output/"development_history.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(dict(arm=arm, member=member, update=update, **metrics))+"\n")
            if any(not torch.equal(value, originals[member].linear.state_dict()[key]) for key, value in model.linear.state_dict().items()):
                raise RuntimeError("frozen normalization changed")
            states = (og.get_state(), ng.get_state(), pg.get_state())
            if member in sampling and not all(torch.equal(x, y) for x, y in zip(states, sampling[member])):
                raise RuntimeError("paired sampling diverged")
            sampling[member] = states
            gate_metrics = evaluate(model, stores["development"], evalcfg, device, model.linear.y_scale)
            counters["prediction_forward_samples"] += gate_metrics["windows"]*8
            final[arm].append(gate_metrics["per_horizon"])
    write_json(output/"prediction_metrics.json", final)
    return prediction_gate(final)


def execute(cfg: dict, c: dict, parent: dict, spec, output: Path, report: dict) -> dict:
    device = resolve_device("cuda")
    originals = model_ensembles(device)["gru_mpc"]
    counters = dict(physical_transitions=0, planning_forward_samples=0, training_updates=0,
                    training_forward_samples=0, prediction_forward_samples=0)
    progress = Progress(output, device)
    for name in ("trajectories", "audit", "checkpoints"):
        (output/name).mkdir()
    records = []; analysis = {}; gate = None
    try:
        if report["phase"] == "train":
            for split in ("train", "development"):
                collected, _ = collect(cfg, c, parent, spec, originals, split, output, device, progress, counters)
                records.extend(collected)
            write_json(output/"data_manifest.json", dict(records=records, status="COLLECTION_COMPLETE"))
            gate = train_models(cfg, c, parent, originals, records, output, device, progress, counters)
            status = "REPAIR_PREDICTION_PASS_REQUIRES_AUDIT" if gate["passed"] else "REPAIR_PREDICTION_FAIL_REQUIRES_AUDIT"
        else:
            training = _project_path(cfg["output_directory"]+("_quick" if c["quick"] else ""))
            values = []
            for name in CONTROLLERS:
                models = []
                if name != "integrator":
                    arm = name.removesuffix("_mpc")
                    for member, original in enumerate(originals):
                        model = deepcopy(original)
                        ck = torch.load(training/"checkpoints"/f"{arm}_{member}_{c['updates']:05d}.pt", map_location=device, weights_only=True)
                        model.load_state_dict(ck["state_dict"]); model.eval().requires_grad_(False); models.append(model)
                collected, means = collect(cfg, c, parent, spec, models, "evaluation", output, device, progress, counters, controller=name)
                records.extend(collected); values.append(means)
            means = torch.stack(values)
            torch.save(dict(means=means.cpu(), controller_order=CONTROLLERS, metric_order=METRICS), output/"metrics.pt")
            write_json(output/"data_manifest.json", dict(records=records, status="COLLECTION_COMPLETE"))
            if not c["quick"]:
                analysis = summarize_closed_loop(means, cfg["bootstrap_seed"], cfg["bootstrap_repeats"])
            status = analysis.get("status", "REPAIR_QUICK_COMPLETE_NO_RANKING")
        planned = report["budget"]
        if (any(counters[k] != planned[k] for k in ("physical_transitions", "training_updates", "training_forward_samples", "prediction_forward_samples"))
                or counters["planning_forward_samples"] > planned["max_planning_forward_samples"]):
            raise RuntimeError(f"repair budget mismatch: {counters} vs {planned}")
        return dict(status="REPAIR_QUICK_COMPLETE_NO_RANKING" if c["quick"] else status,
                    counters=counters, analysis=analysis,
                    prediction_gate=gate if not c["quick"] else None,
                    quick_prediction_diagnostic=gate if c["quick"] else None,
                    completed_batches=len(records), frozen_normalization_verified=True,
                    equal_update_sampling_verified=report["phase"] == "train",
                    primary_checkpoint="final_only", **cfg["boundary"], rl_authorized=False)
    finally:
        progress.close()


def run(path: str | Path, *, phase: str, quick: bool = False, preflight_only: bool = False) -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
        raise RuntimeError("unsupported deterministic CUDA setting")
    cfg, c, parent, spec, output, report = preflight(path, phase, quick)
    if preflight_only:
        return {k: v for k, v in report.items() if k != "frozen_files"}
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False; torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"preflight.json", report); write_json(output/"config.json", cfg)
    write_json(output/"runtime.json", dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                                         gpu=torch.cuda.get_device_name(), git=safe_git_record()))
    try:
        result = execute(cfg, c, parent, spec, output, report)
        for record in read(output/"data_manifest.json")["records"]:
            verify_hashes({record["file"]: record["sha256"], record["audit_file"]: record["audit_sha256"]})
        verify_hashes(report["frozen_files"])
        artifacts = {_relative(p): _file_sha256(p) for p in output.rglob("*") if p.is_file()}
        write_json(output/"artifact_manifest.json", artifacts)
        result.update(artifact_manifest_sha256=_file_sha256(output/"artifact_manifest.json"),
            material_passport=dict(origin_skill="academic-research-suite", origin_mode="run",
                origin_date=datetime.now(timezone.utc).isoformat(), verification_status="REQUIRES_AUDIT",
                version_label="r4_closed_loop_repair_v1"),
            next_action="停止并通知助手只读验收；不自动运行下一阶段。")
        write_json(output/"summary.json", result)
        write_json(output/"SUCCESS.json", dict(summary_sha256=_file_sha256(output/"summary.json")))
        return result
    except BaseException:
        write_json(output/"failure.json", dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise
