"""O2-C 已有短回合的汇总恢复；不调用实验入口、光学链或环境。

原失败目录没有成功封存：本模块只能建立当前保存记录的一致性证据，
不能把历史失败改成成功，也不能声称完成物理回合重新复现。
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import platform
import time
import traceback
from typing import Any

import torch

from observation_bridge import development as development
from src.runtime import resolve_device

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/experiments/observation_bridge_o2_development_summary_recovery_v1.yaml"
CONFIG_R1 = "configs/experiments/observation_bridge_o2_development_summary_recovery_r1.yaml"
STATUS = "O2_C_TECHNICAL_SMOKE_SUMMARY_RECOVERED_ONLY"
ORIGINAL_SOURCE_SHA = "08e6c8955c86328b33d1b7f9c566d23ac70f4cfe71edafd8aa3b1f1edad677de"
CURRENT_SOURCE_SHA = "b022b9c372091a7347b7dc561c747a5c219620eb638e6d179990cb210d7e09b5"
CURRENT_TEST_SHA = "87a6f807ce759ac54a1a29c5ee45fef16a46c6f7a65726092250dd9452bf5623"
short = development.short
ROOT_FILES = {"effective_config.json", "preflight.json", "model_manifest.json", "stream_manifest.json",
              "source_manifest.json", "progress.jsonl", "records.jsonl", "batch_records.jsonl", "failure.json"}


def validate_config(cfg: dict[str, Any]) -> None:
    expected = dict(
        schema="observation_bridge_o2_development_summary_recovery_v1",
        scope="existing_quick_records_summary_recovery_only", device="cuda",
        experiment_config=development.CONFIG,
        experiment_config_sha256="2e9582993ee979e6b15079aa8b2110b7bbff7a29f498be5bf50204de53d90eb2",
        input_directory="outputs/observation_bridge_o2_development_v1_quick",
        failure_sha256="30028803e493a97d5f2183f07ea0e904686538db72eaa0df97aab371eb2fa667",
        output_directory="outputs/observation_bridge_o2_development_v1_quick_summary_recovery_v1",
        new_environment_transitions=0, training_updates=0, scientific_gain_analysis=False,
        saved_decision_replay=True, overwrite_original=False, automatic_retry=False)
    if cfg.get("schema") == "observation_bridge_o2_development_summary_recovery_r1":
        expected.update(schema="observation_bridge_o2_development_summary_recovery_r1",
            output_directory="outputs/observation_bridge_o2_development_v1_quick_summary_recovery_r1",
            previous_recovery_directory="outputs/observation_bridge_o2_development_v1_quick_summary_recovery_v1",
            previous_recovery_config_sha256="4bb201fdd4c213234d4e0601caa13a55348537b64f246a6140c7032e0f66d83d",
            previous_recovery_artifact_sha256={
                "effective_config.json": "0460ae9c72c90cd08b101e473fc3be612ce97b68a90660d1fb097be0d705f011",
                "failure.json": "26db29280733ab24fe2d34efd390a1c17186106c85fd34d1ac8ed0bc5ecda72c",
                "input_manifest.json": "460cc5658f034a876b88c68338b510a7f7bead674c0fae2e5c3ef0fc3cebb185",
                "source_alignment.json": "5fdf82e71f12d8078c84f3423c0cfb7ab1c35bb583866749648229fc4590cc5b"})
    if json.dumps(cfg, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("fixed summary-only recovery contract changed")


def output_guard(output: Path, source: Path, root: Path) -> None:
    """拒绝既有目录、广泛目录、逃逸和源目录别名；不删除任何文件。"""
    base, target, original = (root / "outputs").resolve(), output.resolve(), source.resolve()
    if target == base or not target.is_relative_to(base) or target == original or target.is_relative_to(original):
        raise ValueError("recovery output must be a separate new child of workspace outputs")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"preserve existing recovery output: {output}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("invalid saved JSONL record")
    return rows


def inventory(directory: Path, expected: set[str]) -> dict[str, str]:
    for item in [directory, *directory.rglob("*")]:
        if item.is_symlink() or item.is_junction() or not item.resolve().is_relative_to(directory.resolve()):
            raise ValueError("linked or escaping recovery input is forbidden")
    paths = {p.relative_to(directory).as_posix(): p for p in directory.rglob("*") if p.is_file()}
    if set(paths) != expected:
        raise ValueError("saved artifact inventory incomplete or unexpected")
    return {name: short.optics.file_sha256(paths[name]) for name in sorted(paths)}


def verify_previous_attempt(cfg: dict[str, Any], input_hashes: dict[str, str]) -> dict[str, str]:
    """R1 串联旧恢复故障及其输入指纹；只读，不改写过去的状态。"""
    if "previous_recovery_directory" not in cfg:
        return {}
    prior = ROOT / cfg["previous_recovery_directory"]
    pins = cfg["previous_recovery_artifact_sha256"]
    if inventory(prior, set(pins)) != pins or short.optics.file_sha256(ROOT / CONFIG) != cfg["previous_recovery_config_sha256"]:
        raise ValueError("previous failed recovery artifacts or configuration changed")
    manifest = short.read_json(prior / "input_manifest.json")
    failure = short.read_json(prior / "failure.json")
    if (manifest["input_directory"] != cfg["input_directory"] or manifest["artifact_sha256"] != input_hashes
            or failure["status"] != "RECOVERY_STOPPED_NO_AUTOMATIC_RETRY"
            or failure["exception"] != "ValueError"
            or failure["message"] != "saved candidate index outside candidate bank"
            or failure["new_environment_transitions"] != 0 or failure["training_updates"] != 0
            or failure["original_failure_preserved"] is not True or failure["automatic_retry"] is not False
            or short.read_json(prior / "effective_config.json") != development.read_config(CONFIG)):
        raise ValueError("previous recovery failure or original inputs cannot be aligned")
    return pins


def restored_execution_sha(current: bytes) -> str:
    """仅反向还原已授权的两处末端元数据修复，证明计算代码未改变。"""
    newline = b"\r\n" if b"\r\n" in current else b"\n"
    new_merge, old_merge = b"result = dict(report, status=", b"result = dict(**report, status="
    new_tail = newline.join([
        b'incomplete_batch_size=(3 if context["controller"] is not None',
        b'                                     and context["completed_episode_batches"] < report["episode_batches"] else 0),',
        b'                                 **cfg["boundary"]))'])
    old_tail = b'incomplete_batch_size=3, **cfg["boundary"]))'
    if current.count(new_merge) != 1 or current.count(new_tail) != 1:
        raise ValueError("unexpected experiment source repair")
    return hashlib.sha256(current.replace(new_merge, old_merge).replace(new_tail, old_tail)).hexdigest()


def batches(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    seed = cfg["quick"]["weather_seed_base"]
    return [dict(weather_seed=seed, weather_index=0, camera_condition=camera["id"], **branch,
                 file=f"weather_00_{camera['id']}_{branch['controller']}.pt")
            for camera in cfg["camera_conditions"] for branch in development.controller_specs(cfg)]


def validate_saved_metadata(raw: Path, cfg: dict[str, Any], expected: list[dict]) -> dict[str, Any]:
    report = short.read_json(raw / "preflight.json")
    failure = short.read_json(raw / "failure.json")
    budget = development.budget(cfg["quick"])
    if (report["status"] != "O2_C_READY_FOR_TECHNICAL_SMOKE" or report["quick"] is not True
            or report["scientific_gain_analysis"] is not False or report["episode_length"] != 28
            or report["weather_count"] != 1 or report["device"] != "cuda"
            or any(report.get(k) != v for k, v in {**budget, **cfg["boundary"]}.items())
            or report["preflight_model_forward_calls"] != 0 or report["preflight_environment_transitions"] != 0
            or (report["loaded_policies"], report["loaded_scorers"]) != (3, 12)):
        raise ValueError("saved quick preflight contract mismatch")
    if (failure["status"] != "O2_C_STOPPED_NO_AUTOMATIC_RETRY" or failure["exception"] != "TypeError"
            or failure["message"] != "dict() got multiple values for keyword argument 'status'"
            or failure["last_context"] != dict(physical_transitions=2184, completed_episode_batches=26,
                weather_seed=8460000, camera_condition="small_read_noise", controller="current_2_7564002", action_step=27)
            or failure["incomplete_batch_size"] != 3
            or any(failure.get(k) != v for k, v in cfg["boundary"].items())):
        raise ValueError("only the known completed-quick summary failure can be recovered")
    rows, records, progress = [read_jsonl(raw / name) for name in
                              ("records.jsonl", "batch_records.jsonl", "progress.jsonl")]
    mapping = development.validate_rows(rows, cfg, cfg["quick"])
    if len(records) != 26 or len(progress) != 728:
        raise ValueError("incomplete saved batch/progress budget")
    rng_reference = records[0]["camera_final_rng_sha256"]
    previous_elapsed = 0.0
    for index, (item, record) in enumerate(zip(expected, records, strict=True)):
        desired = {k: item[k] for k in ("controller", "weather_seed", "weather_index", "camera_condition")}
        desired.update(complete_episodes=3, physical_transitions=84, camera_frames_per_family=29,
                       scorer_fold=0 if item["scorer_seed"] is not None else None,
                       policy_forward_calls=28 if item["member"] is not None else 0,
                       scorer_forward_calls=3 if item["scorer_seed"] is not None else 0)
        replay = record["replay"]
        if (any(record.get(k) != v for k, v in desired.items())
                or record["camera_final_rng_sha256"] != rng_reference
                or len(rng_reference) != 3 or any(len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in rng_reference)
                or not math.isfinite(record["modal_error_max_rad"])
                or replay != dict(max_absolute_error=0.0, candidate_indices_exact=True, clocks_exact=True,
                                  mask_exact=True, replayed_steps=28, new_environment_transitions=0,
                                  scope="saved_measurements_interface_and_model_decisions_same_device_backend_batch_layout")
                or (item["scorer_seed"] is not None and record.get("paired_prefix_exact") is not True)):
            raise ValueError("saved branch identity, random stream or replay mismatch")
        for fi, family in enumerate(cfg["family_ids"]):
            if mapping[item["camera_condition"], item["weather_seed"], item["controller"], family]["trajectory_file"] != item["file"]:
                raise ValueError("episode points to another trajectory")
        for step in range(28):
            row = progress[index * 28 + step]
            if (any(row[k] != item[k] for k in ("weather_seed", "controller", "camera_condition"))
                    or row["observation_step"] != step + 1 or row["physical_transitions"] != 3 * (index * 28 + step + 1)
                    or row["total_physical_transitions"] != 2184
                    or any(not math.isfinite(row[k]) or row[k] < 0 for k in
                           ("elapsed_seconds", "eta_seconds", "transitions_per_second", "cuda_allocated_gb", "modal_error_max_rad"))
                    or row["elapsed_seconds"] < previous_elapsed):
                raise ValueError("saved progress sequence mismatch")
            previous_elapsed = row["elapsed_seconds"]
    return dict(report=report, failure=failure, rows=rows, batch_records=records, progress=progress)


def tensor_contract(trace: dict, audit: dict, *, steps: int = 28) -> None:
    f32, f64, i64 = torch.float32, torch.float64, torch.int64
    visible = dict(history=((steps, 3, 8, 79), f32), valid=((steps, 3, 8), torch.bool),
        original=((steps, 3, 11), f32), selected=((steps, 3, 11), f32), choice=((steps, 3), i64),
        prediction=((steps, 3, 23), f32), requested_delta=((steps, 3, 21), f32),
        requested_modal=((steps, 3, 21), f32), residual=((steps + 1, 3, 21), f32),
        measured_power=((steps, 3), f32), next_clock=((steps, 3, 4), f32),
        power_action_step=((steps,), i64), power_arrival_step=((steps,), i64))
    observed = {k: ((steps + 1, 3), f64) for k in ("modal_error_rad", "modal_rmse_rad", "fit_rmse_rad",
        "max_wrapped_neighbor_jump_rad", "batch_true_neighbor_jump_rad", "observation_latency_ms")}
    observed.update(joint_target_rad=((steps + 1, 3, 21), f64), legacy_projection_rad=((steps + 1, 3, 21), f32),
                    camera_negative_clip_fraction=((steps + 1, 3), f32), camera_draw_index=((steps + 1, 3), i64))
    observed.update({k: ((steps, 3), f32) for k in ("action_power", "action_strehl", "action_phase_rmse", "violation", "decision_latency_ms")})
    observed.update(requested_modal=((steps, 3, 21), f32), applied_modal=((steps, 3, 21), f32),
                    observation_step=((steps, 3), i64), power_action_step=((steps, 3), i64))
    for data, schema in ((trace, visible), (audit, observed)):
        if set(data) != set(schema):
            raise ValueError("saved tensor whitelist mismatch")
        for key, (shape, dtype) in schema.items():
            value = data[key]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype or not bool(torch.isfinite(value).all()):
                raise ValueError(f"saved tensor shape/dtype/finiteness mismatch: {key}")


def require_exact(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if not torch.equal(actual, expected):
        raise ValueError(f"saved audit alignment failed: {label}")


def audit_saved_tensors(trace: dict, audit: dict, rows: list[dict], record: dict, *, steps: int = 28) -> None:
    tensor_contract(trace, audit, steps=steps)
    device = trace["residual"].device
    indexes = torch.arange(steps, device=device)
    require_exact(trace["power_action_step"], indexes, "power action clock")
    require_exact(trace["power_arrival_step"], indexes + 1, "power arrival clock")
    require_exact(audit["power_action_step"], indexes[:, None].expand(-1, 3), "audit action clock")
    require_exact(audit["observation_step"], (indexes + 1)[:, None].expand(-1, 3), "audit observation clock")
    require_exact(audit["camera_draw_index"], torch.arange(steps + 1, device=device)[:, None].expand(-1, 3), "camera draw clock")
    require_exact(trace["requested_modal"], audit["requested_modal"], "requested command")
    require_exact(trace["measured_power"], audit["action_power"], "nominal noiseless power meter")
    difference = trace["residual"].double() - audit["joint_target_rad"]
    require_exact(difference.abs().amax(-1), audit["modal_error_rad"], "modal absolute error")
    require_exact(difference.square().mean(-1).sqrt(), audit["modal_rmse_rad"], "modal RMSE")
    if float(audit["modal_error_rad"].max()) != record["modal_error_max_rad"]:
        raise ValueError("saved batch modal maximum mismatch")
    for key in ("fit_rmse_rad", "modal_error_rad", "modal_rmse_rad", "observation_latency_ms", "decision_latency_ms", "action_phase_rmse"):
        if bool((audit[key] < 0).any()):
            raise ValueError("negative saved error or latency")
    for key in ("camera_negative_clip_fraction", "violation", "action_power", "action_strehl"):
        if bool(((audit[key] < 0) | (audit[key] > 1)).any()):
            raise ValueError("saved fraction outside physical bounds")
    for key in ("batch_true_neighbor_jump_rad", "max_wrapped_neighbor_jump_rad"):
        if bool(((audit[key] < 0) | (audit[key] > 1.5)).any()):
            raise ValueError("saved spatial sampling guard failed")
    # 打分器输出 23 个增量分数，另有原动作/等价裁剪两个零分候选。
    # 候选总数来自封存纯函数，不把输出宽度误当候选总数。
    candidate_count = len(short.frozen.source.candidate_commands(trace["original"][0], .1))
    if candidate_count != trace["prediction"].shape[-1] + 2:
        raise ValueError("saved scorer outputs do not match the frozen candidate bank")
    if bool(((trace["choice"] < 0) | (trace["choice"] >= candidate_count)).any()):
        raise ValueError("saved candidate index outside candidate bank")
    values = dict(power=audit["action_power"], strehl=audit["action_strehl"], phase_rmse=audit["action_phase_rmse"],
        violation=audit["violation"], requested_applied_gap_rad=(audit["requested_modal"] - audit["applied_modal"]).abs().mean(-1),
        requested_step_abs_rad=trace["requested_delta"].abs().mean(-1), requested_modal_abs_rad=trace["requested_modal"].abs().mean(-1),
        observation_latency_ms=audit["observation_latency_ms"], decision_latency_ms=audit["decision_latency_ms"],
        camera_negative_clip_fraction=audit["camera_negative_clip_fraction"], modal_rmse_rad=audit["modal_rmse_rad"],
        representation_fit_rmse_rad=audit["fit_rmse_rad"])
    means = {key: value.double().mean(0) for key, value in values.items()}
    selected = trace["choice"].ne(0).float().mean(0)
    for fi, row in enumerate(rows):
        if (row["family_index"] != fi or any(abs(float(value[fi]) - row[key]) > 1e-12 for key, value in means.items())
                or float(selected[fi]) != row["selected_nonoriginal_fraction"]):
            raise ValueError("saved episode metrics differ from saved tensors")


def prepare(path: str | Path) -> tuple:
    cfg = development.read_config(path)
    validate_config(cfg)
    raw, output = ROOT / cfg["input_directory"], ROOT / cfg["output_directory"]
    output_guard(output, raw, ROOT)
    if not raw.is_dir() or not raw.resolve().is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("missing or escaping saved quick input")
    experiment_path = ROOT / cfg["experiment_config"]
    if short.optics.file_sha256(experiment_path) != cfg["experiment_config_sha256"]:
        raise ValueError("experiment configuration changed")
    experiment = development.read_config(experiment_path)
    development.validate_config(experiment)
    if short.read_json(raw / "effective_config.json") != experiment:
        raise ValueError("saved effective configuration changed")
    expected = batches(experiment)
    names = ROOT_FILES | {f"{folder}/{item['file']}" for folder in ("trajectories", "audit") for item in expected}
    hashes = inventory(raw, names)
    if hashes["failure.json"] != cfg["failure_sha256"]:
        raise ValueError("original failure record changed")
    previous_hashes = verify_previous_attempt(cfg, hashes)
    saved = validate_saved_metadata(raw, experiment, expected)
    saved["previous_recovery_hashes"] = previous_hashes
    source = short.read_json(raw / "source_manifest.json")
    current = development.source_manifest(experiment_path)
    if (source["config_sha256"] != cfg["experiment_config_sha256"]
            or source["reused_short_loop_sources"] != current["reused_short_loop_sources"]
            or source["new_source_sha256"] != {
                "observation_bridge/development.py": ORIGINAL_SOURCE_SHA,
                "scripts/evaluate_observation_bridge_o2_development.py": "ff4ec39bc203e49e387c9f088d09c5a10fe60717205580980475304c8228a1b9",
                "tests/test_observation_bridge_o2_development.py": "4de3a5449c16da3f0fe168c84083818b9ab45f4f2444ff8659286189da5e16b6"}
            or current["new_source_sha256"]["observation_bridge/development.py"] != CURRENT_SOURCE_SHA
            or current["new_source_sha256"]["tests/test_observation_bridge_o2_development.py"] != CURRENT_TEST_SHA
            or current["new_source_sha256"]["scripts/evaluate_observation_bridge_o2_development.py"] != source["new_source_sha256"]["scripts/evaluate_observation_bridge_o2_development.py"]
            or restored_execution_sha((ROOT / "observation_bridge/development.py").read_bytes()) != ORIGINAL_SOURCE_SHA):
        raise ValueError("executed source cannot be aligned to the two metadata-only repairs")
    prerequisite = development.verify_short_loop(experiment)
    streams = development.verify_streams(experiment, quick=True)
    if (saved["report"]["frozen_sources"] != prerequisite or saved["report"]["stream_manifest"] != streams
            or short.read_json(raw / "stream_manifest.json") != streams):
        raise ValueError("saved prerequisites or streams changed")
    calibration = short.read_json(ROOT / "configs/experiments/observation_bridge_design_v1.json")["real_calibration"]
    fields = {k: v for k, v in calibration.items() if k != "status"}
    if len(fields) != 18 or any(value is not None for value in fields.values()) or calibration["status"] != "UNVERIFIED_DO_NOT_INFER_DEFAULTS":
        raise ValueError("unknown real calibration fields changed")
    return cfg, experiment, raw, output, expected, hashes, saved, source, current


@torch.no_grad()
def run(path: str | Path = CONFIG, *, preflight_only: bool = False) -> dict[str, Any]:
    cfg, experiment, raw, output, expected, hashes, saved, source, current = prepare(path)
    short.configure_runtime()
    device = resolve_device("cuda")
    if device.type != "cuda":
        raise RuntimeError("summary recovery requires CUDA; no CPU fallback")
    parent = development.read_config(experiment["parent"])
    policies, scorers, models = short.load_assets(device, parent)
    if models != short.read_json(raw / "model_manifest.json") or short.nominal_profile(parent).as_record() != saved["report"]["synthetic_nominal_profile"]:
        raise ValueError("saved frozen model or synthetic profile identity changed")
    ready = dict(status="O2_C_SUMMARY_RECOVERY_READY", input_artifacts=61, complete_episode_rows=78,
                 saved_physical_transitions=2184, new_environment_transitions=0, training_updates=0,
                 loaded_policies=3, loaded_scorers=12, model_forward_calls=0, output_directory=str(output))
    if preflight_only:
        return ready
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        short.optics.write_json(output / "input_manifest.json", dict(input_directory=cfg["input_directory"], artifact_sha256=hashes,
            original_success_marker_absent=True, provenance_scope="current_saved_bytes_not_a_retroactive_original_success_seal"))
        short.optics.write_json(output / "effective_config.json", cfg)
        if saved["previous_recovery_hashes"]:
            short.optics.write_json(output / "previous_recovery_manifest.json", dict(
                input_directory=cfg["previous_recovery_directory"],
                artifact_sha256=saved["previous_recovery_hashes"],
                status="PREVIOUS_FAILURE_PRESERVED_NOT_RECLASSIFIED"))
        config_path = Path(path) if Path(path).is_absolute() else ROOT / path
        config_relative = config_path.resolve().relative_to(ROOT.resolve()).as_posix()
        recovery_sources = ["observation_bridge/development_recovery.py", "scripts/recover_observation_bridge_o2_development_summary.py",
                            "tests/test_observation_bridge_o2_development_recovery.py", config_relative]
        recovery_hashes = {name: short.optics.file_sha256(ROOT / name) for name in recovery_sources}
        short.optics.write_json(output / "source_alignment.json", dict(executed=source, current=current,
            reversed_metadata_repairs_match_executed_sha256=True, tests_added_after_original_run=True, recovery_source_sha256=recovery_hashes))
        rows = development.validate_rows(saved["rows"], experiment, experiment["quick"])
        prefixes, maximum, policy_calls, scorer_calls = 0, 0.0, 0, 0
        ideal_max, noisy_rmse = 0.0, []
        originals, trajectories, replay_records = {}, [], []
        for index, item in enumerate(expected):
            trace = torch.load(raw / "trajectories" / item["file"], map_location=device, weights_only=True)
            audit = torch.load(raw / "audit" / item["file"], map_location=device, weights_only=True)
            part = [rows[item["camera_condition"], item["weather_seed"], item["controller"], family] for family in experiment["family_ids"]]
            record = saved["batch_records"][index]
            audit_saved_tensors(trace, audit, part, record)
            for step, row in enumerate(saved["progress"][index * 28:(index + 1) * 28]):
                if row["modal_error_max_rad"] != float(audit["modal_error_rad"][step + 1].max()):
                    raise ValueError("progress observation error differs from saved audit")
            policy = policies.get(item["member"])
            selector = None if item["scorer_seed"] is None else scorers[0, item["scorer_seed"]]
            # 唯一模型输入是白名单 trace；audit 从不传给模型或控制接口。
            replay = short.replay_visible(trace, {**experiment, "episode_length": 28}, policy, selector)
            if replay != record["replay"]:
                raise ValueError("saved-decision revalidation differs from original recorded replay")
            maximum = max(maximum, replay["max_absolute_error"])
            if selector is None and item["member"] is not None:
                originals[item["camera_condition"], item["member"]] = trace
            if selector is not None:
                short.require_prefix(originals[item["camera_condition"], item["member"]], trace, 25)
                prefixes += 1
            if item["camera_condition"] == "noiseless":
                ideal_max = max(ideal_max, record["modal_error_max_rad"])
                require_exact(audit["camera_negative_clip_fraction"], torch.zeros_like(audit["camera_negative_clip_fraction"]), "noiseless negative clipping")
            else:
                noisy_rmse.append(audit["modal_rmse_rad"])
            policy_calls += record["policy_forward_calls"]
            scorer_calls += record["scorer_forward_calls"]
            trajectories.append(dict(**item, visible_path=f"{cfg['input_directory']}/trajectories/{item['file']}",
                audit_path=f"{cfg['input_directory']}/audit/{item['file']}",
                visible_sha256=hashes[f"trajectories/{item['file']}"], audit_sha256=hashes[f"audit/{item['file']}"],
                scorer_fold=record["scorer_fold"]))
            replay_records.append(dict(file=item["file"], **replay))
            print(f"O2-C 汇总恢复 {index + 1}/26 | 已有记录核验 | 新环境转移=0", flush=True)
        quality = development.observation_quality(ideal_max, noisy_rmse, experiment)
        if (prefixes, policy_calls, scorer_calls, maximum, quality["targets_met"]) != (18, 672, 54, 0.0, True):
            raise ValueError("technical recovery checks did not pass")
        development.verify_short_loop(experiment)
        if inventory(raw, set(hashes)) != hashes or development.source_manifest(ROOT / cfg["experiment_config"]) != current:
            raise ValueError("original input/source changed during summary recovery")
        if verify_previous_attempt(cfg, hashes) != saved["previous_recovery_hashes"]:
            raise ValueError("previous recovery artifacts changed during execution")
        if any(short.optics.file_sha256(ROOT / name) != value for name, value in recovery_hashes.items()):
            raise ValueError("recovery source changed during execution")
        result = dict(status=STATUS, verification_status="ANALYZED", scope=cfg["scope"],
            original_directory=cfg["input_directory"], output_directory=cfg["output_directory"],
            original_status=saved["failure"]["status"], original_failure_sha256=hashes["failure.json"],
            original_failure_preserved=True, original_input_artifacts_unchanged=61,
            previous_recovery_artifacts_unchanged=len(saved["previous_recovery_hashes"]),
            previous_recovery_failure_preserved=bool(saved["previous_recovery_hashes"]),
            completed_episode_batches=26, completed_episodes=78, saved_physical_transitions=2184,
            saved_batched_steps=728, incomplete_batch_size_from_complete_records=0,
            original_failure_incomplete_batch_size_metadata=3,
            paired_prefix_checks=prefixes, camera_rng_pair_checks=26, replay_max_absolute_error=maximum,
            recovery_policy_forward_calls=policy_calls, recovery_scorer_forward_calls=scorer_calls,
            replay_environment_transitions=0, new_environment_transitions=0, training_updates=0,
            invalid_observations=0, failed_or_truncated_episodes=0, observation_quality=quality,
            analysis={}, scientific_gain_analysis=False, independent_weather_clusters=1,
            independent_confirmation=False, old_confirmation_trajectory_access=False, real_data_access=False,
            real_slm_actions=False, historical_gate_reclassification=False, automatic_retry=False,
            physical_trajectory_rerun=False, original_run_retroactively_succeeded=False,
            source_alignment="exact_reverse_of_two_completion_metadata_repairs",
            original_last_logged_elapsed_seconds=saved["progress"][-1]["elapsed_seconds"],
            original_exact_total_elapsed_seconds=None, recovery_elapsed_seconds=time.perf_counter() - started,
            runtime=dict(python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
                         device=str(device), gpu=torch.cuda.get_device_name(device), runtime_scope="recovery_not_original_run"),
            inverse_crime_limitation=True, real_accuracy_verified=False, realtime_verified=False,
            statistical_fallacy_checks_covered=11, real_calibration_unknown_fields_preserved=18,
            next_action="Only user IDE may launch full O2-C; no quick rerun, training, confirmation or hardware actions.")
        for name, value in (("trajectory_manifest.json", trajectories), ("replay_checks.json", replay_records), ("summary.json", result)):
            short.optics.write_json(output / name, value)
        artifacts = {p.relative_to(output).as_posix(): short.optics.file_sha256(p) for p in output.rglob("*") if p.is_file()}
        short.optics.write_json(output / "SUCCESS.json", dict(status=STATUS, summary_sha256=artifacts["summary.json"], artifact_sha256=artifacts))
        return result
    except BaseException as exc:
        short.optics.write_json(output / "failure.json", dict(status="RECOVERY_STOPPED_NO_AUTOMATIC_RETRY", exception=type(exc).__name__,
            message=str(exc), traceback=traceback.format_exc(), new_environment_transitions=0, training_updates=0,
            original_failure_preserved=True, automatic_retry=False))
        raise
