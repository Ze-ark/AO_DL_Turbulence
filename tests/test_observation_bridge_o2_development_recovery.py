"""汇总恢复专用测试：手写 CPU 数学和元数据，零新环境转移。"""
from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from observation_bridge import development_recovery as entry


def forbidden(*args, **kwargs):
    raise AssertionError("summary recovery must not launch any new experiment")


@pytest.fixture
def no_environment(monkeypatch):
    for module, name in ((entry.development, "run"), (entry.development, "rollout"),
                         (entry.development, "preflight"), (entry.short, "run"),
                         (entry.short, "make_environment"), (entry.short.optics, "make_components")):
        monkeypatch.setattr(module, name, forbidden)


def test_import_does_not_initialize_cuda_or_execute():
    command = [sys.executable, "-B", "-X", "utf8", "-c",
               "import torch; import observation_bridge.development_recovery; "
               "assert not torch.cuda.is_initialized()"]
    result = subprocess.run(command, cwd=entry.ROOT, capture_output=True, text=True,
                            encoding="utf-8", errors="strict", timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("key,value", [
    ("device", "cpu"), ("new_environment_transitions", 1), ("new_environment_transitions", False),
    ("training_updates", 1), ("scientific_gain_analysis", True), ("overwrite_original", True),
    ("saved_decision_replay", False), ("automatic_retry", True),
    ("input_directory", "outputs/observation_bridge_o2_development_v1"),
    ("output_directory", "outputs/observation_bridge_o2_development_v1_quick")])
def test_fixed_summary_only_scope(key, value):
    cfg = entry.development.read_config(entry.CONFIG)
    entry.validate_config(cfg)
    with pytest.raises(ValueError, match="summary-only"):
        entry.validate_config({**cfg, key: value})


def test_output_guard_preserves_existing_and_broad_targets(tmp_path):
    base = tmp_path / "outputs"
    source = base / "original"
    source.mkdir(parents=True)
    entry.output_guard(base / "recovery", source, tmp_path)
    for path in (base, source, source / "nested", tmp_path / "outside"):
        with pytest.raises(ValueError):
            entry.output_guard(path, source, tmp_path)
    occupied = base / "occupied"
    occupied.mkdir()
    with pytest.raises(FileExistsError):
        entry.output_guard(occupied, source, tmp_path)
    assert source.is_dir() and occupied.is_dir()


def test_inventory_requires_exact_set_and_detects_changes(tmp_path):
    item = tmp_path / "record.json"
    entry.short.optics.write_json(item, {"handwritten": 1})
    before = entry.inventory(tmp_path, {"record.json"})
    with pytest.raises(ValueError, match="inventory"):
        entry.inventory(tmp_path, {"missing.json"})
    entry.short.optics.write_json(item, {"handwritten": 2})
    assert entry.inventory(tmp_path, {"record.json"}) != before


def test_two_metadata_repairs_reconstruct_original_executed_bytes():
    current = (entry.ROOT / "observation_bridge/development.py").read_bytes()
    assert entry.restored_execution_sha(current) == entry.ORIGINAL_SOURCE_SHA
    with pytest.raises(ValueError, match="source repair"):
        entry.restored_execution_sha(current.replace(b"result = dict(report, status=", b"result = dict(report, result="))


def test_all_provenance_pins_are_sha256():
    cfg = entry.development.read_config(entry.CONFIG)
    for digest in (cfg["experiment_config_sha256"], cfg["failure_sha256"],
                   entry.ORIGINAL_SOURCE_SHA, entry.CURRENT_SOURCE_SHA, entry.CURRENT_TEST_SHA):
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


def test_r1_fixed_config_preserves_scope_and_prior_attempt():
    cfg = entry.development.read_config(entry.CONFIG_R1)
    entry.validate_config(cfg)
    assert cfg["output_directory"].endswith("_summary_recovery_r1")
    assert cfg["new_environment_transitions"] == 0
    for key, value in (("training_updates", 1), ("failure_sha256", "0" * 64),
                       ("previous_recovery_config_sha256", "0" * 64),
                       ("output_directory", cfg["previous_recovery_directory"]),
                       ("previous_recovery_directory", cfg["input_directory"])):
        with pytest.raises(ValueError, match="summary-only"):
            entry.validate_config({**cfg, key: value})
    broken = deepcopy(cfg)
    broken["previous_recovery_artifact_sha256"]["failure.json"] = "0" * 64
    with pytest.raises(ValueError):
        entry.validate_config(broken)


def test_actual_r1_preparation_seals_both_old_directories(no_environment, monkeypatch):
    monkeypatch.setattr(entry, "output_guard", lambda *args: None)
    monkeypatch.setattr(entry.short, "load_assets", forbidden)
    monkeypatch.setattr(entry, "resolve_device", forbidden)
    prepared = entry.prepare(entry.CONFIG_R1)
    cfg, _, raw, _, _, hashes, saved, _, _ = prepared
    assert len(saved["previous_recovery_hashes"]) == 4
    assert entry.verify_previous_attempt(cfg, hashes) == saved["previous_recovery_hashes"]
    broken = dict(hashes)
    broken["records.jsonl"] = "0" * 64
    with pytest.raises(ValueError, match="original inputs"):
        entry.verify_previous_attempt(cfg, broken)
    assert entry.inventory(raw, set(hashes)) == hashes


def test_previous_attempt_changed_files_are_rejected(monkeypatch):
    cfg = entry.development.read_config(entry.CONFIG_R1)
    raw = entry.ROOT / cfg["input_directory"]
    hashes = entry.short.read_json(entry.ROOT / cfg["previous_recovery_directory"] / "input_manifest.json")["artifact_sha256"]
    original_inventory = entry.inventory
    def changed_inventory(path, expected):
        result = original_inventory(path, expected)
        if path != raw:
            result["failure.json"] = "0" * 64
        return result
    monkeypatch.setattr(entry, "inventory", changed_inventory)
    with pytest.raises(ValueError, match="previous failed recovery"):
        entry.verify_previous_attempt(cfg, hashes)


def test_actual_saved_metadata_preparation_is_readonly(no_environment, monkeypatch):
    monkeypatch.setattr(entry.short, "load_assets", forbidden)
    monkeypatch.setattr(entry, "resolve_device", forbidden)
    # 元数据检查可在恢复后再测试；已有输出拒绝覆盖由独立测试负责。
    monkeypatch.setattr(entry, "output_guard", lambda *args: None)
    prepared = entry.prepare(entry.CONFIG)
    cfg, experiment, raw, output, expected, hashes, saved, source, current = prepared
    assert len(hashes) == 61 and len(expected) == 26
    assert len(saved["rows"]) == 78 and len(saved["progress"]) == 728
    assert entry.inventory(raw, set(hashes)) == hashes
    assert source != current  # 原执行指纹与修复后源码指纹分开保留。
    assert not (entry.ROOT / experiment["output_directory"]).exists()
    assert cfg["training_updates"] == 0


@pytest.mark.parametrize("damage", ["missing_row", "duplicate_row", "wrong_file", "progress_clock", "failure", "batch_rng"])
def test_saved_metadata_rejects_incomplete_or_misaligned_records(monkeypatch, damage):
    cfg = entry.development.read_config(entry.development.CONFIG)
    raw = entry.ROOT / cfg["quick_directory"]
    jsonl = {name: entry.read_jsonl(raw / name) for name in ("records.jsonl", "batch_records.jsonl", "progress.jsonl")}
    failure = entry.short.read_json(raw / "failure.json")
    if damage == "missing_row":
        jsonl["records.jsonl"].pop()
    elif damage == "duplicate_row":
        jsonl["records.jsonl"][-1] = deepcopy(jsonl["records.jsonl"][0])
    elif damage == "wrong_file":
        jsonl["records.jsonl"][0]["trajectory_file"] = "another.pt"
    elif damage == "progress_clock":
        jsonl["progress.jsonl"][-1]["observation_step"] = 27
    elif damage == "failure":
        failure["exception"] = "RuntimeError"
    elif damage == "batch_rng":
        jsonl["batch_records.jsonl"][-1]["camera_final_rng_sha256"] = ["0" * 64] * 3
    original_read = entry.short.read_json
    monkeypatch.setattr(entry, "read_jsonl", lambda path: jsonl[path.name])
    monkeypatch.setattr(entry.short, "read_json", lambda path: failure if path.name == "failure.json" else original_read(path))
    with pytest.raises(ValueError):
        entry.validate_saved_metadata(raw, cfg, entry.batches(cfg))


def handwritten_tensors(steps=2):
    """小型零数组只测试格式/公式，不来自仿真或性能实验。"""
    def zeros(*shape, dtype=torch.float32):
        return torch.zeros(shape, dtype=dtype)
    trace = dict(history=zeros(steps, 3, 8, 79), valid=zeros(steps, 3, 8, dtype=torch.bool),
        original=zeros(steps, 3, 11), selected=zeros(steps, 3, 11), choice=zeros(steps, 3, dtype=torch.int64),
        prediction=zeros(steps, 3, 23), requested_delta=zeros(steps, 3, 21), requested_modal=zeros(steps, 3, 21),
        residual=zeros(steps + 1, 3, 21), measured_power=zeros(steps, 3), next_clock=zeros(steps, 3, 4),
        power_action_step=torch.arange(steps), power_arrival_step=torch.arange(steps) + 1)
    audit = {k: zeros(steps + 1, 3, dtype=torch.float64) for k in ("modal_error_rad", "modal_rmse_rad", "fit_rmse_rad",
        "max_wrapped_neighbor_jump_rad", "batch_true_neighbor_jump_rad", "observation_latency_ms")}
    audit.update(joint_target_rad=zeros(steps + 1, 3, 21, dtype=torch.float64), legacy_projection_rad=zeros(steps + 1, 3, 21),
        camera_negative_clip_fraction=zeros(steps + 1, 3), camera_draw_index=torch.arange(steps + 1)[:, None].expand(-1, 3).clone(),
        requested_modal=zeros(steps, 3, 21), applied_modal=zeros(steps, 3, 21),
        observation_step=(torch.arange(steps) + 1)[:, None].expand(-1, 3).clone(),
        power_action_step=torch.arange(steps)[:, None].expand(-1, 3).clone())
    audit.update({k: zeros(steps, 3) for k in ("action_power", "action_strehl", "action_phase_rmse", "violation", "decision_latency_ms")})
    rows = [dict(family_index=fi, selected_nonoriginal_fraction=0.0, **{k: 0.0 for k in entry.development.METRICS}) for fi in range(3)]
    return trace, audit, rows, dict(modal_error_max_rad=0.0)


def test_handwritten_tensor_math_and_row_means():
    trace, audit, rows, record = handwritten_tensors()
    entry.audit_saved_tensors(trace, audit, rows, record, steps=2)


@pytest.mark.parametrize("damage", ["truth_in_visible", "shape", "nonfinite", "clock", "metric", "requested",
                                     "rmse", "draw_index", "fraction", "candidate", "spatial_guard"])
def test_handwritten_tensor_checks_fail_closed(damage):
    trace, audit, rows, record = handwritten_tensors()
    if damage == "truth_in_visible":
        trace["joint_target_rad"] = audit["joint_target_rad"]
    elif damage == "shape":
        trace["valid"] = trace["valid"][:1]
    elif damage == "nonfinite":
        trace["residual"][0, 0, 0] = float("nan")
    elif damage == "clock":
        audit["power_action_step"][0, 0] = 1
    elif damage == "metric":
        rows[0]["power"] = .1
    elif damage == "requested":
        audit["requested_modal"][0, 0, 0] = .1
    elif damage == "rmse":
        audit["modal_rmse_rad"][0, 0] = .1
    elif damage == "draw_index":
        audit["camera_draw_index"][0, 0] = 1
    elif damage == "fraction":
        audit["camera_negative_clip_fraction"][0, 0] = 1.1
    elif damage == "candidate":
        trace["choice"][0, 0] = 25
    elif damage == "spatial_guard":
        audit["batch_true_neighbor_jump_rad"][0, 0] = 1.6
    with pytest.raises(ValueError):
        entry.audit_saved_tensors(trace, audit, rows, record, steps=2)


def test_23_scores_have_25_candidates_and_index_24_is_valid():
    trace, audit, rows, record = handwritten_tensors()
    assert len(entry.short.frozen.source.candidate_commands(trace["original"][0], .1)) == 25
    for index in (23, 24):
        trace["choice"][0, 0] = index
        rows[0]["selected_nonoriginal_fraction"] = .5
        entry.audit_saved_tensors(trace, audit, rows, record, steps=2)
        prediction = torch.zeros((3, 23))
        prediction[0, index - 2] = 1
        assert int(entry.short.frozen.training.candidate_choice(prediction)[0]) == index


def test_no_experiment_or_optics_call_in_recovery_execution():
    source = inspect.getsource(entry.run)
    for call in (".run(", ".rollout(", ".preflight(", ".make_environment(", ".make_components(",
                 ".step(", ".reset(", ".render(", ".reconstruct("):
        assert call not in source
    assert "short.replay_visible(trace," in source
    assert "short.replay_visible(audit," not in source


def mock_file_lifecycle(monkeypatch, tmp_path):
    """隔离文件生命周期；所有模型/设备运算被替换，不生成回合。"""
    monkeypatch.setattr(entry, "output_guard", lambda *args: None)
    prepared = list(entry.prepare(entry.CONFIG))
    output = tmp_path / "handwritten_metadata_recovery"
    prepared[3] = output
    monkeypatch.setattr(entry, "prepare", lambda path: tuple(prepared))
    models = entry.short.read_json(prepared[2] / "model_manifest.json")
    monkeypatch.setattr(entry.short, "configure_runtime", lambda: None)
    monkeypatch.setattr(entry, "resolve_device", lambda device: torch.device(device))
    monkeypatch.setattr(entry.short, "load_assets", lambda device, parent: (
        {i: object() for i in range(3)}, {(f, s): object() for f in range(4) for s in (7564000, 7564001, 7564002)}, models))
    trace, audit, _, _ = handwritten_tensors(28)
    monkeypatch.setattr(entry.torch, "load", lambda path, **kwargs: trace if path.parent.name == "trajectories" else audit)
    monkeypatch.setattr(entry, "audit_saved_tensors", lambda *args: None)
    # 在该文件生命周期测试中只保留进度/模态最大值接口，不计算性能。
    progress = prepared[6]["progress"]
    for row in progress:
        row["modal_error_max_rad"] = 0.0
    monkeypatch.setattr(entry.short, "require_prefix", lambda *args: None)
    replay = prepared[6]["batch_records"][0]["replay"]
    monkeypatch.setattr(entry.short, "replay_visible", lambda *args: deepcopy(replay))
    monkeypatch.setattr(entry.development, "observation_quality", lambda *args: {"targets_met": True, "scope": "handwritten_metadata_unit_test"})
    monkeypatch.setattr(entry.torch.cuda, "get_device_name", lambda device: "NO_GPU_USED_HANDWRITTEN_METADATA_TEST")
    return output, prepared[2], prepared[5]


def test_summary_recovery_success_is_separate_and_sealed(no_environment, monkeypatch, tmp_path):
    output, raw, before = mock_file_lifecycle(monkeypatch, tmp_path)
    result = entry.run()
    assert result["status"] == entry.STATUS
    assert result["new_environment_transitions"] == 0 and result["analysis"] == {}
    assert result["original_run_retroactively_succeeded"] is False
    success = entry.short.read_json(output / "SUCCESS.json")
    files = {p.name for p in output.iterdir()}
    assert files == set(success["artifact_sha256"]) | {"SUCCESS.json"}
    assert len(success["artifact_sha256"]) == 6
    for name, digest in success["artifact_sha256"].items():
        assert entry.short.optics.file_sha256(output / name) == digest
    assert entry.inventory(raw, set(before)) == before
    assert (raw / "failure.json").exists() and not (raw / "SUCCESS.json").exists()
    with pytest.raises(FileExistsError):
        entry.run()


def test_recovery_failure_retains_original_and_does_not_retry(no_environment, monkeypatch, tmp_path):
    output, raw, before = mock_file_lifecycle(monkeypatch, tmp_path)
    monkeypatch.setattr(entry.short, "replay_visible", forbidden)
    with pytest.raises(AssertionError):
        entry.run()
    failure = entry.short.read_json(output / "failure.json")
    assert failure["new_environment_transitions"] == 0 and failure["automatic_retry"] is False
    assert not (output / "SUCCESS.json").exists()
    assert entry.inventory(raw, set(before)) == before


def test_recovery_preflight_does_not_create_output(no_environment, monkeypatch, tmp_path):
    output, raw, before = mock_file_lifecycle(monkeypatch, tmp_path)
    result = entry.run(preflight_only=True)
    assert result["model_forward_calls"] == 0 and result["new_environment_transitions"] == 0
    assert not output.exists()
    assert entry.inventory(raw, set(before)) == before
