"""A8元数据恢复测试；不训练、不推理、不生成正式结果。"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.rl.s4_r3_h16_head_split_recovery import (
    _verify_manifest,
    assert_csv_matches,
    finalize_s4_r3_h16_head_split,
    validate_terminal_git_failure,
)
from src.rl.s4_r3_h16_head_split import _git_record_utf8
from src.rl.s4_training import _file_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/s4_r3_h16_head_split_v1.yaml"
FORMAL_OUTPUT = ROOT / "outputs/s4_r3_h16_head_split_v1"


def test_git_record_explicitly_decodes_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(kwargs)
        stdout = "abc123\n" if command[1] == "rev-parse" else "?? docs/中文记录.md\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("src.rl.s4_r3_h16_head_split.subprocess.run", fake_run)
    assert _git_record_utf8() == {"commit": "abc123", "dirty": True}
    assert all(call["encoding"] == "utf-8" and call["errors"] == "replace" for call in calls)
    assert all("text" not in call for call in calls)


def test_git_record_never_strips_none_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout=None, stderr=None)

    monkeypatch.setattr("src.rl.s4_r3_h16_head_split.subprocess.run", fake_run)
    assert _git_record_utf8() == {"commit": None, "dirty": False}


def test_failure_guard_is_specific_to_terminal_git_metadata_error() -> None:
    valid = {
        "exception": "AttributeError",
        "message": "'NoneType' object has no attribute 'strip'",
        "traceback": "s4_r3_h16_head_split.py -> _git_record -> s4_training.py",
        "automatic_retry": False,
    }
    validate_terminal_git_failure(valid)
    changed = dict(valid, exception="RuntimeError")
    with pytest.raises(RuntimeError, match="not the approved"):
        validate_terminal_git_failure(changed)


def test_manifest_allows_only_declared_metadata_source_drift(tmp_path: Path) -> None:
    path = tmp_path / "source.py"
    path.write_text("old", encoding="utf-8")
    expected = _file_sha256(path)
    path.write_text("new", encoding="utf-8")
    name = str(path).replace("\\", "/")
    drift = _verify_manifest({str(path): expected}, allowed_drift={name})
    assert drift == [{"path": name, "frozen_sha256": expected, "current_sha256": _file_sha256(path)}]
    with pytest.raises(RuntimeError, match="unexpected frozen-file drift"):
        _verify_manifest({str(path): expected}, allowed_drift=set())


def test_csv_reproduction_is_exact(tmp_path: Path) -> None:
    path = tmp_path / "result.csv"
    path.write_text("seed,score,passed\n9301,0.5,True\n", encoding="utf-8")
    assert_csv_matches(path, [{"seed": 9301, "score": 0.5, "passed": True}])
    with pytest.raises(RuntimeError, match="values do not reproduce"):
        assert_csv_matches(path, [{"seed": 9301, "score": 0.6, "passed": True}])


def test_recovery_cli_help_is_utf8_and_defaults_to_read_only() -> None:
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/finalize_s4_r3_h16_head_split.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        check=True,
    )
    assert "默认只读" in result.stdout
    assert "--acknowledge-finalize" in result.stdout
    assert "不训练、不推理" in result.stdout


@pytest.mark.skipif(not (FORMAL_OUTPUT / "failure.json").is_file(), reason="本地没有A8故障目录")
def test_current_formal_output_is_recoverable_without_writes() -> None:
    before = {path.relative_to(FORMAL_OUTPUT): _file_sha256(path) for path in FORMAL_OUTPUT.rglob("*") if path.is_file()}
    result = finalize_s4_r3_h16_head_split(CONFIG, acknowledge_finalize=False)
    assert result["status"] in {"READY_FOR_METADATA_ONLY_FINALIZATION", "ALREADY_FINALIZED"}
    if result["status"] == "READY_FOR_METADATA_ONLY_FINALIZATION":
        assert result["fits"] == 6
        assert result["test_branches"] == 324
        assert result["saved_test_predictions"] == 6
        assert result["existing_csvs_exactly_reproduced"] is True
        assert result["training_or_simulation_will_run"] is False
    after = {path.relative_to(FORMAL_OUTPUT): _file_sha256(path) for path in FORMAL_OUTPUT.rglob("*") if path.is_file()}
    assert after == before
