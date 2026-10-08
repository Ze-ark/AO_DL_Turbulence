"""R4-1C断电恢复的小型确定性契约测试，不产生正式性能结果。"""
from __future__ import annotations

import pytest
import torch

from src.rl.r4_closed_loop_repair_resume import (
    prior_attempt_updates, require_same_generator_state, require_same_record,
    require_same_state, validate_prefix,
)


def histories(updates: int = 8, interval: int = 2, prefix: int = 7) -> tuple[list[dict], list[dict]]:
    losses = []
    dev = []
    for i in range(6):
        arm, member = ("old_data", "mixed_data")[i // 3], i % 3
        final = updates if i < 5 else prefix
        for step in range(1, final + 1):
            losses.append(dict(arm=arm, member=member, update=step, loss=1.0,
                               average_loss=1.0, trajectory=0.2, absolute=0.3,
                               delta=0.5, gradient_norm=2.0))
            if step % interval == 0:
                dev.append(dict(arm=arm, member=member, update=step))
    return losses, dev


def test_only_contiguous_incomplete_last_member_is_accepted() -> None:
    losses, dev = histories()
    assert validate_prefix(losses, dev, 8, 2) == 7
    bad = [dict(row) for row in losses]
    bad[-1]["update"] = 8
    with pytest.raises(ValueError, match="noncontiguous"):
        validate_prefix(bad, dev, 8, 2)


def test_missing_or_extra_development_checkpoint_is_rejected() -> None:
    losses, dev = histories()
    with pytest.raises(ValueError, match="development history"):
        validate_prefix(losses, dev[:-1], 8, 2)
    with pytest.raises(ValueError, match="checkpoint"):
        validate_prefix(losses[:-1], dev, 8, 2)


def test_nonfinite_training_log_is_rejected() -> None:
    losses, dev = histories()
    losses[0]["loss"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        validate_prefix(losses, dev, 8, 2)


def test_exact_state_and_loss_checks_fail_closed() -> None:
    state = {"w": torch.tensor([1.0, 2.0])}
    require_same_state(state, {"w": state["w"].clone()}, "same")
    with pytest.raises(RuntimeError, match="state differs"):
        require_same_state(state, {"w": torch.tensor([1.0, 2.000001])}, "different")
    require_same_record({"loss": 1.0}, {"loss": 1.0}, "same")
    with pytest.raises(RuntimeError, match="log differs"):
        require_same_record({"loss": 1.0}, {"loss": 1.000001}, "different")


def test_sampler_state_comparison_handles_cpu_and_cuda() -> None:
    state = torch.tensor([1, 2, 3], dtype=torch.uint8)
    require_same_generator_state(state, state.clone(), "cpu")
    if torch.cuda.is_available():
        require_same_generator_state(state, state.to("cuda"), "mixed device")
    with pytest.raises(RuntimeError, match="sampler differs"):
        require_same_generator_state(state, torch.tensor([1, 2, 4], dtype=torch.uint8), "different")


def test_prior_attempt_compute_is_counted_without_current_attempt(tmp_path) -> None:
    old = tmp_path / "recovery_attempts" / "old"
    current = tmp_path / "recovery_attempts" / "current"
    old.mkdir(parents=True)
    current.mkdir()
    (old / "progress.jsonl").write_text(
        '{"phase":"R4-1C恢复：重演并核对模型3/3","completed":500}\n'
        '{"phase":"R4-1C恢复：六个最终模型独立预测","completed":6}\n',
        encoding="utf-8",
    )
    (current / "progress.jsonl").write_text(
        '{"phase":"R4-1C恢复：重演并核对模型3/3","completed":2000}\n', encoding="utf-8"
    )
    result = prior_attempt_updates(tmp_path, current)
    assert len(result) == 1 and next(iter(result.values())) == 500
