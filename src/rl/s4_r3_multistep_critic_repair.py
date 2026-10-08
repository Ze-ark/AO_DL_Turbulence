"""R3-D2-A多步评论家修复的目标数学与设计预检。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from src.rl.s4_training import _file_sha256, _load_yaml, _project_path, _relative


@dataclass(frozen=True)
class CriticTargetSpec:
    """冻结演员评论家实验中的单一目标定义。"""

    identifier: str
    kind: str
    horizon: int
    trace_lambda: float | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "CriticTargetSpec":
        spec = cls(
            identifier=str(value["id"]),
            kind=str(value["kind"]),
            horizon=int(value["horizon"]),
            trace_lambda=(
                None
                if value.get("trace_lambda") is None
                else float(value["trace_lambda"])
            ),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("critic target id cannot be empty")
        if self.kind not in {"n_step", "truncated_td_lambda"}:
            raise ValueError(f"unsupported critic target kind: {self.kind}")
        if self.horizon <= 0:
            raise ValueError("critic target horizon must be positive")
        if self.kind == "n_step" and self.trace_lambda is not None:
            raise ValueError("n-step target cannot define trace_lambda")
        if self.kind == "truncated_td_lambda":
            if self.trace_lambda is None or not 0.0 <= self.trace_lambda <= 1.0:
                raise ValueError("truncated TD(lambda) requires lambda in [0, 1]")


def truncated_lambda_weights(
    horizon: int,
    trace_lambda: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """返回截断前向TD(lambda)权重，最后一个时域承接剩余质量。"""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0.0 <= trace_lambda <= 1.0:
        raise ValueError("trace_lambda must be in [0, 1]")
    if horizon == 1:
        return torch.ones(1, dtype=dtype, device=device)
    powers = torch.arange(horizon - 1, dtype=dtype, device=device)
    prefix = (1.0 - trace_lambda) * torch.pow(
        torch.as_tensor(trace_lambda, dtype=dtype, device=device), powers
    )
    tail = torch.as_tensor(
        [trace_lambda ** (horizon - 1)], dtype=dtype, device=device
    )
    weights = torch.cat((prefix, tail))
    if not torch.allclose(
        weights.sum(),
        torch.ones((), dtype=dtype, device=device),
        atol=1e-12,
        rtol=1e-12,
    ):
        raise RuntimeError("truncated TD(lambda) weights do not sum to one")
    return weights


def select_critic_target(
    consecutive_n_step_targets: torch.Tensor,
    spec: CriticTargetSpec,
) -> torch.Tensor:
    """从连续的1..N步目标中选取n步目标或形成截断TD(lambda)目标。"""
    spec.validate()
    if consecutive_n_step_targets.ndim < 1:
        raise ValueError("n-step target tensor must have at least one dimension")
    if consecutive_n_step_targets.shape[-1] < spec.horizon:
        raise ValueError("n-step target tensor does not cover the requested horizon")
    targets = consecutive_n_step_targets[..., : spec.horizon]
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("n-step target tensor contains non-finite values")
    if spec.kind == "n_step":
        return targets[..., spec.horizon - 1]
    weights = truncated_lambda_weights(
        spec.horizon,
        float(spec.trace_lambda),
        dtype=targets.dtype,
        device=targets.device,
    )
    return (targets * weights).sum(dim=-1)


def preflight_s4_r3_multistep_critic_repair_design(
    config_path: str | Path,
) -> dict[str, Any]:
    """验证D2-A设计、上游证据、保护开关和种子隔离。"""
    path = _project_path(config_path)
    experiment = _load_yaml(path)
    metadata = experiment["metadata"]
    if str(metadata["stage"]) != "S4-D2-R3-D2-A-DESIGN":
        raise ValueError("critic-repair design must identify S4-D2-R3-D2-A-DESIGN")
    if not bool(metadata.get("user_authorized_design", False)):
        raise RuntimeError("R3-D2-A design requires user authorization")
    for field in (
        "allow_formal_training",
        "allow_actor_updates",
        "allow_alpha_updates",
        "allow_reward_change",
        "allow_action_budget_change",
        "allow_s4d3_access",
        "allow_real_hardware_actions",
    ):
        if bool(metadata.get(field, True)):
            raise RuntimeError(f"R3-D2-A design protection flag must remain false: {field}")

    upstream = experiment["upstream_d1c"]
    for field in (
        "summary",
        "preflight",
        "effective_config",
        "source_manifest",
        "probe_records",
        "episode_records",
        "experiment_config",
        "audit_record",
    ):
        actual = _file_sha256(_project_path(upstream[field]))
        if actual != str(upstream[f"{field}_sha256"]):
            raise RuntimeError(f"R3-D2-A upstream hash mismatch: {field}")
    summary = json.loads(
        _project_path(upstream["summary"]).read_text(encoding="utf-8")
    )
    if bool(summary["experiment"]["quick"]):
        raise RuntimeError("R3-D2-A cannot consume quick-smoke evidence")
    if str(summary["interpretation"]["status"]) != str(
        upstream["required_interpretation_status"]
    ):
        raise RuntimeError("R3-D2-A upstream interpretation changed")
    if not bool(summary["upstream_alignment"]["pass"]):
        raise RuntimeError("R3-D2-A requires passing D1-C upstream alignment")
    audit_text = _project_path(upstream["audit_record"]).read_text(encoding="utf-8")
    if "Verification Status: `ANALYZED / FINITE-HORIZON-CORRECTION-CONFIRMED`" not in audit_text:
        raise RuntimeError("R3-D2-A upstream audit is not analyzed")

    specs = [CriticTargetSpec.from_mapping(item) for item in experiment["targets"]]
    identifiers = [item.identifier for item in specs]
    required = ["one_step_control", "n_step_16", "n_step_32", "td_lambda_095"]
    if identifiers != required:
        raise RuntimeError(f"R3-D2-A target order must remain {required}")
    if [(item.kind, item.horizon) for item in specs[:3]] != [
        ("n_step", 1),
        ("n_step", 16),
        ("n_step", 32),
    ]:
        raise RuntimeError("R3-D2-A n-step targets changed")
    lambda_spec = specs[-1]
    if lambda_spec.horizon != 32 or lambda_spec.trace_lambda != 0.95:
        raise RuntimeError("R3-D2-A TD(lambda) target must remain lambda=0.95, N=32")

    policy = experiment["frozen_policy"]
    if list(policy["arms"]) != ["student_backbone"]:
        raise RuntimeError("R3-D2-A must use only the three student-backbone actors")
    seeds = [int(value) for value in policy["policy_seeds"]]
    if seeds != [9301, 9302, 9303]:
        raise RuntimeError("R3-D2-A policy seeds changed")
    if not bool(policy["deterministic_actor"]) or any(
        bool(policy[field])
        for field in ("update_actor", "update_alpha", "finetune_student")
    ):
        raise RuntimeError("R3-D2-A frozen-policy contract changed")

    split_ranges: dict[str, list[tuple[int, int]]] = {}
    for split in ("development", "validation", "mechanism_audit"):
        ranges = [tuple(int(x) for x in pair) for pair in experiment["data_splits"][split]]
        if any(len(pair) != 2 or pair[0] > pair[1] for pair in ranges):
            raise ValueError(f"invalid seed range in {split}")
        split_ranges[split] = ranges
    flattened = [
        (split, low, high)
        for split, ranges in split_ranges.items()
        for low, high in ranges
    ]
    for index, (split_a, low_a, high_a) in enumerate(flattened):
        for split_b, low_b, high_b in flattened[index + 1 :]:
            if split_a != split_b and max(low_a, low_b) <= min(high_a, high_b):
                raise RuntimeError(f"seed leakage between {split_a} and {split_b}")
    reserved = int(experiment["data_splits"]["reserved_s4d3_seed_base"])
    if reserved < 4_000_000 or any(high >= reserved for _, _, high in flattened):
        raise RuntimeError("R3-D2-A seed range overlaps the S4-D3 reservation")

    tracked = [str(item) for item in experiment["tracked_source_files"]]
    missing = [item for item in tracked if not _project_path(item).is_file()]
    if missing:
        raise FileNotFoundError(f"R3-D2-A tracked design files missing: {missing}")

    return {
        "status": "READY_FOR_TRAINER_IMPLEMENTATION",
        "stage": str(metadata["stage"]),
        "upstream_interpretation": summary["interpretation"]["status"],
        "policy_seeds": seeds,
        "target_ids": identifiers,
        "planned_critic_fits": len(seeds) * len(specs),
        "actor_updates": 0,
        "alpha_updates": 0,
        "reward_changed": False,
        "action_budget_changed": False,
        "formal_training_authorized": False,
        "sealed_s4d3_access": False,
        "real_slm_actions": False,
        "config": str(_relative(path)),
        "next_action": (
            "Implement the CUDA trainer and quick smoke before asking the user "
            "to start any formal critic fitting."
        ),
    }

