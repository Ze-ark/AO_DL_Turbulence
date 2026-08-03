"""S3数据量与模型容量诊断所需的完整回合抽样和机械分类。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from src.simulation.temporal_dynamics import TemporalDynamicsData


def balanced_episode_subset(
    data: TemporalDynamicsData,
    physical_ids_by_condition: list[str],
    total_episodes: int,
) -> TemporalDynamicsData:
    """在每个物理条件中选择相同数量的完整回合，不随机拆散窗口。"""
    if len(physical_ids_by_condition) <= int(data.condition_index.max()):
        raise ValueError("physical condition mapping does not cover all condition indices")
    unique_physical = sorted(set(physical_ids_by_condition))
    if not unique_physical or total_episodes <= 0:
        raise ValueError("physical conditions and total_episodes must be positive")
    if total_episodes % len(unique_physical):
        raise ValueError("total_episodes must divide evenly across physical conditions")
    per_physical = total_episodes // len(unique_physical)
    selected_mask = torch.zeros(len(data.histories), dtype=torch.bool)

    for physical_id in unique_physical:
        condition_indices = [
            index
            for index, value in enumerate(physical_ids_by_condition)
            if value == physical_id
        ]
        available_by_condition = {
            index: torch.unique(
                data.episode_seed[data.condition_index == index], sorted=True
            )
            for index in condition_indices
        }
        if sum(len(seeds) for seeds in available_by_condition.values()) < per_physical:
            raise ValueError(f"not enough complete episodes for physical condition {physical_id}")

        # 同一物理条件可能由多批独立种子组成。轮流从各批取回合，避免较小
        # 数据规模只来自第一批种子，从而把批次差异误判为数据量效应。
        selected_per_condition = {index: 0 for index in condition_indices}
        for offset in range(per_physical):
            index = condition_indices[offset % len(condition_indices)]
            seed_offset = selected_per_condition[index]
            if seed_offset >= len(available_by_condition[index]):
                raise ValueError(
                    f"seed batches are too imbalanced for physical condition {physical_id}"
                )
            seed = available_by_condition[index][seed_offset]
            selected_per_condition[index] += 1
            selected_mask |= (data.condition_index == index) & (data.episode_seed == seed)

    result = _masked_data(data, selected_mask)
    selected_episode_pairs = set(
        zip(result.condition_index.tolist(), result.episode_seed.tolist())
    )
    if len(selected_episode_pairs) != total_episodes:
        raise RuntimeError("balanced subset did not preserve the requested episode count")
    return result


def classify_learning_curve(
    scale_summaries: dict[int, dict[str, float]],
    *,
    supported_min_192_to_384_rmse_reduction: float,
    supported_min_gap_ratio_reduction_48_to_384: float,
    not_supported_max_192_to_384_rmse_reduction: float,
) -> dict[str, Any]:
    """根据预声明阈值判断现有GRU失败是否得到数据受限解释支持。"""
    required = {48, 192, 384}
    if not required.issubset(scale_summaries):
        raise ValueError("learning curve must contain 48, 192, and 384 episode scales")
    for value in (
        supported_min_192_to_384_rmse_reduction,
        supported_min_gap_ratio_reduction_48_to_384,
        not_supported_max_192_to_384_rmse_reduction,
    ):
        if not 0 <= value <= 1:
            raise ValueError("classification thresholds must be in [0, 1]")

    rmse_192 = scale_summaries[192]["gru_rmse_mean"]
    rmse_384 = scale_summaries[384]["gru_rmse_mean"]
    gap_48 = _gap_ratio(scale_summaries[48])
    gap_384 = _gap_ratio(scale_summaries[384])
    large_scale_reduction = 1 - rmse_384 / rmse_192
    gap_ratio_reduction = 1 - gap_384 / gap_48

    supported = (
        large_scale_reduction >= supported_min_192_to_384_rmse_reduction
        and gap_ratio_reduction >= supported_min_gap_ratio_reduction_48_to_384
    )
    not_supported = (
        large_scale_reduction <= not_supported_max_192_to_384_rmse_reduction
        or gap_ratio_reduction <= 0
    )
    if supported:
        verdict = "DATA_LIMITED_SUPPORTED"
    elif not_supported:
        verdict = "DATA_LIMITED_NOT_SUPPORTED"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "verdict": verdict,
        "gru_rmse_reduction_192_to_384": large_scale_reduction,
        "gru_to_ridge_gap_ratio_48": gap_48,
        "gru_to_ridge_gap_ratio_384": gap_384,
        "gap_ratio_reduction_48_to_384": gap_ratio_reduction,
    }


def classify_capacity(
    hidden_size_summaries: dict[int, dict[str, float]],
    *,
    reference_hidden_size: int,
    min_smaller_model_improvement: float,
) -> dict[str, Any]:
    """判断较小GRU是否稳定优于当前大模型。"""
    if reference_hidden_size not in hidden_size_summaries:
        raise ValueError("reference hidden size is missing")
    if not 0 <= min_smaller_model_improvement <= 1:
        raise ValueError("capacity threshold must be in [0, 1]")
    smaller = {
        size: values
        for size, values in hidden_size_summaries.items()
        if size < reference_hidden_size
    }
    if not smaller:
        raise ValueError("at least one smaller hidden size is required")
    best_size = min(smaller, key=lambda size: smaller[size]["gru_rmse_mean"])
    reference_rmse = hidden_size_summaries[reference_hidden_size]["gru_rmse_mean"]
    improvement = 1 - smaller[best_size]["gru_rmse_mean"] / reference_rmse
    return {
        "verdict": (
            "OVERPARAMETERIZATION_SUPPORTED"
            if improvement >= min_smaller_model_improvement
            else "OVERPARAMETERIZATION_NOT_SUPPORTED"
        ),
        "reference_hidden_size": reference_hidden_size,
        "best_smaller_hidden_size": best_size,
        "best_smaller_model_improvement": improvement,
    }


def summarize_replicates(records: list[dict[str, float | int]], key: str) -> dict[int, dict[str, float]]:
    """按给定整数键汇总多个初始化种子的均值和样本标准差。"""
    grouped: dict[int, list[dict[str, float | int]]] = defaultdict(list)
    for record in records:
        grouped[int(record[key])].append(record)
    result: dict[int, dict[str, float]] = {}
    for group, values in grouped.items():
        gru = torch.tensor([float(item["gru_rmse"]) for item in values], dtype=torch.float64)
        ridge = torch.tensor([float(item["ridge_rmse"]) for item in values], dtype=torch.float64)
        skill = torch.tensor([float(item["skill_score"]) for item in values], dtype=torch.float64)
        result[group] = {
            "runs": float(len(values)),
            "gru_rmse_mean": float(gru.mean()),
            "gru_rmse_std": float(gru.std(unbiased=True)) if len(gru) > 1 else float("nan"),
            "ridge_rmse_mean": float(ridge.mean()),
            "ridge_rmse_std": float(ridge.std(unbiased=True)) if len(ridge) > 1 else float("nan"),
            "skill_score_mean": float(skill.mean()),
            "skill_score_std": float(skill.std(unbiased=True)) if len(skill) > 1 else float("nan"),
        }
    return result


def _masked_data(data: TemporalDynamicsData, mask: torch.Tensor) -> TemporalDynamicsData:
    return TemporalDynamicsData(
        histories=data.histories[mask],
        targets=data.targets[mask],
        episode_seed=data.episode_seed[mask],
        condition_index=data.condition_index[mask],
        target_step=data.target_step[mask],
    )


def _gap_ratio(summary: dict[str, float]) -> float:
    return summary["gru_rmse_mean"] / summary["ridge_rmse_mean"]
