"""S3收敛监视和128/256维容量复查的预声明判断。"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


@dataclass
class ConvergenceTracker:
    """按相对改善幅度累计无实质改善次数，并在最少步数后早停。"""

    min_steps: int
    patience_checks: int
    relative_min_delta: float
    reference_best: float = field(default=float("inf"), init=False)
    stale_checks: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.min_steps <= 0 or self.patience_checks <= 0:
            raise ValueError("min_steps and patience_checks must be positive")
        if not 0 <= self.relative_min_delta < 1:
            raise ValueError("relative_min_delta must be in [0, 1)")

    def update(self, step: int, metric: float) -> bool:
        """更新状态；返回当前是否应停止。"""
        if step <= 0 or not math.isfinite(metric) or metric < 0:
            raise ValueError("step and metric must be finite and non-negative")
        threshold = self.reference_best * (1 - self.relative_min_delta)
        if math.isinf(self.reference_best) or metric < threshold:
            self.reference_best = metric
            self.stale_checks = 0
        else:
            self.stale_checks += 1
        return step >= self.min_steps and self.stale_checks >= self.patience_checks


def convergence_label(
    validation_history: list[float],
    *,
    early_stopped: bool,
    final_window_checks: int,
    max_final_window_improvement: float,
) -> dict[str, float | str]:
    """区分早停收敛、最大步数附近平台和仍在明显改善。"""
    if final_window_checks < 2:
        raise ValueError("final_window_checks must be at least 2")
    if not 0 <= max_final_window_improvement < 1:
        raise ValueError("max_final_window_improvement must be in [0, 1)")
    if len(validation_history) < final_window_checks:
        raise ValueError("validation history is shorter than the requested window")
    first = validation_history[-final_window_checks]
    last = validation_history[-1]
    if first <= 0 or last < 0:
        raise ValueError("validation metrics must be non-negative and start above zero")
    improvement = 1 - last / first
    if early_stopped:
        label = "EARLY_STOPPED_CONVERGED"
    elif improvement <= max_final_window_improvement:
        label = "NEAR_PLATEAU_AT_MAX_STEPS"
    else:
        label = "STILL_IMPROVING_AT_MAX_STEPS"
    return {
        "label": label,
        "final_window_relative_improvement": improvement,
    }


def classify_larger_model(
    setting_summaries: dict[tuple[int, int], dict[str, float]],
    *,
    episode_scales: list[int],
    reference_hidden_size: int,
    candidate_hidden_size: int,
    min_improvement: float,
) -> dict[str, Any]:
    """判断更大模型是否在所有预声明数据规模上都带来足够改善。"""
    if not 0 <= min_improvement <= 1:
        raise ValueError("min_improvement must be in [0, 1]")
    improvements: dict[str, float] = {}
    for episode_scale in episode_scales:
        reference = setting_summaries[(episode_scale, reference_hidden_size)][
            "gru_rmse_mean"
        ]
        candidate = setting_summaries[(episode_scale, candidate_hidden_size)][
            "gru_rmse_mean"
        ]
        improvements[str(episode_scale)] = 1 - candidate / reference
    values = list(improvements.values())
    if all(value >= min_improvement for value in values):
        verdict = "LARGER_MODEL_SUPPORTED"
    elif all(value <= 0 for value in values):
        verdict = "LARGER_MODEL_NOT_SUPPORTED"
    else:
        verdict = "LARGER_MODEL_MIXED"
    return {
        "verdict": verdict,
        "candidate_hidden_size": candidate_hidden_size,
        "reference_hidden_size": reference_hidden_size,
        "rmse_improvement_by_episode_scale": improvements,
    }


def classify_more_data(
    setting_summaries: dict[tuple[int, int], dict[str, float]],
    *,
    smaller_episode_scale: int,
    larger_episode_scale: int,
    hidden_sizes: list[int],
    min_improvement: float,
) -> dict[str, Any]:
    """判断加倍独立回合是否对两个模型容量都带来足够改善。"""
    if not 0 <= min_improvement <= 1:
        raise ValueError("min_improvement must be in [0, 1]")
    improvements: dict[str, float] = {}
    for hidden_size in hidden_sizes:
        smaller = setting_summaries[(smaller_episode_scale, hidden_size)][
            "gru_rmse_mean"
        ]
        larger = setting_summaries[(larger_episode_scale, hidden_size)][
            "gru_rmse_mean"
        ]
        improvements[str(hidden_size)] = 1 - larger / smaller
    values = list(improvements.values())
    if all(value >= min_improvement for value in values):
        verdict = "MORE_DATA_SUPPORTED"
    elif all(value <= 0 for value in values):
        verdict = "MORE_DATA_NOT_SUPPORTED"
    else:
        verdict = "MORE_DATA_MIXED"
    return {
        "verdict": verdict,
        "smaller_episode_scale": smaller_episode_scale,
        "larger_episode_scale": larger_episode_scale,
        "rmse_improvement_by_hidden_size": improvements,
    }
