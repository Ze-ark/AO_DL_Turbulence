"""S3多帧模态动力学数据、GRU预测器和公平线性基线。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import math
from typing import Any

import torch
from torch import nn

from src.simulation.config import S1EnvConfig
from src.simulation.env import AdaptiveOpticsEnv


@dataclass(frozen=True)
class DynamicsCondition:
    """一个可追溯的风速、风向和随机种子条件。"""

    identifier: str
    wind_speed_mps: float
    wind_direction_deg: float
    base_seed: int

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "DynamicsCondition":
        return cls(
            identifier=str(values["id"]),
            wind_speed_mps=float(values["wind_speed_mps"]),
            wind_direction_deg=float(values["wind_direction_deg"]),
            base_seed=int(values["base_seed"]),
        )


@dataclass(frozen=True)
class TemporalDynamicsData:
    """按完整回合切出的历史窗口与下一帧模态目标。"""

    histories: torch.Tensor
    targets: torch.Tensor
    episode_seed: torch.Tensor
    condition_index: torch.Tensor
    target_step: torch.Tensor

    def validate(self, sequence_length: int, num_modes: int) -> None:
        samples = self.histories.shape[0]
        if self.histories.shape != (samples, sequence_length, num_modes):
            raise ValueError("histories have an invalid shape")
        if self.targets.shape != (samples, num_modes):
            raise ValueError("targets have an invalid shape")
        for metadata in (self.episode_seed, self.condition_index, self.target_step):
            if metadata.shape != (samples,):
                raise ValueError("temporal metadata must have one value per sample")


class GRUModalDynamics(nn.Module):
    """用多帧历史预测下一帧模态；输出采用上一帧加学习残差。"""

    def __init__(
        self,
        num_modes: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_modes <= 0 or hidden_size <= 0 or num_layers <= 0:
            raise ValueError("model dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.num_modes = num_modes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.recurrent = nn.GRU(
            input_size=num_modes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, num_modes),
        )

    def forward(self, normalized_history: torch.Tensor) -> torch.Tensor:
        if normalized_history.ndim != 3:
            raise ValueError("history must have shape [batch, sequence, mode]")
        if normalized_history.shape[-1] != self.num_modes:
            raise ValueError("history mode count does not match the model")
        recurrent_output, _ = self.recurrent(normalized_history)
        return normalized_history[:, -1] + self.delta_head(recurrent_output[:, -1])


def generate_temporal_dynamics_data(
    base_config: S1EnvConfig,
    device: torch.device | str,
    conditions: Sequence[DynamicsCondition],
    sequence_length: int,
    frames_per_episode: int,
    random_action_std_rad: float,
    progress_callback: Callable[[int, int], None] | None = None,
) -> TemporalDynamicsData:
    """从公开模态观测重建扰动，并按回合生成时序监督样本。"""
    if not conditions:
        raise ValueError("conditions must not be empty")
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2 for the linear baseline")
    if not sequence_length <= frames_per_episode <= base_config.episode_length:
        raise ValueError("frames_per_episode must cover the sequence and fit the episode")
    if random_action_std_rad < 0:
        raise ValueError("random_action_std_rad must be non-negative")

    actual_device = torch.device(device)
    history_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    seed_parts: list[torch.Tensor] = []
    condition_parts: list[torch.Tensor] = []
    step_parts: list[torch.Tensor] = []
    total_steps = len(conditions) * frames_per_episode
    completed_steps = 0

    with torch.no_grad():
        for condition_index, condition in enumerate(conditions):
            config = replace(
                base_config,
                wind_speed_mps=condition.wind_speed_mps,
                wind_direction_deg=condition.wind_direction_deg,
            )
            config.validate()
            environment = AdaptiveOpticsEnv(config, actual_device)
            observation, _ = environment.reset(seed=condition.base_seed)
            action_generator = torch.Generator(device=actual_device).manual_seed(
                condition.base_seed + 20_000_000
            )
            states = [_disturbance_from_observation(observation, config.num_modes).cpu()]
            for _ in range(frames_per_episode):
                action = random_action_std_rad * torch.randn(
                    config.batch_size,
                    config.num_modes,
                    generator=action_generator,
                    device=actual_device,
                )
                observation, _, terminal, _, _ = environment.step(action)
                states.append(_disturbance_from_observation(observation, config.num_modes).cpu())
                completed_steps += 1
                if progress_callback is not None:
                    progress_callback(completed_steps, total_steps)
                if terminal.all() and completed_steps % frames_per_episode:
                    raise RuntimeError("environment terminated before requested frames were generated")

            state_tensor = torch.stack(states, dim=1)
            windows = state_tensor.unfold(1, sequence_length, 1)
            # unfold把新维度放在末尾：[episode, window, mode, sequence]。
            windows = windows.permute(0, 1, 3, 2)
            sample_count = frames_per_episode - sequence_length + 1
            histories = windows[:, :sample_count]
            targets = state_tensor[:, sequence_length : sequence_length + sample_count]
            history_parts.append(histories.flatten(0, 1))
            target_parts.append(targets.flatten(0, 1))

            episode_seeds = environment.episode_seeds.detach().cpu()
            seed_parts.append(
                episode_seeds.unsqueeze(1).expand(-1, sample_count).reshape(-1)
            )
            condition_parts.append(
                torch.full(
                    (config.batch_size * sample_count,),
                    condition_index,
                    dtype=torch.int64,
                )
            )
            step_parts.append(
                torch.arange(
                    sequence_length,
                    sequence_length + sample_count,
                    dtype=torch.int64,
                )
                .unsqueeze(0)
                .expand(config.batch_size, -1)
                .reshape(-1)
            )

    result = TemporalDynamicsData(
        histories=torch.cat(history_parts),
        targets=torch.cat(target_parts),
        episode_seed=torch.cat(seed_parts),
        condition_index=torch.cat(condition_parts),
        target_step=torch.cat(step_parts),
    )
    result.validate(sequence_length, base_config.num_modes)
    return result


def modal_normalization(data: TemporalDynamicsData) -> tuple[torch.Tensor, torch.Tensor]:
    """只使用训练历史估计逐模态均值和标准差。"""
    flattened = data.histories.reshape(-1, data.histories.shape[-1]).to(torch.float64)
    mean = flattened.mean(dim=0).to(torch.float32)
    scale = flattened.std(dim=0, unbiased=True).clamp_min(1e-6).to(torch.float32)
    return mean, scale


def normalize_modal(value: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (value - mean) / scale


def denormalize_modal(value: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return value * scale + mean


def constant_velocity_forecast(history: torch.Tensor, velocity_gain: float = 1.0) -> torch.Tensor:
    """冻结的S2线性基线：最后一帧加最近一次速度。"""
    if history.ndim != 3 or history.shape[1] < 2:
        raise ValueError("history must contain at least two frames")
    if velocity_gain < 0:
        raise ValueError("velocity_gain must be non-negative")
    return history[:, -1] + velocity_gain * (history[:, -1] - history[:, -2])


def fit_ridge_autoregression(
    histories: torch.Tensor,
    targets: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    ridge_alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """在训练集上拟合与GRU使用相同历史长度的线性岭回归。"""
    if histories.ndim != 3 or targets.ndim != 2:
        raise ValueError("histories and targets must be [sample, sequence, mode] and [sample, mode]")
    if len(histories) != len(targets) or histories.shape[-1] != targets.shape[-1]:
        raise ValueError("history and target sample/mode dimensions must match")
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha must be non-negative")
    normalized_history = normalize_modal(histories, mean, scale).reshape(len(histories), -1)
    normalized_target = normalize_modal(targets, mean, scale)
    design = normalized_history.to(torch.float64)
    response = normalized_target.to(torch.float64)
    ones = torch.ones(len(design), 1, dtype=torch.float64, device=design.device)
    augmented = torch.cat((design, ones), dim=1)
    regularizer = ridge_alpha * torch.eye(
        augmented.shape[1], dtype=torch.float64, device=augmented.device
    )
    regularizer[-1, -1] = 0
    coefficients = torch.linalg.solve(
        augmented.T @ augmented + regularizer,
        augmented.T @ response,
    ).to(torch.float32)
    return coefficients[:-1], coefficients[-1]


def ridge_autoregressive_forecast(
    histories: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """用冻结的同历史线性岭回归预测下一帧物理模态。"""
    normalized = normalize_modal(histories, mean, scale).reshape(len(histories), -1)
    if weight.shape != (normalized.shape[1], histories.shape[-1]):
        raise ValueError("ridge weight shape does not match history and mode dimensions")
    if bias.shape != (histories.shape[-1],):
        raise ValueError("ridge bias shape does not match mode count")
    normalized_prediction = normalized @ weight + bias
    return denormalize_modal(normalized_prediction, mean, scale)


def normalized_modal_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """返回逐样本、按训练集模态尺度归一化的误差平方。"""
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must share [sample, mode]")
    return torch.mean(((prediction - target) / scale).square(), dim=-1)


def paired_episode_comparison(
    prediction: torch.Tensor,
    linear_prediction: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    episode_seed: torch.Tensor,
    condition_index: torch.Tensor,
) -> dict[str, Any]:
    """以完整回合为独立单位比较GRU与线性预测，避免把相邻帧当独立样本。"""
    gru_mse = normalized_modal_mse(prediction, target, scale).cpu()
    linear_mse = normalized_modal_mse(linear_prediction, target, scale).cpu()
    episode_seed = episode_seed.cpu()
    condition_index = condition_index.cpu()
    if not (
        len(gru_mse)
        == len(linear_mse)
        == len(episode_seed)
        == len(condition_index)
    ):
        raise ValueError("prediction metrics and metadata lengths must match")

    episode_gru: list[torch.Tensor] = []
    episode_linear: list[torch.Tensor] = []
    episode_skill: list[torch.Tensor] = []
    episode_conditions: list[int] = []
    for one_condition in torch.unique(condition_index, sorted=True):
        condition_mask = condition_index == one_condition
        for one_seed in torch.unique(episode_seed[condition_mask], sorted=True):
            mask = condition_mask & (episode_seed == one_seed)
            gru_value = gru_mse[mask].mean()
            linear_value = linear_mse[mask].mean()
            skill = 1 - gru_value / linear_value.clamp_min(1e-12)
            episode_gru.append(gru_value)
            episode_linear.append(linear_value)
            episode_skill.append(skill)
            episode_conditions.append(int(one_condition))

    gru_values = torch.stack(episode_gru).to(torch.float64)
    linear_values = torch.stack(episode_linear).to(torch.float64)
    skill_values = torch.stack(episode_skill).to(torch.float64)
    condition_tensor = torch.tensor(episode_conditions, dtype=torch.int64)
    by_condition = {}
    for one_condition in torch.unique(condition_tensor, sorted=True):
        mask = condition_tensor == one_condition
        by_condition[str(int(one_condition))] = {
            "episodes": int(mask.sum()),
            "skill_score": distribution_summary(skill_values[mask], higher_is_better=True),
            "gru_normalized_rmse": distribution_summary(
                torch.sqrt(gru_values[mask]), higher_is_better=False
            ),
            "linear_normalized_rmse": distribution_summary(
                torch.sqrt(linear_values[mask]), higher_is_better=False
            ),
        }
    return {
        "episodes": len(episode_skill),
        "skill_score": distribution_summary(skill_values, higher_is_better=True),
        "gru_normalized_rmse": distribution_summary(
            torch.sqrt(gru_values), higher_is_better=False
        ),
        "linear_normalized_rmse": distribution_summary(
            torch.sqrt(linear_values), higher_is_better=False
        ),
        "by_condition_index": by_condition,
    }


def distribution_summary(values: torch.Tensor, higher_is_better: bool) -> dict[str, float]:
    values = values.to(torch.float64)
    count = int(values.numel())
    if count == 0:
        raise ValueError("cannot summarize an empty tensor")
    mean = values.mean()
    half_width = (
        1.96 * values.std(unbiased=True) / math.sqrt(count)
        if count > 1
        else torch.tensor(float("nan"), dtype=torch.float64)
    )
    quantile = 0.1 if higher_is_better else 0.9
    return {
        "mean": float(mean),
        "median": float(values.median()),
        "ci95_low": float(mean - half_width),
        "ci95_high": float(mean + half_width),
        "worst_decile": float(torch.quantile(values, quantile)),
    }


def condition_gate(
    comparison: dict[str, Any],
    *,
    min_mean_skill_score: float,
    min_ci95_low: float,
    require_every_condition_positive: bool,
) -> dict[str, Any]:
    """应用预声明门槛；只返回机械判断，不扩大科学结论。"""
    overall = comparison["skill_score"]
    condition_means = {
        key: value["skill_score"]["mean"]
        for key, value in comparison["by_condition_index"].items()
    }
    every_condition_positive = all(value > 0 for value in condition_means.values())
    passed = (
        overall["mean"] >= min_mean_skill_score
        and overall["ci95_low"] > min_ci95_low
        and (every_condition_positive or not require_every_condition_positive)
    )
    return {
        "validation_gate": "PASS" if passed else "FAIL",
        "min_mean_skill_score": min_mean_skill_score,
        "min_ci95_low": min_ci95_low,
        "require_every_condition_positive": require_every_condition_positive,
        "every_condition_positive": every_condition_positive,
        "condition_mean_skill_scores": condition_means,
    }


def _disturbance_from_observation(observation: torch.Tensor, num_modes: int) -> torch.Tensor:
    residual = observation[:, :num_modes]
    applied = observation[:, num_modes : 2 * num_modes]
    return residual - applied
