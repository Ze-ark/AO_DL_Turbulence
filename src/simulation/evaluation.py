"""S2 控制器的配对回合评估与轨迹导出。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import h5py
import numpy as np
import torch

from src.simulation.config import S1EnvConfig
from src.simulation.controllers import ModalController
from src.simulation.env import AdaptiveOpticsEnv


@dataclass(frozen=True)
class ControllerRollout:
    """一个控制器在一批完整回合上的逐步记录。"""

    controller_name: str
    seed: int
    episode_seeds: torch.Tensor
    observations: torch.Tensor
    requested_modal: torch.Tensor
    applied_modal: torch.Tensor
    rewards: torch.Tensor
    strehl: torch.Tensor
    power_in_bucket: torch.Tensor
    phase_rmse: torch.Tensor
    action_cost: torch.Tensor
    violation_fraction: torch.Tensor
    terminal: torch.Tensor
    action_latency_ms: torch.Tensor
    oracle_strehl: torch.Tensor | None = None
    oracle_power_in_bucket: torch.Tensor | None = None
    oracle_phase_rmse: torch.Tensor | None = None


def run_controller_rollout(
    config: S1EnvConfig,
    device: torch.device | str,
    controller: ModalController,
    seed: int,
    steps: int | None = None,
    include_oracle_upper_bound: bool = False,
) -> ControllerRollout:
    """在固定种子回合上运行控制器，所有返回张量均移到 CPU。"""
    environment = AdaptiveOpticsEnv(config, device)
    observation, _ = environment.reset(seed=seed)
    controller.reset(config.batch_size, environment.device, observation.dtype)
    max_steps = config.episode_length if steps is None else min(steps, config.episode_length)
    if max_steps <= 0:
        raise ValueError("steps must be positive")

    observations = [observation.detach().cpu()]
    requested_modal: list[torch.Tensor] = []
    applied_modal: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    strehl: list[torch.Tensor] = []
    power_in_bucket: list[torch.Tensor] = []
    phase_rmse: list[torch.Tensor] = []
    action_cost: list[torch.Tensor] = []
    violation_fraction: list[torch.Tensor] = []
    terminals: list[torch.Tensor] = []
    action_latency_ms: list[float] = []
    oracle_strehl: list[torch.Tensor] = []
    oracle_power: list[torch.Tensor] = []
    oracle_phase_rmse: list[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(max_steps):
            if include_oracle_upper_bound:
                upper = environment.oracle_modal_upper_bound()
                oracle_strehl.append(upper["strehl"].detach().cpu())
                oracle_power.append(upper["power_in_bucket"].detach().cpu())
                oracle_phase_rmse.append(upper["phase_rmse"].detach().cpu())

            _synchronize_if_cuda(environment.device)
            start = perf_counter()
            wavefront_observation = environment.ideal_wavefront_observation()
            action = controller.action(observation, wavefront_observation)
            _synchronize_if_cuda(environment.device)
            elapsed_ms = (perf_counter() - start) * 1000

            observation, reward, terminal, _, info = environment.step(action)
            observations.append(observation.detach().cpu())
            requested_modal.append(info["requested_modal"].detach().cpu())
            applied_modal.append(info["applied_modal"].detach().cpu())
            rewards.append(reward.detach().cpu())
            strehl.append(info["reward_strehl"].detach().cpu())
            power_in_bucket.append(info["reward_power_in_bucket"].detach().cpu())
            phase_rmse.append(info["reward_phase_rmse"].detach().cpu())
            action_cost.append(info["action_cost"].detach().cpu())
            violation_fraction.append(info["violation_fraction"].detach().cpu())
            terminals.append(terminal.detach().cpu())
            action_latency_ms.append(elapsed_ms)
            if terminal.all():
                break

    def stack_steps(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values, dim=1)

    return ControllerRollout(
        controller_name=controller.name,
        seed=seed,
        episode_seeds=environment.episode_seeds.detach().cpu(),
        observations=stack_steps(observations),
        requested_modal=stack_steps(requested_modal),
        applied_modal=stack_steps(applied_modal),
        rewards=stack_steps(rewards),
        strehl=stack_steps(strehl),
        power_in_bucket=stack_steps(power_in_bucket),
        phase_rmse=stack_steps(phase_rmse),
        action_cost=stack_steps(action_cost),
        violation_fraction=stack_steps(violation_fraction),
        terminal=stack_steps(terminals),
        action_latency_ms=torch.tensor(action_latency_ms, dtype=torch.float64),
        oracle_strehl=stack_steps(oracle_strehl) if oracle_strehl else None,
        oracle_power_in_bucket=stack_steps(oracle_power) if oracle_power else None,
        oracle_phase_rmse=stack_steps(oracle_phase_rmse) if oracle_phase_rmse else None,
    )


def summarize_rollouts(rollouts: list[ControllerRollout]) -> dict[str, Any]:
    """按回合而不是按单帧计算均值、95%区间和最差十分位。"""
    if not rollouts:
        raise ValueError("rollouts must not be empty")
    name = rollouts[0].controller_name
    if any(item.controller_name != name for item in rollouts):
        raise ValueError("all rollouts must belong to the same controller")

    metrics = {
        "strehl": torch.cat([item.strehl.mean(dim=1) for item in rollouts]),
        "power_in_bucket": torch.cat(
            [item.power_in_bucket.mean(dim=1) for item in rollouts]
        ),
        "phase_rmse": torch.cat([item.phase_rmse.mean(dim=1) for item in rollouts]),
        "reward": torch.cat([item.rewards.mean(dim=1) for item in rollouts]),
        "action_cost": torch.cat([item.action_cost.mean(dim=1) for item in rollouts]),
        "violation_fraction": torch.cat(
            [item.violation_fraction.mean(dim=1) for item in rollouts]
        ),
    }
    result: dict[str, Any] = {
        "controller": name,
        "episodes": int(metrics["strehl"].numel()),
        "steps_per_episode": int(rollouts[0].strehl.shape[1]),
        "episode_seeds": torch.cat([item.episode_seeds for item in rollouts]).tolist(),
    }
    for metric_name, values in metrics.items():
        higher_is_better = metric_name in {"strehl", "power_in_bucket", "reward"}
        result[metric_name] = _distribution_summary(values, higher_is_better)

    latencies = torch.cat([item.action_latency_ms for item in rollouts])
    result["action_latency_ms"] = {
        "mean": float(latencies.mean()),
        "median": float(latencies.median()),
        "maximum": float(latencies.max()),
    }
    return result


def summarize_oracle_upper_bound(rollouts: list[ControllerRollout]) -> dict[str, Any]:
    """汇总瞬时理想模态投影上限；它不是可部署控制器。"""
    if not rollouts or any(item.oracle_strehl is None for item in rollouts):
        raise ValueError("rollouts do not contain oracle upper-bound metrics")
    metrics = {
        "strehl": torch.cat([item.oracle_strehl.mean(dim=1) for item in rollouts]),
        "power_in_bucket": torch.cat(
            [item.oracle_power_in_bucket.mean(dim=1) for item in rollouts]
        ),
        "phase_rmse": torch.cat([item.oracle_phase_rmse.mean(dim=1) for item in rollouts]),
    }
    result: dict[str, Any] = {
        "controller": "oracle_modal_upper_bound",
        "kind": "diagnostic_upper_bound_not_deployable",
        "episodes": int(metrics["strehl"].numel()),
        "steps_per_episode": int(rollouts[0].strehl.shape[1]),
        "episode_seeds": torch.cat([item.episode_seeds for item in rollouts]).tolist(),
    }
    for metric_name, values in metrics.items():
        result[metric_name] = _distribution_summary(
            values,
            higher_is_better=metric_name != "phase_rmse",
        )
    return result


def paired_delta(
    candidate_rollouts: list[ControllerRollout],
    reference_rollouts: list[ControllerRollout],
    metric: str,
) -> dict[str, float]:
    """按相同回合种子计算候选控制器相对参照的配对差值。"""
    candidate = _episode_metric(candidate_rollouts, metric)
    reference = _episode_metric(reference_rollouts, metric)
    candidate_seeds = torch.cat([item.episode_seeds for item in candidate_rollouts])
    reference_seeds = torch.cat([item.episode_seeds for item in reference_rollouts])
    if not torch.equal(candidate_seeds, reference_seeds):
        raise ValueError("paired comparison requires identical episode seeds")
    delta = candidate - reference
    summary = _distribution_summary(delta, higher_is_better=True)
    return {
        "mean": summary["mean"],
        "ci95_low": summary["ci95_low"],
        "ci95_high": summary["ci95_high"],
    }


def export_rollout_h5(
    rollout: ControllerRollout,
    path: str | Path,
    config: S1EnvConfig,
    metadata: dict[str, str] | None = None,
) -> None:
    """按项目动态轨迹契约保存单批回合，不覆盖未授权文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["stage"] = "S2"
        handle.attrs["controller"] = rollout.controller_name
        handle.attrs["observation_kind"] = "oracle_modal_features"
        handle.attrs["seed"] = rollout.seed
        handle.attrs["r0_m"] = config.r0_m
        handle.attrs["wind_speed_mps"] = config.wind_speed_mps
        handle.attrs["wind_direction_deg"] = config.wind_direction_deg
        handle.attrs["dt_s"] = config.dt_s
        handle.attrs["slm_delay_frames"] = config.slm_delay_frames
        for key, value in (metadata or {}).items():
            handle.attrs[key] = value

        _write_tensor(handle, "observation/features", rollout.observations)
        _write_tensor(handle, "action/requested_modal", rollout.requested_modal)
        _write_tensor(handle, "action/applied_modal", rollout.applied_modal)
        _write_tensor(handle, "metric/reward", rollout.rewards)
        _write_tensor(handle, "metric/strehl", rollout.strehl)
        _write_tensor(handle, "metric/power_in_bucket", rollout.power_in_bucket)
        _write_tensor(handle, "metric/phase_residual_rmse", rollout.phase_rmse)
        _write_tensor(handle, "metric/action_cost", rollout.action_cost)
        _write_tensor(handle, "metric/violation_fraction", rollout.violation_fraction)
        _write_tensor(handle, "flags/terminal", rollout.terminal)
        _write_tensor(handle, "meta/episode_seed", rollout.episode_seeds)
        handle.create_dataset(
            "meta/step_id",
            data=np.arange(rollout.observations.shape[1], dtype=np.int64),
        )


def _episode_metric(rollouts: list[ControllerRollout], metric: str) -> torch.Tensor:
    field = {
        "strehl": "strehl",
        "power_in_bucket": "power_in_bucket",
        "phase_rmse": "phase_rmse",
        "reward": "rewards",
    }.get(metric)
    if field is None:
        raise ValueError(f"unsupported paired metric: {metric}")
    return torch.cat([getattr(item, field).mean(dim=1) for item in rollouts])


def _distribution_summary(values: torch.Tensor, higher_is_better: bool) -> dict[str, float]:
    values = values.to(torch.float64)
    count = int(values.numel())
    mean = values.mean()
    if count > 1:
        half_width = 1.96 * values.std(unbiased=True) / count**0.5
    else:
        half_width = torch.tensor(float("nan"), dtype=torch.float64)
    quantile = 0.1 if higher_is_better else 0.9
    return {
        "mean": float(mean),
        "median": float(values.median()),
        "ci95_low": float(mean - half_width),
        "ci95_high": float(mean + half_width),
        "worst_decile": float(torch.quantile(values, quantile)),
    }


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _write_tensor(handle: h5py.File, key: str, value: torch.Tensor) -> None:
    handle.create_dataset(key, data=value.detach().cpu().numpy(), compression="gzip")
