"""S2记忆无关ResUNet的动态回合监督数据与损失。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from src.simulation.config import S1EnvConfig
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.modes import project_phase_to_modes, synthesize_phase


@dataclass(frozen=True)
class WavefrontSupervisedData:
    """按完整回合生成的理想波前输入和可控模态相位目标。"""

    inputs: torch.Tensor
    target_phase: torch.Tensor
    episode_seed: torch.Tensor
    step_id: torch.Tensor


def generate_wavefront_supervised_data(
    config: S1EnvConfig,
    device: torch.device | str,
    base_seeds: list[int],
    frames_per_episode: int,
    random_action_std_rad: float,
    progress_callback: Callable[[int, int], None] | None = None,
) -> WavefrontSupervisedData:
    """生成动态帧；不同集合必须传入互不重叠的完整回合种子。"""
    if not base_seeds:
        raise ValueError("base_seeds must not be empty")
    if not 1 <= frames_per_episode <= config.episode_length:
        raise ValueError("frames_per_episode must be within the episode length")
    if random_action_std_rad < 0:
        raise ValueError("random_action_std_rad must be non-negative")

    all_inputs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_episode_seeds: list[torch.Tensor] = []
    all_steps: list[torch.Tensor] = []
    actual_device = torch.device(device)

    with torch.no_grad():
        total_steps = len(base_seeds) * frames_per_episode
        completed_steps = 0
        for base_seed in base_seeds:
            environment = AdaptiveOpticsEnv(config, actual_device)
            observation, _ = environment.reset(seed=base_seed)
            action_generator = torch.Generator(device=actual_device).manual_seed(
                base_seed + 10_000_000
            )
            inputs: list[torch.Tensor] = []
            targets: list[torch.Tensor] = []
            for _ in range(frames_per_episode):
                inputs.append(environment.ideal_wavefront_observation().cpu())
                residual_modal = observation[:, : config.num_modes]
                targets.append(
                    synthesize_phase(residual_modal, environment.basis).unsqueeze(1).cpu()
                )
                action = random_action_std_rad * torch.randn(
                    config.batch_size,
                    config.num_modes,
                    generator=action_generator,
                    device=actual_device,
                )
                observation, _, terminal, _, _ = environment.step(action)
                completed_steps += 1
                if progress_callback is not None:
                    progress_callback(completed_steps, total_steps)
                if terminal.all() and len(inputs) < frames_per_episode:
                    raise RuntimeError("environment terminated before requested frames were generated")

            input_tensor = torch.stack(inputs, dim=0)
            target_tensor = torch.stack(targets, dim=0)
            all_inputs.append(input_tensor.flatten(0, 1))
            all_targets.append(target_tensor.flatten(0, 1))
            episode_seeds = environment.episode_seeds.cpu()
            all_episode_seeds.append(
                episode_seeds.unsqueeze(0).expand(frames_per_episode, -1).reshape(-1)
            )
            all_steps.append(
                torch.arange(frames_per_episode, dtype=torch.int64)
                .unsqueeze(1)
                .expand(-1, config.batch_size)
                .reshape(-1)
            )

    return WavefrontSupervisedData(
        inputs=torch.cat(all_inputs),
        target_phase=torch.cat(all_targets),
        episode_seed=torch.cat(all_episode_seeds),
        step_id=torch.cat(all_steps),
    )


def resunet_phase_modal_loss(
    predicted_phase: torch.Tensor,
    target_phase: torch.Tensor,
    basis: torch.Tensor,
    pupil: torch.Tensor,
    modal_weight: float,
) -> dict[str, torch.Tensor]:
    """同时约束瞳面相位图和最终使用的低维模态。"""
    if predicted_phase.shape != target_phase.shape or predicted_phase.ndim != 4:
        raise ValueError("predicted and target phase must share [batch, 1, height, width]")
    if modal_weight < 0:
        raise ValueError("modal_weight must be non-negative")
    mask = pupil.to(predicted_phase.dtype).view(1, 1, *pupil.shape)
    phase_mse = ((predicted_phase - target_phase).square() * mask).sum() / (
        mask.sum() * predicted_phase.shape[0]
    )
    predicted_modal = project_phase_to_modes(predicted_phase[:, 0], basis, pupil)
    target_modal = project_phase_to_modes(target_phase[:, 0], basis, pupil)
    modal_mse = torch.mean((predicted_modal - target_modal).square())
    return {
        "total": phase_mse + modal_weight * modal_mse,
        "phase_mse": phase_mse,
        "modal_mse": modal_mse,
    }


def modal_prediction_metrics(
    predicted_phase: torch.Tensor,
    target_phase: torch.Tensor,
    basis: torch.Tensor,
    pupil: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """返回逐样本模态余弦相关和相对误差。"""
    predicted_modal = project_phase_to_modes(predicted_phase[:, 0], basis, pupil)
    target_modal = project_phase_to_modes(target_phase[:, 0], basis, pupil)
    denominator = target_modal.norm(dim=1).clamp_min(1e-12)
    return {
        "modal_cosine": torch.nn.functional.cosine_similarity(
            predicted_modal,
            target_modal,
            dim=1,
        ),
        "relative_modal_rmse": (predicted_modal - target_modal).norm(dim=1) / denominator,
    }
