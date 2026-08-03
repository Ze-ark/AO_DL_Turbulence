"""在 CUDA 上运行一段 S1 动态环境冒烟实验。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime import resolve_device
from src.simulation import AdaptiveOpticsEnv, load_s1_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/environment/s1_taylor_v1.yaml")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output", default="outputs/s1_smoke/summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, requested_device = load_s1_config(args.config)
    device = resolve_device(requested_device)
    environment = AdaptiveOpticsEnv(config, device)
    observation, initial_info = environment.reset(seed=config.seed)
    initial_strehl = initial_info["strehl"].mean().item()
    rewards: list[float] = []
    strehl_values: list[float] = []

    with torch.no_grad():
        for _ in range(min(args.steps, config.episode_length)):
            residual_modal = observation[:, : config.num_modes]
            action = (-0.25 * residual_modal).clamp(-0.2, 0.2)
            observation, reward, terminal, _, info = environment.step(action)
            rewards.append(reward.mean().item())
            strehl_values.append(info["strehl"].mean().item())
            if terminal.all():
                break

    summary = {
        "stage": "S1",
        "status": "smoke_only",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "seed": config.seed,
        "batch_size": config.batch_size,
        "steps": len(rewards),
        "observation_size": config.observation_size,
        "action_size": config.num_modes,
        "initial_mean_strehl": initial_strehl,
        "final_mean_strehl": strehl_values[-1],
        "mean_reward": sum(rewards) / len(rewards),
        "note": "Diagnostic proportional modal actions only; this is not RL training or a baseline result.",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
