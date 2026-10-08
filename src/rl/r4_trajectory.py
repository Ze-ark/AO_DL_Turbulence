"""R4完整回合采集与白名单片段访问；真值审计与模型数据分开保存。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

import torch

from src.rl.r4_control import NominalCalibration, R4Limits
from src.rl.r4_observation import R4Interface, PowerMeasurement, simulation_residual_proxy
from src.simulation.config import S1EnvConfig
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.hardware_effects import HardwareProfile

EPISODE_SCHEMA = "r4_complete_episode_v1"


def anchor_delta(frame: torch.Tensor, parameters: dict) -> torch.Tensor:
    """从白名单当前帧计算采集基座；不是旧真值跟踪控制器。"""
    residual, issued, estimate = frame[:, :21], frame[:, 21:42], frame[:, 42:63]
    return (-parameters["leak"] * issued - parameters["gain"] * residual
            + parameters["tracking_gain"] * (estimate - issued))


@torch.no_grad()
def collect_episode_batch(config: S1EnvConfig, profile: HardwareProfile, basis: torch.Tensor,
                          seed: int, collector: str, cfg: dict,
                          on_step: Callable[[], None]) -> tuple[dict, dict]:
    device = basis.device
    env = AdaptiveOpticsEnv(profile.environment_config(config), device,
                           profile.effects_config(), basis_override=basis)
    raw, _ = env.reset(seed=seed)
    data_cfg = cfg["data"]
    sensor = torch.Generator(device=device).manual_seed(seed + data_cfg["sensor_seed_offset"])
    excite = torch.Generator(device=device).manual_seed(seed + data_cfg["excitation_seed_offset"])
    interface = R4Interface(limits=R4Limits(modal_rad=config.modal_limit_rad),
                            calibration=NominalCalibration(**cfg["nominal_calibration"]))
    proxy = lambda x: simulation_residual_proxy(x, generator=sensor, noise_std_rad=profile.observation_noise_std_rad)
    initial = interface.reset(proxy(raw), episode_id=f"weather_{seed}")
    frames = [initial.features[:, -1].cpu()]
    commands, corrections, powers, done = [], [], [], []
    audit = {key: [] for key in ("applied_modal", "reward_power_in_bucket", "reward_strehl", "reward_phase_rmse", "violation_fraction")}
    correction = torch.zeros(config.batch_size, 11, device=device)
    for t in range(config.episode_length):
        if collector == "bounded_excitation" and t % data_cfg["excitation_hold_steps"] == 0:
            correction = (2 * torch.rand(correction.shape, device=device, generator=excite) - 1) * data_cfg["excitation_amplitude"]
        elif collector not in ("bounded_excitation", "anchor"):
            raise ValueError("unknown R4 collector")
        base = anchor_delta(interface.snapshot().features[:, -1], cfg["collector_anchor"])
        action = interface.issue(base, correction, step=t)
        raw, _, terminated, truncated, info = env.step(action.requested_delta_rad)
        transition = interface.observe_next(proxy(raw), step=t+1, power=PowerMeasurement(
            info["measured_power_in_bucket"], t, t+1))
        if not torch.allclose(action.requested_modal_rad, info["requested_modal"], atol=1e-6, rtol=0):
            raise RuntimeError("R4 collection command alignment failed")
        if bool(terminated.all()) != (t == config.episode_length-1) or bool(truncated.any()):
            raise RuntimeError("R4 unexpected episode termination")
        frames.append(transition.next_history.features[:, -1].cpu())
        commands.append(action.requested_delta_rad.cpu())
        corrections.append(action.normalized_correction.cpu())
        powers.append(transition.action_power.cpu())
        done.append(terminated.cpu())
        for key in audit:
            audit[key].append(info[key].cpu())
        on_step()
    stack = lambda values: torch.stack(values, dim=1).contiguous()
    data = dict(schema=EPISODE_SCHEMA, source="simulation_residual_proxy_not_holography",
                frames=stack(frames), commands=stack(commands), corrections=stack(corrections),
                powers=stack(powers), terminated=stack(done),
                power_valid=torch.ones(config.batch_size, config.episode_length, dtype=torch.bool),
                truncated=torch.zeros(config.batch_size, config.episode_length, dtype=torch.bool),
                weather_seeds=torch.arange(seed, seed+config.batch_size), dt_s=config.dt_s)
    for value in [*data.values(), *(stack(v) for v in audit.values())]:
        if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
            raise RuntimeError("R4 collection non-finite data")
    return data, {k: stack(v) for k, v in audit.items()}


class EpisodeStore:
    """只接收模型白名单；连续片段永不跨回合。可在CPU存储、CUDA训练。"""
    def __init__(self, data: dict[str, torch.Tensor]):
        keys = ("frames", "commands", "corrections", "powers", "weather_seeds")
        self.data = {k: data[k] for k in keys}
        self.count, self.steps = data["commands"].shape[:2]
        if data["frames"].shape != (self.count, self.steps+1, 79):
            raise ValueError("incomplete R4 episode")

    @classmethod
    def from_files(cls, paths: list[Path]) -> "EpisodeStore":
        rows = []
        for path in paths:
            item = torch.load(path, map_location="cpu", weights_only=True)
            if item["schema"] != EPISODE_SCHEMA or item["source"] != "simulation_residual_proxy_not_holography":
                raise ValueError("R4 source/schema mismatch")
            term = item["terminated"]
            if (bool(term[:, :-1].any()) or not bool(term[:, -1].all())
                    or bool(item["truncated"].any()) or not bool(item["power_valid"].all())):
                raise ValueError("R4 incomplete or invalid episode labels")
            for key in ("frames", "commands", "corrections", "powers"):
                if not bool(torch.isfinite(item[key]).all()):
                    raise ValueError("R4 non-finite trajectory")
            if not torch.equal(item["frames"][:, 1:, 74], item["powers"]):
                raise ValueError("R4 power alignment mismatch")
            rows.append(item)
        return cls({k: torch.cat([r[k] for r in rows]) for k in
                    ("frames", "commands", "corrections", "powers", "weather_seeds")})

    def windows(self, episodes: torch.Tensor, starts: torch.Tensor, horizon: int,
                device: torch.device) -> dict[str, torch.Tensor]:
        episodes, starts = episodes.cpu().long(), starts.cpu().long()
        if (horizon < 1 or bool((starts < 0).any()) or bool((starts+horizon > self.steps).any())
                or bool((episodes < 0).any()) or bool((episodes >= self.count).any())):
            raise ValueError("R4 window crosses episode boundary")
        past = starts[:, None] + torch.arange(-7, 1)[None]
        valid = past >= 0
        history = self.data["frames"][episodes[:, None], past.clamp_min(0)].clone()
        history[~valid] = 0
        future = starts[:, None] + torch.arange(horizon)[None]
        result = dict(history=history, valid=valid,
            commands=self.data["commands"][episodes[:, None], future],
            corrections=self.data["corrections"][episodes[:, None], future],
            target_residual=self.data["frames"][episodes[:, None], future+1, :21],
            target_power=self.data["powers"][episodes[:, None], future])
        return {k: v.to(device) for k, v in result.items()}

    def bootstrap_pool(self, seed: int) -> torch.Tensor:
        """按完整天气重采样，保留同天气全部硬件/采集分支。"""
        weather = self.data["weather_seeds"]
        unique = weather.unique(sorted=True)
        g = torch.Generator().manual_seed(seed)
        chosen = unique[torch.randint(len(unique), (len(unique),), generator=g)]
        return torch.cat([torch.nonzero(weather == w).flatten() for w in chosen])

    def sample(self, pool: torch.Tensor, batch: int, horizon: int,
               generator: torch.Generator, device: torch.device) -> dict[str, torch.Tensor]:
        episodes = pool[torch.randint(len(pool), (batch,), generator=generator)]
        starts = torch.randint(self.steps-horizon+1, (batch,), generator=generator)
        return self.windows(episodes, starts, horizon, device)
