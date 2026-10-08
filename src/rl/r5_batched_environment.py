"""R5专用异构批量环境；保留v3逐档位随机流，不向策略暴露档位。"""
from __future__ import annotations

from dataclasses import replace
import torch

from src.rl.r4_observation import simulation_residual_proxy
from src.rl.r5_physics_adapter import BatchedDifferentiableSlm
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.hardware_effects import HardwareProfile, noisy_power_measurement
from src.simulation.config import S1EnvConfig


class R5BatchedEnvironment(AdaptiveOpticsEnv):
    def __init__(self, base: S1EnvConfig, device: torch.device, basis: torch.Tensor,
                 families: list[dict], profiles: list[HardwareProfile], sensor_seed_offset: int):
        self.profiles = profiles
        self.family_count = len(families)
        self.sensor_seed_offset = sensor_seed_offset
        if not families or not profiles:
            raise ValueError('families and profiles must not be empty')
        config = replace(base, batch_size=len(families) * len(profiles))
        super().__init__(config, device, basis_override=basis,
                         condition_batch=[family for _ in profiles for family in families])
        self.slm = BatchedDifferentiableSlm(
            config, [p.as_record() for p in profiles for _ in families], device)
        self._initialize_streams = False

    def reset(self, seed: int | None = None, turbulence_phase: torch.Tensor | None = None):
        if turbulence_phase is not None:
            raise ValueError('R5 parity environment requires seeded turbulence')
        self._base_seed = self.config.seed if seed is None else seed
        self._initialize_streams = True
        self.sensor_generators = [torch.Generator(device=self.device).manual_seed(
            self._base_seed + i * 1000 + self.sensor_seed_offset) for i in range(len(self.profiles))]
        self.power_generators = [torch.Generator(device=self.device).manual_seed(
            self._base_seed + i * 1000 + 60_000_000) for i in range(len(self.profiles))]
        return super().reset(seed=self._base_seed)

    def _new_phase_screens(self) -> torch.Tensor:
        if self._initialize_streams:
            seeds = [self._base_seed + i * 1000 + j
                     for i in range(len(self.profiles)) for j in range(self.family_count)]
            self.generators = [torch.Generator(device=self.device).manual_seed(s) for s in seeds]
            self.episode_seeds = torch.tensor(seeds, device=self.device, dtype=torch.int64)
            self._initialize_streams = False
        return super()._new_phase_screens()

    def step(self, action_delta: torch.Tensor):
        raw, reward, terminated, truncated, info = super().step(action_delta)
        chunks = info['reward_power_in_bucket'].reshape(len(self.profiles), self.family_count)
        noise = torch.cat([
            noisy_power_measurement(chunk, profile.power_noise_relative_std, generator)
            for chunk, profile, generator in zip(chunks, self.profiles, self.power_generators)])
        info['measured_power_in_bucket'] = noise
        return raw, reward, terminated, truncated, info

    def proxy(self, raw: torch.Tensor) -> torch.Tensor:
        chunks = raw.reshape(len(self.profiles), self.family_count, *raw.shape[1:])
        return torch.cat([simulation_residual_proxy(chunk, generator=generator,
                         noise_std_rad=profile.observation_noise_std_rad)
                         for chunk, profile, generator in zip(chunks,
                                                             self.profiles, self.sensor_generators)])
