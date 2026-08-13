"""不依赖Gym的残差Soft Actor-Critic核心实现。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as functional


@dataclass(frozen=True)
class SacConfig:
    """SAC优化参数；动作始终表示归一化到[-1, 1]的残差请求。"""

    state_size: int
    action_size: int
    hidden_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    initial_alpha: float = 0.05
    target_entropy: float | None = None
    log_std_min: float = -5.0
    log_std_max: float = 1.0

    def validate(self) -> None:
        if min(self.state_size, self.action_size, self.hidden_size) <= 0:
            raise ValueError("network dimensions must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if not 0 < self.tau <= 1:
            raise ValueError("tau must be in (0, 1]")
        if self.initial_alpha <= 0:
            raise ValueError("initial_alpha must be positive")
        if self.log_std_max <= self.log_std_min:
            raise ValueError("log_std_max must exceed log_std_min")


class SquashedGaussianActor(nn.Module):
    """输出经过tanh限制的随机动作，零均值初始化从传统控制器附近开始。"""

    def __init__(self, config: SacConfig) -> None:
        super().__init__()
        config.validate()
        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max
        self.backbone = nn.Sequential(
            nn.Linear(config.state_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(config.hidden_size, config.action_size)
        self.log_std_head = nn.Linear(config.hidden_size, config.action_size)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, -2.0)

    def distribution_parameters(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution_parameters(state)
        distribution = Normal(mean, log_std.exp())
        pre_tanh = distribution.rsample()
        action = torch.tanh(pre_tanh)
        correction = torch.log(1 - action.square() + 1e-6)
        log_probability = (distribution.log_prob(pre_tanh) - correction).sum(
            dim=-1,
            keepdim=True,
        )
        return action, log_probability

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        mean, _ = self.distribution_parameters(state)
        return torch.tanh(mean)


class QNetwork(nn.Module):
    """评价一个状态和残差动作组合的长期回报。"""

    def __init__(self, config: SacConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.state_size + config.action_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((state, action), dim=-1))


class TransitionReplayBuffer:
    """按单帧转移保存经验；默认放在CPU，采样后再送往训练CUDA。"""

    def __init__(
        self,
        capacity: int,
        state_size: int,
        action_size: int,
        *,
        storage_device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        if min(capacity, state_size, action_size) <= 0:
            raise ValueError("buffer dimensions must be positive")
        self.capacity = capacity
        self.state_size = state_size
        self.action_size = action_size
        self.storage_device = torch.device(storage_device)
        self.states = torch.empty(capacity, state_size, device=self.storage_device)
        self.actions = torch.empty(capacity, action_size, device=self.storage_device)
        self.rewards = torch.empty(capacity, 1, device=self.storage_device)
        self.next_states = torch.empty(capacity, state_size, device=self.storage_device)
        self.dones = torch.empty(capacity, 1, device=self.storage_device)
        self.position = 0
        self.size = 0
        self.generator = torch.Generator(device=self.storage_device).manual_seed(seed)

    def add_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        batch = states.shape[0]
        if states.shape != (batch, self.state_size):
            raise ValueError("states have the wrong shape")
        if actions.shape != (batch, self.action_size):
            raise ValueError("actions have the wrong shape")
        if next_states.shape != states.shape:
            raise ValueError("next_states have the wrong shape")
        rewards = rewards.reshape(batch, 1)
        dones = dones.reshape(batch, 1)
        if batch > self.capacity:
            states = states[-self.capacity :]
            actions = actions[-self.capacity :]
            rewards = rewards[-self.capacity :]
            next_states = next_states[-self.capacity :]
            dones = dones[-self.capacity :]
            batch = self.capacity

        indices = (torch.arange(batch, device=self.storage_device) + self.position) % self.capacity
        self.states[indices] = states.detach().to(self.storage_device)
        self.actions[indices] = actions.detach().to(self.storage_device)
        self.rewards[indices] = rewards.detach().to(self.storage_device)
        self.next_states[indices] = next_states.detach().to(self.storage_device)
        self.dones[indices] = dones.detach().to(self.storage_device, torch.float32)
        self.position = (self.position + batch) % self.capacity
        self.size = min(self.size + batch, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        if batch_size <= 0 or self.size < batch_size:
            raise ValueError("replay buffer does not contain a full batch")
        indices = torch.randint(
            self.size,
            (batch_size,),
            generator=self.generator,
            device=self.storage_device,
        )
        return (
            self.states[indices].to(device),
            self.actions[indices].to(device),
            self.rewards[indices].to(device),
            self.next_states[indices].to(device),
            self.dones[indices].to(device),
        )

    def __len__(self) -> int:
        return self.size


class ResidualSacAgent:
    """双评价网络、自动熵温度和软目标更新组成的SAC训练器。"""

    def __init__(self, config: SacConfig, device: torch.device) -> None:
        config.validate()
        if device.type != "cuda" and not _unit_test_mode_allowed():
            raise RuntimeError("formal Residual SAC training requires CUDA")
        self.config = config
        self.device = device
        self.actor = SquashedGaussianActor(config).to(device)
        self.q1 = QNetwork(config).to(device)
        self.q2 = QNetwork(config).to(device)
        self.target_q1 = deepcopy(self.q1).to(device).eval()
        self.target_q2 = deepcopy(self.q2).to(device).eval()
        for parameter in (*self.target_q1.parameters(), *self.target_q2.parameters()):
            parameter.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.learning_rate
        )
        self.q_optimizer = torch.optim.Adam(
            (*self.q1.parameters(), *self.q2.parameters()),
            lr=config.learning_rate,
        )
        self.log_alpha = torch.tensor(
            math.log(config.initial_alpha),
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=config.learning_rate
        )
        self.target_entropy = (
            -float(config.action_size)
            if config.target_entropy is None
            else float(config.target_entropy)
        )
        self.update_count = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, state: torch.Tensor, *, deterministic: bool) -> torch.Tensor:
        state = state.to(self.device)
        if deterministic:
            return self.actor.deterministic(state)
        return self.actor.sample(state)[0]

    def update(
        self,
        batch: tuple[torch.Tensor, ...],
    ) -> dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        with torch.no_grad():
            next_actions, next_log_probabilities = self.actor.sample(next_states)
            next_q = torch.minimum(
                self.target_q1(next_states, next_actions),
                self.target_q2(next_states, next_actions),
            ) - self.alpha.detach() * next_log_probabilities
            target = rewards + self.config.gamma * (1 - dones) * next_q

        q1_prediction = self.q1(states, actions)
        q2_prediction = self.q2(states, actions)
        q1_loss = functional.mse_loss(q1_prediction, target)
        q2_loss = functional.mse_loss(q2_prediction, target)
        critic_loss = q1_loss + q2_loss
        self.q_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.q_optimizer.step()

        for parameter in (*self.q1.parameters(), *self.q2.parameters()):
            parameter.requires_grad_(False)
        sampled_actions, log_probabilities = self.actor.sample(states)
        sampled_q = torch.minimum(
            self.q1(states, sampled_actions),
            self.q2(states, sampled_actions),
        )
        actor_loss = (self.alpha.detach() * log_probabilities - sampled_q).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        for parameter in (*self.q1.parameters(), *self.q2.parameters()):
            parameter.requires_grad_(True)

        alpha_loss = -(
            self.log_alpha * (log_probabilities.detach() + self.target_entropy)
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)
        self.update_count += 1
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha_loss": float(alpha_loss.detach()),
            "alpha": float(self.alpha.detach()),
            "mean_q": float(sampled_q.detach().mean()),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for source_parameter, target_parameter in zip(
                source.parameters(), target.parameters(), strict=True
            ):
                target_parameter.lerp_(source_parameter, self.config.tau)

    def checkpoint(self) -> dict[str, Any]:
        return {
            "algorithm": "residual_sac",
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "update_count": self.update_count,
        }

    def load_actor(self, checkpoint: dict[str, Any]) -> None:
        self.actor.load_state_dict(checkpoint["actor"])


def _unit_test_mode_allowed() -> bool:
    """仅允许显式pytest进程在CPU上检查小型确定性网络。"""
    import sys

    return "pytest" in sys.modules
