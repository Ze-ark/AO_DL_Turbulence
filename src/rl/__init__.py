"""S4及后续阶段的强化学习组件。

库模块导入时不会创建环境、占用CUDA或启动训练。
"""

from src.rl.residual_control import ResidualTrackingController
from src.rl.residual_sac import ResidualSacAgent, SacConfig, TransitionReplayBuffer

__all__ = [
    "ResidualSacAgent",
    "ResidualTrackingController",
    "SacConfig",
    "TransitionReplayBuffer",
]
