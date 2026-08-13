"""S1 时序自适应光学仿真组件。"""

from src.simulation.config import S1EnvConfig, load_s1_config
from src.simulation.controller_selection import select_robust_controller
from src.simulation.complex_dynamics import (
    fit_regime_conditioned_ridge,
    initialization_seed_gate,
    student_t_summary,
)
from src.simulation.controllers import (
    DirectProjectionController,
    LeakyIntegratorController,
    LinearPredictiveController,
    ModalController,
    NoCorrectionController,
    RidgePredictiveController,
    TrackingLeakyIntegratorController,
    ResUNetModalController,
    make_controller,
)
from src.simulation.env import AdaptiveOpticsEnv
from src.simulation.hardware_effects import (
    HardwareAwareSlmModel,
    HardwareEffectsConfig,
    HardwareProfile,
    apply_registration_error,
    noisy_power_measurement,
)
from src.simulation.robust_control import (
    RobustnessCondition,
    closed_loop_robustness_gate,
    sealed_closed_loop_gate,
)
from src.simulation.temporal_dynamics import (
    DynamicsCondition,
    GRUModalDynamics,
    TemporalDynamicsData,
    condition_gate,
    constant_velocity_forecast,
    fit_ridge_autoregression,
    generate_temporal_dynamics_data,
    modal_normalization,
    paired_episode_comparison,
    ridge_autoregressive_forecast,
)

__all__ = [
    "AdaptiveOpticsEnv",
    "DirectProjectionController",
    "DynamicsCondition",
    "GRUModalDynamics",
    "HardwareAwareSlmModel",
    "HardwareEffectsConfig",
    "HardwareProfile",
    "LeakyIntegratorController",
    "LinearPredictiveController",
    "ModalController",
    "NoCorrectionController",
    "RidgePredictiveController",
    "TrackingLeakyIntegratorController",
    "ResUNetModalController",
    "RobustnessCondition",
    "S1EnvConfig",
    "TemporalDynamicsData",
    "condition_gate",
    "apply_registration_error",
    "closed_loop_robustness_gate",
    "constant_velocity_forecast",
    "fit_ridge_autoregression",
    "fit_regime_conditioned_ridge",
    "generate_temporal_dynamics_data",
    "load_s1_config",
    "select_robust_controller",
    "make_controller",
    "modal_normalization",
    "noisy_power_measurement",
    "initialization_seed_gate",
    "paired_episode_comparison",
    "ridge_autoregressive_forecast",
    "sealed_closed_loop_gate",
    "student_t_summary",
]
