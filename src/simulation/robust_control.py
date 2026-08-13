"""S4-A复杂动态闭环条件与预声明鲁棒性门槛。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

from src.simulation.config import S1EnvConfig


@dataclass(frozen=True)
class RobustnessCondition:
    """一个闭环动态条件；每个基础种子展开为一批独立完整回合。"""

    identifier: str
    regime: str
    base_seed: int
    wind_speed_mps: float
    wind_direction_deg: float
    frozen_flow_rho: float = 1.0
    wind_speed_modulation_fraction: float = 0.0
    wind_direction_modulation_deg: float = 0.0
    wind_modulation_period_frames: int = 0
    wind_modulation_phase_deg: float = 0.0
    observation_noise_std_rad: float = 0.0
    slm_delay_frames: int = 2

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "RobustnessCondition":
        return cls(
            identifier=str(values["id"]),
            regime=str(values["regime"]),
            base_seed=int(values["base_seed"]),
            wind_speed_mps=float(values["wind_speed_mps"]),
            wind_direction_deg=float(values["wind_direction_deg"]),
            frozen_flow_rho=float(values.get("frozen_flow_rho", 1.0)),
            wind_speed_modulation_fraction=float(
                values.get("wind_speed_modulation_fraction", 0.0)
            ),
            wind_direction_modulation_deg=float(
                values.get("wind_direction_modulation_deg", 0.0)
            ),
            wind_modulation_period_frames=int(
                values.get("wind_modulation_period_frames", 0)
            ),
            wind_modulation_phase_deg=float(values.get("wind_modulation_phase_deg", 0.0)),
            observation_noise_std_rad=float(values.get("observation_noise_std_rad", 0.0)),
            slm_delay_frames=int(values.get("slm_delay_frames", 2)),
        )

    def environment_config(self, base: S1EnvConfig) -> S1EnvConfig:
        if self.observation_noise_std_rad < 0:
            raise ValueError("observation_noise_std_rad must be non-negative")
        if self.slm_delay_frames < 0:
            raise ValueError("slm_delay_frames must be non-negative")
        config = replace(
            base,
            wind_speed_mps=self.wind_speed_mps,
            wind_direction_deg=self.wind_direction_deg,
            frozen_flow_rho=self.frozen_flow_rho,
            wind_speed_modulation_fraction=self.wind_speed_modulation_fraction,
            wind_direction_modulation_deg=self.wind_direction_modulation_deg,
            wind_modulation_period_frames=self.wind_modulation_period_frames,
            wind_modulation_phase_deg=self.wind_modulation_phase_deg,
            slm_delay_frames=self.slm_delay_frames,
        )
        config.validate()
        return config


def closed_loop_robustness_gate(
    controller_summary: dict[str, Any],
    condition_records: Sequence[dict[str, Any]],
    *,
    min_power_delta_ci95_low: float,
    min_strehl_delta_ci95_low: float,
    max_phase_rmse_delta_ci95_high: float,
    max_violation_fraction: float,
    require_every_condition_power_positive: bool,
) -> dict[str, Any]:
    """检查候选控制器是否稳定优于不校正且不以违规换性能。"""
    paired = controller_summary["paired_delta_vs_no_correction"]
    power_pass = paired["power_in_bucket"]["ci95_low"] > min_power_delta_ci95_low
    strehl_pass = paired["strehl"]["ci95_low"] > min_strehl_delta_ci95_low
    phase_pass = paired["phase_rmse"]["ci95_high"] < max_phase_rmse_delta_ci95_high
    violation_pass = (
        controller_summary["violation_fraction"]["mean"] <= max_violation_fraction
    )
    matching = [
        item
        for item in condition_records
        if item["controller"] == controller_summary["controller"]
    ]
    if not matching:
        raise ValueError("condition records do not contain the selected controller")
    every_condition_power_positive = all(
        float(item["power_delta_vs_no_correction"]) > 0 for item in matching
    )
    every_condition_safe = all(
        float(item["violation_fraction"]) <= max_violation_fraction
        for item in matching
    )
    passed = (
        power_pass
        and strehl_pass
        and phase_pass
        and violation_pass
        and every_condition_safe
        and (
            every_condition_power_positive
            or not require_every_condition_power_positive
        )
    )
    return {
        "validation_gate": "PASS" if passed else "FAIL",
        "selected_controller": controller_summary["controller"],
        "power_ci_pass": power_pass,
        "strehl_ci_pass": strehl_pass,
        "phase_rmse_ci_pass": phase_pass,
        "overall_violation_pass": violation_pass,
        "every_condition_power_positive": every_condition_power_positive,
        "every_condition_safe": every_condition_safe,
        "thresholds": {
            "min_power_delta_ci95_low": min_power_delta_ci95_low,
            "min_strehl_delta_ci95_low": min_strehl_delta_ci95_low,
            "max_phase_rmse_delta_ci95_high": max_phase_rmse_delta_ci95_high,
            "max_violation_fraction": max_violation_fraction,
            "require_every_condition_power_positive": (
                require_every_condition_power_positive
            ),
        },
    }


def sealed_closed_loop_gate(
    controller_summary: dict[str, Any],
    reference_summary: dict[str, Any],
    condition_records: Sequence[dict[str, Any]],
    *,
    min_power_delta_ci95_low: float,
    min_mean_relative_power_gain: float,
    min_strehl_delta_ci95_low: float,
    max_phase_rmse_delta_ci95_high: float,
    max_violation_fraction: float,
    require_every_condition_power_positive: bool,
) -> dict[str, Any]:
    """S4-B封存门槛：在方向性门槛外增加预声明的实际桶功率增益。"""
    if min_mean_relative_power_gain < 0:
        raise ValueError("min_mean_relative_power_gain must be non-negative")
    reference_power = float(reference_summary["power_in_bucket"]["mean"])
    if reference_power <= 0:
        raise ValueError("reference power_in_bucket mean must be positive")
    base_gate = closed_loop_robustness_gate(
        controller_summary,
        condition_records,
        min_power_delta_ci95_low=min_power_delta_ci95_low,
        min_strehl_delta_ci95_low=min_strehl_delta_ci95_low,
        max_phase_rmse_delta_ci95_high=max_phase_rmse_delta_ci95_high,
        max_violation_fraction=max_violation_fraction,
        require_every_condition_power_positive=require_every_condition_power_positive,
    )
    candidate_power = float(controller_summary["power_in_bucket"]["mean"])
    relative_power_gain = (candidate_power - reference_power) / reference_power
    relative_power_pass = relative_power_gain >= min_mean_relative_power_gain
    passed = base_gate["validation_gate"] == "PASS" and relative_power_pass
    return {
        **base_gate,
        "validation_gate": "PASS" if passed else "FAIL",
        "mean_relative_power_gain": relative_power_gain,
        "relative_power_gain_pass": relative_power_pass,
        "thresholds": {
            **base_gate["thresholds"],
            "min_mean_relative_power_gain": min_mean_relative_power_gain,
        },
    }
