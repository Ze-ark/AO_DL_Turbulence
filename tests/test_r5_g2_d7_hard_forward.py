"""G2-D7 硬前向诊断的确定性 CPU 单测；不执行正式仿真。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts import diagnose_s4_r5_g2_d7_hard_forward as d7
from src.rl.r4_observation import R4Interface
from src.rl.s4_training import _load_yaml, _project_path


def test_frozen_contract_and_streams() -> None:
    cfg = _load_yaml(_project_path(d7.CONFIG))
    d7._contract(cfg)
    for path, value in ((["deployment_scales"], [1.6, 1.9]),
                        (["boundary", "confirmation_access"], True),
                        (["data", "weather_count"], 3)):
        altered = deepcopy(cfg)
        cursor = altered
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = value
        with pytest.raises(ValueError, match="合同"):
            d7._contract(altered)
    assert d7.SPAN == pytest.approx(0.175)
    assert d7.stream_manifest(False)["weather_bases"] == [7_340_000, 7_340_010]
    assert d7.stream_manifest(True)["weather_bases"] == [7_360_000]
    d7._check_stream_isolation()


class _ConstantPolicy(torch.nn.Module):
    def forward(self, history: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        assert not torch.is_grad_enabled()
        raw = history.new_zeros((len(history), 11))
        raw[:, 0] = 0.2
        return raw


class _TinyDelayedEnv:
    def __init__(self, steps: int = 200) -> None:
        self.device = torch.device("cpu")
        self.basis = torch.zeros(21, 2, 2, dtype=torch.float64)
        self.config = SimpleNamespace(batch_size=1, episode_length=steps)
        self.queue = [torch.zeros(1, 21, dtype=torch.float64) for _ in range(2)]
        self.step_count = 0

    def proxy(self, raw: torch.Tensor) -> torch.Tensor:
        return raw[:, :21]

    def step(self, delta: torch.Tensor):
        delayed = self.queue.pop(0)
        self.queue.append(delta)
        physical = 0.7 + 0.25 * delayed[:, 10]
        measured = physical + 0.01
        self.step_count += 1
        done = torch.tensor([self.step_count == self.config.episode_length])
        info = {
            "measured_power_in_bucket": measured,
            "reward_power_in_bucket": physical,
            "violation_fraction": torch.zeros(1, dtype=torch.float64),
            "saturated_fraction": torch.zeros(1, dtype=torch.float64),
            "slew_limited_fraction": torch.zeros(1, dtype=torch.float64),
            "requested_modal": delta,
            "applied_modal": delayed,
        }
        raw = torch.zeros(1, 44, dtype=torch.float64)
        return raw, torch.zeros(1, dtype=torch.float64), done, torch.zeros_like(done), info


def test_hard_forward_uses_no_grad_and_exact_200_frame_decomposition() -> None:
    env = _TinyDelayedEnv(200)
    interface = R4Interface()
    interface.reset(torch.zeros(1, 21, dtype=torch.float64), episode_id="cpu-unit")
    counter = SimpleNamespace(steps=0)

    def tick() -> None:
        counter.steps += 1

    result = d7._hard_rollout((env, interface), _ConstantPolicy(), scale_value=1.8375,
                              action_weight=.01, smooth_weight=.001,
                              progress=SimpleNamespace(tick=tick))
    assert counter.steps == 200
    assert len(result["per_frame_batch_mean"]["training_objective"]) == 200
    assert result["per_frame_batch_mean"]["physical_power"][:2] == [0.7, 0.7]
    assert result["per_frame_batch_mean"]["physical_power"][2] != 0.7
    mean = result["group_mean"]
    assert mean["measured_power"] != mean["physical_power"]
    assert result["full_training_objective"] == pytest.approx(
        mean["measured_power"] - mean["action_penalty"] - mean["smooth_penalty"], abs=1e-12)


def _triplet(j_surrogate: float) -> tuple[dict, dict, dict, dict]:
    def make(scale: float, physical: float, measured: float,
             action: float, smooth: float) -> dict:
        return {
            "hardware_condition": "nominal_clone", "weather_seed": 7_340_000,
            "member": 0, "controller": d7.ARMS[0], "source_state_sha256": "same",
            "deployment_scale": scale,
            "group_mean": {**{name: 0.0 for name in d7.METRICS},
                           "physical_power": physical, "measured_power": measured,
                           "action_penalty": action, "smooth_penalty": smooth,
                           "training_objective": measured - action - smooth},
        }
    minus = make(d7.SCALES[0], .70, .70, .010, .001)
    center = make(d7.CENTER, .705, .7025, .015, .001)
    center["d_objective_d_deployment_scale"] = j_surrogate
    plus = make(d7.SCALES[1], .71, .705, .020, .001)
    cfg = _load_yaml(_project_path(d7.CONFIG))
    return minus, center, plus, cfg


def test_centered_finite_difference_and_penalty_reversal_are_not_ste_values() -> None:
    minus, center, plus, cfg = _triplet(-.03)
    result = d7._compare_triplet(minus, center, plus, cfg)
    assert result["hard_centered_derivative"]["training_objective"] == pytest.approx(
        (plus["group_mean"]["training_objective"]
         - minus["group_mean"]["training_objective"]) / .175)
    assert result["d6_surrogate_d_objective_d_scale"] == -.03
    assert result["objective_direction_agreement"] == "AGREE"
    assert result["physical_up_objective_down_penalty_reversal"] is True
    assert result["one_sided_deltas"]["physical_power"]["center_to_plus"] == pytest.approx(.005)
    center["d_objective_d_deployment_scale"] = .03
    assert d7._compare_triplet(minus, center, plus, cfg)[
        "objective_direction_agreement"] == "OPPOSITE_REQUIRES_STE_CHECK"
    center["d_objective_d_deployment_scale"] = 0.0
    assert d7._compare_triplet(minus, center, plus, cfg)[
        "objective_direction_agreement"] == "INDETERMINATE_FLAT"


def _synthetic(quick: bool) -> tuple[list[dict], list[dict], dict, dict]:
    cfg = _load_yaml(_project_path(d7.CONFIG))
    steps = cfg["quick" if quick else "data"]["episode_length"]
    members = cfg["quick" if quick else "data"]["initializations"]
    mean = {name: 0.0 for name in d7.METRICS}
    mean.update({"measured_power": .7, "physical_power": .69,
                 "action_penalty": .01, "smooth_penalty": .001,
                 "training_objective": .689})
    groups, rows, center = [], [], {}
    for condition in d7.CONDITIONS:
        for seed in d7.stream_manifest(quick)["weather_bases"]:
            for member in range(members):
                for arm in d7.ARMS:
                    key = condition, seed, member, arm
                    if not quick:
                        center[key] = {"source_state_sha256": "same", "group_mean": dict(mean),
                                       "d_objective_d_deployment_scale": -.01}
                    for scale in d7.SCALES:
                        group = {"hardware_condition": condition, "weather_seed": seed,
                                 "member": member, "controller": arm,
                                 "source_state_sha256": "same", "deployment_scale": scale,
                                 "episode_length": steps, "group_mean": dict(mean),
                                 "per_frame_batch_mean": {name: [mean[name]] * steps
                                                          for name in d7.POWER_TERMS},
                                 "full_training_objective": .689}
                        groups.append(group)
                        for slot, profile in enumerate(d7.d6.d5.d4.d3.PROFILES):
                            for family_index, family in enumerate(d7.d6.d5.d4.d3.FAMILIES):
                                rows.append({
                                    "hardware_condition": condition, "weather_seed": seed,
                                    "member": member, "controller": arm,
                                    "source_state_sha256": "same", "deployment_scale": scale,
                                    "episode_length": steps, "family": family, "slot": slot,
                                    "profile": (f"nominal_for_{profile}"
                                                if condition == "nominal_clone" else profile),
                                    "turbulence_stream_seed": seed + 1000 * slot + family_index,
                                    **mean,
                                })
    return groups, rows, cfg, center


def test_summary_requires_complete_paired_sides_and_center_state() -> None:
    groups, rows, cfg, center = _synthetic(False)
    analysis, comparisons = d7._summarize(groups, rows, cfg, center, quick=False)
    assert analysis["status"] == "EXPLORATORY_HARD_FORWARD_FINITE_DIFFERENCE_NO_GATE"
    assert len(comparisons) == 24
    assert all(c["objective_direction_agreement"] == "INDETERMINATE_FLAT"
               for c in comparisons)
    with pytest.raises(RuntimeError, match="组别缺失或重复"):
        d7._summarize(groups[:-1], rows, cfg, center, quick=False)
    wrong_state = deepcopy(groups)
    wrong_state[1]["source_state_sha256"] = "different"
    with pytest.raises(RuntimeError, match="初态哈希"):
        d7._summarize(wrong_state, rows, cfg, center, quick=False)
    wrong_center = deepcopy(center)
    wrong_center[next(iter(wrong_center))]["source_state_sha256"] = "different"
    with pytest.raises(RuntimeError, match="G2-D6 中心初态哈希"):
        d7._summarize(groups, rows, cfg, wrong_center, quick=False)
    with pytest.raises(RuntimeError, match="记录索引缺失或重复"):
        d7._summarize(groups, rows[:-1], cfg, center, quick=False)


def test_quick_has_no_d6_center_comparison() -> None:
    groups, rows, cfg, center = _synthetic(True)
    analysis, comparisons = d7._summarize(groups, rows, cfg, center, quick=True)
    assert analysis["status"] == "QUICK_SMOKE_ONLY_NO_D6_CENTER_COMPARISON"
    assert comparisons == []


def test_import_has_no_output_writes_or_cuda_initialization() -> None:
    root = Path(__file__).resolve().parents[1]
    code = (
        "from pathlib import Path\n"
        "from unittest.mock import patch\n"
        "import torch\n"
        "with patch.object(Path, 'mkdir', side_effect=AssertionError('write')), "
        "patch.object(Path, 'write_text', side_effect=AssertionError('write')), "
        "patch.object(torch.cuda, 'init', side_effect=AssertionError('cuda')):\n"
        " import scripts.diagnose_s4_r5_g2_d7_hard_forward\n"
    )
    subprocess.run([sys.executable, "-B", "-c", code], cwd=root,
                   check=True, capture_output=True, text=True)
