"""G2-D6 完整目标诊断的确定性 CPU 测试；不启动训练或正式仿真。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts import diagnose_s4_r5_g2_d6_full_objective as d6
from src.rl.r4_observation import R4Interface
from src.rl.s4_training import _load_yaml, _project_path


def test_contract_and_seed_stream_isolation() -> None:
    cfg = _load_yaml(_project_path(d6.CONFIG))
    d6._contract(cfg)
    altered = deepcopy(cfg)
    altered["boundary"]["confirmation_access"] = True
    with pytest.raises(ValueError, match="合同"):
        d6._contract(altered)

    formal, quick = d6.stream_manifest(False), d6.stream_manifest(True)
    assert formal["weather_bases"] == [7_340_000, 7_340_010]
    assert quick["weather_bases"] == [7_350_000]
    historical = (
        d6.d5.stream_manifest(False), d6.d5.stream_manifest(True),
        d6.d5.d4.stream_manifest(False), d6.d5.d4.stream_manifest(True),
        d6.d5.d4.d3.stream_manifest(False), d6.d5.d4.d3.stream_manifest(True),
        d6.d5.d4.d3.train.stream_manifest(quick=False),
        d6.d5.d4.d3.train.stream_manifest(quick=True),
        d6.d5.d4.d3.train.g2.stream_manifest(False),
        d6.d5.d4.d3.train.g2.stream_manifest(True),
    )
    for name in ("turbulence", "sensor", "power"):
        assert len(set(formal[name])) == len(formal[name])
        assert len(set(quick[name])) == len(quick[name])
        assert set(formal[name]).isdisjoint(quick[name])
        past = set().union(*(set(stream[name]) for stream in historical))
        assert set(formal[name]).isdisjoint(past)
        assert set(quick[name]).isdisjoint(past)
    d6._check_stream_isolation()


class _ConstantPolicy(torch.nn.Module):
    def forward(self, history: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        raw = history.new_zeros((len(history), 11))
        raw[:, 0] = 0.2
        return raw


class _TinyDelayedEnv:
    """只模拟确定性两帧功率延迟，供目标分解单测使用。"""

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
        # 物理功率只依赖两帧前动作；另加固定测量偏差以分清两种功率。
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


def test_exact_200_frame_objective_decomposition_with_delayed_power() -> None:
    env = _TinyDelayedEnv(200)
    interface = R4Interface()
    interface.reset(torch.zeros(1, 21, dtype=torch.float64), episode_id="cpu-unit")
    counter = SimpleNamespace(steps=0)

    def tick() -> None:
        counter.steps += 1

    result = d6._rollout(
        (env, interface), _ConstantPolicy(), scale_value=1.75,
        action_weight=0.01, smooth_weight=0.001,
        progress=SimpleNamespace(tick=tick),
    )
    assert counter.steps == 200
    assert len(result["per_frame_batch_mean"]["training_objective"]) == 200
    assert result["per_frame_batch_mean"]["physical_power"][:2] == [0.7, 0.7]
    assert result["per_frame_batch_mean"]["physical_power"][2] != 0.7
    assert result["group_mean"]["measured_power"] != result["group_mean"]["physical_power"]
    expected = (result["group_mean"]["measured_power"]
                - result["group_mean"]["action_penalty"]
                - result["group_mean"]["smooth_penalty"])
    assert result["full_training_objective"] == pytest.approx(expected, abs=1e-12)
    assert result["group_mean"]["training_objective"] == pytest.approx(expected, abs=1e-12)
    assert result["radial_d_objective_d_alpha"] == pytest.approx(
        1.75 * result["d_objective_d_deployment_scale"], abs=1e-12)
    assert result["d_objective_d_deployment_scale"] != 0


def _synthetic_groups_and_rows() -> tuple[list[dict], list[dict], dict]:
    cfg = _load_yaml(_project_path(d6.CONFIG))
    seed = d6.stream_manifest(True)["weather_bases"][0]
    steps = cfg["quick"]["episode_length"]
    mean = {name: 0.0 for name in d6.METRICS}
    mean.update({"measured_power": 0.7, "physical_power": 0.69,
                 "action_penalty": 0.0004, "smooth_penalty": 0.00004,
                 "training_objective": 0.69956})
    groups, rows = [], []
    for condition in d6.CONDITIONS:
        for arm in d6.ARMS:
            group = {
                "hardware_condition": condition, "weather_seed": seed, "member": 0,
                "controller": arm, "source_state_sha256": "same",
                "deployment_scale": 1.75,
                "group_mean": dict(mean),
                "per_frame_batch_mean": {name: [value] * steps for name, value in mean.items()
                                         if name in ("measured_power", "action_penalty",
                                                     "smooth_penalty", "training_objective",
                                                     "physical_power")},
                "full_training_objective": mean["training_objective"],
                "d_objective_d_deployment_scale": 0.01,
                "radial_d_objective_d_alpha": 0.0175,
            }
            groups.append(group)
            for slot, profile in enumerate(d6.d5.d4.d3.PROFILES):
                for family_index, family in enumerate(d6.d5.d4.d3.FAMILIES):
                    rows.append({
                        "hardware_condition": condition, "weather_seed": seed,
                        "member": 0, "controller": arm,
                        "family": family, "slot": slot,
                        "profile": (f"nominal_for_{profile}"
                                    if condition == "nominal_clone" else profile),
                        "turbulence_stream_seed": seed + 1000 * slot + family_index,
                        "source_state_sha256": "same", **mean,
                    })
    return groups, rows, cfg


def test_summary_rejects_missing_duplicate_and_mismatched_pairs() -> None:
    groups, rows, cfg = _synthetic_groups_and_rows()
    analysis = d6._summarize(groups, rows, cfg, quick=True)
    assert analysis["status"] == "EXPLORATORY_FULL_HORIZON_OBJECTIVE_NO_GATE"

    with pytest.raises(RuntimeError, match="组别缺失或重复"):
        d6._summarize(groups[:-1], rows, cfg, quick=True)
    duplicated = deepcopy(groups)
    duplicated[1] = deepcopy(duplicated[0])
    with pytest.raises(RuntimeError, match="组别缺失或重复"):
        d6._summarize(duplicated, rows, cfg, quick=True)
    wrong_state = deepcopy(groups)
    wrong_state[1]["source_state_sha256"] = "different"
    with pytest.raises(RuntimeError, match="初态哈希"):
        d6._summarize(wrong_state, rows, cfg, quick=True)
    with pytest.raises(RuntimeError, match="记录索引缺失或重复"):
        d6._summarize(groups, rows[:-1], cfg, quick=True)
    duplicated_row = deepcopy(rows)
    duplicated_row[1] = deepcopy(duplicated_row[0])
    with pytest.raises(RuntimeError, match="记录索引缺失或重复"):
        d6._summarize(groups, duplicated_row, cfg, quick=True)


def test_import_has_no_output_writes_or_cuda_initialization() -> None:
    root = Path(__file__).resolve().parents[1]
    code = (
        "from pathlib import Path\n"
        "from unittest.mock import patch\n"
        "import torch\n"
        "with patch.object(Path, 'mkdir', side_effect=AssertionError('write')), "
        "patch.object(Path, 'write_text', side_effect=AssertionError('write')), "
        "patch.object(torch.cuda, 'init', side_effect=AssertionError('cuda')):\n"
        " import scripts.diagnose_s4_r5_g2_d6_full_objective\n"
    )
    subprocess.run([sys.executable, "-B", "-c", code], cwd=root,
                   check=True, capture_output=True, text=True)
