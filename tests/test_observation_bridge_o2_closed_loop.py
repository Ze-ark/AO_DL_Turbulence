"""O2-B 确定性接口测试；CUDA 微型单元回合不是 39 回合技术验收。"""
from copy import deepcopy
from dataclasses import replace
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from observation_bridge import closed_loop as entry
from scripts.verify_observation_bridge_o2_optics import BUNDLE_SHA256, sealed_bundle_sha256


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load((entry.ROOT / entry.CONFIG).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def gpu(cfg):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable: checks unrun, not CPU fallback")
    entry.configure_runtime()
    device = entry.resolve_device("cuda")
    optics_cfg = yaml.safe_load((entry.ROOT / cfg["optics_config"]).read_text(encoding="utf-8"))
    sensor, bridge = entry.optics.make_components(optics_cfg, device)
    parent = yaml.safe_load((entry.ROOT / cfg["parent"]).read_text(encoding="utf-8"))
    return device, sensor, bridge, parent


@pytest.mark.parametrize("key,value", [("episode_length", 31), ("weather_seed", 8300001),
                                       ("camera_noise_std", .001), ("policy_scale", 1.8),
                                       ("device", "cpu"), ("complete_episodes", True),
                                       ("selector_start_step", 24), ("candidate_epsilon", .2)])
def test_budget_seed_scope_and_controller_frozen(cfg, key, value):
    with pytest.raises(ValueError, match="fixed technical"):
        entry.validate_config({**cfg, key: value})


def test_predeclared_gates_boundary_and_exact_13_branches(cfg):
    entry.validate_config(cfg)
    branches = entry.frozen.controller_specs(dict(policy_initializations=3, scorer_seeds=cfg["scorer_seeds"]))
    assert len(branches) == len({b["controller"] for b in branches}) == 13
    assert branches[0]["controller"] == "integrator"
    assert sum(b["member"] is not None for b in branches) == 12
    assert sum(b["scorer_seed"] is not None for b in branches) == 9
    for key in ("thresholds", "boundary"):
        broken = deepcopy(cfg)
        broken[key][next(iter(broken[key]))] = 1.0
        with pytest.raises(ValueError):
            entry.validate_config(broken)


def test_exact_stream_manifest_no_historical_collision_and_fixed_fold(cfg):
    manifest = entry.verify_streams(cfg)
    assert manifest["turbulence"] == [8300000, 8300001, 8300002]
    assert manifest["power"] == [68300000]
    assert manifest["fold_assignment"] == {"8300000": 0}
    assert manifest["camera_random_draws"] == manifest["proxy_random_draws"] == 0
    assert manifest["historical_manifests_checked"] == 12
    with pytest.raises(RuntimeError, match="collision"):
        entry.verify_streams({**cfg, "weather_seed": 7800000})


def test_synthetic_nominal_profile_cannot_change_under_frozen_name(tmp_path):
    path = tmp_path / "hardware.yaml"
    profile = {"id": "nominal", "label": "基准", "severity": "nominal", "required_for_gate": True}
    path.write_text(yaml.safe_dump({"hardware_profiles": [profile]}), encoding="utf-8")
    assert entry.nominal_profile({"hardware_profile_source": str(path)}).slm_delay_frames == 2
    path.write_text(yaml.safe_dump({"hardware_profiles": [{**profile, "slm_delay_frames": 3}]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="nominal actuator/power parameters changed"):
        entry.nominal_profile({"hardware_profile_source": str(path)})


def test_cpu_tiny_choose_whitelist_disabled_prefix_and_selector_enable(cfg, monkeypatch):
    class UnitPolicy(torch.nn.Module):
        def forward(self, history, valid):
            return history.new_full((len(history), 11), .2)
    h, v = torch.zeros(2, 8, 79), torch.ones(2, 8, dtype=torch.bool)
    calls = []
    def unit_select(history, valid, command, model, normalizer):
        calls.append(True)
        return command + .1, torch.ones(2, dtype=torch.long), torch.zeros(2, 23)
    monkeypatch.setattr(entry.frozen, "select_command", unit_select)
    original, selected, choice, _ = entry.choose_command(h, v, UnitPolicy(), (None, {}), step=24, cfg=cfg)
    assert not calls and choice.eq(0).all()
    torch.testing.assert_close(original, torch.full((2, 11), .35))
    assert torch.equal(original, selected)
    _, selected, choice, _ = entry.choose_command(h, v, UnitPolicy(), (None, {}), step=25, cfg=cfg)
    assert calls and choice.eq(1).all()
    assert selected[0, 0] == pytest.approx(.45)
    assert list(inspect.signature(entry.choose_command).parameters) == ["history", "valid", "policy", "selector", "step", "cfg"]


def test_cpu_tiny_replay_comparator_requires_exact_indices_and_finite_float():
    with pytest.raises(RuntimeError, match="exact index"):
        entry.compare_tensor(torch.tensor([1]), torch.tensor([2]), "candidate", 1.0)
    with pytest.raises(RuntimeError, match="nonfinite"):
        entry.compare_tensor(torch.tensor([float("nan")]), torch.tensor([0.0]), "history", 1e-6)
    with pytest.raises(RuntimeError, match="tolerance"):
        entry.compare_tensor(torch.tensor([.01]), torch.tensor([0.0]), "command", 1e-6)
    with pytest.raises(RuntimeError, match="dtype"):
        entry.compare_tensor(torch.tensor([1.0]), torch.tensor([1.0], dtype=torch.float64), "command", 1e-6)
    assert entry.compare_tensor(torch.zeros(1), torch.zeros(1), "unit", 1e-6) == 0


@pytest.fixture(scope="module")
def micro_trace(cfg, gpu):
    _, sensor, bridge, parent = gpu
    # 与事前 B 天气分离，显式微型 CUDA 单元预算：4 步×3 家族=12 次转移。
    unit_cfg = {**cfg, "weather_seed": 8330000, "episode_length": 4}
    context = {"physical_transitions": 0}
    trace, audit, record = entry.rollout(unit_cfg, parent, {"controller": "integrator"}, sensor, bridge,
                                        None, None, lambda _: None, context)
    assert context["physical_transitions"] == 12
    return trace, audit, record, unit_cfg


def test_cuda_micro_12_transitions_full_phase_observation_clock_and_interface(micro_trace):
    trace, audit, record, cfg = micro_trace
    assert trace["residual"].shape == (5, 3, 21)
    assert trace["history"].shape == (4, 3, 8, 79)
    assert record["physical_transitions"] == 12 and record["complete_episodes"] == 3
    assert record["modal_error_max_rad"] <= .001
    assert audit["joint_target_rad"].shape == (5, 3, 21)
    assert audit["action_power"].shape == (4, 3)
    assert trace["valid"][0].sum() == 3
    assert trace["history"][0, :, -1, 77].eq(-1).all()
    assert trace["history"][0, :, -1, 78].eq(0).all()
    assert trace["power_action_step"].tolist() == [0, 1, 2, 3]
    assert trace["power_arrival_step"].tolist() == [1, 2, 3, 4]
    assert torch.equal(trace["next_clock"][:, 0, 0], torch.arange(1, 5, device=trace["residual"].device).float())
    assert trace["next_clock"][..., -1].eq(1).all()
    assert trace["requested_delta"].device.type == "cuda"


def test_cuda_micro_disk_replay_no_new_environment_transitions(micro_trace, tmp_path):
    trace, _, _, cfg = micro_trace
    target = tmp_path / "visible.pt"
    torch.save({k: v.cpu() for k, v in trace.items()}, target)
    loaded = torch.load(target, map_location=trace["residual"].device, weights_only=True)
    result = entry.replay_visible(loaded, cfg, None, None)
    assert result["max_absolute_error"] == 0
    assert result["candidate_indices_exact"] and result["clocks_exact"]
    assert result["new_environment_transitions"] == 0


@pytest.mark.parametrize("field", ["choice", "requested_delta", "history", "power_action_step"])
def test_cuda_micro_saved_decision_or_clock_corruption_fails(micro_trace, field):
    trace, _, _, cfg = micro_trace
    broken = {k: v.clone() for k, v in trace.items()}
    broken[field].flatten()[0] += 1
    with pytest.raises((RuntimeError, ValueError)):
        entry.replay_visible(broken, cfg, None, None)


def test_cuda_micro_prefix_detects_unpaired_history(micro_trace):
    trace = micro_trace[0]
    entry.require_prefix(trace, trace, 2)
    altered = {k: v.clone() for k, v in trace.items()}
    altered["residual"][0, 0, 0] += .1
    with pytest.raises(RuntimeError, match="observations differ"):
        entry.require_prefix(trace, altered, 2)


def test_cuda_environment_proxy_and_privileged_return_poison_does_not_change_measurement(cfg, gpu, monkeypatch):
    _, sensor, bridge, parent = gpu
    first = entry.make_environment(parent, bridge.basis, 8330100, 1)
    second = entry.make_environment(parent, bridge.basis, 8330100, 1)
    a = entry.HolographicEnvironmentPort(first, sensor, bridge, .001)
    b = entry.HolographicEnvironmentPort(second, sensor, bridge, .001)
    reset, step = second.reset, second.step
    def poison_reset(*args, **kwargs):
        raw, info = reset(*args, **kwargs)
        return raw.fill_(float("nan")), {key: value.new_full(value.shape, float("nan")) for key, value in info.items()}
    def poison_step(*args, **kwargs):
        raw, reward, done, truncated, info = step(*args, **kwargs)
        # 仅保留必须用于合法功率/独立审计的字段；privileged raw/下一状态审计字段失效。
        for key in ("strehl", "phase_rmse", "power_in_bucket"):
            info[key].fill_(float("nan"))
        return raw.fill_(float("nan")), reward.fill_(float("nan")), done, truncated, info
    def forbidden_proxy(*args, **kwargs):
        raise AssertionError("oracle modal proxy was used")
    monkeypatch.setattr(second, "reset", poison_reset)
    monkeypatch.setattr(second, "step", poison_step)
    monkeypatch.setattr(first, "proxy", forbidden_proxy)
    monkeypatch.setattr(second, "proxy", forbidden_proxy)
    observed_a, _ = a.reset(8330100)
    observed_b, _ = b.reset(8330100)
    assert torch.equal(observed_a.residual, observed_b.residual)
    action = observed_a.residual.new_zeros((3, 21))
    next_a, power_a, _ = a.step(action, 0, lambda: None)
    next_b, power_b, _ = b.step(action, 0, lambda: None)
    assert torch.equal(next_a.residual, next_b.residual)
    assert torch.equal(power_a.value, power_b.value)


def test_cuda_environment_causal_wrong_clock_and_invalid_observation_fail_closed(gpu, monkeypatch):
    _, sensor, bridge, parent = gpu
    env = entry.make_environment(parent, bridge.basis, 8330200, 1)
    port = entry.HolographicEnvironmentPort(env, sensor, bridge, .001)
    readout, _ = port.reset(8330200)
    calls = []
    with pytest.raises(RuntimeError, match="clock mismatch"):
        port.step(readout.residual.new_zeros((3, 21)), 1, lambda: calls.append(True))
    assert not calls and env.step_count == 0
    def invalid_field(value):
        raise ValueError("hole inside pupil")
    monkeypatch.setattr(bridge, "measure", invalid_field)
    with pytest.raises(ValueError, match="hole"):
        port.observe()  # 没有真值回退。


def test_cuda_command_changes_image_observation_only_after_frozen_actuator_delay(gpu):
    _, sensor, bridge, parent = gpu
    a = entry.HolographicEnvironmentPort(entry.make_environment(parent, bridge.basis, 8330300, 3), sensor, bridge, .001)
    b = entry.HolographicEnvironmentPort(entry.make_environment(parent, bridge.basis, 8330300, 3), sensor, bridge, .001)
    first, _ = a.reset(8330300)
    other, _ = b.reset(8330300)
    assert torch.equal(first.residual, other.residual)
    zero = first.residual.new_zeros((3, 21))
    for step in range(3):
        requested = zero.clone()
        if step == 0:
            requested[:, 0] = .15
        next_a, _, _ = a.step(requested, step, lambda: None)
        next_b, _, _ = b.step(zero, step, lambda: None)
        if step < 2:
            assert torch.equal(next_a.residual, next_b.residual)
        else:
            # 同一天气，只改变动作；正模态应在两帧执行延迟后进入图像读数。
            assert (next_a.residual[:, 0] - next_b.residual[:, 0]).min() > .1


def test_preflight_preserves_output_and_path_boundary(cfg, tmp_path, monkeypatch):
    kept = tmp_path / "outputs/kept"
    kept.mkdir(parents=True)
    sentinel = kept / "original.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    path = tmp_path / "config.yaml"
    for output, exception in (("outputs/kept", FileExistsError), ("outputs", ValueError), ("../outside", ValueError)):
        path.write_text(yaml.safe_dump({**cfg, "output_directory": output}), encoding="utf-8")
        with pytest.raises(exception):
            entry.preflight(path)
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_frozen_weights_and_saved_normalizers_load_without_data_read(cfg, gpu, monkeypatch):
    device, _, _, parent = gpu
    original = torch.load
    reads = []
    def checkpoint_only(path, *args, **kwargs):
        assert "checkpoints" in str(path)
        reads.append(str(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(torch, "load", checkpoint_only)
    monkeypatch.setattr(entry.frozen, "load_assets", lambda *a, **k: pytest.fail("old dataset loader called"))
    monkeypatch.setattr(entry.frozen.training, "fit_normalizer", lambda *a, **k: pytest.fail("normalizer refit"))
    policies, scorers, manifest = entry.load_assets(device, parent)
    assert len(policies) == 3 and len(scorers) == 12 and len(reads) == len(manifest) == 15
    assert all(not m.training for m in policies.values())
    assert all(not p.requires_grad for m in policies.values() for p in m.parameters())
    assert all(part[1]["std"].min() > 0 for part in scorers.values())


def test_cuda_actual_frozen_models_accept_three_row_whitelist_and_safe_selection(cfg, gpu, micro_trace):
    device, _, _, parent = gpu
    policies, scorers, _ = entry.load_assets(device, parent)
    h, v = micro_trace[0]["history"][0], micro_trace[0]["valid"][0]
    for member in range(3):
        original, selected, choice, predicted = entry.choose_command(
            h, v, policies[member], scorers[0, cfg["scorer_seeds"][member]], step=25, cfg=cfg)
        assert original.shape == selected.shape == (3, 11) and predicted.shape == (3, 23)
        assert predicted.device.type == "cuda" and torch.isfinite(predicted).all()
        assert choice.dtype == torch.long and choice.min() >= 0 and choice.max() < 25
        candidates = entry.frozen.source.candidate_commands(original, .1)
        expected = torch.stack(list(candidates.values()), 1)[torch.arange(3, device=device), choice]
        assert torch.equal(selected, expected)
        assert all(not p.requires_grad for p in policies[member].parameters())


def test_failed_run_preserves_evidence_without_retry(cfg, tmp_path, monkeypatch):
    # 纯 CPU 文件生命周期单元测试：没有相机运算、环境或模型前向。
    output = tmp_path / "outputs/failed_unit"
    report = {"stream_manifest": {}}
    monkeypatch.setattr(entry, "preflight", lambda *_: (cfg, output, torch.device("cpu"), {}, {}, {}, [], report))
    monkeypatch.setattr(entry, "source_manifest", lambda *_: {"unit_only": True})
    def crash(*args, **kwargs):
        raise RuntimeError("intentional unit failure before environment creation")
    monkeypatch.setattr(entry.optics, "make_components", crash)
    with pytest.raises(RuntimeError, match="intentional unit failure"):
        entry.run()
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["last_context"]["physical_transitions"] == 0
    assert "intentional unit failure" in failure["message"]
    assert not (output / "SUCCESS.json").exists()
    with pytest.raises(FileExistsError):
        entry.run()


def test_import_inert_cli_help_and_sealed_bundle_unchanged():
    code = "from pathlib import Path\nfrom unittest.mock import patch\nimport torch\nwith patch.object(Path,'mkdir',side_effect=AssertionError('write')), patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('GPU')):\n import observation_bridge.closed_loop\n import scripts.verify_observation_bridge_o2_closed_loop\n"
    subprocess.run([sys.executable, "-B", "-c", code], cwd=entry.ROOT, check=True, capture_output=True)
    help_result = subprocess.run([sys.executable, "-X", "utf8", "-B", "scripts/verify_observation_bridge_o2_closed_loop.py", "--help"],
                                 cwd=entry.ROOT, check=True, capture_output=True, text=True, encoding="utf-8")
    assert "--preflight-only" in help_result.stdout
    assert sealed_bundle_sha256() == BUNDLE_SHA256
