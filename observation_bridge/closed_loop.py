"""O2-B 纯软件全息反馈技术入口；冻结控制器，不提供收益排名。

只复用冻结代码的纯函数和类，不调用旧实验入口、数据加载或标准化拟合。
环境真值只用于生成人工相机图和独立审计；控制器签名是白名单张量。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback
from typing import Any, Callable

import torch
import yaml

from observation_bridge.cuda_modal_observation import CudaModalObservationBridge
from observation_bridge.holography import SyntheticOffAxisSensor
from scripts import evaluate_s4_r5_g2_d13_closed_loop as frozen
from scripts import verify_observation_bridge_o2_optics as optics
from src.rl.r4_observation import HistoryView, PowerMeasurement, R4Interface
from src.rl.r4_trajectory import anchor_delta
from src.rl.r5_batched_environment import R5BatchedEnvironment
from src.rl.r5_policy_training import ResidualGRUPolicy
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.simulation.hardware_effects import HardwareProfile
from src.simulation.modes import project_phase_to_modes
from src.simulation.optics import focal_plane_metrics
from src.simulation.robust_control import RobustnessCondition

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/experiments/observation_bridge_o2_closed_loop_quick_v1.yaml"
OPTICS_SUMMARY_SHA = "606c3763201b0fe1f4ce490445cc434c5412526d1bb00f854d0d9aab7247161f"
Scorer = tuple[torch.nn.Module, dict[str, torch.Tensor]]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(cfg: dict[str, Any]) -> None:
    fixed = {
        "schema": "observation_bridge_o2_closed_loop_quick_v1",
        "scope": "cuda_synthetic_holography_closed_loop_technical_only", "device": "cuda",
        "design_path": "configs/experiments/observation_bridge_o2_design_v1.json",
        "design_sha256": optics.DESIGN_SHA256,
        "optics_config": "configs/experiments/observation_bridge_o2_optics_v1.yaml",
        "optics_summary": "outputs/observation_bridge_o2_optics_v1/summary.json",
        "optics_summary_sha256": OPTICS_SUMMARY_SHA,
        "parent": "configs/experiments/s4_r5_policy_training_v1.yaml",
        "weather_seed": 8_300_000, "weather_index": 0,
        "family_ids": ["frozen", "boiling", "varying"], "episode_length": 32, "batch_size": 3,
        "controller_branches": 13, "complete_episodes": 39, "physical_transitions": 1248,
        "selector_start_step": 25, "policy_initializations": [0, 1, 2],
        "scorer_seeds": [7564000, 7564001, 7564002],
        "integrator": {"gain": .15, "leak": .1, "tracking_gain": .5},
        "policy_scale": 1.75, "candidate_epsilon": .1, "camera_noise_std": 0.0,
        "thresholds": {"ideal_modal_error_max_rad": .001, "replay_max_absolute_error": 1e-6,
                       "invalid_observations": 0, "failed_or_truncated_episodes": 0},
        "boundary": {"training_updates": 0, "scientific_gain_analysis": False,
                     "independent_confirmation": False, "old_confirmation_trajectory_access": False,
                     "real_data_access": False, "real_slm_actions": False, "automatic_retry": False},
    }
    # JSON 化保留 bool/int 的类型差异，不把 True 当成预算 1。
    if set(cfg) != set(fixed) | {"output_directory"} or any(
            json.dumps(cfg.get(k), sort_keys=True) != json.dumps(v, sort_keys=True) for k, v in fixed.items()):
        raise ValueError("O2-B fixed technical scope, budget or thresholds changed")


def configure_runtime() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8", ":16:8"):
        raise RuntimeError("unsupported deterministic cuBLAS workspace setting")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def verify_optics_prerequisite(cfg: dict[str, Any]) -> dict[str, Any]:
    design_path = ROOT / cfg["design_path"]
    if optics.file_sha256(design_path) != optics.DESIGN_SHA256:
        raise RuntimeError("historical O2 design changed")
    sources = optics.check_sources(read_json(design_path))
    # 用已封存上游代码中的指纹串联实际调用的外部脚本；仅校验字节，不加载旧数据。
    helper_pins = {
        Path(frozen.local.__file__): frozen.LOCAL_ENTRY_SHA256,
        Path(frozen.training.__file__): frozen.local.TRAINING_ENTRY_SHA256,
        Path(frozen.training.pairing.__file__): frozen.training.PAIRING_ENTRY_SHA256,
        Path(frozen.source.__file__): frozen.training.pairing.SOURCE_ENTRY_SHA256,
    }
    for helper, expected in helper_pins.items():
        if optics.file_sha256(helper) != expected:
            raise RuntimeError(f"frozen controller helper changed: {helper.name}")
    sources["frozen_controller_helper_sha256"] = {p.relative_to(ROOT).as_posix(): h for p, h in helper_pins.items()}
    summary_path = ROOT / cfg["optics_summary"]
    if optics.file_sha256(summary_path) != OPTICS_SUMMARY_SHA:
        raise RuntimeError("O2-A technical summary changed")
    summary, success = read_json(summary_path), read_json(summary_path.parent / "SUCCESS.json")
    if ((summary_path.parent / "failure.json").exists()
            or summary["status"] != "O2_A_SYNTHETIC_OPTICS_TECHNICAL_PASS_ONLY"
            or summary["positive_fixtures_completed"] != 47
            or success["summary_sha256"] != OPTICS_SUMMARY_SHA):
        raise RuntimeError("O2-A prerequisite incomplete")
    for filename, expected in success["artifact_sha256"].items():
        if Path(filename).name != filename or optics.file_sha256(summary_path.parent / filename) != expected:
            raise RuntimeError("O2-A successful artifacts changed")
    for filename, expected in summary["source_manifest"]["new_source_sha256"].items():
        if optics.file_sha256(ROOT / filename) != expected:
            raise RuntimeError(f"O2-A executed source changed: {filename}")
    if optics.file_sha256(ROOT / cfg["optics_config"]) != summary["source_manifest"]["config_sha256"]:
        raise RuntimeError("O2-A executed optical calibration changed")
    return sources


def stream_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    seed = cfg["weather_seed"]
    return dict(weather_bases=[seed], turbulence=[seed + i for i in range(3)],
                power=[seed + 60_000_000], unused_proxy_sensor=[seed + 50_000_000],
                camera_reserved=[seed + i + 170_000_000 for i in range(3)],
                camera_random_draws=0, proxy_random_draws=0,
                fold_assignment={str(seed): cfg["weather_index"] % 4},
                controllers_share_exogenous_streams=True)


def _integers(value: Any) -> set[int]:
    if type(value) is int:
        return {value}
    if isinstance(value, dict):
        return set().union(*(_integers(v) for v in value.values())) if value else set()
    if isinstance(value, (list, tuple)):
        return set().union(*(_integers(v) for v in value)) if value else set()
    return set()


def verify_streams(cfg: dict[str, Any]) -> dict[str, Any]:
    # 仅调用历史纯种子清单函数，不读取历史轨迹/成绩或启动环境。
    from scripts import confirm_s4_r5_g2_c1 as c1
    s = frozen.source
    factories = [s.stream_manifest, s.d10.d9.stream_manifest,
                 s.d10.d9.d8.stream_manifest, s.d10.d9.d3.stream_manifest,
                 s.d10.d9.d8.d2.g2.stream_manifest]
    old = [fn(quick=q) for fn in factories for q in (False, True)]
    old_cfg = yaml.safe_load((ROOT / c1.CONFIG).read_text(encoding="utf-8"))
    old.extend(c1.stream_manifest(old_cfg[k]) for k in ("data", "quick"))
    manifest = stream_manifest(cfg)
    current = set().union(*(set(manifest[k]) for k in
                            ("weather_bases", "turbulence", "power", "unused_proxy_sensor", "camera_reserved")))
    if any(current & _integers(item) for item in old):
        raise RuntimeError("O2-B exact stream collision with declared G2/C1 history")
    # 历史更早 R5 及 720/760 万预留命名空间不得使用；C 的新空间仍预留。
    if cfg["weather_seed"] != 8_300_000 or current & {8_400_000 + i for i in range(3)}:
        raise RuntimeError("O2-B weather namespace is not the declared technical reservation")
    manifest["historical_manifests_checked"] = len(old)
    manifest["disjointness_scope"] = "exact_declared_G2_and_C1_streams_plus_new_namespace_guard"
    return manifest


def load_assets(device: torch.device, parent: dict[str, Any]) -> tuple[dict, dict, list[dict]]:
    """只加载 SHA 封存的末次权重和其中的标准化；不加载数据、不重新拟合。"""
    archive = read_json(ROOT / "configs/experiments/s4_r5_g2_c1_archive_v1.json")["frozen_files"]
    policies, scorers, manifest = {}, {}, []
    fold_metadata = {}
    for member in range(3):
        relative = f"outputs/s4_r5_g2_d8_action_penalty_zero_v1/checkpoints/action_penalty_0_policy_{member}_00512.pt"
        if optics.file_sha256(ROOT / relative) != archive[relative]:
            raise RuntimeError("frozen policy changed")
        checkpoint = torch.load(ROOT / relative, map_location=device, weights_only=True)
        if (checkpoint["arm"], checkpoint["init"], checkpoint["update"], checkpoint["deployment_scale"]) != (
                "action_penalty_0", member, 512, 1.75):
            raise ValueError("frozen policy identity mismatch")
        model = ResidualGRUPolicy(parent["policy"]["hidden_size"], parent["policy"]["output_size"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval().requires_grad_(False)
        policies[member] = model
        manifest.append(dict(kind="policy", member=member, path=relative, sha256=archive[relative]))
    for fold in range(4):
        for seed in (7564000, 7564001, 7564002):
            relative = f"outputs/s4_r5_g2_d12_candidate_scorer_v1/checkpoints/fold_{fold}_current_seed_{seed}_01000.pt"
            if optics.file_sha256(ROOT / relative) != archive[relative]:
                raise RuntimeError("frozen scorer changed")
            checkpoint = torch.load(ROOT / relative, map_location=device, weights_only=True)
            split = dict(fold=fold, train_weather=checkpoint["train_weather"], held_out_weather=checkpoint["held_out_weather"])
            frozen.local.validate_checkpoint(checkpoint, split, "current", seed, quick=False)
            if fold in fold_metadata and split != fold_metadata[fold]:
                raise ValueError("scorer seeds disagree on frozen weather fold")
            fold_metadata[fold] = split
            model = frozen.training.CandidateScorer("current").to(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval().requires_grad_(False)
            scorers[fold, seed] = (model, checkpoint["normalizer"])
            manifest.append(dict(kind="scorer", fold=fold, seed=seed, path=relative, sha256=archive[relative],
                                 train_weather=split["train_weather"], held_out_weather=split["held_out_weather"],
                                 normalizer_source="saved_checkpoint_no_refit"))
    for model in list(policies.values()) + [part[0] for part in scorers.values()]:
        if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            raise ValueError("nonfinite frozen model")
    return policies, scorers, manifest


def preflight(path: str | Path = CONFIG) -> tuple:
    config_path = Path(path) if Path(path).is_absolute() else ROOT / path
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(cfg)
    output = (ROOT / cfg["output_directory"]).resolve()
    output_root = (ROOT / "outputs").resolve()
    if output == output_root or not output.is_relative_to(output_root):
        raise ValueError("O2-B output must be a new child of workspace outputs")
    if output.exists():
        raise FileExistsError(f"preserve existing O2-B output: {output}")
    sources = verify_optics_prerequisite(cfg)
    streams = verify_streams(cfg)
    configure_runtime()
    device = resolve_device(cfg["device"])
    if device.type != "cuda":
        raise RuntimeError("O2-B requires CUDA; no CPU fallback")
    parent = yaml.safe_load((ROOT / cfg["parent"]).read_text(encoding="utf-8"))
    profile = nominal_profile(parent)
    policies, scorers, models = load_assets(device, parent)
    if any(cfg["weather_seed"] in row.get("train_weather", []) + row.get("held_out_weather", []) for row in models):
        raise RuntimeError("O2-B weather was used by frozen scorer training/development")
    report = dict(status="O2_B_READY_FOR_USER_TECHNICAL_RUN", **cfg["boundary"],
                  device=str(device), complete_episodes=39, physical_transitions=1248,
                  batch_size=3, episode_length=32, controller_branches=13,
                  loaded_policies=len(policies), loaded_scorers=len(scorers),
                  preflight_model_forward_calls=0, preflight_environment_transitions=0,
                  stream_manifest=streams, frozen_sources=sources,
                  synthetic_nominal_profile=profile.as_record(), proxy_observation_noise_used=False,
                  output_directory=str(output), assigned_scorer_fold=cfg["weather_index"] % 4)
    return cfg, output, device, parent, policies, scorers, models, report


@dataclass(frozen=True)
class SensorReadout:
    residual: torch.Tensor
    observation_step: int


class HolographicEnvironmentPort:
    """物理/相机侧边界。真值留在生成器和审计，不进入 choose_command。"""

    def __init__(self, env: R5BatchedEnvironment, sensor: SyntheticOffAxisSensor,
                 bridge: CudaModalObservationBridge, max_modal_error: float):
        self.env, self.sensor, self.bridge = env, sensor, bridge
        self.max_modal_error = max_modal_error

    @torch.no_grad()
    def observe(self) -> tuple[SensorReadout, dict[str, torch.Tensor]]:
        if self.env.slm.current_phase is None:
            raise RuntimeError("unknown actuator initial state")
        phase = self.env._current_turbulence_window() + self.env.slm.current_phase
        field = torch.polar(self.env.pupil.to(phase.dtype).expand_as(phase), phase)
        reconstructed = self.sensor.reconstruct(self.sensor.render(field))
        measured = self.bridge.measure(reconstructed)
        target, true_jump = optics.audit_known_phase(phase.double(), self.bridge)
        if true_jump > self.bridge.tolerances.max_neighbor_jump_rad:
            raise RuntimeError("O2-B known spatial sampling precondition failed")
        error = (measured.residual_rad.double() - target).abs().amax(dim=1)
        if float(error.max()) > self.max_modal_error:
            raise RuntimeError("O2-B predeclared modal observation tolerance failed")
        audit = dict(joint_target_rad=target, legacy_projection_rad=project_phase_to_modes(phase, self.env.basis, self.env.pupil),
                     modal_error_rad=error, fit_rmse_rad=measured.fit_rmse_rad,
                     max_wrapped_neighbor_jump_rad=measured.max_neighbor_jump_rad,
                     batch_true_neighbor_jump_rad=error.new_full(error.shape, true_jump))
        return SensorReadout(measured.residual_rad, self.env.step_count), audit

    @torch.no_grad()
    def reset(self, seed: int) -> tuple[SensorReadout, dict[str, torch.Tensor]]:
        _raw, _info = self.env.reset(seed=seed)  # 审计返回字段全部丢弃，不调用 env.proxy。
        if not bool(self.env.slm.current_phase.eq(0).all()):
            raise RuntimeError("synthetic zero initial actuator state not verified")
        return self.observe()

    @torch.no_grad()
    def step(self, requested_delta: torch.Tensor, step: int,
             on_transition: Callable[[], None]) -> tuple[SensorReadout, PowerMeasurement, dict]:
        if self.env.step_count != step:
            raise RuntimeError("O2-B command/observation clock mismatch")
        previous_turbulence = self.env._current_turbulence_window().clone()
        _raw, _reward, terminated, truncated, info = self.env.step(requested_delta)
        on_transition()
        if (self.env.step_count != step + 1 or bool(truncated.any())
                or not torch.equal(terminated, torch.full_like(terminated, step + 1 == self.env.config.episode_length))):
            raise RuntimeError("O2-B failed, truncated or mis-timestamped episode")
        # 独立验证动作功率在推进湍流之前测量，不将下一观测质量冒充动作奖励。
        action_quality = focal_plane_metrics(
            previous_turbulence + self.env.slm.current_phase, self.env.pupil,
            self.env.config.bucket_radius_pixels, ideal_intensity=self.env._ideal_intensity,
            bucket_mask=self.env._bucket_mask)
        for name, key in (("power_in_bucket", "reward_power_in_bucket"), ("strehl", "reward_strehl"),
                          ("phase_rmse", "reward_phase_rmse")):
            if float((action_quality[name] - info[key]).abs().max()) > 1e-6:
                raise RuntimeError("O2-B action reward/next phase clocks mixed")
        readout, audit = self.observe()  # env 已推进：这是下一观测，不是动作前真值。
        power = PowerMeasurement(info["measured_power_in_bucket"].clone(), step, step + 1)
        audit.update(action_power=info["reward_power_in_bucket"], action_strehl=info["reward_strehl"],
                     action_phase_rmse=info["reward_phase_rmse"], violation=info["violation_fraction"],
                     requested_modal=info["requested_modal"], applied_modal=info["applied_modal"],
                     observation_step=torch.full_like(terminated, step + 1, dtype=torch.int64),
                     power_action_step=torch.full_like(terminated, step, dtype=torch.int64))
        return readout, power, audit


@torch.no_grad()
def choose_command(history: torch.Tensor, valid: torch.Tensor, policy: torch.nn.Module | None,
                   selector: Scorer | None, *, step: int, cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """仅白名单历史和固定步索引；无 env、truth、profile、info 或未来安全参数。"""
    original = history.new_zeros((len(history), 11)) if policy is None else policy(history, valid) * cfg["policy_scale"]
    selected, choice = original, torch.zeros(len(history), dtype=torch.long, device=history.device)
    prediction = history.new_zeros((len(history), 23))
    if selector is not None and step >= cfg["selector_start_step"]:
        selected, choice, prediction = frozen.select_command(history, valid, original, *selector)
    return original, selected, choice, prediction


def nominal_profile(parent: dict[str, Any]) -> HardwareProfile:
    profile_cfg = yaml.safe_load((ROOT / parent["hardware_profile_source"]).read_text(encoding="utf-8"))
    nominal = next(row for row in profile_cfg["hardware_profiles"] if row["id"] == "nominal")
    profile = HardwareProfile.from_mapping(nominal)
    expected = HardwareProfile("nominal", "基准", "nominal", True)
    if profile != expected:
        raise RuntimeError("O2-B frozen synthetic nominal actuator/power parameters changed")
    return profile


def make_environment(parent: dict[str, Any], basis: torch.Tensor, seed: int, steps: int) -> R5BatchedEnvironment:
    base, _ = load_s1_config(ROOT / parent["environment_config"])
    base = replace(base, num_modes=21, batch_size=1, episode_length=steps)
    profile = nominal_profile(parent)
    first = RobustnessCondition.from_mapping(dict(parent["families"][0], base_seed=seed))
    return R5BatchedEnvironment(first.environment_config(base), basis.device, basis, parent["families"],
                                [profile], parent["data"]["sensor_seed_offset"])


def compare_tensor(actual: torch.Tensor, expected: torch.Tensor, name: str, tolerance: float) -> float:
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.device != expected.device:
        raise RuntimeError(f"O2-B replay shape/dtype/device mismatch: {name}")
    if actual.dtype in (torch.bool, torch.int64):
        if not torch.equal(actual, expected):
            raise RuntimeError(f"O2-B exact index/mask replay failed: {name}")
        return 0.0
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
        raise RuntimeError(f"O2-B nonfinite replay: {name}")
    error = float((actual - expected).abs().max())
    if error > tolerance:
        raise RuntimeError(f"O2-B replay tolerance failed: {name}={error}")
    return error


@torch.no_grad()
def replay_visible(trace: dict[str, torch.Tensor], cfg: dict[str, Any], policy, selector) -> dict[str, Any]:
    """重放保存的测量/功率 → 历史 → 模型决策/安全请求；不读取 audit。

    不重新生成物理回合，新增环境转移为 0；不能冒称物理轨迹复现。
    """
    steps = len(trace["requested_delta"])
    if steps != cfg["episode_length"] or trace["residual"].shape != (steps + 1, 3, 21):
        raise RuntimeError("O2-B incomplete saved visible trajectory")
    interface = R4Interface()
    interface.reset(trace["residual"][0], episode_id="o2-b-visible-replay")
    maximum = 0.0
    for step in range(steps):
        view = interface.snapshot()
        original, selected, choice, prediction = choose_command(view.features, view.valid, policy, selector, step=step, cfg=cfg)
        action = interface.issue(anchor_delta(view.features[:, -1], cfg["integrator"]), selected, step=step)
        for name, value in dict(history=view.features, valid=view.valid, original=original, selected=selected,
                                choice=choice, prediction=prediction, requested_delta=action.requested_delta_rad,
                                requested_modal=action.requested_modal_rad).items():
            maximum = max(maximum, compare_tensor(value, trace[name][step], name, cfg["thresholds"]["replay_max_absolute_error"]))
        transition = interface.observe_next(trace["residual"][step + 1], step=step + 1,
            power=PowerMeasurement(trace["measured_power"][step], int(trace["power_action_step"][step]),
                                   int(trace["power_arrival_step"][step])))
        if not bool(transition.action_power_valid.all()):
            raise RuntimeError("O2-B power failed causal arrival check")
        compare_tensor(transition.next_history.features[:, -1, 75:79], trace["next_clock"][step],
                       "observation/command/power clocks", 0.0)
    return dict(max_absolute_error=maximum, candidate_indices_exact=True, clocks_exact=True,
                mask_exact=True, replayed_steps=steps, new_environment_transitions=0,
                scope="saved_measurements_interface_and_model_decisions_same_device_backend_batch_layout")


def require_prefix(original: dict[str, torch.Tensor], selected: dict[str, torch.Tensor], start: int) -> None:
    for name in ("history", "valid", "original", "selected", "requested_delta", "requested_modal", "measured_power", "next_clock"):
        if not torch.equal(original[name][:start], selected[name][:start]):
            raise RuntimeError(f"O2-B selector-disabled paired prefix differs: {name}")
    if not torch.equal(original["residual"][:start + 1], selected["residual"][:start + 1]):
        raise RuntimeError("O2-B selector-disabled observations differ")


def _stack(parts: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack(value) for key, value in parts.items() if value}


@torch.no_grad()
def rollout(cfg: dict[str, Any], parent: dict, branch: dict, sensor, bridge, policy, selector,
            progress: Callable[[dict], None], context: dict) -> tuple[dict, dict, dict]:
    context.update(controller=branch["controller"], action_step=None)
    env = make_environment(parent, bridge.basis, cfg["weather_seed"], cfg["episode_length"])
    port = HolographicEnvironmentPort(env, sensor, bridge, cfg["thresholds"]["ideal_modal_error_max_rad"])
    readout, initial_audit = port.reset(cfg["weather_seed"])
    interface = R4Interface()
    interface.reset(readout.residual, episode_id=f"o2-b-{branch['controller']}")
    visible = {k: [] for k in ("history", "valid", "original", "selected", "choice", "prediction",
                              "requested_delta", "requested_modal", "residual", "measured_power", "next_clock",
                              "power_action_step", "power_arrival_step")}
    visible["residual"].append(readout.residual.clone())
    audits = {k: [v.clone()] for k, v in initial_audit.items()}
    actions_audit: dict[str, list[torch.Tensor]] = {}
    def on_transition() -> None:
        context["physical_transitions"] += 3
    for step in range(cfg["episode_length"]):
        context.update(controller=branch["controller"], action_step=step)
        view = interface.snapshot()
        if readout.observation_step != step or view.observation_step != step:
            raise RuntimeError("O2-B current observation clock mismatch")
        original, selected, choice, prediction = choose_command(view.features, view.valid, policy, selector, step=step, cfg=cfg)
        action = interface.issue(anchor_delta(view.features[:, -1], cfg["integrator"]), selected, step=step)
        readout, power, audit = port.step(action.requested_delta_rad, step, on_transition)
        transition = interface.observe_next(readout.residual, step=readout.observation_step, power=power)
        if not bool(transition.action_power_valid.all()):
            raise RuntimeError("O2-B missing properly indexed action power")
        for key, value in dict(history=view.features, valid=view.valid, original=original, selected=selected,
                               choice=choice, prediction=prediction, requested_delta=action.requested_delta_rad,
                               requested_modal=action.requested_modal_rad, measured_power=power.value,
                               next_clock=transition.next_history.features[:, -1, 75:79],
                               power_action_step=torch.tensor(power.action_step, device=bridge.device),
                               power_arrival_step=torch.tensor(power.arrival_observation_step, device=bridge.device)).items():
            visible[key].append(value.detach().clone())
        visible["residual"].append(readout.residual.clone())
        for key, value in audit.items():
            collection = audits if key in initial_audit else actions_audit
            collection.setdefault(key, []).append(value.detach().clone())
        progress(dict(controller=branch["controller"], observation_step=readout.observation_step,
                      physical_transitions=context["physical_transitions"], total_physical_transitions=cfg["physical_transitions"],
                      modal_error_max_rad=float(audit["modal_error_rad"].max())))
    trace, audit_trace = _stack(visible), {**_stack(audits), **_stack(actions_audit)}
    info = dict(controller=branch["controller"], complete_episodes=3, physical_transitions=3 * cfg["episode_length"],
                modal_error_max_rad=float(audit_trace["modal_error_rad"].max()),
                representation_fit_rmse_max_rad=float(audit_trace["fit_rmse_rad"].max()),
                legacy_reference_difference_max_rad=float((audit_trace["legacy_projection_rad"].double()
                                                            - audit_trace["joint_target_rad"]).abs().max()),
                policy_forward_calls=cfg["episode_length"] if policy is not None else 0,
                scorer_forward_calls=cfg["episode_length"] - cfg["selector_start_step"] if selector is not None else 0)
    return trace, audit_trace, info


def source_manifest(path: str | Path) -> dict[str, Any]:
    config_path = Path(path) if Path(path).is_absolute() else ROOT / path
    paths = ["observation_bridge/closed_loop.py", "scripts/verify_observation_bridge_o2_closed_loop.py",
             "tests/test_observation_bridge_o2_closed_loop.py", "scripts/evaluate_s4_r5_g2_d13_closed_loop.py",
             "scripts/train_s4_r5_g2_d12_candidate_scorer.py", "scripts/evaluate_s4_r5_g2_d12_held_out.py",
             "scripts/prepare_s4_r5_g2_d12_causal_dataset.py", "scripts/diagnose_s4_r5_g2_d11_same_state_r1.py",
             "configs/experiments/s4_hardware_stress_v1.yaml"]
    return dict(source_sha256={p: optics.file_sha256(ROOT / p) for p in paths},
                config_sha256=optics.file_sha256(config_path), sealed_src_bundle_sha256=optics.BUNDLE_SHA256)


def run(path: str | Path = CONFIG, *, preflight_only: bool = False) -> dict[str, Any]:
    cfg, output, device, parent, policies, scorers, models, report = preflight(path)
    if preflight_only:
        return report
    output.mkdir(parents=True, exist_ok=False)
    context: dict[str, Any] = dict(physical_transitions=0, controller=None, action_step=None)
    started = time.perf_counter()
    try:
        (output / "trajectories").mkdir()
        (output / "audit").mkdir()
        optics.write_json(output / "effective_config.json", cfg)
        optics.write_json(output / "preflight.json", report)
        optics.write_json(output / "model_manifest.json", models)
        optics.write_json(output / "stream_manifest.json", report["stream_manifest"])
        optics.write_json(output / "source_manifest.json", source_manifest(path))
        optics_cfg = yaml.safe_load((ROOT / cfg["optics_config"]).read_text(encoding="utf-8"))
        sensor, bridge = optics.make_components(optics_cfg, device)
        original_traces, records, artifacts = {}, [], {}
        fold = cfg["weather_index"] % 4
        spec = dict(policy_initializations=3, scorer_seeds=cfg["scorer_seeds"])
        branches = frozen.controller_specs(spec)
        def progress(row: dict) -> None:
            elapsed = time.perf_counter() - started
            count = row["physical_transitions"]
            row.update(elapsed_seconds=elapsed, eta_seconds=elapsed / count * (1248 - count),
                       cuda_allocated_gb=torch.cuda.memory_allocated(device) / 2**30)
            with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            if row["observation_step"] % 4 == 0 or count == 1248:
                print(f"O2-B 技术闭环 {count}/1248 | {row['controller']} 第{row['observation_step']}/32帧 | "
                      f"观测误差={row['modal_error_max_rad']:.3g} rad | 剩余={row['eta_seconds']:.1f}s | "
                      f"显存={row['cuda_allocated_gb']:.2f}GB", flush=True)
        for branch in branches:
            policy = policies.get(branch["member"])
            selector = None if branch["scorer_seed"] is None else scorers[fold, branch["scorer_seed"]]
            trace, audit, record = rollout(cfg, parent, branch, sensor, bridge, policy, selector, progress, context)
            name = branch["controller"] + ".pt"
            for directory, value in (("trajectories", trace), ("audit", audit)):
                saved = output / directory / name
                # 只在完整回合保存时转 CPU；不是每帧测量处理或控制回退。
                torch.save({k: v.cpu() for k, v in value.items()}, saved)
                artifacts[f"{directory}/{name}"] = optics.file_sha256(saved)
            # 从磁盘重新加载白名单记录重放；audit 不交给重放函数。
            loaded = torch.load(output / "trajectories" / name, map_location=device, weights_only=True)
            record["replay"] = replay_visible(loaded, cfg, policy, selector)
            if branch["member"] is not None and selector is None:
                original_traces[branch["member"]] = trace
            if selector is not None:
                require_prefix(original_traces[branch["member"]], trace, cfg["selector_start_step"])
                record["paired_prefix_exact"] = True
            records.append(record)
            with (output / "records.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        if context["physical_transitions"] != 1248 or sum(r["complete_episodes"] for r in records) != 39:
            raise RuntimeError("O2-B complete episode/transition budget mismatch")
        verify_optics_prerequisite(cfg)
        if read_json(output / "source_manifest.json") != source_manifest(path):
            raise RuntimeError("O2-B own code/config changed during execution")
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, encoding="utf-8", errors="replace", check=False)
        result = dict(status="O2_B_SYNTHETIC_CLOSED_LOOP_TECHNICAL_PASS_ONLY", **cfg["boundary"],
                      complete_episodes=39, physical_transitions=1248, controllers=13,
                      invalid_observations=0, failed_or_truncated_episodes=0,
                      max_modal_error_rad=max(r["modal_error_max_rad"] for r in records),
                      replay_max_absolute_error=max(r["replay"]["max_absolute_error"] for r in records),
                      paired_prefix_checks=9, action_reward_clock_checks=416,
                      policy_forward_calls=sum(r["policy_forward_calls"] for r in records),
                      scorer_forward_calls=sum(r["scorer_forward_calls"] for r in records),
                      replay_policy_forward_calls=sum(r["policy_forward_calls"] for r in records),
                      replay_scorer_forward_calls=sum(r["scorer_forward_calls"] for r in records),
                      replay_environment_transitions=0, records=records, analysis={},
                      output_directory=str(output), raw_camera_frames_saved=0,
                      runtime=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name(device), deterministic=True,
                                   allow_tf32=False, cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
                                   batch_layout="three_families_one_nominal_profile"),
                      git_head=git.stdout.strip() if git.returncode == 0 else None,
                      elapsed_seconds=time.perf_counter() - started,
                      inverse_crime_limitation=True, real_accuracy_verified=False, realtime_verified=False,
                      next_action="Read-only audit first; then prepare O2-C development, not training or independent confirmation.")
        optics.write_json(output / "summary.json", result)
        artifacts.update({p.name: optics.file_sha256(p) for p in output.iterdir() if p.is_file()})
        optics.write_json(output / "SUCCESS.json", dict(status=result["status"], artifact_sha256=artifacts,
                                                       summary_sha256=artifacts["summary.json"]))
        return result
    except BaseException as exc:
        optics.write_json(output / "failure.json", dict(status="O2_B_TECHNICAL_FAILED", exception=type(exc).__name__,
                         message=str(exc), traceback=traceback.format_exc(), last_context=context, **cfg["boundary"]))
        raise
