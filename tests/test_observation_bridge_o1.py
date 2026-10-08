"""O1 小型确定性 CPU 单元测试；不产生光学排名或真实硬件结论。"""
from dataclasses import replace
import inspect
from itertools import product
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from scripts.verify_observation_bridge_o1 import known_field, preflight, run
from observation_bridge.adapter import (
    FieldOrientation, FrozenModalObservationBridge, O1Tolerances,
    SyntheticFieldContract, orient_complex_field,
)
from src.rl.r4_observation import PowerMeasurement, R4Interface
from src.rl.r5_physics_adapter import causal_policy_features

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bridge():
    cfg = yaml.safe_load((ROOT / "configs/experiments/observation_bridge_o1_v1.yaml").read_text(encoding="utf-8"))
    values = dict(cfg["contract"])
    values["orientation"] = FieldOrientation(**values["orientation"])
    return FrozenModalObservationBridge(SyntheticFieldContract(**values), O1Tolerances(**cfg["tolerances"]))


@pytest.mark.parametrize("transpose,flip_y,flip_x,phase_sign", list(product((False, True), (False, True), (False, True), (-1, 1))))
def test_asymmetric_non_square_axis_and_sign(transpose, flip_y, flip_x, phase_sign):
    canonical = torch.complex(torch.arange(15).reshape(1, 3, 5).double(),
                              torch.arange(20, 35).reshape(1, 3, 5).double())
    raw = canonical.conj() if phase_sign == -1 else canonical.clone()
    if flip_x:
        raw = raw.flip(-1)
    if flip_y:
        raw = raw.flip(-2)
    if transpose:
        raw = raw.transpose(-2, -1)
    original = raw.clone()
    result = orient_complex_field(raw, FieldOrientation(transpose, flip_y, flip_x, phase_sign))
    assert torch.equal(result, canonical)
    assert torch.equal(raw, original)
    assert result.data_ptr() != raw.data_ptr()


def test_all_21_modes_both_signs_roundtrip(bridge):
    coefficients = torch.cat((torch.eye(21), -torch.eye(21))) * 0.05
    field, _ = known_field(coefficients, bridge)
    measured = bridge.measure(field)
    torch.testing.assert_close(measured.residual_rad, coefficients, atol=1e-5, rtol=0)
    assert measured.residual_rad.shape == (42, 21)
    assert measured.residual_rad.dtype == torch.float32
    assert measured.fit_rmse_rad.max() < 1e-5


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("constant", [-7.0, -3.7, 0.0, 3.7, 8.0])
def test_wrapped_phase_global_reference_and_tilt_are_preserved(bridge, dtype, constant):
    coefficients = torch.zeros(1, 21)
    coefficients[0, 0], coefficients[0, 2], coefficients[0, 10] = 4.0, 0.7, 0.1
    field, phase = known_field(coefficients, bridge, constant=constant)
    measured = bridge.measure(field.to(dtype))
    assert (phase[:, bridge.pupil].abs() > torch.pi).any()
    torch.testing.assert_close(measured.residual_rad, coefficients, atol=1e-5, rtol=0)
    assert measured.residual_rad[0, 0] == pytest.approx(4.0, abs=1e-5)
    # 常量参考只识别到 2π；不会误称恢复了绝对光学相位。
    phase_difference = measured.unwrapped_phase_rad[:, bridge.pupil] - phase[:, bridge.pupil]
    circular_difference = torch.atan2(phase_difference.sin(), phase_difference.cos())
    assert circular_difference.abs().max() < 1e-5


def test_conjugation_has_known_modal_sign(bridge):
    coefficients = torch.randn(2, 21, generator=torch.Generator().manual_seed(831001)) * 0.02
    field, _ = known_field(coefficients, bridge)
    torch.testing.assert_close(bridge.measure(field.conj()).residual_rad, -coefficients, atol=1e-5, rtol=0)
    corrected = FrozenModalObservationBridge(
        replace(bridge.contract, orientation=FieldOrientation(False, False, False, -1)), bridge.tolerances)
    torch.testing.assert_close(corrected.measure(field.conj()).residual_rad, coefficients, atol=1e-5, rtol=0)


def test_relative_amplitude_does_not_fabricate_power(bridge):
    coefficients = torch.zeros(1, 21)
    coefficients[0, 4] = 0.1
    field, _ = known_field(coefficients, bridge)
    first, scaled = bridge.measure(field), bridge.measure(field * 300)
    torch.testing.assert_close(first.residual_rad, scaled.residual_rad, atol=1e-5, rtol=0)
    assert "power" not in first.__dataclass_fields__
    assert not hasattr(first, "arrived_power")


@pytest.mark.parametrize("fault", ["zero", "hole", "low_intensity", "nan", "inf", "wrong_grid", "real_dtype", "no_batch", "oversized_batch"])
def test_bad_field_fails_closed(bridge, fault):
    field = torch.ones(1, 64, 64, dtype=torch.complex128)
    r, c = bridge.pupil.nonzero()[0].tolist()
    if fault == "zero":
        field.zero_()
    elif fault == "hole":
        field[0, r, c] = 0
    elif fault == "low_intensity":
        field[0, r, c] = 1e-3
    elif fault in ("nan", "inf"):
        field[0, 0, 0] = float(fault)
    elif fault == "wrong_grid":
        field = field[:, :63]
    elif fault == "real_dtype":
        field = field.real
    elif fault == "no_batch":
        field = field[0]
    else:
        field = field.repeat(65, 1, 1)
    with pytest.raises(ValueError):
        bridge.measure(field)


@pytest.mark.parametrize("field,value", [
    ("source_kind", "real_matlab_hdf5"), ("measurement_plane", "unknown"),
    ("phase_unit", "degree"), ("reference", "unknown"),
    ("spatial_sampling_verified", False), ("orientation", None),
])
def test_unknown_contract_and_real_source_rejected(bridge, field, value):
    with pytest.raises(ValueError):
        FrozenModalObservationBridge(replace(bridge.contract, **{field: value}), bridge.tolerances)


@pytest.mark.parametrize("orientation", [FieldOrientation(None, False, False, 1),
                                       FieldOrientation(False, False, False, 0),
                                       FieldOrientation(False, False, False, True)])
def test_unknown_orientation_is_not_assumed(orientation):
    with pytest.raises(ValueError):
        orient_complex_field(torch.ones(3, 5, dtype=torch.complex128), orientation)


def test_spatial_jump_guard(bridge):
    phase = (torch.arange(64, dtype=torch.float64) % 2 * 2.0).expand(1, 64, 64)
    with pytest.raises(ValueError, match="spatial phase jump"):
        bridge.measure(torch.polar(torch.ones_like(phase), phase))


def test_loop_inconsistency_is_rejected(bridge):
    coordinates = torch.arange(64, dtype=torch.float64) - 31.5
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    phase = torch.atan2(y, x)[None]
    guarded = FrozenModalObservationBridge(bridge.contract, replace(bridge.tolerances, max_neighbor_jump_rad=2.0))
    with pytest.raises(ValueError, match="loop inconsistency"):
        guarded.measure(torch.polar(torch.ones_like(phase), phase))


def test_unrepresented_local_phase_is_not_mislabeled_as_modes(bridge):
    phase = torch.zeros(1, 64, 64, dtype=torch.float64)
    phase[0, 31, 31] = 0.01
    with pytest.raises(ValueError, match="does not fit"):
        bridge.measure(torch.polar(torch.ones_like(phase), phase))


def test_sampling_is_a_precondition_not_an_alias_detector(bridge):
    # 每像素多转一整圈与零相位的复场不可区分；必须由来源/采样验证排除。
    aliased = (2 * torch.pi * torch.arange(64, dtype=torch.float64)).expand(1, 64, 64)
    recovered = bridge.measure(torch.polar(torch.ones_like(aliased), aliased)).residual_rad
    assert recovered.abs().max() < 1e-5
    assert float((aliased[..., 1:] - aliased[..., :-1]).abs().max()) > torch.pi
    with pytest.raises(ValueError, match="spatial sampling"):
        replace(bridge.contract, spatial_sampling_verified=False).validate()


def test_residual_only_and_no_double_pseudo_open_loop(bridge):
    coefficients = torch.zeros(1, 21)
    coefficients[0, 0] = 0.2
    field, _ = known_field(coefficients, bridge)
    measured = bridge.measure(field).residual_rad
    adapter = R4Interface()
    adapter.reset(measured, episode_id="synthetic-known-initial-state")
    adapter.issue(torch.zeros_like(measured), torch.full((1, 11), 0.2), step=0)
    transition = adapter.observe_next(measured, step=1)
    view = transition.next_history
    assert list(inspect.signature(bridge.measure).parameters) == ["field"]
    torch.testing.assert_close(view.features[:, -1, :21], coefficients, atol=1e-5, rtol=0)
    transformed = causal_policy_features(view.features, view.valid)
    torch.testing.assert_close(transformed[:, -1, :21], measured - view.features[:, -1, 42:63])
    assert not transition.action_power_valid.any()
    assert view.features[0, -1, 77] == -1
    assert view.features[0, -1, 78] == 0


def test_causality_reset_and_power_clock_use_existing_interface(bridge):
    coefficients = torch.zeros(1, 21)
    field, _ = known_field(coefficients, bridge)
    measured = bridge.measure(field).residual_rad
    adapter = R4Interface()
    adapter.reset(measured, episode_id="unit-temperature-A")
    adapter.issue(torch.zeros(1, 21), torch.zeros(1, 11), step=0)
    with pytest.raises(ValueError, match="future"):
        adapter.observe_next(measured, step=1, power=PowerMeasurement(torch.tensor([0.3]), 1, 1))
    first = adapter.observe_next(measured, step=1)
    before = first.next_history.features.clone()
    adapter.issue(torch.zeros(1, 21), torch.zeros(1, 11), step=1)
    second = adapter.observe_next(measured, step=2, power=PowerMeasurement(torch.tensor([0.3]), 0, 2))
    assert not second.action_power_valid.any()  # 迟到值不能成为动作 1 的标签。
    assert torch.equal(first.next_history.features, before)
    assert second.next_history.features[0, -1, 77] == 0
    reset = adapter.reset(measured, episode_id="unit-temperature-B")
    assert reset.episode_id == "unit-temperature-B" and reset.observation_step == 0
    assert reset.valid.sum() == 1
    assert reset.features[:, :-1].eq(0).all()
    assert reset.features[:, -1, 21:75].eq(0).all()
    assert reset.features[0, -1, 77] == -1 and reset.features[0, -1, 78] == 0


def test_no_state_or_mutation_or_future_field(bridge):
    coefficients = torch.zeros(1, 21)
    field, _ = known_field(coefficients, bridge)
    original = field.clone()
    first = bridge.measure(field)
    future = coefficients.clone()
    future[:, 3] = 0.1
    bridge.measure(known_field(future, bridge)[0])
    assert torch.equal(field, original)
    torch.testing.assert_close(first.residual_rad, coefficients, atol=1e-5, rtol=0)
    public_basis = bridge.basis
    public_basis.zero_()
    assert bridge.basis.abs().max() > 0


def test_non_cpu_input_is_rejected_without_transfer(bridge):
    # meta 无真实存储或 GPU 分配，检查设备守卫而非执行 CUDA 仿真。
    with pytest.raises(ValueError, match="not a CUDA fallback"):
        bridge.measure(torch.empty(1, 64, 64, device="meta", dtype=torch.complex64))


def test_config_preflight_does_not_write_and_existing_output_is_preserved(tmp_path, monkeypatch):
    cfg = yaml.safe_load((ROOT / "configs/experiments/observation_bridge_o1_v1.yaml").read_text(encoding="utf-8"))
    task_root = tmp_path / "workspace"
    existing = task_root / "outputs" / "kept"
    existing.mkdir(parents=True)
    evidence = existing / "original.txt"
    evidence.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr("scripts.verify_observation_bridge_o1.ROOT", task_root)
    path = tmp_path / "config.yaml"
    cfg["output_directory"] = "outputs/new_unit"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    report = run(path, preflight_only=True)
    assert report["status"] == "O1_UNIT_PREFLIGHT_ONLY"
    assert not (task_root / "outputs" / "new_unit").exists()
    cfg["output_directory"] = "outputs/kept"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(FileExistsError, match="preserve existing"):
        run(path)
    assert evidence.read_text(encoding="utf-8") == "preserve"
    cfg["output_directory"] = "outputs"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="new child"):
        preflight(path)
    cfg["output_directory"] = "../outside-unit-output"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="new child"):
        preflight(path)
    cfg["device"] = "cuda"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed unit"):
        preflight(path)


def test_import_is_inert_and_cli_help_is_available():
    code = "from pathlib import Path\nfrom unittest.mock import patch\nimport torch\nwith patch.object(Path,'mkdir',side_effect=AssertionError('write')), patch.object(torch.cuda,'init',side_effect=AssertionError('GPU')):\n import observation_bridge.adapter\n import scripts.verify_observation_bridge_o1\n"
    subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, check=True, capture_output=True)
    help_result = subprocess.run([sys.executable, "-X", "utf8", "-B", "scripts/verify_observation_bridge_o1.py", "--help"],
                                 cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8")
    assert "--preflight-only" in help_result.stdout
