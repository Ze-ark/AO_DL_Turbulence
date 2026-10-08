"""O2 小型确定性技术单元测试；不是动态仿真成绩、训练或硬件验证。"""
from dataclasses import replace
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from observation_bridge.adapter import FieldOrientation, FrozenModalObservationBridge, O1Tolerances
from observation_bridge.cuda_modal_observation import CudaModalObservationBridge, O2ModalTolerances
from observation_bridge.holography import (
    SyntheticOffAxisSensor, SyntheticOpticsCalibration, centered_fft2, centered_ifft2, interpolate_complex_field,
)
from scripts.verify_observation_bridge_o2_optics import (
    ROOT, audit_known_phase, fixture_groups, make_components, preflight, run,
)
from src.runtime import resolve_device
from src.simulation.modes import synthesize_phase


@pytest.fixture(scope="module")
def config():
    return yaml.safe_load((ROOT / "configs/experiments/observation_bridge_o2_optics_v1.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def components(config):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable: GPU checks unrun, not CPU fallback")
    return make_components(config, resolve_device("cuda"))


@pytest.mark.parametrize("shape,target", [((3, 5), (8, 12)), ((4, 7), (9, 14)),
                                          ((5, 4), (10, 9)), ((6, 8), (12, 16))])
def test_cpu_tiny_asymmetric_odd_even_fft_and_complex_amplitude(shape, target):
    # 仅十几个像素的数学检查；正式相机入口明确拒绝 CPU。
    y = torch.arange(shape[0], dtype=torch.float64) - shape[0] // 2
    x = torch.arange(shape[1], dtype=torch.float64) - shape[1] // 2
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    field = torch.polar(torch.full_like(yy, 1.7), 2 * torch.pi * (yy / shape[0] - xx / shape[1]))[None]
    spectrum = centered_fft2(field)
    peak = spectrum.abs().flatten().argmax()
    expected_index = (shape[0] // 2 + 1) * shape[1] + shape[1] // 2 - 1
    assert int(peak) == expected_index
    torch.testing.assert_close(centered_ifft2(spectrum), field, atol=1e-12, rtol=0)
    enlarged = interpolate_complex_field(field, target)
    expanded_spectrum = centered_fft2(enlarged)
    start_y, start_x = target[0] // 2 - shape[0] // 2, target[1] // 2 - shape[1] // 2
    recovered = centered_ifft2(expanded_spectrum[:, start_y:start_y + shape[0], start_x:start_x + shape[1]]
                             * (shape[0] * shape[1] / (target[0] * target[1])) ** 0.5)
    torch.testing.assert_close(recovered, field, atol=1e-12, rtol=0)
    torch.testing.assert_close(enlarged.abs(), torch.full_like(enlarged.real, 1.7), atol=1e-12, rtol=0)


def test_cpu_tiny_even_nyquist_asymmetric_peak_is_not_discarded():
    spectrum = torch.zeros(1, 4, 6, dtype=torch.complex128)
    spectrum[0, 0, 0] = 2 + 3j  # 两轴都为负 Nyquist 频率，不能因共轭边界丢失。
    field = centered_ifft2(spectrum)
    expanded = centered_fft2(interpolate_complex_field(field, (12, 18)))
    assert expanded[0, 4, 6] == pytest.approx((2 + 3j) * 3, abs=1e-12)


@pytest.mark.parametrize("shape", [(2, 3), (0, 10), (True, 10)])
def test_cpu_primitive_rejects_bad_target(shape):
    with pytest.raises(ValueError):
        interpolate_complex_field(torch.ones(1, 3, 5, dtype=torch.complex128), shape)


@pytest.mark.parametrize("name", ["zero_phase", "signed_mode_pulses", "seeded_mixtures",
                                  "wrapped_smooth_phase", "wrapped_constant_reference"])
def test_cuda_47_positive_known_fixtures(config, components, name):
    sensor, bridge = components
    groups = fixture_groups(config["fixture_seed"], bridge.device)
    assert sum(len(c) for _, c, _ in groups) == 47
    _, coefficients, constant = next(g for g in groups if g[0] == name)
    for offset in range(0, len(coefficients), 4):
        known = coefficients[offset:offset + 4]
        phase = synthesize_phase(known.double(), bridge.basis.double()) + constant
        field = (torch.polar(torch.ones_like(phase), phase) * bridge.pupil).to(torch.complex64)
        intensity = sensor.render(field)
        rebuilt = sensor.reconstruct(intensity)
        measured = bridge.measure(rebuilt)
        expected, true_jump = audit_known_phase(phase, bridge)
        assert true_jump <= 1.5
        relative = ((rebuilt - field).abs().square().sum((-2, -1)) / field.abs().square().sum((-2, -1))).sqrt()
        assert relative.max() < 1e-5
        torch.testing.assert_close(measured.residual_rad.double(), expected, atol=1e-4, rtol=0)
        assert measured.residual_rad.device.type == "cuda"
        assert measured.residual_rad.dtype == torch.float32
        if name.startswith("wrapped"):
            assert phase[:, bridge.pupil].abs().max() > torch.pi


def test_cuda_opposite_sideband_conjugates_full_complex_field_with_nyquist(components):
    sensor, bridge = components
    coefficient = torch.zeros(1, 21, device=bridge.device)
    coefficient[0, 0], coefficient[0, 2] = 4.0, 0.7
    phase = synthesize_phase(coefficient.double(), bridge.basis.double()) + 0.4
    field = (torch.polar(torch.ones_like(phase), phase) * bridge.pupil).to(torch.complex64)
    opposite = sensor.reconstruct_opposite_for_test(sensor.render(field))
    torch.testing.assert_close(opposite, field.conj(), atol=1e-5, rtol=0)
    torch.testing.assert_close(bridge.measure(opposite).residual_rad, -coefficient, atol=1e-4, rtol=0)


def test_cuda_non21_subspace_records_residual_and_does_not_use_o1_strict_gate(components):
    sensor, bridge = components
    phase = torch.zeros(1, 64, 64, dtype=torch.float64, device=bridge.device)
    phase[0, 31, 31] = 0.01
    field = (torch.polar(torch.ones_like(phase), phase) * bridge.pupil).to(torch.complex64)
    measured = bridge.measure(sensor.reconstruct(sensor.render(field)))
    expected, _ = audit_known_phase(phase, bridge)
    assert measured.fit_rmse_rad.min() > 1e-5  # 不是相机失败，也不能抹掉高阶残差。
    torch.testing.assert_close(measured.residual_rad.double(), expected, atol=1e-4, rtol=0)


@pytest.mark.parametrize("constant", [0.0, 3.7, -7.0])
def test_cuda_parity_with_o1_only_tiny_explicit_unit_reference(components, constant):
    sensor, bridge = components
    coefficients = torch.zeros(1, 21, device=bridge.device)
    coefficients[0, 0], coefficients[0, 2] = 4.0, 0.7
    phase = synthesize_phase(coefficients.double(), bridge.basis.double()) + constant
    field = torch.polar(torch.ones_like(phase), phase).to(torch.complex64)
    gpu = bridge.measure(sensor.reconstruct(sensor.render(field * bridge.pupil)))
    # 唯一允许测量复制到 CPU 的场景：显式标注的小型参考一致性单元测试。
    reference = FrozenModalObservationBridge(bridge.contract, O1Tolerances(1e-4, 1.5, 1e-5, 1e-5))
    cpu = reference.measure(field.cpu())
    torch.testing.assert_close(gpu.residual_rad.cpu(), cpu.residual_rad, atol=1e-4, rtol=0)


@pytest.mark.parametrize("fault", ["zero", "hole", "low_intensity", "nan", "inf", "wrong_grid", "real", "no_batch", "oversized"])
def test_cuda_modal_bad_input_fails_closed(components, fault):
    _, bridge = components
    field = torch.ones(1, 64, 64, dtype=torch.complex64, device=bridge.device)
    if fault == "zero":
        field.zero_()
    elif fault == "hole":
        field[0, 31, 31] = 0
    elif fault == "low_intensity":
        field[0, 31, 31] = 0.001
    elif fault in ("nan", "inf"):
        field[0, 0, 0] = float(fault)
    elif fault == "wrong_grid":
        field = field[:, :63]
    elif fault == "real":
        field = field.real
    elif fault == "no_batch":
        field = field[0]
    else:
        field = field.repeat(65, 1, 1)
    with pytest.raises(ValueError):
        bridge.measure(field)


@pytest.mark.parametrize("fault", ["nan", "negative", "grid", "complex", "future_stack"])
def test_cuda_camera_invalid_current_intensity_rejected(components, fault):
    sensor, bridge = components
    intensity = torch.ones(1, 512, 512, device=bridge.device)
    if fault == "nan":
        intensity[0, 0, 0] = float("nan")
    elif fault == "negative":
        intensity[0, 0, 0] = -1
    elif fault == "grid":
        intensity = intensity[:, :511]
    elif fault == "complex":
        intensity = intensity.to(torch.complex64)
    else:
        intensity = intensity[:, None]  # 不接 [batch,time,row,column] 未来序列。
    with pytest.raises(ValueError):
        sensor.reconstruct(intensity)


def test_cuda_spatial_jump_and_loop_singularity_rejected(components):
    _, bridge = components
    phase = (torch.arange(64, device=bridge.device, dtype=torch.float64) % 2 * 2).expand(1, 64, 64)
    with pytest.raises(ValueError, match="spatial phase jump"):
        bridge.measure(torch.polar(torch.ones_like(phase), phase))
    coordinates = torch.arange(64, device=bridge.device, dtype=torch.float64) - 31.5
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    phase = torch.atan2(y, x)[None]
    relaxed = CudaModalObservationBridge(bridge.contract, replace(bridge.tolerances, max_neighbor_jump_rad=2.0), bridge.device)
    with pytest.raises(ValueError, match="loop inconsistency"):
        relaxed.measure(torch.polar(torch.ones_like(phase), phase))


def test_cuda_sampling_guard_is_not_a_two_pi_alias_detector(components):
    _, bridge = components
    phase = (2 * torch.pi * torch.arange(64, device=bridge.device, dtype=torch.float64)).expand(1, 64, 64)
    result = bridge.measure(torch.polar(torch.ones_like(phase), phase))
    assert result.residual_rad.abs().max() < 1e-4
    with pytest.raises(ValueError, match="spatial sampling"):
        replace(bridge.contract, spatial_sampling_verified=False).validate()


def test_cuda_whitelist_stateless_and_fixed_crop_with_no_host_tensor_transfer(components, monkeypatch):
    sensor, bridge = components
    coefficients = torch.zeros(1, 21, device=bridge.device)
    coefficients[0, 3] = 0.1
    phase = synthesize_phase(coefficients.double(), bridge.basis.double())
    field = (torch.polar(torch.ones_like(phase), phase) * bridge.pupil).to(torch.complex64)
    original = field.clone()
    intensity = sensor.render(field)
    before = intensity.clone()
    # 已创建静态几何以后，当前链不得 .cpu()/.numpy()/.tolist() 回传测量张量。
    def forbidden(*args, **kwargs):
        raise AssertionError("current tensor CPU roundtrip")
    with monkeypatch.context() as patch:
        for method in ("cpu", "numpy", "tolist"):
            patch.setattr(torch.Tensor, method, forbidden)
        first = bridge.measure(sensor.reconstruct(intensity))
        sensor.reconstruct(intensity * 1.1)  # 后一帧不改变前一帧或窗口。
        again = bridge.measure(sensor.reconstruct(intensity))
    assert torch.equal(first.residual_rad, again.residual_rad)
    assert torch.equal(field, original) and torch.equal(intensity, before)
    assert list(inspect.signature(sensor.reconstruct).parameters) == ["current_intensity"]
    assert list(inspect.signature(bridge.measure).parameters) == ["current_field"]
    assert "power" not in first.__dataclass_fields__
    assert sensor.calibration.crop_rows == (96, 160)
    copy = bridge.basis
    copy.zero_()
    assert bridge.basis.abs().max() > 0


def test_cuda_components_reject_cpu_measurements_without_transfer(components):
    sensor, bridge = components
    with pytest.raises(ValueError, match="CUDA device"):
        sensor.reconstruct(torch.ones(1, 512, 512))
    with pytest.raises(ValueError, match="CUDA device"):
        sensor.render(torch.ones(1, 64, 64, dtype=torch.complex64))
    with pytest.raises(ValueError, match="CUDA device"):
        bridge.measure(torch.ones(1, 64, 64, dtype=torch.complex64))
    with pytest.raises(ValueError, match="requires CUDA"):
        SyntheticOffAxisSensor(sensor.calibration, torch.device("cpu"))
    with pytest.raises(ValueError, match="requires CUDA"):
        CudaModalObservationBridge(bridge.contract, bridge.tolerances, torch.device("cpu"))


@pytest.mark.parametrize("field,value", [("measurement_plane", "unknown"), ("reference_amplitude", 2.0),
                                        ("orientation", None), ("crop_rows", (95, 159))])
def test_unknown_optics_contract_not_silently_inferred(config, field, value):
    optics = dict(config["optics"])
    optics["orientation"] = FieldOrientation(**optics["orientation"])
    for name in ("reference_carrier_bins_yx", "crop_rows", "crop_columns"):
        optics[name] = tuple(optics[name])
    with pytest.raises(ValueError):
        replace(SyntheticOpticsCalibration(**optics), **{field: value}).validate()


@pytest.mark.parametrize("value", [FieldOrientation(None, False, False, 1),
                                  FieldOrientation(False, False, False, 0),
                                  FieldOrientation(False, False, False, True)])
def test_unknown_orientation_rejected_without_gpu(value):
    with pytest.raises(ValueError):
        value.validate()


def test_preflight_readonly_preserves_output_and_frozen_files(config, tmp_path, monkeypatch):
    import scripts.verify_observation_bridge_o2_optics as entry
    path = tmp_path / "config.yaml"
    cfg = dict(config)
    cfg["output_directory"] = "outputs/observation_bridge_o2_optics_v1"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    if torch.cuda.is_available() and not (ROOT / cfg["output_directory"]).exists():
        report = run(path, preflight_only=True)
        assert report["frozen_C1_files_checked"] == 55
        assert not (ROOT / cfg["output_directory"]).exists()
    # 小型独立工作区验证拒绝覆盖和逃逸路径，不复制历史大权重。
    local_root = tmp_path / "workspace"
    (local_root / "configs/experiments").mkdir(parents=True)
    design_bytes = (ROOT / config["design_path"]).read_bytes()
    (local_root / config["design_path"]).write_bytes(design_bytes)
    kept = local_root / "outputs/kept"
    kept.mkdir(parents=True)
    sentinel = kept / "original.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(entry, "ROOT", local_root)
    cfg["output_directory"] = "outputs/kept"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(FileExistsError, match="preserve existing"):
        run(path)
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    for output in ("outputs", "../outside"):
        cfg["output_directory"] = output
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        with pytest.raises(ValueError, match="new child"):
            preflight(path)


def test_missing_cuda_preflight_fails_not_cpu_fallback(config, monkeypatch):
    import scripts.verify_observation_bridge_o2_optics as entry
    monkeypatch.setattr(entry, "check_sources", lambda _: {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # 不受技术输出完成与否影响，只针对设备解析守卫。
    monkeypatch.setattr(Path, "exists", lambda self: False)
    with pytest.raises(RuntimeError, match="CUDA enabled"):
        preflight("configs/experiments/observation_bridge_o2_optics_v1.yaml")


def test_config_cannot_increase_budget_or_relax_gate(config, tmp_path):
    path = tmp_path / "invalid.yaml"
    for name, value in (("positive_fixture_count", 48), ("device", "cpu"),
                        ("thresholds", {"relative_complex_field_l2_error_maximum": 0.1,
                                        "modal_coefficient_max_error_rad": 1.0})):
        cfg = {**config, name: value}
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        with pytest.raises(ValueError):
            preflight(path)


def test_imports_inert_cli_help_and_no_frozen_src_edits():
    code = "from pathlib import Path\nfrom unittest.mock import patch\nimport torch\nwith patch.object(Path,'mkdir',side_effect=AssertionError('write')), patch.object(torch.cuda,'init',side_effect=AssertionError('GPU')), patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('GPU')):\n import observation_bridge.holography\n import observation_bridge.cuda_modal_observation\n import scripts.verify_observation_bridge_o2_optics\n"
    subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, check=True, capture_output=True)
    result = subprocess.run([sys.executable, "-X", "utf8", "-B", "scripts/verify_observation_bridge_o2_optics.py", "--help"],
                            cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8")
    assert "--preflight-only" in result.stdout
    from scripts.verify_observation_bridge_o2_optics import BUNDLE_SHA256, DESIGN_SHA256, file_sha256, sealed_bundle_sha256
    assert sealed_bundle_sha256() == BUNDLE_SHA256
    assert file_sha256(ROOT / "configs/experiments/observation_bridge_o2_design_v1.json") == DESIGN_SHA256
