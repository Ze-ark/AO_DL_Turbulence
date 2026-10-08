"""O2 理想瞳面共轭成像的人工离轴全息；不代表真实相机标定。

频谱补零的是复光场，不是包裹相位。这里只实现受控带限往返，
不传播光场、不寻找频谱峰、不读取未来帧或模态真值。
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from observation_bridge.adapter import FieldOrientation
from src.runtime import resolve_device


def centered_fft2(value: torch.Tensor) -> torch.Tensor:
    """末两轴为行 y、列 x；支持奇偶和非方形的小型数学单元测试。"""
    return torch.fft.fftshift(torch.fft.fft2(
        torch.fft.ifftshift(value, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def centered_ifft2(value: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.ifft2(
        torch.fft.ifftshift(value, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def interpolate_complex_field(field: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """人为带限插值，保留振幅；不把它解释为真实光学放大率。"""
    if field.ndim < 2 or not field.is_complex():
        raise ValueError("complex field with two spatial axes required")
    if (len(shape) != 2 or any(type(n) is not int for n in shape)
            or any(m < n for m, n in zip(shape, field.shape[-2:]))):
        raise ValueError("target axes must not be smaller than source axes")
    ny, nx = field.shape[-2:]
    my, mx = shape
    padded = torch.zeros((*field.shape[:-2], my, mx), dtype=field.dtype, device=field.device)
    row, column = my // 2 - ny // 2, mx // 2 - nx // 2
    padded[..., row:row + ny, column:column + nx] = centered_fft2(field) * math.sqrt(my * mx / (ny * nx))
    return centered_ifft2(padded)


@dataclass(frozen=True)
class SyntheticOpticsCalibration:
    measurement_plane: str
    pupil_grid_size: int
    camera_grid_size: int
    reference_amplitude: float
    reference_constant_phase_rad: float
    reference_carrier_bins_yx: tuple[int, int]
    crop_rows: tuple[int, int]
    crop_columns: tuple[int, int]
    orientation: FieldOrientation

    def validate(self) -> None:
        fixed = {
            "measurement_plane": "ideal_conjugate_image_of_frozen_simulation_pupil",
            "pupil_grid_size": 64, "camera_grid_size": 512,
            "reference_amplitude": 1.0, "reference_constant_phase_rad": 0.0,
            "reference_carrier_bins_yx": (128, 128),
            "crop_rows": (96, 160), "crop_columns": (96, 160),
        }
        if any(type(getattr(self, key)) is not type(value) or getattr(self, key) != value
               for key, value in fixed.items()):
            raise ValueError("O2-A requires the fixed declared synthetic optical calibration")
        if not isinstance(self.orientation, FieldOrientation):
            raise ValueError("unknown camera orientation")
        self.orientation.validate()
        if self.orientation != FieldOrientation(False, False, False, 1):
            raise ValueError("O2-A camera axes and sideband sign must be explicitly canonical")


class SyntheticOffAxisSensor:
    """无状态 GPU 人工相机及固定窗口重建。控制入口只接当前强度图。"""

    def __init__(self, calibration: SyntheticOpticsCalibration, device: torch.device):
        calibration.validate()
        if device.type != "cuda":
            raise ValueError("O2 measurement requires CUDA; no CPU fallback")
        self.device = resolve_device(str(device))
        self.calibration = calibration
        size = calibration.camera_grid_size
        # 双精度预计算载频，避免 float32 的大角度三角函数产生伪误差。
        coordinate = torch.arange(size, device=self.device, dtype=torch.float64) - size // 2
        y, x = torch.meshgrid(coordinate, coordinate, indexing="ij")
        ky, kx = calibration.reference_carrier_bins_yx
        phase = 2 * math.pi * (ky * y + kx * x) / size + calibration.reference_constant_phase_rad
        self._reference = torch.polar(torch.full_like(phase, calibration.reference_amplitude), phase).to(torch.complex64)

    def _validate(self, value: torch.Tensor, size: int, *, complex_field: bool) -> None:
        if value.device.type != "cuda" or value.device != self._reference.device:
            raise ValueError("measurement tensor must stay on the configured CUDA device")
        allowed = (torch.complex64, torch.complex128) if complex_field else (torch.float32, torch.float64)
        if (value.ndim != 3 or not 1 <= value.shape[0] <= 64
                or value.shape[-2:] != (size, size) or value.dtype not in allowed):
            raise ValueError("invalid current-frame batch, grid or dtype")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite measurement")
        if not complex_field and bool((value < 0).any()):
            raise ValueError("negative camera intensity")

    @torch.no_grad()
    def render(self, current_pupil_field: torch.Tensor) -> torch.Tensor:
        """生成器可读当前完整物光；不先投影到 21 模态。"""
        self._validate(current_pupil_field, 64, complex_field=True)
        camera_field = interpolate_complex_field(current_pupil_field, (512, 512))
        return (camera_field + self._reference).abs().square()

    @torch.no_grad()
    def reconstruct(self, current_intensity: torch.Tensor) -> torch.Tensor:
        """负载频窗口取 O R*；无相位真值、未来帧或环境对象参数。"""
        self._validate(current_intensity, 512, complex_field=False)
        rows, columns = self.calibration.crop_rows, self.calibration.crop_columns
        spectrum = centered_fft2(current_intensity)
        selected = spectrum[:, rows[0]:rows[1], columns[0]:columns[1]]
        return centered_ifft2(selected * (64 / 512) / self.calibration.reference_amplitude)

    @torch.no_grad()
    def reconstruct_opposite_for_test(self, current_intensity: torch.Tensor) -> torch.Tensor:
        """仅用于方向单元检查，取正侧 O* R；不供控制器选侧。

        偶数网格的共轭频谱支撑为 [-31,32]，不是 [-32,31]。
        因此取 [353:417] 并将 +32 折回 -32；漏掉此边界会伪造误差。
        """
        self._validate(current_intensity, 512, complex_field=False)
        spectrum = centered_fft2(current_intensity)
        selected = spectrum[:, 353:417, 353:417].roll((1, 1), dims=(-2, -1))
        return centered_ifft2(selected * (64 / 512) / self.calibration.reference_amplitude)
