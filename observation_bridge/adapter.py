"""O1 人工复光场到冻结 21 模态的参考转换；不是实测标定或实时控制器。

模块独立于已封存的 src 源码包。仅接受小型、已知瞳面及足够空间采样的
CPU 人工单元例子。相位解包裹
无法仅凭复场排除空间混叠；调用者的采样声明不能替代实际标定。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import torch

from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.simulation.config import S1EnvConfig


@dataclass(frozen=True)
class FieldOrientation:
    """顺序固定为：转置 → 行翻转 → 列翻转 → 相位符号校正。无隐式默认。"""

    transpose: bool
    flip_y: bool
    flip_x: bool
    phase_sign: int

    def validate(self) -> None:
        if any(type(v) is not bool for v in (self.transpose, self.flip_y, self.flip_x)):
            raise ValueError("orientation must be explicitly known")
        if type(self.phase_sign) is not int or self.phase_sign not in (-1, 1):
            raise ValueError("phase sign must be explicitly +1 or -1")


@dataclass(frozen=True)
class SyntheticFieldContract:
    source_kind: str
    measurement_plane: str
    phase_unit: str
    reference: str
    spatial_sampling_verified: bool
    orientation: FieldOrientation

    def validate(self) -> None:
        if (self.source_kind != "synthetic_known_pupil"
                or self.measurement_plane != "frozen_simulation_pupil"
                or self.phase_unit != "rad"
                or self.reference != "joint_constant_and_frozen_modes"):
            raise ValueError("O1 only supports the declared synthetic pupil/reference contract")
        if self.spatial_sampling_verified is not True:
            raise ValueError("unknown spatial sampling; cannot unwrap safely")
        if not isinstance(self.orientation, FieldOrientation):
            raise ValueError("unknown orientation")
        self.orientation.validate()


@dataclass(frozen=True)
class O1Tolerances:
    min_relative_intensity: float
    max_neighbor_jump_rad: float
    loop_consistency_rad: float
    max_fit_rmse_rad: float

    def validate(self) -> None:
        if any(isinstance(v, bool) or not math.isfinite(v) for v in (
            self.min_relative_intensity, self.max_neighbor_jump_rad,
            self.loop_consistency_rad, self.max_fit_rmse_rad,
        )):
            raise ValueError("finite numeric tolerances required")
        if not 0 < self.min_relative_intensity <= 1:
            raise ValueError("invalid relative intensity floor")
        if not 0 < self.max_neighbor_jump_rad < math.pi:
            raise ValueError("neighbor jump limit must be below pi")
        if not 0 < self.loop_consistency_rad < 0.01 or not 0 < self.max_fit_rmse_rad < 0.01:
            raise ValueError("invalid O1 phase consistency tolerance")


@dataclass(frozen=True)
class ModalObservation:
    residual_rad: torch.Tensor
    constant_phase_rad: torch.Tensor
    unwrapped_phase_rad: torch.Tensor
    pupil: torch.Tensor
    fit_rmse_rad: torch.Tensor
    max_neighbor_jump_rad: torch.Tensor
    minimum_relative_intensity: torch.Tensor


def orient_complex_field(field: torch.Tensor, orientation: FieldOrientation) -> torch.Tensor:
    """只处理明确的轴变换，不裁剪、插值、归一化强度，也不生成功率读数。"""
    orientation.validate()
    if field.ndim < 2 or not field.is_complex():
        raise ValueError("complex field with two spatial axes required")
    result = field.transpose(-2, -1) if orientation.transpose else field
    if orientation.flip_y:
        result = result.flip(-2)
    if orientation.flip_x:
        result = result.flip(-1)
    if orientation.phase_sign == -1:
        result = result.conj()
    return result.clone()


class FrozenModalObservationBridge:
    """光瞳图上的四邻域参考解包裹，再联合拟合常量相位和原 21 模态。

    不先减相位均值：离散冻结模态可能有非零均值。联合常量项避免把
    参考相位误投进模态，也不重造或更改冻结基。常量项仅确定到 2π。
    此 CPU 参考实现无实时性承诺；未知标定、孔洞和不一致绕行均拒绝。
    """

    def __init__(self, contract: SyntheticFieldContract, tolerances: O1Tolerances):
        contract.validate()
        tolerances.validate()
        self.contract = contract
        self.tolerances = tolerances
        self._basis, self._pupil, self.basis_diagnostics = build_action_basis(
            S1EnvConfig(grid_size=64, pupil_radius_fraction=0.4, num_modes=21, batch_size=1),
            ActionRepresentation("O1_frozen_21_unit_fixture", "zernike", 21),
            torch.device("cpu"),
        )
        coordinates = self._pupil.nonzero().tolist()
        index = {tuple(rc): i for i, rc in enumerate(coordinates)}
        self._neighbors = [
            [index[rc] for rc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)) if rc in index]
            for r, c in coordinates
        ]
        reached = {0}
        queue = deque([0])
        while queue:
            for neighbor in self._neighbors[queue.popleft()]:
                if neighbor not in reached:
                    reached.add(neighbor)
                    queue.append(neighbor)
        if len(reached) != len(coordinates):
            raise ValueError("reference pupil must be connected")
        design = torch.cat((torch.ones(len(coordinates), 1, dtype=torch.float64),
                            self._basis[:, self._pupil].T.double()), dim=1)
        if int(torch.linalg.matrix_rank(design)) != 22:
            raise ValueError("constant phase and frozen modes are not identifiable")
        self.design_condition_number = float(torch.linalg.cond(design))
        if self.design_condition_number > 100:
            raise ValueError("ill-conditioned phase reference fit")
        self._design = design
        self._inverse = torch.linalg.pinv(design)

    @property
    def basis(self) -> torch.Tensor:
        return self._basis.clone()

    @property
    def pupil(self) -> torch.Tensor:
        return self._pupil.clone()

    def _unwrap(self, wrapped: list[float]) -> tuple[list[float], float]:
        phases: list[float | None] = [None] * len(wrapped)
        phases[0] = wrapped[0]
        queue = deque([0])
        max_jump = 0.0
        while queue:
            current = queue.popleft()
            for neighbor in self._neighbors[current]:
                delta = (wrapped[neighbor] - wrapped[current] + math.pi) % (2 * math.pi) - math.pi
                max_jump = max(max_jump, abs(delta))
                if abs(delta) > self.tolerances.max_neighbor_jump_rad:
                    raise ValueError("spatial phase jump exceeds the O1 sampling guard")
                candidate = phases[current] + delta
                if phases[neighbor] is None:
                    phases[neighbor] = candidate
                    queue.append(neighbor)
                elif abs(phases[neighbor] - candidate) > self.tolerances.loop_consistency_rad:
                    raise ValueError("phase loop inconsistency; singular or ambiguous field")
        return [float(value) for value in phases], max_jump

    @torch.no_grad()
    def measure(self, field: torch.Tensor) -> ModalObservation:
        """只读取当前复光场；不接收真值、未来帧、动作、env.info 或功率。"""
        if field.device.type != "cpu":
            raise ValueError("O1 is an explicit CPU unit reference, not a CUDA fallback")
        if (field.ndim != 3 or not 1 <= field.shape[0] <= 64
                or field.dtype not in (torch.complex64, torch.complex128)):
            raise ValueError("small [batch,64,64] complex64/complex128 unit fixture required")
        registered = orient_complex_field(field, self.contract.orientation)
        if registered.shape[1:] != self._pupil.shape:
            raise ValueError("grid mismatch; no implicit camera resizing/registration")
        if not bool(torch.isfinite(registered).all()):
            raise ValueError("nonfinite complex field")
        sampled = registered[:, self._pupil].to(torch.complex128)
        intensity = sampled.abs().square()
        peak = intensity.amax(dim=1)
        if not bool(torch.isfinite(intensity).all()) or bool((peak <= 0).any()):
            raise ValueError("invalid or zero pupil intensity")
        relative_min = (intensity / peak[:, None]).amin(dim=1)
        if bool((relative_min < self.tolerances.min_relative_intensity).any()):
            raise ValueError("low intensity or hole inside the known pupil")
        values, jumps = zip(*(self._unwrap(row) for row in sampled.angle().tolist()))
        unwrapped = torch.tensor(values, dtype=torch.float64)
        solution = unwrapped @ self._inverse.T
        error = unwrapped - solution @ self._design.T
        rmse = error.square().mean(dim=1).sqrt()
        if not bool(torch.isfinite(solution).all()) or bool((rmse > self.tolerances.max_fit_rmse_rad).any()):
            raise ValueError("field does not fit constant phase plus frozen 21 modes")
        phase_map = torch.zeros_like(registered.real, dtype=torch.float64)
        phase_map[:, self._pupil] = unwrapped
        return ModalObservation(
            residual_rad=solution[:, 1:].float(), constant_phase_rad=solution[:, 0],
            unwrapped_phase_rad=phase_map, pupil=self.pupil, fit_rmse_rad=rmse,
            max_neighbor_jump_rad=torch.tensor(jumps, dtype=torch.float64),
            minimum_relative_intensity=relative_min,
        )
