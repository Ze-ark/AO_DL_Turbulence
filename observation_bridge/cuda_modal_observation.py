"""O2 固定光瞳图上的批量 CUDA 解包裹与冻结模态拟合。

静态几何可在 CPU 预计算；所有当前复场、相位和拟合张量留在 GPU。
21 模态拟合残差是表示容量误差，不作为坏观测而被静默排除。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import torch

from observation_bridge.adapter import ModalObservation, SyntheticFieldContract, orient_complex_field
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.runtime import resolve_device
from src.simulation.config import S1EnvConfig


@dataclass(frozen=True)
class O2ModalTolerances:
    min_relative_intensity: float
    max_neighbor_jump_rad: float
    loop_consistency_rad: float

    def validate(self) -> None:
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (
                self.min_relative_intensity, self.max_neighbor_jump_rad, self.loop_consistency_rad)):
            raise ValueError("finite numeric O2 tolerances required")
        if not 0 < self.min_relative_intensity <= 1:
            raise ValueError("invalid relative intensity floor")
        if not 0 < self.max_neighbor_jump_rad < math.pi:
            raise ValueError("neighbor jump guard must be below pi")
        if not 0 < self.loop_consistency_rad < 0.01:
            raise ValueError("invalid phase loop tolerance")


class CudaModalObservationBridge:
    def __init__(self, contract: SyntheticFieldContract, tolerances: O2ModalTolerances,
                 device: torch.device):
        contract.validate()
        tolerances.validate()
        if device.type != "cuda":
            raise ValueError("O2 modal measurement requires CUDA; no CPU fallback")
        self.device = resolve_device(str(device))
        self.contract, self.tolerances = contract, tolerances
        basis, pupil, self.basis_diagnostics = build_action_basis(
            S1EnvConfig(grid_size=64, pupil_radius_fraction=0.4, num_modes=21, batch_size=1),
            ActionRepresentation("O2_frozen_21_geometry", "zernike", 21), torch.device("cpu"))
        coordinates = pupil.nonzero().tolist()  # 静态几何，不是观测张量回传。
        index = {tuple(rc): i for i, rc in enumerate(coordinates)}
        root = min(range(len(coordinates)), key=lambda i: sum((v - 31.5) ** 2 for v in coordinates[i]))
        depth, parents = {root: 0}, {root: root}
        queue = deque([root])
        edges: list[tuple[int, int]] = []
        while queue:
            current = queue.popleft()
            r, c = coordinates[current]
            for rc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if rc not in index:
                    continue
                child = index[rc]
                if child not in depth:
                    depth[child], parents[child] = depth[current] + 1, current
                    queue.append(child)
                if current < child:
                    edges.append((current, child))
        if len(depth) != len(coordinates):
            raise ValueError("frozen pupil graph is disconnected")
        design = torch.cat((torch.ones(len(coordinates), 1, dtype=torch.float64),
                            basis[:, pupil].T.double()), dim=1)
        if int(torch.linalg.matrix_rank(design)) != 22:
            raise ValueError("phase reference is singular")
        self.design_condition_number = float(torch.linalg.cond(design))
        if self.design_condition_number > 100:
            raise ValueError("ill-conditioned phase reference fit")
        self._basis, self._pupil = basis.to(self.device), pupil.to(self.device)
        self._design, self._inverse = design.to(self.device), torch.linalg.pinv(design).to(self.device)
        self._root = root
        self._layers = []
        for level in range(1, max(depth.values()) + 1):
            children = [i for i in range(len(coordinates)) if depth[i] == level]
            self._layers.append((torch.tensor(children, device=self.device),
                                 torch.tensor([parents[i] for i in children], device=self.device)))
        pairs = torch.tensor(edges, dtype=torch.long, device=self.device)
        self._edge_from, self._edge_to = pairs[:, 0], pairs[:, 1]

    @property
    def basis(self) -> torch.Tensor:
        return self._basis.clone()

    @property
    def pupil(self) -> torch.Tensor:
        return self._pupil.clone()

    @torch.no_grad()
    def measure(self, current_field: torch.Tensor) -> ModalObservation:
        """白名单仅有当前重建复场；不接收真值、动作、info 或功率。"""
        if current_field.device.type != "cuda" or current_field.device != self._pupil.device:
            raise ValueError("current measurement must stay on the configured CUDA device")
        if (current_field.ndim != 3 or not 1 <= current_field.shape[0] <= 64
                or current_field.dtype not in (torch.complex64, torch.complex128)):
            raise ValueError("current complex [batch,64,64] field required")
        field = orient_complex_field(current_field, self.contract.orientation)
        if field.shape[-2:] != (64, 64):
            raise ValueError("unknown grid; no implicit resizing")
        if not bool(torch.isfinite(field).all()):
            raise ValueError("nonfinite reconstructed field")
        values = field[:, self._pupil].to(torch.complex128)
        intensity = values.abs().square()
        peak = intensity.amax(dim=1)
        if not bool(torch.isfinite(intensity).all()) or bool((peak <= 0).any()):
            raise ValueError("invalid pupil intensity")
        minimum = (intensity / peak[:, None]).amin(dim=1)
        if bool((minimum < self.tolerances.min_relative_intensity).any()):
            raise ValueError("low intensity or hole inside frozen pupil")
        wrapped = values.angle()
        edge_delta = wrapped[:, self._edge_to] - wrapped[:, self._edge_from]
        principal = torch.atan2(edge_delta.sin(), edge_delta.cos())
        max_jump = principal.abs().amax(dim=1)
        if bool((max_jump > self.tolerances.max_neighbor_jump_rad).any()):
            raise ValueError("spatial phase jump exceeds sampling guard")
        unwrapped = torch.zeros_like(wrapped)
        unwrapped[:, self._root] = wrapped[:, self._root]
        for children, parents in self._layers:
            delta = wrapped[:, children] - wrapped[:, parents]
            unwrapped[:, children] = unwrapped[:, parents] + torch.atan2(delta.sin(), delta.cos())
        inconsistency = (unwrapped[:, self._edge_to] - unwrapped[:, self._edge_from] - principal).abs()
        if bool((inconsistency.amax(dim=1) > self.tolerances.loop_consistency_rad).any()):
            raise ValueError("phase loop inconsistency; singular or ambiguous field")
        solution = unwrapped @ self._inverse.T
        rmse = (unwrapped - solution @ self._design.T).square().mean(dim=1).sqrt()
        if not bool(torch.isfinite(solution).all()) or not bool(torch.isfinite(rmse).all()):
            raise ValueError("nonfinite modal fit")
        phase_map = torch.zeros_like(field.real, dtype=torch.float64)
        phase_map[:, self._pupil] = unwrapped
        return ModalObservation(
            residual_rad=solution[:, 1:].float(), constant_phase_rad=solution[:, 0],
            unwrapped_phase_rad=phase_map, pupil=self.pupil, fit_rmse_rad=rmse,
            max_neighbor_jump_rad=max_jump, minimum_relative_intensity=minimum)
