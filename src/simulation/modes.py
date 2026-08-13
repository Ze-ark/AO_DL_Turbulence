"""实 Zernike 与空间动作表示的离散正交基。"""

from __future__ import annotations

import math

import torch


MODE_NAMES = (
    "水平倾斜",
    "垂直倾斜",
    "离焦",
    "零度像散",
    "四十五度像散",
    "水平彗差",
    "垂直彗差",
    "水平三叶",
    "垂直三叶",
    "球差",
    "四阶二次余弦",
    "四阶二次正弦",
    "四阶四次余弦",
    "四阶四次正弦",
    "五阶一次余弦",
    "五阶一次正弦",
    "五阶三次余弦",
    "五阶三次正弦",
    "五阶五次余弦",
    "五阶五次正弦",
    "六阶零次",
    "六阶二次余弦",
    "六阶二次正弦",
    "六阶四次余弦",
    "六阶四次正弦",
    "六阶六次余弦",
    "六阶六次正弦",
    "七阶一次余弦",
    "七阶一次正弦",
    "七阶三次余弦",
    "七阶三次正弦",
    "七阶五次余弦",
    "七阶五次正弦",
    "七阶七次余弦",
    "七阶七次正弦",
    "八阶零次",
)


def make_pupil_mask(
    grid_size: int,
    pupil_radius_fraction: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """构造与环境一致的圆形光瞳掩膜。"""
    if grid_size <= 1:
        raise ValueError("grid_size must exceed one")
    if not 0 < pupil_radius_fraction <= 0.5:
        raise ValueError("pupil_radius_fraction must be in (0, 0.5]")
    coordinate = torch.linspace(-1, 1, grid_size, device=device, dtype=dtype)
    y, x = torch.meshgrid(coordinate, coordinate, indexing="ij")
    normalized_radius = torch.sqrt(x.square() + y.square()) / (
        2 * pupil_radius_fraction
    )
    return normalized_radius <= 1


def make_low_order_zernike_basis(
    grid_size: int,
    pupil_radius_fraction: float,
    num_modes: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """构造去掉 piston 的前36个实模态，并在离散瞳面内正交化。

    前十个候选保留项目原有多项式及顺序，保证旧实验的10维动作语义不变。
    """
    if not 1 <= num_modes <= len(MODE_NAMES):
        raise ValueError(f"num_modes must be between 1 and {len(MODE_NAMES)}")
    coordinate = torch.linspace(-1, 1, grid_size, device=device, dtype=dtype)
    Y, X = torch.meshgrid(coordinate, coordinate, indexing="ij")
    normalized_radius = torch.sqrt(X.square() + Y.square()) / (2 * pupil_radius_fraction)
    x = X / (2 * pupil_radius_fraction)
    y = Y / (2 * pupil_radius_fraction)
    r2 = x.square() + y.square()
    pupil = normalized_radius <= 1
    candidates: list[torch.Tensor] = [
        x,
        y,
        2 * r2 - 1,
        x.square() - y.square(),
        2 * x * y,
        (3 * r2 - 2) * x,
        (3 * r2 - 2) * y,
        x * (x.square() - 3 * y.square()),
        y * (3 * x.square() - y.square()),
        6 * r2.square() - 6 * r2 + 1,
    ]
    if num_modes > 10:
        radius = torch.sqrt(r2)
        angle = torch.atan2(y, x)
        generic: list[torch.Tensor] = []
        radial_order = 1
        while len(generic) < num_modes:
            for azimuthal_order in range(radial_order % 2, radial_order + 1, 2):
                radial = _zernike_radial(
                    radial_order,
                    azimuthal_order,
                    radius,
                )
                if azimuthal_order == 0:
                    generic.append(radial)
                else:
                    generic.append(radial * torch.cos(azimuthal_order * angle))
                    generic.append(radial * torch.sin(azimuthal_order * angle))
            radial_order += 1
        candidates.extend(generic[10:num_modes])
    orthonormal: list[torch.Tensor] = []
    mask = pupil.to(dtype)
    for candidate in candidates[:num_modes]:
        mode = candidate * mask
        for previous in orthonormal:
            mode = mode - (mode * previous).sum() / previous.square().sum() * previous
        mode = mode / torch.sqrt(mode[pupil].square().mean())
        orthonormal.append(mode)
    return torch.stack(orthonormal), pupil


def make_hybrid_spatial_basis(
    grid_size: int,
    pupil_radius_fraction: float,
    num_modes: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    *,
    anchor_modes: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """构造“前十个Zernike＋低频空间余弦”的正交动作基。

    该基专用于非学习能力上限。前 ``anchor_modes`` 个向量与传统控制器完全
    一致，其余向量从低频二维余弦候选中剔除锚点分量后正交化。计算固定在
    CPU float64 上，再转到目标设备，避免CPU/GPU分别分解造成基向量漂移。
    """
    if not 1 <= anchor_modes <= 10:
        raise ValueError("anchor_modes must be between 1 and 10")
    if not anchor_modes < num_modes <= 256:
        raise ValueError("hybrid spatial basis requires anchor_modes < num_modes <= 256")

    cpu = torch.device("cpu")
    anchor, pupil = make_low_order_zernike_basis(
        grid_size,
        pupil_radius_fraction,
        anchor_modes,
        cpu,
        torch.float64,
    )
    pupil_points = int(pupil.sum())
    if pupil_points < num_modes:
        raise ValueError("pupil has fewer samples than requested spatial modes")
    scale = math.sqrt(pupil_points)
    anchor_columns = anchor[:, pupil].transpose(0, 1) / scale

    sample = torch.arange(grid_size, dtype=torch.float64) + 0.5
    one_dimensional = [
        torch.cos(math.pi * frequency * sample / grid_size)
        for frequency in range(grid_size)
    ]
    frequency_pairs = sorted(
        (
            (fx, fy)
            for fy in range(grid_size)
            for fx in range(grid_size)
            if fx != 0 or fy != 0
        ),
        key=lambda pair: (pair[0] + pair[1], max(pair), pair[1], pair[0]),
    )
    required = num_modes - anchor_modes
    candidate_count = min(len(frequency_pairs), required + 64)
    candidates = []
    for fx, fy in frequency_pairs[:candidate_count]:
        pattern = torch.outer(one_dimensional[fy], one_dimensional[fx])
        candidates.append(pattern[pupil])
    candidate_matrix = torch.stack(candidates, dim=1)
    candidate_matrix = candidate_matrix - anchor_columns @ (
        anchor_columns.transpose(0, 1) @ candidate_matrix
    )
    spatial_columns, singular_values, _ = torch.linalg.svd(
        candidate_matrix,
        full_matrices=False,
    )
    rank = int((singular_values > singular_values.max() * 1e-10).sum())
    if rank < required:
        raise RuntimeError("spatial candidate basis is rank deficient inside the pupil")
    spatial_columns = spatial_columns[:, :required]

    combined = torch.cat((anchor_columns, spatial_columns), dim=1) * scale
    basis = torch.zeros(
        num_modes,
        grid_size,
        grid_size,
        dtype=torch.float64,
    )
    basis[:, pupil] = combined.transpose(0, 1)
    return basis.to(device=device, dtype=dtype), pupil.to(device=device)


def _zernike_radial(
    radial_order: int,
    azimuthal_order: int,
    radius: torch.Tensor,
) -> torch.Tensor:
    """计算未归一化的Zernike径向多项式。"""
    if radial_order < 0 or azimuthal_order < 0:
        raise ValueError("Zernike orders must be non-negative")
    if azimuthal_order > radial_order or (radial_order - azimuthal_order) % 2:
        raise ValueError("invalid Zernike radial/azimuthal order pair")
    result = torch.zeros_like(radius)
    for index in range((radial_order - azimuthal_order) // 2 + 1):
        coefficient = (
            (-1) ** index
            * math.factorial(radial_order - index)
            / (
                math.factorial(index)
                * math.factorial((radial_order + azimuthal_order) // 2 - index)
                * math.factorial((radial_order - azimuthal_order) // 2 - index)
            )
        )
        result = result + coefficient * radius.pow(radial_order - 2 * index)
    return result


def project_phase_to_modes(
    phase: torch.Tensor,
    basis: torch.Tensor,
    pupil: torch.Tensor,
) -> torch.Tensor:
    """把相位投影为离散正交模态系数。"""
    masked_phase = phase * pupil.to(phase.dtype)
    numerator = torch.einsum("...hw,mhw->...m", masked_phase, basis)
    denominator = basis.square().sum(dim=(-2, -1))
    return numerator / denominator


def synthesize_phase(coefficients: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """由模态系数合成二维相位。"""
    if coefficients.shape[-1] != basis.shape[0]:
        raise ValueError("coefficient count does not match basis")
    return torch.einsum("...m,mhw->...hw", coefficients, basis)
