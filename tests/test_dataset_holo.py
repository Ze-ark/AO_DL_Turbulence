"""模拟湍流 HDF5 数据集及索引划分测试。"""

import h5py
import numpy as np
import pytest
import torch

from src.dataset_holo import HoloH5Dataset, split_indices


def _write_sample_h5(path, n=4, h=8, w=8):
    """创建供测试使用的最小首维帧布局 HDF5 文件。"""
    with h5py.File(path, "w") as f:
        f.create_dataset("input/intensity_turb", data=np.ones((n, h, w), dtype=np.float32) * 4.0)
        phase = np.linspace(-np.pi, np.pi, n * h * w, dtype=np.float32).reshape(n, h, w)
        f.create_dataset("input/phase_turb", data=phase)
        f.create_dataset("target/intensity_clean", data=np.ones((n, h, w), dtype=np.float32))
        f.create_dataset("target/phase_clean", data=np.zeros((n, h, w), dtype=np.float32))
        f.create_dataset("meta/frame_id", data=np.arange(n, dtype=np.int64))
        f.create_dataset("meta/turbulence_strength", data=np.linspace(0.2, 1.0, n, dtype=np.float32))
        f.create_dataset("meta/r0_or_equivalent", data=np.linspace(0.08, 0.02, n, dtype=np.float32))


def test_holo_h5_dataset_returns_normalized_input_and_targets(tmp_path):
    """验证数据集返回归一化双通道输入、目标和元数据。"""
    h5_path = tmp_path / "sample.h5"
    _write_sample_h5(h5_path)

    dataset = HoloH5Dataset(h5_path)
    sample = dataset[0]

    assert len(dataset) == 4
    assert sample["input"].shape == (2, 8, 8)
    assert torch.allclose(sample["input"][0], torch.ones(8, 8))
    assert sample["target_intensity"].shape == (1, 8, 8)
    assert sample["target_phase"].shape == (1, 8, 8)
    assert sample["meta"]["frame_id"] == 0
    assert sample["meta"]["turbulence_strength"] == pytest.approx(0.2)


def test_split_indices_is_deterministic_and_covers_dataset():
    """验证相同种子产生确定划分且所有样本恰好覆盖一次。"""
    split_a = split_indices(length=10, train_fraction=0.6, val_fraction=0.2, seed=7)
    split_b = split_indices(length=10, train_fraction=0.6, val_fraction=0.2, seed=7)

    assert split_a == split_b
    assert len(split_a["train"]) == 6
    assert len(split_a["val"]) == 2
    assert len(split_a["test"]) == 2
    assert sorted(split_a["train"] + split_a["val"] + split_a["test"]) == list(range(10))


def test_holo_h5_dataset_reads_matlab_frame_last_and_row_meta(tmp_path):
    """验证 MATLAB 末维帧布局及行向量元数据读取。"""
    # 构造 H×W×N 数据与 1×N 元数据，模拟 MATLAB 导出格式。
    h5_path = tmp_path / "matlab_layout.h5"
    n, h, w = 3, 6, 5
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("input/intensity_turb", data=np.ones((h, w, n), dtype=np.float32))
        f.create_dataset("input/phase_turb", data=np.zeros((h, w, n), dtype=np.float32))
        f.create_dataset("target/intensity_clean", data=np.ones((h, w, n), dtype=np.float32))
        f.create_dataset("target/phase_clean", data=np.zeros((h, w, n), dtype=np.float32))
        f.create_dataset("meta/frame_id", data=np.arange(n, dtype=np.int64).reshape(1, n))
        f.create_dataset("meta/turbulence_strength", data=np.array([[0.2, 0.6, 1.0]], dtype=np.float32))
        f.create_dataset("meta/r0_or_equivalent", data=np.array([[0.08, 0.04, 0.02]], dtype=np.float32))

    dataset = HoloH5Dataset(h5_path)
    sample = dataset[2]

    assert len(dataset) == 3
    assert sample["input"].shape == (2, h, w)
    assert sample["meta"]["frame_id"] == 2
    assert sample["meta"]["turbulence_strength"] == pytest.approx(1.0)
