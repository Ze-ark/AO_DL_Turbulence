import h5py
import numpy as np

from src.dataset_holo import RealComplexH5Dataset


def test_real_complex_h5_dataset_reads_unlabeled_complex_fields(tmp_path):
    h5_path = tmp_path / "real_validation.h5"
    n, h, w = 3, 8, 6
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("input/intensity_turb", data=np.ones((h, w, n), dtype=np.float32))
        f.create_dataset("input/phase_turb", data=np.zeros((h, w, n), dtype=np.float32))
        f.create_dataset("meta/frame_id", data=np.arange(n, dtype=np.int64).reshape(1, n))
        f.create_dataset("meta/temperature", data=np.array([[40, 80, 100]], dtype=np.float32))

    dataset = RealComplexH5Dataset(h5_path)
    sample = dataset[1]

    assert len(dataset) == 3
    assert sample["input"].shape == (2, h, w)
    assert sample["input_intensity"].shape == (1, h, w)
    assert sample["input_phase"].shape == (1, h, w)
    assert sample["meta"]["frame_id"] == 1
    assert sample["meta"]["temperature"] == 80
