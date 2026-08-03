"""训练进度工具的小型确定性测试。"""

import torch

from src.training_progress import gpu_memory_status, progress_bar, update_progress


def test_cpu_progress_has_no_cuda_memory_text():
    assert gpu_memory_status(torch.device("cpu")) == ""


def test_progress_bar_tracks_iterable_and_accepts_metrics():
    bar = progress_bar(range(2), description="测试进度", unit="批", leave=False)
    observed = []
    for value in bar:
        observed.append(value)
        update_progress(
            bar,
            device=torch.device("cpu"),
            metrics={"损失": float(2 - value)},
        )

    assert observed == [0, 1]
    assert bar.n == 2
