"""R5 开发诊断的随机流合同，不生成正式性能结果。"""
import pytest

from src.rl.r5_margin_development import effective_stream_seeds


def test_weather_stride_prevents_cross_family_stream_reuse() -> None:
    seeds = effective_stream_seeds(5240000, 32, 10, 6, 3)
    assert len(seeds) == len(set(seeds)) == 576
    assert min(seeds) == 5240000
    assert max(seeds) == 5245312


def test_consecutive_weather_base_reuses_family_stream() -> None:
    with pytest.raises(ValueError, match="reused turbulence stream"):
        effective_stream_seeds(5240000, 32, 1, 6, 3)
