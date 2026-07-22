"""评估脚本绘图后端配置测试。"""

import matplotlib

import src.evaluate_compensation  # noqa: F401


def test_evaluate_uses_noninteractive_matplotlib_backend():
    """验证评估模块使用适合无界面环境的 Agg 后端。"""
    assert matplotlib.get_backend().lower() == "agg"
