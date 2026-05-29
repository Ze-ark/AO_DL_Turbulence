import matplotlib

import src.evaluate_compensation  # noqa: F401


def test_evaluate_uses_noninteractive_matplotlib_backend():
    assert matplotlib.get_backend().lower() == "agg"
