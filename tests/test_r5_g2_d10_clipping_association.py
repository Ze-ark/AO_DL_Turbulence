"""G2-D10 CPU 确定性小型单元测试；合成数据不是正式性能结果。"""
from copy import deepcopy

import pytest
import torch

from scripts import diagnose_s4_r5_g2_d10_clipping_association as d10
from src.rl.s4_training import _load_yaml, _project_path


@pytest.mark.parametrize("field,value", [
    ("runtime", {"device": "cpu", "automatic_retry": False}),
    ("boundary", {"causal_performance_claim": True}),
    ("d9_config_sha256", "0" * 64),
    ("analysis", {"bootstrap_unit": "frame"}),
    ("output_directory", "outputs/overwrite"),
])
def test_contract_preserves_evidence_and_noncausal_boundary(field, value):
    cfg = _load_yaml(_project_path(d10.CONFIG))
    d10._contract(cfg)
    cfg[field] = value
    with pytest.raises(ValueError, match="合同"):
        d10._contract(cfg)


def test_within_cell_centering_exposes_pooled_confounding():
    variation = torch.linspace(-1, 1, 32, dtype=torch.float64)[:, None]
    offsets = torch.tensor([[0., 100.]], dtype=torch.float64)
    x, y = offsets + variation, offsets - variation
    assert d10._correlation(x, y, within=False) > .99
    result = d10.association(x, y, seed=22, repeats=100)
    assert result["within_cell_correlation"] == pytest.approx(-1)
    assert result["weather_cluster_ci95"] == pytest.approx([-1, -1])
    assert result["valid_bootstrap_draws"] == 100


def test_whole_weather_bootstrap_is_deterministic():
    x = torch.arange(32 * 3, dtype=torch.float64).reshape(32, 3) / 100
    y = x * 2
    first = d10.association(x, y, seed=19, repeats=100)
    assert first == d10.association(x, y, seed=19, repeats=100)
    assert first["within_cell_correlation"] == pytest.approx(1)
    assert first["weather_cluster_ci95"] == pytest.approx([1, 1])


def test_constant_within_cell_returns_null_not_nan():
    x = torch.tensor([[1., 2.]] * 32, dtype=torch.float64)
    result = d10.association(x, x, seed=19, repeats=100)
    assert result["pooled_correlation"] == pytest.approx(1)
    assert result["within_cell_correlation"] is None
    assert result["weather_cluster_ci95"] is None
    assert result["undefined_reason"] == "no_within_cell_variance"


@pytest.mark.parametrize("bad", ["shape", "nan", "empty", "integer"])
def test_bad_association_inputs_fail_closed(bad):
    x = torch.ones(3, 2, dtype=torch.float64)
    y = x.clone()
    if bad == "shape":
        y = y.flatten()
    elif bad == "nan":
        y[0, 0] = float("nan")
    elif bad == "empty":
        x, y = x[:0], y[:0]
    else:
        x, y = x.long(), y.long()
    with pytest.raises(ValueError, match="矩阵"):
        d10.association(x, y, seed=19, repeats=100)


def test_bootstrap_rejects_zero_draws():
    with pytest.raises(ValueError, match="正整数"):
        d10.association(torch.ones(3, 2), torch.ones(3, 2), seed=19, repeats=0)


def test_controlled_direct_projection_gradient_not_physical_saturation():
    result = d10.controlled_jacobian(torch.device("cpu"))
    expected = [0., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0.]
    assert result["normalized_output_gradient"] == pytest.approx(expected)
    assert result["requested_high_order_delta_gradient"] == pytest.approx([.0125 * v for v in expected])
    assert result["outside_direct_gradient_zero"]
    assert result["not_observed_training_gradient_fraction"]
    assert "direct_current_action_path_only" in result["scope"]


def synthetic_rows():
    d9 = d10.d9
    rows = []
    for condition in d9.CONDITIONS:
        for index, controller in enumerate(d9.CONTROLLERS):
            baseline, new = index == 0, index >= 4
            for family_index, family in enumerate(d9.FAMILIES):
                for seed in d9.stream_manifest(False)["weather_bases"]:
                    for slot, profile in enumerate(d9.PROFILES):
                        row = dict.fromkeys(d9.METRICS, 0.)
                        row.update(hardware_condition=condition, controller=controller,
                                   family=family, weather_seed=seed, slot=slot,
                                   profile=f"nominal_for_{profile}" if condition == "nominal_clone" else profile,
                                   episode_length=200, scale=0. if baseline else 1.75,
                                   turbulence_stream_seed=seed + 1000 * slot + family_index,
                                   power=.7 if baseline else .7072 if new else .707,
                                   strehl=.7 if baseline else .72 if new else .71,
                                   phase_rmse=.3 if baseline else .28 if new else .29,
                                   normalized_correction_clipped_fraction=0. if baseline else .25 if new else .04)
                        rows.append(row)
    return rows


def test_full_synthetic_grid_preserves_failure_and_all_groups():
    cfg = _load_yaml(_project_path(d10.CONFIG))
    cfg["analysis"]["bootstrap_repeats"] = 10
    rows = synthetic_rows()
    original = deepcopy(rows)
    result = d10.analyze(rows, cfg, device=torch.device("cpu"))
    assert rows == original
    assert result["prior_continue_criteria"]["all"] is False
    assert len(result["group_table"]) == 36
    assert len({(r["hardware_condition"], r["family"], r["slot"]) for r in result["group_table"]}) == 36
    for cell in result["cells"].values():
        assert cell["new_clip_fraction"] == pytest.approx(.25)
        assert cell["old_clip_fraction"] == pytest.approx(.04)
        assert cell["power_shortfall_to_development_margin"] == pytest.approx(.00015)
        assert cell["clip_vs_new_minus_old_power_association"]["within_cell_correlation"] is None
    assert "not_causal" in result["status"].lower()


def test_frozen_source_change_fails_before_cuda_or_output(monkeypatch):
    monkeypatch.setattr(d10, "_file_sha256", lambda path: "0" * 64)
    with pytest.raises(RuntimeError, match="冻结入口"):
        d10.preflight()
