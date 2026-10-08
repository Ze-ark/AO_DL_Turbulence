"""O2-D1：小型元数据/几何单元和明确的 CUDA 技术读取，零动态转移。"""
from copy import deepcopy
from dataclasses import replace
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from observation_bridge import read_noise_diagnostic as entry


@pytest.fixture(scope="module")
def cfg():
    return entry.read_config()


@pytest.mark.parametrize("key,value", [
    ("device", "cpu"), ("fixture_seed", 0), ("fixture_count", 54),
    ("measurement_batch_size", True), ("shared_draw_across_levels", False),
    ("noise_units", "electrons"), ("noise_distribution", "poisson"), ("unexpected", True),
])
def test_fixed_single_factor_contract(cfg, key, value):
    with pytest.raises(ValueError, match="single-factor"):
        entry.validate_config({**cfg, key: value})


@pytest.mark.parametrize("key,sub,value", [
    ("data", "repetitions", 17), ("data", "noise_stds", [0.0, .002]),
    ("quick", "extra_fixture_seed", 8500000), ("thresholds", "zero_noise_modal_error_max_rad", .01),
    ("boundary", "truth_fallback", True), ("boundary", "dynamic_environment_steps", 1),
    ("boundary", "real_slm_actions", True), ("boundary", "model_forward_calls", 1),
])
def test_nested_budget_guards_cannot_drift(cfg, key, sub, value):
    broken = deepcopy(cfg)
    broken[key][sub] = value
    with pytest.raises(ValueError):
        entry.validate_config(broken)


def test_budget_and_pure_seed_manifest(cfg):
    entry.validate_config(cfg)
    assert entry.budget(cfg["data"])["measurement_attempts"] == 7040
    assert entry.budget(cfg["data"])["standard_normal_draws"] == 880
    assert entry.budget(cfg["quick"])["measurement_attempts"] == 9
    formal = entry.stream_manifest(cfg, quick=False)
    quick = entry.stream_manifest(cfg, quick=True)
    assert formal["historical_manifests_checked"] == 15
    assert len(formal["noise_seeds"]) == 880
    assert not set(formal["noise_seeds"]) & set(quick["noise_seeds"])
    assert entry.noise_seed(cfg["data"], 54, 15) == 8501879
    broken = deepcopy(cfg)
    broken["data"]["noise_seed_base"] = 8400000
    with pytest.raises(RuntimeError, match="collision"):
        entry.stream_manifest(broken, quick=False)
    with pytest.raises(ValueError):
        entry.noise_seed(cfg["quick"], 2, 0)


def test_frozen_sources_read_only_no_old_entries(cfg, monkeypatch):
    from observation_bridge import development
    monkeypatch.setattr(development, "run", lambda *a, **k: pytest.fail("old C entry called"))
    monkeypatch.setattr(development.short, "run", lambda *a, **k: pytest.fail("old B entry called"))
    result = entry.verify_prerequisites(cfg)
    assert result["C_artifacts_checked"] == 426
    assert result["short_loop_artifacts_checked"] == 34
    assert result["frozen_C1_files_checked"] == 55
    assert result["real_calibration_unknown_fields"] == 18


def test_pure_cpu_geometry_outside_span_and_fixed_8_phases(cfg):
    # 仅静态几何预计算检查，不生成图像、测量误差或性能结果。
    from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
    from src.simulation.config import S1EnvConfig
    basis, pupil, _ = build_action_basis(
        S1EnvConfig(grid_size=64, pupil_radius_fraction=.4, num_modes=21, batch_size=1),
        ActionRepresentation("unit_geometry", "zernike", 21), torch.device("cpu"))
    first = entry.extra_phase_templates(basis, pupil, 8500000, 1e-8)
    assert torch.equal(first, entry.extra_phase_templates(basis, pupil, 8500000, 1e-8))
    design = torch.cat((torch.ones(int(pupil.sum()), 1).double(), basis[:, pupil].T.double()), 1)
    coefficients = torch.linalg.lstsq(design, first[:, pupil].T, driver="gelsd").solution
    errors = (first[:, pupil].T - design @ coefficients).square().mean(0).sqrt()
    assert errors.tolist() == pytest.approx([.1, .1, .2, .2, .4, .4, .8, .8])
    fake = SimpleNamespace(device=torch.device("cpu"), basis=basis, pupil=pupil,
                           tolerances=SimpleNamespace(max_neighbor_jump_rad=1.5))
    phases, ids = entry.make_fixtures(cfg, cfg["data"], fake)
    assert phases.shape == (55, 64, 64) and len(ids) == 55
    assert sum(i["fixture_group"] == "outside_21_modes" for i in ids) == 8
    with pytest.raises(RuntimeError, match="sampling"):
        entry.make_fixtures(cfg, cfg["data"], SimpleNamespace(**{**fake.__dict__, "tolerances":SimpleNamespace(max_neighbor_jump_rad=.01)}))


def test_cpu_only_tiny_handwritten_statistics():
    result = entry.valid_statistics(torch.tensor([1., 2., 3.], dtype=torch.float64))
    assert result == pytest.approx(dict(mean=2, p95=2.9, maximum=3))
    with pytest.raises(ValueError):
        entry.valid_statistics(torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="require CUDA"):
        entry.summarize([], {}, torch.device("cpu"))
    with pytest.raises(ValueError, match="CUDA"):
        entry.paired_intensity(torch.ones(1), torch.zeros(1), .001)


@pytest.mark.parametrize("reason,category", list(entry.MEASUREMENT_REJECTIONS.items()))
def test_only_predeclared_measurement_rejections(reason, category):
    assert entry.rejection_type(ValueError(reason)) == category


@pytest.mark.parametrize("reason", ["nonfinite reconstructed field", "invalid pupil intensity",
                                     "nonfinite modal fit", "current measurement must stay on the configured CUDA device"])
def test_unexpected_error_not_swallowed(reason):
    assert entry.rejection_type(ValueError(reason)) is None


def handmade_rows(spec):
    rows = []
    for f in spec["fixture_indices"]:
        for rep in range(spec["repetitions"]):
            for level, std in enumerate(spec["noise_stds"]):
                row = dict(fixture_index=f, fixture_id=f"unit_{f}", fixture_group="controlled" if f<47 else "outside_21_modes",
                           repetition=rep, noise_level_index=level, read_noise_std=std,
                           noise_seed=entry.noise_seed(spec, f, rep), noise_rng_after_draw_sha256="a"*64,
                           status="valid", rejection_type=None, rejection_reason=None, modal_bias_rad=[0.]*21,
                           **{c:0. for c in entry.NUMERIC_COLUMNS}, clean_intensity_mean=1.,
                           noise_std_over_clean_mean=std, negative_clip_fraction=0., observation_latency_ms=0.)
                if std == 10:
                    row.update(status="rejected", rejection_type="sampling_jump",
                               rejection_reason="spatial phase jump exceeds sampling guard", modal_bias_rad=None,
                               **{c:None for c in entry.NUMERIC_COLUMNS})
                rows.append(row)
    return rows


def test_all_attempts_grid_includes_rejected_not_imputed(cfg):
    rows = handmade_rows(cfg["quick"])
    entry.validate_rows(rows, cfg["quick"])
    assert len(rows) == 9 and sum(r["status"] == "rejected" for r in rows) == 3
    for broken in (rows[:-1], rows+[rows[0]]):
        with pytest.raises(ValueError):
            entry.validate_rows(broken, cfg["quick"])
    bad = deepcopy(rows); bad[2]["modal_rmse_rad"] = 0.0
    with pytest.raises(ValueError, match="invented"):
        entry.validate_rows(bad, cfg["quick"])
    bad = deepcopy(rows); bad[1]["noise_rng_after_draw_sha256"] = "b"*64
    with pytest.raises(ValueError, match="differs"):
        entry.validate_rows(bad, cfg["quick"])
    bad = deepcopy(rows); bad[0]["modal_rmse_rad"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        entry.validate_rows(bad, cfg["quick"])
    bad = deepcopy(rows); bad[0].update(status="rejected", rejection_type="sampling_jump",
        rejection_reason="spatial phase jump exceeds sampling guard",modal_bias_rad=None,
        **{c:None for c in entry.NUMERIC_COLUMNS})
    with pytest.raises(ValueError, match="zero-noise"):
        entry.validate_rows(bad, cfg["quick"])


def test_output_path_preserved_and_preflight_no_writes(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    kept = tmp_path/"outputs/kept"
    kept.mkdir(parents=True)
    for directory, exception in [("outputs/kept", FileExistsError), ("outputs", ValueError), ("../outside", ValueError)]:
        path = tmp_path/"unit_config.yaml"
        path.write_text(yaml.safe_dump({**cfg,"output_directory":directory}), encoding="utf-8")
        with pytest.raises(exception):
            entry.preflight(path)
    assert kept.exists()
    monkeypatch.setattr(entry, "preflight", lambda *a, **k: (cfg, cfg["quick"], tmp_path/"nothing", torch.device("cpu"), {"unit":True}))
    assert entry.run(preflight_only=True) == {"unit":True}
    assert not (tmp_path/"nothing").exists()


def test_failure_file_lifecycle_no_retry_or_overwrite(cfg, tmp_path, monkeypatch):
    output = tmp_path/"outputs/unit_failure"
    monkeypatch.setattr(entry, "preflight", lambda *a, **k: (cfg,cfg["quick"],output,torch.device("cpu"),{"quick":True,"frozen_sources":{},"stream_manifest":{}}))
    monkeypatch.setattr(entry, "source_manifest", lambda *a: {})
    def crash(*a):
        raise RuntimeError("intentional unit error")
    monkeypatch.setattr(entry, "execute", crash)
    with pytest.raises(RuntimeError, match="intentional"):
        entry.run(quick=True)
    failure = json.loads((output/"failure.json").read_text(encoding="utf-8"))
    assert failure["dynamic_environment_steps"] == failure["model_forward_calls"] == 0
    assert failure["last_context"]["measurement_attempts_completed"] == 0
    assert not (output/"SUCCESS.json").exists()
    before = (output/"failure.json").read_bytes()
    with pytest.raises(FileExistsError):
        entry.run(quick=True)
    assert (output/"failure.json").read_bytes() == before


def test_complete_metadata_lifecycle_no_duplicate_status(cfg, tmp_path, monkeypatch):
    output = tmp_path/"outputs/unit_complete"
    report = dict(quick=True,status="PREFLIGHT_UNIT",frozen_sources={},stream_manifest={})
    monkeypatch.setattr(entry, "preflight", lambda *a, **k:(cfg,cfg["quick"],output,torch.device("cpu"),report))
    monkeypatch.setattr(entry, "source_manifest", lambda *a: {})
    monkeypatch.setattr(entry, "verify_prerequisites", lambda *a: {})
    monkeypatch.setattr(entry, "execute", lambda *a: dict(report,status="O2_D1_TECHNICAL_SMOKE_ONLY"))
    result = entry.run(quick=True)
    assert result["status"] != report["status"]
    success = json.loads((output/"SUCCESS.json").read_text(encoding="utf-8"))
    assert success["summary_sha256"] == entry.optics.file_sha256(output/"summary.json")
    assert not (output/"failure.json").exists()


@pytest.fixture(scope="module")
def gpu(cfg):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; no CPU optical fallback")
    entry.configure_runtime()
    device = entry.resolve_device("cuda")
    sensor, bridge = entry.optics.make_components(entry.read_config(cfg["optics_config"]), device)
    return device, sensor, bridge


def test_cuda_exact_noise_pair_and_zero_level_still_uses_same_draw(gpu):
    device = gpu[0]
    generator = torch.Generator(device=device).manual_seed(8520001)
    samples = torch.randn(1,512,512,device=device,generator=generator)
    repeated = torch.randn(1,512,512,device=device,generator=torch.Generator(device=device).manual_seed(8520001))
    assert torch.equal(samples, repeated)
    clean = torch.zeros_like(samples)
    zero, clip_zero = entry.paired_intensity(clean,samples,0.)
    small, clip_small = entry.paired_intensity(clean,samples,.001)
    large, clip_large = entry.paired_intensity(clean,samples,1.)
    assert torch.equal(zero,clean) and clip_zero == 0
    assert torch.equal(small,(samples*.001).clamp_min(0))
    assert clip_small == clip_large and .48 < clip_small < .52
    assert torch.equal(large,samples.clamp_min(0))


def test_cuda_complete_phase_retained_zero_measurement_and_no_truth_argument(cfg,gpu):
    _,sensor,bridge=gpu
    spec=deepcopy(cfg["quick"]); spec["extra_fixture_seed"]=8520002
    phases,ids=entry.make_fixtures(cfg,spec,bridge)
    assert [i["fixture_index"] for i in ids] == [0,1,47]
    field=(torch.polar(torch.ones_like(phases),phases)*bridge.pupil).to(torch.complex64)
    measured=bridge.measure(sensor.reconstruct(sensor.render(field)))
    # 测量完成后独立拟合真值，读数入口无 phase/target/info 参数。
    target,_=entry.optics.audit_known_phase(phases,bridge)
    assert float((measured.residual_rad.double()-target).abs().max())<.001
    assert float(measured.fit_rmse_rad[-1])>.09
    assert list(inspect.signature(bridge.measure).parameters)==["current_field"]


def test_cuda_summary_denominators_and_empty_valid_cell(cfg,gpu):
    rows=handmade_rows(cfg["quick"])
    cells=entry.summarize(rows,cfg["quick"],gpu[0])
    high=next(c for c in cells if c["fixture_group"]=="all" and c["read_noise_std"]==10)
    assert (high["planned_attempts"],high["valid_readings"],high["rejected_readings"])==(3,0,3)
    assert high["valid_only_modal_rmse_rad"] is None and high["rejection_fraction"]==1
    assert high["whole_level_reliability_claim"] is False


def test_import_inert_help_and_source_bundle():
    code="from pathlib import Path\nfrom unittest.mock import patch\nimport torch\nwith patch.object(Path,'mkdir',side_effect=AssertionError('write')),patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('GPU')):\n import observation_bridge.read_noise_diagnostic\n import scripts.diagnose_observation_bridge_o2_read_noise\n"
    subprocess.run([sys.executable,"-B","-c",code],cwd=entry.ROOT,capture_output=True,check=True)
    result=subprocess.run([sys.executable,"-B","-X","utf8","scripts/diagnose_observation_bridge_o2_read_noise.py","--help"],
                          cwd=entry.ROOT,capture_output=True,text=True,encoding="utf-8",check=True)
    assert "--quick" in result.stdout and "--preflight-only" in result.stdout
    assert entry.optics.sealed_bundle_sha256()==entry.optics.BUNDLE_SHA256
