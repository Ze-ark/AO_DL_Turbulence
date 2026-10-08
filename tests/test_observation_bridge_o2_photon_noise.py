"""O2-D4 小型元数据单元与 CUDA 合成读取；不运行正式扫描或控制器。"""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from observation_bridge import photon_noise_diagnostic as entry


@pytest.fixture(scope="module")
def cfg():
    return entry.read_config()


@pytest.mark.parametrize("key,value", [
    ("device", "cpu"), ("fixture_seed", 0), ("measurement_batch_size", True),
    ("shared_draw_across_levels", True), ("image_dependent_normalization", True),
    ("read_noise_std", .001), ("count_dtype", "float32"), ("unexpected", 1),
    ("noise_units", "real_electrons"), ("noise_distribution", "gaussian"),
])
def test_single_factor_contract(cfg, key, value):
    with pytest.raises(ValueError, match="single-factor"):
        entry.validate_config({**cfg, key:value})


@pytest.mark.parametrize("key,sub,value", [
    ("data", "repetitions", 17), ("data", "count_scales", [None,100.]),
    ("quick", "extra_fixture_seed", 8800000), ("thresholds", "zero_noise_modal_error_max_rad", .01),
    ("boundary", "truth_fallback", True), ("boundary", "real_slm_actions", True),
    ("boundary", "dynamic_environment_steps", 1), ("boundary", "model_loads", 1),
])
def test_nested_contract_cannot_drift(cfg, key, sub, value):
    bad = deepcopy(cfg)
    bad[key][sub] = value
    with pytest.raises(ValueError):
        entry.validate_config(bad)


def test_budgets_and_unique_streams(cfg):
    entry.validate_config(cfg)
    assert entry.budget(cfg["data"])["measurement_attempts"] == 6160
    assert entry.budget(cfg["data"])["poisson_draws"] == 5280
    assert entry.budget(cfg["quick"])["measurement_attempts"] == 9
    assert entry.budget(cfg["quick"])["poisson_draws"] == 6
    full = entry.stream_manifest(cfg, quick=False)
    quick = entry.stream_manifest(cfg, quick=True)
    assert len(full["noise_seeds"]) == 5280 and full["historical_manifests_checked"] == 21
    assert not set(full["noise_seeds"]) & set(quick["noise_seeds"])
    assert entry.noise_seed(cfg["data"], 54, 15, 6) == 8806279
    assert entry.noise_seed(cfg["data"], 0, 0, 0) is None
    for args in [(True,0,0), (55,0,0), (0,16,0), (0,0,7)]:
        with pytest.raises(ValueError):
            entry.noise_seed(cfg["data"], *args)


@pytest.mark.parametrize("collision", [8501000, 8700000, 8790000, 8870000])
def test_historical_and_unit_seed_collisions(cfg, collision):
    broken = deepcopy(cfg)
    broken["data"]["noise_seed_base"] = collision
    with pytest.raises(RuntimeError, match="collision"):
        entry.stream_manifest(broken, quick=False)


def test_sealed_prerequisites_without_old_entry_or_models(monkeypatch):
    monkeypatch.setattr(entry.completed, "run", lambda *a, **k:pytest.fail("old D3 run"))
    monkeypatch.setattr(entry.static, "run", lambda *a, **k:pytest.fail("old D1 run"))
    monkeypatch.setattr(entry.short, "load_assets", lambda *a, **k:pytest.fail("model load"))
    result = entry.verify_prerequisites()
    assert result["D3_artifacts_checked"] == 904
    assert result["D2_artifacts_checked"] == 486
    assert result["D1_artifacts_checked"] == 16
    assert result["real_calibration_unknown_fields"] == 18


def test_cpu_only_source_sampling_geometry(cfg):
    # 明确的几何预计算单元，不在 CPU 生成相机图像或误差曲线。
    from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
    from src.simulation.config import S1EnvConfig
    basis, pupil, _ = build_action_basis(
        S1EnvConfig(grid_size=64, pupil_radius_fraction=.4, num_modes=21, batch_size=1),
        ActionRepresentation("unit_geometry", "zernike", 21), torch.device("cpu"))
    fake = SimpleNamespace(device=torch.device("cpu"), basis=basis, pupil=pupil,
                           tolerances=SimpleNamespace(max_neighbor_jump_rad=1.5))
    phases, ids = entry.static.make_fixtures(cfg, cfg["data"], fake)
    assert phases.shape == (55,64,64) and len(ids) == 55
    assert sum(i["fixture_group"] == "outside_21_modes" for i in ids) == 8
    assert torch.equal(phases, entry.static.make_fixtures(cfg, cfg["data"], fake)[0])


def handmade_rows(spec):
    rows = []
    for f in spec["fixture_indices"]:
        for r in range(spec["repetitions"]):
            for l, scale in enumerate(spec["count_scales"]):
                row = dict(fixture_index=f, fixture_id=f"unit_{f}", fixture_group="controlled" if f < 47 else "outside_21_modes",
                           repetition=r, light_level_index=l, counts_per_intensity_unit=scale,
                           noise_seed=entry.noise_seed(spec,f,r,l), noise_rng_after_draw_sha256="a"*64 if scale else None,
                           counts_sha256="b"*64 if scale else None, clean_intensity_sha256="c"*64,
                           status="valid", rejection_type=None, rejection_reason=None,
                           modal_bias_rad=[0.]*21, **{c:0. for c in entry.NUMERIC_COLUMNS},
                           clean_intensity_mean=1., observation_latency_ms=0.,
                           expected_count_mean=scale, sampled_count_mean=scale, sampled_count_max=scale,
                           zero_count_fraction=0. if scale else None)
                if scale == 1:
                    row.update(status="rejected", rejection_type="sampling_jump",
                               rejection_reason="spatial phase jump exceeds sampling guard", modal_bias_rad=None,
                               **{c:None for c in entry.NUMERIC_COLUMNS})
                rows.append(row)
    return rows


def test_reading_grid_all_attempts_no_imputation(cfg):
    rows = handmade_rows(cfg["quick"])
    entry.validate_rows(rows, cfg["quick"])
    for broken in (rows[:-1], rows+[rows[0]]):
        with pytest.raises(ValueError):
            entry.validate_rows(broken, cfg["quick"])
    for column, value in [("modal_rmse_rad", 0.), ("modal_bias_rad", [0.]*21)]:
        bad = deepcopy(rows)
        bad[2][column] = value
        with pytest.raises(ValueError, match="invented"):
            entry.validate_rows(bad, cfg["quick"])


@pytest.mark.parametrize("index,key,value", [
    (0,"noise_seed",1), (0,"counts_sha256","a"*64), (1,"noise_seed",1),
    (1,"expected_count_mean",1001.), (1,"sampled_count_mean",float("nan")),
    (1,"zero_count_fraction",2.), (1,"counts_sha256","wrong"),
    (1,"clean_intensity_sha256","d"*64), (1,"modal_rmse_rad",float("inf")),
    (1,"fixture_group","outside_21_modes"), (1,"fixture_id","changed"),
])
def test_bad_rows_rejected(cfg, index, key, value):
    rows = handmade_rows(cfg["quick"])
    rows[index][key] = value
    with pytest.raises(ValueError):
        entry.validate_rows(rows, cfg["quick"])


def test_unknown_and_noiseless_rejections_not_allowed(cfg):
    rows = handmade_rows(cfg["quick"])
    rows[0].update(status="rejected", rejection_reason="spatial phase jump exceeds sampling guard", rejection_type="sampling_jump")
    with pytest.raises(ValueError, match="zero-noise"):
        entry.validate_rows(rows, cfg["quick"])
    assert entry.static.rejection_type(ValueError("nonfinite measurement")) is None


def test_preflight_path_preservation_no_writes(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    kept = tmp_path/"outputs/kept"
    kept.mkdir(parents=True)
    for directory, exception in [("outputs/kept", FileExistsError), ("outputs", ValueError), ("../outside", ValueError)]:
        config = tmp_path/"unit_config.yaml"
        config.write_text(yaml.safe_dump({**cfg,"output_directory":directory}),encoding="utf-8")
        with pytest.raises(exception):
            entry.preflight(config)
    monkeypatch.setattr(entry, "preflight", lambda *a, **k:(cfg,cfg["quick"],tmp_path/"absent",torch.device("cpu"),{"unit":True}))
    assert entry.run(preflight_only=True) == {"unit":True}
    assert not (tmp_path/"absent").exists()


def test_no_cpu_fallback(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    monkeypatch.setattr(entry, "verify_prerequisites", lambda:{})
    monkeypatch.setattr(entry, "stream_manifest", lambda *a,**k:{})
    monkeypatch.setattr(entry, "resolve_device", lambda *a:torch.device("cpu"))
    path = tmp_path/"unit.yaml"
    path.write_text(yaml.safe_dump(cfg),encoding="utf-8")
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        entry.preflight(path)
    with pytest.raises(ValueError, match="CUDA"):
        entry.photon_intensity(torch.ones(1),None,None)
    with pytest.raises(ValueError, match="CUDA"):
        entry.summarize([],{},torch.device("cpu"))


def test_failure_and_success_lifecycle(cfg, tmp_path, monkeypatch):
    output = tmp_path/"outputs/unit_failure"
    report = dict(quick=True,frozen_sources={},stream_manifest={},status="UNIT_PREFLIGHT")
    monkeypatch.setattr(entry, "preflight", lambda *a,**k:(cfg,cfg["quick"],output,torch.device("cpu"),report))
    monkeypatch.setattr(entry, "source_manifest", lambda *a:{})
    def crash(*args):
        raise RuntimeError("intentional unit failure")
    monkeypatch.setattr(entry, "execute", crash)
    with pytest.raises(RuntimeError, match="intentional"):
        entry.run(quick=True)
    before = (output/"failure.json").read_bytes()
    failure = json.loads(before)
    assert failure["last_context"]["poisson_draws"] == 0 and failure["model_loads"] == 0
    assert not (output/"SUCCESS.json").exists()
    with pytest.raises(FileExistsError):
        entry.run(quick=True)
    assert (output/"failure.json").read_bytes() == before
    fresh = tmp_path/"outputs/unit_success"
    monkeypatch.setattr(entry, "preflight", lambda *a,**k:(cfg,cfg["quick"],fresh,torch.device("cpu"),report))
    monkeypatch.setattr(entry, "verify_prerequisites", lambda:{})
    monkeypatch.setattr(entry, "execute", lambda *a:dict(report,status="O2_D4_TECHNICAL_SMOKE_ONLY"))
    result = entry.run(quick=True)
    success = json.loads((fresh/"SUCCESS.json").read_text(encoding="utf-8"))
    assert result["status"] != report["status"]
    assert success["summary_sha256"] == entry.optics.file_sha256(fresh/"summary.json")
    assert not (fresh/"failure.json").exists()


@pytest.fixture(scope="module")
def gpu(cfg):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; no CPU optical fallback")
    entry.static.configure_runtime()
    device = entry.resolve_device("cuda")
    sensor, bridge = entry.optics.make_components(entry.read_config(cfg["optics_config"]),device)
    return device, sensor, bridge


def test_cuda_exact_counts_and_rng_independent_fixed_scale(gpu):
    device = gpu[0]
    clean = torch.tensor([[[0.,.25,1.,2.]]],device=device)
    intensity, record = entry.photon_intensity(clean,100.,8870001)
    generator = torch.Generator(device=device).manual_seed(8870001)
    counts = torch.poisson(clean.double()*100., generator=generator)
    assert torch.equal(intensity,(counts/100.).float())
    assert record["counts_sha256"] == entry.tensor_sha256(counts)
    assert record["noise_rng_after_draw_sha256"] == entry.tensor_sha256(generator.get_state())
    second, metadata = entry.photon_intensity(clean,100.,8870001)
    assert torch.equal(intensity,second) and record == metadata
    assert intensity[0,0,0] == 0 and record["expected_count_mean"] == 81.25
    zero, record = entry.photon_intensity(clean,None,None)
    assert torch.equal(zero,clean) and all(v is None for v in record.values())
    assert list(inspect.signature(entry.photon_intensity).parameters) == ["clean","scale","seed"]


@pytest.mark.parametrize("scale,seed", [(0.,8870002),(-1.,8870002),(float("nan"),8870002),(True,8870002),(1.,None),(None,8870002)])
def test_cuda_invalid_scale_seed(gpu,scale,seed):
    with pytest.raises(ValueError):
        entry.photon_intensity(torch.ones(1,device=gpu[0]),scale,seed)


def test_cuda_nonfinite_and_negative_input(gpu):
    for value in (-1.,float("nan"),float("inf")):
        with pytest.raises(ValueError,match="clean intensity"):
            entry.photon_intensity(torch.tensor([value],device=gpu[0]),1.,8870003)


def test_cuda_toy_poisson_mean_and_variance(gpu):
    # 明确的固定计数 CUDA 抽样单元，不是正式相机校准或噪声曲线。
    clean = torch.ones(1,256,256,device=gpu[0])
    values,_ = entry.photon_intensity(clean,10.,8870004)
    counts = values.double()*10
    assert float(counts.mean()) == pytest.approx(10.,abs=.08)
    assert float(counts.var(correction=0)) == pytest.approx(10.,abs=.3)


def test_cuda_full_phase_no_truth_in_readout(cfg,gpu):
    spec = {**cfg["quick"],"extra_fixture_seed":8870005}
    _,sensor,bridge = gpu
    phases,ids = entry.static.make_fixtures(cfg,spec,bridge)
    field = (torch.polar(torch.ones_like(phases),phases)*bridge.pupil).to(torch.complex64)
    images = sensor.render(field)
    measured = bridge.measure(sensor.reconstruct(images))
    target,_ = entry.optics.audit_known_phase(phases,bridge)
    assert float((measured.residual_rad.double()-target).abs().max()) < .001
    assert float(measured.fit_rmse_rad[-1]) > .09 and ids[-1]["fixture_index"] == 47
    assert list(inspect.signature(bridge.measure).parameters) == ["current_field"]


def test_cuda_summary_rejections_and_null_empty_cells(cfg,gpu):
    cells = entry.summarize(handmade_rows(cfg["quick"]),cfg["quick"],gpu[0])
    assert len(cells) == 9
    dark = next(c for c in cells if c["fixture_group"] == "all" and c["counts_per_intensity_unit"] == 1.)
    assert (dark["planned_attempts"],dark["valid_readings"],dark["rejected_readings"]) == (3,0,3)
    assert dark["valid_only_modal_rmse_rad"] is None and dark["rejection_fraction"] == 1
    assert dark["rejection_counts"]["sampling_jump"] == 3
    assert dark["whole_level_reliability_claim"] is False


@pytest.mark.parametrize("reason", [*entry.static.MEASUREMENT_REJECTIONS,"nonfinite measurement"])
def test_cuda_micro_execution_known_rejection_only(cfg,gpu,tmp_path,monkeypatch,reason):
    # 两次明确的 CUDA 技术读取；第二次模拟保护拒绝或程序错误，不执行正式扫描。
    device,sensor,bridge = gpu
    calls = []
    def measured(field):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError(reason)
        return bridge.measure(field)
    wrapper = SimpleNamespace(device=bridge.device,basis=bridge.basis,pupil=bridge.pupil,
                              tolerances=bridge.tolerances,measure=measured)
    monkeypatch.setattr(entry.optics,"make_components",lambda *a:(sensor,wrapper))
    spec = dict(fixture_indices=[0],repetitions=1,count_scales=[None,1.],
                extra_fixture_seed=8870006,noise_seed_base=8871000)
    context = dict(clean_images_generated=0,poisson_draws=0,measurement_attempts_completed=0,valid_readings=0,rejected_readings=0)
    if reason not in entry.static.MEASUREMENT_REJECTIONS:
        with pytest.raises(ValueError,match="nonfinite"):
            entry.execute(cfg,spec,tmp_path,device,{"quick":True},context)
        assert context["measurement_attempts_completed"] == 1
        assert not (tmp_path/"fixtures.pt").exists()
    else:
        result = entry.execute(cfg,spec,tmp_path,device,{"quick":True},context)
        assert result["completed_measurement_attempts"] == 2 and result["completed_poisson_draws"] == 1
        assert result["valid_readings"] == result["rejected_readings"] == 1 and result["cells"] == []
        import csv
        with (tmp_path/"measurements.csv").open(encoding="utf-8",newline="") as table:
            rows = list(csv.DictReader(table))
        assert rows[-1]["status"] == "rejected" and rows[-1]["modal_bias_rad"] == "null"
        assert rows[-1]["modal_rmse_rad"] == ""
    assert len(calls) == 2 and context["poisson_draws"] == 1


def test_import_inert_help_and_sealed_src_bundle():
    code="from pathlib import Path\nfrom unittest.mock import patch\nimport torch\nwith patch.object(Path,'mkdir',side_effect=AssertionError('write')),patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('GPU')):\n import observation_bridge.photon_noise_diagnostic\n import scripts.diagnose_observation_bridge_o2_photon_noise\n"
    subprocess.run([sys.executable,"-B","-c",code],cwd=entry.ROOT,capture_output=True,check=True)
    result = subprocess.run([sys.executable,"-B","-X","utf8","scripts/diagnose_observation_bridge_o2_photon_noise.py","--help"],
                            cwd=entry.ROOT,capture_output=True,text=True,encoding="utf-8",check=True)
    assert "--quick" in result.stdout and "--preflight-only" in result.stdout
    assert entry.optics.sealed_bundle_sha256() == entry.optics.BUNDLE_SHA256
