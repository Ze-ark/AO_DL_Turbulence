"""O2-D6 小型确定性测试；手写统计与微型回合不是正式科学结果。"""
from copy import deepcopy
import hashlib
import inspect
import json
import subprocess
import sys

import pytest
import torch

from observation_bridge import photon_development as entry


@pytest.fixture(scope="module")
def cfg():
    return entry.read_config()


@pytest.mark.parametrize("key,value", [("device", "cpu"), ("gain_threshold", .01), ("policy_scale", 2),
    ("selector_start_step", 24), ("read_noise_std", .001), ("noise_units", "electrons"),
    ("shared_draw_across_levels", True), ("image_dependent_normalization", True), ("batch_size", 1)])
def test_fixed_contract(cfg, key, value):
    with pytest.raises(ValueError, match="fixed photon development"):
        entry.validate_config({**cfg, key: value})


@pytest.mark.parametrize("key,sub,value", [("data", "episode_length", 32), ("data", "weather_count", 4),
    ("data", "weather_seed_stride", 1000), ("quick", "weather_seed_base", 9000000),
    ("integrator", "gain", .2), ("boundary", "training_updates", 1), ("boundary", "truth_fallback", True),
    ("boundary", "independent_confirmation", True), ("boundary", "real_slm_actions", True),
    ("statistics", "cluster", "frame"), ("statistics", "bootstrap_seed", 9000000)])
def test_nested_contract(cfg, key, sub, value):
    bad = deepcopy(cfg); bad[key][sub] = value
    with pytest.raises(ValueError):
        entry.validate_config(bad)


def test_fixed_light_levels_and_distinct_paths(cfg):
    entry.validate_config(cfg)
    bad = deepcopy(cfg); bad["camera_conditions"][2]["counts_per_intensity_unit"] = 1.0
    with pytest.raises(ValueError):
        entry.validate_config(bad)
    with pytest.raises(ValueError, match="distinct output"):
        entry.validate_config({**cfg, "quick_directory": cfg["output_directory"]})
    assert cfg["statistics"]["interval_scope"] == "pointwise_descriptive_not_simultaneous_not_confirmation"


def test_budgets_and_fresh_namespaces(cfg):
    full, quick = entry.budget(cfg["data"]), entry.budget(cfg["quick"])
    assert (full["episode_batches"], full["complete_episodes"], full["physical_transitions"]) == (312, 936, 187200)
    assert (full["batched_steps"], full["policy_forward_calls"], full["scorer_forward_calls"]) == (62400, 57600, 37800)
    assert (full["paired_prefix_checks"], full["camera_family_frames"], full["poisson_draws"]) == (216, 188136, 125424)
    assert (quick["episode_batches"], quick["complete_episodes"], quick["physical_transitions"]) == (26, 78, 2184)
    assert (quick["policy_forward_calls"], quick["scorer_forward_calls"], quick["poisson_draws"]) == (672, 54, 1131)
    for q in (False, True):
        streams = entry.stream_manifest(cfg, quick=q)
        assert streams["historical_manifests_checked"] == 25
        assert streams["bootstrap_draws_used"] is (not q)
        assert streams["zero_noise_draws"] == 0 and not streams["shared_draw_across_levels"]
    s = entry.stream_manifest(cfg, quick=False)
    assert s["weather_bases"] == [9000000+10000*i for i in range(8)]
    assert list(s["fold_assignment"].values()) == [0, 1, 2, 3, 0, 1, 2, 3]
    assert len(s["camera"]) == len(set(s["camera"])) == 8*2*201*3


@pytest.mark.parametrize("seed", [8900000, 8960000, 8970000, 8800000, 8700000, 9160000, 9170000])
def test_prior_or_reserved_weather_collision(cfg, seed):
    bad = deepcopy(cfg); bad["data"]["weather_seed_base"] = seed
    with pytest.raises(RuntimeError, match="collision"):
        entry.stream_manifest(bad, quick=False)


@pytest.mark.parametrize("seed", [8900000, 9000000, 9160000, 9170000, 8790000, 8490000])
def test_bootstrap_collision(cfg, seed):
    bad = deepcopy(cfg); bad["statistics"]["bootstrap_seed"] = seed
    with pytest.raises(RuntimeError, match="collision"):
        entry.stream_manifest(bad, quick=False)


def test_200_frame_stride_must_not_reuse_camera_seed(cfg):
    bad = deepcopy(cfg); bad["data"]["weather_seed_stride"] = 1000
    with pytest.raises(RuntimeError, match="internal random stream collision"):
        entry.stream_manifest(bad, quick=False)


def test_upstream_seals_read_only_no_old_entry_or_tensor_load(monkeypatch):
    def forbidden(*a, **k):
        pytest.fail("old experiment or tensor load invoked")
    for module in (entry.technical, entry.prior, entry.short, entry.technical.photon, entry.technical.photon.completed):
        monkeypatch.setattr(module, "run", forbidden)
        monkeypatch.setattr(module, "preflight", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    report = entry.verify_prerequisites()
    assert report["D5_artifacts_checked"] == 564 and report["D4_artifacts_checked"] == 16
    assert report["D3_artifacts_checked"] == 904 and report["D2_artifacts_checked"] == 486
    assert report["real_calibration_unknown_fields"] == 18


def test_preflight_preserves_existing_or_broad_output(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    (tmp_path / "outputs/already").mkdir(parents=True)
    for location, error in (("outputs/already", FileExistsError), ("outputs", ValueError), ("../outside", ValueError)):
        monkeypatch.setattr(entry, "read_config", lambda _: {**cfg, "output_directory": location})
        with pytest.raises(error):
            entry.preflight()


def test_preflight_no_cpu_fallback_or_output(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    monkeypatch.setattr(entry, "read_config", lambda _: cfg)
    monkeypatch.setattr(entry, "verify_prerequisites", lambda: {})
    monkeypatch.setattr(entry, "stream_manifest", lambda *a, **k: {})
    monkeypatch.setattr(entry.short, "configure_runtime", lambda: None)
    monkeypatch.setattr(entry, "resolve_device", lambda _: torch.device("cpu"))
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        entry.preflight()
    assert not (tmp_path / "outputs").exists()


def synthetic_rows(cfg, spec):
    """手写完整天气小网格；不同光照设不同确定性效应，检验配对公式。"""
    rows = []
    for ci, camera in enumerate(cfg["camera_conditions"]):
        if camera["id"] not in spec["camera_ids"]:
            continue
        for wi in range(spec["weather_count"]):
            seed = spec["weather_seed_base"] + spec["weather_seed_stride"]*wi
            sequence = [entry.technical.frame_seed(cfg, seed, camera["id"], t, f)
                        for t in range(spec["episode_length"]+1) for f in range(3)]
            sha = hashlib.sha256(json.dumps(sequence).encode("utf-8")).hexdigest()
            for branch in entry.prior.controller_specs(cfg):
                factor = 1 if branch["member"] is None else (1.01 if branch["scorer_seed"] is None else 1.015-.002*ci)
                for fi, family in enumerate(cfg["family_ids"]):
                    metrics = {k: 0.0 for k in entry.prior.METRICS}
                    metrics["power"] = (.2+.04*wi+.01*fi)*(1-.02*ci)*factor
                    rows.append(dict(camera_condition=camera["id"], counts_per_intensity_unit=camera["counts_per_intensity_unit"],
                        read_noise_std=0.0, weather_seed=seed, weather_index=wi, family=family, family_index=fi, **branch,
                        episode_length=spec["episode_length"], camera_frames=spec["episode_length"]+1,
                        scorer_fold=wi % 4 if branch["scorer_seed"] is not None else None,
                        turbulence_stream_seed=seed+fi, power_stream_seed=seed+60000000,
                        camera_initial_frame_seed=entry.technical.frame_seed(cfg, seed, camera["id"], 0, fi),
                        camera_seed_sequence_sha256=sha, failed=False, truncated=False, selected_nonoriginal_fraction=0.0,
                        trajectory_file=f"weather_{wi:02d}_{camera['id']}_{branch['controller']}.pt", **metrics))
    return rows


def test_complete_grid_quick_no_gain_or_bootstrap(cfg):
    assert len(entry.validate_rows(synthetic_rows(cfg, cfg["data"]), cfg, cfg["data"])) == 936
    rows = synthetic_rows(cfg, cfg["quick"])
    assert len(rows) == 78
    assert entry.summarize(rows, cfg, cfg["quick"], device=torch.device("cpu"), quick=True) == {}
    with pytest.raises(ValueError, match="must not bootstrap"):
        entry.summarize(rows, cfg, cfg["quick"], device=torch.device("cpu"), quick=True, draws=torch.zeros(1))
    for bad in (rows[:-1], rows+[rows[0]]):
        with pytest.raises(ValueError):
            entry.validate_rows(bad, cfg, cfg["quick"])


@pytest.mark.parametrize("key,value", [("counts_per_intensity_unit", 3), ("read_noise_std", .001), ("scorer_fold", 3),
    ("trajectory_file", "old.pt"), ("episode_length", 199), ("camera_frames", 200), ("power", 1.1),
    ("phase_rmse", -.1), ("power", float("nan")), ("failed", True), ("camera_initial_frame_seed", 1),
    ("camera_seed_sequence_sha256", "b"*64), ("camera_negative_clip_fraction", .1)])
def test_grid_rejects_wrong_provenance_or_metric(cfg, key, value):
    rows = synthetic_rows(cfg, cfg["quick"])
    next(r for r in rows if r["scorer_seed"] is not None)[key] = value
    with pytest.raises(ValueError):
        entry.validate_rows(rows, cfg, cfg["quick"])


@pytest.fixture(scope="module")
def gpu(cfg):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; never CPU fallback")
    entry.short.configure_runtime()
    device = entry.resolve_device("cuda")
    sensor, bridge = entry.optics.make_components(entry.read_config(cfg["optics_config"]), device)
    return device, sensor, bridge, entry.read_config(cfg["parent"])


def test_cuda_paired_three_level_statistics_and_ratio_of_means(cfg, gpu):
    cfg = deepcopy(cfg); cfg["statistics"]["bootstrap_seed"] = 9170400
    rows = synthetic_rows(cfg, cfg["data"])
    result = entry.summarize(rows, cfg, cfg["data"], device=gpu[0], quick=False)
    assert result["statistical_backend"] == "cuda" and result["independent_weather_clusters"] == 8
    assert not result["frame_independence_assumed"] and not result["multiple_comparison_significance_claims"]
    assert result["gain_threshold"] is None and not result["independent_confirmation"]
    assert len(result["family_table"]) == 9
    for ci, camera in enumerate(cfg["camera_conditions"]):
        comparison = result["cells"][camera["id"]]["comparisons"]["current_vs_integrator"]
        expected = .015-.002*ci
        assert comparison["relative_gain"] == pytest.approx(expected)
        assert comparison["relative_gain_descriptive_ci95"] == pytest.approx([expected, expected])
        assert len(result["cells"][camera["id"]]["per_controller_means"]) == 13
        if ci:
            effect = result["photon_noise_effects_vs_noiseless"][camera["id"]]
            assert effect["current_relative_gain_change_fraction"] == pytest.approx(-.002*ci)
            assert effect["current_relative_gain_change_descriptive_ci95_fraction"] == pytest.approx([-.002*ci]*2)
            assert effect["method_power_changes"]["integrator"]["noisy_minus_noiseless_power"] == pytest.approx(-.02*ci*.35)
    with pytest.raises(ValueError, match="require CUDA"):
        entry.summarize(rows, cfg, cfg["data"], device=torch.device("cpu"), quick=False)


def test_cuda_zero_effect_and_nonpositive_baseline(cfg, gpu):
    cfg = deepcopy(cfg); cfg["statistics"]["bootstrap_seed"] = 9170401
    rows = synthetic_rows(cfg, cfg["data"])
    for row in rows:
        row["power"] = .3
    result = entry.summarize(rows, cfg, cfg["data"], device=gpu[0], quick=False)
    assert all(c["comparisons"]["current_vs_integrator"]["relative_gain"] == 0 for c in result["cells"].values())
    for row in rows:
        if row["member"] is None:
            row["power"] = 0
    with pytest.raises(ValueError, match="nonpositive"):
        entry.summarize(rows, cfg, cfg["data"], device=gpu[0], quick=False)


def test_cuda_ratio_of_means_not_mean_of_percentages_and_same_weather_pairing(cfg, gpu):
    cfg = deepcopy(cfg); cfg["statistics"]["bootstrap_seed"] = 9170403
    rows = synthetic_rows(cfg, cfg["data"])
    base = torch.tensor([.1+.05*i for i in range(8)], dtype=torch.float64, device=gpu[0])
    delta = torch.tensor([-.003+.001*i for i in range(8)], dtype=torch.float64, device=gpu[0])
    for row in rows:
        wi = row["weather_index"]
        # 三次/九个初始化均贡献，不挑最佳；家族取等权平均。
        row["power"] = float(base[wi]) + (float(delta[wi]) if row["scorer_seed"] is not None else 0.0)
    result = entry.summarize(rows, cfg, cfg["data"], device=gpu[0], quick=False)
    part = result["cells"]["noiseless"]["comparisons"]["current_vs_integrator"]
    assert part["relative_gain"] == pytest.approx(float(delta.mean()/base.mean()))
    assert part["relative_gain"] != pytest.approx(float((delta/base).mean()))
    assert part["positive_weather_count"] == 4
    draws = entry.prior.stratified_draws(8, repeats=5000, seed=9170403, device=gpu[0])
    assert part["relative_gain_descriptive_ci95"] == pytest.approx(entry.prior.paired_interval(delta, draws, base))
    assert all(effect["current_relative_gain_change_descriptive_ci95_fraction"] == pytest.approx([0.0, 0.0])
               for effect in result["photon_noise_effects_vs_noiseless"].values())


def test_cuda_fixed_fold_draws_and_invalid_draw_rejected(cfg, gpu):
    draws = entry.prior.stratified_draws(8, repeats=5000, seed=9170402, device=gpu[0])
    assert torch.equal(draws, entry.prior.stratified_draws(8, repeats=5000, seed=9170402, device=gpu[0]))
    rows = synthetic_rows(cfg, cfg["data"])
    for invalid in (draws.cpu(), draws.float(), draws[:2], draws.roll(1, 1), draws+8):
        with pytest.raises(ValueError, match="fixed-fold"):
            entry.summarize(rows, cfg, cfg["data"], device=gpu[0], quick=False, draws=invalid)


def test_cuda_micro_three_light_rows_clocks_replay_and_metadata(cfg, gpu, tmp_path):
    device, sensor, bridge, parent = gpu
    spec = dict(weather_seed_base=9170100, weather_count=1, weather_seed_stride=10000, episode_length=3,
                camera_ids=cfg["data"]["camera_ids"])
    rows = []
    for camera in cfg["camera_conditions"]:
        context = {"physical_transitions": 0}
        trace, audit, record, camera_rows = entry.technical.rollout(cfg, spec, parent, entry.prior.controller_specs(cfg)[0],
            seed=9170100, weather_index=0, camera=camera, sensor=sensor, bridge=bridge, policy=None, selector=None,
            progress=lambda _: None, context=context, partial_directory=tmp_path / "partial")
        assert context["physical_transitions"] == 9
        entry.technical.validate_camera_rows(camera_rows, cfg, spec, weather=9170100, camera=camera)
        part = entry.episode_rows(trace, audit, record, cfg, spec, f"weather_00_{camera['id']}_integrator.pt")
        assert part[0]["power"] == float(audit["action_power"][:, 0].double().mean())
        assert part[0]["counts_per_intensity_unit"] == camera["counts_per_intensity_unit"]
        assert torch.equal(trace["measured_power"], audit["action_power"])
        assert trace["power_action_step"].tolist() == [0, 1, 2] and trace["power_arrival_step"].tolist() == [1, 2, 3]
        assert entry.short.replay_visible(trace, {**cfg, "episode_length": 3}, None, None)["max_absolute_error"] == 0
        bad = dict(audit); bad["action_power"] = audit["action_power"].cpu()
        with pytest.raises(ValueError, match="finite CUDA"):
            entry.episode_rows(trace, bad, record, cfg, spec, "unit.pt")
        rows.extend(part)
    assert len(rows) == 9


def test_failure_keeps_logs_no_retry(cfg, tmp_path, monkeypatch):
    output = tmp_path / "outputs/unit_failure"
    monkeypatch.setattr(entry, "preflight", lambda *a, **k: (cfg, cfg["quick"], output, torch.device("cpu"), {}, {}, {}, [], {"stream_manifest": {}}))
    monkeypatch.setattr(entry, "source_manifest", lambda *_: {"unit_only": True})
    calls = []
    def crash(*a, **k):
        calls.append(1)
        raise ValueError("hole inside frozen pupil")
    monkeypatch.setattr(entry.optics, "make_components", crash)
    with pytest.raises(ValueError, match="hole"):
        entry.run(quick=True)
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["last_context"]["physical_transitions"] == 0
    assert not failure["automatic_retry"] and not failure["truth_fallback"]
    assert not (output / "SUCCESS.json").exists() and len(calls) == 1
    with pytest.raises(FileExistsError):
        entry.run(quick=True)
    assert len(calls) == 1


def test_quick_file_completion_seal_and_no_statistics(cfg, tmp_path, monkeypatch):
    # 纯文件生命周期替身：零图像、零物理转移、零模型前向。
    spec = {**cfg["quick"], "weather_seed_base": 9170500, "episode_length": 1}
    output = tmp_path / "outputs/unit_completion"
    report = dict(status="O2_D6_READY_FOR_TECHNICAL_SMOKE", quick=True, unit_scope="file_lifecycle_only",
                  stream_manifest=entry.technical._streams(cfg, spec), **entry.budget(spec), **cfg["boundary"])
    policies = {m: object() for m in range(3)}
    scorers = {(0, s): object() for s in cfg["scorer_seeds"]}
    monkeypatch.setattr(entry, "preflight", lambda *a, **k: (cfg, spec, output, torch.device("cpu"), {}, policies, scorers, [], report))
    monkeypatch.setattr(entry, "source_manifest", lambda *_: {"unit_only": True})
    monkeypatch.setattr(entry, "verify_prerequisites", lambda: {})
    monkeypatch.setattr(entry.optics, "make_components", lambda *a, **k: (None, None))
    monkeypatch.setattr(entry.short, "replay_visible", lambda *a, **k: {"max_absolute_error": 0.0, "new_environment_transitions": 0})
    monkeypatch.setattr(entry.short, "require_prefix", lambda *a, **k: None)
    monkeypatch.setattr(entry.technical, "validate_camera_rows", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *_: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *_: 0)
    monkeypatch.setattr(entry.technical, "observation_quality", lambda *a: {"targets_met": True, "unit_only": True})
    monkeypatch.setattr(entry.technical.photon.static, "_runtime", lambda *a: {"unit_only": True})
    rows = synthetic_rows(cfg, spec)
    def fake_rollout(cfg, spec, parent, branch, *, seed, weather_index, camera, sensor, bridge, policy, selector, progress, context, partial_directory):
        context["physical_transitions"] += 3
        progress(dict(weather_seed=seed, camera_condition=camera["id"], controller=branch["controller"],
                      observation_step=1, physical_transitions=context["physical_transitions"]))
        part = next(r for r in rows if r["controller"] == branch["controller"] and r["camera_condition"] == camera["id"])
        record = dict(**branch, weather_seed=seed, weather_index=0, camera_condition=camera["id"],
            counts_per_intensity_unit=camera["counts_per_intensity_unit"], read_noise_std=0.0, complete_episodes=3,
            physical_transitions=3, scorer_fold=0 if branch["scorer_seed"] is not None else None,
            camera_frames_per_family=2, poisson_draws=0 if camera["counts_per_intensity_unit"] is None else 6,
            camera_seed_sequence_sha256=part["camera_seed_sequence_sha256"], modal_error_max_rad=0.0,
            policy_forward_calls=1 if branch["member"] is not None else 0, scorer_forward_calls=0)
        return {"unit_only": torch.zeros(1)}, {"modal_rmse_rad": torch.zeros(2, 3)}, record, []
    monkeypatch.setattr(entry.technical, "rollout", fake_rollout)
    monkeypatch.setattr(entry, "episode_rows", lambda trace, audit, record, *a: [r for r in rows
                        if r["controller"] == record["controller"] and r["camera_condition"] == record["camera_condition"]])
    result = entry.run(quick=True)
    assert result["status"] == "O2_D6_TECHNICAL_SMOKE_ONLY" and report["status"] == "O2_D6_READY_FOR_TECHNICAL_SMOKE"
    assert result["analysis"] == {} and result["bootstrap_index_artifact"] is None
    assert (result["completed_episode_batches"], result["completed_episodes"]) == (26, 78)
    success = json.loads((output / "SUCCESS.json").read_text(encoding="utf-8"))
    assert success["summary_sha256"] == entry.optics.file_sha256(output / "summary.json")
    for name, digest in success["artifact_sha256"].items():
        assert entry.optics.file_sha256(output / name) == digest
    assert not (output / "bootstrap_draws.pt").exists() and not (output / "failure.json").exists()


def test_import_inert_help_and_frozen_whitelist():
    code = "from pathlib import Path\nfrom unittest.mock import patch\nimport torch\nwith patch.object(Path,'mkdir',side_effect=AssertionError('write')),patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('GPU')):\n import observation_bridge.photon_development\n import scripts.evaluate_observation_bridge_o2_photon_development\n"
    subprocess.run([sys.executable, "-B", "-c", code], cwd=entry.ROOT, capture_output=True, check=True)
    result = subprocess.run([sys.executable, "-B", "-X", "utf8", "scripts/evaluate_observation_bridge_o2_photon_development.py", "--help"],
                            cwd=entry.ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
    assert "--quick" in result.stdout and "--preflight-only" in result.stdout
    assert list(inspect.signature(entry.short.choose_command).parameters) == ["history", "valid", "policy", "selector", "step", "cfg"]
    assert entry.optics.sealed_bundle_sha256() == entry.optics.BUNDLE_SHA256
