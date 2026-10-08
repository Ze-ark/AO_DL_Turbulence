"""O2-D3 小型确定性检查；手写统计和微型回合不是开发性能结果。"""
from copy import deepcopy
import inspect
import json
import subprocess
import sys

import pytest
import torch

from observation_bridge import noise_development as entry


@pytest.fixture(scope="module")
def cfg():
    return entry.read_config()


@pytest.mark.parametrize("key,value", [("device", "cpu"), ("gain_threshold", .01), ("policy_scale", 2),
    ("selector_start_step", 24), ("camera_noise_units", "electrons"), ("camera_draw_including_noiseless", False),
    ("batch_size", 1), ("controller_branches", True), ("candidate_epsilon", .2)])
def test_fixed_contract(cfg, key, value):
    with pytest.raises(ValueError, match="fixed development"):
        entry.validate_config({**cfg, key: value})


@pytest.mark.parametrize("key,subkey,value", [("data", "episode_length", 32), ("data", "weather_count", 4),
    ("quick", "weather_seed_base", 8700000), ("integrator", "gain", .2),
    ("boundary", "training_updates", 1), ("boundary", "truth_fallback", True),
    ("boundary", "independent_confirmation", True), ("boundary", "real_slm_actions", True),
    ("statistics", "cluster", "frame"), ("statistics", "stratification", "none"),
    ("statistics", "bootstrap_seed", 8700000)])
def test_nested_contract(cfg, key, subkey, value):
    bad = deepcopy(cfg); bad[key][subkey] = value
    with pytest.raises(ValueError):
        entry.validate_config(bad)


def test_levels_and_interval_scope_fixed(cfg):
    entry.validate_config(cfg)
    bad = deepcopy(cfg); bad["camera_conditions"][2]["read_noise_std"] = .5
    with pytest.raises(ValueError):
        entry.validate_config(bad)
    assert cfg["statistics"]["interval_scope"] == "pointwise_descriptive_not_simultaneous_not_confirmation"
    assert all(not v for v in cfg["boundary"].values())


def test_budgets_folds_and_new_namespaces(cfg):
    full, quick = entry.budget(cfg["data"]), entry.budget(cfg["quick"])
    assert (full["episode_batches"], full["complete_episodes"], full["physical_transitions"]) == (416, 1248, 249600)
    assert (full["batched_steps"], full["policy_forward_calls"], full["scorer_forward_calls"]) == (83200, 76800, 50400)
    assert (full["paired_prefix_checks"], full["camera_batch_draws"], full["camera_family_frame_draws"]) == (288, 83616, 250848)
    assert (quick["episode_batches"], quick["complete_episodes"], quick["physical_transitions"]) == (26, 78, 2184)
    assert (quick["policy_forward_calls"], quick["scorer_forward_calls"], quick["paired_prefix_checks"]) == (672, 54, 18)
    for q in (False, True):
        streams = entry.stream_manifest(cfg, quick=q)
        assert streams["historical_manifests_checked"] == 19
        assert streams["fixed_draw_including_noiseless_and_initial_frame"]
        assert streams["bootstrap_draws_used"] is (not q)
    full_streams = entry.stream_manifest(cfg, quick=False)
    assert full_streams["weather_bases"] == [8700000+10*i for i in range(8)]
    assert list(full_streams["fold_assignment"].values()) == [0,1,2,3,0,1,2,3]


@pytest.mark.parametrize("seed", [8400000, 8460000, 8501000, 8511000, 8600000, 8660000, 8670000, 8770000, 8760000])
def test_weather_history_units_and_other_mode_collision(cfg, seed):
    bad = deepcopy(cfg); bad["data"]["weather_seed_base"] = seed
    with pytest.raises(RuntimeError, match="collision"):
        entry.stream_manifest(bad, quick=False)


@pytest.mark.parametrize("seed", [8300000, 8600000, 8501000, 8700000, 8760000, 8770000, 8670000])
def test_bootstrap_history_and_current_collision(cfg, seed):
    bad = deepcopy(cfg); bad["statistics"]["bootstrap_seed"] = seed
    with pytest.raises(RuntimeError, match="collision"):
        entry.stream_manifest(bad, quick=False)


def test_upstream_seals_without_executing_completed_experiments(monkeypatch):
    def forbidden(*a, **k):
        pytest.fail("sealed experiment entry invoked")
    for module in (entry.technical, entry.prior, entry.technical.static, entry.short):
        monkeypatch.setattr(module, "run", forbidden)
        monkeypatch.setattr(module, "preflight", forbidden)
    report = entry.verify_prerequisites()
    assert report["D2_artifacts_checked"] == 486
    assert report["D1_artifacts_checked"] == 16
    assert report["C_artifacts_checked"] == 426
    assert report["short_loop_artifacts_checked"] == 34
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
    """手写小网格；不调用环境、模型，也不当成物理性能结果。"""
    rows = []
    for ci, camera in enumerate(cfg["camera_conditions"]):
        if camera["id"] not in spec["camera_ids"]:
            continue
        for wi in range(spec["weather_count"]):
            seed = spec["weather_seed_base"] + spec["weather_seed_stride"]*wi
            for branch in entry.prior.controller_specs(cfg):
                factor = 1 if branch["member"] is None else (1.01 if branch["scorer_seed"] is None else 1.015-.002*ci)
                for fi, family in enumerate(cfg["family_ids"]):
                    metrics = {k:0.0 for k in entry.prior.METRICS}
                    metrics["power"] = (.2+.04*wi+.01*fi)*(1-.02*ci)*factor
                    rows.append(dict(camera_condition=camera["id"],read_noise_std=camera["read_noise_std"],
                        weather_seed=seed,weather_index=wi,family=family,family_index=fi,**branch,
                        episode_length=spec["episode_length"],camera_frames=spec["episode_length"]+1,
                        scorer_fold=wi%4 if branch["scorer_seed"] is not None else None,
                        turbulence_stream_seed=seed+fi,camera_stream_seed=seed+fi+cfg["camera_seed_offset"],
                        power_stream_seed=seed+60000000,failed=False,truncated=False,selected_nonoriginal_fraction=0.0,
                        trajectory_file=f"weather_{wi:02d}_{camera['id']}_{branch['controller']}.pt",**metrics))
    return rows


def test_complete_grid_and_quick_has_no_gain_statistics(cfg):
    rows = synthetic_rows(cfg,cfg["data"])
    assert len(entry.validate_rows(rows,cfg,cfg["data"])) == 1248
    quick_rows = synthetic_rows(cfg,cfg["quick"])
    assert len(quick_rows) == 78
    assert entry.summarize(quick_rows,cfg,cfg["quick"],device=torch.device("cpu"),quick=True) == {}
    for bad in (rows[:-1], rows+[rows[0]], [{**r,"power":float("nan")} for r in rows]):
        with pytest.raises(ValueError):
            entry.validate_rows(bad,cfg,cfg["data"])


@pytest.mark.parametrize("key,value", [("read_noise_std", .2), ("scorer_fold", 3), ("trajectory_file", "other.pt"),
    ("episode_length", 199), ("camera_frames", 200), ("power", 1.1), ("phase_rmse", -.1),
    ("failed", True), ("camera_stream_seed", 1)])
def test_grid_rejects_wrong_provenance_or_metric(cfg,key,value):
    rows = synthetic_rows(cfg,cfg["data"])
    next(r for r in rows if r["scorer_seed"] is not None)[key] = value
    with pytest.raises(ValueError):
        entry.validate_rows(rows,cfg,cfg["data"])


@pytest.fixture(scope="module")
def gpu(cfg):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; never CPU fallback")
    entry.short.configure_runtime()
    device = entry.resolve_device("cuda")
    sensor,bridge = entry.optics.make_components(entry.read_config(cfg["optics_config"]),device)
    return device,sensor,bridge,entry.read_config(cfg["parent"])


def test_cuda_handwritten_four_level_statistics_ratio_of_means_and_paired_effects(cfg,gpu):
    rows = synthetic_rows(cfg,cfg["data"])
    result = entry.summarize(rows,cfg,cfg["data"],device=gpu[0],quick=False)
    assert result["statistical_backend"] == "cuda"
    assert result["independent_weather_clusters"] == 8 and not result["frame_independence_assumed"]
    assert result["gain_threshold"] is None and not result["historical_gate_reclassification"]
    assert not result["multiple_comparison_significance_claims"] and not result["independent_confirmation"]
    assert "camera_noise_effect" not in result and "current_increment_change_noisy_minus_noiseless" not in result
    assert len(result["family_table"]) == 12
    for ci,camera in enumerate(cfg["camera_conditions"]):
        cell = result["cells"][camera["id"]]
        expected = .015-.002*ci
        comparison = cell["comparisons"]["current_vs_integrator"]
        assert comparison["relative_gain"] == pytest.approx(expected)
        assert comparison["relative_gain_descriptive_ci95"] == pytest.approx([expected,expected])
        assert len(cell["per_controller_means"]) == 13
        if ci:
            effect = result["camera_noise_effects_vs_noiseless"][camera["id"]]
            assert effect["current_relative_gain_change_fraction"] == pytest.approx(-.002*ci)
            assert effect["current_relative_gain_change_descriptive_ci95_fraction"] == pytest.approx([-.002*ci]*2)
            baseline_mean = .35
            assert effect["method_power_changes"]["integrator"]["noisy_minus_noiseless_power"] == pytest.approx(-.02*ci*baseline_mean)
            increment = baseline_mean*((1-.02*ci)*expected-.015)
            assert effect["current_increment_change_noisy_minus_noiseless"]["mean_power_difference"] == pytest.approx(increment)
            assert len(effect["method_power_changes"]["current"]["per_weather_power_difference"]) == 8
    with pytest.raises(ValueError,match="require CUDA"):
        entry.summarize(rows,cfg,cfg["data"],device=torch.device("cpu"),quick=False)


def test_cuda_zero_increment_and_nonpositive_baseline(cfg,gpu):
    rows = synthetic_rows(cfg,cfg["data"])
    for row in rows:
        row["power"] = .3
    result = entry.summarize(rows,cfg,cfg["data"],device=gpu[0],quick=False)
    assert all(c["comparisons"]["current_vs_integrator"]["relative_gain"] == 0 for c in result["cells"].values())
    assert all(c["current_relative_gain_change_fraction"] == 0 for c in result["camera_noise_effects_vs_noiseless"].values())
    for row in rows:
        if row["member"] is None:
            row["power"] = 0
    with pytest.raises(ValueError,match="nonpositive"):
        entry.summarize(rows,cfg,cfg["data"],device=gpu[0],quick=False)


def test_cuda_stratified_draws_keep_whole_weather_and_same_draw(cfg,gpu):
    draws = entry.prior.stratified_draws(8,repeats=40,seed=8770400,device=gpu[0])
    assert draws.shape == (40,8)
    for fold in range(4):
        assert (draws[:,2*fold:2*fold+2] % 4 == fold).all()
    assert torch.equal(draws,entry.prior.stratified_draws(8,repeats=40,seed=8770400,device=gpu[0]))


def test_cuda_four_level_micro_rows_replay_and_clocks(cfg,gpu,tmp_path):
    device,sensor,bridge,parent = gpu
    spec = dict(weather_seed_base=8770100,weather_count=1,weather_seed_stride=10,episode_length=3,
                camera_ids=cfg["data"]["camera_ids"])
    rng = []
    for camera in cfg["camera_conditions"]:
        context = {"physical_transitions":0}
        trace,audit,record = entry.technical.rollout(cfg,spec,parent,entry.prior.controller_specs(cfg)[0],seed=8770100,
            weather_index=0,camera=camera,sensor=sensor,bridge=bridge,policy=None,selector=None,
            progress=lambda _:None,context=context,partial_directory=tmp_path/"partial")
        assert context["physical_transitions"] == 9
        rng.append(record["camera_final_rng_sha256"])
        rows = entry.episode_rows(trace,audit,record,cfg,spec,f"weather_00_{camera['id']}_integrator.pt")
        assert len(rows) == 3 and all(r["camera_frames"] == 4 for r in rows)
        assert all(r["read_noise_std"] == camera["read_noise_std"] for r in rows)
        assert rows[0]["power"] == float(audit["action_power"][:,0].double().mean())
        assert trace["power_action_step"].tolist() == [0,1,2]
        assert trace["power_arrival_step"].tolist() == [1,2,3]
        assert torch.equal(trace["measured_power"],audit["action_power"])
        replay = entry.short.replay_visible(trace,{**cfg,"episode_length":3},None,None)
        assert replay["max_absolute_error"] == replay["new_environment_transitions"] == 0
        bad = dict(audit); bad["action_power"] = audit["action_power"].cpu()
        with pytest.raises(ValueError,match="finite CUDA"):
            entry.episode_rows(trace,bad,record,cfg,spec,"unit.pt")
    assert all(r == rng[0] for r in rng)


def test_failure_keeps_logs_and_no_automatic_retry(cfg,tmp_path,monkeypatch):
    # 纯文件生命周期测试：0 环境、0 模型、0 CUDA 运算。
    output = tmp_path/"outputs/intentional_failure"
    monkeypatch.setattr(entry,"preflight",lambda *a,**k:(cfg,cfg["quick"],output,torch.device("cpu"),{}, {}, {},[],{"stream_manifest":{}}))
    monkeypatch.setattr(entry,"source_manifest",lambda *_:{"unit_only":True})
    calls = []
    def crash(*a,**k):
        calls.append(1)
        raise ValueError("low intensity or hole inside frozen pupil")
    monkeypatch.setattr(entry.optics,"make_components",crash)
    with pytest.raises(ValueError,match="low intensity"):
        entry.run(quick=True)
    failure = json.loads((output/"failure.json").read_text(encoding="utf-8"))
    assert failure["last_context"]["physical_transitions"] == 0
    assert failure["last_context"]["incomplete_batch_size"] == 0
    assert not failure["automatic_retry"] and not failure["truth_fallback"]
    assert not (output/"SUCCESS.json").exists() and len(calls) == 1
    with pytest.raises(FileExistsError):
        entry.run(quick=True)
    assert len(calls) == 1


def test_complete_file_lifecycle_without_duplicate_status(cfg,tmp_path,monkeypatch):
    # 手写文件流程替身，不生成物理指标或科学结论。
    spec = {**cfg["quick"],"weather_seed_base":8770500,"episode_length":1}
    output = tmp_path/"outputs/unit_completion"
    report = dict(status="O2_D3_READY_FOR_TECHNICAL_SMOKE",quick=True,unit_scope="file_lifecycle_only",
                  stream_manifest=entry.prior._streams(cfg,spec),**entry.budget(spec),**cfg["boundary"])
    policies = {m:object() for m in range(3)}
    scorers = {(0,s):object() for s in cfg["scorer_seeds"]}
    monkeypatch.setattr(entry,"preflight",lambda *a,**k:(cfg,spec,output,torch.device("cpu"),{},policies,scorers,[],report))
    monkeypatch.setattr(entry,"source_manifest",lambda *_:{"unit_only":True})
    monkeypatch.setattr(entry,"verify_prerequisites",lambda:{})
    monkeypatch.setattr(entry.optics,"make_components",lambda *a,**k:(None,None))
    monkeypatch.setattr(entry.short,"replay_visible",lambda *a,**k:{"max_absolute_error":0.0,"new_environment_transitions":0})
    monkeypatch.setattr(entry.short,"require_prefix",lambda *a,**k:None)
    monkeypatch.setattr(torch.cuda,"memory_allocated",lambda *_:0)
    monkeypatch.setattr(torch.cuda,"get_device_name",lambda *_:"NO_GPU_FILE_UNIT")
    monkeypatch.setattr(entry.technical,"observation_quality",lambda *a:{"unit_only":True})
    rows = synthetic_rows(cfg,spec)
    def fake_rollout(cfg,spec,parent,branch,*,seed,weather_index,camera,sensor,bridge,policy,selector,progress,context,partial_directory):
        context["physical_transitions"] += 3
        progress(dict(weather_seed=seed,camera_condition=camera["id"],controller=branch["controller"],
                      observation_step=1,physical_transitions=context["physical_transitions"]))
        record = dict(**branch,weather_seed=seed,weather_index=0,camera_condition=camera["id"],
            read_noise_std=camera["read_noise_std"],complete_episodes=3,physical_transitions=3,
            scorer_fold=0 if branch["scorer_seed"] is not None else None,camera_frames_per_family=2,
            modal_error_max_rad=0.0,policy_forward_calls=1 if branch["member"] is not None else 0,scorer_forward_calls=0,
            camera_final_rng_sha256=["unit-a","unit-b","unit-c"])
        return {"unit_only":torch.zeros(1)},{"modal_rmse_rad":torch.zeros(2,3)},record
    monkeypatch.setattr(entry.technical,"rollout",fake_rollout)
    monkeypatch.setattr(entry,"episode_rows",lambda trace,audit,record,*a:[r for r in rows if r["controller"] == record["controller"] and r["camera_condition"] == record["camera_condition"]])
    result = entry.run(quick=True)
    assert result["status"] == "O2_D3_TECHNICAL_SMOKE_ONLY" and report["status"] == "O2_D3_READY_FOR_TECHNICAL_SMOKE"
    assert result["analysis"] == {} and result["unit_scope"] == "file_lifecycle_only"
    assert (result["completed_episode_batches"],result["completed_episodes"]) == (26,78)
    success = json.loads((output/"SUCCESS.json").read_text(encoding="utf-8"))
    assert success["summary_sha256"] == entry.optics.file_sha256(output/"summary.json")
    assert not (output/"failure.json").exists()


def test_import_inert_help_whitelist_and_frozen_src_bundle():
    code = "from pathlib import Path\nfrom unittest.mock import patch\nimport torch\nwith patch.object(Path,'mkdir',side_effect=AssertionError('write')),patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('GPU')):\n import observation_bridge.noise_development\n import scripts.evaluate_observation_bridge_o2_noise_development\n"
    subprocess.run([sys.executable,"-B","-c",code],cwd=entry.ROOT,capture_output=True,check=True)
    result = subprocess.run([sys.executable,"-B","-X","utf8","scripts/evaluate_observation_bridge_o2_noise_development.py","--help"],
                            cwd=entry.ROOT,capture_output=True,text=True,encoding="utf-8",check=True)
    assert "--quick" in result.stdout and "--preflight-only" in result.stdout
    assert list(inspect.signature(entry.short.choose_command).parameters) == ["history","valid","policy","selector","step","cfg"]
    assert list(inspect.signature(entry.short.replay_visible).parameters) == ["trace","cfg","policy","selector"]
    assert entry.optics.sealed_bundle_sha256() == entry.optics.BUNDLE_SHA256
