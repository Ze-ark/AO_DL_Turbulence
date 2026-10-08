"""O2-D2 确定性单元检查；微型 CUDA 回合不支撑正式性能结论。"""
from copy import deepcopy
from dataclasses import replace
import inspect
from pathlib import Path

import pytest
import torch

from observation_bridge import noise_closed_loop as entry


@pytest.fixture(scope="module")
def cfg():
    return entry.read_config()


@pytest.mark.parametrize("key,value", [("device", "cpu"), ("gain_threshold", .01), ("policy_scale", 2),
    ("selector_start_step", 24), ("camera_noise_units", "electrons"), ("camera_draw_including_noiseless", False),
    ("batch_size", 1), ("controller_branches", True), ("candidate_epsilon", .2)])
def test_fixed_contract(cfg, key, value):
    with pytest.raises(ValueError, match="fixed technical"):
        entry.validate_config({**cfg, key: value})


@pytest.mark.parametrize("key,subkey,value", [("data", "episode_length", 16), ("data", "weather_count", 1),
    ("quick", "weather_seed_base", 8600000), ("integrator", "gain", .2),
    ("boundary", "training_updates", 1), ("boundary", "truth_fallback", True),
    ("boundary", "scientific_gain_analysis", True), ("boundary", "real_slm_actions", True)])
def test_nested_contract(cfg, key, subkey, value):
    bad = deepcopy(cfg); bad[key][subkey] = value
    with pytest.raises(ValueError):
        entry.validate_config(bad)


def test_noise_levels_fixed(cfg):
    entry.validate_config(cfg)
    bad = deepcopy(cfg); bad["camera_conditions"][2]["read_noise_std"] = .5
    with pytest.raises(ValueError):
        entry.validate_config(bad)


def test_budget_all_folds_and_selector_actually_enabled(cfg):
    full, smoke = entry.budget(cfg["data"]), entry.budget(cfg["quick"])
    assert (full["episode_batches"], full["complete_episodes"], full["physical_transitions"]) == (208, 624, 19968)
    assert (full["policy_forward_calls"], full["scorer_forward_calls"], full["paired_prefix_checks"]) == (6144, 1008, 144)
    assert (full["camera_batch_draws"], full["camera_family_frame_draws"]) == (6864, 20592)
    assert (smoke["complete_episodes"], smoke["physical_transitions"], smoke["scorer_forward_calls"]) == (78, 2184, 54)
    for q in (False, True):
        m = entry.stream_manifest(cfg, quick=q)
        assert m["historical_manifests_checked"] == 17
        assert m["fixed_draw_including_noiseless_and_initial_frame"]
    assert list(entry.stream_manifest(cfg, quick=False)["fold_assignment"].values()) == [0, 1, 2, 3]


@pytest.mark.parametrize("seed", [8400000, 8460000, 8501000, 8511000, 8670000])
def test_declared_C_D1_and_unit_stream_collision(cfg, seed):
    bad = deepcopy(cfg); bad["data"]["weather_seed_base"] = seed
    with pytest.raises(RuntimeError, match="collision"):
        entry.stream_manifest(bad, quick=False)


def test_completed_experiments_not_reexecuted(monkeypatch):
    def forbidden(*a, **k):
        pytest.fail("sealed experiment entry invoked")
    for module in (entry.prior, entry.static, entry.short):
        monkeypatch.setattr(module, "run", forbidden)
        monkeypatch.setattr(module, "preflight", forbidden)
    report = entry.verify_prerequisites()
    assert report["D1_artifacts_checked"] == 16
    assert report["C_artifacts_checked"] == 426
    assert report["short_loop_artifacts_checked"] == 34
    assert report["real_calibration_unknown_fields"] == 18


def test_preflight_rejects_existing_or_broad_output(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    (tmp_path / "outputs" / "already").mkdir(parents=True)
    for location, error in (("outputs/already", FileExistsError), ("outputs", ValueError), ("../outside", ValueError)):
        modified = {**cfg, "output_directory": location}
        monkeypatch.setattr(entry, "read_config", lambda _: modified)
        with pytest.raises(error):
            entry.preflight()


def test_preflight_cpu_does_not_fallback(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    monkeypatch.setattr(entry, "read_config", lambda _: cfg)
    monkeypatch.setattr(entry, "verify_prerequisites", lambda: {})
    monkeypatch.setattr(entry, "stream_manifest", lambda *a, **k: {})
    monkeypatch.setattr(entry.short, "configure_runtime", lambda: None)
    monkeypatch.setattr(entry, "resolve_device", lambda _: torch.device("cpu"))
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        entry.preflight()
    assert not (tmp_path / "outputs").exists()


def test_no_truth_argument_in_policy_decision_or_replay():
    assert list(inspect.signature(entry.short.choose_command).parameters) == ["history", "valid", "policy", "selector", "step", "cfg"]
    assert list(inspect.signature(entry.short.replay_visible).parameters) == ["trace", "cfg", "policy", "selector"]
    assert entry.PairedReadNoise.apply is entry.prior.PairedReadNoise.apply
    assert entry.PairedReadNoise.final_state_sha256 is entry.prior.PairedReadNoise.final_state_sha256


def test_camera_refuses_cpu_and_undeclared_level():
    for std in (0.0, .001, .3, 1.0):
        with pytest.raises(ValueError, match="CUDA"):
            entry.PairedReadNoise(torch.device("cpu"), [1, 2, 3], std)
    for std in (True, .1, 3, float("nan")):
        with pytest.raises(ValueError):
            entry.PairedReadNoise(torch.device("cuda"), [1, 2, 3], std)


@pytest.fixture(scope="module")
def gpu(cfg):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; never CPU fallback")
    entry.short.configure_runtime()
    device = entry.resolve_device("cuda")
    sensor, bridge = entry.optics.make_components(entry.read_config(cfg["optics_config"]), device)
    return device, sensor, bridge, entry.read_config(cfg["parent"])


def test_cuda_four_level_fixed_draws_clipping_and_independent_families(gpu):
    device = gpu[0]
    cameras = [entry.PairedReadNoise(device, [178670000+i for i in range(3)], std) for std in (0, .001, .3, 1)]
    repeat = entry.PairedReadNoise(device, [178670000+i for i in range(3)], 1)
    zero = torch.zeros((3,512,512), device=device)
    for _ in range(2):
        images = [c.apply(zero) for c in cameras]
        assert images[0][0].eq(0).all() and images[0][1].eq(0).all()
        assert torch.equal(images[-1][0], repeat.apply(zero)[0])
        for image, clip in images[1:]:
            assert image.min() == 0 and .48 < float(clip.mean()) < .52
            assert not torch.equal(image[0], image[1])
        assert torch.allclose(images[2][0]/.3, images[3][0], atol=1e-6, rtol=1e-6)
    assert len({tuple(c.final_state_sha256()) for c in cameras+[repeat]}) == 1
    assert all(c.frames == 2 for c in cameras)


@pytest.fixture(scope="module")
def micro(cfg, gpu, tmp_path_factory):
    _, sensor, bridge, parent = gpu
    spec = dict(weather_seed_base=8670100, weather_count=1, weather_seed_stride=10, episode_length=3,
                camera_ids=[c["id"] for c in cfg["camera_conditions"]])
    results = []
    for camera in cfg["camera_conditions"]:
        context = {"physical_transitions": 0}
        result = entry.rollout(cfg, spec, parent, entry.prior.controller_specs(cfg)[0], seed=8670100,
            weather_index=0, camera=camera, sensor=sensor, bridge=bridge, policy=None, selector=None,
            progress=lambda _: None, context=context, partial_directory=tmp_path_factory.mktemp("partial") / "failed")
        assert context["physical_transitions"] == 9
        results.append(result)
    return spec, results


def test_cuda_36_micro_transitions_clocks_and_replay(cfg, micro):
    spec, parts = micro
    assert len({tuple(r["camera_final_rng_sha256"]) for _, _, r in parts}) == 1
    for trace, audit, record in parts:
        assert trace["history"].shape == (3,3,8,79)
        assert trace["residual"].shape == (4,3,21)
        assert torch.equal((trace["residual"].double()-audit["joint_target_rad"]).square().mean(-1).sqrt(), audit["modal_rmse_rad"])
        assert audit["camera_draw_index"][:,0].tolist() == [0,1,2,3]
        assert trace["power_action_step"].tolist() == [0,1,2]
        assert trace["power_arrival_step"].tolist() == [1,2,3]
        assert torch.equal(trace["measured_power"], audit["action_power"])
        replay = entry.short.replay_visible(trace, {**cfg, "episode_length": 3}, None, None)
        assert replay["max_absolute_error"] == replay["new_environment_transitions"] == 0
    assert not torch.equal(parts[0][0]["residual"], parts[3][0]["residual"])
    quality = entry.observation_quality({c["id"]:[p[1]["modal_rmse_rad"]] for c,p in zip(cfg["camera_conditions"],parts)}, parts[0][2]["modal_error_max_rad"], cfg)
    assert len(quality["cells"]) == 4 and all(c["observed_family_frames"] == 12 for c in quality["cells"])
    assert quality["targets_met"] and not quality["noisy_error_used_to_replace_measurements"]


def test_cuda_quality_failure_does_not_impute_or_promote_gain(cfg, gpu):
    high = torch.full((2,3), .1, device=gpu[0])
    quality = entry.observation_quality({"large_read_noise": [high]}, 0, cfg)
    assert not quality["targets_met"] and not quality["noisy_error_used_to_replace_measurements"]
    assert cfg["gain_threshold"] is None and not cfg["boundary"]["scientific_gain_analysis"]


def test_cuda_invalid_next_image_stops_saves_pending_without_fallback(cfg, gpu, tmp_path, monkeypatch):
    _, sensor, bridge, parent = gpu
    normal = bridge.measure
    calls = 0
    def invalid_next(field):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("low intensity or hole inside frozen pupil")
        return normal(field)
    monkeypatch.setattr(bridge, "measure", invalid_next)
    context = {"physical_transitions":0}
    with pytest.raises(ValueError, match="low intensity"):
        entry.rollout(cfg, {"episode_length":3}, parent, entry.prior.controller_specs(cfg)[0], seed=8670200,
            weather_index=0, camera=cfg["camera_conditions"][3], sensor=sensor, bridge=bridge, policy=None,
            selector=None, progress=lambda _:None, context=context, partial_directory=tmp_path/"partial")
    assert calls == 2 and context["physical_transitions"] == 3
    assert context["camera_frames"] == 2 and context["current_batch_completed_steps"] == 0
    assert context["pending_action"] and context["incomplete_batch_size"] == 3
    visible = torch.load(tmp_path/"partial/visible.pt", map_location=gpu[0], weights_only=True)
    pending = torch.load(tmp_path/"partial/pending_request.pt", map_location=gpu[0], weights_only=True)
    assert visible["residual"].shape == (1,3,21) and "measured_power" not in visible
    assert pending["requested_delta"].shape == (3,21)
    assert not (tmp_path/"SUCCESS.json").exists()


def test_cuda_measured_bias_is_never_replaced(cfg, gpu, monkeypatch):
    _, sensor, bridge, parent = gpu
    normal = bridge.measure
    def biased(field):
        measured = normal(field)
        return replace(measured, residual_rad=measured.residual_rad+.05)
    monkeypatch.setattr(bridge, "measure", biased)
    env = entry.short.make_environment(parent, bridge.basis, 8670300, 1)
    noise = entry.PairedReadNoise(bridge.device, [178670300+i for i in range(3)], 1)
    readout, audit = entry.prior.DevelopmentPort(env, sensor, bridge, noise, .001).reset(8670300)
    assert float(audit["modal_rmse_rad"].min()) > .04
    assert float((readout.residual.double()-audit["joint_target_rad"]).mean()) > .04
    assert env.step_count == 0


def test_records_full_grid_no_drops_duplicates_noise_or_fold_change(cfg):
    spec = cfg["data"]
    stds = {c["id"]:c["read_noise_std"] for c in cfg["camera_conditions"]}
    records = []
    for wi in range(spec["weather_count"]):
        for camera in spec["camera_ids"]:
            for branch in entry.prior.controller_specs(cfg):
                records.append(dict(**branch, weather_seed=spec["weather_seed_base"]+10*wi, weather_index=wi,
                    camera_condition=camera, read_noise_std=stds[camera], complete_episodes=3, physical_transitions=96,
                    scorer_fold=wi%4 if branch["scorer_seed"] is not None else None, camera_frames_per_family=33,
                    policy_forward_calls=32 if branch["member"] is not None else 0,
                    scorer_forward_calls=7 if branch["scorer_seed"] is not None else 0))
    entry.validate_records(records, cfg, spec)
    for bad in (records[:-1], records+[records[0]]):
        with pytest.raises(ValueError):
            entry.validate_records(bad, cfg, spec)
    for key,value in (("read_noise_std",.2), ("scorer_fold",3), ("camera_frames_per_family",32), ("scorer_forward_calls",0)):
        bad = deepcopy(records)
        next(r for r in bad if r["scorer_seed"] is not None)[key] = value
        with pytest.raises(ValueError, match="identity"):
            entry.validate_records(bad,cfg,spec)


def test_quality_no_empty_unknown_or_cpu_cells(cfg):
    for parts in ({}, {"typo":[torch.zeros(1)]}, {"noiseless":[]}, {"noiseless":[torch.zeros(1)]}):
        with pytest.raises(ValueError):
            entry.observation_quality(parts,0,cfg)


def test_cuda_all_four_folds_and_candidates_use_saved_only(cfg,gpu,micro,monkeypatch):
    def forbidden(*a,**k):
        pytest.fail("old trajectory loading or normalizer refit")
    monkeypatch.setattr(entry.short.frozen,"load_assets",forbidden)
    monkeypatch.setattr(entry.short.frozen.training,"fit_normalizer",forbidden)
    original_load=torch.load
    loaded=[]
    def checkpoint_only(path,*a,**k):
        assert "checkpoints" in str(path)
        loaded.append(str(path))
        return original_load(path,*a,**k)
    monkeypatch.setattr(torch,"load",checkpoint_only)
    policies,scorers,manifest=entry.short.load_assets(gpu[0],gpu[3])
    assert len(loaded)==len(manifest)==15
    trace=micro[1][3][0]
    h,v=trace["history"][0],trace["valid"][0]
    for fold in range(4):
        for member in range(3):
            for scorer_seed in cfg["scorer_seeds"]:
                original,selected,choice,prediction=entry.short.choose_command(h,v,policies[member],scorers[fold,scorer_seed],step=25,cfg=cfg)
                candidates=torch.stack(list(entry.short.frozen.source.candidate_commands(original,.1).values()),1)
                assert candidates.shape==(3,25,11) and prediction.shape==(3,23)
                assert torch.equal(selected,candidates[torch.arange(3,device=gpu[0]),choice])
                assert bool(torch.isfinite(prediction).all())
    assert all(not p.requires_grad for model in policies.values() for p in model.parameters())
