"""O2-D7 小型确定性单元测试；不生成正式性能结果，不重跑旧入口。"""
from copy import deepcopy
from dataclasses import replace
import inspect

import pytest
import torch

from observation_bridge import combined_noise_closed_loop as entry


@pytest.fixture(scope="module")
def cfg():
    return entry.read_config()


@pytest.mark.parametrize("key,value", [
    ("device", "cpu"), ("gain_threshold", .01), ("policy_scale", 2), ("selector_start_step", 24),
    ("candidate_epsilon", .2), ("noise_units", "electrons"), ("noise_order", "gaussian_then_poisson"),
    ("component_seed_identity_shared_across_conditions", False), ("noise_components_independent", False),
    ("image_dependent_normalization", True), ("zero_noise_draws", True), ("camera_frame_seed_stride", 1),
    ("photon_seed_offset", 210000000), ("read_seed_offset", 200000000), ("batch_size", 1), ("controller_branches", True)])
def test_fixed_contract(cfg, key, value):
    with pytest.raises(ValueError, match="fixed four-cell"):
        entry.validate_config({**cfg, key: value})


@pytest.mark.parametrize("key,subkey,value", [
    ("data", "episode_length", 200), ("data", "weather_count", 1), ("data", "weather_seed_stride", 10),
    ("quick", "weather_seed_base", 9200000), ("integrator", "gain", .2), ("boundary", "training_updates", 1),
    ("boundary", "scientific_gain_analysis", True), ("boundary", "truth_fallback", True),
    ("boundary", "real_slm_actions", True), ("boundary", "automatic_retry", True),
    ("boundary", "old_confirmation_trajectory_access", True), ("boundary", "independent_confirmation", True)])
def test_nested_contract(cfg, key, subkey, value):
    bad = deepcopy(cfg)
    bad[key][subkey] = value
    with pytest.raises(ValueError):
        entry.validate_config(bad)


def test_four_cells_fixed_and_inherited_targets(cfg):
    entry.validate_config(cfg)
    assert [(c["counts_per_intensity_unit"], c["read_noise_std"]) for c in cfg["camera_conditions"]] == [
        (None, 0), (None, 1), (10, 0), (10, 1)]
    for key, value in (("counts_per_intensity_unit", 1.0), ("read_noise_std", 3.0)):
        bad = deepcopy(cfg)
        bad["camera_conditions"][3][key] = value
        with pytest.raises(ValueError):
            entry.validate_config(bad)
    assert cfg["thresholds"] == entry.technical.read_config()["thresholds"]
    assert cfg["gain_threshold"] is None and not cfg["boundary"]["scientific_gain_analysis"]


def test_budget_streams_four_folds_component_sharing(cfg):
    full, quick = entry.budget(cfg, cfg["data"]), entry.budget(cfg, cfg["quick"])
    assert (full["episode_batches"], full["complete_episodes"], full["physical_transitions"], full["batched_steps"]) == (208, 624, 19968, 6656)
    assert (full["policy_forward_calls"], full["scorer_forward_calls"], full["paired_prefix_checks"]) == (6144, 1008, 144)
    assert (full["camera_family_frames"], full["poisson_draws"], full["read_noise_draws"]) == (20592, 10296, 10296)
    assert (quick["complete_episodes"], quick["physical_transitions"], quick["poisson_draws"], quick["read_noise_draws"]) == (78, 2184, 1131, 1131)
    manifest = entry.stream_manifest(cfg, quick=False)
    other = entry.stream_manifest(cfg, quick=True)
    assert list(manifest["fold_assignment"].values()) == [0, 1, 2, 3]
    assert manifest["historical_manifests_checked"] == 27 and manifest["technical_unit_namespace"] == 9370000
    assert len(manifest["photon_camera"]) == len(set(manifest["photon_camera"])) == 396
    assert len(manifest["read_camera"]) == 396 and len(other["read_camera"]) == 87
    assert not set(manifest["photon_camera"]) & set(manifest["read_camera"])
    assert not entry.short._integers(manifest) & set(other["weather_bases"])
    assert manifest["component_seed_identity_shared_across_conditions"] and manifest["equal_poisson_rng_end_state_not_assumed"]


@pytest.mark.parametrize("seed", [8400000, 8460000, 8501000, 8600000, 8700000, 8801000, 8900000, 9000000,
                                      9160000, 9170000, 9180000, 9360000, 9370000])
def test_seed_collision_rejected(cfg, seed):
    bad = deepcopy(cfg)
    bad["data"]["weather_seed_base"] = seed
    with pytest.raises(RuntimeError, match="collision"):
        entry.stream_manifest(bad, quick=False)


def test_component_seeds_clock_and_zero_draws(cfg):
    assert entry.component_seed(cfg, 9200000, "noiseless", 0, 0, "read") is None
    assert entry.component_seed(cfg, 9200000, "noiseless", 0, 0, "photon") is None
    assert entry.component_seed(cfg, 9200000, "combined", 1, 2, "photon") == 209200012
    assert entry.component_seed(cfg, 9200000, "combined", 1, 2, "read") == 219200012
    assert entry.component_seed(cfg, 9200000, "read_only", 1, 2, "read") == 219200012
    assert entry.component_seed(cfg, 9200000, "photon_only", 1, 2, "photon") == 209200012
    for camera, frame, family, kind in (("typo", 0, 0, "read"), ("combined", -1, 0, "read"),
                                      ("combined", 0, 3, "read"), ("combined", True, 0, "read"),
                                      ("combined", 0, 0, "typo")):
        with pytest.raises(ValueError):
            entry.component_seed(cfg, 9200000, camera, frame, family, kind)
    bad = deepcopy(cfg)
    bad["data"]["weather_seed_stride"] = 10
    with pytest.raises(RuntimeError, match="internal.*collision"):
        entry.stream_manifest(bad, quick=False)


def test_no_old_experiment_entry_called(monkeypatch):
    def forbidden(*a, **k):
        pytest.fail("completed experiment reexecuted")
    for module in (entry.completed, entry.technical, entry.technical.technical, entry.photon,
                   entry.photon.completed, entry.photon.static, entry.prior, entry.short):
        monkeypatch.setattr(module, "run", forbidden)
        monkeypatch.setattr(module, "preflight", forbidden)
    report = entry.verify_prerequisites()
    assert report["D6_artifacts_checked"] == 1035 and report["D5_artifacts_checked"] == 564
    assert report["real_calibration_unknown_fields"] == 18


def test_output_preservation_and_scope(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    (tmp_path / "outputs/already").mkdir(parents=True)
    for location, error in (("outputs/already", FileExistsError), ("outputs", ValueError),
                            ("../outside", ValueError), (cfg["quick_directory"], ValueError)):
        changed = {**cfg, "output_directory": location}
        monkeypatch.setattr(entry, "read_config", lambda _: changed)
        with pytest.raises(error):
            entry.preflight()


def test_cpu_fallback_fails_before_output(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    monkeypatch.setattr(entry, "read_config", lambda _: cfg)
    monkeypatch.setattr(entry, "verify_prerequisites", lambda: {})
    monkeypatch.setattr(entry, "stream_manifest", lambda *a, **k: {})
    monkeypatch.setattr(entry.short, "configure_runtime", lambda: None)
    monkeypatch.setattr(entry, "resolve_device", lambda _: torch.device("cpu"))
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        entry.preflight()
    assert not (tmp_path / "outputs").exists()


def test_causal_api_does_not_receive_audit_or_camera_labels():
    assert list(inspect.signature(entry.short.choose_command).parameters) == ["history", "valid", "policy", "selector", "step", "cfg"]
    assert list(inspect.signature(entry.short.replay_visible).parameters) == ["trace", "cfg", "policy", "selector"]
    assert list(inspect.signature(entry.CombinedCamera.apply).parameters) == ["self", "clean"]
    assert list(inspect.signature(entry.combined_intensity).parameters) == ["clean", "scale", "read_std", "photon_seed", "read_seed"]
    assert entry.CombinedPort.step is entry.short.HolographicEnvironmentPort.step
    assert entry.CombinedPort.reset is entry.short.HolographicEnvironmentPort.reset


@pytest.fixture(scope="module")
def gpu(cfg):
    if not torch.cuda.is_available():
        pytest.fail("O2-D7 CUDA validation unavailable; no CPU fallback")
    entry.short.configure_runtime()
    device = entry.resolve_device("cuda")
    sensor, bridge = entry.optics.make_components(entry.read_config(cfg["optics_config"]), device)
    return device, sensor, bridge, entry.read_config(cfg["parent"])


def test_cuda_four_cell_composition_order_zero_draws_and_global_rng(cfg, gpu, monkeypatch):
    cameras = [entry.CombinedCamera(gpu[0], cfg, 9370100, c) for c in cfg["camera_conditions"]]
    clean = torch.full((3, 512, 512), 1.5, device=gpu[0], dtype=torch.float32)
    poisson, randn = torch.poisson, torch.randn
    calls = {"poisson": 0, "read": 0}

    def count_poisson(*a, **k):
        calls["poisson"] += 1
        return poisson(*a, **k)

    def count_read(*a, **k):
        calls["read"] += 1
        return randn(*a, **k)

    monkeypatch.setattr(torch, "poisson", count_poisson)
    monkeypatch.setattr(torch, "randn", count_read)
    global_state = torch.cuda.get_rng_state(gpu[0])
    images = [c.apply(clean) for c in cameras]
    assert calls == {"poisson": 6, "read": 6}
    assert torch.equal(global_state, torch.cuda.get_rng_state(gpu[0]))
    assert torch.equal(images[0], clean) and cameras[0].poisson_draws == cameras[0].read_noise_draws == 0
    for family in range(3):
        ps = entry.component_seed(cfg, 9370100, "combined", 0, family, "photon")
        rs = entry.component_seed(cfg, 9370100, "combined", 0, family, "read")
        normal = randn((512, 512), device=gpu[0], dtype=torch.float32,
                       generator=torch.Generator(device=gpu[0]).manual_seed(rs))
        counts = poisson(clean[family].double() * 10, generator=torch.Generator(device=gpu[0]).manual_seed(ps))
        expected = ((counts / 10).float() + normal).clamp_min(0)
        assert torch.equal(images[3][family], expected)
        assert torch.equal(images[2][family], (counts / 10).float())
        assert torch.equal(images[1][family], (clean[family] + normal).clamp_min(0))
        assert cameras[2].records[family]["counts_sha256"] == cameras[3].records[family]["counts_sha256"]
        assert cameras[1].records[family]["read_normal_sha256"] == cameras[3].records[family]["read_normal_sha256"]
    assert not torch.equal(images[3][0], images[3][1])
    repeat = entry.CombinedCamera(gpu[0], cfg, 9370100, cfg["camera_conditions"][3])
    assert torch.equal(images[3], repeat.apply(clean))
    # 分叉改变计数图，但不会改变下一帧身份；固定高斯分量仍可精确配对。
    cameras[3].apply(clean * 2)
    repeat.apply(clean)
    assert cameras[3].seed_sequence_sha256() == repeat.seed_sequence_sha256()
    assert cameras[3].records[-1]["counts_sha256"] != repeat.records[-1]["counts_sha256"]
    assert cameras[3].records[-1]["read_normal_sha256"] == repeat.records[-1]["read_normal_sha256"]
    for c in cameras + [repeat]:
        entry.validate_camera_rows(c.records, cfg, {"episode_length": c.frames - 1}, weather=9370100, camera=c.camera)


@pytest.mark.parametrize("problem", ["cpu", "shape", "dtype", "negative", "nan"])
def test_cuda_bad_camera_input_no_draws(cfg, gpu, problem):
    camera = entry.CombinedCamera(gpu[0], cfg, 9370100, cfg["camera_conditions"][3])
    clean = torch.ones((3, 512, 512), device=gpu[0], dtype=torch.float32)
    if problem == "cpu":
        clean = clean.cpu()
    elif problem == "shape":
        clean = clean[:1]
    elif problem == "dtype":
        clean = clean.double()
    elif problem == "negative":
        clean[0, 0, 0] = -1
    else:
        clean[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        camera.apply(clean)
    assert camera.frames == camera.poisson_draws == camera.read_noise_draws == 0 and camera.records == []


@pytest.mark.parametrize("scale,std,pseed,rseed", [
    (None, 0, 1, None), (10, 0, 1, 2), (10, 1, 1, 1), (10, 1, 1, None),
    (10, float("nan"), 1, 2), (10, -1, 1, 2), (0, 1, 1, 2), (10, 1, True, 2)])
def test_cuda_component_contract(cfg, gpu, scale, std, pseed, rseed):
    with pytest.raises(ValueError):
        entry.combined_intensity(torch.ones((4, 4), device=gpu[0]), scale, std, pseed, rseed)


def test_cuda_negative_clip_is_recorded_not_hidden(cfg, gpu):
    clean = torch.zeros((16, 16), device=gpu[0])
    image, metadata = entry.combined_intensity(clean, 10, 1, 9370150, 9370151)
    assert bool((image >= 0).all()) and 0 < metadata["negative_clip_fraction"] < 1
    assert metadata["sampled_count_mean"] == 0 and metadata["zero_count_fraction"] == 1
    assert metadata["preclip_intensity_sha256"] != metadata["measured_intensity_sha256"]


@pytest.fixture(scope="module")
def micro(cfg, gpu, tmp_path_factory):
    _, sensor, bridge, parent = gpu
    spec = {"episode_length": 3}
    parts = []
    for camera in cfg["camera_conditions"]:
        context = {"physical_transitions": 0}
        parts.append(entry.rollout(cfg, spec, parent, entry.prior.controller_specs(cfg)[0], seed=9370200,
            weather_index=0, camera=camera, sensor=sensor, bridge=bridge, policy=None, selector=None,
            progress=lambda _: None, context=context, partial_directory=tmp_path_factory.mktemp("d7") / "partial"))
        assert context["physical_transitions"] == 9
    return spec, parts


def test_cuda_36_micro_transitions_clocks_error_and_replay(cfg, micro):
    spec, parts = micro
    for trace, audit, record, rows in parts:
        assert trace["history"].shape == (3, 3, 8, 79) and trace["residual"].shape == (4, 3, 21)
        assert torch.equal((trace["residual"].double() - audit["joint_target_rad"]).square().mean(-1).sqrt(), audit["modal_rmse_rad"])
        assert torch.equal(trace["measured_power"], audit["action_power"])
        assert trace["power_action_step"].tolist() == [0, 1, 2] and trace["power_arrival_step"].tolist() == [1, 2, 3]
        assert audit["camera_frame_index"][:, 0].tolist() == [0, 1, 2, 3]
        for kind in ("photon", "read"):
            assert audit[f"camera_{kind}_seed"].reshape(-1).tolist() == [r[f"{kind}_seed"] if r[f"{kind}_seed"] is not None else -1 for r in rows]
        assert audit["camera_negative_clip_fraction"].reshape(-1).tolist() == [r["negative_clip_fraction"] for r in rows]
        camera = next(c for c in cfg["camera_conditions"] if c["id"] == record["camera_condition"])
        entry.validate_camera_rows(rows, cfg, spec, weather=9370200, camera=camera)
        replay = entry.short.replay_visible(trace, {**cfg, "episode_length": 3}, None, None)
        assert replay["max_absolute_error"] == replay["new_environment_transitions"] == 0
    quality = entry.observation_quality({c["id"]: [p[1]["modal_rmse_rad"]] for c, p in zip(cfg["camera_conditions"], parts)},
                                      parts[0][2]["modal_error_max_rad"], cfg)
    assert quality["targets_met"] and len(quality["cells"]) == 4
    assert all(c["observed_family_frames"] == 12 for c in quality["cells"])
    assert not quality["noisy_error_used_to_replace_measurements"]


def test_cuda_invalid_next_observation_preserves_pending_no_retry(cfg, gpu, tmp_path, monkeypatch):
    normal, calls = gpu[2].measure, 0

    def fail_next(field):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("spatial phase jump exceeds sampling guard")
        return normal(field)

    monkeypatch.setattr(gpu[2], "measure", fail_next)
    context = {"physical_transitions": 0}
    with pytest.raises(ValueError, match="sampling guard"):
        entry.rollout(cfg, {"episode_length": 3}, gpu[3], entry.prior.controller_specs(cfg)[0], seed=9370300,
            weather_index=0, camera=cfg["camera_conditions"][3], sensor=gpu[1], bridge=gpu[2], policy=None, selector=None,
            progress=lambda _: None, context=context, partial_directory=tmp_path / "partial")
    assert calls == 2 and context["physical_transitions"] == 3 and context["current_batch_completed_steps"] == 0
    assert context["pending_action"] and context["camera_frames"] == 2 and context["incomplete_batch_size"] == 3
    assert context["poisson_draws_in_current_batch"] == context["read_noise_draws_in_current_batch"] == 6
    visible = torch.load(tmp_path / "partial/visible.pt", map_location=gpu[0], weights_only=True)
    pending = torch.load(tmp_path / "partial/pending_request.pt", map_location=gpu[0], weights_only=True)
    assert visible["residual"].shape == (1, 3, 21) and "measured_power" not in visible
    assert pending["requested_delta"].shape == (3, 21)
    assert len(entry.short.read_json(tmp_path / "partial/camera_frames.json")) == 6
    assert not (tmp_path / "SUCCESS.json").exists()


def test_cuda_read_only_bias_not_mistaken_for_ideal_or_replaced(cfg, gpu, monkeypatch):
    normal = gpu[2].measure

    def biased(field):
        measured = normal(field)
        return replace(measured, residual_rad=measured.residual_rad + .05)

    monkeypatch.setattr(gpu[2], "measure", biased)
    env = entry.short.make_environment(gpu[3], gpu[2].basis, 9370400, 1)
    camera = entry.CombinedCamera(gpu[0], cfg, 9370400, cfg["camera_conditions"][1])
    readout, audit = entry.CombinedPort(env, gpu[1], gpu[2], camera, .001).reset(9370400)
    assert float(audit["modal_rmse_rad"].min()) > .04
    assert float((readout.residual.double() - audit["joint_target_rad"]).mean()) > .04
    quality = entry.observation_quality({"read_only": [audit["modal_rmse_rad"]]}, 0, cfg)
    assert not quality["targets_met"] and not quality["noisy_error_used_to_replace_measurements"]


@pytest.mark.parametrize("key,value", [("photon_seed", 0), ("read_seed", 0), ("family_index", 2),
    ("expected_count_mean", 1), ("sampled_count_max", .5), ("zero_count_fraction", 1.1),
    ("counts_sha256", None), ("read_normal_sha256", None), ("negative_clip_fraction", -1), ("read_noise_std", 0)])
def test_metadata_identity_tampering_rejected(cfg, micro, key, value):
    rows = deepcopy(micro[1][3][3])
    rows[0][key] = value
    with pytest.raises(ValueError):
        entry.validate_camera_rows(rows, cfg, micro[0], weather=9370200, camera=cfg["camera_conditions"][3])


def test_missing_duplicate_ideal_metadata_and_quality_guards(cfg, micro):
    rows = micro[1][3][3]
    for bad in (rows[:-1], rows + [rows[0]]):
        with pytest.raises(ValueError):
            entry.validate_camera_rows(bad, cfg, micro[0], weather=9370200, camera=cfg["camera_conditions"][3])
    bad = deepcopy(micro[1][0][3])
    bad[0]["read_normal_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        entry.validate_camera_rows(bad, cfg, micro[0], weather=9370200, camera=cfg["camera_conditions"][0])
    for parts in ({}, {"typo": [torch.zeros(1)]}, {"noiseless": []}, {"noiseless": [torch.zeros(1)]}):
        with pytest.raises(ValueError):
            entry.observation_quality(parts, 0, cfg)


def test_records_grid_fold_and_budget(cfg):
    rows, spec = [], cfg["data"]
    for wi, weather in enumerate(entry._streams(cfg, spec)["weather_bases"]):
        for camera in cfg["camera_conditions"]:
            for branch in entry.prior.controller_specs(cfg):
                rows.append(dict(**branch, weather_seed=weather, weather_index=wi, camera_condition=camera["id"],
                    counts_per_intensity_unit=camera["counts_per_intensity_unit"], read_noise_std=camera["read_noise_std"],
                    complete_episodes=3, physical_transitions=96, scorer_fold=wi % 4 if branch["scorer_seed"] is not None else None,
                    camera_frames_per_family=33, poisson_draws=99 if camera["counts_per_intensity_unit"] is not None else 0,
                    read_noise_draws=99 if camera["read_noise_std"] > 0 else 0,
                    camera_seed_sequence_sha256=entry.seed_sequence(cfg, weather, camera["id"], 32),
                    policy_forward_calls=32 if branch["member"] is not None else 0,
                    scorer_forward_calls=7 if branch["scorer_seed"] is not None else 0))
    entry.validate_records(rows, cfg, spec)
    for bad in (rows[:-1], rows + [rows[0]]):
        with pytest.raises(ValueError):
            entry.validate_records(bad, cfg, spec)
    for key, value in (("scorer_fold", 3), ("read_noise_std", .001), ("poisson_draws", 0),
                       ("read_noise_draws", 0), ("camera_seed_sequence_sha256", "0" * 64)):
        bad = deepcopy(rows)
        next(r for r in bad if r["scorer_seed"] is not None and r["camera_condition"] == "combined")[key] = value
        with pytest.raises(ValueError, match="identity"):
            entry.validate_records(bad, cfg, spec)


def test_cuda_model_assets_checkpoint_only_all_folds_no_refit(cfg, gpu, micro, monkeypatch):
    def forbidden(*a, **k):
        pytest.fail("old trajectory access or normalization refit")
    monkeypatch.setattr(entry.short.frozen, "load_assets", forbidden)
    monkeypatch.setattr(entry.short.frozen.training, "fit_normalizer", forbidden)
    normal, loaded = torch.load, []

    def checkpoint_only(path, *a, **k):
        assert "checkpoints" in str(path)
        loaded.append(str(path))
        return normal(path, *a, **k)

    monkeypatch.setattr(torch, "load", checkpoint_only)
    policies, scorers, manifest = entry.short.load_assets(gpu[0], gpu[3])
    assert len(loaded) == len(manifest) == 15
    trace = micro[1][3][0]
    for fold in range(4):
        for member in range(3):
            for seed in cfg["scorer_seeds"]:
                original, selected, choice, prediction = entry.short.choose_command(
                    trace["history"][0], trace["valid"][0], policies[member], scorers[fold, seed], step=25, cfg=cfg)
                candidates = torch.stack(list(entry.short.frozen.source.candidate_commands(original, .1).values()), 1)
                assert prediction.shape == (3, 23)
                assert torch.equal(selected, candidates[torch.arange(3, device=gpu[0]), choice])
    assert all(not p.requires_grad for model in policies.values() for p in model.parameters())


def test_preflight_only_no_output_no_forward(cfg, gpu, tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    monkeypatch.setattr(entry, "read_config", lambda p: cfg if p == entry.CONFIG else gpu[3])
    monkeypatch.setattr(entry, "verify_prerequisites", lambda: {})
    monkeypatch.setattr(entry, "stream_manifest", lambda *a, **k: {"weather_bases": [9200000]})
    monkeypatch.setattr(entry.short, "load_assets", lambda *a: ({i: object() for i in range(3)}, {i: object() for i in range(12)}, []))

    def forbidden(*a, **k):
        pytest.fail("preflight executed model/environment")
    monkeypatch.setattr(entry, "rollout", forbidden)
    monkeypatch.setattr(entry.short, "make_environment", forbidden)
    monkeypatch.setattr(entry.short, "choose_command", forbidden)
    report = entry.run(preflight_only=True)
    assert report["preflight_model_forward_calls"] == report["preflight_environment_transitions"] == 0
    assert report["complete_episodes"] == 624 and report["loaded_policies"] == 3 and report["loaded_scorers"] == 12
    assert not (tmp_path / "outputs").exists()


def test_preflight_rejects_seen_weather(cfg, gpu, tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    monkeypatch.setattr(entry, "read_config", lambda p: cfg if p == entry.CONFIG else gpu[3])
    monkeypatch.setattr(entry, "verify_prerequisites", lambda: {})
    monkeypatch.setattr(entry, "stream_manifest", lambda *a, **k: {"weather_bases": [9200000]})
    monkeypatch.setattr(entry.short, "load_assets", lambda *a: ({}, {}, [{"train_weather": [9200000]}]))
    with pytest.raises(RuntimeError, match="weather used"):
        entry.preflight()
    assert not (tmp_path / "outputs").exists()


def test_failure_keeps_original_error_no_retry_no_success(cfg, gpu, tmp_path, monkeypatch):
    output, calls = tmp_path / "new_d7", 0
    report = {**entry.budget(cfg, cfg["quick"]), "stream_manifest": {"weather_bases": [9360000]}}
    monkeypatch.setattr(entry, "preflight", lambda *a, **k: (cfg, cfg["quick"], output, gpu[0], gpu[3], {}, {}, [], report))
    monkeypatch.setattr(entry, "source_manifest", lambda *a: {})

    def fail(*a, **k):
        nonlocal calls
        calls += 1
        raise RuntimeError("preserved original failure")
    monkeypatch.setattr(entry, "rollout", fail)
    with pytest.raises(RuntimeError, match="preserved original"):
        entry.run(quick=True)
    failure = entry.short.read_json(output / "failure.json")
    assert calls == 1 and not (output / "SUCCESS.json").exists()
    assert failure["message"] == "preserved original failure"
    assert not failure["automatic_retry"] and not failure["truth_fallback"]


def test_cuda_quality_nonfinite_negative_empty(cfg, gpu):
    for value in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError):
            entry.observation_quality({"combined": [torch.full((1,), value, device=gpu[0])]}, 0, cfg)
    with pytest.raises(ValueError):
        entry.observation_quality({"combined": [torch.empty(0, device=gpu[0])]}, 0, cfg)
