"""O2-D5 确定性小型单元测试；不产生正式性能结果。"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import inspect
import json

import pytest
import torch

from observation_bridge import photon_closed_loop as entry


@pytest.fixture(scope="module")
def cfg():
    return entry.read_config()


@pytest.mark.parametrize("key,value",[("device","cpu"),("gain_threshold",.01),("policy_scale",2),
    ("selector_start_step",24),("noise_units","electrons"),("read_noise_std",.001),
    ("shared_draw_across_levels",True),("image_dependent_normalization",True),("zero_noise_draws",True),
    ("camera_frame_seed_stride",1),("batch_size",1),("controller_branches",True),("candidate_epsilon",.2)])
def test_fixed_contract(cfg,key,value):
    with pytest.raises(ValueError,match="fixed single-factor"):
        entry.validate_config({**cfg,key:value})


@pytest.mark.parametrize("key,subkey,value",[("data","episode_length",16),("data","weather_count",1),
    ("data","weather_seed_stride",10),("quick","weather_seed_base",8900000),
    ("integrator","gain",.2),("boundary","training_updates",1),("boundary","truth_fallback",True),
    ("boundary","scientific_gain_analysis",True),("boundary","real_slm_actions",True)])
def test_nested_contract(cfg,key,subkey,value):
    bad = deepcopy(cfg); bad[key][subkey] = value
    with pytest.raises(ValueError):
        entry.validate_config(bad)


def test_light_levels_fixed_no_lowest_or_read_noise(cfg):
    entry.validate_config(cfg)
    bad = deepcopy(cfg); bad["camera_conditions"][2]["counts_per_intensity_unit"] = 1.0
    with pytest.raises(ValueError):
        entry.validate_config(bad)
    assert cfg["read_noise_std"] == 0 and cfg["gain_threshold"] is None
    assert not cfg["boundary"]["scientific_gain_analysis"]


def test_budget_and_all_four_folds(cfg):
    full,quick = entry.budget(cfg["data"]),entry.budget(cfg["quick"])
    assert (full["episode_batches"],full["complete_episodes"],full["physical_transitions"]) == (156,468,14976)
    assert (full["policy_forward_calls"],full["scorer_forward_calls"],full["paired_prefix_checks"]) == (4608,756,108)
    assert (full["camera_family_frames"],full["poisson_draws"]) == (15444,10296)
    assert (quick["complete_episodes"],quick["physical_transitions"],quick["poisson_draws"]) == (78,2184,1131)
    full_manifest = entry.stream_manifest(cfg,quick=False)
    quick_manifest = entry.stream_manifest(cfg,quick=True)
    assert len(full_manifest["camera"]) == len(set(full_manifest["camera"])) == 792
    assert len(quick_manifest["camera"]) == 87
    assert list(full_manifest["fold_assignment"].values()) == [0,1,2,3]
    assert full_manifest["historical_manifests_checked"] == 23
    assert not full_manifest["shared_draw_across_levels"] and full_manifest["zero_noise_draws"] == 0
    assert full_manifest["equal_poisson_rng_end_state_not_assumed"]


@pytest.mark.parametrize("seed",[8400000,8460000,8501000,8600000,8700000,8801000,8861000,8970000])
def test_history_and_unit_stream_collisions(cfg,seed):
    bad = deepcopy(cfg); bad["data"]["weather_seed_base"] = seed
    with pytest.raises(RuntimeError,match="collision"):
        entry.stream_manifest(bad,quick=False)


def test_frame_seed_identity_noiseless_no_seed(cfg):
    assert entry.frame_seed(cfg,8900000,"noiseless",0,0) is None
    assert entry.frame_seed(cfg,8900000,"photon_k100",0,0) == 189900000
    assert entry.frame_seed(cfg,8900000,"photon_k10",1,2) == 190900012
    for camera,frame,family in (("typo",0,0),("photon_k10",-1,0),("photon_k10",0,3),("photon_k10",True,0)):
        with pytest.raises(ValueError):
            entry.frame_seed(cfg,8900000,camera,frame,family)
    bad = deepcopy(cfg); bad["data"]["weather_seed_stride"] = 10
    with pytest.raises(RuntimeError,match="internal.*collision"):
        entry.stream_manifest(bad,quick=False)


def test_completed_experiments_not_reexecuted(monkeypatch):
    def forbidden(*a,**k):
        pytest.fail("sealed experiment entry invoked")
    for module in (entry.photon,entry.technical,entry.prior,entry.short,entry.photon.static,entry.photon.completed):
        monkeypatch.setattr(module,"run",forbidden)
        monkeypatch.setattr(module,"preflight",forbidden)
    report = entry.verify_prerequisites()
    assert report["D4_artifacts_checked"] == 16 and report["D3_artifacts_checked"] == 904
    assert report["real_calibration_unknown_fields"] == 18


def test_preflight_preserves_existing_broad_or_same_output(cfg,tmp_path,monkeypatch):
    monkeypatch.setattr(entry,"ROOT",tmp_path)
    (tmp_path/"outputs/already").mkdir(parents=True)
    for location,error in (("outputs/already",FileExistsError),("outputs",ValueError),("../outside",ValueError),
                           (cfg["quick_directory"],ValueError)):
        changed = {**cfg,"output_directory":location}
        monkeypatch.setattr(entry,"read_config",lambda _:changed)
        with pytest.raises(error):
            entry.preflight()


def test_preflight_cpu_no_fallback_no_output(cfg,tmp_path,monkeypatch):
    monkeypatch.setattr(entry,"ROOT",tmp_path)
    monkeypatch.setattr(entry,"read_config",lambda _:cfg)
    monkeypatch.setattr(entry,"verify_prerequisites",lambda:{})
    monkeypatch.setattr(entry,"stream_manifest",lambda *a,**k:{})
    monkeypatch.setattr(entry.short,"configure_runtime",lambda:None)
    monkeypatch.setattr(entry,"resolve_device",lambda _:torch.device("cpu"))
    with pytest.raises(RuntimeError,match="no CPU fallback"):
        entry.preflight()
    assert not (tmp_path/"outputs").exists()


def test_no_truth_or_light_label_argument_in_decision_and_camera():
    assert list(inspect.signature(entry.short.choose_command).parameters) == ["history","valid","policy","selector","step","cfg"]
    assert list(inspect.signature(entry.short.replay_visible).parameters) == ["trace","cfg","policy","selector"]
    assert list(inspect.signature(entry.PhotonCamera.apply).parameters) == ["self","clean"]
    assert list(inspect.signature(entry.photon.photon_intensity).parameters) == ["clean","scale","seed"]
    assert entry.PhotonPort.step is entry.short.HolographicEnvironmentPort.step
    assert entry.PhotonPort.reset is entry.short.HolographicEnvironmentPort.reset


def test_camera_cpu_or_unknown_level_rejected(cfg):
    with pytest.raises(ValueError,match="CUDA"):
        entry.PhotonCamera(torch.device("cpu"),cfg,8970000,cfg["camera_conditions"][0])
    with pytest.raises(ValueError):
        entry.PhotonCamera(torch.device("cuda"),cfg,8970000,{"id":"photon_k1","counts_per_intensity_unit":1.0})


@pytest.fixture(scope="module")
def gpu(cfg):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; never CPU fallback")
    entry.short.configure_runtime()
    device = entry.resolve_device("cuda")
    sensor,bridge = entry.optics.make_components(entry.read_config(cfg["optics_config"]),device)
    return device,sensor,bridge,entry.read_config(cfg["parent"])


def test_cuda_poisson_clock_reproducibility_independent_family_levels_and_no_zero_draw(cfg,gpu,monkeypatch):
    cameras = [entry.PhotonCamera(gpu[0],cfg,8970100,c) for c in cfg["camera_conditions"]]
    repeat = entry.PhotonCamera(gpu[0],cfg,8970100,cfg["camera_conditions"][2])
    clean = torch.full((3,512,512),1.5,device=gpu[0],dtype=torch.float32)
    poisson = torch.poisson
    calls = 0
    def track(*a,**k):
        nonlocal calls
        calls += 1
        return poisson(*a,**k)
    monkeypatch.setattr(torch,"poisson",track)
    images = [c.apply(clean) for c in cameras]
    assert cameras[0].draws == 0 and calls == 6 and torch.equal(images[0],clean)
    assert torch.equal(images[2],repeat.apply(clean)) and calls == 9
    assert not torch.equal(images[2][0],images[2][1])
    assert not torch.equal(images[1]*100,images[2]*10)
    for c in cameras+[repeat]:
        entry.validate_camera_rows(c.records,cfg,{"episode_length":0},weather=8970100,camera=c.camera)
    # 当前强度不同，但下一帧种子仍相同，不能因泊松消耗差异发生跨帧漂移。
    cameras[2].apply(clean*2); repeat.apply(clean)
    assert [r["noise_seed"] for r in cameras[2].records] == [r["noise_seed"] for r in repeat.records]
    assert cameras[2].seed_sequence_sha256() == repeat.seed_sequence_sha256()
    assert cameras[2].records[-1]["counts_sha256"] != repeat.records[-1]["counts_sha256"]


@pytest.mark.parametrize("problem",["cpu","shape","dtype","negative","nan"])
def test_cuda_bad_current_camera_input_rejected_without_draw(cfg,gpu,problem):
    camera = entry.PhotonCamera(gpu[0],cfg,8970100,cfg["camera_conditions"][2])
    clean = torch.ones((3,512,512),device=gpu[0],dtype=torch.float32)
    if problem == "cpu": clean = clean.cpu()
    elif problem == "shape": clean = clean[:1]
    elif problem == "dtype": clean = clean.double()
    elif problem == "negative": clean[0,0,0] = -1
    else: clean[0,0,0] = float("nan")
    with pytest.raises(ValueError):
        camera.apply(clean)
    assert camera.frames == camera.draws == 0 and camera.records == []


@pytest.fixture(scope="module")
def micro(cfg,gpu,tmp_path_factory):
    _,sensor,bridge,parent = gpu
    spec = {"episode_length":3}
    results = []
    for camera in cfg["camera_conditions"]:
        context = {"physical_transitions":0}
        result = entry.rollout(cfg,spec,parent,entry.prior.controller_specs(cfg)[0],seed=8970200,weather_index=0,
            camera=camera,sensor=sensor,bridge=bridge,policy=None,selector=None,progress=lambda _:None,
            context=context,partial_directory=tmp_path_factory.mktemp("partial")/"failed")
        assert context["physical_transitions"] == 9
        results.append(result)
    return spec,results


def test_cuda_27_micro_transitions_clocks_observation_and_replay(cfg,micro):
    spec,parts = micro
    for trace,audit,record,rows in parts:
        assert trace["history"].shape == (3,3,8,79) and trace["residual"].shape == (4,3,21)
        assert torch.equal((trace["residual"].double()-audit["joint_target_rad"]).square().mean(-1).sqrt(),audit["modal_rmse_rad"])
        assert audit["camera_frame_index"][:,0].tolist() == [0,1,2,3]
        assert trace["power_action_step"].tolist() == [0,1,2]
        assert trace["power_arrival_step"].tolist() == [1,2,3]
        assert torch.equal(trace["measured_power"],audit["action_power"])
        assert audit["camera_noise_seed"].reshape(-1).tolist() == [r["noise_seed"] if r["noise_seed"] is not None else -1 for r in rows]
        camera = next(c for c in cfg["camera_conditions"] if c["id"] == record["camera_condition"])
        entry.validate_camera_rows(rows,cfg,spec,weather=8970200,camera=camera)
        replay = entry.short.replay_visible(trace,{**cfg,"episode_length":3},None,None)
        assert replay["max_absolute_error"] == replay["new_environment_transitions"] == 0
    assert not torch.equal(parts[0][0]["residual"],parts[2][0]["residual"])
    quality = entry.observation_quality({c["id"]:[p[1]["modal_rmse_rad"]] for c,p in zip(cfg["camera_conditions"],parts)},parts[0][2]["modal_error_max_rad"],cfg)
    assert len(quality["cells"]) == 3 and all(c["observed_family_frames"] == 12 for c in quality["cells"])
    assert quality["targets_met"] and not quality["noisy_error_used_to_replace_measurements"]


def test_cuda_invalid_next_image_stops_saves_pending_no_fallback(cfg,gpu,tmp_path,monkeypatch):
    _,sensor,bridge,parent = gpu
    normal = bridge.measure
    calls = 0
    def invalid_next(field):
        nonlocal calls
        calls += 1
        if calls == 2: raise ValueError("spatial phase jump exceeds sampling guard")
        return normal(field)
    monkeypatch.setattr(bridge,"measure",invalid_next)
    context = {"physical_transitions":0}
    with pytest.raises(ValueError,match="sampling guard"):
        entry.rollout(cfg,{"episode_length":3},parent,entry.prior.controller_specs(cfg)[0],seed=8970300,
            weather_index=0,camera=cfg["camera_conditions"][2],sensor=sensor,bridge=bridge,policy=None,selector=None,
            progress=lambda _:None,context=context,partial_directory=tmp_path/"partial")
    assert calls == 2 and context["physical_transitions"] == 3 and context["camera_frames"] == 2
    assert context["current_batch_completed_steps"] == 0 and context["pending_action"]
    assert context["incomplete_batch_size"] == 3 and context["poisson_draws_in_current_batch"] == 6
    visible = torch.load(tmp_path/"partial/visible.pt",map_location=gpu[0],weights_only=True)
    pending = torch.load(tmp_path/"partial/pending_request.pt",map_location=gpu[0],weights_only=True)
    assert visible["residual"].shape == (1,3,21) and "measured_power" not in visible
    assert pending["requested_delta"].shape == (3,21)
    assert len(entry.short.read_json(tmp_path/"partial/camera_frames.json")) == 6
    assert not (tmp_path/"SUCCESS.json").exists()


def test_cuda_biased_measurement_never_replaced(cfg,gpu,monkeypatch):
    _,sensor,bridge,parent = gpu
    normal = bridge.measure
    def biased(field):
        m = normal(field)
        return replace(m,residual_rad=m.residual_rad+.05)
    monkeypatch.setattr(bridge,"measure",biased)
    env = entry.short.make_environment(parent,bridge.basis,8970400,1)
    camera = entry.PhotonCamera(bridge.device,cfg,8970400,cfg["camera_conditions"][2])
    readout,audit = entry.PhotonPort(env,sensor,bridge,camera,.001).reset(8970400)
    assert float(audit["modal_rmse_rad"].min()) > .04
    assert float((readout.residual.double()-audit["joint_target_rad"]).mean()) > .04
    assert env.step_count == 0
    quality = entry.observation_quality({"photon_k10":[audit["modal_rmse_rad"]]},0,cfg)
    assert not quality["targets_met"] and not quality["noisy_error_used_to_replace_measurements"]


def test_camera_metadata_detects_missing_duplicate_seed_scale_or_count(cfg,micro):
    rows = micro[1][2][3]
    spec,camera = micro[0],cfg["camera_conditions"][2]
    for bad in (rows[:-1],rows+[rows[0]]):
        with pytest.raises(ValueError): entry.validate_camera_rows(bad,cfg,spec,weather=8970200,camera=camera)
    for k,v in (("noise_seed",0),("family_index",2),("expected_count_mean",1),
                 ("sampled_count_max",.5),("zero_count_fraction",1.1),("counts_sha256",None)):
        bad = deepcopy(rows); bad[0][k] = v
        with pytest.raises(ValueError): entry.validate_camera_rows(bad,cfg,spec,weather=8970200,camera=camera)
    bad = deepcopy(micro[1][0][3]); bad[0]["sampled_count_mean"] = 0
    with pytest.raises(ValueError,match="consumed a draw"):
        entry.validate_camera_rows(bad,cfg,spec,weather=8970200,camera=cfg["camera_conditions"][0])


def test_records_complete_grid_fold_seed_and_budget_identity(cfg):
    spec = cfg["data"]
    rows = []
    for wi,w in enumerate(entry._streams(cfg,spec)["weather_bases"]):
        for camera in cfg["camera_conditions"]:
            identity = hashlib.sha256(json.dumps([entry.frame_seed(cfg,w,camera["id"],t,f)
                for t in range(33) for f in range(3)]).encode("utf-8")).hexdigest()
            for branch in entry.prior.controller_specs(cfg):
                rows.append(dict(**branch,weather_seed=w,weather_index=wi,camera_condition=camera["id"],
                    counts_per_intensity_unit=camera["counts_per_intensity_unit"],read_noise_std=0.0,
                    complete_episodes=3,physical_transitions=96,scorer_fold=wi%4 if branch["scorer_seed"] is not None else None,
                    camera_frames_per_family=33,poisson_draws=0 if camera["id"] == "noiseless" else 99,
                    camera_seed_sequence_sha256=identity,policy_forward_calls=32 if branch["member"] is not None else 0,
                    scorer_forward_calls=7 if branch["scorer_seed"] is not None else 0))
    entry.validate_records(rows,cfg,spec)
    for bad in (rows[:-1],rows+[rows[0]]):
        with pytest.raises(ValueError): entry.validate_records(bad,cfg,spec)
    for key,value in (("scorer_fold",3),("read_noise_std",.001),("poisson_draws",0),
                      ("camera_seed_sequence_sha256","0"*64),("counts_per_intensity_unit",1.0)):
        bad = deepcopy(rows)
        next(r for r in bad if r["scorer_seed"] is not None and r["camera_condition"] == "photon_k10")[key] = value
        with pytest.raises(ValueError,match="identity"):
            entry.validate_records(bad,cfg,spec)


def test_quality_empty_unknown_cpu_rejected(cfg):
    for parts in ({},{"typo":[torch.zeros(1)]},{"noiseless":[]},{"noiseless":[torch.zeros(1)]}):
        with pytest.raises(ValueError): entry.observation_quality(parts,0,cfg)


def test_cuda_models_all_folds_checkpoint_only_no_refit(cfg,gpu,micro,monkeypatch):
    def forbidden(*a,**k): pytest.fail("old trajectory loading or normalizer refit")
    monkeypatch.setattr(entry.short.frozen,"load_assets",forbidden)
    monkeypatch.setattr(entry.short.frozen.training,"fit_normalizer",forbidden)
    normal = torch.load
    loaded = []
    def checkpoint_only(path,*a,**k):
        assert "checkpoints" in str(path)
        loaded.append(str(path))
        return normal(path,*a,**k)
    monkeypatch.setattr(torch,"load",checkpoint_only)
    policies,scorers,manifest = entry.short.load_assets(gpu[0],gpu[3])
    assert len(loaded) == len(manifest) == 15
    trace = micro[1][2][0]
    h,v = trace["history"][0],trace["valid"][0]
    for fold in range(4):
        for member in range(3):
            for scorer_seed in cfg["scorer_seeds"]:
                original,selected,choice,prediction = entry.short.choose_command(h,v,policies[member],scorers[fold,scorer_seed],step=25,cfg=cfg)
                candidates = torch.stack(list(entry.short.frozen.source.candidate_commands(original,.1).values()),1)
                assert candidates.shape == (3,25,11) and prediction.shape == (3,23)
                assert torch.equal(selected,candidates[torch.arange(3,device=gpu[0]),choice])
    assert all(not p.requires_grad for model in policies.values() for p in model.parameters())


def test_preflight_only_never_creates_output_or_calls_model_env(cfg,gpu,tmp_path,monkeypatch):
    monkeypatch.setattr(entry,"ROOT",tmp_path)
    monkeypatch.setattr(entry,"read_config",lambda p:cfg if p == entry.CONFIG else gpu[3])
    monkeypatch.setattr(entry,"verify_prerequisites",lambda:{})
    monkeypatch.setattr(entry,"stream_manifest",lambda *a,**k:{"weather_bases":[8900000]})
    monkeypatch.setattr(entry.short,"load_assets",lambda *a:({0:object(),1:object(),2:object()},{i:object() for i in range(12)},[]))
    def forbidden(*a,**k): pytest.fail("preflight triggered physical/model execution")
    monkeypatch.setattr(entry,"rollout",forbidden)
    monkeypatch.setattr(entry.short,"make_environment",forbidden)
    monkeypatch.setattr(entry.short,"choose_command",forbidden)
    report = entry.run(preflight_only=True)
    assert report["preflight_environment_transitions"] == report["preflight_model_forward_calls"] == 0
    assert report["loaded_policies"] == 3 and report["loaded_scorers"] == 12
    assert not (tmp_path/"outputs").exists()


def test_preflight_rejects_weather_seen_by_scorer(cfg,gpu,tmp_path,monkeypatch):
    monkeypatch.setattr(entry,"ROOT",tmp_path)
    monkeypatch.setattr(entry,"read_config",lambda p:cfg if p == entry.CONFIG else gpu[3])
    monkeypatch.setattr(entry,"verify_prerequisites",lambda:{})
    monkeypatch.setattr(entry,"stream_manifest",lambda *a,**k:{"weather_bases":[8900000]})
    monkeypatch.setattr(entry.short,"load_assets",lambda *a:({}, {}, [{"train_weather":[8900000]}]))
    with pytest.raises(RuntimeError,match="weather used"):
        entry.preflight()
    assert not (tmp_path/"outputs").exists()


def test_failure_preserves_report_no_success_and_no_retry(cfg,gpu,tmp_path,monkeypatch):
    output = tmp_path/"new_diagnostic"
    report = {**entry.budget(cfg["quick"]),"stream_manifest":{"weather_bases":[8960000]}}
    monkeypatch.setattr(entry,"preflight",lambda *a,**k:(cfg,cfg["quick"],output,gpu[0],gpu[3],{}, {}, [],report))
    monkeypatch.setattr(entry,"source_manifest",lambda *a:{})
    calls = 0
    def fail(*a,**k):
        nonlocal calls
        calls += 1
        raise RuntimeError("preserved original simulation failure")
    monkeypatch.setattr(entry,"rollout",fail)
    with pytest.raises(RuntimeError,match="preserved original"):
        entry.run(quick=True)
    failure = entry.short.read_json(output/"failure.json")
    assert calls == 1 and not (output/"SUCCESS.json").exists()
    assert failure["message"] == "preserved original simulation failure"
    assert not failure["automatic_retry"] and not failure["truth_fallback"]
    assert (output/"effective_config.json").exists()


def test_cuda_quality_rejects_nonfinite_negative_or_empty(cfg,gpu):
    for value in (float("nan"),float("inf"),-1.0):
        with pytest.raises(ValueError):
            entry.observation_quality({"photon_k10":[torch.full((1,),value,device=gpu[0])]},0,cfg)
    with pytest.raises(ValueError):
        entry.observation_quality({"photon_k10":[torch.empty(0,device=gpu[0])]},0,cfg)
