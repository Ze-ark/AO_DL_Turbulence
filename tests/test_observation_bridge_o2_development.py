"""O2-C 小型确定性测试；微型 CUDA 回合不是正式开发性能结果。"""
from copy import deepcopy
from dataclasses import replace
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from observation_bridge import development as entry


@pytest.fixture(scope="module")
def cfg():
    return entry.read_config(entry.CONFIG)


@pytest.mark.parametrize("key,value", [("device", "cpu"), ("policy_scale", 2.0), ("gain_threshold", .01),
                                       ("selector_start_step", 24), ("camera_draw_including_noiseless", False),
                                       ("controller_branches", True), ("camera_seed_offset", 0)])
def test_fixed_development_contract(cfg, key, value):
    with pytest.raises(ValueError, match="fixed development"):
        entry.validate_config({**cfg, key: value})


def test_nested_conditions_and_boundaries_cannot_change(cfg):
    entry.validate_config(cfg)
    for key, subkey, value in (("data", "episode_length", 199), ("quick", "weather_seed_base", 8400000),
                              ("thresholds", "noisy_modal_rmse_mean_rad", 1),
                              ("statistics", "stratification", "frame"),
                              ("boundary", "independent_confirmation", True)):
        broken = deepcopy(cfg); broken[key][subkey] = value
        with pytest.raises(ValueError):
            entry.validate_config(broken)
    broken = deepcopy(cfg); broken["camera_conditions"][1]["read_noise_std"] = .01
    with pytest.raises(ValueError):
        entry.validate_config(broken)


def test_exact_budget_stream_disjointness_and_fixed_fold(cfg):
    formal, quick = entry.budget(cfg["data"]), entry.budget(cfg["quick"])
    assert (formal["episode_batches"], formal["complete_episodes"], formal["physical_transitions"]) == (208, 624, 124800)
    assert (formal["policy_forward_calls"], formal["scorer_forward_calls"]) == (38400, 25200)
    assert (formal["camera_batch_draws"], formal["camera_family_frame_draws"]) == (41808, 125424)
    assert (quick["complete_episodes"], quick["physical_transitions"]) == (78, 2184)
    for q in (False, True):
        manifest = entry.verify_streams(cfg, quick=q)
        assert manifest["historical_manifests_checked"] == 13
        assert manifest["fixed_draw_including_noiseless_and_initial_frame"]
    manifest = entry.verify_streams(cfg, quick=False)
    assert manifest["weather_bases"] == [8400000+10*i for i in range(8)]
    assert list(manifest["fold_assignment"].values()) == [0,1,2,3,0,1,2,3]
    broken = deepcopy(cfg); broken["data"]["weather_seed_base"] = 8300000
    with pytest.raises(RuntimeError, match="collision"):
        entry.verify_streams(broken, quick=False)


def test_completed_B_preserved_and_no_old_entry_rerun(cfg, monkeypatch):
    def forbidden(*a, **k):
        pytest.fail("completed experiment entry called")
    monkeypatch.setattr(entry.short, "run", forbidden)
    monkeypatch.setattr(entry.short, "preflight", forbidden)
    sources = entry.verify_short_loop(cfg)
    assert sources["short_loop_artifacts_checked"] == 34
    assert sources["frozen_C1_files_checked"] == 55
    assert sources["audited_short_loop_summary_sha256"] == entry.B_SUMMARY_SHA


def test_cpu_tiny_stratified_whole_weather_and_ratio_of_means():
    # 8 个手写数字的数学单元测试；不是正式控制器比较。
    draws = entry.stratified_draws(8, repeats=40, seed=1, device=torch.device("cpu"))
    assert draws.shape == (40, 8)
    for fold in range(4):
        assert (draws[:, 2*fold:2*fold+2] % 4 == fold).all()
    base = torch.arange(1,9).double()
    assert entry.paired_interval(base*.1, draws, base) == pytest.approx([.1,.1])
    assert torch.equal(draws,entry.stratified_draws(8,repeats=40,seed=1,device=torch.device("cpu")))
    with pytest.raises(ValueError,match="nonpositive"):
        entry.paired_interval(base,draws,base*0)


@pytest.fixture(scope="module")
def gpu(cfg):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; not CPU fallback")
    entry.short.configure_runtime()
    device = entry.resolve_device("cuda")
    sensor, bridge = entry.short.optics.make_components(entry.read_config(cfg["optics_config"]),device)
    return device, sensor, bridge, entry.read_config(cfg["parent"])


def test_camera_requires_cuda():
    with pytest.raises(ValueError, match="CUDA"):
        entry.PairedReadNoise(torch.device("cpu"),[1,2,3],.001)


def test_cuda_paired_gaussian_draws_noiseless_still_draws_and_negative_clips(gpu):
    device = gpu[0]
    seeds = [8470000,8470001,8470002]
    clean = entry.PairedReadNoise(device,seeds,0.0)
    noisy = entry.PairedReadNoise(device,seeds,.001)
    repeat = entry.PairedReadNoise(device,seeds,.001)
    zero = torch.zeros((3,512,512),device=device)
    for _ in range(2):
        a, clip_a = clean.apply(zero)
        b, clip_b = noisy.apply(zero)
        c, clip_c = repeat.apply(zero)
        assert torch.equal(a,zero) and clip_a.eq(0).all()
        assert torch.equal(b,c) and torch.equal(clip_b,clip_c)
        assert b.min()==0 and .48 < float(clip_b.mean()) < .52
    assert clean.frames==noisy.frames==2
    assert clean.final_state_sha256()==noisy.final_state_sha256()==repeat.final_state_sha256()
    assert not torch.equal(b[0],b[1])


@pytest.fixture(scope="module")
def micro(cfg,gpu):
    _,sensor,bridge,parent=gpu
    spec = dict(weather_seed_base=8470100,weather_count=1,weather_seed_stride=10,episode_length=4)
    result=[]
    for camera in cfg["camera_conditions"]:
        context={"physical_transitions":0}
        trace,audit,rows,record=entry.rollout(cfg,spec,parent,entry.controller_specs(cfg)[0],seed=8470100,
            weather_index=0,camera=camera,sensor=sensor,bridge=bridge,policy=None,selector=None,
            progress=lambda _:None,context=context)
        assert context["physical_transitions"]==12
        result.append((trace,audit,rows,record))
    return spec,result


def test_cuda_micro_24_transitions_noise_quality_and_causal_clock(cfg,micro):
    spec,parts=micro
    assert not torch.equal(parts[0][0]["residual"],parts[1][0]["residual"])
    assert parts[0][3]["camera_final_rng_sha256"]==parts[1][3]["camera_final_rng_sha256"]
    for trace,audit,rows,record in parts:
        assert trace["history"].shape==(4,3,8,79)
        assert trace["residual"].shape==(5,3,21)
        assert audit["joint_target_rad"].shape==(5,3,21)
        assert audit["camera_draw_index"][:,0].tolist()==[0,1,2,3,4]
        assert len(rows)==3 and all(r["camera_frames"]==5 and not r["failed"] for r in rows)
        assert trace["power_action_step"].tolist()==[0,1,2,3]
        assert trace["power_arrival_step"].tolist()==[1,2,3,4]
        assert torch.equal(trace["measured_power"],audit["action_power"])
        assert audit["observation_latency_ms"].min()>=0
        assert audit["decision_latency_ms"].shape==(4,3)
        assert record["scorer_forward_calls"]==0
        result=entry.short.replay_visible(trace,{**cfg,"episode_length":4},None,None)
        assert result["max_absolute_error"]==result["new_environment_transitions"]==0
    quality=entry.observation_quality(parts[0][3]["modal_error_max_rad"],[parts[1][1]["modal_rmse_rad"]],cfg)
    assert quality["noisy_family_observations"]==15 and quality["targets_met"]


def test_noisy_quality_not_substituted_or_used_for_gain_gate(cfg,micro):
    _,parts=micro
    trace,audit,_,_=parts[1]
    error=(trace["residual"].double()-audit["joint_target_rad"]).square().mean(-1).sqrt()
    assert torch.equal(error,audit["modal_rmse_rad"])
    quality=entry.observation_quality(0,[error.new_full((2,3),.1)],cfg)
    assert not quality["targets_met"] and not quality["noisy_error_used_to_replace_measurements"]
    assert quality["scope"]=="technical_observation_targets_not_compensation_gain_gate"


def test_cuda_noisy_measured_bias_remains_in_readout_no_truth_fallback(cfg,gpu,monkeypatch):
    _,sensor,bridge,parent=gpu
    env=entry.short.make_environment(parent,bridge.basis,8470200,1)
    noise=entry.PairedReadNoise(bridge.device,[178470200+i for i in range(3)],.001)
    port=entry.DevelopmentPort(env,sensor,bridge,noise,.001)
    normal=bridge.measure
    def measured_bias(field):
        measured=normal(field)
        return replace(measured,residual_rad=measured.residual_rad+.05)
    monkeypatch.setattr(bridge,"measure",measured_bias)
    readout,audit=port.reset(8470200)  # 零环境转移，故意注入读数误差。
    assert float(audit["modal_rmse_rad"].min())>.04
    assert float((readout.residual.double()-audit["joint_target_rad"]).mean())>.04
    def invalid(field):
        raise ValueError("intentional invalid sensor")
    monkeypatch.setattr(bridge,"measure",invalid)
    with pytest.raises(ValueError,match="invalid sensor"):
        port.observe()
    assert env.step_count==0


def test_cuda_all_folds_use_saved_normalizers_and_no_old_dataset(cfg,gpu,micro,monkeypatch):
    device,_,_,parent=gpu
    def forbidden(*a,**k):
        pytest.fail("old experiment/data loader or normalizer fit called")
    monkeypatch.setattr(entry.short.frozen,"load_assets",forbidden)
    monkeypatch.setattr(entry.short.frozen.training,"fit_normalizer",forbidden)
    normal_load=torch.load
    reads=[]
    def checkpoints_only(path,*a,**k):
        assert "checkpoints" in str(path)
        reads.append(str(path))
        return normal_load(path,*a,**k)
    monkeypatch.setattr(torch,"load",checkpoints_only)
    policies,scorers,manifest=entry.short.load_assets(device,parent)
    assert len(reads)==len(manifest)==15
    h,v=micro[1][1][0]["history"][0],micro[1][1][0]["valid"][0]
    for fold in range(4):
        for member in range(3):
            original,selected,choice,predicted=entry.short.choose_command(h,v,policies[member],
                scorers[fold,cfg["scorer_seeds"][member]],step=25,cfg=cfg)
            assert predicted.shape==(3,23) and bool(torch.isfinite(predicted).all())
            candidates=torch.stack(list(entry.short.frozen.source.candidate_commands(original,.1).values()),1)
            expected=candidates[torch.arange(3,device=device),choice]
            assert torch.equal(selected,expected)
    assert all(not p.requires_grad for model in policies.values() for p in model.parameters())


def synthetic_rows(cfg,spec):
    rows=[]
    for camera in cfg["camera_conditions"]:
        for wi,w in enumerate(entry._streams(cfg,spec)["weather_bases"]):
            for branch in entry.controller_specs(cfg):
                for fi,family in enumerate(cfg["family_ids"]):
                    metrics={k:0.0 for k in entry.METRICS}
                    metrics["power"]=1.0 if branch["controller"]=="integrator" else 1.1
                    rows.append(dict(camera_condition=camera["id"],weather_seed=w,weather_index=wi,
                        family=family,family_index=fi,**branch,episode_length=spec["episode_length"],
                        camera_frames=spec["episode_length"]+1,scorer_fold=wi%4 if branch["scorer_seed"] else None,
                        turbulence_stream_seed=w+fi,camera_stream_seed=w+fi+cfg["camera_seed_offset"],
                        power_stream_seed=w+60000000,failed=False,truncated=False,selected_nonoriginal_fraction=0.0,**metrics))
    return rows


def test_metadata_grid_rejects_drops_duplicates_wrong_fold_and_nonfinite(cfg):
    rows=synthetic_rows(cfg,cfg["quick"])
    assert len(entry.validate_rows(rows,cfg,cfg["quick"]))==78
    assert entry.summarize(rows,cfg,cfg["quick"],device=torch.device("cpu"),quick=True)=={}
    for broken in (rows[:-1],rows+[rows[0]], [{**r,"power":float("nan")} for r in rows]):
        with pytest.raises(ValueError):
            entry.validate_rows(broken,cfg,cfg["quick"])
    broken=deepcopy(rows)
    next(r for r in broken if r["scorer_seed"] is not None)["scorer_fold"]=1
    with pytest.raises(ValueError,match="identity"):
        entry.validate_rows(broken,cfg,cfg["quick"])


def test_cuda_tiny_synthetic_statistics_no_gate_promotion(cfg,gpu):
    # 手写常数网格的 GPU 数学检查；没有物理回合，不生成正式性能结论。
    rows=synthetic_rows(cfg,cfg["data"])
    result=entry.summarize(rows,cfg,cfg["data"],device=gpu[0],quick=False)
    assert result["independent_weather_clusters"]==8 and not result["frame_independence_assumed"]
    assert result["gain_threshold"] is None and not result["historical_gate_reclassification"]
    for cell in result["cells"].values():
        assert cell["comparisons"]["current_vs_integrator"]["relative_gain"]==pytest.approx(.1)
        assert cell["comparisons"]["current_vs_integrator"]["relative_gain_descriptive_ci95"]==pytest.approx([.1,.1])
    assert len(result["family_table"])==6
    with pytest.raises(ValueError,match="require CUDA"):
        entry.summarize(rows,cfg,cfg["data"],device=torch.device("cpu"),quick=False)


def test_preflight_preserves_existing_output_and_path_boundary(cfg,tmp_path,monkeypatch):
    monkeypatch.setattr(entry,"ROOT",tmp_path)
    kept=tmp_path/"outputs/kept"; kept.mkdir(parents=True)
    for name,exception in (("outputs/kept",FileExistsError),("outputs",ValueError),("../outside",ValueError)):
        path=tmp_path/"config.yaml"
        path.write_text(yaml.safe_dump({**cfg,"output_directory":name}),encoding="utf-8")
        with pytest.raises(exception):
            entry.preflight(path)
    assert kept.exists()


def test_failure_leaves_logs_no_auto_retry(cfg,tmp_path,monkeypatch):
    # 纯文件生命周期测试，不执行 GPU、环境或模型。
    output=tmp_path/"outputs/intentional_failure"
    monkeypatch.setattr(entry,"preflight",lambda *a,**k:(cfg,cfg["quick"],output,torch.device("cpu"),{}, {}, {},[],
                                                       {"stream_manifest":{}}))
    monkeypatch.setattr(entry,"source_manifest",lambda *_:{"unit_only":True})
    def crash(*a,**k):
        raise RuntimeError("intentional unit crash before environment")
    monkeypatch.setattr(entry.short.optics,"make_components",crash)
    with pytest.raises(RuntimeError,match="intentional"):
        entry.run(quick=True)
    failure=json.loads((output/"failure.json").read_text(encoding="utf-8"))
    assert failure["last_context"]["physical_transitions"]==0
    assert failure["incomplete_batch_size"]==0
    assert not (output/"SUCCESS.json").exists()
    with pytest.raises(FileExistsError):
        entry.run(quick=True)


def test_complete_file_lifecycle_overrides_preflight_status_without_duplicate_keyword(cfg,tmp_path,monkeypatch):
    # 纯 CPU 文件生命周期单元：手写记录，0 环境/模型/GPU 运算，无科学分析。
    spec=dict(weather_seed_base=8470400,weather_count=1,weather_seed_stride=10,episode_length=1)
    output=tmp_path/"outputs/unit_completion"
    report=dict(status="O2_C_READY_FOR_TECHNICAL_SMOKE",quick=True,unit_scope="file_lifecycle_only",
                stream_manifest=entry._streams(cfg,spec),**entry.budget(spec),**cfg["boundary"])
    monkeypatch.setattr(entry,"preflight",lambda *a,**k:(cfg,spec,output,torch.device("cpu"),{}, {}, {},[],report))
    monkeypatch.setattr(entry,"source_manifest",lambda *_:{"unit_only":True})
    monkeypatch.setattr(entry,"verify_short_loop",lambda *_:{})
    monkeypatch.setattr(entry.short.optics,"make_components",lambda *a,**k:(None,None))
    monkeypatch.setattr(entry.short,"replay_visible",lambda *a,**k:{"max_absolute_error":0.0,"new_environment_transitions":0})
    monkeypatch.setattr(entry.short,"require_prefix",lambda *a,**k:None)
    monkeypatch.setattr(torch.cuda,"memory_allocated",lambda *_:0)
    monkeypatch.setattr(torch.cuda,"get_device_name",lambda *_:"NO_GPU_FILE_UNIT")
    unit_rows=synthetic_rows(cfg,spec)
    def fake_rollout(cfg,spec,parent,branch,*,seed,weather_index,camera,sensor,bridge,policy,selector,progress,context):
        context["physical_transitions"]+=3
        progress(dict(weather_seed=seed,camera_condition=camera["id"],controller=branch["controller"],
                      observation_step=1,physical_transitions=context["physical_transitions"],modal_error_max_rad=0.0))
        rows=[r for r in unit_rows if r["controller"]==branch["controller"] and r["camera_condition"]==camera["id"]]
        record=dict(controller=branch["controller"],weather_seed=seed,weather_index=0,camera_condition=camera["id"],
            scorer_fold=0 if branch["scorer_seed"] is not None else None,
            modal_error_max_rad=0.0,policy_forward_calls=1 if branch["member"] is not None else 0,scorer_forward_calls=0,
            camera_final_rng_sha256=["unit-a","unit-b","unit-c"])
        return {"unit_only":torch.zeros(1)},{"modal_rmse_rad":torch.zeros(2,3)},rows,record
    monkeypatch.setattr(entry,"rollout",fake_rollout)
    # 占位模型只作为字典身份，rollout/replay 均被手写文件测试替身替代。
    policies={m:object() for m in range(3)}
    scorers={(0,s):(object(),{}) for s in cfg["scorer_seeds"]}
    monkeypatch.setattr(entry,"preflight",lambda *a,**k:(cfg,spec,output,torch.device("cpu"),{},policies,scorers,[],report))
    result=entry.run(quick=True)
    assert result["status"]=="O2_C_TECHNICAL_SMOKE_ONLY" and report["status"]=="O2_C_READY_FOR_TECHNICAL_SMOKE"
    assert result["analysis"]=={} and result["unit_scope"]=="file_lifecycle_only"
    assert (result["completed_episode_batches"],result["completed_episodes"])==(26,78)
    success=json.loads((output/"SUCCESS.json").read_text(encoding="utf-8"))
    assert success["summary_sha256"]==entry.short.optics.file_sha256(output/"summary.json")
    assert not (output/"failure.json").exists()
    output=tmp_path/"outputs/unit_summary_failure"
    normal_write=entry.short.optics.write_json
    def fail_only_summary(path,value):
        if path.name=="summary.json":
            raise OSError("intentional metadata write failure after complete unit grid")
        return normal_write(path,value)
    monkeypatch.setattr(entry.short.optics,"write_json",fail_only_summary)
    with pytest.raises(OSError,match="metadata write failure"):
        entry.run(quick=True)
    failure=json.loads((output/"failure.json").read_text(encoding="utf-8"))
    assert failure["last_context"]["completed_episode_batches"]==26
    assert failure["incomplete_batch_size"]==0  # 元数据故障不能伪装成物理回合失败。


def test_import_inert_whitelist_signature_help_and_frozen_bundle():
    code="from pathlib import Path\nfrom unittest.mock import patch\nimport torch\nwith patch.object(Path,'mkdir',side_effect=AssertionError('write')),patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('GPU')):\n import observation_bridge.development\n import scripts.evaluate_observation_bridge_o2_development\n"
    subprocess.run([sys.executable,"-B","-c",code],cwd=entry.ROOT,capture_output=True,check=True)
    result=subprocess.run([sys.executable,"-B","-X","utf8","scripts/evaluate_observation_bridge_o2_development.py","--help"],
        cwd=entry.ROOT,capture_output=True,text=True,encoding="utf-8",check=True)
    assert "--quick" in result.stdout and "--preflight-only" in result.stdout
    assert list(inspect.signature(entry.short.choose_command).parameters)==["history","valid","policy","selector","step","cfg"]
    assert entry.short.optics.sealed_bundle_sha256()==entry.short.optics.BUNDLE_SHA256
