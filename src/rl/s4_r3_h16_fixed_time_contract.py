"""A9冻结设计、数据隔离与只读预检。导入时不占用GPU或写文件。"""
from __future__ import annotations

from copy import deepcopy
import json
import re
from pathlib import Path
from typing import Any

from src.rl.s4_r3_h16_data_scaling import _split_seeds
from src.rl.s4_r3_multistep_critic_training import (
    _effective_settings as collector_settings, preflight_s4_r3_multistep_critic_training,
)
from src.rl.s4_training import _file_sha256, _load_yaml, _project_path, _relative
from src.runtime import resolve_device

ARMS = ("shared", "split", "regression_only", "classification_only")
SPLITS = ("training", "selection", "confirmation_id", "confirmation_shift")
BOUNDARY = dict(actor_updates=0, original_critic_updates=0, alpha_updates=0,
                student_updates=0, full_rl_trained=False, s4d3_accessed=False, real_slm_actions=False)
DESIGN_PATH = "configs/experiments/s4_r3_h16_fixed_time_confirmation_design_v1.yaml"
DESIGN_SHA = "d3a7003c2b98b7eab30b37ceca3780b0ddd31f1b6915994b6aef32603027e1b3"


def safe_path(value: str) -> Path:
    root = _project_path(".").resolve()
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("A9 paths must be repository relative")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("A9 path escapes repository")
    return resolved


def pin(value: str, digest: str, manifest: dict[str, str]) -> None:
    if _file_sha256(safe_path(value)) != digest:
        raise RuntimeError(f"A9 frozen hash mismatch: {value}")
    manifest[value] = digest


def load_contract(config_path: str | Path, *, quick: bool, quick_run_tag: str | None = None) -> tuple[dict, dict]:
    cfg = _load_yaml(_project_path(config_path))
    if (cfg.get("stage") != "S4-D2-R3-D2-A9" or cfg.get("design") != DESIGN_PATH
            or cfg.get("design_sha256") != DESIGN_SHA or cfg.get("runnable") is not True):
        raise RuntimeError("A9 requires the pinned runnable configuration, not a design-only file")
    if cfg.get("runtime") != dict(device="cuda", require_cuda=True, deterministic_algorithms=True):
        raise RuntimeError("A9 requires deterministic CUDA; CPU fallback is forbidden")
    if cfg.get("boundary") != BOUNDARY:
        raise RuntimeError("A9 update boundary changed")
    if set(cfg) != {"stage", "runnable", "design", "design_sha256", "runtime", "boundary", "frozen_inputs", "tracked_source_files"}:
        raise RuntimeError("A9 unexpected runtime override")
    pin(DESIGN_PATH, DESIGN_SHA, {})
    d = _load_yaml(safe_path(DESIGN_PATH))
    s = effective_settings(d, quick=quick)
    if quick_run_tag is not None:
        if not quick or re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", quick_run_tag) is None:
            raise ValueError("A9 quick run tag requires --quick and a safe lowercase label")
        # 只改变快速运行目录；不改冻结设计、种子、训练预算或正式输出目录。
        s["output_directory"] += "_" + quick_run_tag
        s["quick_run_tag"] = quick_run_tag
    return cfg, s


def effective_settings(d: dict, *, quick: bool) -> dict:
    """运行参数仅从冻结设计推导；测试可对返回副本构造小型确定性案例。"""
    t, data, q = d["training_design"], d["data_design"], d["quick_design"]
    source = _load_yaml(safe_path(d["upstream"]["collection_config"]))
    s = deepcopy(t)
    s.update(quick=quick, model=deepcopy(d["model"]), statistics=deepcopy(d["statistics_design"]),
             design=d, candidate_actions=deepcopy(source["candidate_actions"]))
    if quick:
        for k in ("policy_seeds", "replicate_indices", "maximum_updates", "fixed_checkpoint_updates",
                  "primary_early_update", "primary_late_update", "batch_size", "log_interval_updates", "selection_interval_updates"):
            s[k] = deepcopy(q[k])
        s["statistics"]["bootstrap_replicates"] = q["bootstrap_replicates"]
    s["arms"] = list(ARMS)
    s["output_directory"] = d["proposed_output_paths"]["quick_directory" if quick else "directory"]
    s["splits"] = {}
    for i, spec in enumerate(data["splits"]):
        conditions = deepcopy(source["data"][spec["condition_template"]]["conditions"])
        conditions.sort(key=lambda x: data["condition_family_order"].index(x["id"].split("_")[-1]))
        for j, condition in enumerate(conditions):
            original = deepcopy(condition)
            base = q["namespace_base"] if quick else data["namespace_base"]
            offset = q["split_seed_offsets"][i] if quick else spec["seed_offset"]
            co = q["condition_seed_offsets"][j] if quick else data["condition_seed_offsets"][j]
            condition["id"] = f"a9_{spec['id']}_{data['condition_family_order'][j]}"
            condition["base_seed"] = base + offset + co
            if {k: v for k, v in condition.items() if k not in ("id", "base_seed")} != {
                    k: v for k, v in original.items() if k not in ("id", "base_seed")}:
                raise RuntimeError("A9 physical template changed")
        s["splits"][spec["id"]] = dict(
            conditions=conditions, episodes_per_condition=q["episodes_per_condition"] if quick else spec["episodes_per_condition"],
            profile_ids=list(q["profile_ids"] if quick else data["profile_ids"]),
            probe_steps=list(q["probe_steps"] if quick else data["probe_steps"]),
            episode_length_steps=data["episode_length_steps"], episode_index_offset=0,
        )
    s["planned_fits"] = len(s["policy_seeds"]) * len(s["replicate_indices"]) * len(ARMS)
    return s


def budget(s: dict) -> dict[str, int]:
    f, m = s["planned_fits"], len(s["fixed_checkpoint_updates"])
    episodes = {k: len(_split_seeds(v)) for k, v in s["splits"].items()}
    return dict(planned_fits=f, maximum_updates_per_fit=s["maximum_updates"],
                fixed_checkpoints=f*m, selected_checkpoints=f, inventory_records=f*(m+1),
                endpoint_records=f*m*4, selected_records=f*4,
                gradient_records=f//2*m*sum(episodes[k] for k in ("training", "confirmation_id", "confirmation_shift")),
                comparison_records=len(s["policy_seeds"])*3,
                dataset_files=len(s["policy_seeds"])*4, unique_weather_episodes=sum(episodes.values()),
                collection_branches=len(s["policy_seeds"])*sum(len(x["conditions"])*len(x["profile_ids"])*len(x["probe_steps"])*2 for x in s["splits"].values()))


def namespace_check(formal: dict, quick: dict, protected_blocks: set[int]) -> list[dict]:
    seen: set[int] = set()
    report = []
    for mode, s in (("formal", formal), ("quick", quick)):
        for name, split in s["splits"].items():
            seeds = _split_seeds(split)
            if seen & seeds or any(v >= 4000000 or v < 0 or v//10000 in protected_blocks for v in seeds):
                raise RuntimeError(f"A9 seed namespace overlap/reserved: {mode}/{name}")
            seen |= seeds
            report.append(dict(mode=mode, split=name, episodes=len(seeds), minimum=min(seeds), maximum=max(seeds)))
    return report


def _seed_blocks(obj: Any) -> set[int]:
    """保守封闭旧种子所在的整个一万整数块，含偏移扩样和旧测试。"""
    result: set[int] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in ("base_seed", "episode_seed", "test_base_seeds", "episode_seeds"):
                values = value if isinstance(value, list) else [value]
                result.update(int(v)//10000 for v in values if isinstance(v, int) and not isinstance(v, bool))
            result |= _seed_blocks(value)
    elif isinstance(obj, list):
        for value in obj:
            result |= _seed_blocks(value)
    return result


def preflight(config_path: str | Path, cfg: dict, s: dict) -> tuple[dict, dict, list[dict]]:
    device = resolve_device(cfg["runtime"]["device"])
    if device.type != "cuda":
        raise RuntimeError("A9 CUDA required")
    output = safe_path(s["output_directory"])
    if output.parent != safe_path("outputs") or not output.name.startswith("s4_r3_h16_fixed_time_confirmation_v1"):
        raise RuntimeError("A9 invalid dedicated output")
    if output.exists():
        raise FileExistsError(f"A9 output already exists; preserve it: {output}")
    inputs: dict[str, str] = {}
    pin(DESIGN_PATH, DESIGN_SHA, inputs)
    for path, digest in cfg["frozen_inputs"].items():
        pin(path, digest, inputs)
    d = s["design"]
    summary = json.loads(safe_path(d["upstream"]["summary"]).read_text(encoding="utf-8"))
    pin(d["upstream"]["summary"], d["upstream"]["summary_sha256"], inputs)
    if summary["experiment"]["quick"] or summary["interpretation"]["status"] != d["upstream"]["required_status"]:
        raise RuntimeError("A9 upstream diagnostic status changed")
    # 固定物理代码与A8梯度代码；读取清单和哈希，不载入旧测试样本。
    for path in ("outputs/s4_r3_physical_target_confirmation_v1/source_manifest.json",
                 "outputs/s4_r3_h16_checkpoint_endpoints_v1/source_manifest.json"):
        for file, digest in json.loads(safe_path(path).read_text(encoding="utf-8")).items():
            pin(file, digest, inputs)
    source_path = safe_path(d["upstream"]["collection_config"])
    source = _load_yaml(source_path)
    link = source
    for field in ("upstream_d1c", "upstream_d1b", "upstream_d1a", "upstream_d1"):
        contract = link[field]
        pin(contract["experiment_config"], contract["experiment_config_sha256"], inputs)
        link = _load_yaml(safe_path(contract["experiment_config"]))
    old = collector_settings(source, quick=False)
    _, physical, policies = preflight_s4_r3_multistep_critic_training(source_path, source, old, quick=False)
    policies = [p for p in policies if p["policy_seed"] in s["policy_seeds"]]
    if sorted(p["policy_seed"] for p in policies) != sorted(s["policy_seeds"]):
        raise RuntimeError("A9 frozen policy coverage mismatch")
    for p in policies:
        pin(p["path"], p["sha256"], inputs)
        pin(p["student_path"], p["student_sha256"], inputs)
    pin(physical["environment_config"], physical["environment_config_sha256"], inputs)
    protected: set[int] = set()
    metadata_files = list(safe_path("configs/experiments").glob("*.yaml"))
    for name in ("data_manifest.json", "effective_config.json"):
        metadata_files += list(safe_path("outputs").glob(f"*/{name}"))
    for path in metadata_files:
        if "s4_r3_h16_fixed_time_confirmation" in str(path):
            continue
        content = _load_yaml(path) if path.suffix == ".yaml" else json.loads(path.read_text(encoding="utf-8"))
        protected |= _seed_blocks(content)
        inputs[_relative(path)] = _file_sha256(path)
    namespaces = namespace_check(effective_settings(d, quick=False), effective_settings(d, quick=True), protected)
    sources = {p: _file_sha256(safe_path(p)) for p in cfg["tracked_source_files"]}
    sources[_relative(_project_path(config_path))] = _file_sha256(_project_path(config_path))
    result = dict(status="READY_FOR_QUICK_SMOKE" if s["quick"] else "READY_FOR_USER_TRAINING",
                  quick=s["quick"], device=str(device), **budget(s), namespace=namespaces,
                  output_directory=s["output_directory"], frozen_input_hashes=inputs,
                  frozen_source_hashes=sources, old_sample_tensors_loaded=False,
                  new_data_generated=False, training_run=False, **BOUNDARY)
    return result, physical, policies


def verify_manifests(report: dict) -> None:
    for key in ("frozen_input_hashes", "frozen_source_hashes"):
        for path, digest in report[key].items():
            pin(path, digest, {})
