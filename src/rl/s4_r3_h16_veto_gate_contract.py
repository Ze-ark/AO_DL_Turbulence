"""A10冻结契约和只读预检；导入时不写文件或占用GPU。"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

from src.rl.s4_r3_h16_fixed_time_contract import _seed_blocks
from src.rl.s4_r3_multistep_critic_training import (
    _effective_settings as collector_settings,
    preflight_s4_r3_multistep_critic_training,
)
from src.rl.s4_training import _file_sha256, _load_yaml, _project_path, _relative
from src.runtime import resolve_device

DESIGN_PATH = "configs/experiments/s4_r3_h16_veto_gate_confirmation_design_v1.yaml"
DESIGN_SHA = "5fc137c164206563a704271330149a9733fdb512ead79f6c068aa954e8121a77"
SPLITS = ("confirmation_id", "confirmation_shift")
BOUNDARY = dict(supervised_updates=0, actor_updates=0, critic_updates=0, alpha_updates=0,
                student_updates=0, full_rl_trained=False, closed_loop_rollout=False,
                s4d3_accessed=False, real_slm_actions=False)


def safe_path(value: str | Path) -> Path:
    root = _project_path(".").resolve()
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("A10 path must be repository relative")
    result = (root / path).resolve()
    if not result.is_relative_to(root):
        raise ValueError("A10 path escapes repository")
    return result


def pin(path: str | Path, digest: str, manifest: dict[str, str]) -> None:
    p = safe_path(path)
    if not p.is_file() or _file_sha256(p) != digest:
        raise RuntimeError(f"A10 frozen hash mismatch: {path}")
    manifest[_relative(p)] = digest


def load_contract(config_path: str | Path, *, quick: bool,
                  quick_run_tag: str | None = None) -> tuple[dict, dict]:
    cfg = _load_yaml(_project_path(config_path))
    if (cfg.get("stage") != "S4-D2-R3-D2-A10" or cfg.get("runnable") is not True
            or cfg.get("design") != DESIGN_PATH or cfg.get("design_sha256") != DESIGN_SHA):
        raise RuntimeError("A10 requires the pinned runnable configuration")
    if cfg.get("runtime") != dict(device="cuda", require_cuda=True, deterministic_algorithms=True):
        raise RuntimeError("A10 requires deterministic CUDA without CPU fallback")
    if cfg.get("boundary") != BOUNDARY:
        raise RuntimeError("A10 update boundary changed")
    allowed = {"stage", "runnable", "design", "design_sha256", "runtime", "boundary",
               "frozen_inputs", "tracked_source_files"}
    if set(cfg) != allowed:
        raise RuntimeError("A10 unexpected runtime override")
    pin(DESIGN_PATH, DESIGN_SHA, {})
    design = _load_yaml(safe_path(DESIGN_PATH))
    settings = effective_settings(design, quick=quick)
    if quick_run_tag is not None:
        if not quick or re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", quick_run_tag) is None:
            raise ValueError("A10 quick run tag requires --quick and a safe lowercase label")
        # 只改变快速运行目录，不改变冻结设计、种子、模型、门槛或正式输出目录。
        settings["output_directory"] += "_" + quick_run_tag
        settings["quick_run_tag"] = quick_run_tag
    return cfg, settings


def effective_settings(design: dict, *, quick: bool) -> dict:
    data, stats = design["data"], deepcopy(design["statistics"])
    upstream_summary = json.loads(safe_path(design["upstream"]["summary"]).read_text(encoding="utf-8"))
    effective_path = safe_path("outputs/s4_r3_h16_fixed_time_confirmation_v1/effective_config.json")
    if upstream_summary["records"].get(_relative(effective_path)) != _file_sha256(effective_path):
        raise RuntimeError("A10 A9 effective configuration is not frozen")
    a9 = json.loads(effective_path.read_text(encoding="utf-8"))
    q = design["quick_design"]
    policy_seeds = list(q["policy_seeds"] if quick else design["upstream"]["policy_seeds"])
    s: dict[str, Any] = dict(
        quick=quick, design=design, policy_seeds=policy_seeds,
        replicate_indices=list(design["upstream"]["replicate_indices"]),
        profile_ids=list(q["profile_ids"] if quick else data["profile_ids"]),
        probe_steps=list(q["probe_steps"] if quick else data["probe_steps"]),
        target_horizon=data["target_horizon"], episode_length_steps=data["episode_length_steps"],
        candidate_actions=deepcopy(a9["candidate_actions"]), gate=deepcopy(design["gate"]),
        statistics=stats,
        output_directory=design["proposed_output"]["quick_directory" if quick else "directory"],
    )
    if quick:
        s["statistics"]["bootstrap_replicates"] = q["bootstrap_replicates"]
    s["splits"] = {}
    templates = {"development": a9["splits"]["training"],
                 "validation": a9["splits"]["confirmation_shift"]}
    for i, spec in enumerate(data["splits"]):
        template = templates[spec["condition_template"]]
        conditions = deepcopy(template["conditions"])
        for j, condition in enumerate(conditions):
            original = deepcopy(condition)
            base = q["namespace_base"] if quick else data["namespace_base"]
            split_offset = q["split_seed_offsets"][i] if quick else spec["seed_offset"]
            offsets = q["condition_seed_offsets"] if quick else data["condition_seed_offsets"]
            family = str(condition["id"]).split("_")[-1]
            condition["id"] = f"a10_{spec['id']}_{family}"
            condition["base_seed"] = base + split_offset + offsets[j]
            left = {k: v for k, v in condition.items() if k not in ("id", "base_seed")}
            right = {k: v for k, v in original.items() if k not in ("id", "base_seed")}
            if left != right:
                raise RuntimeError("A10 physical template changed")
        s["splits"][spec["id"]] = dict(
            conditions=conditions,
            episodes_per_condition=q["episodes_per_condition"] if quick else spec["episodes_per_condition"],
            profile_ids=list(s["profile_ids"]), probe_steps=list(s["probe_steps"]),
            episode_length_steps=s["episode_length_steps"], episode_index_offset=0,
        )
    return s


def split_seeds(split: dict) -> set[int]:
    return {int(c["base_seed"]) + i for c in split["conditions"]
            for i in range(int(split["episodes_per_condition"]))}


def budget(s: dict) -> dict[str, int]:
    episodes = {name: len(split_seeds(split)) for name, split in s["splits"].items()}
    pairs_per_episode = len(s["profile_ids"]) * len(s["probe_steps"])
    return dict(dataset_files=len(s["policy_seeds"]) * len(SPLITS),
                unique_weather_episodes=sum(episodes.values()),
                policy_episode_instances=len(s["policy_seeds"]) * sum(episodes.values()),
                paired_action_samples=len(s["policy_seeds"]) * sum(episodes.values()) * pairs_per_episode,
                collection_branches=len(s["policy_seeds"]) * len(SPLITS) * 3 * pairs_per_episode * 2,
                frozen_models=len(s["policy_seeds"]) * len(s["replicate_indices"]),
                episode_decision_records=len(s["policy_seeds"]) * sum(episodes.values()),
                primary_comparison_records=len(s["policy_seeds"]) * 3,
                shifted_comparison_records=len(s["policy_seeds"]) * 3)


def selected_models(summary: dict, s: dict, manifest: dict[str, str]) -> dict[int, list[dict]]:
    result: dict[int, list[dict]] = {}
    expected = {(p, r) for p in s["policy_seeds"] for r in s["replicate_indices"]}
    found: dict[tuple[int, int], dict] = {}
    for fit in summary["fits"]:
        if fit["arm"] != "regression_only" or fit["policy_seed"] not in s["policy_seeds"]:
            continue
        selected = [x for x in fit["inventory"] if x["role"] == "selected"]
        if len(selected) != 1:
            raise RuntimeError("A10 selected checkpoint coverage changed")
        rec = selected[0]
        key = int(fit["policy_seed"]), int(fit["replicate"])
        if key in found or rec["arm"] != "regression_only" or rec["update"] != fit["selected_update"]:
            raise RuntimeError("A10 selected checkpoint identity changed")
        pin(rec["path"], rec["sha256"], manifest)
        payload = __import__("torch").load(safe_path(rec["path"]), map_location="cpu", weights_only=False)
        if payload.get("independent_confirmation_used_for_selection") is not False:
            raise RuntimeError("A10 checkpoint selection accessed confirmation data")
        found[key] = deepcopy(rec)
    if set(found) != expected:
        raise RuntimeError("A10 frozen selected model set is incomplete")
    for policy in s["policy_seeds"]:
        result[policy] = [found[policy, r] for r in s["replicate_indices"]]
    return result


def preflight(config_path: str | Path, cfg: dict, s: dict) -> tuple[dict, dict, list[dict], dict[int, list[dict]]]:
    device = resolve_device(cfg["runtime"]["device"])
    if device.type != "cuda":
        raise RuntimeError("A10 CUDA required")
    output = safe_path(s["output_directory"])
    if output.parent != safe_path("outputs") or not output.name.startswith("s4_r3_h16_veto_gate_confirmation_v1"):
        raise RuntimeError("A10 invalid dedicated output")
    if output.exists():
        raise FileExistsError(f"A10 output already exists; preserve it: {output}")
    inputs: dict[str, str] = {}
    pin(DESIGN_PATH, DESIGN_SHA, inputs)
    for path, digest in cfg["frozen_inputs"].items():
        pin(path, digest, inputs)
    design = s["design"]
    pin(design["upstream"]["summary"], design["upstream"]["summary_sha256"], inputs)
    summary = json.loads(safe_path(design["upstream"]["summary"]).read_text(encoding="utf-8"))
    if summary["interpretation"]["status"] != design["upstream"]["required_status"] or summary["experiment"]["quick"]:
        raise RuntimeError("A10 upstream A9 status changed")
    success = json.loads(safe_path(design["upstream"]["success"]).read_text(encoding="utf-8"))
    if success["summary_sha256"] != design["upstream"]["summary_sha256"]:
        raise RuntimeError("A10 A9 success marker mismatch")
    pin(design["upstream"]["training_freeze"], summary["training_frozen"]["sha256"], inputs)
    for name in ("input_manifest.json", "source_manifest.json"):
        path = f"outputs/s4_r3_h16_fixed_time_confirmation_v1/{name}"
        pin(path, summary["records"][path], inputs)
        for frozen_path, digest in json.loads(safe_path(path).read_text(encoding="utf-8")).items():
            pin(frozen_path, digest, inputs)
    models = selected_models(summary, s, inputs)
    a9 = json.loads(safe_path("outputs/s4_r3_h16_fixed_time_confirmation_v1/effective_config.json").read_text(encoding="utf-8"))
    source_path = safe_path(a9["design"]["upstream"]["collection_config"])
    source = _load_yaml(source_path)
    old = collector_settings(source, quick=False)
    _, physical, policies = preflight_s4_r3_multistep_critic_training(source_path, source, old, quick=False)
    policies = [p for p in policies if p["policy_seed"] in s["policy_seeds"]]
    if sorted(p["policy_seed"] for p in policies) != sorted(s["policy_seeds"]):
        raise RuntimeError("A10 frozen actor coverage mismatch")
    for policy in policies:
        pin(policy["path"], policy["sha256"], inputs)
        pin(policy["student_path"], policy["student_sha256"], inputs)
    pin(physical["environment_config"], physical["environment_config_sha256"], inputs)
    protected: set[int] = set()
    files = list(safe_path("configs/experiments").glob("*.yaml"))
    for name in ("effective_config.json", "data_manifest.json"):
        files.extend(safe_path("outputs").glob(f"*/{name}"))
    for path in files:
        if "s4_r3_h16_veto_gate_confirmation" in str(path):
            continue
        obj = _load_yaml(path) if path.suffix == ".yaml" else json.loads(path.read_text(encoding="utf-8"))
        protected |= _seed_blocks(obj)
        inputs[_relative(path)] = _file_sha256(path)
    formal = effective_settings(design, quick=False)
    quick = effective_settings(design, quick=True)
    all_seen: set[int] = set()
    namespace = []
    for mode, settings in (("formal", formal), ("quick", quick)):
        for name, split in settings["splits"].items():
            seeds = split_seeds(split)
            if seeds & all_seen or any(x >= design["data"]["forbid_seed_at_or_above"] or x // 10000 in protected for x in seeds):
                raise RuntimeError(f"A10 seed namespace conflict: {mode}/{name}")
            all_seen |= seeds
            namespace.append(dict(mode=mode, split=name, episodes=len(seeds), minimum=min(seeds), maximum=max(seeds)))
    sources = {path: _file_sha256(safe_path(path)) for path in cfg["tracked_source_files"]}
    sources[_relative(_project_path(config_path))] = _file_sha256(_project_path(config_path))
    report = dict(status="READY_FOR_QUICK_SMOKE" if s["quick"] else "READY_FOR_USER_CONFIRMATION",
                  quick=s["quick"], device=str(device), output_directory=s["output_directory"],
                  namespace=namespace, frozen_input_hashes=inputs, frozen_source_hashes=sources,
                  a9_samples_loaded=False, new_data_generated=False, **budget(s), **BOUNDARY)
    return report, physical, policies, models


def verify_manifests(report: dict) -> None:
    for field in ("frozen_input_hashes", "frozen_source_hashes"):
        for path, digest in report[field].items():
            pin(path, digest, {})
