"""A9运行编排：采集开发数据→监督训练→冻结→新确认→统计。"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import time
import traceback

import torch

from src.rl.s4_r3_h16_fixed_time_contract import (
    ARMS, BOUNDARY, budget, load_contract, preflight, safe_path, verify_manifests,
)
from src.rl.s4_r3_h16_fixed_time_training import (
    FixedTimeProbe, fit_probe, require_frozen, seal_training, validate_data, write_json, write_rows,
)
from src.rl.s4_r3_h16_fixed_time_analysis import evaluate_all
from src.rl.s4_r3_h16_data_scaling import _collect_and_save, verify_dataset_separation
from src.rl.s4_r3_h16_head_split import _git_record_utf8
from src.rl.s4_representation_capacity import ActionRepresentation, build_action_basis
from src.rl.s4_training import _file_sha256, _project_path, _relative, _runtime_record
from src.runtime import resolve_device
from src.simulation.config import load_s1_config
from src.training_progress import counted_progress, progress_message


def run_s4_r3_h16_fixed_time_confirmation(config_path: str | Path, *, quick: bool = False,
                                         preflight_only: bool = False, quick_run_tag: str | None = None) -> dict:
    cfg, s = load_contract(config_path, quick=quick, quick_run_tag=quick_run_tag)
    report, physical, policies = preflight(config_path, cfg, s)
    if preflight_only:
        return report
    device = resolve_device(cfg["runtime"]["device"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    output = safe_path(s["output_directory"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "preflight.json", report)
    write_json(output / "effective_config.json", s)
    write_json(output / "input_manifest.json", report["frozen_input_hashes"])
    write_json(output / "source_manifest.json", report["frozen_source_hashes"])
    try:
        return _execute(s, report, physical, policies, output, device)
    except Exception as exc:
        write_json(output / "failure.json", dict(exception=type(exc).__name__, message=str(exc),
                   traceback=traceback.format_exc(), automatic_retry=False, **BOUNDARY))
        raise


def _execute(s: dict, report: dict, physical: dict, policies: list[dict], output: Path,
             device: torch.device) -> dict:
    started = time.perf_counter()
    dataset_directory = output / "datasets"
    dataset_directory.mkdir()
    base, _ = load_s1_config(_project_path(physical["environment_config"]))
    representation = ActionRepresentation.from_mapping(physical["representation"])
    base = replace(base, num_modes=representation.num_modes)
    basis, _, diagnostics = build_action_basis(base, representation, device)
    data = {p: {} for p in s["policy_seeds"]}
    records, fits = [], []
    collection_bar = counted_progress(total=budget(s)["collection_branches"], description="A9 新回合采集", unit="批量分支")
    collection_started = time.perf_counter()
    frozen = None

    def collect(names: tuple[str, ...]) -> None:
        for policy in policies:
            seed = policy["policy_seed"]
            for name in names:
                if name.startswith("confirmation"):
                    if frozen is None:
                        raise RuntimeError("A9 confirmation attempted before training freeze")
                    require_frozen(output, frozen, s)
                progress_message(f"A9 采集 {seed}/{name}")
                pairs, record = _collect_and_save(name=name, split=s["splits"][name], checkpoint=policy,
                    physical_experiment=physical, base_config=base, basis=basis, settings=s,
                    bar=collection_bar, progress_path=output / "collection_progress.jsonl",
                    collection_started=collection_started, dataset_directory=dataset_directory, device=device)
                validate_data(pairs, s["splits"][name], seed, collection_record=record)
                data[seed][name] = pairs
                verify_dataset_separation(data[seed])
                record["physical_split_config"] = s["splits"][name]
                records.append(record)
                write_json(output / "data_manifest.json", dict(records=records))

    progress_message("A9 阶段1/4：采集新的训练与模型选择回合")
    collect(("training", "selection"))
    progress_message("A9 阶段2/4：四组固定预算监督训练")
    for policy in s["policy_seeds"]:
        for replicate in s["replicate_indices"]:
            seed = s["initialization_seed_offset"] + policy + s["replicate_seed_stride"]*replicate
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            init_model = FixedTimeProbe(221, s["model"]["hidden_sizes"][0], "shared").to(device)
            initial = {k: v.detach().clone() for k, v in init_model.state_dict().items()}
            del init_model
            for arm in ARMS:
                progress_message(f"A9 模型 {len(fits)+1}/{s['planned_fits']}：策略{policy}/初始化{replicate}/{arm}")
                fits.append(fit_probe(training=data[policy]["training"], selection=data[policy]["selection"],
                    s=s, policy_seed=policy, replicate=replicate, task=arm, initial=initial,
                    directory=output / f"policy_{policy}" / f"replicate_{replicate}" / arm, device=device))
                write_json(output / "training_fits.json", dict(fits=fits, confirmation_opened=False))
    verify_manifests(report)
    frozen = seal_training(fits, s, output)
    inventory = require_frozen(output, frozen, s)
    write_rows(output / "checkpoint_inventory.csv", inventory)
    write_json(output / "CONFIRMATION_OPENED.json", dict(training_frozen=frozen,
               created_at=datetime.now(timezone.utc).isoformat(), all_fits_completed=True,
               confirmation_splits=["confirmation_id", "confirmation_shift"], **BOUNDARY))
    progress_message("A9 阶段3/4：全部模型已冻结，采集两套独立确认回合")
    collect(("confirmation_id", "confirmation_shift"))
    collection_bar.close()
    if len(records) != budget(s)["dataset_files"] or collection_bar.n != collection_bar.total:
        raise RuntimeError("A9 incomplete collection")
    progress_message("A9 阶段4/4：固定模型评估、梯度记录和九项配对比较")
    evaluation = evaluate_all(inventory, data, s, output, device)
    require_frozen(output, frozen, s)
    verify_manifests(report)
    for record in records:
        if _file_sha256(_project_path(record["path"])) != record["sha256"]:
            raise RuntimeError("A9 collected dataset changed")
    file_hashes = {_relative(p): _file_sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    result = dict(material_passport=dict(origin_skill="academic-research-suite / experiment-agent",
                  origin_mode="run", origin_date=datetime.now(timezone.utc).isoformat(),
                  verification_status="UNVERIFIED", version_label="s4d2_r3_d2a9_fixed_time_v1"),
                  experiment=dict(id="AO-S4-D2-R3-D2-A9-FIXED-TIME-CONFIRMATION", quick=s["quick"],
                  status="completed_pending_audit", device=str(device), gpu_name=torch.cuda.get_device_name(device),
                  duration_seconds=time.perf_counter()-started),
                  budget=budget(s), fits=fits, training_frozen=frozen, basis_diagnostics=diagnostics,
                  evaluation={k: v for k, v in evaluation.items() if k != "interpretation"},
                  interpretation=evaluation["interpretation"],
                  evidence_boundary=dict(training_run=True, supervised_probe_updates=sum(f["updates_completed"] for f in fits),
                  new_data_generated=True, independent_confirmation_opened=True, old_test_samples_loaded=False, **BOUNDARY),
                  runtime=_runtime_record(), git=_git_record_utf8(), records=file_hashes,
                  next_action="训练完成后停止，等待用户通知与只读审计；不要自动重训或进入完整RL。")
    write_json(output / "summary.json", result)
    write_json(output / "SUCCESS.json", dict(status="COMPLETED_PENDING_AUDIT", quick=s["quick"],
               summary_sha256=_file_sha256(output / "summary.json"), **BOUNDARY))
    return result
