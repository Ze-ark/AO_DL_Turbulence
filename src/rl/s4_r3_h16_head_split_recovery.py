"""A8末端元数据故障恢复：只核验既有产物并补写汇总，不重训或重新仿真。"""

from __future__ import annotations

from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import torch

from src.rl.s4_r3_h16_data_scaling import (
    _episode_seeds,
    _load_pairs,
    _split_seeds,
)
from src.rl.s4_r3_h16_head_split import (
    ARMS,
    NO_UPDATES,
    _git_record_utf8,
    compare_heads,
    evaluate_outputs,
    interpret_heads,
    subgroup_outputs,
    validate_coverage,
)
from src.rl.s4_training import (
    _file_sha256,
    _load_yaml,
    _project_path,
    _relative,
    _runtime_record,
    _write_json,
    json_safe,
)


RECOVERABLE_SOURCE_DRIFT = {"src/rl/s4_r3_h16_head_split.py"}
APPROVED_A8_RECOVERY_SOURCE_SHA256 = "fe2aacb85f2f7ec5d594e0706b3677b00385926ff7c7d3088ad38cca295e7fd8"
REQUIRED_OUTPUT_FILES = {
    "data_manifest.json",
    "effective_config.json",
    "failure.json",
    "fit_summary.csv",
    "head_comparison.csv",
    "input_manifest.json",
    "NEW_TEST_OPENED.json",
    "preflight.json",
    "source_manifest.json",
    "test_collection_progress.jsonl",
    "test_subgroups.csv",
    "training_fits.json",
    "TRAINING_FROZEN.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_terminal_git_failure(payload: dict[str, Any]) -> None:
    """只接受已知的Git元数据解码故障，其他训练故障一律拒绝恢复。"""
    traceback = str(payload.get("traceback", ""))
    if (
        payload.get("exception") != "AttributeError"
        or payload.get("message") != "'NoneType' object has no attribute 'strip'"
        or payload.get("automatic_retry") is not False
        or "_git_record" not in traceback
        or "s4_training.py" not in traceback
        or "s4_r3_h16_head_split.py" not in traceback
    ):
        raise RuntimeError("A8 failure is not the approved terminal Git-metadata failure")


def _verify_manifest(
    manifest: dict[str, str], *, allowed_drift: set[str]
) -> list[dict[str, str]]:
    drift: list[dict[str, str]] = []
    for name, expected in manifest.items():
        path = _project_path(name)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _file_sha256(path)
        if actual != expected:
            normalized = str(name).replace("\\", "/")
            if normalized not in allowed_drift:
                raise RuntimeError(f"A8 recovery found unexpected frozen-file drift: {normalized}")
            drift.append({"path": normalized, "frozen_sha256": expected, "current_sha256": actual})
    return drift


def _verify_record(record: dict[str, Any], *, label: str) -> None:
    path = _project_path(record["path"])
    if not path.is_file() or _file_sha256(path) != record["sha256"]:
        raise RuntimeError(f"A8 recovery hash mismatch: {label}")


def _read_progress(path: Path, expected: int) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected or [int(row["completed_branches"]) for row in rows] != list(range(1, expected + 1)):
        raise RuntimeError("A8 recovery found incomplete or reordered test progress")
    if any(int(row["total_branches"]) != expected for row in rows):
        raise RuntimeError("A8 recovery test-progress total changed")
    return rows[-1]


def _prediction_payload(path: Path, dataset: Any) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"reward_prediction", "ranking_logit", "reward_delta", "power_delta", "rows"}
    if set(payload) != required:
        raise RuntimeError(f"A8 recovery prediction schema changed: {path}")
    if (
        payload["rows"] != dataset.rows
        or not torch.equal(payload["reward_delta"], dataset.reward_delta)
        or not torch.equal(payload["power_delta"], dataset.power_delta)
        or payload["reward_prediction"].shape != dataset.reward_delta.shape
        or payload["ranking_logit"].shape != dataset.reward_delta.shape
        or not bool(torch.isfinite(payload["reward_prediction"]).all())
        or not bool(torch.isfinite(payload["ranking_logit"]).all())
    ):
        raise RuntimeError(f"A8 recovery prediction/data alignment failed: {path}")
    return payload


def _comparison_rows(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "policy_seed": comparison["policy_seed"],
            **{
                f"{metric}_{key}": value
                for metric in ("relative_mae_reduction", "ranking_ba_delta")
                for key, value in comparison[metric].items()
            },
        }
        for comparison in comparisons
    ]


def _fit_rows(fits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "policy_seed": fit["policy_seed"],
            "arm": fit["arm"],
            "best_update": fit["best_update"],
            "parameter_count": fit["parameter_count"],
            **{
                f"{name}_{key}": value
                for name in ("training", "validation", "independent_test")
                for key, value in {
                    "ranking_ba": fit[name]["ranking"]["balanced_accuracy"],
                    "value_mae": fit[name]["value"]["mae"],
                    "value_mae_better_than_constant": fit[name]["value"]["mae_better_than_constant"],
                    "head_disagreement": fit[name]["head_sign_disagreement_fraction"],
                }.items()
            },
        }
        for fit in fits
    ]


def _csv_value(value: Any) -> str:
    return "" if value is None else str(value)


def assert_csv_matches(path: Path, expected: list[dict[str, Any]]) -> None:
    if not expected:
        raise ValueError("A8 recovery cannot validate an empty CSV expectation")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        actual = list(reader)
        fields = list(reader.fieldnames or [])
    if fields != list(expected[0]) or len(actual) != len(expected):
        raise RuntimeError(f"A8 recovery CSV shape changed: {path.name}")
    serialized = [{key: _csv_value(value) for key, value in row.items()} for row in expected]
    if actual != serialized:
        raise RuntimeError(f"A8 recovery CSV values do not reproduce: {path.name}")


def _validate_fits(
    frozen: dict[str, Any], training_fits: dict[str, Any], settings: dict[str, Any]
) -> list[dict[str, Any]]:
    fits = deepcopy(frozen.get("fits", []))
    expected = {(int(seed), arm) for seed in settings["policy_seeds"] for arm in ARMS}
    if len(fits) != len(expected) or {(int(fit["policy_seed"]), fit["arm"]) for fit in fits} != expected:
        raise RuntimeError("A8 recovery found incomplete frozen fits")
    if training_fits.get("fits") != frozen.get("fits") or training_fits.get("new_test_opened") is not False:
        raise RuntimeError("A8 recovery training_fits and TRAINING_FROZEN differ")
    for fit in fits:
        if int(fit["updates_completed"]) != int(settings["maximum_updates"]):
            raise RuntimeError("A8 recovery found an incomplete training budget")
        if fit["arm"] == "shared" and fit.get("control_parity", {}).get("matched") is not True:
            raise RuntimeError("A8 recovery found missing A7 control parity")
        for label, path_key, hash_key in (
            ("best checkpoint", "checkpoint", "checkpoint_sha256"),
            ("last checkpoint", "last_checkpoint", "last_checkpoint_sha256"),
            ("progress log", "progress_log", "progress_sha256"),
            ("loss history", "loss_history", "loss_history_sha256"),
        ):
            _verify_record({"path": fit[path_key], "sha256": fit[hash_key]}, label=label)
        for split in ("training", "validation"):
            _verify_record(fit["predictions"][split], label=f"{split} predictions")
    return fits


def _reconstruct_saved_results(
    output: Path, fits: list[dict[str, Any]], settings: dict[str, Any],
    data_manifest: dict[str, Any], test_marker: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    records = data_manifest.get("generated_test", [])
    by_seed = {int(record["policy_seed"]): record for record in records}
    if set(by_seed) != {int(seed) for seed in settings["policy_seeds"]} or len(records) != len(by_seed):
        raise RuntimeError("A8 recovery found incomplete generated-test records")
    expected_test_seeds = _split_seeds(test_marker["settings"])
    datasets: dict[int, Any] = {}
    for seed, record in by_seed.items():
        _verify_record(record, label=f"seed {seed} raw test")
        dataset = _load_pairs({"path": record["path"]})
        validate_coverage(dataset, episodes=96, pairs=1728)
        if _episode_seeds(dataset) != expected_test_seeds:
            raise RuntimeError(f"A8 recovery test seed namespace changed: {seed}")
        datasets[seed] = dataset

    rebuilt: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    outputs: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
    saved_prediction_hashes: dict[str, str] = {}
    for seed in settings["policy_seeds"]:
        dataset = datasets[int(seed)]
        for arm in ARMS:
            fit = deepcopy(next(item for item in fits if int(item["policy_seed"]) == int(seed) and item["arm"] == arm))
            path = output / f"seed_{seed}" / arm / "predictions_independent_test.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = _prediction_payload(path, dataset)
            checkpoint = torch.load(_project_path(fit["checkpoint"]), map_location="cpu", weights_only=False)
            training_mean = float(checkpoint["training_reward_mean"])
            value = payload["reward_prediction"]
            score = payload["ranking_logit"]
            fit["independent_test"] = evaluate_outputs(value, score, dataset, training_mean, settings, int(seed))
            fit["test_subgroups"] = subgroup_outputs(
                value, score, dataset, training_mean, policy_seed=int(seed), arm=arm
            )
            prediction_record = {"path": _relative(path), "sha256": _file_sha256(path)}
            fit["predictions"]["independent_test"] = prediction_record
            saved_prediction_hashes[prediction_record["path"]] = prediction_record["sha256"]
            groups.extend(fit["test_subgroups"])
            outputs[int(seed), arm] = (value, score)
            rebuilt.append(fit)

    comparisons = [
        compare_heads(outputs[int(seed), "shared"], outputs[int(seed), "split"], datasets[int(seed)], settings, int(seed))
        for seed in settings["policy_seeds"]
    ]
    return rebuilt, comparisons, groups, saved_prediction_hashes


def _prepare_recovery(config_path: str | Path) -> dict[str, Any]:
    config_path = _project_path(config_path)
    experiment = _load_yaml(config_path)
    if experiment["metadata"]["stage"] != "S4-D2-R3-D2-A8":
        raise RuntimeError("A8 recovery received the wrong experiment config")
    output = _project_path(experiment["outputs"]["directory"])
    if not output.is_dir():
        raise FileNotFoundError(output)
    summary_path = output / "summary.json"
    success_path = output / "RECOVERED_SUCCESS.json"
    if summary_path.exists():
        if success_path.is_file() and _read_json(success_path).get("summary_sha256") == _file_sha256(summary_path):
            return {"already_finalized": True, "output": output, "summary": _read_json(summary_path)}
        raise FileExistsError("A8 summary exists without a valid recovery-success marker; preserve it for audit")
    missing = sorted(name for name in REQUIRED_OUTPUT_FILES if not (output / name).is_file())
    if missing:
        raise RuntimeError(f"A8 recovery missing required outputs: {missing}")
    if (output / "RECOVERY_MANIFEST.json").exists():
        raise FileExistsError("A8 partial recovery manifest already exists; do not retry automatically")

    failure = _read_json(output / "failure.json")
    validate_terminal_git_failure(failure)
    effective = _read_json(output / "effective_config.json")
    if effective.get("experiment") != experiment:
        raise RuntimeError("A8 recovery config differs from the frozen effective config")
    settings = effective["settings"]
    if settings.get("quick") is not False or _project_path(settings["output_directory"]).resolve() != output.resolve():
        raise RuntimeError("A8 recovery only accepts the formal output directory")

    source_manifest = _read_json(output / "source_manifest.json")
    input_manifest = _read_json(output / "input_manifest.json")
    source_drift = _verify_manifest(source_manifest, allowed_drift=RECOVERABLE_SOURCE_DRIFT)
    if source_drift != [{
        "path": "src/rl/s4_r3_h16_head_split.py",
        "frozen_sha256": source_manifest["src/rl/s4_r3_h16_head_split.py"],
        "current_sha256": APPROVED_A8_RECOVERY_SOURCE_SHA256,
    }]:
        raise RuntimeError("A8 recovery source drift is not the approved UTF-8 metadata patch")
    _verify_manifest(input_manifest, allowed_drift=set())

    frozen_path = output / "TRAINING_FROZEN.json"
    frozen = _read_json(frozen_path)
    test_marker = _read_json(output / "NEW_TEST_OPENED.json")
    frozen_hash = _file_sha256(frozen_path)
    if test_marker.get("quick") is not False or test_marker.get("training_frozen_sha256") != frozen_hash:
        raise RuntimeError("A8 recovery training freeze marker does not align")
    if test_marker.get("settings") != settings["test_split"]:
        raise RuntimeError("A8 recovery test settings differ from the frozen settings")
    fits = _validate_fits(frozen, _read_json(output / "training_fits.json"), settings)

    preflight = _read_json(output / "preflight.json")
    expected_branches = int(preflight["expected_test_branches"])
    if expected_branches != 324:
        raise RuntimeError("A8 recovery expected-test coverage changed")
    progress = _read_progress(output / "test_collection_progress.jsonl", expected_branches)
    data_manifest = _read_json(output / "data_manifest.json")
    if data_manifest.get("reused") != preflight.get("data_specs"):
        raise RuntimeError("A8 recovery reused-data manifest changed")
    rebuilt, comparisons, groups, prediction_hashes = _reconstruct_saved_results(
        output, fits, settings, data_manifest, test_marker
    )
    assert_csv_matches(output / "fit_summary.csv", _fit_rows(rebuilt))
    assert_csv_matches(output / "head_comparison.csv", _comparison_rows(comparisons))
    assert_csv_matches(output / "test_subgroups.csv", groups)

    upstream_summary = _read_json(_project_path(experiment["upstream_a7"]["summary"]))
    basis = upstream_summary["basis_diagnostics"]
    recorded_work_duration = sum(float(fit["duration_seconds"]) for fit in fits) + float(progress["elapsed_seconds"])
    return {
        "already_finalized": False,
        "config_path": config_path,
        "experiment": experiment,
        "settings": settings,
        "output": output,
        "failure": failure,
        "source_drift": source_drift,
        "fits": rebuilt,
        "comparisons": comparisons,
        "groups": groups,
        "prediction_hashes": prediction_hashes,
        "basis_diagnostics": basis,
        "frozen_hash": frozen_hash,
        "expected_branches": expected_branches,
        "recorded_work_duration": recorded_work_duration,
    }


def _atomic_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    _write_json(temporary, payload)
    temporary.replace(path)


def _recovery_code_hashes() -> dict[str, str]:
    paths = [
        "src/rl/s4_r3_h16_head_split.py",
        "src/rl/s4_r3_h16_head_split_recovery.py",
        "scripts/finalize_s4_r3_h16_head_split.py",
        "tests/test_s4_r3_h16_head_split_recovery.py",
    ]
    return {name: _file_sha256(_project_path(name)) for name in paths}


def _recovery_manifest(state: dict[str, Any]) -> dict[str, Any]:
    output = state["output"]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "operation": "A8_METADATA_ONLY_FINALIZATION",
        "original_failure": {
            "path": _relative(output / "failure.json"),
            "sha256": _file_sha256(output / "failure.json"),
            "exception": state["failure"]["exception"],
            "message": state["failure"]["message"],
        },
        "validated": {
            "fits": len(state["fits"]),
            "updates_per_fit": state["settings"]["maximum_updates"],
            "test_branches": state["expected_branches"],
            "saved_test_predictions": len(state["prediction_hashes"]),
            "existing_csvs_exactly_reproduced": True,
            "training_frozen_sha256": state["frozen_hash"],
        },
        "source_drift": state["source_drift"],
        "saved_prediction_hashes": state["prediction_hashes"],
        "recovery_code_hashes": _recovery_code_hashes(),
        "scientific_boundary": {
            "training_run": False,
            "optimizer_updates": 0,
            "model_inference_run": False,
            "simulation_run": False,
            "new_test_generated": False,
            "existing_failure_preserved": True,
        },
    }


def _summary(state: dict[str, Any]) -> dict[str, Any]:
    output = state["output"]
    experiment = state["experiment"]
    settings = state["settings"]
    records = {
        _relative(path): _file_sha256(path)
        for path in output.rglob("*")
        if path.is_file() and path.name not in {"summary.json", "RECOVERED_SUCCESS.json"}
    }
    return {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": experiment["metadata"]["version_label"],
        },
        "experiment": {
            "id": experiment["metadata"]["experiment_id"],
            "quick": False,
            "status": "completed_pending_audit",
            "completion_mode": "metadata_only_recovery",
            "duration_seconds": state["recorded_work_duration"],
            "duration_basis": "sum_of_recorded_fit_and_test_collection_durations",
            "device": settings.get("device", experiment["runtime"]["device"]),
        },
        "fits": state["fits"],
        "comparisons": state["comparisons"],
        "interpretation": interpret_heads(state["fits"], state["comparisons"], settings),
        "basis_diagnostics": state["basis_diagnostics"],
        "training_frozen_sha256": state["frozen_hash"],
        "records": records,
        "recovery": {
            "operation": "A8_METADATA_ONLY_FINALIZATION",
            "original_failure_preserved": True,
            "reconstructed_from_saved_predictions": True,
            "existing_csvs_exactly_reproduced": True,
            "source_drift": state["source_drift"],
            "original_wall_duration_unavailable": True,
        },
        "evidence_boundary": {
            "supervised_only": True,
            "ranking_score_is_reward": False,
            "old_test_used_for_training_or_selection": False,
            "new_training_data_generated": False,
            "full_rl_trained": False,
            "s4d3_accessed": False,
            "real_slm_actions": False,
            **NO_UPDATES,
        },
        "runtime": _runtime_record(),
        "git": _git_record_utf8(),
        "next_action": "停止并进行只读审计；不得自动重训。",
    }


def finalize_s4_r3_h16_head_split(
    config_path: str | Path, *, acknowledge_finalize: bool = False
) -> dict[str, Any]:
    """默认只读预检；显式确认后仅补写恢复清单、summary和成功标记。"""
    state = _prepare_recovery(config_path)
    if state["already_finalized"]:
        return {
            "status": "ALREADY_FINALIZED",
            "output_directory": _relative(state["output"]),
            "summary_sha256": _file_sha256(state["output"] / "summary.json"),
            "interpretation": state["summary"]["interpretation"],
        }
    ready = {
        "status": "READY_FOR_METADATA_ONLY_FINALIZATION",
        "output_directory": _relative(state["output"]),
        "fits": len(state["fits"]),
        "updates_per_fit": state["settings"]["maximum_updates"],
        "test_branches": state["expected_branches"],
        "saved_test_predictions": len(state["prediction_hashes"]),
        "existing_csvs_exactly_reproduced": True,
        "training_or_simulation_will_run": False,
        "source_drift": state["source_drift"],
    }
    if not acknowledge_finalize:
        return ready

    output = state["output"]
    manifest_path = output / "RECOVERY_MANIFEST.json"
    summary_path = output / "summary.json"
    success_path = output / "RECOVERED_SUCCESS.json"
    _atomic_json(manifest_path, _recovery_manifest(state))
    summary = _summary(state)
    _atomic_json(summary_path, summary)
    _atomic_json(
        success_path,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "RECOVERED_METADATA_ONLY",
            "summary_sha256": _file_sha256(summary_path),
            "recovery_manifest_sha256": _file_sha256(manifest_path),
            "failure_preserved_sha256": _file_sha256(output / "failure.json"),
            "automatic_retraining": False,
        },
    )
    return json_safe(
        {
            **ready,
            "status": "RECOVERED_METADATA_ONLY",
            "summary_sha256": _file_sha256(summary_path),
            "interpretation": summary["interpretation"],
        }
    )
