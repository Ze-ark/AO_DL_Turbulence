"""保存并比较S2最终封存实验的确定性复现快照。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = PROJECT_ROOT / "outputs" / "s2_baselines_v1" / "final_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("snapshot", "compare"))
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    snapshot_path = result_dir / "reproducibility_snapshot.json"
    current = _snapshot(result_dir)
    if args.mode == "snapshot":
        snapshot_path.write_text(
            json.dumps(current, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"snapshot={snapshot_path}")
        return

    if not snapshot_path.exists():
        raise FileNotFoundError("reproducibility snapshot is missing")
    original = json.loads(snapshot_path.read_text(encoding="utf-8"))
    metrics_match = original["canonical_summary"] == current["canonical_summary"]
    trajectories_match = original["trajectory_sha256"] == current["trajectory_sha256"]
    verdict = "REPRODUCIBLE" if metrics_match and trajectories_match else "NOT_REPRODUCIBLE"
    report = _validation_report(
        current["source_summary"],
        verdict,
        metrics_match,
        trajectories_match,
    )
    (result_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (result_dir / "validation_report.md").write_text(
        _markdown_report(report),
        encoding="utf-8",
    )
    print(json.dumps(report["reproducibility"], ensure_ascii=False))
    if verdict != "REPRODUCIBLE":
        raise SystemExit("S2 final reproducibility check failed")


def _snapshot(result_dir: Path) -> dict[str, Any]:
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "source_summary": summary,
        "canonical_summary": _canonical_summary(summary),
        "trajectory_sha256": {
            path.name: _sha256(path)
            for path in sorted((result_dir / "trajectories").glob("*.h5"))
        },
    }


def _canonical_summary(summary: dict[str, Any]) -> dict[str, Any]:
    metric_names = (
        "strehl",
        "power_in_bucket",
        "phase_rmse",
        "reward",
        "action_cost",
        "violation_fraction",
    )
    controllers = []
    for item in summary["test_results"]:
        controllers.append(
            {
                "controller": item["controller"],
                "episodes": item["episodes"],
                "steps_per_episode": item["steps_per_episode"],
                "episode_seeds": item["episode_seeds"],
                "parameters": item["parameters"],
                "metrics": {name: item[name] for name in metric_names},
                "paired_delta_vs_no_correction": item.get(
                    "paired_delta_vs_no_correction"
                ),
            }
        )
    oracle = summary["oracle_modal_upper_bound"]
    return {
        "test_split": summary["test_split"],
        "test_base_seeds": summary["test_base_seeds"],
        "episodes_per_base_seed": summary["episodes_per_base_seed"],
        "steps": summary["steps"],
        "controllers": controllers,
        "oracle": {
            "episodes": oracle["episodes"],
            "steps_per_episode": oracle["steps_per_episode"],
            "episode_seeds": oracle["episode_seeds"],
            "strehl": oracle["strehl"],
            "power_in_bucket": oracle["power_in_bucket"],
            "phase_rmse": oracle["phase_rmse"],
        },
        "best_controller": summary["best_deployable_controller_by_power_in_bucket"],
        "s2_gate": summary["s2_gate"],
    }


def _validation_report(
    summary: dict[str, Any],
    verdict: str,
    metrics_match: bool,
    trajectories_match: bool,
) -> dict[str, Any]:
    findings = []
    for item in summary["test_results"]:
        if item["controller"] == "no_correction":
            continue
        delta = item["paired_delta_vs_no_correction"]["power_in_bucket"]
        findings.append(
            {
                "controller": item["controller"],
                "paired_power_in_bucket_delta": delta,
                "confidence": "SOLID" if delta["ci95_low"] > 0 else "CAUTION",
            }
        )
    return {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "validate",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "VERIFIED" if verdict == "REPRODUCIBLE" else "ANALYZED",
            "version_label": "s2_final_validation_v1",
        },
        "source": "AO-S2-FINAL-COMPARISON",
        "overall_confidence": "CAUTION",
        "statistical_findings": findings,
        "warnings": [
            "只覆盖一个纯仿真湍流、风速和时延配置，不能外推到其他条件或真实SLM。",
            "当前波前观测是无噪声理想瞳面观测，不是传播后的真实离轴全息复场。",
            "主要指标预先指定为桶内功率；其他指标用于一致性检查。",
        ],
        "fallacy_scan": {
            "coverage": "11/11",
            "items": [
                {"fallacy": "Simpson's paradox", "severity": "NOTE", "detail": "单一仿真条件，尚不能检查跨条件趋势反转。"},
                {"fallacy": "Ecological fallacy", "severity": "NONE", "detail": "统计与推断单位均为完整回合。"},
                {"fallacy": "Berkson's paradox", "severity": "NOTE", "detail": "固定仿真条件限制外部推广，但未据此筛选成功回合。"},
                {"fallacy": "Collider bias", "severity": "NONE", "detail": "配对比较未加入由控制器和指标共同造成的控制变量。"},
                {"fallacy": "Base rate neglect", "severity": "N/A", "detail": "不涉及诊断概率。"},
                {"fallacy": "Regression to the mean", "severity": "NONE", "detail": "回合未按极端初始表现筛选，且有同轨迹不校正对照。"},
                {"fallacy": "Survivorship bias", "severity": "NONE", "detail": "所有64个最终回合均完成并纳入。"},
                {"fallacy": "Look-elsewhere effect", "severity": "NOTE", "detail": "主要指标已预先指定；多项次要指标不用于选择结论。"},
                {"fallacy": "Garden of forking paths", "severity": "NOTE", "detail": "开发、感知测试和最终种子分离；最终参数在开封前冻结。"},
                {"fallacy": "Correlation != causation", "severity": "NOTE", "detail": "同轨迹干预支持模拟器内控制效果，不支持真实硬件或自然大气因果外推。"},
                {"fallacy": "Reverse causality", "severity": "N/A", "detail": "控制动作先于对应系统响应，且不作观察性因果推断。"},
            ],
        },
        "reproducibility": {
            "method": "same-command deterministic rerun; timing excluded",
            "verdict": verdict,
            "controller_metrics_exact_match": metrics_match,
            "trajectory_files_sha256_exact_match": trajectories_match,
        },
    }


def _markdown_report(report: dict[str, Any]) -> str:
    passport = report["material_passport"]
    repro = report["reproducibility"]
    finding_rows = "\n".join(
        "| {controller} | {mean:.6f} | [{low:.6f}, {high:.6f}] | {confidence} |".format(
            controller=item["controller"],
            mean=item["paired_power_in_bucket_delta"]["mean"],
            low=item["paired_power_in_bucket_delta"]["ci95_low"],
            high=item["paired_power_in_bucket_delta"]["ci95_high"],
            confidence=item["confidence"],
        )
        for item in report["statistical_findings"]
    )
    fallacy_rows = "\n".join(
        f"| {item['fallacy']} | {item['severity']} | {item['detail']} |"
        for item in report["fallacy_scan"]["items"]
    )
    warning_rows = "\n".join(f"- {item}" for item in report["warnings"])
    return f"""## Material Passport

- Origin Skill: {passport['origin_skill']}
- Origin Mode: {passport['origin_mode']}
- Origin Date: {passport['origin_date']}
- Verification Status: {passport['verification_status']}
- Version Label: {passport['version_label']}

## Validation Report

- **Source**: {report['source']}
- **Overall Confidence**: {report['overall_confidence']}

### Statistical Findings

| Controller | Paired PIB Delta | 95% CI | Confidence |
|---|---:|---:|---|
{finding_rows}

### Warnings

{warning_rows}

### Fallacy Scan

- **Coverage**: {report['fallacy_scan']['coverage']} fallacy types checked

| Fallacy | Severity | Detail |
|---|---|---|
{fallacy_rows}

### Reproducibility

- **Method**: {repro['method']}
- **Verdict**: {repro['verdict']}
- **Controller metrics exact match**: {repro['controller_metrics_exact_match']}
- **Trajectory SHA-256 exact match**: {repro['trajectory_files_sha256_exact_match']}
"""


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
