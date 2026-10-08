"""A10冻结模型有害动作否决开关确认；正式运行由用户在IDE启动。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.s4_r3_h16_veto_gate_confirmation import run_s4_r3_h16_veto_gate_confirmation


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if callable(getattr(stream, "reconfigure", None)):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/experiments/s4_r3_h16_veto_gate_confirmation_v1.yaml"))
    parser.add_argument("--quick", action="store_true", help="独立种子的小型CUDA快速冒烟")
    parser.add_argument("--quick-run-tag", help="仅快速模式：显式指定新目录后缀，保留已有运行；不会自动重试")
    parser.add_argument("--preflight-only", action="store_true", help="只读预检，不采集、不评估")
    args = parser.parse_args()
    result = run_s4_r3_h16_veto_gate_confirmation(
        args.config, quick=args.quick, preflight_only=args.preflight_only,
        quick_run_tag=args.quick_run_tag)
    keys = ("status", "device", "dataset_files", "unique_weather_episodes", "paired_action_samples",
            "collection_branches", "output_directory") if args.preflight_only else ("experiment", "interpretation")
    print(json.dumps({k: result[k] for k in keys}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
