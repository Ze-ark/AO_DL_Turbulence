"""R5 动作力度开发诊断入口；正式运行由用户在 IDE 启动。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rl.r5_margin_development import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/s4_r5_margin_development_v1.yaml")
    parser.add_argument("--quick", action="store_true", help="小型 CUDA 冒烟，不用于算法排名")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, quick=args.quick, preflight_only=args.preflight_only),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
