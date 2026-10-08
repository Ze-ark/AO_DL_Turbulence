"""R5-R2 全新天气确认入口；正式运行须由用户在 IDE 启动。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rl.r5_independent_confirmation_r2 import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/s4_r5_independent_confirmation_r2.yaml")
    parser.add_argument("--quick", action="store_true", help="16帧CUDA冒烟，不作性能结论")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, quick=args.quick, preflight_only=args.preflight_only),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
