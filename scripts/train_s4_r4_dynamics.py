"""R4-1A连续轨迹与监督预测模型；正式运行由用户IDE启动，不训练RL。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl.r4_dynamics_experiment import run


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if callable(getattr(stream, "reconfigure", None)):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/s4_r4_dynamics_v1.yaml"))
    parser.add_argument("--quick", action="store_true", help="独立小预算CUDA冒烟，不用于科学结论")
    parser.add_argument("--preflight-only", action="store_true", help="只读预检，不采集、不训练")
    args = parser.parse_args()
    print(json.dumps(run(args.config, quick=args.quick, preflight_only=args.preflight_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
