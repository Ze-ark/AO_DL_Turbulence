"""S4-D2-R1受限理想控制能力上限入口；不训练RL。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.s4_oracle_bound import run_s4_r1_oracle_bound
from src.rl.s4_training import json_safe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行S4-D2-R1受限理想控制能力上限")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/s4_r1_oracle_bound_v1.yaml"),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只检查R1负结果、动作约束、源码和种子隔离，不占用CUDA或创建输出",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="运行CUDA快速冒烟；结果不得用于判断RL能力",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_s4_r1_oracle_bound(
        args.config,
        quick=args.quick,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
