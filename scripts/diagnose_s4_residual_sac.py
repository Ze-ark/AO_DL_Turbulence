"""S4-D2残差SAC失败诊断入口；不执行训练或检查点更新。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.s4_diagnostic import run_s4_residual_diagnostic
from src.rl.s4_training import json_safe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="诊断S4-D2残差SAC失败原因")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/s4_residual_sac_diagnostic_v1.yaml"),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只核对训练失败、检查点、源码和种子隔离，不占用CUDA或创建输出",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="运行CUDA快速诊断冒烟；结果不得用于失败原因结论",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_s4_residual_diagnostic(
        args.config,
        quick=args.quick,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
