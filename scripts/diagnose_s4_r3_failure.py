"""只读诊断R3学生锚定SAC的动作、评论家和奖励失败机制。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.s4_r3_failure_diagnostic import run_s4_r3_failure_diagnostic
from src.rl.s4_training import json_safe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/s4_r3_failure_diagnostic_v1.yaml"),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="运行极小CUDA接口冒烟；结果不得用于机制判断。",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只核对R3封存结果、检查点、种子和只读边界。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_s4_r3_failure_diagnostic(
        args.config,
        quick=args.quick,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
