"""执行一次受限学生状态数据聚合、同预算监督训练和闭环诊断。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.s4_student_state_aggregation import run_s4_student_state_aggregation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/s4_student_state_aggregation_v1.yaml",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="只运行极小CUDA冒烟；结果不得用于科学判断。",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只检查封存证据、数据预算、种子、CUDA和安全边界。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_s4_student_state_aggregation(
        args.config,
        quick=args.quick,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
