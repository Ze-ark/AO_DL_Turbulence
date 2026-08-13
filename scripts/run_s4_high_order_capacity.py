"""运行S4-D2-R2新增高阶子空间理想容量诊断。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.s4_high_order_capacity import run_s4_high_order_capacity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="只允许新增高阶模式的非学习理想容量诊断"
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/s4_high_order_capacity_v1.yaml",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = run_s4_high_order_capacity(
        args.config,
        quick=args.quick,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
