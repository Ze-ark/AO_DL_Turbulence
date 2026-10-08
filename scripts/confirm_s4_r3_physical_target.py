"""运行R3-D2-A5-R1固定16步纯物理目标独立确认。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.s4_r3_physical_target_confirmation import (
    run_s4_r3_physical_target_confirmation,
)
from src.rl.s4_training import json_safe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiments/s4_r3_physical_target_confirmation_v1.yaml"
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="运行小型CUDA冒烟；输出不得作为科研结论。",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只检查固定步长、独立种子、上游哈希和安全保护。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_s4_r3_physical_target_confirmation(
        args.config,
        quick=args.quick,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

