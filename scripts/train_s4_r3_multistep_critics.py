"""训练R3-D2-A冻结演员多步评论家；正式训练由用户在IDE启动。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.s4_r3_multistep_critic_training import (
    run_s4_r3_multistep_critic_training,
)
from src.rl.s4_training import json_safe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/s4_r3_multistep_critic_repair_v1.yaml"),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="运行极小CUDA训练冒烟；输出不得用于科学结论。",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只检查上游、数据拆分、训练规模和安全保护。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_s4_r3_multistep_critic_training(
        args.config,
        quick=args.quick,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

