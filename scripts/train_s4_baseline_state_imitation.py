"""在传统控制器基线轨迹上训练新增11维监督模型，并做配对闭环诊断。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.s4_baseline_state_imitation import run_s4_baseline_state_imitation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/s4_baseline_state_imitation_v1.yaml",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="只运行极小CUDA冒烟；结果不得用于科学判断。",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只检查证据、数据拆分、模型规模、CUDA和安全边界。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_s4_baseline_state_imitation(
        args.config,
        quick=args.quick,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
