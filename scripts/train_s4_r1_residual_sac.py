"""S4-D2-R1小残差SAC入口；正式训练由用户在IDE终端启动。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.s4_r1_training import run_s4_r1_residual_sac
from src.rl.s4_training import json_safe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练S4-D2-R1小残差Soft Actor-Critic")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/s4_residual_sac_r1_v1.yaml"),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只检查单因素变更、上游证据和种子隔离，不创建输出或占用CUDA",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="运行CUDA快速冒烟；结果不得用于R1科学结论",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_s4_r1_residual_sac(
        args.config,
        quick=args.quick,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
