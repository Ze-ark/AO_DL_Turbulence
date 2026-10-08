"""训练A6十六步实际累计奖励成对探针；正式训练由用户在IDE启动。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.s4_r3_h16_reward_pairwise_probe import (
    run_s4_r3_h16_reward_pairwise_probe,
)
from src.rl.s4_training import json_safe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiments/s4_r3_h16_reward_pairwise_probe_v1.yaml"
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="运行极小CUDA训练冒烟；输出不得用于科学结论。",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只检查上游哈希、三集合拆分、单因素对照和安全保护。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_s4_r3_h16_reward_pairwise_probe(
        args.config,
        quick=args.quick,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
