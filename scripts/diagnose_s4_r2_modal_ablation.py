"""S4-D2-R2前10维与新增高阶维的只读配对消融入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.s4_r2_modal_ablation import run_s4_r2_modal_ablation
from src.rl.s4_training import json_safe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="消融R2策略的前10维动作与新增高阶动作"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiments/s4_residual_sac_r2_modal_ablation_v1.yaml"
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只核对上游审计、检查点、配对种子和只读边界，不占用CUDA",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="运行CUDA快速冒烟；结果不得用于模态归因结论",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_s4_r2_modal_ablation(
        args.config,
        quick=args.quick,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
