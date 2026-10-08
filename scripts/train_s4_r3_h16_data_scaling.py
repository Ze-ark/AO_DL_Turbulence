"""A7：96与384条独立训练回合的监督学习对照，由用户在IDE启动。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.s4_r3_h16_data_scaling import run_s4_r3_h16_data_scaling
from src.rl.s4_training import json_safe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "configs/experiments/s4_r3_h16_data_scaling_v1.yaml"
    ))
    parser.add_argument("--quick", action="store_true", help="小型CUDA冒烟，不能作科学结论。")
    parser.add_argument("--preflight-only", action="store_true", help="只读核对，不生成回合或训练。")
    args = parser.parse_args()
    result = run_s4_r3_h16_data_scaling(
        args.config, quick=args.quick, preflight_only=args.preflight_only
    )
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
