from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.s4_r3_student_anchored_sac import run_s4_r3_student_anchored_sac
from src.rl.s4_training import json_safe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="训练S4-D2-R3学生锚定残差SAC（正式训练由用户在IDE终端启动）"
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/s4_student_anchored_sac_r3_v1.yaml",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = run_s4_r3_student_anchored_sac(
        args.config,
        quick=args.quick,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
