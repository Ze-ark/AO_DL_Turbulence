"""检查R3-D2-A多步评论家修复设计，不训练任何模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.s4_r3_multistep_critic_repair import (
    preflight_s4_r3_multistep_critic_repair_design,
)
from src.rl.s4_training import json_safe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiments/s4_r3_multistep_critic_repair_design_v1.yaml"
        ),
    )
    return parser.parse_args()


def main() -> None:
    result = preflight_s4_r3_multistep_critic_repair_design(parse_args().config)
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

