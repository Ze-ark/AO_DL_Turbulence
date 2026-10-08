"""A8末端元数据故障恢复；默认只读，显式确认后才补写最终汇总。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.s4_r3_h16_head_split_recovery import finalize_s4_r3_h16_head_split


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/s4_r3_h16_head_split_v1.yaml"),
    )
    parser.add_argument(
        "--acknowledge-finalize",
        action="store_true",
        help="确认只补写恢复清单与summary；不训练、不推理、不生成测试回合。",
    )
    args = parser.parse_args()
    result = finalize_s4_r3_h16_head_split(
        args.config,
        acknowledge_finalize=args.acknowledge_finalize,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
