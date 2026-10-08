"""A8共享/分工输出监督对照；正式训练由用户在IDE启动。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.s4_r3_h16_head_split import run_s4_r3_h16_head_split


def main() -> None:
    # IDE和非交互管道在Windows上可能使用不同默认编码，统一中文进度输出。
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/s4_r3_h16_head_split_v1.yaml"))
    parser.add_argument("--quick", action="store_true", help="小型CUDA冒烟，不是正式训练。")
    parser.add_argument("--preflight-only", action="store_true", help="只读核对，不训练、不生成回合。")
    args = parser.parse_args()
    result = run_s4_r3_h16_head_split(args.config, quick=args.quick, preflight_only=args.preflight_only)
    compact = ({k: result[k] for k in ("status", "planned_fits", "maximum_updates_per_fit", "parameter_counts", "test_namespace", "output_directory")}
               if args.preflight_only else {"experiment": result["experiment"], "interpretation": result["interpretation"]})
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
