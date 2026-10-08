"""R4-1C断电恢复：先严格重演和核对，再补齐最后模型；正式运行由用户在IDE启动。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl.r4_closed_loop_repair_resume import run


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if callable(getattr(stream, "reconfigure", None)):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/experiments/s4_r4_closed_loop_repair_v1.yaml"))
    parser.add_argument("--preflight-only", action="store_true", help="只读检查可恢复性")
    parser.add_argument("--smoke", action="store_true", help="只核对前两批；不写正式结果")
    args = parser.parse_args()
    if args.preflight_only and args.smoke:
        parser.error("--preflight-only and --smoke are mutually exclusive")
    print(json.dumps(run(args.config, preflight_only=args.preflight_only, smoke=args.smoke),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
