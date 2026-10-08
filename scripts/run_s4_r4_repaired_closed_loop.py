"""R4-1C训练验收后运行三组完整闭环；预测门槛不通过则立即停止。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.rl.r4_closed_loop_repair import run


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if callable(getattr(stream, "reconfigure", None)):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/s4_r4_closed_loop_repair_v1.yaml"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, phase="evaluate", quick=args.quick,
                         preflight_only=args.preflight_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
