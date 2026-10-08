"""R4-0因果观测与动作时间对齐CUDA诊断；不训练模型。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl.r4_interface_smoke import run_interface_smoke


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if callable(getattr(stream, "reconfigure", None)):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/experiments/s4_r4_interface_smoke_v1.yaml"))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_interface_smoke(args.config, preflight_only=args.preflight_only),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
