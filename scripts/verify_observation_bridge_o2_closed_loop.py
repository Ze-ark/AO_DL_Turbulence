"""O2-B CUDA 短闭环技术入口；不是训练或正式性能评价。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from observation_bridge.closed_loop import CONFIG, run


def main() -> None:
    parser = argparse.ArgumentParser(description="O2-B 人工全息观测短闭环技术检查（CUDA；无训练）")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = run(args.config, preflight_only=args.preflight_only)
    print(json.dumps({k: v for k, v in result.items() if k not in ("records", "frozen_sources")},
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
