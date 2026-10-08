"""O2-C 人工全息观测开发对照；CUDA，仅推理，不训练。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from observation_bridge.development import CONFIG, run


def main() -> None:
    parser = argparse.ArgumentParser(description="O2-C 冻结控制器全息观测开发对照（CUDA；无训练）")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--quick", action="store_true", help="独立种子的技术冒烟，不报告收益")
    parser.add_argument("--preflight-only", action="store_true", help="只校验来源、预算和模型；不创建输出")
    args = parser.parse_args()
    result = run(args.config, quick=args.quick, preflight_only=args.preflight_only)
    print(json.dumps({k: v for k, v in result.items() if k not in ("records", "frozen_sources")},
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
