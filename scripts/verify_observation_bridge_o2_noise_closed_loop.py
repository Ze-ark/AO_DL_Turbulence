"""O2-D2 四档读出噪声短闭环技术检查（CUDA；不训练）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from observation_bridge.noise_closed_loop import CONFIG, run


def main() -> None:
    parser = argparse.ArgumentParser(description="O2-D2 冻结模型读出噪声短闭环检查，不评收益")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--quick", action="store_true", help="独立种子的两档技术冒烟，不是正式检查")
    parser.add_argument("--preflight-only", action="store_true", help="只检查封存来源、种子、模型，不创建输出")
    args = parser.parse_args()
    result = run(args.config, quick=args.quick, preflight_only=args.preflight_only)
    print(json.dumps({k: v for k, v in result.items() if k != "frozen_sources"}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
