"""O2-D7 联合噪声短闭环技术检查；不训练，不给出科学收益排名。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from observation_bridge.combined_noise_closed_loop import CONFIG, run


def main() -> None:
    parser = argparse.ArgumentParser(description="O2-D7 冻结控制器的联合相机噪声短闭环检查")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--quick", action="store_true", help="独立天气、两条件技术冒烟，不是正式检查")
    parser.add_argument("--preflight-only", action="store_true", help="只检查来源、种子、模型，不创建输出")
    args = parser.parse_args()
    result = run(args.config, quick=args.quick, preflight_only=args.preflight_only)
    print(json.dumps({k: v for k, v in result.items() if k not in ("frozen_sources", "stream_manifest")},
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
