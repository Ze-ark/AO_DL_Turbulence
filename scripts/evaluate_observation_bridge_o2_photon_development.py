"""O2-D6 单因素光子噪声完整开发对照；不训练、不作独立确认。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from observation_bridge.photon_development import CONFIG, run


def main() -> None:
    parser = argparse.ArgumentParser(description="O2-D6 冻结控制器单因素光子噪声完整开发对照")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--quick", action="store_true", help="独立天气两档技术冒烟，不输出科学收益")
    parser.add_argument("--preflight-only", action="store_true", help="只核对来源、种子和模型，不创建输出")
    args = parser.parse_args()
    result = run(args.config, quick=args.quick, preflight_only=args.preflight_only)
    print(json.dumps({k: v for k, v in result.items() if k not in ("frozen_sources", "stream_manifest")},
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
