"""仅恢复 O2-C 已有技术冒烟汇总，绝不启动环境或训练。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from observation_bridge.development_recovery import CONFIG, run


def main() -> None:
    parser = argparse.ArgumentParser(description="核验已有 O2-C 记录，汇总另存；新增物理转移为零")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--preflight-only", action="store_true", help="只核验元数据/封存权重；不写输出")
    args = parser.parse_args()
    print(json.dumps(run(args.config, preflight_only=args.preflight_only), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
