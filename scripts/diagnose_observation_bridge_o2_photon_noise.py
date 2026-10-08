"""O2-D4 静态光子噪声标尺；完整诊断由用户在 IDE 启动。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from observation_bridge.photon_noise_diagnostic import CONFIG, run


def main() -> None:
    parser = argparse.ArgumentParser(description="O2-D4：固定静态光场、单因素光子计数，零控制/训练")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--quick", action="store_true", help="独立种子 9 次技术读取，不给正式噪声曲线")
    parser.add_argument("--preflight-only", action="store_true", help="只检查来源/种子/CUDA，不生成图像、加载模型或写输出")
    args = parser.parse_args()
    print(json.dumps(run(args.config, quick=args.quick, preflight_only=args.preflight_only),
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
