"""A8-D3最佳与末次模型对照：不训练、不生成数据、不访问独立测试。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.s4_r3_h16_checkpoint_endpoints import run_s4_r3_h16_checkpoint_endpoints


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        configure = getattr(stream, "reconfigure", None)
        if callable(configure):
            configure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/s4_r3_h16_checkpoint_endpoints_v1.yaml"))
    parser.add_argument("--preflight-only", action="store_true", help="只读核对输入、模型、哈希和范围，不创建输出。")
    parser.add_argument("--quick", action="store_true", help="48条回合梯度的CUDA快速冒烟，不作正式判断。")
    args = parser.parse_args()
    result = run_s4_r3_h16_checkpoint_endpoints(args.config, quick=args.quick, preflight_only=args.preflight_only)
    keys = ("status", "policy_seeds", "selected_episodes_per_split", "expected_episode_records",
            "expected_endpoint_metrics", "expected_gap_comparisons", "output_directory") if args.preflight_only else ("experiment", "alignment", "interpretation")
    print(json.dumps({k: result[k] for k in keys}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
