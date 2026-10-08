"""从已完成的v3排序轨迹生成并列感知摘要，不重跑仿真。"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rl.r4_closed_loop_ranking_finalize import run


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/s4_r4_closed_loop_ranking_finalize_v1.yaml")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, preflight_only=args.preflight_only), ensure_ascii=False, indent=2))
