"""运行R4既有五候选的分层交叉拟合重选诊断；不训练、不重跑仿真。"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rl.r4_candidate_reselection import run


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/s4_r4_candidate_reselection_v1.yaml")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, preflight_only=args.preflight_only),
                     ensure_ascii=False, indent=2))

