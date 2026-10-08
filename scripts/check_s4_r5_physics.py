"""R5-0物理诊断；不训练模型。"""
from pathlib import Path
import argparse
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/s4_r5_physics_check_v1.yaml")
    parser.add_argument("--quick", action="store_true", help="小型CUDA冒烟，不给科学PASS")
    parser.add_argument("--preflight-only", action="store_true", help="只读检查，不创建输出")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from src.rl.r5_contract_check import run
    print(json.dumps(run(args.config,quick=args.quick,preflight_only=args.preflight_only),
                     ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
