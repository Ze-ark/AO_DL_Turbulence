"""A8-D1共享主干梯度冲突诊断；不训练、不更新权重、不读取独立测试样本。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.s4_r3_h16_gradient_conflict import run_s4_r3_h16_gradient_conflict


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/s4_r3_h16_gradient_conflict_v1.yaml"),
    )
    parser.add_argument("--quick", action="store_true", help="小型CUDA诊断冒烟，不产生正式科学结论。")
    parser.add_argument("--preflight-only", action="store_true", help="只读核对输入、哈希和安全边界，不创建输出。")
    args = parser.parse_args()
    result = run_s4_r3_h16_gradient_conflict(
        args.config,
        quick=args.quick,
        preflight_only=args.preflight_only,
    )
    if args.preflight_only:
        compact = {key: result[key] for key in (
            "status", "policy_seeds", "selected_episodes_per_split",
            "expected_episode_records", "gradient_evaluations_per_record", "output_directory",
        )}
    else:
        compact = {"experiment": result["experiment"], "interpretation": result["interpretation"]}
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
