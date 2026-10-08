"""A8-D2训练—验证梯度翻转来源诊断；不训练、不加载检查点、不访问独立测试样本。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.s4_r3_h16_gradient_shift import run_s4_r3_h16_gradient_shift


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/s4_r3_h16_gradient_shift_v1.yaml"),
    )
    parser.add_argument("--quick", action="store_true", help="小型CUDA冒烟，不产生正式科学结论。")
    parser.add_argument("--preflight-only", action="store_true", help="只读核对输入、哈希和范围，不创建输出。")
    args = parser.parse_args()
    result = run_s4_r3_h16_gradient_shift(
        args.config,
        quick=args.quick,
        preflight_only=args.preflight_only,
    )
    if args.preflight_only:
        compact = {key: result[key] for key in (
            "status", "policy_seeds", "selected_episodes", "expected_descriptor_records",
            "expected_gradient_shift_comparisons", "expected_matching_comparisons", "output_directory",
        )}
    else:
        compact = {"experiment": result["experiment"], "interpretation": result["interpretation"]}
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
