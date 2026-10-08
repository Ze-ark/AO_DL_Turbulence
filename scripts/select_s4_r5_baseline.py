"""运行 R5-1 传统基座开发集筛选；正式运行由用户在 IDE 启动。"""
from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main() -> None:
    from src.rl.r5_baseline_selection import run
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/s4_r5_baseline_selection_v1.yaml")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, quick=args.quick, preflight_only=args.preflight_only), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
