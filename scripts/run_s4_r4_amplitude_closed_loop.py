"""完整闭环力度对照，必须显式选择运行模式。"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rl.r4_amplitude_closed_loop import run

if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/s4_r4_amplitude_closed_loop_v1.yaml')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run-formal', action='store_true')
    group.add_argument('--quick', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.config, mode='quick' if args.quick else 'formal',
                         preflight_only=args.preflight_only), ensure_ascii=False, indent=2))
