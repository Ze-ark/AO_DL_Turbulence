"""同状态受限动作力度诊断；不训练、不操作硬件。"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rl.r4_amplitude_probe import run

if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/s4_r4_amplitude_probe_v1.yaml')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run-formal', action='store_true')
    mode.add_argument('--quick', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.config, quick=args.quick, preflight_only=args.preflight_only),
                     ensure_ascii=False, indent=2))
