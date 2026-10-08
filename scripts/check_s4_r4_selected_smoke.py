from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rl.r4_selected_smoke import run

if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'): sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='诊断性短闭环，不训练、不排名')
    parser.add_argument('--config', default='configs/experiments/s4_r4_selected_smoke_v1.yaml')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.config, preflight_only=args.preflight_only), ensure_ascii=False, indent=2))
