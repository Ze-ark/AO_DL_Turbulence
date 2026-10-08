from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.rl.r4_closed_loop import run

if __name__=='__main__':
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'): stream.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser(description='三组冻结闭环；正式运行须显式选择--run-formal')
    p.add_argument('--config',default='configs/experiments/s4_r4_closed_loop_v1.yaml')
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--run-formal',action='store_true')
    group.add_argument('--quick',action='store_true')
    group.add_argument('--batch-check',action='store_true')
    p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args()
    mode='formal' if args.run_formal else 'quick' if args.quick else 'batch'
    print(json.dumps(run(args.config,mode=mode,preflight_only=args.preflight_only),ensure_ascii=False,indent=2))
