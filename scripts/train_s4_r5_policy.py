"""R5-2物理梯度残差策略训练；正式训练由用户在IDE启动。"""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def main():
    from src.rl.r5_policy_training import run
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',default='configs/experiments/s4_r5_policy_training_v1.yaml'); p.add_argument('--quick',action='store_true'); p.add_argument('--preflight-only',action='store_true'); a=p.parse_args(); print(json.dumps(run(a.config,a.quick,a.preflight_only),ensure_ascii=False,indent=2))
if __name__=='__main__': main()
