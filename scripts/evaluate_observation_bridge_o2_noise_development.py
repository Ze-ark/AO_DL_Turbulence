"""O2-D3 四档人工读出噪声完整回合开发对照（CUDA；不训练）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from observation_bridge.noise_development import CONFIG,run


def main()->None:
    parser=argparse.ArgumentParser(description='O2-D3 冻结控制器四档噪声开发对照，不训练或独立确认')
    parser.add_argument('--config',default=CONFIG)
    parser.add_argument('--quick',action='store_true',help='独立种子两档技术冒烟，不报告收益')
    parser.add_argument('--preflight-only',action='store_true',help='只检查封存来源、预算和模型，不创建输出')
    args=parser.parse_args()
    result=run(args.config,quick=args.quick,preflight_only=args.preflight_only)
    print(json.dumps({k:v for k,v in result.items() if k not in ('frozen_sources','analysis')},ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
