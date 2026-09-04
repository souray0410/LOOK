#!/usr/bin/env python3
"""Retrain fixed layer3 backbones and fit joint LOOK, both selected by Macro-F1."""
import argparse
import json
import os
from pathlib import Path


def main():
    from look_core.cli import add_runtime_arguments,resolve_runtime_arguments
    from look_core.distributed import parse_gpu_devices
    p=argparse.ArgumentParser(description=__doc__);add_runtime_arguments(p)
    p.add_argument('--gpus',default='0,1');p.add_argument('--spec',type=Path)
    p.add_argument('--execute',action='store_true');args=p.parse_args()
    devices=parse_gpu_devices(args.gpus);os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str,devices))
    paths=resolve_runtime_arguments(args)
    spec=args.spec or paths.project_root/'configs/macro_f1_study.json'
    from look_core.macro_f1_study import run_macro_f1_study
    print(json.dumps(run_macro_f1_study(paths,spec,devices,args.execute),indent=2))

if __name__=='__main__':main()
