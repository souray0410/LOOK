#!/usr/bin/env python3
"""Bounded method controls after the parent and suffix studies; test sealed."""
import argparse
import json
import os
from pathlib import Path

def main():
    from look_core.cli import add_runtime_arguments,resolve_runtime_arguments
    from look_core.distributed import parse_gpu_devices
    p=argparse.ArgumentParser(description=__doc__);add_runtime_arguments(p)
    p.add_argument('--parent-summary',type=Path,required=True)
    p.add_argument('--parent-source',type=Path,required=True)
    predecessor=p.add_mutually_exclusive_group(required=True)
    predecessor.add_argument('--suffix-summary',type=Path)
    predecessor.add_argument('--start-evidence-manifest',type=Path)
    p.add_argument('--gpus',default='0,1');p.add_argument('--execute',action='store_true')
    p.add_argument('--wait-predecessors',action='store_true');p.add_argument('--no-render',action='store_true')
    args=p.parse_args();devices=parse_gpu_devices(args.gpus)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str,devices))
    from look_core.method_study import run_method_study
    result=run_method_study(resolve_runtime_arguments(args),args.parent_summary,args.parent_source,args.suffix_summary,
        devices,execute=args.execute,wait=args.wait_predecessors,render=not args.no_render,
        start_evidence_manifest=args.start_evidence_manifest)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
