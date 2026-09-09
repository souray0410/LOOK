#!/usr/bin/env python3
"""Run the suffix-start supplement after its parent study completes; test stays sealed."""
import argparse
import json
import os
from pathlib import Path


def main():
    from look_core.cli import add_runtime_arguments, resolve_runtime_arguments
    from look_core.distributed import parse_gpu_devices
    p = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(p)
    p.add_argument('--parent-summary', type=Path, required=True)
    p.add_argument('--parent-source', type=Path, required=True)
    p.add_argument('--gpus', default='0,1')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--wait-parent', action='store_true')
    p.add_argument('--no-render', action='store_true')
    args = p.parse_args()
    devices = parse_gpu_devices(args.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, devices))
    paths = resolve_runtime_arguments(args)
    from look_core.start_study import run_start_study
    result = run_start_study(paths, args.parent_summary, args.parent_source, devices,
        execute=args.execute, wait_parent=args.wait_parent, render=not args.no_render)
    print(json.dumps({k:v for k,v in result.items() if k not in ('identity','completed_cases')},indent=2))


if __name__ == '__main__':
    main()
