#!/usr/bin/env python3
"""Queue the three self-input cases after the fixed 15-case method study."""
import argparse,json,os
from pathlib import Path
from look_core.cli import add_runtime_arguments,resolve_runtime_arguments
from look_core.distributed import parse_gpu_devices

def main():
    p=argparse.ArgumentParser(description=__doc__);add_runtime_arguments(p)
    p.add_argument('--predecessor-summary',type=Path,required=True)
    p.add_argument('--gpus',default='0,1');p.add_argument('--execute',action='store_true')
    p.add_argument('--wait-predecessor',action='store_true')
    a=p.parse_args();devices=parse_gpu_devices(a.gpus)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str,devices))
    from look_core.self_input_study import run_self_input_study
    print(json.dumps(run_self_input_study(resolve_runtime_arguments(a),a.predecessor_summary,devices,
        execute=a.execute,wait=a.wait_predecessor),indent=2))
if __name__=='__main__':main()
