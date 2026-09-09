#!/usr/bin/env python3
"""Declare and run bounded train/validation stability, calibration and cost analysis."""
import argparse
import importlib.util
from pathlib import Path
from look_core.dual_queue import read,check_source,atomic


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',type=int,choices=[3407,3408,3409]);p.add_argument('--microbatch',type=int,choices=[1,2,4,8],default=8)
    p.add_argument('--cpu-only',action='store_true')
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    if a.worker:
        from look_core.supplement_runtime import run_seed
        manifest=a.output/'diagnostic_manifest.json';plan=read(manifest)
        if check_source(root)!=plan['source_manifest_sha256']:raise ValueError('Worker source changed')
        case=a.output/'gpu'/f'seed_{a.worker}';attempt=case/f'execution_micro_{a.microbatch}'
        # Tiny actual-data acceptance precedes each execution batch. It cannot count as full evidence.
        run_seed(manifest,attempt/'acceptance',a.worker,a.microbatch,limit=2)
        result=run_seed(manifest,attempt/'full',a.worker,a.microbatch)
        atomic(result,case/'complete.json')
    elif a.cpu_only:
        from look_core.supplement_diagnostics import freeze,cpu_analysis
        freeze(a.request,a.output,check_source(root));cpu_analysis(a.output)
    else:
        path=root/'tool/operations/supplement_queue.py';spec=importlib.util.spec_from_file_location('supplement_queue',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.supervise(root,a.request,a.output,Path(__file__).resolve())


if __name__=='__main__':main()
