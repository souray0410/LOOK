#!/usr/bin/env python3
"""Wait for frozen validation dependencies, replay on validation, evaluate test."""
import argparse
import importlib.util
from pathlib import Path
from look_core.dual_queue import read,record,check_source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker')
    p.add_argument('--phase',choices=['validation_replay','test'])
    p.add_argument('--microbatch',type=int,choices=[1,2,4,8],default=8)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    if a.worker:
        from look_core.test_runtime import run_job
        frozen=a.output/'frozen_manifest.json';f=read(frozen)
        if check_source(root)!=f['source_manifest_sha256']:raise ValueError('Worker source identity changed')
        job=next(j for j in f['jobs'] if j['id']==a.worker)
        run_job(job,a.output/a.phase/a.worker,a.phase,record(frozen),microbatch=a.microbatch,
                gate=a.output/'replay_gate.json' if a.phase=='test' else None)
    else:
        path=root/'tool/operations/test_queue.py'
        spec=importlib.util.spec_from_file_location('primary_queue',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.supervise(root,a.request,a.output,Path(__file__).resolve())


if __name__=='__main__':main()
