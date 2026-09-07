#!/usr/bin/env python3
"""Select any physical GPUs; preserve frozen case/worker identity across switches.

PYTHONPATH must point to WORKER_ROOT/tool. This entrypoint and its execution
adapters may come from a newer controller release; numerical code may not.
"""
import argparse
import importlib.util
from pathlib import Path


def load(name):
    path=Path(__file__).resolve().parents[1]/'tool/operations'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker-root',type=Path)
    p.add_argument('--plan',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--lock-path',type=Path)
    p.add_argument('--control',type=Path)
    p.add_argument('--set-devices',nargs='*',type=int)
    p.add_argument('--mode',choices=['drain','interrupt'],default='drain')
    p.add_argument('--worker')
    p.add_argument('--execution-acceptance',type=Path)
    p.add_argument('--collect-resumed-starts',action='store_true')
    a=p.parse_args();queue=load('flexible_queue')
    if a.set_devices is not None:
        if not a.control:p.error('--control required')
        print(queue.set_devices(a.control,a.set_devices,a.mode));return
    if not all((a.worker_root,a.plan,a.output)):p.error('Worker root, plan and output required')
    import look_core.dual_queue
    if Path(look_core.dual_queue.__file__).resolve().parents[1]!=(a.worker_root/'tool').resolve():
        raise ValueError('PYTHONPATH must resolve to the frozen worker root')
    if a.worker:
        load('flexible_worker').worker(a.worker_root,a.plan,a.output,a.worker);return
    if not a.control or not a.lock_path:p.error('--control and --lock-path required')
    try:
        queue.supervise(a.worker_root,a.plan,a.output,a.lock_path,a.control,Path(__file__),execution_acceptance=a.execution_acceptance)
    finally:
        if a.collect_resumed_starts and (a.output/'queue_status.json').exists():
            from look_core.resumed_starts import collect
            collect(a.output.parent)


if __name__=='__main__':main()
