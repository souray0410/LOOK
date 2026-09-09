#!/usr/bin/env python3
"""Operational adapter: park an idle frozen graph on CPU during GAN workers.

The immutable scientific entrypoint and worker modules are executed unchanged.
No batch, seed, objective, checkpoint or correction identity is modified.
"""
from __future__ import annotations
import argparse
import gc
import runpy
import sys
import weakref
from pathlib import Path


def park_while(graph,action,audit_path):
    import torch
    from look_core.distributed import module_state_sha256
    from look_core.state import atomic_write_json,utc_now,file_sha256
    target=graph.device
    if target.type!='cuda':return action()
    if any(p.requires_grad for p in graph.parameters()):raise ValueError('Only an idle frozen graph may be parked')
    torch.cuda.synchronize(target)
    before=module_state_sha256(graph)
    record=dict(operation='idle_frozen_graph_cpu_parking_v1',started_at_utc=utc_now(),
        operation_source=str(Path(__file__).resolve()),operation_sha256=file_sha256(Path(__file__)),
        original_device=str(target),frozen_state_sha256=before,test_access=False,
        allocated_before=torch.cuda.memory_allocated(target),reserved_before=torch.cuda.memory_reserved(target))
    graph.to('cpu');torch.cuda.synchronize(target);gc.collect();torch.cuda.empty_cache()
    record.update(status='parked_for_gan',allocated_parked=torch.cuda.memory_allocated(target),reserved_parked=torch.cuda.memory_reserved(target))
    atomic_write_json(record,audit_path)
    print(f"Idle frozen graph parked: allocated {record['allocated_before']} -> {record['allocated_parked']} bytes",flush=True)
    try:
        result=action()
        graph.to(target);torch.cuda.synchronize(target)
        if module_state_sha256(graph)!=before:raise RuntimeError('Frozen graph state changed during parking')
    except BaseException as error:
        record.update(status='failed',error=repr(error),finished_at_utc=utc_now());atomic_write_json(record,audit_path)
        raise
    record.update(status='restored',finished_at_utc=utc_now(),allocated_restored=torch.cuda.memory_allocated(target))
    atomic_write_json(record,audit_path)
    return result


def install(audit_root):
    from look_core.pipeline import ExperimentRunner
    from look_core.state import utc_now
    load=ExperimentRunner._load_frozen_graph;prepare=ExperimentRunner._prepare_filler
    def remember(self,*args,**kwargs):
        graph,checkpoint=load(self,*args,**kwargs)
        self._runtime_idle_graph=weakref.ref(graph)
        return graph,checkpoint
    def prepare_with_parking(self,*args,**kwargs):
        graph=getattr(self,'_runtime_idle_graph',lambda:None)()
        if graph is None or self.selection.filling_strategy!='paired_cgan':return prepare(self,*args,**kwargs)
        stamp=utc_now().replace(':','').replace('+','_')
        path=Path(audit_root)/self.experiment_id/f'{stamp}.json'
        return park_while(graph,lambda:prepare(self,*args,**kwargs),path)
    ExperimentRunner._load_frozen_graph=remember
    ExperimentRunner._prepare_filler=prepare_with_parking


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--project-root',type=Path,required=True);p.add_argument('--runtime-audit-root',type=Path,required=True)
    a,remaining=p.parse_known_args()
    import look_core.pipeline as pipeline
    if Path(pipeline.__file__).resolve().parents[2]!=a.project_root.resolve():
        raise RuntimeError('Launcher PYTHONPATH does not identify the requested immutable source')
    install(a.runtime_audit_root)
    entry=a.project_root/'pipeline/38_run_unified_study.py'
    sys.argv=[str(entry),'--project-root',str(a.project_root),*remaining]
    runpy.run_path(str(entry),run_name='__main__')

if __name__=='__main__':main()
