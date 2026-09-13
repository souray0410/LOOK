"""Read-only train reconstruction diagnostics and controlled inference timing."""
import os
import resource
import subprocess
import sys
import time
import numpy as np
import torch
from look.runtime.host_checkpoint import capture_rng,restore_rng,cpu_tree
from look.methods.operator import forward_with_look,apply_artifact
from look.methods.imputation import NormalizedMeanFiller
from look.training.mechanism_training import state_equal


@torch.no_grad()
def feature_diagnostics(graph,loader,artifacts,pattern,device):
    modes={m:m.training for m in graph.modules()};graph.eval();rng=capture_rng();state=cpu_tree(graph.state_dict());rows=[];upstream=[]
    try:
        for a in artifacts:
            sums=dict(input_ss=0.,residual_ss=0.,full_ss=0.,before_error_ss=0.,after_error_ss=0.,elements=0)
            for b in loader:
                oct_x,cfp=b['oct'].to(device),b['cfp'].to(device)
                full=forward_with_look(graph,oct_x,cfp,stop_node=a.node_name,counts=b['counts']).detach()
                o,c=NormalizedMeanFiller().fill(oct_x,cfp,pattern)
                x=forward_with_look(graph,o,c,upstream,stop_node=a.node_name,counts=b['counts']).detach()
                corrected=apply_artifact(x,a);delta=corrected-x
                for key,value in (('input_ss',x),('residual_ss',delta),('full_ss',full),
                                  ('before_error_ss',x-full),('after_error_ss',corrected-full)):
                    sums[key]+=float(value.double().square().sum())
                sums['elements']+=x.numel()
            rows.append(dict(node=a.node_name,**sums,
                residual_ratio=(sums['residual_ss']/sums['input_ss'])**.5 if sums['input_ss'] else None,
                before_relative_error=(sums['before_error_ss']/sums['full_ss'])**.5 if sums['full_ss'] else None,
                after_relative_error=(sums['after_error_ss']/sums['full_ss'])**.5 if sums['full_ss'] else None,
                definition='ratio_of_full_train_sum_squared_norms; writeback_before_downstream_forward'))
            upstream.append(a)
        state_equal(graph,state)
        return rows
    finally:
        restore_rng(rng)
        for m,mode in modes.items():m.training=mode


def latency(forward,device):
    rng=capture_rng();values=[]
    try:
        with torch.no_grad():
            for _ in range(10):forward()
            if device.type=='cuda':torch.cuda.synchronize(device)
            for _ in range(50):
                start=time.perf_counter();forward()
                if device.type=='cuda':torch.cuda.synchronize(device)
                values.append(time.perf_counter()-start)
        others=None
        if device.type=='cuda':
            try:
                text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True,timeout=10)
                others=[line for line in text.splitlines() if line.split(',')[0].strip()!=str(os.getpid())]
            except (OSError,subprocess.SubprocessError):pass
        return dict(warmup=10,repeats=50,median_seconds=float(np.median(values)),
            q25_seconds=float(np.quantile(values,.25)),q75_seconds=float(np.quantile(values,.75)),
            other_gpu_processes=others,interference='unknown' if others is None else 'present' if others else 'not_observed',
            scope='fixed_train_batch; synchronized_forward_only; no_unqualified_speed_claim')
    finally:restore_rng(rng)


def resources(device):
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    from importlib.metadata import version,PackageNotFoundError
    versions={}
    for package in ('numpy','scipy','scikit-learn','torchvision','mhd-framework'):
        try:versions[package]=version(package)
        except PackageNotFoundError:versions[package]='source_binding_required'
    return dict(python_version=sys.version,packages=versions,peak_rss_bytes=int(rss if sys.platform=='darwin' else rss*1024),
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,
        torch_version=torch.__version__,compiled_cuda=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
        precision='fp32',tf32=False,world_size=1)
