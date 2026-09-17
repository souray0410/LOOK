"""Train-only, frozen real-host oracle for shared extraction; not research results."""
import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import time
import psutil
import torch
from torch.utils.data import DataLoader
from look.methods import operator as op
from look.methods.shared_latent import SharedLatentFitter
from look.methods.linear_operator import fingerprint
from look.studies.search_case import dependencies, load_basis, TrainProbe
from look.studies.mechanism_case import load_original
from look.data.observed_pair import collate_observed
from look.runtime.state import atomic_write_json, file_sha256, stable_hash
from look.training.mechanism_training import state_equal
from look.runtime.host_checkpoint import cpu_tree


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    assert os.environ.get('SLURM_JOB_ID'),'Existing admitted Slurm allocation required'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(device).total_memory)
    spec=json.loads(Path(a.spec).read_text());base,source,manifest=dependencies(spec)
    _,parents,g,data=load_original(base,source,device);del parents;gc.collect()
    g.eval()
    for param in g.parameters():param.requires_grad_(False)
    frozen=cpu_tree(g.state_dict());nodes=[(n.id,n.name) for n in g.nodes]
    sites=spec['eligible_sites'];bank=load_basis(spec,manifest)
    bases={n:next(b for (name,f),b in bank.items() if name==n and f==(1 if len(b.feature_shape)==1 else 16)) for n in sites}
    def guard():
        if psutil.Process().memory_info().rss>6*1024**3*.85:raise MemoryError('Probe RAM limit')
        if torch.cuda.mem_get_info()[0]<10*1024**3:raise MemoryError('Whole GPU reserve')
        return False
    loader=DataLoader(TrainProbe(data['fit'],range(32)),batch_size=16,shuffle=False,num_workers=0,collate_fn=collate_observed)
    # Warmup uses only train and does not update this frozen host.
    batch=next(iter(loader));op.forward_with_look(g,batch['oct'].to(device),batch['cfp'].to(device),counts=batch['counts'])
    records=[];upstream=[]
    for index,(pattern,targets) in enumerate([('oct_missing',sites),('cfp_missing',sites),('oct_missing',sites[1:])]):
        oracle={};torch.cuda.synchronize();t=time.monotonic()
        for n in targets:
            guard();b=bases[n];stats=op.LatentSufficientStatistics(32)
            for full,missing,_,_ in op.iter_feature_pairs(g,loader,n,pattern,b.factor,device,upstream):
                proj=lambda x:(((x-b.mean)/b.std)-b.pca_mean)@b.components[:32].T
                stats.update(proj(missing),proj(full))
            oracle[n]=stats
        torch.cuda.synchronize();old_s=time.monotonic()-t
        fitter=SharedLatentFitter(g,loader,bases,32,device,out/f'moments{index}',out/'reference',dict(spec=spec['source'],n=32),guard)
        t=time.monotonic();actual=fitter.statistics(pattern,upstream,targets);torch.cuda.synchronize();new_s=time.monotonic()-t
        artifacts={}
        for n in targets:
            assert fingerprint(vars(oracle[n]))==fingerprint(vars(actual[n])),('moments',n)
            kw=dict(graph=g,loader=loader,node_name=n,missing_pattern=pattern,factor=bases[n].factor,
                    latent_dims=[32],max_rank=32,device=device,pca=bases[n],upstream_artifacts=upstream)
            left=op.fit_look_node(**kw,latent_statistics=oracle[n])[32]
            right=op.fit_look_node(**kw,latent_statistics=actual[n])[32]
            assert fingerprint(asdict(left))==fingerprint(asdict(right)),('solution',n)
            artifacts[n]=right
            args=(g,batch['oct'].to(device),batch['cfp'].to(device))
            x=op.forward_with_look(*args,artifacts=upstream+[left],counts=batch['counts']).detach().clone()
            y=op.forward_with_look(*args,artifacts=upstream+[right],counts=batch['counts']).detach().clone()
            assert torch.equal(x,y),('logits',n)
        records.append(dict(pattern=pattern,upstream=[x.node_name for x in upstream],sites=len(targets),
          single_site_seconds=old_s,shared_seconds=new_s,metrics=fitter.metrics,exact_moments=True,exact_parameters=True,exact_logits=True))
        if index==1:
            # Refit the next pass under a real nonzero correction; complete references remain original.
            upstream=[artifacts[sites[0]]]
    state_equal(g,frozen);assert nodes==[(n.id,n.name) for n in g.nodes]
    atomic_write_json(dict(state='accepted',scope='32_train_participants_real_host_operator_oracle_not_formal_performance',
        records=records,test_access=False,frozen=True,source_identity=stable_hash(spec),
        tested_source_sha256={str(p.relative_to(Path(__file__).parents[2])):file_sha256(p) for p in
            (Path(__file__).parents[2]/'src/look').rglob('*.py')},gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        rss_bytes=psutil.Process().memory_info().rss,concurrent_load=True),out/'accepted.json')
    print(json.dumps(records),flush=True)

if __name__=='__main__':
    with torch.no_grad():main()
