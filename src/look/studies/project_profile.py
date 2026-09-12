"""Isolated complete host-update/save/reload resource verification on real train data."""
import gc
from pathlib import Path
import time
import numpy as np
import torch
from look.models.native_materialization import load_selected
from look.models.native_host import build_native_host,forward_host
from look.models.graph import optimizer_parameter_groups
from look.data.observed_pair import from_parent_specs,collate_observed
from look.runtime.host_checkpoint import save,load,cpu_tree,capture_rng,restore_rng
from look.runtime.state import atomic_write_json,file_sha256,stable_hash
from look.training.observed_host import HostSchedule
from look.studies.project_case import read,validate_spec


def profile(spec,output,device):
    from expanded.native import Inputs
    from mhd_framework.models import create_model
    import psutil
    validate_spec(spec);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    cfg=spec['training'];identity=stable_hash(spec)
    if device.type!='cuda':raise ValueError('Production resource admission requires a GPU')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    total=torch.cuda.get_device_properties(device).total_memory
    budget=min(.875*total,total-10*1024**3)
    if total<78*1024**3:raise ValueError('This production profile requires the requested A10080')
    torch.cuda.set_per_process_memory_fraction((budget-2*1024**3)/1.2/total,device)
    torch.manual_seed(spec['seed']);np.random.seed(spec['seed'])
    torch.cuda.reset_peak_memory_stats(device)
    specs=[read(Path(spec['parents'][k]['path'])/'spec.json') for k in ('first','second')]
    parents=[load_selected(spec['parents'][k]['path'],create_model,device='cpu',allow_inference_equivalence=True) for k in ('first','second')]
    graph=build_native_host(*parents,spec['position'],device=device);del parents;gc.collect()
    train=from_parent_specs(*specs,'train',Inputs,augment=True,seed=spec['seed'])
    dev=from_parent_specs(*specs,'development',Inputs)
    # Maximum valid eye count is checked, not guessed from the first random batch.
    indices=[i for i,n in enumerate(train.counts) if n==2]
    if len(indices)<cfg['microbatch']:raise ValueError('Missing full maximum-eye resource fixture')
    batch=collate_observed([train[i] for i in indices[:cfg['microbatch']]])
    opt=torch.optim.AdamW(optimizer_parameter_groups(graph,cfg['pretrained_lr'],cfg['new_layer_lr']),weight_decay=cfg['weight_decay'])
    schedule=HostSchedule(opt,cfg);opt.zero_grad(set_to_none=True)
    nodes=[(n.id,n.name) for n in sorted(graph.nodes,key=lambda n:n.id)]
    samples=[];process=psutil.Process();start=time.time()
    def update():
        graph.train()
        for _ in range(cfg['effective_batch']//cfg['microbatch']):
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=cfg['precision']=='bf16'):
                forward_host(graph,batch['oct'].to(device),batch['cfp'].to(device),batch['counts'],batch['label'].to(device),loss_scale=cfg['microbatch']/cfg['effective_batch'])
            graph.backward(levels=graph.backward_levels)
        torch.nn.utils.clip_grad_norm_(graph.parameters(),cfg['clip'],error_if_nonfinite=True)
        opt.step();opt.zero_grad(set_to_none=True);torch.cuda.synchronize(device)
        samples.append(dict(time=time.time(),rss=process.memory_info().rss,allocated=torch.cuda.max_memory_allocated(device),reserved=torch.cuda.max_memory_reserved(device),device_used=total-torch.cuda.mem_get_info(device)[0]))
    for _ in range(5):update()
    for _ in range(20):update()
    save(out/'resume.pt',model=graph,optimizer=opt,scheduler=schedule,identity=identity,progress={'updates':25},node_ids=nodes)
    update();expected=cpu_tree(graph.state_dict());expected_optimizer=cpu_tree(opt.state_dict())
    load(out/'resume.pt',model=graph,optimizer=opt,scheduler=schedule,identity=identity,node_ids=nodes)
    update()
    def same(a,b):
        if isinstance(a,torch.Tensor):return torch.equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
        if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        return a==b
    if not same(expected,cpu_tree(graph.state_dict())) or not same(expected_optimizer,cpu_tree(opt.state_dict())):
        raise ValueError('Profile complete optimizer recovery diverged')
    graph.eval()
    with torch.no_grad():
        v=collate_observed([dev[i] for i in range(min(cfg['microbatch'],len(dev)))])
        z=forward_host(graph,v['oct'].to(device),v['cfp'].to(device),v['counts'])
        if z.shape!=(len(v['label']),2) or not torch.isfinite(z).all():raise ValueError('Profile development inference failed')
    peak=max(s['device_used'] for s in samples)
    if peak*1.2+2*1024**3>budget:raise ValueError('Measured peak fails final 10GiB reserve')
    atomic_write_json(samples,out/'samples.json')
    receipt=dict(schema='look_project_resource_v1',status='accepted',case_identity=identity,
        scope='train_development_resource_only',warmups=5,updates=20,max_eyes_per_person=2,
        full_checkpoint_resume_exact=True,development_forward=True,peak_device_bytes=peak,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        host_rss_bytes=max(s['rss'] for s in samples),device=str(torch.cuda.get_device_properties(device)),
        seconds=time.time()-start,test_access=False,files={name:file_sha256(out/name) for name in ('resume.pt','samples.json')})
    atomic_write_json(receipt,out/'accepted.json')
    return receipt
