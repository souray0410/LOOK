"""Actual-parent GPU resource admission in an isolated, non-scientific attempt."""
from pathlib import Path
import os
import shutil
import torch
from torch.nn import functional as F
from look.studies.mechanism_case import context,loader
from look.studies.mechanism_protocol import TRAINING
from look.data.observed_pair import collate_observed
from look.data.mechanism_samples import mask_training_batch
from look.models.graph import optimizer_parameter_groups
from look.models.native_host import forward_host
from look.methods.operator import forward_with_look,load_selected_bank
from look.evaluation.evaluator import evaluate_missing
from look.evaluation.mechanism_diagnostics import resources
from look.runtime.host_checkpoint import atomic_save,cpu_tree,capture_rng,restore_rng
from look.runtime.state import atomic_write_json,stable_hash
from look.training.mechanism_training import distillation_loss,state_equal


def profile(spec,out,device):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    task=spec['task']
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise ValueError('Production admission needs an actual GPU')
    props=torch.cuda.get_device_properties(device);total=props.total_memory
    budget=min(.875*total,total-10*1024**3)
    free,_=torch.cuda.mem_get_info(device)
    # No optimistic concurrent probe: this workflow owns an otherwise empty card.
    existing=max(0,total-free-torch.cuda.memory_reserved(device))
    if existing>2*1024**3:raise ValueError('Unknown co-resident GPU work; keep old work, defer resource probe')
    torch.cuda.set_per_process_memory_fraction(max(.01,(budget-existing-2*1024**3)/total))
    torch.cuda.reset_peak_memory_stats(device)
    base,parents,graph,data=context(spec,device)
    ds=data['train'];counts=ds.counts
    indices=sorted(range(len(ds)),key=lambda i:(-counts[i],str(ds.participant_ids[i])))[:128]
    if len(indices)!=128:raise ValueError('Full effective-batch resource sample unavailable')
    batches=[collate_observed([ds[i] for i in indices[j:j+16]]) for j in range(0,128,16)]
    dev_batches=[collate_observed([data['development'][i] for i in range(j,j+16)]) for j in (0,16)]
    # CPU storage/feature-fit scaling is recorded, not inferred from 128 samples.
    fit_rows=sum(counts);host_state=cpu_tree(graph.state_dict())
    teacher=None;student=task['kind']=='student_training';training=task['kind'].endswith('_training')
    if student:
        model=parents[0 if task['track']=='cfp' else 1].to(device)
        teacher=graph if task['arm']=='distill' else None
        if teacher is None:graph.cpu()
        else:
            teacher.eval()
            for p in teacher.parameters():p.requires_grad_(False)
        last=model.graph.endpoint_nodes['logits']
        eid=next(e for e,inputs,o in model.graph._definitions if o==last)
        heads={id(p) for _,p in model.graph.get_edge_by_id(eid).named_edge_parameters()}
        groups=[dict(params=[p for p in model.parameters() if id(p) not in heads],lr=1e-5),
                dict(params=[p for p in model.parameters() if id(p) in heads],lr=1e-4)]
    else:model=graph;groups=optimizer_parameter_groups(graph,1e-5,1e-4)
    rng=torch.Generator().manual_seed(spec['task']['host']['seed']+910731)
    if training:
        optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
        def update():
            model.train();optimizer.zero_grad(set_to_none=True)
            for original in batches:
                b,_=mask_training_batch(original,rng) if task['arm']=='continue_missing' else (original,None)
                if student:
                    logits=model(b[task['track']].to(device),b['counts']);labels=b['label'].to(device)
                    if teacher is not None:
                        with torch.no_grad():target=forward_with_look(teacher,b['oct'].to(device),b['cfp'].to(device),counts=b['counts'])
                        loss=distillation_loss(logits,target,labels)
                    else:loss=F.cross_entropy(logits,labels)
                    (loss/8).backward()
                else:
                    forward_host(graph,b['oct'].to(device),b['cfp'].to(device),b['counts'],b['label'].to(device),loss_scale=1/8)
                    graph.backward(levels=graph.backward_levels)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
        for _ in range(25):update()
        saved=dict(model=cpu_tree(model.state_dict()),optimizer=cpu_tree(optimizer.state_dict()),
                   rng=capture_rng(),mask_rng=rng.get_state())
        atomic_save(out/'resume.pt',saved);update();expected=cpu_tree(model.state_dict())
        restored=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(restored['model']);optimizer.load_state_dict(restored['optimizer']);restore_rng(restored['rng']);rng.set_state(restored['mask_rng'])
        update();state_equal(model,expected)
        model.eval()
        with torch.no_grad():
            for b in dev_batches:
                if student:model(b[task['track']].to(device),b['counts'])
                else:
                    for pattern in ('complete','oct_missing','cfp_missing'):
                        from look.methods.imputation import NormalizedMeanFiller
                        o,c=NormalizedMeanFiller().fill(b['oct'].to(device),b['cfp'].to(device),pattern)
                        forward_with_look(graph,o,c,counts=b['counts'])
        if teacher is not None:state_equal(teacher,host_state)
    else:
        graph.eval()
        for p in graph.parameters():p.requires_grad_(False)
        source=Path(spec['source']['run_dir'])/'corrections/look'/task['pattern']
        artifacts=load_selected_bank(source)
        from look.methods.mechanism_operator import MechanismArtifact,hidden_width
        if task['arm']=='mlp':
            converted=[]
            for a in artifacts:
                h=hidden_width(a.latent_dim)
                module=torch.nn.Sequential(torch.nn.Linear(a.latent_dim,h),torch.nn.ReLU(),torch.nn.Linear(h,a.latent_dim))
                # Exercise learned nonlinear execution without altering the saved parent.
                converted.append(MechanismArtifact(a,mlp_state=module.state_dict(),hidden=h))
            artifacts=converted
        # Exercise fitting and reload on a train-only probe, then the selected
        # original geometries on maximum-eye batches. Probe artifacts are never
        # promoted to scientific results.
        from torch.utils.data import Subset,DataLoader
        from look.methods.mechanism_operator import fit_variant
        fit_probe=Subset(data['fit'],indices)
        fit_probe.participant_ids=[data['fit'].participant_ids[i] for i in indices]
        fit_probe.counts=[counts[i] for i in indices]
        fit_probe.split='train';fit_probe.augment=False
        probe_loader=DataLoader(fit_probe,batch_size=16,collate_fn=collate_observed,shuffle=False)
        if artifacts:
            bases=load_selected_bank(source)
            arm=task['arm'] if task['kind']=='correction' else 'sequential'
            fitted,_=fit_variant(graph,probe_loader,bases,arm,device,7341618+task.get('repeat',0),out/'fit_probe')
            reloaded,_=fit_variant(graph,probe_loader,bases,arm,device,7341618+task.get('repeat',0),out/'fit_probe')
            b=batches[0]
            from look.methods.imputation import NormalizedMeanFiller
            o,c=NormalizedMeanFiller().fill(b['oct'].to(device),b['cfp'].to(device),task['pattern'])
            with torch.no_grad():
                torch.testing.assert_close(forward_with_look(graph,o,c,fitted,counts=b['counts']),
                    forward_with_look(graph,o,c,reloaded,counts=b['counts']),rtol=0,atol=0)
        from look.methods.imputation import NormalizedMeanFiller
        with torch.no_grad():
            for b in batches:
                o,c=NormalizedMeanFiller().fill(b['oct'].to(device),b['cfp'].to(device),task['pattern'])
                forward_with_look(graph,o,c,artifacts,counts=b['counts'])
        state_equal(graph,host_state)
        atomic_save(out/'fixed_operators.pt',[a.record() if hasattr(a,'record') else a for a in artifacts])
    import psutil
    report=resources(device);peak=torch.cuda.max_memory_reserved(device)
    # Latent full/missing pairs plus diagnostic workspaces and streamed PCA;
    # this is an explicit conservative CPU estimate, not a measured full fit.
    estimated_latent_bytes=fit_rows*512*4*8
    host_rss=psutil.Process().memory_info().rss
    memory_limit=int(os.environ.get('LOOK_WORKER_MEMORY_BYTES',100*1024**3))
    if host_rss+estimated_latent_bytes>.85*memory_limit:raise ValueError('Projected full-fit host memory exceeds worker budget')
    if existing+1.2*peak+2*1024**3>budget:raise ValueError('GPU full-peak reserve failed')
    if shutil.disk_usage(out).free<100*1024**3:raise ValueError('Less than 100 GiB storage reserve')
    report.update(status='accepted',case_identity=stable_hash(spec),training_updates=25 if training else 0,
        exact_next_update_reload=training,maximum_observed_eyes_per_batch=max(sum(b['counts']) for b in batches),
        estimated_full_latent_bytes=estimated_latent_bytes,host_memory_measurement='probe_RSS_plus_conservative_full_fit_estimate',
        effective_batch=128,microbatch=16,budget_bytes=budget,existing_gpu_bytes=existing,
        scope='actual_parent_non_scientific_resource_probe; not_a_completed_experiment',
        development_forward=True,fit_probe_participants=128 if not training else None)
    atomic_write_json(report,out/'accepted.json')
    return report
