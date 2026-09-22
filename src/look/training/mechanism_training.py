"""Matched continuation/student training with private missingness RNG and resume."""
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from look.training.observed_host import HostSchedule, validate_config
from look.runtime.host_checkpoint import atomic_save, cpu_tree, save, load, save_selected, read_selected
from look.runtime.state import atomic_write_json, file_sha256
from look.data.observed_pair import collate_observed
from look.data.mechanism_samples import mask_training_batch
from look.models.native_host import forward_host
from look.models.graph import optimizer_parameter_groups
from look.methods.operator import forward_with_look
from look.evaluation.evaluator import evaluate_missing, save_prediction_bundle
from look.evaluation.stability import logit_metrics, probabilities_from_logits


def state_equal(model, state):
    if set(model.state_dict()) != set(state) or any(not torch.equal(v.detach().cpu(), state[k]) for k,v in model.state_dict().items()):
        raise ValueError('Frozen model/BN changed')


def distillation_loss(student, teacher, labels):
    if teacher.requires_grad:
        raise ValueError('Teacher logits must be detached')
    return F.cross_entropy(student, labels) + 4*F.kl_div(
        F.log_softmax(student/2, -1), F.softmax(teacher/2, -1), reduction='batchmean')


@torch.no_grad()
def evaluate_student(model, loader, device, track):
    model.eval(); zs, ys, ids = [], [], []
    for b in loader:
        zs.append(model(b[track].to(device), b['counts']).cpu().numpy())
        ys.extend(b['label'].tolist()); ids.extend(b['participant_id'])
    z=np.concatenate(zs).astype(np.float64); y=np.asarray(ys)
    pattern='oct_missing' if track=='cfp' else 'cfp_missing'
    return dict(logits=z, labels=y, participant_ids=np.asarray(ids),
        probabilities=probabilities_from_logits(z), scores=z[:,1]-z[:,0],
        patterns=np.asarray([pattern]*len(y)), metrics=logit_metrics(y,z))


def train(model, train_data, dev_data, config, seed, output, identity, device,
          arm, track=None, teacher=None, should_pause=lambda:False, preflight_updates=None):
    validate_config(config)
    if arm not in ('continue_complete','continue_missing','ce','distill'):
        raise ValueError('Unknown training arm')
    if train_data.split!='train' or dev_data.split!='development' or set(train_data.participant_ids)&set(dev_data.participant_ids):
        raise ValueError('Training/development provenance failed')
    student=arm in ('ce','distill')
    if student and track not in ('cfp','oct'): raise ValueError('Student track required')
    if (arm=='distill') != (teacher is not None): raise ValueError('Teacher contract mismatch')
    if teacher is not None:
        teacher.eval()
        for p in teacher.parameters(): p.requires_grad_(False)
        teacher_state=cpu_tree(teacher.state_dict())
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    if student:
        head={id(p) for e in model.graph.edges if e.name=='head' or e.name=='logits' for _,p in e.named_edge_parameters()}
        # Native definitions can expose the classifier under another exact edge ID.
        endpoint=model.graph.endpoint_nodes['logits']
        eid=next(e for e,inputs,o in model.graph._definitions if o==endpoint)
        head.update(id(p) for _,p in model.graph.get_edge_by_id(eid).named_edge_parameters())
        groups=[dict(params=[p for p in model.parameters() if id(p) not in head],lr=config['pretrained_lr']),
                dict(params=[p for p in model.parameters() if id(p) in head],lr=config['new_layer_lr'])]
        graph=model.graph
    else:
        groups=optimizer_parameter_groups(model,config['pretrained_lr'],config['new_layer_lr']);graph=model
    owned=[id(p) for g in groups for p in g['params']]
    if len(owned)!=len(set(owned)) or set(owned)!={id(p) for p in model.parameters()}:
        raise ValueError('Optimizer parameters omitted or duplicated')
    opt=torch.optim.AdamW(groups,weight_decay=config['weight_decay']); opt.zero_grad(set_to_none=True)
    schedule=HostSchedule(opt,config)
    ids=[(n.id,n.name) for n in sorted(graph.nodes,key=lambda n:n.id)]
    mask_rng=torch.Generator().manual_seed(seed+910731)
    progress=dict(epoch=1,offset=0,updates=0,history=[],epoch_loss=0.,epoch_seen=0,seconds=0.,mask_rng=mask_rng.get_state())
    if (out/'last.pt').exists():
        progress=load(out/'last.pt',model=model,optimizer=opt,scheduler=schedule,identity=identity,node_ids=ids)
        mask_rng.set_state(progress['mask_rng'])
    clock=time.monotonic()
    def checkpoint():
        nonlocal clock
        now=time.monotonic(); progress['seconds']+=now-clock; clock=now
        progress['mask_rng']=mask_rng.get_state()
        save(out/'last.pt',model=model,optimizer=opt,scheduler=schedule,identity=identity,progress=progress,node_ids=ids)
    def status(state):
        atomic_write_json(dict(state=state,arm=arm,epoch=progress['epoch'],offset=progress['offset'],
            updates=progress['updates'],identity=identity,time=time.time(),test_access=False),out/'status.json')
    dev_loader=DataLoader(dev_data,batch_size=config['microbatch'],collate_fn=collate_observed,
        shuffle=False,num_workers=config['num_workers'],generator=torch.Generator().manual_seed(seed))
    def evaluate():
        if student:
            r=evaluate_student(model,dev_loader,device,track)
            return r['metrics']['macro_f1'], {'student':r}
        values={p:evaluate_missing(model,dev_loader,device,fixed_pattern=p) for p in ('complete','cfp_missing','oct_missing')}
        return float(np.mean([v['metrics']['macro_f1'] for v in values.values()])),values
    def select(epoch,score,values):
        if schedule.step(score,epoch):
            save_selected(out/'best.pt',identity=identity,epoch=epoch,model=model,node_ids=ids)
            for key,r in values.items(): save_prediction_bundle(r,out/(key+'.npz'))
    if not (out/'best.pt').exists():
        score,values=evaluate();select(0,score,values);checkpoint()
    launch=0
    while progress['epoch']<=config['epochs']:
        if schedule.stall>=config['patience'] and progress['epoch']>config['minimum_epochs']:break
        epoch=progress['epoch'];train_data.set_epoch(epoch);schedule.begin(epoch);model.train()
        order=torch.randperm(len(train_data),generator=torch.Generator().manual_seed(seed+epoch)).tolist()
        while progress['offset']<len(order):
            if should_pause(): checkpoint();status('paused');return {'state':'paused'}
            block=order[progress['offset']:progress['offset']+config['effective_batch']]
            loader=DataLoader(Subset(train_data,block),batch_size=config['microbatch'],collate_fn=collate_observed,
                shuffle=False,num_workers=config['num_workers'],generator=torch.Generator().manual_seed(seed+epoch))
            for b in loader:
                if arm=='continue_missing': b,_=mask_training_batch(b,mask_rng)
                n=len(b['label']); labels=b['label'].to(device)
                if student:
                    logits=model(b[track].to(device),b['counts'])
                    if teacher is not None:
                        with torch.no_grad(): target=forward_with_look(teacher,b['oct'].to(device),b['cfp'].to(device),counts=b['counts'])
                        loss=distillation_loss(logits,target,labels)
                    else: loss=F.cross_entropy(logits,labels)
                    if not torch.isfinite(loss): raise ValueError('Nonfinite student loss')
                    (loss*n/len(block)).backward()
                    progress['epoch_loss']+=float(loss.detach())*n
                else:
                    forward_host(model,b['oct'].to(device),b['cfp'].to(device),b['counts'],labels,loss_scale=n/len(block))
                    loss=model.get_node_by_name('loss').feature_message.current_state
                    if not torch.isfinite(loss): raise ValueError('Nonfinite host loss')
                    model.backward(levels=model.backward_levels)
                    progress['epoch_loss']+=float(loss.detach())*len(block)
                progress['epoch_seen']+=n
            torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip'],error_if_nonfinite=True)
            opt.step();opt.zero_grad(set_to_none=True)
            progress['offset']+=len(block);progress['updates']+=1;launch+=1;status('training')
            if should_pause() or (preflight_updates is not None and launch>=preflight_updates):
                checkpoint();status('paused');return dict(state='paused',updates=launch)
        status('validating');score,values=evaluate();select(epoch,score,values)
        if teacher is not None: state_equal(teacher,teacher_state)
        progress['history'].append(dict(epoch=epoch,selection_score=score,
            loss=progress['epoch_loss']/progress['epoch_seen'],metrics={k:r['metrics'] for k,r in values.items()}))
        progress.update(epoch=epoch+1,offset=0,epoch_loss=0.,epoch_seen=0)
        atomic_write_json(progress['history'],out/'history.json');checkpoint()
    if schedule.stall<config['patience']:
        status('needs_review_epoch_cap');return dict(state='needs_review_epoch_cap')
    best=read_selected(out/'best.pt',identity=identity,node_ids=ids)
    if best['identity']!=identity or best['node_ids']!=ids: raise ValueError('Selected model provenance changed')
    model.load_state_dict(best['model'],strict=True);score,values=evaluate()
    for key,r in values.items():
        with np.load(out/(key+'.npz'),allow_pickle=False) as old:
            if not np.array_equal(old['participant_ids'],r['participant_ids']) or not np.array_equal(old['labels'],r['labels']):
                raise ValueError('Selected prediction identity mismatch')
            if not np.allclose(old['probabilities'],r['probabilities'],rtol=1e-4,atol=1e-5) or not np.array_equal(old['logits'].argmax(1),r['logits'].argmax(1)):
                raise ValueError('Selected model replay failed')
    receipt=dict(schema='look_mechanism_training_v1',state='accepted',identity=identity,
        plateau=True,best_epoch=best['epoch'],stop_epoch=progress['epoch']-1,seconds=progress['seconds'],
        selection='student_dev_macro_f1' if student else 'equal_mean_three_dev_states',test_access=False,
        files={p.name:file_sha256(p) for p in out.iterdir() if p.name in ('best.pt','last.pt','history.json') or p.suffix=='.npz'})
    atomic_write_json(receipt,out/'accepted.json');status('completed');return receipt
