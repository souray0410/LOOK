"""Resumable complete-input host training, independent of native-model runs."""
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from look.data.observed_pair import collate_observed
from look.models.native_host import forward_host
from look.models.graph import optimizer_parameter_groups
from look.evaluation.evaluator import evaluate_missing, save_prediction_bundle
from look.runtime.host_checkpoint import atomic_save, cpu_tree, save, load
from look.runtime.state import atomic_write_json, file_sha256, stable_hash


DEFAULTS = dict(epochs=100, patience=15, minimum_epochs=8, warmup_epochs=5,
    microbatch=16, effective_batch=128, pretrained_lr=1e-4, new_layer_lr=1e-3,
    weight_decay=1e-4, clip=5.0, precision='fp32', num_workers=0,
    loss='unweighted_cross_entropy', primary_metric='macro_f1')


def validate_config(config):
    if set(config) != set(DEFAULTS):
        raise ValueError('Host training configuration must be complete and versioned')
    if config['loss'] != DEFAULTS['loss'] or config['primary_metric'] != 'macro_f1':
        raise ValueError('Unregistered loss/selection protocol')
    if config['precision'] not in ('fp32','bf16'):
        raise ValueError('Explicit validated precision required')
    if config['effective_batch'] % config['microbatch'] or min(config['microbatch'],config['epochs'],config['patience'])<1:
        raise ValueError('Invalid batch or stop configuration')


class HostSchedule:
    def __init__(self, optimizer, config):
        self.optimizer=optimizer;self.config=config
        self.base=[group['lr'] for group in optimizer.param_groups]
        self.best=-float('inf');self.best_epoch=0;self.stall=0
    def begin(self, epoch):
        warm=self.config['warmup_epochs'];limit=self.config['epochs']
        scale=epoch/max(1,warm) if epoch<=warm else .5*(1+math.cos(math.pi*(epoch-warm)/max(1,limit-warm)))
        for group,base in zip(self.optimizer.param_groups,self.base):group['lr']=base*scale
    def step(self, score, epoch):
        if not math.isfinite(score):raise ValueError('Non-finite development selection score')
        improved=score>self.best
        if improved:self.best=score;self.best_epoch=epoch;self.stall=0
        else:self.stall+=1
        return improved
    def state_dict(self):
        return dict(best=self.best,best_epoch=self.best_epoch,stall=self.stall,base=self.base,config=self.config)
    def load_state_dict(self,s):
        if s['config'] != self.config:raise ValueError('Host schedule changed on resume')
        self.best,self.best_epoch,self.stall,self.base=s['best'],s['best_epoch'],s['stall'],s['base']


def train_host(graph, train, development, config, seed, output, identity, device, should_pause=lambda:False, *, preflight_updates=None):
    validate_config(config)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if train.split != 'train' or development.split != 'development' or set(train.participant_ids)&set(development.participant_ids):
        raise ValueError('Host split provenance or disjointness failed')
    opt=torch.optim.AdamW(optimizer_parameter_groups(graph,config['pretrained_lr'],config['new_layer_lr']),weight_decay=config['weight_decay'])
    ids=[(n.id,n.name) for n in sorted(graph.nodes,key=lambda n:n.id)]
    schedule=HostSchedule(opt,config);opt.zero_grad(set_to_none=True)
    progress=dict(epoch=1,offset=0,updates=0,history=[],epoch_loss=0.,epoch_seen=0,seconds=0.)
    if (out/'last.pt').exists():
        progress=load(out/'last.pt',model=graph,optimizer=opt,scheduler=schedule,identity=identity,node_ids=ids)
    t0=time.monotonic()
    def checkpoint():
        nonlocal t0
        now=time.monotonic();progress['seconds']+=now-t0;t0=now
        save(out/'last.pt',model=graph,optimizer=opt,scheduler=schedule,identity=identity,progress=progress,node_ids=ids)
    def status(state):
        atomic_write_json(dict(state=state,identity=identity,epoch=progress['epoch'],offset=progress['offset'],
            updates=progress['updates'],updated_at=time.time(),test_access=False),out/'status.json')
    dev_loader=DataLoader(development,batch_size=config['microbatch'],collate_fn=collate_observed,shuffle=False,num_workers=config['num_workers'],generator=torch.Generator().manual_seed(seed))
    if not (out/'best.pt').exists():
        result=evaluate_missing(graph,dev_loader,device,fixed_pattern='complete')
        schedule.step(result['metrics']['macro_f1'],0)
        atomic_save(out/'best.pt',dict(identity=identity,epoch=0,model=cpu_tree(graph.state_dict()),node_ids=ids))
        save_prediction_bundle(result,out/'development_predictions.npz')
        checkpoint()
    launch_updates=0
    while progress['epoch'] <= config['epochs']:
        if schedule.stall>=config['patience'] and progress['epoch']>config['minimum_epochs']:break
        epoch=progress['epoch'];train.set_epoch(epoch);schedule.begin(epoch);graph.train()
        order=torch.randperm(len(train),generator=torch.Generator().manual_seed(seed+epoch)).tolist()
        while progress['offset']<len(order):
            if should_pause():
                checkpoint();status('paused');return dict(state='paused')
            block=order[progress['offset']:progress['offset']+config['effective_batch']]
            loader=DataLoader(Subset(train,block),batch_size=config['microbatch'],collate_fn=collate_observed,
                shuffle=False,num_workers=config['num_workers'],generator=torch.Generator().manual_seed(seed+epoch))
            for batch in loader:
                n=len(batch['label'])
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=config['precision']=='bf16'):
                    forward_host(graph,batch['oct'].to(device),batch['cfp'].to(device),batch['counts'],batch['label'].to(device),loss_scale=n/len(block))
                    loss=graph.get_node_by_name('loss').feature_message.current_state
                if not torch.isfinite(loss):raise ValueError('Non-finite host loss')
                graph.backward(levels=graph.backward_levels)
                progress['epoch_loss']+=float(loss.detach())*len(block);progress['epoch_seen']+=n
            torch.nn.utils.clip_grad_norm_(graph.parameters(),config['clip'],error_if_nonfinite=True)
            opt.step();opt.zero_grad(set_to_none=True)
            progress['offset']+=len(block);progress['updates']+=1;launch_updates+=1
            status('training')
            if should_pause() or (preflight_updates is not None and launch_updates>=preflight_updates):
                checkpoint();status('paused');return dict(state='paused',updates=launch_updates)
        status('validating')
        result=evaluate_missing(graph,dev_loader,device,fixed_pattern='complete')
        score=result['metrics']['macro_f1']
        if schedule.step(score,epoch):
            atomic_save(out/'best.pt',dict(identity=identity,epoch=epoch,model=cpu_tree(graph.state_dict()),node_ids=ids))
            save_prediction_bundle(result,out/'development_predictions.npz')
        progress['history'].append(dict(epoch=epoch,loss=progress['epoch_loss']/progress['epoch_seen'],metrics=result['metrics']))
        progress.update(epoch=epoch+1,offset=0,epoch_loss=0.,epoch_seen=0)
        atomic_write_json(progress['history'],out/'history.json');checkpoint()
    if schedule.stall<config['patience']:
        status('needs_review_epoch_cap');return dict(state='needs_review_epoch_cap')
    selected=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    if selected['identity']!=identity or selected['node_ids']!=ids:raise ValueError('Selected host identity changed')
    graph.load_state_dict(selected['model'],strict=True);graph.eval()
    replay=evaluate_missing(graph,dev_loader,device,fixed_pattern='complete')
    saved=np.load(out/'development_predictions.npz',allow_pickle=False)
    if not np.array_equal(saved['participant_ids'].astype(str),replay['participant_ids'].astype(str)) or not np.array_equal(saved['labels'],replay['labels']):
        raise ValueError('Host selected replay identity mismatch')
    if not np.allclose(saved['probabilities'],replay['probabilities'],rtol=1e-4,atol=1e-5) or not np.array_equal(saved['logits'].argmax(1),replay['logits'].argmax(1)):
        raise ValueError('Host selected replay numeric mismatch')
    receipt=dict(schema='look_observed_host_v1',state='accepted',identity=identity,best_epoch=schedule.best_epoch,
        stop_epoch=progress['epoch']-1,plateau=True,selection='development_macro_f1',test_access=False,
        metrics=replay['metrics'],seconds=progress['seconds'],source_nodes=graph.native_host_provenance,
        files={name:file_sha256(out/name) for name in ('best.pt','last.pt','history.json','development_predictions.npz')})
    atomic_write_json(receipt,out/'accepted.json');status('completed')
    return receipt
