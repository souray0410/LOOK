"""Published scale-and-shift idea adapted to the frozen LOOK graph and data.

Reference: Reza et al., arXiv:2310.03986v6, sections 3.2 and S2/S3.
This is a matched-task implementation, not the authors' original benchmark.
"""
from __future__ import annotations
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn

from .graph import reset_and_forward
from .look import forward_with_look
from .method_kernels import evaluate_control
from .filling import NormalizedMeanFiller
from .reproducibility import seed_everything
from .distributed import module_state_sha256
from .state import atomic_write_json, durable_replace, stable_hash, file_sha256


class ScaleShift(nn.Module):
    def __init__(self, width, axis):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))
        self.shift = nn.Parameter(torch.zeros(width))
        self.axis = axis

    def forward(self, value):
        shape = [1]*value.ndim; shape[self.axis] = len(self.scale)
        return value*self.scale.reshape(shape) + self.shift.reshape(shape)


class SSFAdapter(nn.Module):
    """Hooks transform active encoder/fusion operations; prediction head stays fixed."""
    def __init__(self, graph, pattern):
        super().__init__()
        if pattern not in ('oct_missing', 'cfp_missing'): raise ValueError(pattern)
        retained = 'cfp' if pattern == 'oct_missing' else 'oct'
        self.pattern, self.enabled, self.handles, self.slots = pattern, False, [], []
        self.layers = nn.ModuleDict()
        for edge in sorted(graph.edges, key=lambda e:e.id):
            if not (edge.name.startswith(retained+'_') or edge.name.startswith('fusion_') or edge.name.startswith('fuse_')):
                continue
            if 'classifier' in edge.name: continue
            for oi, operation in enumerate(edge.edge_operations):
                if not isinstance(operation.function, nn.Module): continue
                for name, module in operation.function.named_modules():
                    width, axis = None, 1
                    if isinstance(module, nn.Conv2d): width = module.out_channels
                    elif isinstance(module, nn.BatchNorm2d): width = module.num_features
                    elif isinstance(module, nn.Linear): width, axis = module.out_features, -1
                    elif isinstance(module, nn.LayerNorm) and len(module.normalized_shape) == 1:
                        width, axis = module.normalized_shape[0], -1
                    if width is None: continue
                    key = f'slot_{len(self.slots):03d}'
                    self.layers[key] = ScaleShift(width, axis).to(graph.device)
                    self.slots.append(dict(key=key, edge=edge.name, operation=oi, module=name,
                                           type=type(module).__name__, width=width, axis=axis))
                    def hook(_module, _inputs, output, key=key):
                        return self.layers[key](output) if self.enabled else output
                    self.handles.append(module.register_forward_hook(hook))
        if not self.slots: raise ValueError('No eligible SSF modules')

    def close(self):
        for handle in self.handles: handle.remove()
        self.handles.clear(); self.enabled = False


def save_torch(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.partial')
    torch.save(value, temp); durable_replace(temp, path)


def train_ssf(graph, train, validation, device, pattern, spec, output, source_hash, seed):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    frozen_hash = module_state_sha256(graph)
    if any(p.requires_grad for p in graph.parameters()):
        raise ValueError('SSF requires a frozen backbone and prediction head')
    filler = NormalizedMeanFiller()
    off = evaluate_control(graph, validation, device, filler=filler, fixed_pattern=pattern)
    off_score = off['metrics']['macro_f1']
    candidates = []
    for li, lr in enumerate(spec['learning_rates']):
        root = output/f'lr_{li:02d}'
        root.mkdir(parents=True, exist_ok=True)
        identity = dict(method='ssf', spec=spec, lr=lr, pattern=pattern, source=source_hash, seed=seed)
        path = root/'identity.json'
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError('SSF resume identity mismatch')
        atomic_write_json(identity, path)
        seed_everything(seed)
        adapter = SSFAdapter(graph, pattern)
        try:
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=spec['weight_decay'],eps=1e-8)
            history, epoch0, stale, best = [], 0, 0, off_score
            best_path, last_path = root/'best.pt', root/'last.pt'
            if (root/'complete.json').exists():
                complete=json.loads((root/'complete.json').read_text())
                if complete['identity']!=identity or file_sha256(last_path)!=complete['last_sha256'] or file_sha256(best_path)!=complete['best_sha256']:
                    raise ValueError('Completed SSF checkpoint changed')
            if last_path.exists():
                last = torch.load(last_path, map_location='cpu', weights_only=False)
                if last['identity'] != identity: raise ValueError('SSF checkpoint identity mismatch')
                adapter.load_state_dict(last['adapter']); optimizer.load_state_dict(last['optimizer'])
                epoch0, stale, best, history = last['epoch'], last['stale'], last['best'], last['history']
                # best.pt may have advanced during an uncommitted interrupted epoch.
                save_torch(last['best_adapter'],best_path)
                torch.set_rng_state(last['torch_rng']); np.random.set_state(last['numpy_rng']);random.setstate(last['python_rng'])
                if device.type == 'cuda': torch.cuda.set_rng_state_all(last['cuda_rng'])
                if train.generator is not None: train.generator.set_state(last['loader_rng'])
            else:
                save_torch(adapter.state_dict(),best_path)
                if train.generator is not None: train.generator.manual_seed(seed)
            if not (root/'complete.json').exists():
                for epoch in range(epoch0, spec['epochs']):
                    if stale >= spec['patience']: break
                    if hasattr(train.dataset, 'set_epoch'): train.dataset.set_epoch(epoch)
                    graph.eval(); adapter.enabled = True
                    optimizer.zero_grad(set_to_none=True)
                    started = time.perf_counter(); total_loss, total_n = 0., 0
                    accumulation = spec['effective_batch_size']//train.batch_size
                    if spec['effective_batch_size'] % train.batch_size:
                        raise ValueError('Execution microbatch must divide effective batch')
                    for step, batch in enumerate(train):
                        first = (step//accumulation)*accumulation
                        end = min(first+accumulation, len(train))
                        group_n = min((end-first)*train.batch_size, len(train.dataset)-first*train.batch_size)
                        o,c = filler.fill(batch['oct'].to(device),batch['cfp'].to(device),pattern)
                        prediction = reset_and_forward(graph,o,c,batch['label'].to(device))
                        loss = graph.get_node_by_name('loss').feature_message.current_state
                        n = len(batch['label'])
                        total_loss += float(loss.detach())*n; total_n += n
                        graph.get_node_by_name('loss').feature_message.current_state = loss*(n/group_n)
                        graph.backward(levels=graph.backward_levels)
                        if step+1 == end:
                            optimizer.step();optimizer.zero_grad(set_to_none=True)
                    if device.type == 'cuda': torch.cuda.synchronize(device)
                    train_seconds = time.perf_counter()-started
                    started = time.perf_counter()
                    val = evaluate_control(graph, validation, device, filler=filler, fixed_pattern=pattern)
                    score = val['metrics']['macro_f1']
                    if not np.isfinite(score): raise ValueError('Non-finite SSF score')
                    if score > best:
                        best, stale = score, 0;save_torch(adapter.state_dict(),best_path)
                    else: stale += 1
                    history.append(dict(epoch=epoch+1, execution_micro_batch_size=train.batch_size, train_loss=total_loss/total_n, macro_f1=score,
                        best_macro_f1=best, training_seconds=train_seconds,
                        validation_seconds=time.perf_counter()-started))
                    save_torch(dict(identity=identity, epoch=epoch+1, stale=stale,best=best,history=history,
                        best_adapter=torch.load(best_path,map_location='cpu',weights_only=True),
                        adapter=adapter.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
                        numpy_rng=np.random.get_state(),python_rng=random.getstate(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                        loader_rng=train.generator.get_state() if train.generator is not None else None),last_path)
                    atomic_write_json(history,root/'history.json')
                    print(f'SSF {pattern} lr={lr} epoch={epoch+1} val={score:.4f} best={best:.4f}',flush=True)
                atomic_write_json(dict(identity=identity, best=best, epochs=len(history),last_sha256=file_sha256(last_path),best_sha256=file_sha256(best_path)),root/'complete.json')
            adapter.load_state_dict(torch.load(best_path,map_location=device,weights_only=True));adapter.enabled=True
            reproduced = evaluate_control(graph,validation,device,filler=filler,fixed_pattern=pattern)
            if reproduced['metrics']['macro_f1'] != best: raise ValueError('SSF reload changed validation score')
            candidates.append(dict(lr=lr, ordinal=li, macro_f1=best, checkpoint=str(best_path),
                best_epoch=next((h['epoch'] for h in history if h['macro_f1']==best),0) if best>off_score else 0,
                checkpoint_sha256=file_sha256(best_path), epochs=len(history), slots=adapter.slots,
                parameters=sum(p.numel() for p in adapter.parameters()),
                training_seconds=sum(h['training_seconds'] for h in history),
                validation_seconds=sum(h['validation_seconds'] for h in history)))
        finally: adapter.close()
        if module_state_sha256(graph) != frozen_hash:
            raise RuntimeError('SSF changed frozen parameters or normalization buffers')
    chosen = min(candidates,key=lambda x:(-x['macro_f1'],x['ordinal']))
    result = dict(method='ssf',reference='https://arxiv.org/abs/2310.03986',
        implementation='matched_LOOK_task_adaptation_not_original_benchmark_reproduction',
        baseline_macro_f1=off_score, selected=chosen, candidates=candidates,
        selection_rule='validation_macro_f1_then_declared_lr_order; epoch_0_identity_allowed',
        supervision='train_labels_cross_entropy',precision='fp32',frozen_state_sha256=frozen_hash,
        training_seconds=sum(x['training_seconds'] for x in candidates),
        validation_seconds=sum(x['validation_seconds'] for x in candidates))
    atomic_write_json(result,output/'selection.json')
    return result
