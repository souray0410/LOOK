"""Single-process full-state recovery at completed optimizer boundaries."""
import hashlib
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value


def capture_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']:
        if len(state['cuda'])!=torch.cuda.device_count():raise ValueError('CUDA topology changed')
        torch.cuda.set_rng_state_all(state['cuda'])


def atomic_save(path, state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.partial',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            torch.save(state,f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def save_selected(path, *, model, identity, epoch, node_ids, **extra):
    """Write the current selected-host contract, separate from resume state."""
    if {'framework_api', 'schema', 'optimizer', 'scheduler', 'rng'} & extra.keys():
        raise ValueError('Selected host metadata cannot override the state contract')
    state = dict(extra, framework_api='V5', identity=identity, epoch=epoch,
                 node_ids=node_ids, model=cpu_tree(model.state_dict()))
    atomic_save(path, state)
    return state


def read_selected(path, *, identity, node_ids):
    """Reject unconverted V4 or resume files before callers modify a model."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    if (not isinstance(state, dict) or state.get('framework_api') != 'V5'
            or state.get('identity') != identity or state.get('node_ids') != node_ids
            or type(state.get('epoch')) is not int or state['epoch'] < 0
            or not isinstance(state.get('model'), dict)
            or any(key in state for key in ('optimizer', 'scheduler', 'rng', 'schema'))):
        raise ValueError('Current V5 selected host required; explicitly migrate old weights')
    return state


def save(path, *, model, optimizer, scheduler, identity, progress, node_ids=None):
    # Caller must finish optimizer.step and clear gradients; no partial accumulation.
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Checkpoint requires cleared gradients at an optimizer boundary')
    state={'schema':'optimizer_boundary_v2','framework_api':'V5','identity':identity,'model':cpu_tree(model.state_dict()),
           'optimizer':cpu_tree(optimizer.state_dict()),'scheduler':cpu_tree(scheduler.state_dict()),
           'progress':cpu_tree(progress),'rng':capture_rng(),'node_ids':node_ids,'world_size':1}
    atomic_save(path,state)
    return state


def load(path, *, model, optimizer, scheduler, identity, node_ids=None):
    # Only trusted, task-owned checkpoints are accepted; pickle is not an interchange API.
    state=torch.load(path,map_location='cpu',weights_only=False)
    if (state.get('schema')!='optimizer_boundary_v2' or state.get('framework_api')!='V5'
            or state['identity']!=identity or state['world_size']!=1):
        raise ValueError('Current V5 checkpoint identity/world-size mismatch; explicitly migrate old checkpoints')
    if state['node_ids']!=node_ids:raise ValueError('MHD Node IDs changed')
    model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler']);restore_rng(state['rng'])
    return state['progress']
