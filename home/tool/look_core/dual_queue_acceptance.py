"""Small real-validation frozen-graph probe; no new fits or test access."""
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .bounded_runtime import BudgetRunner
from .method_study import make_resource_case
from .method_kernels import forward_control
from .look import load_selected_bank
from .filling import NormalizedMeanFiller
from .distributed import module_state_sha256
from .reproducibility import seed_everything
from .dual_queue import atomic, record


def probe(source, output, seed):
    out = Path(output)
    seed_everything(seed)
    runner = BudgetRunner(source, make_resource_case(seed), out/'resources', out/'cache',
                          torch.device('cuda:0'), (0,))
    _, datasets = runner._build_loaders()
    ds = datasets['validation']
    indices = [int(i) for label in (0, 1) for i in np.flatnonzero(ds.labels == label)[:4]]
    loader = DataLoader(Subset(ds, indices), batch_size=8, shuffle=False, num_workers=0)
    graph, _ = runner._load_frozen_graph(runner._train_or_resume()); graph.eval()
    before = module_state_sha256(graph)
    bank_root = Path(source['result']['path']).parent/'look'
    filler = NormalizedMeanFiller(); values = {}
    batch = next(iter(loader))
    with torch.no_grad():
        for pattern in ('oct_missing', 'cfp_missing'):
            bank = load_selected_bank(bank_root/pattern)
            o, c = filler.fill(batch['oct'].to('cuda:0'), batch['cfp'].to('cuda:0'), pattern)
            values[pattern] = forward_control(graph, o, c, bank, policy='joint', pattern=pattern).cpu().numpy()
    torch.cuda.synchronize()
    if module_state_sha256(graph) != before: raise ValueError('Probe changed frozen backbone')
    path = out/'probe_predictions.npz'
    with path.with_suffix('.partial').open('wb') as f: np.savez_compressed(f, **values)
    path.with_suffix('.partial').replace(path)
    # Keeps contexts overlapping long enough for the process-level NVML audit.
    time.sleep(2)
    result = dict(status='complete', acceptance_only=True, test_access=False, new_fits=0,
                  validation_n=len(indices), seed=seed, frozen_backbone_unchanged=True,
                  cuda_device_name=torch.cuda.get_device_name(0), visible_devices=torch.cuda.device_count(),
                  torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                  provenance=[record(path)])
    rp = out/'probe_result.json'; atomic(result, rp)
    return rp
