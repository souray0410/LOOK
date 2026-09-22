"""Numerical acceptance of a selected parent using every development participant."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from look.models.native_materialization import verify_selected
from look.runtime.state import atomic_write_json, file_sha256


@torch.no_grad()
def replay_selected(model, root, inputs_factory, collate, device, batch_size=16, output=None):
    _, spec, receipt = verify_selected(root)
    dataset = inputs_factory(spec['development_manifest'],spec['development_manifest_sha256'],spec['track'],
                             recipe=spec['training'].get('recipe'))
    saved = np.load(Path(root)/'development_predictions.npz',allow_pickle=False)
    ids = np.asarray(dataset.ids).astype(str)
    if not np.array_equal(saved['ids'].astype(str),ids):
        raise ValueError('Selected prediction participant order differs')
    was_training=model.training; model.eval(); predictions=[];labels=[];indices=[]
    try:
        loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=collate,
                          generator=torch.Generator().manual_seed(0))
        for x,n,y,i in loader:
            predictions.append(model(x.to(device),n).softmax(1).cpu().numpy());labels.extend(y.tolist());indices.extend(i)
    finally:
        model.train(was_training)
    p=np.concatenate(predictions)
    if indices != list(range(len(dataset))) or not np.array_equal(saved['labels'],labels):
        raise ValueError('Selected replay labels/coverage mismatch')
    difference=float(np.max(np.abs(p-saved['probabilities'])))
    if not np.allclose(p,saved['probabilities'],rtol=1e-4,atol=1e-5) or not np.array_equal(p.argmax(1),saved['probabilities'].argmax(1)):
        raise ValueError(f'Selected parent numerical replay failed: max error {difference}')
    result=dict(schema='look_native_replay_v1',status='accepted',participants=len(ids),
                best_sha256=file_sha256(Path(root)/'best.pt'), source_receipt_sha256=file_sha256(Path(root)/'source_accepted.json'),
                maximum_probability_error=difference, argmax_identical=True, split='development',test_access=False,
                execution_provenance=model.execution_provenance)
    if output is not None: atomic_write_json(result,Path(output))
    return result
