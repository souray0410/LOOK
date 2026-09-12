"""Native-only references on exactly the host's paired development participants."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from look.data.observed_pair import collate_observed
from look.evaluation.stability import logit_metrics, probabilities_from_logits
from look.evaluation.evaluator import save_prediction_bundle
from look.runtime.state import file_sha256


@torch.no_grad()
def evaluate_available(models,dataset,device,output,batch_size):
    records=[]
    for model,modality,scenario in zip(models,('cfp','oct'),('oct_missing','cfp_missing')):
        model.to(device).eval();zs=[];ys=[];ids=[]
        for batch in DataLoader(dataset,batch_size=batch_size,collate_fn=collate_observed,
            num_workers=0,shuffle=False,generator=torch.Generator().manual_seed(0)):
            zs.append(model(batch[modality].to(device),batch['counts']).cpu().numpy());ys.extend(batch['label'].tolist());ids.extend(batch['participant_id'])
        model.cpu();z=np.concatenate(zs).astype(np.float64);y=np.asarray(ys)
        result=dict(logits=z,probabilities=probabilities_from_logits(z),scores=z[:,1]-z[:,0],labels=y,
            participant_ids=np.asarray(ids),patterns=np.asarray([scenario]*len(ids)),metrics=logit_metrics(y,z))
        path=Path(output)/('available_parent__'+scenario+'.npz');save_prediction_bundle(result,path)
        records.append(dict(method='available_parent',scenario=scenario,metrics=result['metrics'],path=str(path),sha256=file_sha256(path)))
    return records
