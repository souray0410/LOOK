"""Bounded real MHD train/resume/correction/report validation, not study evidence."""
import argparse,copy,json,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from mhd_framework.models import create_model
from look.models.observed_participant import ObservedParticipantModel
from look.models.native_host import build_native_host
from look.data.observed_pair import collate_observed
from look.training.observed_host import train_host,DEFAULTS
from look.runtime.host_checkpoint import capture_rng,restore_rng
from look.methods.operator import prepare_complete_pca_bank,greedy_fit_look,load_selected_bank
from look.evaluation.evaluator import evaluate_missing
from look.evaluation.observed_suite import fit_logit_controls,evaluate_suite
from look.analysis.observed_report import write_report
from look.runtime.state import atomic_write_json


class Fixture(Dataset):
    def __init__(self,n,split):
        self.split=split;self.augment=False;self.participant_ids=[split+str(i) for i in range(n)]
    def __len__(self):return len(self.participant_ids)
    def set_epoch(self,e):pass
    def __getitem__(self,i):
        g=torch.Generator().manual_seed(i+(100 if self.split=='train' else 200))
        n=i%2+1
        return dict(oct=torch.rand(n,3,224,224,generator=g),cfp=torch.rand(n,3,224,224,generator=g),label=i%2,participant_id=self.participant_ids[i])



class RealFixture(Dataset):
    def __init__(self,root,split):
        from look.runtime.state import file_sha256
        self.root=Path(root);self.split=split;self.augment=False
        sources=[json.loads((self.root/t/(split+'.json')).read_text()) for t in ('cfp_2d','oct_bscan_2d')]
        assert all(x['role']==split for x in sources)
        rows=sources[0]['samples'];other=sources[1]['samples']
        assert [(r['id'],r['label']) for r in rows]==[(r['id'],r['label']) for r in other]
        chosen=[i for label in (0,1) for i in [j for j,r in enumerate(rows) if r['label']==label][:2]]
        assert len(chosen)==4
        self.rows=[[x['samples'][i] for i in chosen] for x in sources]
        self.participant_ids=[r['id'] for r in self.rows[0]]
    def __len__(self):return len(self.participant_ids)
    def set_epoch(self,e):pass
    def __getitem__(self,i):
        from look.runtime.state import file_sha256
        values=[]
        for track,rows in zip(('cfp_2d','oct_bscan_2d'),self.rows):
            row=rows[i];path=self.root/track/row['file'];assert file_sha256(path)==row['sha256']
            a=np.load(path,allow_pickle=False);x=torch.from_numpy(a.copy()).float()/255
            if x.shape[1]==1:x=x.repeat(1,3,1,1)
            x=(x-torch.tensor([.485,.456,.406]).reshape(1,3,1,1))/torch.tensor([.229,.224,.225]).reshape(1,3,1,1)
            values.append(x)
        return dict(cfp=values[0],oct=values[1],label=self.rows[0][i]['label'],participant_id=self.participant_ids[i])


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--architecture',default='resnet50');p.add_argument('--real-root');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);d=torch.device(a.device)
    torch.set_num_threads(2);torch.manual_seed(3416);np.random.seed(3416);random.seed(3416)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    parents=[ObservedParticipantModel(create_model(dict(name=a.architecture,num_classes=2))) for _ in range(2)]
    config=dict(DEFAULTS,microbatch=2,effective_batch=4,epochs=10,patience=2,minimum_epochs=2,warmup_epochs=1)
    train,dev=([RealFixture(a.real_root,x) for x in ('train','development')] if a.real_root else [Fixture(4,x) for x in ('train','development')])
    base=build_native_host(*parents,'deep',d)
    initial=copy.deepcopy(base.state_dict());rng=capture_rng()
    train_host(base,train,dev,config,3416,out/'continuous','fixture',d,preflight_updates=2)
    expected=torch.load(out/'continuous/last.pt',map_location='cpu',weights_only=False)
    del base;torch.cuda.empty_cache()
    restored=build_native_host(*parents,'deep',d);restored.load_state_dict(initial,strict=True);restore_rng(rng)
    train_host(restored,train,dev,config,3416,out/'resumed','fixture',d,preflight_updates=1)
    # Rebuild in a new graph, restoring full optimizer/BN/RNG/offset.
    del restored;torch.cuda.empty_cache()
    resumed=build_native_host(*parents,'deep',d)
    train_host(resumed,train,dev,config,3416,out/'resumed','fixture',d,preflight_updates=1)
    actual=torch.load(out/'resumed/last.pt',map_location='cpu',weights_only=False)
    def exact(x,y):
        if isinstance(x,torch.Tensor):assert torch.equal(x,y)
        elif isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x:exact(x[k],y[k])
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y)
            for u,v in zip(x,y):exact(u,v)
        else:assert x==y
    for key in ('model','optimizer','scheduler','node_ids'):exact(expected[key],actual[key])
    for key in ('epoch','offset','updates','history','epoch_loss','epoch_seen'):exact(expected['progress'][key],actual['progress'][key])
    frozen={k:v.cpu().clone() for k,v in resumed.state_dict().items()}
    resumed.eval()
    for v in resumed.parameters():v.requires_grad_(False)
    tl=DataLoader(train,batch_size=2,collate_fn=collate_observed);dl=DataLoader(dev,batch_size=2,collate_fn=collate_observed)
    sites=[('fusion_denseblock4' if a.architecture=='densenet121' else 'fusion_stage4'),'fusion_participant_feature']
    bank=prepare_complete_pca_bank(resumed,tl,sites,[8],2,d,out/'pca',{'fixture':a.architecture},out/'quarantine')
    banks={m:{} for m in ('look','single_final','all_on')}
    controls={}
    full=evaluate_missing(resumed,tl,d,fixed_pattern='complete')
    for pattern in ('oct_missing','cfp_missing'):
        for method in banks:
            banks[method][pattern],_=greedy_fit_look(resumed,tl,dl,pattern,
                sites[-1:] if method=='single_final' else sites,[8],[1,2],2,d,out/method/pattern,bank,force_enable=method=='all_on')
        for method in banks:
            loaded=load_selected_bank(out/method/pattern)
            assert len(loaded)==len(banks[method][pattern])
            banks[method][pattern]=loaded
        controls[pattern]=fit_logit_controls(full,evaluate_missing(resumed,tl,d,fixed_pattern=pattern))
    records=evaluate_suite(resumed,dl,d,banks,controls,out/'development',[.2,.4,.6,.8,1.],3407)
    write_report(records,out/'report',100,3416)
    for k,v in resumed.state_dict().items():assert torch.equal(v.cpu(),frozen[k])
    atomic_write_json(dict(status='accepted',scope='real_small_train_development_runtime_not_scientific_results' if a.real_root else 'synthetic_runtime_not_scientific_results',
        actual_mhd_architecture=a.architecture,exact_resume=True,explicit_mhd_backward=True,correction_fit=True,
        selected_bank_reload=True,matched_evaluation=True,report=True,host_bn_unchanged=True,
        views=len(records),test_access=False),out/'accepted.json')

if __name__=='__main__':main()
