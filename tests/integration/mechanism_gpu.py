"""Independent synthetic MHD GPU correctness, not UKB scientific results."""
import argparse
import copy
from pathlib import Path
import torch
from torch.utils.data import Dataset
from mhd_framework.models import create_model
from look.models.observed_participant import ObservedParticipantModel
from look.models.native_host import build_native_host
from look.studies.mechanism_protocol import TRAINING
from look.training.mechanism_training import train
from look.runtime.state import atomic_write_json


class Samples(Dataset):
    def __init__(self,role,size=32,pixels=32):
        self.pixels=pixels
        self.split=role;self.augment=role=='train';self.participant_ids=[role+str(i) for i in range(size)];self.epoch=0
    def __len__(self):return len(self.participant_ids)
    def set_epoch(self,epoch):self.epoch=epoch
    def __getitem__(self,i):
        g=torch.Generator().manual_seed(17+i+self.epoch)
        return dict(cfp=torch.randn(2,3,self.pixels,self.pixels,generator=g),oct=torch.randn(2,3,self.pixels,self.pixels,generator=g),
                    label=i%2,participant_id=self.participant_ids[i])


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--architecture',default='resnet50');p.add_argument('--device',default='cuda:0');p.add_argument('--pixels',type=int,default=32)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device)
    parents=[ObservedParticipantModel(create_model(dict(name=args.architecture,num_classes=2,views=1))) for _ in range(2)]
    data,dev=Samples('train',pixels=args.pixels),Samples('development',16,pixels=args.pixels)
    initial=build_native_host(*parents,'middle',device=device)
    reference=copy.deepcopy(initial.state_dict());initial.cpu();records=[]
    for arm in ('continue_complete','continue_missing','ce','distill'):
        def build():
            host=build_native_host(*parents,'middle',device=device);host.load_state_dict(reference)
            if arm in ('ce','distill'):
                student=copy.deepcopy(parents[0]).to(device)
                return student,host if arm=='distill' else None
            return host,None
        model,teacher=build();torch.manual_seed(93)
        train(model,data,dev,TRAINING,3416,out/arm/'whole','probe',device,arm,'cfp' if arm in ('ce','distill') else None,teacher,preflight_updates=2)
        whole=torch.load(out/arm/'whole/last.pt',map_location='cpu',weights_only=False)
        model.cpu()
        if teacher is not None:teacher.cpu()
        del model,teacher
        torch.cuda.empty_cache()
        model,teacher=build();torch.manual_seed(93)
        train(model,data,dev,TRAINING,3416,out/arm/'split','probe',device,arm,'cfp' if arm in ('ce','distill') else None,teacher,preflight_updates=1)
        model.cpu()
        if teacher is not None:teacher.cpu()
        del model,teacher
        torch.cuda.empty_cache()
        model,teacher=build()
        train(model,data,dev,TRAINING,3416,out/arm/'split','probe',device,arm,'cfp' if arm in ('ce','distill') else None,teacher,preflight_updates=1)
        split=torch.load(out/arm/'split/last.pt',map_location='cpu',weights_only=False)
        for key in whole['model']:torch.testing.assert_close(whole['model'][key],split['model'][key],rtol=0,atol=0)
        def compare(a,b):
            if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
            elif isinstance(a,dict):
                assert a.keys()==b.keys()
                for k in a:compare(a[k],b[k])
            elif isinstance(a,(list,tuple)):
                assert len(a)==len(b)
                for x,y in zip(a,b):compare(x,y)
            else:assert a==b
        compare(whole['optimizer'],split['optimizer']);compare(whole['scheduler'],split['scheduler'])
        compare(whole['progress']['mask_rng'],split['progress']['mask_rng'])
        assert whole['node_ids']==split['node_ids']
        records.append(dict(arm=arm,exact_resume=True,optimizer_equal=True,node_ids_equal=True))
        model.cpu()
        if teacher is not None:teacher.cpu()
        del model,teacher,whole,split
        torch.cuda.empty_cache()
    atomic_write_json(dict(status='accepted',scope='synthetic_MHD_correctness_not_production_resource_admission',pixels=args.pixels,
        architecture=args.architecture,records=records,torch=torch.__version__),out/'accepted.json')

if __name__=='__main__':main()
