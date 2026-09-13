"""Synthetic MHD source-fit and read-only operator validation; never UKB evidence."""
import argparse
import copy
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from mhd_framework.models import create_model
from look.models.observed_participant import ObservedParticipantModel
from look.models.native_host import build_native_host
from look.data.observed_pair import collate_observed
from tests.integration.mechanism_gpu import Samples
from look.methods.operator import fit_complete_pca,fit_look_node,forward_with_look
from look.methods.mechanism_operator import fit_variant,MechanismArtifact
from look.evaluation.mechanism_diagnostics import feature_diagnostics
from look.runtime.host_checkpoint import cpu_tree
from look.training.mechanism_training import state_equal
from look.runtime.state import atomic_write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);device=torch.device('cuda:0')
    torch.set_num_threads(2);torch.manual_seed(7)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    parents=[ObservedParticipantModel(create_model(dict(name='resnet50',num_classes=2,views=1))) for _ in range(2)]
    graph=build_native_host(*parents,'middle',device=device).eval()
    for par in graph.parameters():par.requires_grad_(False)
    state=cpu_tree(graph.state_dict());ds=Samples('train');ds.augment=False;ds.counts=[2]*len(ds)
    loader=DataLoader(ds,batch_size=16,collate_fn=collate_observed,shuffle=False)
    templates=[]
    for node,factor in [('joint_stem',8),('fusion_participant_feature',1)]:
        pca=fit_complete_pca(graph,loader,node,factor,4,device,'synthetic')
        artifact=fit_look_node(graph,loader,node,'oct_missing',factor,[4],4,device,pca,templates)[4]
        templates.append(artifact)
    records=[];batch=next(iter(loader));o=torch.zeros_like(batch['oct']).to(device);c=batch['cfp'].to(device)
    for arm in ('independent','shuffle','missing_readout','available_readout','missing_refit','available_refit','mlp','affine_equivalence'):
        artifacts,info=fit_variant(graph,loader,templates,arm,device,3416,out/arm)
        restored,other=fit_variant(graph,loader,templates,arm,device,3416,out/arm)
        with torch.no_grad():
            pred=forward_with_look(graph,o,c,artifacts,counts=batch['counts'])
            pred2=forward_with_look(graph,o,c,restored,counts=batch['counts'])
            torch.testing.assert_close(pred,pred2,rtol=0,atol=0)
            if arm=='affine_equivalence':
                torch.testing.assert_close(pred,forward_with_look(graph,o,c,templates,counts=batch['counts']),rtol=1e-5,atol=1e-5)
        state_equal(graph,state)
        diag=feature_diagnostics(graph,loader,artifacts,'oct_missing',device);state_equal(graph,state)
        records.append(dict(arm=arm,strict_artifact_reload=True,frozen_host_bn=True,nodes=[x['node'] for x in diag]))
    atomic_write_json(dict(status='accepted',scope='synthetic_32px_two_site_fit_not_UKB_performance',records=records),out/'accepted.json')

if __name__=='__main__':main()
