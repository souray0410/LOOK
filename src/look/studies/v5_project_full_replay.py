"""One-time formal-V5 full-cohort PCA and first-decision replay.

This migration entrypoint is deliberately separate from current V5 readers. It
accepts only the pinned historical project-case contract and never opens test.
"""
from __future__ import annotations
import argparse,json,pathlib,time
import torch
from torch.utils.data import DataLoader
from mhd_framework.models import create_model
from look.data.observed_pair import collate_observed
from look.models.observed_participant import ObservedParticipantModel
from look.models.native_host import build_native_host
from look.methods.joint import correction_sites
from look.methods.operator import prepare_complete_pca_bank,greedy_fit_look
from look.runtime.host_checkpoint import read_selected
from look.runtime.provenance import write_json_atomic
from look.runtime.state import file_sha256,stable_hash
from look.studies.v5_project_feature_replay import datasets,_selected_parent

FRAMEWORK='1287681c08846e11364c81653048435482e772a7'
MODELS='cc16e74a8cfc705d69b3d31efe2daeec9404471f'

def read(p): return json.loads(pathlib.Path(p).read_text())
def _parent(root,manifest):
 spec=_selected_parent(root,manifest); graph=create_model(spec['model'],device='cpu')
 model=ObservedParticipantModel(graph); state=torch.load(pathlib.Path(root)/'best.pt',map_location='cpu',weights_only=False)
 model.load_state_dict(state['model'],strict=True); model.eval(); return model

def execute(*,source_run,checkpoint,output,inputs_factory,device,gpu_budget_bytes):
 from mhd_framework import __api_version__
 if __api_version__!='V5': raise ValueError('Formal V5 required')
 source_run=pathlib.Path(source_run).resolve();output=pathlib.Path(output).resolve();checkpoint=pathlib.Path(checkpoint).resolve()
 if not gpu_budget_bytes or gpu_budget_bytes<=0: raise ValueError('Measured GPU budget required')
 if output==source_run or output.is_relative_to(source_run): raise ValueError('Isolated output required')
 spec=read(source_run/'spec.json')
 if spec.get('schema')!='look_project_case_v1' or spec.get('test_access') is not False: raise ValueError('Historical train/development project case required')
 total=torch.cuda.get_device_properties(device).total_memory
 if gpu_budget_bytes>=total: raise ValueError('GPU budget must retain headroom')
 torch.cuda.set_per_process_memory_fraction(gpu_budget_bytes/total,device)
 parents=[_parent(spec['parents'][k]['path'],spec['parents'][k]['manifest_sha256']) for k in ('first','second')]
 graph=build_native_host(*parents,spec['position'],device=device); del parents
 accepted=read(source_run/'host/accepted.json'); ids=[(n.id,n.name) for n in sorted(graph.nodes,key=lambda n:n.id)]
 state=read_selected(checkpoint,identity=accepted['identity'],node_ids=ids);graph.load_state_dict(state['model'],strict=True);graph.eval()
 data,cohort=datasets(source_run,inputs_factory,seed=spec['seed'])
 def loader(role): return DataLoader(data[role],batch_size=spec['training']['microbatch'],shuffle=False,num_workers=spec['training']['num_workers'],collate_fn=collate_observed,generator=torch.Generator().manual_seed(spec['seed']))
 train,dev=loader('train'),loader('development');sites=correction_sites(graph)
 if len(sites)!=9: raise ValueError('Exact nine-site host required')
 identity={'schema':'look_formal_v5_project_full_replay_v1','framework_commit':FRAMEWORK,'models_commit':MODELS,'source_spec_sha256':file_sha256(source_run/'spec.json'),'target_checkpoint_sha256':file_sha256(checkpoint),'cohort':cohort,'sites':sites,'test_access':False}
 write_json_atomic(identity,output/'identity.json')
 cfg=spec['look'];bank=prepare_complete_pca_bank(graph,train,sites,cfg['factors'],cfg['max_rank'],device,output/'pca',identity,output/'quarantine')
 # The protocol's first deterministic replacement decision is the first site
 # under its first declared missing pattern and spatial factor.
 pattern='oct_missing';factor=cfg['factors'][0]
 artifacts,history,completion=greedy_fit_look(graph,train,dev,pattern,[sites[0]],[factor],cfg['latent_dims'],cfg['max_rank'],device,output/'first_decision',bank,resume=True)
 receipt={'schema':'look_formal_v5_project_full_replay_receipt_v1','state':'accepted','identity_sha256':stable_hash(identity),'sites':sites,'pca_entries':len(bank),'train_participants':len(data['train']),'development_participants':len(data['development']),'first_decision':completion['decisions'][0],'test_access':False,'completed_at':time.time()}
 write_json_atomic(receipt,output/'accepted.json');return receipt

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--source-run',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--gpu-budget-bytes',required=True,type=int);a=p.parse_args(argv)
 from expanded.native import Inputs
 print(json.dumps(execute(source_run=a.source_run,checkpoint=a.checkpoint,output=a.output,inputs_factory=Inputs,device=torch.device(a.device),gpu_budget_bytes=a.gpu_budget_bytes)),flush=True)
if __name__=='__main__':main()
