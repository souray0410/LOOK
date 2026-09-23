"""Chunked formal-V5 nine-site train feature materialization for offline migration."""
from __future__ import annotations
import argparse,hashlib,json,pathlib,time
import torch
from torch.utils.data import DataLoader
from look.data.observed_pair import collate_observed
from look.methods.joint import correction_sites,read_site
from look.methods.operator import forward_with_look
from look.runtime.provenance import write_json_atomic
from look.runtime.state import file_sha256,stable_hash
from look.studies.v5_project_feature_replay import datasets
from look.studies.v5_project_full_replay import _parent,read,validate_claim,FRAMEWORK,MODELS

def participant_sha(ids): return hashlib.sha256(json.dumps(list(ids),separators=(',',':')).encode()).hexdigest()
def materialize(*,source_run,checkpoint,output,inputs_factory,device,gpu_budget_bytes,max_batches=None):
 from look.models.native_host import build_native_host
 from look.runtime.host_checkpoint import read_selected
 source_run=pathlib.Path(source_run).resolve();checkpoint=pathlib.Path(checkpoint).resolve();output=pathlib.Path(output).resolve()
 if not gpu_budget_bytes or gpu_budget_bytes<=0: raise ValueError('Measured GPU budget required')
 spec=read(source_run/'spec.json');accepted=read(source_run/'host/accepted.json')
 if spec.get('test_access') is not False: raise ValueError('Train/development source required')
 total=torch.cuda.get_device_properties(device).total_memory
 if gpu_budget_bytes>=total: raise ValueError('GPU budget must retain headroom')
 torch.cuda.set_per_process_memory_fraction(gpu_budget_bytes/total,device)
 parents=[_parent(spec['parents'][k]['path'],spec['parents'][k]['manifest_sha256']) for k in ('first','second')]
 graph=build_native_host(*parents,spec['position'],device=device);del parents
 ids=[(n.id,n.name) for n in sorted(graph.nodes,key=lambda n:n.id)];state=read_selected(checkpoint,identity=accepted['identity'],node_ids=ids)
 graph.load_state_dict(state['model'],strict=True);graph.eval();data,cohort=datasets(source_run,inputs_factory,seed=spec['seed']);sites=correction_sites(graph)
 if len(sites)!=9: raise ValueError('Exact nine-site host required')
 identity={'schema':'look_formal_v5_train_feature_cache_v1','framework_commit':FRAMEWORK,'models_commit':MODELS,'source_spec_sha256':file_sha256(source_run/'spec.json'),'target_checkpoint_sha256':file_sha256(checkpoint),'train':cohort['roles']['train'],'sites':sites,'test_access':False}
 output.mkdir(parents=True,exist_ok=True);ip=output/'identity.json'
 if ip.exists() and read(ip)!=identity: raise ValueError('Feature cache identity changed')
 write_json_atomic(identity,ip);records=output/'records';records.mkdir(exist_ok=True)
 loader=DataLoader(data['train'],batch_size=spec['training']['microbatch'],shuffle=False,num_workers=spec['training']['num_workers'],collate_fn=collate_observed,generator=torch.Generator().manual_seed(spec['seed']))
 completed=0;processed_this_run=0;started=time.time()
 with torch.no_grad():
  for index,batch in enumerate(loader):
   path=records/f'{index:06d}.pt';meta=path.with_suffix('.json')
   if path.exists() or meta.exists():
    row=read(meta)
    if row['batch']!=index or row['participants_sha256']!=participant_sha(batch['participant_id']) or row['sha256']!=file_sha256(path): raise ValueError('Existing feature chunk changed')
    completed=index+1;continue
   forward_with_look(graph,batch['oct'].to(device),batch['cfp'].to(device),counts=batch['counts'])
   payload={'batch':index,'participant_id':list(batch['participant_id']),'sites':{s:read_site(graph,s).detach().cpu() for s in sites}}
   tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)
   row={'batch':index,'participants_sha256':participant_sha(batch['participant_id']),'sha256':file_sha256(path),'bytes':path.stat().st_size,'sites':sites,'test_access':False};write_json_atomic(row,meta);completed=index+1
   write_json_atomic({'state':'running','completed_batches':completed,'participants_consumed':min(completed*spec['training']['microbatch'],len(data['train'])),'elapsed_seconds':time.time()-started,'test_access':False},output/'status.json')
   if max_batches is not None and processed_this_run>=max_batches: break
 total_batches=(len(data['train'])+spec['training']['microbatch']-1)//spec['training']['microbatch'];complete=completed==total_batches
 receipt={'schema':'look_formal_v5_train_feature_materialization_v1','state':'accepted_complete' if complete else 'accepted_partial','identity_sha256':stable_hash(identity),'completed_batches':completed,'processed_this_run':processed_this_run,'total_batches':total_batches,'participants_total':len(data['train']),'bytes':sum(p.stat().st_size for p in records.glob('*.pt')),'elapsed_seconds':time.time()-started,'test_access':False}
 write_json_atomic(receipt,output/('accepted.json' if complete else 'partial.json'));return receipt

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--source-run',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--gpu-budget-bytes',required=True,type=int);p.add_argument('--max-batches',type=int);a=p.parse_args(argv)
 validate_claim();from expanded.native import Inputs
 print(json.dumps(materialize(source_run=a.source_run,checkpoint=a.checkpoint,output=a.output,inputs_factory=Inputs,device=torch.device(a.device),gpu_budget_bytes=a.gpu_budget_bytes,max_batches=a.max_batches)),flush=True)
if __name__=='__main__':main()
