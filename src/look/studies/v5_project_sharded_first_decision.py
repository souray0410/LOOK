"""Run the registered first prefix decision from an accepted sharded PCA bank."""
from __future__ import annotations
import argparse,json,pathlib,time
import torch
from torch.utils.data import DataLoader
from look.data.observed_pair import collate_observed
from look.methods.joint import correction_sites
from look.methods.operator import FullFeaturePCA,greedy_fit_look
from look.runtime.host_checkpoint import read_selected
from look.runtime.provenance import write_json_atomic
from look.runtime.state import file_sha256,stable_hash
from look.studies.v5_project_feature_replay import datasets
from look.studies.v5_project_full_replay import FRAMEWORK,MODELS,_parent,read,validate_claim

def load_pca(output:pathlib.Path,expected_cache_sha:str):
 output=pathlib.Path(output).resolve();identity=read(output/'identity.json');receipt=read(output/'accepted.json')
 if receipt.get('schema')!='look_formal_v5_sharded_pca_receipt_v1' or receipt.get('state')!='accepted': raise ValueError('Accepted PCA required')
 if stable_hash(identity)!=receipt.get('identity_sha256') or identity.get('cache_acceptance_sha256')!=expected_cache_sha: raise ValueError('PCA/cache identity mismatch')
 if identity.get('test_access') is not False or receipt.get('test_access') is not False: raise ValueError('Train/development PCA required')
 bank={}
 for row in receipt['entries']:
  path=output/row['path']
  if path.is_symlink() or not path.is_relative_to(output/'bank') or file_sha256(path)!=row['sha256']: raise ValueError('PCA entry hash mismatch')
  basis=FullFeaturePCA.load(path)
  if basis.source_id!=row['source_id'] or basis.node_name!=row['site'] or basis.factor!=row['factor']: raise ValueError('PCA entry identity mismatch')
  bank[(row['site'],row['factor'])]=basis
 return identity,bank

def execute(*,source_run,checkpoint,cache,pca,output,inputs_factory,device,gpu_budget_bytes):
 source_run=pathlib.Path(source_run).resolve();checkpoint=pathlib.Path(checkpoint).resolve();cache=pathlib.Path(cache).resolve();output=pathlib.Path(output).resolve()
 if output.exists() and not output.is_dir(): raise ValueError('Output must be directory')
 total=torch.cuda.get_device_properties(device).total_memory
 if not 0<gpu_budget_bytes<total: raise ValueError('Measured GPU budget with headroom required')
 torch.cuda.set_per_process_memory_fraction(gpu_budget_bytes/total,device)
 cache_accept=read(cache/'accepted.json')
 if cache_accept.get('state')!='accepted_complete': raise ValueError('Complete cache required before downstream')
 pca_identity,bank=load_pca(pathlib.Path(pca),file_sha256(cache/'accepted.json'))
 spec=read(source_run/'spec.json');accepted=read(source_run/'host/accepted.json')
 if spec.get('test_access') is not False: raise ValueError('Train/development source required')
 parents=[_parent(spec['parents'][k]['path'],spec['parents'][k]['manifest_sha256']) for k in ('first','second')]
 from look.models.native_host import build_native_host
 graph=build_native_host(*parents,spec['position'],device=device);del parents
 ids=[(n.id,n.name) for n in sorted(graph.nodes,key=lambda n:n.id)]
 state=read_selected(checkpoint,identity=accepted['identity'],node_ids=ids);graph.load_state_dict(state['model'],strict=True);graph.eval()
 data,cohort=datasets(source_run,inputs_factory,seed=spec['seed'])
 def loader(role): return DataLoader(data[role],batch_size=spec['training']['microbatch'],shuffle=False,num_workers=spec['training']['num_workers'],collate_fn=collate_observed,generator=torch.Generator().manual_seed(spec['seed']))
 sites=correction_sites(graph)
 if sites!=pca_identity['sites'] or len(sites)!=9: raise ValueError('Exact nine-site PCA/graph order required')
 cfg=spec['look'];factor=cfg['factors'][0]
 # Registered migration contract: first deterministic prefix is the first site.
 artifacts,history=greedy_fit_look(graph,loader('train'),loader('development'),'oct_missing',[sites[0]],[factor],cfg['latent_dims'],cfg['max_rank'],device,output/'first_prefix',bank,resume=True)
 decision=read(output/'first_prefix'/'factors'/f'x{factor}'/'decisions'/f'01_{sites[0]}.json')
 identity={'schema':'look_formal_v5_first_prefix_v1','framework_commit':FRAMEWORK,'models_commit':MODELS,
  'cache_acceptance_sha256':file_sha256(cache/'accepted.json'),'pca_acceptance_sha256':file_sha256(pathlib.Path(pca)/'accepted.json'),
  'source_spec_sha256':file_sha256(source_run/'spec.json'),'checkpoint_sha256':file_sha256(checkpoint),'sites':sites,
  'train':cohort['roles']['train'],'development':cohort['roles']['development'],'test_access':False}
 receipt={'schema':'look_formal_v5_first_prefix_receipt_v1','state':'accepted','identity_sha256':stable_hash(identity),
  'evaluated_site':sites[0],'all_nine_sites_bound':True,'decision':decision,'artifacts':len(artifacts),'history_rows':len(history),'completed_at':time.time(),'test_access':False}
 write_json_atomic(identity,output/'identity.json');write_json_atomic(receipt,output/'accepted.json');return receipt

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--source-run',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--cache',required=True);p.add_argument('--pca',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--gpu-budget-bytes',required=True,type=int)
 p.add_argument('--comparison-contract');p.add_argument('--comparison-output');a=p.parse_args(argv)
 if bool(a.comparison_contract)!=bool(a.comparison_output): raise ValueError('Comparison contract and output must be configured together')
 validate_claim();from expanded.native import Inputs
 receipt=execute(source_run=a.source_run,checkpoint=a.checkpoint,cache=a.cache,pca=a.pca,output=a.output,inputs_factory=Inputs,device=torch.device(a.device),gpu_budget_bytes=a.gpu_budget_bytes)
 if a.comparison_contract:
  from look.studies.v5_full_cohort_first_decision_gate import execute as compare
  comparison=compare(first_prefix=pathlib.Path(a.output),contract_path=pathlib.Path(a.comparison_contract),output=pathlib.Path(a.comparison_output))
  print(json.dumps({'first_prefix':receipt,'comparison':comparison}),flush=True)
  if comparison['state']!='eligible_handoff': raise SystemExit(75)
 else: print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
