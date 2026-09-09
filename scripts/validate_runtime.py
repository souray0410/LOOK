"""Finite real-data infrastructure acceptance; never a scientific training result."""
from pathlib import Path
import argparse,hashlib,json,os,time,gc

def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def write(path,value):
 path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--project',choices=['LOOK','Radon_Bridge'],required=True)
 p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--manifest-sha',required=True);p.add_argument('--prepare-only',action='store_true')
 args=p.parse_args();root=args.data_root;out=args.output;out.mkdir(parents=True,exist_ok=True)
 assert file_sha(root/'manifest.json')==args.manifest_sha,'Data manifest changed'
 assert json.loads((root/'accepted.json').read_text())['manifest_sha256']==args.manifest_sha
 os.environ['TORCH_HOME']=str(root/'weights')
 import torch,numpy as np
 from torch.utils.data import DataLoader
 import mhd_framework
 assert mhd_framework.__api_version__=='V4'
 if args.project=='LOOK':
  from look.data.dataset import UKBBilateralVisitDataset
  train=UKBBilateralVisitDataset(root/'look/reference_labels_train_development.csv',root/'look/raw_not_included','train',augment=False,preprocess_cache_root=root/'look/cache')
  dev=UKBBilateralVisitDataset(root/'look/reference_labels_train_development.csv',root/'look/raw_not_included','validation',augment=False,preprocess_cache_root=root/'look/cache')
 else:
  from radon_bridge.data.dataset import PairedDataset
  train=PairedDataset(root/'radon/cache','train',cfp_size=224);dev=PairedDataset(root/'radon/cache','validation',cfp_size=224)
 assert (len(train),len(dev))==(1264,296)
 first=next(iter(DataLoader(train,batch_size=16,shuffle=False,num_workers=0)))
 record=dict(project=args.project,scope='disposable infrastructure acceptance only',test_used=False,scientific_training_result=False,participants={'train':len(train),'development':len(dev)},batch=16,manifest_sha256=args.manifest_sha,framework=mhd_framework.__version__,torch=torch.__version__,source_data_readonly=True)
 if args.prepare_only:
  record['state']='cpu_data_ready';write(out/'prepare.json',record);print(json.dumps(record),flush=True);return
 assert torch.cuda.is_available() and torch.cuda.device_count()==1,'One allocated CUDA device is required'
 device=torch.device('cuda:0');limit=10 if args.project=='Radon_Bridge' else 14
 torch.cuda.set_per_process_memory_fraction((limit-1)*1024**3/torch.cuda.get_device_properties(0).total_memory)
 torch.cuda.reset_peak_memory_stats();torch.manual_seed(9181);started=time.monotonic()
 record.update(state='running',gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,project_memory_limit_gib=limit,slurm_job_id=os.environ.get('SLURM_JOB_ID'))
 def progress(stage,**kw):
  record.update(stage=stage,elapsed_seconds=time.monotonic()-started,**kw);write(out/'status.json',record);print(json.dumps({'stage':stage,**kw}),flush=True)
 progress('load_reference_model')
 if args.project=='Radon_Bridge':
  from radon_bridge.models.model import PilotGraph
  from radon_bridge.training.optimization import configure_optimizer,clip_task_gradients
  dep=json.loads((root/'radon/dependencies_relative.json').read_text())
  def load_parents(model):
   for branch,ref in dep['parents']['3416'].items():
    path=root/ref['path'];assert file_sha(path)==ref['sha256'];model.load_native_state(torch.load(path,map_location='cpu',weights_only=False)['model'],branch=branch)
  def inputs(batch):return [v.to(device) for v in batch[:3]]
  values=inputs(first);plain=PilotGraph(seed=3416,device=device);load_parents(plain);plain.graph.eval()
  with torch.no_grad():expected={k:v.detach().cpu().clone() for k,v in plain.forward(*values)[0].items()}
  del plain;gc.collect();torch.cuda.empty_cache()
  refs={k:{'path':str(root/v['path']),'sha256':v['sha256']} for k,v in dep['bases']['3416'].items()}
  config={'nodes':['cfp_stage3','oct_stage3'],'M':32,'S':64,'rho':.125,'mode':'radon','kernel_size':3,'compression':'fixed_svd_channel','basis_files':refs}
  model=PilotGraph(seed=3416,device=device,bridge_configs=[config]);load_parents(model);graph=model.graph;graph.eval()
  with torch.no_grad():actual=model.forward(*values)[0]
  for k in expected:torch.testing.assert_close(actual[k].cpu(),expected[k],rtol=1e-6,atol=1e-6)
  record['zero_initialization_matches_parent']=True
  policy={'adapt_stages':[1,2,3,4],'training_regime':'full_finetune','backbone_lr':6e-5,'head_lr':1e-4,'bridge_lr':1e-4,'weight_decay':.01}
  optimizer=configure_optimizer(model,policy)
  def predict(batch):return model.forward(*inputs(batch))[0]
  def step(batch):
   optimizer.zero_grad(set_to_none=True);outputs,loss=model.forward(*inputs(batch));assert torch.isfinite(loss);model.backward();clip_task_gradients(model,5.);optimizer.step();return float(loss.detach())
 else:
  from look.models.graph import build_resnet50_mhd_graph,reset_and_forward,optimizer_parameter_groups
  from mhd_framework.utils import MHD_Trainer,MHD_Monitor,MHD_DistributedContext
  checkpoint=torch.load(root/'look/parent/best.pt',map_location='cpu',weights_only=False)
  cfg=checkpoint['config']
  graph=build_resnet50_mhd_graph('layer3',batch_size=16,device=device,pretrained=False,classifier_dropout=cfg.get('classifier_dropout',0.),label_smoothing=cfg.get('label_smoothing',0.))
  graph.load_state_dict(checkpoint['graph_state_dict'],strict=True);del checkpoint
  optimizer=torch.optim.AdamW(optimizer_parameter_groups(graph,3e-4,.003),weight_decay=.0001)
  trainer=MHD_Trainer(graph,optimizer,MHD_Monitor(['loss']),graph.forward_levels,graph.backward_levels,criteria=lambda g:g.get_node_by_name('loss').feature_message.current_state,save_dir=str(out/'disposable_trainer'),input_nodes=['oct_input','cfp_input','label_gt'],output_nodes=['fusion_logits','loss'],precision='fp32',distributed_context=MHD_DistributedContext(0,0,1,device,'gloo'))
  def predict(batch):return {'fusion':reset_and_forward(graph,batch['oct'].to(device),batch['cfp'].to(device))}
  def step(batch):
   metrics=trainer.train_step({'oct_input':batch['oct'].to(device),'cfp_input':batch['cfp'].to(device),'label_gt':batch['label'].to(device)})
   assert np.isfinite(metrics['loss']);return float(metrics['loss'])
 record['strict_reference_checkpoint_loaded']=True
 node_ids=sorted((n.id,n.name) for n in graph.nodes)
 tracked=next(p for p in graph.parameters() if p.requires_grad);before=tracked.detach().clone()
 progress('four_disposable_optimizer_updates');graph.train()
 for i,batch in enumerate(DataLoader(train,batch_size=16,shuffle=False,num_workers=2)):
  if i==4:break
  value=step(batch);progress('optimizer_update',updates=i+1,finite_loss=True)
 assert not torch.equal(before,tracked.detach()),'Native parameters did not update'
 del before
 assert node_ids==sorted((n.id,n.name) for n in graph.nodes)
 if args.project=='Radon_Bridge':
  communication=[p for n,m in model.modules_by_name().items() if n.startswith('bridge_') for p in m.parameters()]
  assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in communication),'No bridge gradient'
 record.update(optimizer_updates=4,native_parameters_updated=True,node_ids_preserved=True)
 graph.eval()
 with torch.no_grad():reference={k:v.detach().cpu().clone() for k,v in predict(first).items()}
 state={k:v.detach().cpu() for k,v in graph.state_dict().items()};torch.save({'model':state,'optimizer':optimizer.state_dict(),'scope':'disposable_runtime_acceptance'},out/'roundtrip.pt');del state
 saved=torch.load(out/'roundtrip.pt',map_location='cpu',weights_only=False);graph.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);del saved
 with torch.no_grad():reloaded=predict(first)
 for k in reference:torch.testing.assert_close(reloaded[k].cpu(),reference[k],rtol=0,atol=0)
 record['checkpoint_output_exact']=True
 for split,dataset in [('train',train),('development',dev)]:
  seen=0;digest=hashlib.sha256();progress('read_and_infer_'+split)
  with torch.no_grad():
   for batch in DataLoader(dataset,batch_size=16,shuffle=False,num_workers=2):
    pred=predict(batch)
    for name,value in pred.items():
     assert value.ndim==2 and value.shape[1]==2 and torch.isfinite(value).all()
     digest.update(name.encode());digest.update(value.detach().float().cpu().numpy().tobytes())
    seen+=len(next(iter(pred.values())))
    if seen%256==0:progress('read_and_infer_'+split,seen=seen)
  assert seen==len(dataset);record[split+'_inference']={'count':seen,'logit_sha256':digest.hexdigest()}
 torch.cuda.synchronize();record.update(state='accepted',stage='complete',elapsed_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),multigpu_training_tested=False)
 write(out/'accepted.json',record);write(out/'status.json',record);print(json.dumps(record),flush=True)
if __name__=='__main__':main()
