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
 p.add_argument('--project',choices=['LOOK'],required=True,
                help='This repository validates LOOK only; Radon_Bridge owns its independent V5 artifact contract')
 p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--manifest-sha',required=True);p.add_argument('--prepare-only',action='store_true')
 p.add_argument('--gpu-budget-bytes',type=int,
                help='Accepted complete-lifecycle budget for this exact workload; no fixed reserve is implied')
 args=p.parse_args();root=args.data_root;out=args.output;out.mkdir(parents=True,exist_ok=True)
 assert file_sha(root/'manifest.json')==args.manifest_sha,'Data manifest changed'
 assert json.loads((root/'accepted.json').read_text())['manifest_sha256']==args.manifest_sha
 os.environ['TORCH_HOME']=str(root/'weights')
 import torch,numpy as np
 from torch.utils.data import DataLoader
 import mhd_framework
 assert mhd_framework.__api_version__=='V5'
 from look.data.dataset import UKBBilateralVisitDataset
 train=UKBBilateralVisitDataset(root/'look/reference_labels_train_development.csv',root/'look/raw_not_included','train',augment=False,preprocess_cache_root=root/'look/cache')
 dev=UKBBilateralVisitDataset(root/'look/reference_labels_train_development.csv',root/'look/raw_not_included','validation',augment=False,preprocess_cache_root=root/'look/cache')
 assert (len(train),len(dev))==(1264,296)
 first=next(iter(DataLoader(train,batch_size=16,shuffle=False,num_workers=0)))
 record=dict(project=args.project,scope='disposable infrastructure acceptance only',test_used=False,scientific_training_result=False,participants={'train':len(train),'development':len(dev)},batch=16,manifest_sha256=args.manifest_sha,framework=mhd_framework.__version__,torch=torch.__version__,source_data_readonly=True)
 if args.prepare_only:
  record['state']='cpu_data_ready';write(out/'prepare.json',record);print(json.dumps(record),flush=True);return
 assert torch.cuda.is_available() and torch.cuda.device_count()==1,'One allocated CUDA device is required'
 device=torch.device('cuda:0');total=torch.cuda.get_device_properties(0).total_memory
 assert args.gpu_budget_bytes is not None and 0 < args.gpu_budget_bytes <= total, \
  'Pass the accepted complete-lifecycle --gpu-budget-bytes for this workload/device'
 torch.cuda.set_per_process_memory_fraction(args.gpu_budget_bytes/total)
 torch.cuda.reset_peak_memory_stats();torch.manual_seed(9181);started=time.monotonic()
 record.update(state='running',gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
               gpu_budget_bytes=args.gpu_budget_bytes,gpu_total_bytes=total,
               gpu_budget_source='explicit_complete_lifecycle_profile',
               fixed_gpu_reserve_bytes=0,slurm_job_id=os.environ.get('SLURM_JOB_ID'))
 def progress(stage,**kw):
  record.update(stage=stage,elapsed_seconds=time.monotonic()-started,**kw);write(out/'status.json',record);print(json.dumps({'stage':stage,**kw}),flush=True)
 progress('load_reference_model')
 from look.models.graph import build_resnet50_mhd_graph,reset_and_forward,optimizer_parameter_groups
 from mhd_framework.utils import MHD_Trainer,MHD_Monitor,MHD_DistributedContext
 from look.runtime.host_checkpoint import read_selected
 selected=json.loads((root/'look/parent/selected.json').read_text())
 assert (selected.get('schema')=='look_v5_selected_reference_v1'
         and selected.get('framework_api')=='V5'
         and isinstance(selected.get('identity'),str)
         and isinstance(selected.get('sha256'),str)
         and isinstance(selected.get('config'),dict)), 'Current LOOK V5 selected reference required'
 checkpoint_path=root/'look/parent/best.pt'
 assert file_sha(checkpoint_path)==selected['sha256'],'Selected V5 checkpoint changed'
 cfg=selected['config']
 graph=build_resnet50_mhd_graph('layer3',batch_size=16,device=device,pretrained=False,classifier_dropout=cfg.get('classifier_dropout',0.),label_smoothing=cfg.get('label_smoothing',0.))
 ids=[(n.id,n.name) for n in sorted(graph.nodes,key=lambda n:n.id)]
 checkpoint=read_selected(checkpoint_path,identity=selected['identity'],node_ids=ids)
 graph.load_state_dict(checkpoint['model'],strict=True);del checkpoint
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
