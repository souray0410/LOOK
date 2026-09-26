"""Explicit full-cohort modern EmbraceNet contract using current V5 parent assets."""
from pathlib import Path
import json
import torch
from look.runtime.state import file_sha256
from look.models.native_materialization import verify_selected

SCHEMA='look_modern_embracenet_20260926_v1'
TRAINING={'epochs':100,'patience':15,'minimum_epochs':8,'warmup_epochs':5,'microbatch':16,'effective_batch':128,
          'pretrained_lr':1e-4,'new_layer_lr':1e-3,'weight_decay':1e-4,'clip':5.0,'precision':'fp32','num_workers':0,
          'loss':'unweighted_cross_entropy','primary_metric':'macro_f1'}
SITES=['joint_input','joint_stem','joint_stage1','joint_stage2','joint_stage3','joint_stage4','joint_features','joint_participant_feature','embraced_feature']
LOOK={'arms':['pca_free_mean','residual_rrr'],'patterns':['oct_missing','cfp_missing'],'search':'positive_forward_tree','factor':16,'rank':32,'penalty_policy':'prefix_train_pca_gcv','sites':SITES}

def read(p):return json.loads(Path(p).read_text())

def parent_specs(spec):return [read(Path(x['path'])/'spec.json') for x in spec['parents']]

def validate(spec):
    from look.runtime.device_budget import validate as budget
    budget(spec)
    if (spec.get('schema')!=SCHEMA or spec.get('test_access') is not False or spec.get('architecture')!='convnext_base'
        or spec.get('seed')!=3416 or spec.get('embracement_size')!=256 or spec.get('training')!=TRAINING or spec.get('look')!=LOOK):
        raise ValueError('Unregistered full-cohort modern method contract')
    if spec.get('cohort')!={'train':58403,'development':12510}:raise ValueError('Complete registered cohort required')
    if spec.get('lease_safety_seconds')!=900:raise ValueError('Lease-safe stage continuation required')
    if len(spec.get('parents',[]))!=2:raise ValueError('Two current parent assets required')
    for record,track in zip(spec['parents'],('cfp_2d','oct_bscan_2d')):
        root=Path(record['path'])
        if file_sha256(root/'selected_artifact.json')!=record['manifest_sha256']:raise ValueError('Parent asset identity changed')
        _,parent,_=verify_selected(root)
        if parent['model']['name']!='convnext_base' or parent['track']!=track or parent['disease']!='cataract':raise ValueError('Parent architecture/modality/disease mismatch')
        from mhd_framework.models.artifacts import verify_runtime
        verify_runtime(parent['framework'])
    parents=parent_specs(spec)
    for role,count in spec['cohort'].items():
        ids=[]
        for parent in parents:
            path=Path(parent[role+'_manifest'])
            if file_sha256(path)!=parent[role+'_manifest_sha256']:raise ValueError('Data identity changed')
            manifest=read(path)
            if manifest['role']!=role or len(manifest['samples'])!=count:raise ValueError('Data role/count mismatch')
            ids.append([(r['id'],r['eyes'],r['label']) for r in manifest['samples']])
        if ids[0]!=ids[1] or len({r[0] for r in ids[0]})!=count:raise ValueError('Participant pairing mismatch')
    train={x['id'] for x in read(parents[0]['train_manifest'])['samples']}
    if train & {x['id'] for x in read(parents[0]['development_manifest'])['samples']}:raise ValueError('Participant leakage')
    if not spec.get('source_pins'):raise ValueError('Pinned execution source required')
    for row in spec['source_pins']:
        if file_sha256(row['path'])!=row['sha256']:raise ValueError('Execution source changed')
    if spec.get('bootstrap')!={'iterations':10000,'seed':3416,'metrics':['macro_f1','macro_auroc_ovr','negative_log_likelihood','multiclass_brier']}:raise ValueError('Matched report contract changed')


def make_graph(spec,device):
    from mhd_framework.models import create_model
    from look.models.observed_participant import ObservedParticipantModel
    from look.models.embracenet import build_embracenet_host
    parents=[]
    for record,parent in zip(spec['parents'],parent_specs(spec)):
        model=ObservedParticipantModel(create_model(parent['model']))
        state=torch.load(Path(record['path'])/'best.pt',map_location='cpu',weights_only=False)
        receipt=read(Path(record['path'])/'source_accepted.json')
        if state.get('framework_api')!='V5' or state['identity']!=receipt['identity']:raise ValueError('Current selected execution identity required')
        model.load_state_dict(state['model'],strict=True);parents.append(model)
    graph=build_embracenet_host(*parents,device=device,embracement_size=spec['embracement_size'],sampling_seed=spec['seed'])
    graph.to(device);return graph


def dataset(spec,role,augment=False):
    from mhd_models.workflows.native import Inputs
    from look.data.observed_pair import from_parent_specs,ObservedPair
    class VerifiedObservedPair(ObservedPair):
        @property
        def verified(self):return {(0,i) for i in self.first.verified}|{(1,i) for i in self.second.verified}
    paired=from_parent_specs(*parent_specs(spec),role,Inputs,augment=augment,seed=spec['seed'])
    return VerifiedObservedPair(paired.first,paired.second,role,augment)


def fresh_process_resume(spec,root,check):
    """Executed as a separate pipeline subprocess after the continuous profile."""
    import os,shutil
    from torch.utils.data import DataLoader
    from look.data.observed_pair import collate_observed
    from look.training.embracenet_host import train_embracenet_host
    from look.evaluation.embracenet import evaluate_complete_analytical
    from look.runtime.state import stable_hash,atomic_write_json
    from look.studies.embracenet_delivery import _tree_equal,StagePaused
    profile=read(root/'profile/accepted.json')
    if profile['identity']!=stable_hash(spec) or profile['state']!='accepted':raise ValueError('Accepted uninterrupted reference required')
    dest=root/'modern_resume';dest.mkdir(exist_ok=True)
    if not (dest/'run').exists():shutil.copytree(root/'profile/resume_base',dest/'run')
    graph=make_graph(spec,torch.device('cuda:0'))
    train,dev=dataset(spec,'train',True),dataset(spec,'development')
    result=train_embracenet_host(graph,train,dev,spec['training'],spec['seed'],dest/'run',stable_hash(spec),torch.device('cuda:0'),check,preflight_target_updates=2)
    if check():raise StagePaused()
    if result['state']!='paused' or result['total_updates']!=2:raise ValueError('Fresh process did not reach exact update boundary')
    actual=torch.load(dest/'run/last.pt',map_location='cpu',weights_only=False)
    reference=torch.load(root/'profile/continuous_two_updates/last.pt',map_location='cpu',weights_only=False)
    _tree_equal(actual,reference)
    # The independent profile already compared all-dev logits; here replay the
    # restored real model over the complete development set as a consumer check.
    evaluated=evaluate_complete_analytical(graph,DataLoader(dev,batch_size=spec['training']['microbatch'],collate_fn=collate_observed),torch.device('cuda:0'),check)
    if check():raise StagePaused()
    atomic_write_json({'state':'accepted','identity':stable_hash(spec),'pid':os.getpid(),'qualification_updates':2,
        'scientific_training_updates':0,'checkpoint_sha256':file_sha256(dest/'run/last.pt'),'reference_sha256':file_sha256(root/'profile/continuous_two_updates/last.pt'),
        'development_participants':len(dev),'test_access':False},dest/'accepted.json')
