"""Prepare a complete registry now; evaluate only after an explicit audited release.

This entry point is independent of training. It never fits PCA, corrections,
thresholds or model weights. Existing training adapters continue to reject test.
"""
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from look.runtime.state import atomic_write_json,file_sha256,stable_hash
from look.evaluation.mechanism_gate import validate_release


def read(p):return json.loads(Path(p).read_text())


def prepare(feed,base_feed,output):
    from look.studies.project_case import verify_case as verify_base
    from look.studies.mechanism_case import verify_case
    models=[];unresolved=[];excluded=[]
    for kind,queue,verify in (('original',base_feed,verify_base),('supplement',feed,verify_case)):
        for t in queue['tasks']:
            run=Path(t['run_dir']);spec=read(t['spec'])
            if not (run/'accepted.json').exists():unresolved.append(t['id']);continue
            from look.runtime.mechanism_receipts import monitored_acceptance
            receipt=monitored_acceptance(run,spec,verify,Path(output).parent/'test_preparation_verification_cache')
            if receipt.get('state')=='infeasible':excluded.append(dict(id=t['id'],reason=receipt.get('reason')));continue
            if kind=='original':checkpoint=run/'host/best.pt';disease=spec['disease']
            else:
                disease=spec['task']['host']['disease']
                checkpoint=run/'training/best.pt' if spec['task']['kind'].endswith('_training') else Path(spec['source']['run_dir'])/'host/best.pt'
            models.append(dict(id=t['id'],kind=kind,spec=t['spec'],spec_sha256=file_sha256(t['spec']),run_dir=str(run),
                checkpoint=str(checkpoint),checkpoint_sha256=file_sha256(checkpoint),disease=disease,
                acceptance_sha256=file_sha256(run/'accepted.json'),state='accepted',development_replay='required',
                development_replay_reason='accepted training predictions do not replace an independent full registry replay'))
    result=dict(schema='look_test_models_v1',models=models,unresolved=unresolved,exclusions=excluded,
        all_training_resolved=not unresolved and len(feed['tasks'])==feed['expected'] and len(base_feed['tasks'])==81,
        unique_neural_checkpoint_count=len({m['checkpoint_sha256'] for m in models}),
        registered_execution_views=len(models),
        count_note='execution views include distinct corrections and aliases; not distinct neural trainings',
        test_access=False,state='prepared_not_released')
    atomic_write_json(result,output)
    return result


def evaluate_model(model,manifest,output,device,role="test"):
    """Called only after validate_release; one complete selected model at a time."""
    from mhd_models.workflows.native import Inputs
    from look.studies.mechanism_case import context,loader,verify_case
    from look.data.observed_pair import ObservedPair
    from look.models.observed_participant import verify_pair_rows
    from look.methods.operator import load_selected_bank
    from look.methods.mechanism_operator import MechanismArtifact
    from look.training.mechanism_training import evaluate_student,state_equal
    from look.runtime.host_checkpoint import cpu_tree
    from look.evaluation.observed_suite import evaluate_suite
    from look.evaluation.evaluator import evaluate_missing,save_prediction_bundle
    if role not in ("development", "test"):raise ValueError("Invalid evaluation role")
    spec=read(model['spec']);run=Path(model['run_dir']);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if file_sha256(model['spec'])!=model['spec_sha256'] or file_sha256(run/'accepted.json')!=model['acceptance_sha256']:
        raise ValueError('Evaluation provenance changed')
    if model['kind']=='original':
        from look.studies.project_case import verify_case as verify_original
        from look.studies.mechanism_case import load_original
        verify_original(run,spec);base,parents,graph,data=load_original(spec,run,device)
        task=dict(kind='original')
    else:
        verify_case(run,spec);base,parents,graph,data=context(spec,device);task=spec['task']
    if role=="development":
        pair=data["development"]
    else:
        parent_specs=[read(Path(base['parents'][role]['path'])/'spec.json') for role in ('first','second')]
        datasets=[Inputs(manifest['path'],manifest['sha256'],track,seed=base['seed'],augment=False,
            recipe=p['training'].get('recipe')) for p,track in zip(parent_specs,('cfp_2d','oct_bscan_2d'))]
        # Intentionally separate constructor: the ordinary fitting dataset rejects test.
        pair=object.__new__(ObservedPair);pair.first,pair.second=datasets
        pair.pairing=verify_pair_rows(datasets[0].rows,datasets[1].rows)
        pair.counts=[len(r['eyes']) for r in datasets[0].rows];pair.participant_ids=[str(r['id']) for r in datasets[0].rows]
        pair.split='test';pair.augment=False
        if stable_hash(pair.participant_ids)!=manifest['participant_sha256']:raise ValueError('Test participant manifest mismatch')
    dl=loader(pair,base['seed']);banks={};records=[]
    if task['kind']=='original':
        graph.eval();frozen=cpu_tree(graph.state_dict())
        for p in graph.parameters():p.requires_grad_(False)
        for method in ('look','single_final','all_on'):
            banks[method]={p:load_selected_bank(run/'corrections'/method/p) for p in ('oct_missing','cfp_missing')}
        records=evaluate_suite(graph,dl,device,banks,read(run/'controls.json'),out,base['look']['ratios'],base['look']['mask_seed'])
        state_equal(graph,frozen)
        from look.evaluation.observed_native import evaluate_available
        records+=evaluate_available(parents,pair,device,out,16)
    elif task['kind'].endswith('_training'):
        checkpoint=torch.load(model['checkpoint'],map_location='cpu',weights_only=False)
        target=parents[0 if task.get('track')=='cfp' else 1] if task['kind']=='student_training' else graph
        target.to(device).load_state_dict(checkpoint['model'],strict=True);target.eval()
        for p in target.parameters():p.requires_grad_(False)
        frozen=cpu_tree(target.state_dict())
        if task['kind']=='student_training':
            result=evaluate_student(target,dl,device,task['track'])
            save_prediction_bundle(result,out/'student.npz');records=[dict(method=task['arm'],scenario='oct_missing' if task['track']=='cfp' else 'cfp_missing',path=str(out/'student.npz'))]
        else:
            for method in ('look','single_final','all_on'):
                banks[method]={p:load_selected_bank(run/'corrections'/method/p) for p in ('oct_missing','cfp_missing')}
            records=evaluate_suite(graph,dl,device,banks,read(run/'controls.json'),out,base['look']['ratios'],base['look']['mask_seed'])
        state_equal(target,frozen)
        if task['kind']=='host_training':
            from look.evaluation.observed_native import evaluate_available
            records+=evaluate_available(parents,pair,device,out,16)
    else:
        for p in graph.parameters():p.requires_grad_(False)
        graph.eval();frozen=cpu_tree(graph.state_dict());pattern=task['pattern']
        if (run/'alias.json').exists():artifacts=load_selected_bank(Path(spec['source']['run_dir'])/'corrections/look'/pattern)
        else:artifacts=[MechanismArtifact.from_record(r) for r in torch.load(run/'artifacts.pt',map_location='cpu',weights_only=False)]
        result=evaluate_missing(graph,dl,device,fixed_pattern=pattern,artifact_banks={pattern:artifacts})
        path=out/'prediction.npz';save_prediction_bundle(result,path);records=[dict(method=task['arm'],scenario=pattern,path=str(path))]
        state_equal(graph,frozen)
    atomic_write_json(dict(model_id=model['id'],data_sha256=manifest['sha256'] if role=='test' else None,
        records=records,role=role,test_access=role=='test'),out/('test_access.json' if role=='test' else 'replay.json'))
    return records



def compare_replay(expected,actual,atol=1e-6,rtol=1e-5):
    """No new selection: require matching identities, decisions and numeric logits."""
    def index(records):
        result={(r['method'],r['scenario']):r for r in records}
        if len(result)!=len(records):raise ValueError('Duplicate replay view')
        return result
    e,a=index(expected),index(actual)
    if e.keys()!=a.keys():raise ValueError('Replay view coverage changed')
    for key,row in e.items():
        if file_sha256(row['path'])!=row['sha256']:raise ValueError('Original prediction changed')
        with np.load(row['path'],allow_pickle=False) as x,np.load(a[key]['path'],allow_pickle=False) as y:
            for field in ('participant_ids','labels','patterns'):
                if not np.array_equal(x[field],y[field]):raise ValueError('Replay '+field+' changed')
            if not np.allclose(x['logits'],y['logits'],atol=atol,rtol=rtol):raise ValueError('Replay logits changed')
            if not np.array_equal(x['logits'].argmax(1),y['logits'].argmax(1)):raise ValueError('Replay decisions changed')
    return dict(state='accepted',views=len(e),atol=atol,rtol=rtol,
        actual_files={r['path']:file_sha256(r['path']) for r in actual})


def replay_registry(registry,output,device):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    # This registry is separate from the continuously regenerated preparation.
    result=json.loads(json.dumps(registry));result['test_access']=False
    result['state']='development_replayed'
    for model in result['models']:
        run=Path(model['run_dir'])
        expected=read(run/'development/suite.json')['records'] if model['kind']=='original' else read(run/'records.json')
        actual=evaluate_model(model,None,out/model['id'],device,role='development')
        receipt=compare_replay(expected,actual)
        path=out/model['id']/'accepted.json';atomic_write_json(receipt,path)
        model.update(development_replay='accepted',development_replay_receipt=str(path),
            development_replay_sha256=file_sha256(path))
        model.pop('development_replay_reason',None)
        atomic_write_json(result,out/'models.json')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--feed');p.add_argument('--base-feed');p.add_argument('--output',required=True)
    p.add_argument('--replay-development',action='store_true');p.add_argument('--release');p.add_argument('--models');p.add_argument('--comparisons');p.add_argument('--data');p.add_argument('--audit')
    a=p.parse_args()
    if a.replay_development:
        replay_registry(read(a.models),a.output,torch.device('cuda:0'));return
    if not a.release:
        prepare(read(a.feed),read(a.base_feed),a.output);return
    models=read(a.models);data=read(a.data)
    validate_release(read(a.release),models,read(a.comparisons),data,read(a.audit))
    # Gate is checked before importing/constructing any test dataset.
    for model in models['models']:evaluate_model(model,data['diseases'][model['disease']],Path(a.output)/model['id'],torch.device('cuda:0'))

if __name__=='__main__':main()
