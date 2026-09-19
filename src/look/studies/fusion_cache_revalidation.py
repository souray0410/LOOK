"""Fresh-host revalidation of a cached fusion-stage family tree.

The original profile/fitting tree is immutable audit evidence.  Revalidation
creates a separate tree, hard-links immutable artifacts/moments, recomputes all
cached development evidence on a freshly loaded accepted host, and lets the
unchanged positive-forward-tree algorithm extend only newly required branches.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from look.evaluation.evaluator import evaluate_missing, save_prediction_bundle
from look.methods.affine_family import FamilyArtifact
from look.methods.family_greedy import fit_family_trajectory
from look.methods.linear_operator import fingerprint
from look.methods.positive_forward_tree import prefix_identity
from look.runtime.state import atomic_write_json, file_sha256, stable_hash


AUTHORITY_FILES=('selection.json','tree_progress.json','bank.pt','replay.json','feature_costs.json')
MUTABLE_OPERATIONAL_FIELDS={
    'feature_costs.json':['full_forwards','missing_forwards','completed_hits','skipped_batches'],
    'tree_progress.json':['state','completed_prefixes','site_attempts','candidate_evaluations','best_score','best_path'],
}


def raw_manifest(root):
    root=Path(root)
    return {str(path.relative_to(root)):file_sha256(path)
            for path in sorted(root.rglob('*')) if path.is_file()}


def write_manifest(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    atomic_write_json(value,path)
    return file_sha256(path)


def descriptor(row):
    return {key:row[key] for key in ('index','node','key','artifact','sha256')}


def prediction_values_sha(result):
    h=hashlib.sha256()
    for name in ('participant_ids','labels','logits'):
        arr=np.ascontiguousarray(result[name])
        h.update(name.encode());h.update(str(arr.dtype).encode());h.update(str(arr.shape).encode());h.update(arr.tobytes())
    return h.hexdigest()


def evaluate_bank(graph,loader,device,pattern,bank,root):
    dev_loader=loader('development')
    result=evaluate_missing(graph,dev_loader,device,fixed_pattern=pattern,artifact_banks={pattern:bank})
    values_sha=prediction_values_sha(result)
    key=fingerprint([artifact.record() for artifact in bank])
    path=Path(root)/'predictions'/f'{key}_{values_sha}.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as saved:
            if any(not np.array_equal(saved[name],result[name]) for name in ('participant_ids','labels','logits')):
                raise ValueError('Fresh revalidation prediction evidence changed')
    else:
        path.parent.mkdir(parents=True,exist_ok=True);save_prediction_bundle(result,path)
    return dict(role='development',data_role=dev_loader.dataset.split,score=float(result['metrics']['macro_f1']),
        prediction=str(path),sha256=file_sha256(path),values_sha256=values_sha,metrics=result['metrics'])


def load_artifact(root,row):
    path=(Path(root)/row['artifact']).resolve()
    if not path.is_relative_to(Path(root).resolve()) or file_sha256(path)!=row['sha256']:
        raise ValueError('Cached family artifact changed')
    return FamilyArtifact.from_record(torch.load(path,map_location='cpu',weights_only=False))


def copy_audit_tree(source,target):
    source,target=Path(source),Path(target)
    if target.exists():
        return
    if not source.exists():
        target.mkdir(parents=True)
        return
    shutil.copytree(source,target,copy_function=os.link)
    # These are outputs of the selection/replay process, not immutable inputs.
    for name in AUTHORITY_FILES:
        (target/name).unlink(missing_ok=True)
    for path in target.glob('prefixes/**/decision.json'):
        path.unlink()


def refresh_cached_evidence(source,target,graph,loader,device,pattern):
    source,target=Path(source),Path(target)
    if not source.exists():
        return dict(source_decisions=0,refreshed_evaluations=0)
    selection=json.loads((source/'selection.json').read_text())
    contract=json.loads((target/'contract.json').read_text())
    memo={}
    def evidence(path_rows):
        key=stable_hash([descriptor(row) for row in path_rows])
        if key not in memo:
            bank=[load_artifact(target,row) for row in path_rows]
            memo[key]=evaluate_bank(graph,loader,device,pattern,bank,target)
        return memo[key]
    for decision in selection['decisions']:
        path_rows=decision['path']
        pid=prefix_identity(contract,path_rows)
        folder=target/'prefixes'/('root' if not path_rows else pid)
        baseline_path=folder/'baseline.json'
        baseline=json.loads(baseline_path.read_text())
        if baseline.get('identity')!=pid:
            raise ValueError('Cached prefix identity changed')
        baseline['evidence']=evidence(path_rows);atomic_write_json(baseline,baseline_path)
        for site_path in sorted(folder.glob('site_*.json')):
            cache=json.loads(site_path.read_text())
            for row in cache['candidates']:
                fresh=evidence([*path_rows,descriptor(row)])
                row['score']=fresh['score'];row['evidence']=fresh
            atomic_write_json(cache,site_path)
    return dict(source_decisions=len(selection['decisions']),refreshed_evaluations=len(memo))


def science_projection(root):
    root=Path(root);selection=json.loads((root/'selection.json').read_text())
    bank=torch.load(root/'bank.pt',map_location='cpu',weights_only=False)
    moments={}
    for path in sorted((root/'family_moments').glob('*.pt')):
        record=torch.load(path,map_location='cpu',weights_only=False)
        if fingerprint(record['payload'])!=record['sha256']:
            raise ValueError('Family moment payload fingerprint changed')
        moments[path.stem]=record['sha256']
    predictions={}
    for path in sorted((root/'predictions').glob('*.npz')):
        with np.load(path,allow_pickle=False) as saved:
            h=hashlib.sha256()
            for name in ('participant_ids','labels','logits'):
                arr=np.ascontiguousarray(saved[name]);h.update(name.encode());h.update(str(arr.dtype).encode());h.update(str(arr.shape).encode());h.update(arr.tobytes())
            predictions[path.name]=h.hexdigest()
    return dict(contract_sha256=file_sha256(root/'contract.json'),
        selected_path=selection['selected_path'],final=selection['final'],
        bank_fingerprint=fingerprint(bank['bank']),bank_identity=bank['identity'],
        moment_payload_fingerprints=moments,prediction_values=predictions)


def revalidate_or_fit(*,source,target,graph,loader,device,arm,pattern,sites,factor,pca_bank,identity,
                      workspace_bytes,should_pause,load_fresh_graph):
    source,target=Path(source),Path(target)
    audit=target.parent/'audit'/target.name
    source_manifest=raw_manifest(source) if source.exists() else {}
    source_manifest_sha=write_manifest(source_manifest,audit/'source_raw_manifest.json')
    copy_audit_tree(source,target)
    refresh=refresh_cached_evidence(source,target,graph,loader,device,pattern)
    bank,result=fit_family_trajectory(graph,loader('train'),loader('development'),arm=arm,pattern=pattern,sites=sites,
        factor=factor,candidates=[dict(rank=32,ridge_lambda=None)],pca_bank=pca_bank,identity=identity,
        output=target,device=device,workspace_bytes=workspace_bytes,mode='positive_forward_tree',
        should_pause=should_pause,penalty_policy='prefix_train_pca_gcv')
    # fit_family_trajectory already requires serialized-bank replay exact.  Repeat on
    # a newly loaded accepted host to reject any process-history-dependent result.
    fresh=load_fresh_graph()
    replay=evaluate_missing(fresh,loader('development'),device,fixed_pattern=pattern,artifact_banks={pattern:bank})
    values_sha=prediction_values_sha(replay)
    if values_sha!=result['final']['values_sha256'] or replay['metrics']!=result['final']['metrics']:
        raise ValueError('Fresh-host revalidated selection is not independently replay-exact')
    target_manifest=raw_manifest(target)
    target_manifest_sha=write_manifest(target_manifest,audit/'revalidated_raw_manifest.json')
    projection=science_projection(target)
    receipt=dict(schema='look_fusion_fresh_revalidation_v1',state='accepted',identity=identity,test_access=False,
        arm=arm,pattern=pattern,source_tree_present=source.exists(),source_raw_manifest_sha256=source_manifest_sha,
        revalidated_raw_manifest_sha256=target_manifest_sha,refresh=refresh,
        mutable_operational_fields=MUTABLE_OPERATIONAL_FIELDS,science=projection,
        final_values_sha256=values_sha,final_metrics=replay['metrics'],
        participant_count=len(replay['participant_ids']),
        participant_ids_sha256=hashlib.sha256(np.ascontiguousarray(replay['participant_ids']).tobytes()).hexdigest(),
        no_host_retraining=True)
    atomic_write_json(receipt,audit/'accepted.json')
    return bank,result,receipt
