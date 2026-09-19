"""Strict fresh-host revalidation of cached fusion-stage family trees.

The original profile/fitting tree is immutable audit evidence. Revalidation
copies mutable metadata, hard-links only immutable tensor/prediction payloads,
verifies every committed prefix/candidate evidence record, and accepts a complete
cached tree only when the entire scientific selection projection and a fresh
final-bank replay are unchanged. Missing branches alone may be computed.
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


def saved_prediction_values_sha(path):
    with np.load(path,allow_pickle=False) as saved:
        return prediction_values_sha({name:saved[name] for name in ('participant_ids','labels','logits')})


def formal_runtime_fingerprint():
    value=dict(
        CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        torch_threads=torch.get_num_threads(),
        torch=str(torch.__version__),cuda=str(torch.version.cuda),
        cudnn=str(torch.backends.cudnn.version()),
    )
    required=dict(CUBLAS_WORKSPACE_CONFIG=':4096:8',deterministic_algorithms=True,
        cudnn_benchmark=False,matmul_tf32=False,cudnn_tf32=False,torch_threads=2)
    if any(value[k]!=v for k,v in required.items()):
        raise ValueError('Fusion revalidation runtime differs from formal deterministic worker')
    return value


def _tensor_sha(tensor):
    arr=tensor.detach().cpu().contiguous().numpy()
    h=hashlib.sha256();h.update(str(arr.dtype).encode());h.update(str(arr.shape).encode());h.update(arr.tobytes())
    return h.hexdigest()


def graph_state_projection(graph):
    # Ephemeral MHD message states are reset before capture; scientific module
    # parameters/buffers, BN state, modes, hooks and requires_grad must match.
    for node in graph.nodes:
        node.reset()
    graph.eval()
    state={key:_tensor_sha(value) for key,value in graph.state_dict().items()}
    modules={}
    bn={}
    for name,module in graph.named_modules():
        modules[name]=dict(training=module.training,forward_pre_hooks=len(module._forward_pre_hooks),
            forward_hooks=len(module._forward_hooks),backward_hooks=len(module._backward_hooks))
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
            bn[name]=dict(training=module.training,
                running_mean=None if module.running_mean is None else _tensor_sha(module.running_mean),
                running_var=None if module.running_var is None else _tensor_sha(module.running_var),
                num_batches_tracked=None if module.num_batches_tracked is None else int(module.num_batches_tracked.item()))
    value=dict(graph_training=graph.training,state_dict=state,batchnorm=bn,modules=modules,
        requires_grad={name:p.requires_grad for name,p in graph.named_parameters()})
    value['sha256']=stable_hash(value)
    return value


def evidence_projection(value):
    return dict(role=value['role'],data_role=value['data_role'],score=value['score'],
        values_sha256=value['values_sha256'],metrics=value['metrics'])


def candidate_projection(row):
    return dict(index=row['index'],node=row['node'],key=row['key'],score=row['score'],
        artifact=row['artifact'],sha256=row['sha256'],evidence=evidence_projection(row['evidence']))


def decision_projection(value):
    return dict(identity=value['identity'],path=[descriptor(row) for row in value['path']],
        baseline=evidence_projection(value['baseline']),
        candidates=[candidate_projection(row) for row in value['candidates']],
        children=[dict(path=[descriptor(row) for row in child['path']],
            winner=candidate_projection(child['winner'])) for child in value['children']],
        terminal=value['terminal'])


def selection_projection(selection):
    return dict(schema=selection['schema'],contract_sha256=selection['contract_sha256'],mode=selection['mode'],
        decisions=[decision_projection(value) for value in selection['decisions']],
        final=evidence_projection(selection['final']),
        selected_path=[descriptor(row) for row in selection['selected_path']],
        site_attempts=selection['site_attempts'],candidate_evaluations=selection['candidate_evaluations'],
        prefix_count=selection['prefix_count'],test_access=selection['test_access'],
        scientific_acceptance=selection['scientific_acceptance'])


def prefix_projection(root):
    root=Path(root);value={}
    for path in sorted((root/'prefixes').rglob('*.json')):
        rel=str(path.relative_to(root))
        row=json.loads(path.read_text())
        if path.name=='baseline.json':
            value[rel]=dict(identity=row['identity'],evidence=evidence_projection(row['evidence']))
        elif path.name.startswith('site_'):
            value[rel]=dict(identity=row['identity'],candidates=[candidate_projection(x) for x in row['candidates']])
        elif path.name=='decision.json':
            value[rel]=decision_projection(row)
    return value


def science_projection(root):
    root=Path(root)
    selection=json.loads((root/'selection.json').read_text())
    bank=torch.load(root/'bank.pt',map_location='cpu',weights_only=False)
    moments={}
    for path in sorted((root/'family_moments').glob('*.pt')):
        record=torch.load(path,map_location='cpu',weights_only=False)
        if fingerprint(record['payload'])!=record['sha256']:
            raise ValueError('Family moment payload fingerprint changed')
        moments[path.stem]=record['sha256']
    artifacts={str(path.relative_to(root)):file_sha256(path)
        for path in sorted((root/'prefixes').rglob('artifacts/*.pt'))}
    predictions={}
    for path in sorted((root/'predictions').glob('*.npz')):
        predictions[path.name]=saved_prediction_values_sha(path)
    tree_progress=json.loads((root/'tree_progress.json').read_text())
    value=dict(contract=json.loads((root/'contract.json').read_text()),
        selection=selection_projection(selection),tree_progress=tree_progress,
        prefix_evidence=prefix_projection(root),
        bank_fingerprint=fingerprint(bank['bank']),bank_identity=bank['identity'],
        moment_payload_fingerprints=moments,artifact_sha256=artifacts,
        prediction_values=predictions,replay=json.loads((root/'replay.json').read_text()))
    value['sha256']=stable_hash(value)
    return value


def evaluate_bank(graph,loader,device,pattern,bank,root):
    formal_runtime_fingerprint()
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


def _audit_copy(src,dst):
    srcp=Path(src)
    # JSON/status/lock metadata can be rewritten during revalidation: give it an
    # independent inode. Immutable tensor/prediction payloads may be hard-linked.
    if srcp.suffix in ('.pt','.npz'):
        os.link(src,dst)
    else:
        shutil.copy2(src,dst)
    return dst


def copy_audit_tree(source,target):
    source,target=Path(source),Path(target)
    if target.exists():
        return
    if not source.exists():
        target.mkdir(parents=True)
        return
    shutil.copytree(source,target,copy_function=_audit_copy)
    # Rebuild selection/replay authority from refreshed evidence.
    for name in AUTHORITY_FILES:
        (target/name).unlink(missing_ok=True)
    for path in target.glob('prefixes/**/decision.json'):
        path.unlink()


def refresh_cached_evidence(source,target,graph,loader,device,pattern):
    source,target=Path(source),Path(target)
    if not source.exists() or not (source/'selection.json').exists():
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


def assert_references_within_tree(root):
    root=Path(root).resolve()
    def walk(value):
        if isinstance(value,dict):
            if 'prediction' in value:
                path=Path(value['prediction']).resolve()
                if not path.is_relative_to(root) or not path.exists():
                    raise ValueError('Revalidated prediction reference escaped its tree')
                if value.get('values_sha256')!=saved_prediction_values_sha(path):
                    raise ValueError('Revalidated prediction values changed')
            if 'artifact' in value and 'sha256' in value:
                path=(root/value['artifact']).resolve()
                if not path.is_relative_to(root) or file_sha256(path)!=value['sha256']:
                    raise ValueError('Revalidated artifact reference changed')
            for child in value.values():walk(child)
        elif isinstance(value,list):
            for child in value:walk(child)
    walk(json.loads((root/'selection.json').read_text()))
    for path in (root/'prefixes').rglob('*.json'):
        walk(json.loads(path.read_text()))


def revalidate_or_fit(*,source,target,graph,loader,device,arm,pattern,sites,factor,pca_bank,identity,
                      workspace_bytes,should_pause,load_fresh_graph):
    source,target=Path(source),Path(target)
    audit=target.parent/'audit'/target.name
    runtime=formal_runtime_fingerprint()
    start_state=graph_state_projection(graph)
    source_manifest=raw_manifest(source) if source.exists() else {}
    source_manifest_sha=write_manifest(source_manifest,audit/'source_raw_manifest.json')
    source_complete=all((source/name).exists() for name in ('selection.json','bank.pt','tree_progress.json','replay.json'))
    source_science=science_projection(source) if source_complete else None
    copy_audit_tree(source,target)
    cache_verification=dict(mode='verify_committed_prefix_candidate_evidence_without_rewrite',
        source_complete=source_complete,
        source_decisions=0 if not source_complete else len(json.loads((source/'selection.json').read_text())['decisions']))
    bank,result=fit_family_trajectory(graph,loader('train'),loader('development'),arm=arm,pattern=pattern,sites=sites,
        factor=factor,candidates=[dict(rank=32,ridge_lambda=None)],pca_bank=pca_bank,identity=identity,
        output=target,device=device,workspace_bytes=workspace_bytes,mode='positive_forward_tree',
        should_pause=should_pause,penalty_policy='prefix_train_pca_gcv')
    end_state=graph_state_projection(graph)
    if end_state!=start_state:
        raise ValueError('Fusion revalidation changed accepted host parameters/buffers/modes/hooks')
    fresh=load_fresh_graph()
    fresh_state=graph_state_projection(fresh)
    if fresh_state!=start_state:
        raise ValueError('Fresh accepted host state differs during fusion revalidation')
    replay=evaluate_missing(fresh,loader('development'),device,fixed_pattern=pattern,artifact_banks={pattern:bank})
    values_sha=prediction_values_sha(replay)
    if values_sha!=result['final']['values_sha256'] or replay['metrics']!=result['final']['metrics']:
        raise ValueError('Fresh-host revalidated selection is not independently replay-exact')
    assert_references_within_tree(target)
    target_science=science_projection(target)
    if source_science is not None and target_science!=source_science:
        raise ValueError('Fresh revalidation changed complete prefix/candidate/selection scientific evidence')
    source_manifest_final=raw_manifest(source) if source.exists() else {}
    if source_manifest_final!=source_manifest:
        raise ValueError('Fusion source audit tree changed during revalidation')
    source_manifest_final_sha=write_manifest(source_manifest_final,audit/'source_raw_manifest_final.json')
    target_manifest=raw_manifest(target)
    target_manifest_sha=write_manifest(target_manifest,audit/'revalidated_raw_manifest.json')
    feature_costs=read_json=lambda p: json.loads(p.read_text()) if p.exists() else {}
    receipt=dict(schema='look_fusion_fresh_revalidation_v2',state='accepted',identity=identity,test_access=False,
        arm=arm,pattern=pattern,source_tree_present=source.exists(),source_complete=source_complete,
        source_raw_manifest_sha256=source_manifest_sha,source_raw_manifest_final_sha256=source_manifest_final_sha,
        source_raw_manifest_unchanged=True,revalidated_raw_manifest_sha256=target_manifest_sha,
        cache_verification=cache_verification,
        mutable_operational_fields=MUTABLE_OPERATIONAL_FIELDS,runtime=runtime,
        graph_state_sha256=start_state['sha256'],graph_state_exact_before_after_and_fresh=True,
        source_science_sha256=None if source_science is None else source_science['sha256'],
        revalidated_science_sha256=target_science['sha256'],
        complete_source_science_exact=None if source_science is None else True,
        feature_costs_source=feature_costs(source/'feature_costs.json'),
        feature_costs_revalidated=feature_costs(target/'feature_costs.json'),
        final_values_sha256=values_sha,final_metrics=replay['metrics'],
        participant_count=len(replay['participant_ids']),
        participant_ids_sha256=hashlib.sha256(np.ascontiguousarray(replay['participant_ids']).tobytes()).hexdigest(),
        references_within_revalidated_tree=True,no_host_retraining=True)
    atomic_write_json(receipt,audit/'accepted.json')
    return bank,result,receipt
