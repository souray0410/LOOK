import copy
import json
from pathlib import Path

import numpy as np
import pytest

from look.analysis import fusion_stage_publication as pub
from look.runtime.state import file_sha256, stable_hash
from look.studies import cohort_delivery, cohort_fusion_stage as fusion
from look.training.observed_host import DEFAULTS


def base_spec(tmp_path, position='middle'):
    data=tmp_path/'data';data.mkdir(exist_ok=True)
    (data/'accepted.json').write_text('{}')
    init=tmp_path/'init.pt';init.write_bytes(b'init')
    source=tmp_path/'source.py';source.write_text('source')
    return dict(
        schema='look_fresh_cohort_delivery_v1',run_id='run-'+position,source_commit='source',
        framework_commit='framework',architecture='resnet18',position=position,seed=3416,
        factor=16,rank=32,arms=['residual_rrr','pca_free_mean'],search='positive_forward_tree',
        data_root=str(data),data_audit_sha256=file_sha256(data/'accepted.json'),test_access=False,
        devices=[0],lock_root=str(tmp_path/'locks'),
        initialization=dict(kind='public_imagenet_fresh_host',path=str(init),sha256=file_sha256(init)),
        training=dict(DEFAULTS),source_pins=[dict(path=str(source),sha256=file_sha256(source))],
        gpu_budget_bytes=100,gpu_reserve_bytes=100,ram_budget_bytes=1000,workspace_bytes=100,
        disk_reserve_bytes=50*1024**3,
    )


def register(spec):
    spec=copy.deepcopy(spec)
    spec.update(study_kind=fusion.STUDY_KIND,fusion_stage_task=fusion.TASK_ID,
        fusion_stage_reference=dict(run_id=fusion.DEEP_RUN_ID,spec_sha256=fusion.DEEP_SPEC_SHA256,
            delivery_sha256=fusion.DEEP_DELIVERY_SHA256),
        fusion_stage_structure=fusion.host_structure(spec['position']))
    return spec


def test_registered_member_only_opens_middle_and_features(tmp_path):
    ordinary=base_spec(tmp_path,'middle')
    with pytest.raises(ValueError,match='Unregistered scope'):
        cohort_delivery.validate(ordinary)
    cohort_delivery.validate(register(ordinary))
    feature=base_spec(tmp_path,'features')
    cohort_delivery.validate(register(feature))
    bad=register(ordinary);bad['architecture']='resnet34'
    with pytest.raises(ValueError,match='member scope'):
        cohort_delivery.validate(bad)
    bad=register(ordinary);bad['mmtm']={'stage':'stage3','ratio':4,'gate_scale':1.0}
    with pytest.raises(ValueError,match='member changed'):
        cohort_delivery.validate(bad)
    bad=register(ordinary);bad['disk_reserve_bytes']=0
    with pytest.raises(ValueError,match='disk reserve'):
        cohort_delivery.validate(bad)


def test_host_structure_contract_is_position_specific():
    middle=fusion.host_structure('middle')
    deep=fusion.host_structure('deep')
    features=fusion.host_structure('features')
    assert middle['fusion_endpoint']=='stage2'
    assert deep['fusion_endpoint']=='stage4'
    assert features['fusion_endpoint']=='features'
    assert middle['correction_sites']==[
        'joint_input','joint_stem','joint_stage1','joint_stage2','fusion_stage2',
        'fusion_stage3','fusion_stage4','fusion_features','fusion_participant_feature']
    assert deep['correction_sites']==[
        'joint_input','joint_stem','joint_stage1','joint_stage2','joint_stage3',
        'joint_stage4','fusion_stage4','fusion_features','fusion_participant_feature']
    assert features['correction_sites']==[
        'joint_input','joint_stem','joint_stage1','joint_stage2','joint_stage3',
        'joint_stage4','joint_features','fusion_features','fusion_participant_feature']
    assert middle['parameters']==11893634
    assert deep['parameters']==features['parameters']==22879362
    assert middle['parameters']<deep['parameters']
    assert deep['architecture_id']!=features['architecture_id']


def test_cross_host_family_is_exactly_eight_prespecified_contrasts():
    expected=[
        (position,method,pattern)
        for position in ('middle','features')
        for method in ('residual_rrr','pca_free_mean')
        for pattern in ('oct_missing','cfp_missing')
    ]
    assert [(r['position'],r['method'],r['pattern']) for r in fusion.CROSS_HOST_COMPARISONS]==expected
    assert all(r['reference_position']=='deep' for r in fusion.CROSS_HOST_COMPARISONS)


def _write_predictions(root,spec,reverse_ids=False):
    root=Path(root);ids=np.array(['a','b','c','d']);labels=np.array([0,0,1,1])
    if reverse_ids:ids=ids[::-1]
    logits=np.array([[2.,0.],[1.,0.],[0.,1.],[0.,2.]])
    for arm in fusion.METHODS:
        folder=root/arm;folder.mkdir(parents=True,exist_ok=True);records=[]
        for pattern in fusion.PATTERNS:
            for method in (arm,'host'):
                path=folder/f'{method}_{pattern}.npz'
                np.savez(path,participant_ids=ids,labels=labels,logits=logits)
                records.append(dict(method=method,scenario=pattern,path=str(path),sha256=file_sha256(path),metrics={'macro_f1':1.0}))
        (folder/'accepted.json').write_text(json.dumps(dict(state='accepted',identity=stable_hash(spec),records=records)))


def test_cross_host_statistics_requires_same_participant_order(tmp_path):
    specs=[]
    for position in fusion.POSITIONS:
        root=tmp_path/position;root.mkdir()
        spec={'run_id':position,'output':str(root),'position':position}
        specs.append(spec);_write_predictions(root,spec)
    plan={'cross_host_comparisons':fusion.CROSS_HOST_COMPARISONS}
    stats=pub._cross_host_statistics(plan,specs)
    assert stats['participants']==4 and len(stats['contrasts'])==8
    assert all(row['difference']==0 for row in stats['contrasts'])
    _write_predictions(Path(specs[-1]['output']),specs[-1],reverse_ids=True)
    with pytest.raises(ValueError,match='participant order'):
        pub._cross_host_statistics(plan,specs)

from look.studies import cohort_sequence


class ImmediateChild:
    def __init__(self, code):
        self.returncode=code
    def poll(self):
        return self.returncode


def test_sequence_never_launches_deep_reference_and_orders_new_hosts(tmp_path,monkeypatch):
    specs=[]
    tasks=[]
    for role,position in [('accepted_reference','deep'),('execute','middle'),('execute','features')]:
        out=tmp_path/position;out.mkdir()
        spec={'run_id':position,'output':str(out),'position':position}
        spec_path=tmp_path/(position+'.json');spec_path.write_text(json.dumps(spec))
        specs.append(spec);tasks.append({'role':role,'position':position,'spec':str(spec_path)})
    (Path(specs[0]['output'])/'delivery').mkdir()
    (Path(specs[0]['output'])/'delivery/accepted.json').write_text('{}')
    plan={'root':str(tmp_path/'sequence'),'publication':str(tmp_path/'publication'),'tasks':tasks}
    monkeypatch.setattr(cohort_sequence,'validate_sequence',lambda p:specs)
    monkeypatch.setattr(cohort_sequence,'publish',lambda root,out:{'matched_package':'accepted'})
    monkeypatch.setattr(cohort_sequence,'refresh',lambda plan,statuses:None)
    launched=[]
    def launch(spec,row):
        launched.append(spec['run_id']);return ImmediateChild(0)
    cohort_sequence.run(plan,launch=launch,interval=0)
    assert launched==['middle','features']
    status=json.loads((tmp_path/'sequence/status.json').read_text())
    assert status['tasks']['deep']['state']=='accepted_reference'
    assert status['tasks']['middle']['state']==status['tasks']['features']['state']=='accepted'
    assert json.loads((tmp_path/'sequence/completion.json').read_text())['state']=='accepted'


def test_sequence_failure_isolated_and_features_still_runs(tmp_path,monkeypatch):
    specs=[];tasks=[]
    for role,position in [('accepted_reference','deep'),('execute','middle'),('execute','features')]:
        out=tmp_path/position;out.mkdir()
        spec={'run_id':position,'output':str(out),'position':position}
        spec_path=tmp_path/(position+'.json');spec_path.write_text(json.dumps(spec))
        specs.append(spec);tasks.append({'role':role,'position':position,'spec':str(spec_path)})
    (Path(specs[0]['output'])/'delivery').mkdir()
    (Path(specs[0]['output'])/'delivery/accepted.json').write_text('{}')
    plan={'root':str(tmp_path/'sequence'),'publication':str(tmp_path/'publication'),'tasks':tasks}
    monkeypatch.setattr(cohort_sequence,'validate_sequence',lambda p:specs)
    monkeypatch.setattr(cohort_sequence,'publish',lambda root,out:{'matched_package':'accepted'})
    monkeypatch.setattr(cohort_sequence,'refresh',lambda plan,statuses:None)
    launched=[]
    def launch(spec,row):
        launched.append(spec['run_id']);return ImmediateChild(1 if spec['run_id']=='middle' else 0)
    cohort_sequence.run(plan,launch=launch,interval=0)
    assert launched==['middle','features']
    status=json.loads((tmp_path/'sequence/status.json').read_text())
    assert status['tasks']['middle']['state']=='needs_review'
    assert status['tasks']['features']['state']=='accepted'
    assert json.loads((tmp_path/'sequence/completion.json').read_text())['state']=='needs_review'


@pytest.mark.parametrize('position,needle',[
    ('middle','Stage2 ┐'),('deep','deep融合'),('features','features融合')])
def test_fusion_diagram_matches_position(position,needle):
    from look.analysis.cohort_publication import fusion_diagram
    assert needle in '\n'.join(fusion_diagram(position))

def test_runtime_structure_uses_same_schema_as_prespecified():
    from types import SimpleNamespace
    from mhd_framework.models import create_model
    from look.models.native_host import build_native_host
    cfg=dict(name='resnet18',spatial_dims=2,in_channels=3,num_classes=2,views=1,granularity='block')
    first=create_model(cfg,weights=None);second=create_model(cfg,weights=None)
    graph=build_native_host(SimpleNamespace(graph=first),SimpleNamespace(graph=second),'middle','cpu')
    actual=fusion.runtime_host_structure(graph,'middle')
    assert actual==fusion.host_structure('middle')
    wrong=dict(actual);wrong['position']='deep'
    assert wrong!=fusion.host_structure('middle')


def test_revalidation_copy_preserves_raw_tree_and_drops_old_authority(tmp_path):
    from look.studies.fusion_cache_revalidation import copy_audit_tree,raw_manifest
    source=tmp_path/'raw';source.mkdir()
    (source/'family_moments').mkdir();(source/'family_moments/a.pt').write_bytes(b'moment')
    (source/'prefixes/root').mkdir(parents=True);(source/'prefixes/root/site_000.json').write_text('{}')
    (source/'prefixes/root/decision.json').write_text('{}')
    for name in ('selection.json','tree_progress.json','bank.pt','replay.json','feature_costs.json'):
        (source/name).write_bytes(b'old')
    before=raw_manifest(source)
    target=tmp_path/'revalidated';copy_audit_tree(source,target)
    assert raw_manifest(source)==before
    assert (target/'family_moments/a.pt').exists() and (target/'prefixes/root/site_000.json').exists()
    assert not (target/'prefixes/root/decision.json').exists()
    for name in ('selection.json','tree_progress.json','bank.pt','replay.json','feature_costs.json'):
        assert not (target/name).exists()


def test_verify_formal_arm_requires_logical_v2_migration(tmp_path):
    from look.studies import cohort_delivery as delivery
    root=tmp_path/'run';root.mkdir()
    spec={'x':1};identity=stable_hash(spec)
    (root/'host').mkdir();(root/'host/accepted.json').write_text(json.dumps({'files':{'best.pt':'hostsha'}}))
    profile=root/'profile/fitting';profile.mkdir(parents=True);(profile/'accepted.json').write_text('{}')
    correction=root/'profile/revalidated/residual_rrr/oct_missing';correction.mkdir(parents=True)
    (correction/'selection.json').write_text(json.dumps({'selected_path':[],'final':{'values_sha256':'v'}}))
    (correction/'bank.pt').write_bytes(b'bank')
    audit=correction.parent/'audit'/correction.name;audit.mkdir(parents=True)
    runtime={'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    revalidation=dict(schema='look_fusion_fresh_revalidation_v2',state='accepted',identity=identity,test_access=False,
        source_raw_manifest_unchanged=True,graph_state_exact_before_after_and_fresh=True,
        references_within_revalidated_tree=True,relocation_no_scientific_value_change=True,
        relocation_receipt_sha256='reloc',relocation_mapping_count=2,complete_source_science_exact=True,
        revalidated_science_sha256='science',source_science_sha256='source',
        runtime=runtime,graph_state_sha256='graph',
        source_raw_manifest_sha256='manifest',source_raw_manifest_final_sha256='manifest')
    (audit/'accepted.json').write_text(json.dumps(revalidation))
    out=root/'residual_rrr';(out/'development').mkdir(parents=True)
    pred=out/'development/residual_rrr_oct_missing.npz';pred.write_bytes(b'pred')
    row=dict(revalidated_root=str(correction.relative_to(root)),
        revalidation_receipt=str((audit/'accepted.json').relative_to(root)),
        revalidation_receipt_sha256=file_sha256(audit/'accepted.json'),
        selection_sha256=file_sha256(correction/'selection.json'),bank_sha256=file_sha256(correction/'bank.pt'),
        selected_path=[],final_values_sha256='v',revalidated_science_sha256='science',
        source_science_sha256='source',runtime_sha256=stable_hash(runtime),graph_state_sha256='graph',
        source_raw_manifest_sha256='manifest',source_raw_manifest_final_sha256='manifest',
        relocation_receipt_sha256='reloc',relocation_mapping_count=2)
    migration=dict(schema='look_fusion_profile_migration_v2',no_refit=True,no_physical_relocation=True,
        source_role='fresh_revalidated_same_run_same_identity',profile_receipt_sha256=file_sha256(profile/'accepted.json'),
        patterns={'oct_missing':row,'cfp_missing':dict(row)})
    # create second referenced path for cfp to satisfy coverage without sharing missing files
    migration['patterns']['cfp_missing']=dict(row)
    receipt=dict(state='accepted',identity=identity,test_access=False,host_best_sha256='hostsha',replay_exact=True,
        records=[dict(path=str(pred),sha256=file_sha256(pred))],profile_migration=migration)
    (out/'accepted.json').write_text(json.dumps(receipt))
    assert delivery.verify_fusion_formal_arm(spec,root,'residual_rrr')['profile_migration']['no_refit']
    bad=json.loads((out/'accepted.json').read_text());bad['profile_migration']['no_physical_relocation']=False
    (out/'accepted.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='logical profile migration'):
        delivery.verify_fusion_formal_arm(spec,root,'residual_rrr')


def test_needs_review_requires_one_exact_repair_marker(tmp_path,monkeypatch):
    from look.studies import cohort_sequence
    spec={'run_id':'middle','output':str(tmp_path/'middle'),'position':'middle'}
    Path(spec['output']).mkdir()
    spec_path=tmp_path/'middle.json';spec_path.write_text(json.dumps(spec))
    plan={'study_kind':'fusion_stage_v1','root':str(tmp_path/'sequence'),
        'publication':str(tmp_path/'publication'),
        'tasks':[{'role':'execute','position':'middle','spec':str(spec_path)}]}
    root=Path(plan['root']);root.mkdir()
    identity=stable_hash(plan)
    (Path(spec['output'])/'pipeline_status.json').write_text(json.dumps({'state':'needs_review','error':'old'}))
    (root/'status.json').write_text(json.dumps({'identity':identity,'tasks':{'middle':{'state':'needs_review'}}}))
    monkeypatch.setattr(cohort_sequence,'validate_sequence',lambda p:[spec])
    monkeypatch.setattr(cohort_sequence,'refresh',lambda plan,statuses:None)
    monkeypatch.setattr(cohort_sequence,'publish',lambda root,out:{'matched_package':'accepted'})
    launched=[]
    cohort_sequence.run(plan,launch=lambda s,row: launched.append(s['run_id']) or ImmediateChild(0),interval=0)
    assert launched==[]
    pipeline=Path(spec['output'])/'pipeline_status.json'
    marker=dict(schema='look_fusion_stage_repair_resume_v2',task_id=fusion.TASK_ID,run_id='middle',
        spec_sha256=file_sha256(spec_path),prior_pipeline_status_sha256=file_sha256(pipeline),
        repair_packet_sha256=fusion.REPAIR_0207_SHA256,test_access=False)
    (Path(spec['output'])/'repair_resume_0207.json').write_text(json.dumps(marker))
    overlay={'ok':True};(root/'management_overlay.json').write_text(json.dumps(overlay))
    (Path(spec['output'])/'management_overlay.json').write_text(json.dumps(overlay))
    monkeypatch.setattr(fusion,'verify_management_overlay',lambda path:overlay)
    cohort_sequence.run(plan,launch=lambda s,row: launched.append(s['run_id']) or ImmediateChild(0),interval=0)
    assert launched==['middle']
    status=json.loads((root/'status.json').read_text())
    assert status['tasks']['middle']['state']=='accepted'
    assert list((root/'incidents').glob('middle_before_repair_*.json'))

def test_fresh_evaluate_bank_uses_explicit_development_loader(tmp_path,monkeypatch):
    from look.studies import fusion_cache_revalidation as fresh
    calls=[]
    class Loader:
        class Dataset:
            split='development'
        dataset=Dataset()
    def loader(role):
        calls.append(role)
        assert role=='development'
        return Loader()
    result=dict(participant_ids=np.array(['a','b']),labels=np.array([0,1]),
        logits=np.array([[2.,0.],[0.,2.]]),metrics={'macro_f1':1.0})
    monkeypatch.setattr(fresh,'formal_runtime_fingerprint',lambda:{'formal':True})
    monkeypatch.setattr(fresh,'evaluate_missing',lambda *a,**k:result)
    monkeypatch.setattr(fresh,'save_prediction_bundle',
        lambda value,path: np.savez(path,participant_ids=value['participant_ids'],labels=value['labels'],logits=value['logits']))
    value=fresh.evaluate_bank(None,loader,'cpu','oct_missing',[],tmp_path)
    assert calls==['development']
    assert value['data_role']=='development' and value['score']==1.0


def test_revalidation_operational_fields_do_not_hide_tree_selection():
    from look.studies import fusion_cache_revalidation as fresh
    assert set(fresh.MUTABLE_OPERATIONAL_FIELDS)=={'feature_costs.json'}
    assert 'best_path' not in sum(fresh.MUTABLE_OPERATIONAL_FIELDS.values(),[])
    selection=dict(schema='look_positive_forward_tree_v1',contract_sha256='c',mode='positive_forward_tree',
        decisions=[dict(identity='i',path=[],baseline=dict(role='development',data_role='development',score=.5,
            values_sha256='a',metrics={'macro_f1':.5}),
            candidates=[dict(index=0,node='n',key='k',score=.6,artifact='a.pt',sha256='s',
                evidence=dict(role='development',data_role='development',score=.6,values_sha256='b',
                    metrics={'macro_f1':.6}))],
            children=[],terminal=True)],
        final=dict(role='development',data_role='development',score=.6,values_sha256='b',metrics={'macro_f1':.6}),
        selected_path=[dict(index=0,node='n',key='k',artifact='a.pt',sha256='s')],
        site_attempts=1,candidate_evaluations=1,prefix_count=1,test_access=False,scientific_acceptance=False)
    projected=fresh.selection_projection(selection)
    changed=json.loads(json.dumps(selection));changed['decisions'][0]['candidates'][0]['score']=.61
    assert fresh.selection_projection(changed)!=projected


def test_revalidation_copy_separates_mutable_json_and_preserves_source_manifest(tmp_path):
    from look.studies.fusion_cache_revalidation import copy_audit_tree,raw_manifest
    from look.runtime.state import atomic_write_json
    import os
    source=tmp_path/'source';source.mkdir()
    (source/'prefixes/root').mkdir(parents=True)
    (source/'prefixes/root/site_000.json').write_text(json.dumps({'identity':'x','candidates':[]}))
    (source/'family_moments').mkdir();(source/'family_moments/a.pt').write_bytes(b'immutable')
    before=raw_manifest(source)
    target=tmp_path/'target';copy_audit_tree(source,target)
    assert os.stat(source/'prefixes/root/site_000.json').st_ino!=os.stat(target/'prefixes/root/site_000.json').st_ino
    assert os.stat(source/'family_moments/a.pt').st_ino==os.stat(target/'family_moments/a.pt').st_ino
    atomic_write_json({'identity':'x','candidates':[{'changed':True}]},target/'prefixes/root/site_000.json')
    assert raw_manifest(source)==before


def test_formal_runtime_fingerprint_requires_exact_worker_flags(monkeypatch):
    from look.studies import fusion_cache_revalidation as fresh
    import torch
    old=dict(deterministic=torch.are_deterministic_algorithms_enabled(),
        benchmark=torch.backends.cudnn.benchmark,matmul=torch.backends.cuda.matmul.allow_tf32,
        cudnn=torch.backends.cudnn.allow_tf32,threads=torch.get_num_threads())
    try:
        monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        value=fresh.formal_runtime_fingerprint()
        assert value['CUBLAS_WORKSPACE_CONFIG']==':4096:8' and value['deterministic_algorithms'] is True
        monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':16:8')
        with pytest.raises(ValueError,match='formal deterministic worker'):
            fresh.formal_runtime_fingerprint()
    finally:
        torch.set_num_threads(old['threads']);torch.use_deterministic_algorithms(old['deterministic'])
        torch.backends.cudnn.benchmark=old['benchmark'];torch.backends.cuda.matmul.allow_tf32=old['matmul']
        torch.backends.cudnn.allow_tf32=old['cudnn']


def test_relocation_map_moves_prediction_references_without_changing_values_or_source(tmp_path):
    from look.studies import fusion_cache_revalidation as fresh
    source=tmp_path/'source';target=tmp_path/'target';audit=tmp_path/'audit'
    (source/'prefixes/root').mkdir(parents=True);(source/'predictions').mkdir()
    ids=np.array(['a','b']);labels=np.array([0,1]);logits=np.array([[2.,0.],[0.,2.]])
    pred=source/'predictions/p.npz';np.savez(pred,participant_ids=ids,labels=labels,logits=logits)
    result=dict(participant_ids=ids,labels=labels,logits=logits)
    values=fresh.prediction_values_sha(result)
    evidence=dict(role='development',data_role='development',score=1.0,prediction=str(pred),
        sha256=file_sha256(pred),values_sha256=values,metrics={'macro_f1':1.0})
    (source/'prefixes/root/baseline.json').write_text(json.dumps({'identity':'x','evidence':evidence}))
    before=fresh.raw_manifest(source)
    fresh.copy_audit_tree(source,target)
    receipt,path=fresh.relocate_cached_prediction_references(source,target,audit,'pre_fit')
    row=json.loads((target/'prefixes/root/baseline.json').read_text())['evidence']
    assert Path(row['prediction']).resolve().is_relative_to(target.resolve())
    assert row['sha256']==file_sha256(pred)==file_sha256(row['prediction'])
    assert row['values_sha256']==fresh.saved_prediction_values_sha(row['prediction'])==values
    assert receipt['mapping_count']==1 and receipt['no_scientific_value_change'] is True
    assert file_sha256(path)
    assert fresh.raw_manifest(source)==before


def test_formal_verify_rejects_missing_relocation_binding(tmp_path):
    from look.studies import cohort_delivery as delivery
    root=tmp_path/'run';root.mkdir();spec={'x':1};identity=stable_hash(spec)
    (root/'host').mkdir();(root/'host/accepted.json').write_text(json.dumps({'files':{'best.pt':'hostsha'}}))
    profile=root/'profile/fitting';profile.mkdir(parents=True);(profile/'accepted.json').write_text('{}')
    correction=root/'profile/revalidated/residual_rrr/oct_missing';correction.mkdir(parents=True)
    (correction/'selection.json').write_text(json.dumps({'selected_path':[],'final':{'values_sha256':'v'}}))
    (correction/'bank.pt').write_bytes(b'bank')
    audit=correction.parent/'audit'/correction.name;audit.mkdir(parents=True)
    runtime={'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    revalidation=dict(schema='look_fusion_fresh_revalidation_v2',state='accepted',identity=identity,test_access=False,
        source_raw_manifest_unchanged=True,graph_state_exact_before_after_and_fresh=True,
        references_within_revalidated_tree=True,relocation_no_scientific_value_change=True,
        relocation_receipt_sha256='reloc',relocation_mapping_count=1,complete_source_science_exact=True,
        revalidated_science_sha256='science',source_science_sha256='source',runtime=runtime,
        graph_state_sha256='graph',source_raw_manifest_sha256='manifest',source_raw_manifest_final_sha256='manifest')
    (audit/'accepted.json').write_text(json.dumps(revalidation))
    out=root/'residual_rrr';(out/'development').mkdir(parents=True)
    pred=out/'development/p.npz';pred.write_bytes(b'pred')
    row=dict(revalidated_root=str(correction.relative_to(root)),revalidation_receipt=str((audit/'accepted.json').relative_to(root)),
        revalidation_receipt_sha256=file_sha256(audit/'accepted.json'),selection_sha256=file_sha256(correction/'selection.json'),
        bank_sha256=file_sha256(correction/'bank.pt'),selected_path=[],final_values_sha256='v',
        revalidated_science_sha256='science',source_science_sha256='source',runtime_sha256=stable_hash(runtime),
        graph_state_sha256='graph',source_raw_manifest_sha256='manifest',source_raw_manifest_final_sha256='manifest',
        relocation_receipt_sha256='wrong',relocation_mapping_count=1)
    migration=dict(schema='look_fusion_profile_migration_v2',no_refit=True,no_physical_relocation=True,
        source_role='fresh_revalidated_same_run_same_identity',profile_receipt_sha256=file_sha256(profile/'accepted.json'),
        patterns={'oct_missing':row,'cfp_missing':dict(row)})
    receipt=dict(state='accepted',identity=identity,test_access=False,host_best_sha256='hostsha',replay_exact=True,
        records=[dict(path=str(pred),sha256=file_sha256(pred))],profile_migration=migration)
    (out/'accepted.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='revalidation changed'):
        delivery.verify_fusion_formal_arm(spec,root,'residual_rrr')

def test_graph_state_capture_is_pure_and_mode_drift_is_rejected():
    from types import SimpleNamespace
    import torch
    from mhd_framework.models import create_model
    from look.models.native_host import build_native_host
    from look.studies import fusion_cache_revalidation as fresh
    cfg=dict(name='resnet18',spatial_dims=2,in_channels=3,num_classes=2,views=1,granularity='block')
    first=create_model(cfg,weights=None);second=create_model(cfg,weights=None)
    graph=build_native_host(SimpleNamespace(graph=first),SimpleNamespace(graph=second),'deep','cpu')
    graph.eval()
    node=graph.get_node_by_name('oct_input')
    node.feature_message.current_state=torch.ones(1,3,8,8)
    before_training={name:m.training for name,m in graph.named_modules()}
    before_state={k:v.detach().clone() for k,v in graph.state_dict().items()}
    capture=fresh.capture_graph_state(graph)
    assert capture['node_messages']['oct_input']['feature_current_matches_initial'] is False
    assert not torch.equal(node.feature_message.current_state,node.feature_message.initial_state)
    assert {name:m.training for name,m in graph.named_modules()}==before_training
    assert all(torch.equal(before_state[k],v) for k,v in graph.state_dict().items())
    prepared=fresh.prepare_for_evaluation(graph)
    assert prepared['transient_messages_reset'] is True
    assert torch.equal(node.feature_message.current_state,node.feature_message.initial_state)
    assert prepared['after']['node_messages']['oct_input']['feature_current_matches_initial'] is True
    assert prepared['before']['scientific']==prepared['after']['scientific']
    # Fault injection: any child train-mode drift must be visible and rejected.
    child=next(m for name,m in graph.named_modules() if name and hasattr(m,'training'))
    child.train()
    drift=fresh.capture_graph_state(graph)
    assert drift['scientific']['modules'][next(name for name,m in graph.named_modules() if m is child)]['training'] is True
    with pytest.raises(ValueError,match='mode drifted from eval'):
        fresh.prepare_for_evaluation(graph)


def test_capture_graph_state_does_not_silently_eval_graph():
    from types import SimpleNamespace
    from mhd_framework.models import create_model
    from look.models.native_host import build_native_host
    from look.studies import fusion_cache_revalidation as fresh
    cfg=dict(name='resnet18',spatial_dims=2,in_channels=3,num_classes=2,views=1,granularity='block')
    graph=build_native_host(SimpleNamespace(graph=create_model(cfg,weights=None)),
        SimpleNamespace(graph=create_model(cfg,weights=None)),'features','cpu')
    graph.train()
    capture=fresh.capture_graph_state(graph)
    assert capture['scientific']['graph_training'] is True
    assert graph.training is True
    with pytest.raises(ValueError,match='mode drifted from eval'):
        fresh.assert_evaluation_mode(capture)
    assert graph.training is True

def test_two_phase_relocation_normalizes_prefix_and_authority_without_touching_source(tmp_path):
    from look.studies import fusion_cache_revalidation as fresh
    source=tmp_path/'source';target=tmp_path/'target';audit=tmp_path/'audit'
    (source/'predictions').mkdir(parents=True);(source/'prefixes/root').mkdir(parents=True)
    ids=np.array(['a','b']);labels=np.array([0,1]);logits=np.array([[2.,0.],[0.,2.]])
    pred=source/'predictions/p.npz';np.savez(pred,participant_ids=ids,labels=labels,logits=logits)
    evidence=dict(role='development',data_role='development',score=1.0,prediction=str(pred),
        sha256=file_sha256(pred),values_sha256=fresh.saved_prediction_values_sha(pred),metrics={'macro_f1':1.0})
    (source/'prefixes/root/baseline.json').write_text(json.dumps({'identity':'root','evidence':evidence}))
    before=fresh.raw_manifest(source)
    fresh.copy_audit_tree(source,target)
    pre,pre_path=fresh.relocate_cached_prediction_references(source,target,audit,'pre_fit')
    prefix=json.loads((target/'prefixes/root/baseline.json').read_text())
    assert Path(prefix['evidence']['prediction']).is_relative_to(target)
    assert pre['mapping_count']==1 and pre['verified_target_refs']==0
    assert file_sha256(pre_path)
    # Simulate authority files written after fit from the original source evidence.
    selection=dict(decisions=[],selected_path=[],final=dict(evidence),schema='s',contract_sha256='c',
        mode='positive_forward_tree',site_attempts=0,candidate_evaluations=0,prefix_count=0,
        test_access=False,scientific_acceptance=False)
    (target/'selection.json').write_text(json.dumps(selection))
    post,post_path=fresh.relocate_cached_prediction_references(source,target,audit,'post_fit')
    final=json.loads((target/'selection.json').read_text())['final']
    assert Path(final['prediction']).is_relative_to(target)
    assert post['mapping_count']==1
    assert file_sha256(post_path)
    assert fresh.raw_manifest(source)==before


def test_relocation_rejects_unrelated_absolute_prediction_path(tmp_path):
    from look.studies import fusion_cache_revalidation as fresh
    source=tmp_path/'source';target=tmp_path/'target';audit=tmp_path/'audit';other=tmp_path/'other'
    (source/'prefixes/root').mkdir(parents=True);(target/'prefixes/root').mkdir(parents=True);other.mkdir()
    pred=other/'p.npz';np.savez(pred,participant_ids=np.array(['a']),labels=np.array([0]),logits=np.array([[1.,0.]]))
    evidence=dict(role='development',data_role='development',score=1.0,prediction=str(pred),
        sha256=file_sha256(pred),values_sha256=fresh.saved_prediction_values_sha(pred),metrics={'macro_f1':1.0})
    (target/'prefixes/root/baseline.json').write_text(json.dumps({'identity':'root','evidence':evidence}))
    with pytest.raises(ValueError,match='outside both source and target'):
        fresh.relocate_cached_prediction_references(source,target,audit,'pre_fit')
