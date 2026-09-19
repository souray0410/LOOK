import importlib.util
import json
import os
from pathlib import Path
import pytest
from look.runtime.state import file_sha256, stable_hash

candidate=os.environ.get('LOOK_PUBLICATION_CANDIDATE')
if candidate:
    spec=importlib.util.spec_from_file_location('publication_candidate',candidate)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    publish=module.publish
else:
    from look.analysis.cohort_publication import publish


def fixture(root):
    s=dict(run_id='synthetic',source_commit='source',framework_commit='framework',architecture='resnet50',
        position='deep',seed=3416,factor=16,rank=32,search='positive_forward_tree',
        arms=['residual_rrr','pca_free_mean'],training={},initialization={'kind':'public','path':'SECRET_PATH','sha256':'init'})
    (root/'spec.json').write_text(json.dumps(s))
    for arm in s['arms']:
        p=root/arm;p.mkdir();rows=[]
        for pattern in ('oct_missing','cfp_missing'):
            q=p/'corrections'/pattern;q.mkdir(parents=True);(q/'bank.pt').write_bytes(b'bank')
            for name in (arm,'host'):
                f=p/f'{name}_{pattern}.npz';f.write_bytes(b'private')
                rows.append(dict(method=name,scenario=pattern,path=str(f),sha256=file_sha256(f),
                    metrics={'macro_f1':.6,'macro_auroc_ovr':.7}))
        (p/'accepted.json').write_text(json.dumps(dict(state='accepted',identity=stable_hash(s),records=rows)))
        (root/(arm+'_status.json')).write_text(json.dumps(dict(state='completed',time=1000)))
    return s


def test_idempotence_and_unique_results(tmp_path):
    fixture(tmp_path);out=tmp_path/'public';v=publish(tmp_path,out)
    assert len(v['results'])==6
    assert 'SECRET_PATH' not in (out/'current.json').read_text()
    original={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in out.iterdir()}
    assert publish(tmp_path,out)==v
    assert original=={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in out.iterdir()}
    assert v['evidence_cutoff_utc'].startswith('1970-01-01T00:16:40')


@pytest.mark.parametrize('fault',['identity','prediction','duplicate'])
def test_corruption_preserves_last_good_snapshot(tmp_path,fault):
    fixture(tmp_path);out=tmp_path/'public';publish(tmp_path,out)
    original={p.name:p.read_bytes() for p in out.iterdir()}
    p=tmp_path/'pca_free_mean/accepted.json';v=json.loads(p.read_text())
    if fault=='identity':v['identity']='bad'
    elif fault=='prediction':Path(v['records'][0]['path']).write_bytes(b'changed')
    else:v['records'][1]['metrics']['macro_f1']=.2
    p.write_text(json.dumps(v))
    with pytest.raises(ValueError):publish(tmp_path,out)
    assert original=={p.name:p.read_bytes() for p in out.iterdir()}


def test_reject_incomplete_delivery_and_then_publish_increment(tmp_path):
    s=fixture(tmp_path);out=tmp_path/'public';v=publish(tmp_path,out)
    delivery=tmp_path/'delivery';delivery.mkdir()
    def receipt(rows):
        (delivery/'results.json').write_text(json.dumps(dict(results=rows,comparisons=[],statistics={'contrasts':[]})))
        (delivery/'accepted.json').write_text(json.dumps(dict(state='accepted',identity=stable_hash(s),files={'results.json':file_sha256(delivery/'results.json')})))
    receipt([])
    with pytest.raises(ValueError,match='disagree'):publish(tmp_path,out)
    assert json.loads((out/'current.json').read_text())==v
    receipt(v['results']);new=publish(tmp_path,out)
    assert new['matched_package']=='accepted' and len(new['results'])==6
    assert new['source_evidence_sha256']!=v['source_evidence_sha256']


def test_tree_detail_provenance_and_readable_matching_table(tmp_path):
    s=fixture(tmp_path);folder=tmp_path/'residual_rrr/corrections/oct_missing'
    contract=dict(identity=dict(identity=stable_hash(s),candidates=[dict(rank=32,ridge_lambda=None)],penalty_policy='prefix_train_pca_gcv'),sites=['fusion_features'])
    (folder/'contract.json').write_text(json.dumps(contract));artifact=folder/'selected.pt';artifact.write_bytes(b'map')
    node=dict(node='fusion_features',index=7,artifact='selected.pt',sha256=file_sha256(artifact),score=.6)
    tree=dict(contract_sha256=file_sha256(folder/'contract.json'),mode='positive_forward_tree',
        decisions=[dict(path=[],baseline={'score':.5},candidates=[node])],selected_path=[node],
        final={'metrics':{'macro_f1':.6,'macro_auroc_ovr':.7}},candidate_evaluations=1,prefix_count=2,site_attempts=1)
    (folder/'selection.json').write_text(json.dumps(tree));(folder/'feature_costs.json').write_text(json.dumps(dict(full_forwards=1,missing_forwards=1,completed_hits=0,skipped_batches=0)))
    out=tmp_path/'public';v=publish(tmp_path,out)
    assert v['search_details'][0]['selected_path'][0]['node']=='fusion_features'
    content=(out/'README.md').read_text()
    assert '|输入情况|不修正|PCA方向约束＋树|自由低秩残差＋树|' in content
    assert '⑧每眼特征向量' in content and '正收益树' in content
    before=(out/'current.json').read_bytes();artifact.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Selected search artifact changed'):publish(tmp_path,out)
    assert (out/'current.json').read_bytes()==before


def test_mmtm_identity_is_visible_without_changing_plain_host(tmp_path):
    s=fixture(tmp_path);s['mmtm']=dict(stage='stage3',ratio=4,gate_scale=1.0)
    (tmp_path/'spec.json').write_text(json.dumps(s))
    for arm in s['arms']:
        path=tmp_path/arm/'accepted.json';v=json.loads(path.read_text());v['identity']=stable_hash(s);path.write_text(json.dumps(v))
    out=tmp_path/'public';v=publish(tmp_path,out)
    assert v['mmtm']==s['mmtm']
    assert 'MMTM适配宿主' in (out/'README.md').read_text()

def test_fusion_logical_migration_resolves_revalidated_correction(tmp_path):
    from look.analysis.cohort_publication import correction_folder
    root=tmp_path/'run';root.mkdir()
    folder=root/'profile/revalidated/residual_rrr/oct_missing';folder.mkdir(parents=True)
    (folder/'selection.json').write_text('{}');(folder/'bank.pt').write_bytes(b'bank')
    audit=root/'profile/revalidated/residual_rrr/audit/oct_missing';audit.mkdir(parents=True)
    (audit/'accepted.json').write_text(json.dumps({'state':'accepted'}))
    row=dict(revalidated_root=str(folder.relative_to(root)),
        revalidation_receipt=str((audit/'accepted.json').relative_to(root)),
        revalidation_receipt_sha256=file_sha256(audit/'accepted.json'),
        selection_sha256=file_sha256(folder/'selection.json'),
        bank_sha256=file_sha256(folder/'bank.pt'))
    receipt=dict(profile_migration=dict(schema='look_fusion_profile_migration_v2',no_refit=True,
        no_physical_relocation=True,source_role='fresh_revalidated_same_run_same_identity',
        patterns={'oct_missing':row}))
    assert correction_folder(root,'residual_rrr','oct_missing',receipt)==folder.resolve()
    (folder/'bank.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='logical migration evidence changed'):
        correction_folder(root,'residual_rrr','oct_missing',receipt)


def test_plain_publication_correction_path_is_unchanged(tmp_path):
    from look.analysis.cohort_publication import correction_folder
    root=tmp_path/'run';root.mkdir()
    expected=(root/'residual_rrr/corrections/oct_missing').resolve()
    assert correction_folder(root,'residual_rrr','oct_missing',{})==expected
