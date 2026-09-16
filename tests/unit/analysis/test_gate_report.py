import json
from pathlib import Path
import numpy as np
import pytest
from look.analysis.gate_report import compute, report
from look.studies.gate_manifest import import_run, PROTOCOL
from look.runtime.state import file_sha256, atomic_write_json, stable_hash
from look.evaluation.correction_gate import macro_f1


def fixture(root):
    root.mkdir(); records=[];y=np.array([0,1]*12);methods=['shared_pca_ridge','rrr_shared_intercept','residual_rrr']
    for m in ['host',*methods]:
        for p in ['complete','oct_missing','cfp_missing']:
            # PCA rejected, both RRR accepted for OCT; CFP rejects all.
            pred=y if p=='complete' or (p=='cfp_missing' and m=='host') or (p=='oct_missing' and m in methods[1:]) else 1-y
            logits=np.eye(2)[pred];path=root/f'{m}_{p}.npz'
            np.savez(path,logits=logits,labels=y,participant_ids=np.arange(len(y)),patterns=np.full(len(y),p))
            records.append(dict(method=m,scenario=p,path=str(path),sha256=file_sha256(path),metrics={'macro_f1':macro_f1(y,logits)}))
    spec=dict(schema='look_terminal_stage_v1',test_access=False,host=dict(disease='cataract',architecture='resnet50',position='features',seed=3416))
    atomic_write_json(spec,root/'spec.json');atomic_write_json(dict(records=records,test_access=False),root/'development/suite.json')
    receipt=dict(schema=spec['schema'],test_access=False,state='accepted',profile=False,identity=stable_hash(spec),host_frozen=True,reload_exact=True,rng_restored=True,full_development_mhd_replay=True,
        files={n:file_sha256(root/n) for n in ['spec.json','development/suite.json']})
    atomic_write_json(receipt,root/'accepted.json');return root


def test_each_method_independent_same_rule_identity_no_mutation(tmp_path):
    root=fixture(tmp_path/'source');manifest=import_run(root)
    before={str(p):file_sha256(p) for p in root.rglob('*') if p.is_file()}
    assert ['residual_rrr','shared_pca_ridge'] in manifest['comparisons']
    rows,gates,chosen,stats=compute(manifest,iterations=50)
    assert not gates['shared_pca_ridge','oct_missing']['enabled']
    assert gates['residual_rrr','oct_missing']['enabled']
    assert not gates['residual_rrr','cfp_missing']['enabled']
    assert any(r['macro_f1']==0 for r in rows if r['view']=='forced_on_mechanism')
    assert all(r['macro_f1']==1 for r in rows if r['view']=='dev_selected_strategy' and r['scenario']=='cfp_missing')
    report(manifest,tmp_path/'report',iterations=50)
    # Completed review can be reused, original source remains byte identical.
    assert report(manifest,tmp_path/'report',iterations=50)['state']=='accepted'
    assert before=={str(p):file_sha256(p) for p in root.rglob('*') if p.is_file()}
    assert stats['selection_bias_removed'] is False
    a=next(r for r in manifest['records'] if r['method']=='host');a['metrics']['macro_f1']=.1
    with pytest.raises(ValueError,match='metric'):compute(manifest,iterations=10)


def test_import_rejects_unaccepted_or_test_evidence(tmp_path):
    root=fixture(tmp_path/'source');r=json.loads((root/'accepted.json').read_text());r['profile']=True
    atomic_write_json(r,root/'accepted.json')
    with pytest.raises(ValueError,match='accepted'):import_run(root)
