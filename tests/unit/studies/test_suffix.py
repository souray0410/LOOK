import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from look.studies.suffix_protocol import protocol,sites,validate,VERSION
from look.runtime.state import atomic_write_json,file_sha256,stable_hash


def specification():
    h=dict(disease='cataract',architecture='resnet50',position='deep',seed=3416);nodes=sites('resnet50','deep')
    return dict(schema=VERSION,protocol=protocol(),host=h,start_ordinal=3,candidate_sites=nodes,eligible_sites=nodes[2:],test_access=False)


def test_all_starts_count_and_prefix_gate():
    for a in ('resnet50','densenet121','swin_b'):
        for p in ('middle','deep','features'):
            n=sites(a,p);assert len(n)==9 and len(set(n))==9 and n[0]=='joint_input' and n[-1]=='fusion_participant_feature'
    s=specification();validate(s)
    for change in ({'test_access':True},{'start_ordinal':1},{'eligible_sites':s['candidate_sites']}):
        with pytest.raises(ValueError):validate(dict(s,**change))
    s['host']['seed']=3417
    with pytest.raises(ValueError,match='technical'):validate(s)
    assert 81*(9-2)==567


def test_no_partial_winner_and_ties():
    from look.analysis.suffix_report import choose
    with pytest.raises(ValueError):choose([dict(start_ordinal=1,macro_f1=.8)],9)
    rows=[dict(start_ordinal=i,macro_f1=.5) for i in range(1,10)]
    assert choose(rows,9)['start_ordinal']==1
    rows[4]['macro_f1']=.7;assert choose(rows,9)['start_ordinal']==5
    rows[1]['start_ordinal']=1
    with pytest.raises(ValueError):choose(rows,9)


def test_forged_acceptance_and_cross_run_files(tmp_path):
    from look.studies.suffix_case import verify_case,check_files
    s=specification();atomic_write_json(dict(schema=VERSION,identity=stable_hash(s),state='accepted',test_access=False,profile=False,files={}),tmp_path/'accepted.json')
    with pytest.raises(ValueError):verify_case(tmp_path,s)
    other=tmp_path.parent/'foreign';other.write_text('x')
    with pytest.raises(ValueError):check_files(tmp_path,{'../foreign':file_sha256(other)})


def test_dispatch_suffix_stays_in_existing_claim_system(tmp_path):
    from look.runtime.project_dispatch import work,eligible
    p=tmp_path/'spec.json';atomic_write_json(specification(),p)
    t=dict(id='suffix',spec=str(p),spec_sha256=file_sha256(p),run_dir=str(tmp_path/'run'))
    q=tmp_path/'queue.json';atomic_write_json(dict(schema='look_suffix_work_feed_v1',test_access=False,tasks=[t]),q)
    config=dict(project_feed=str(tmp_path/'none'),native_feed=str(tmp_path/'none'),suffix_feeds=[str(q),str(q)])
    rows=work(config);assert len(rows)==1 and rows[0]['execution']=='look_suffix'
    claim=tmp_path/'claim';atomic_write_json({'state':'running'},claim)
    assert not eligible(rows[0],SimpleNamespace(path=lambda r:claim))
    atomic_write_json(dict(schema='look_suffix_work_feed_v1',test_access=True,tasks=[t]),q)
    with pytest.raises(ValueError):work(config)


def test_report_reuses_anchors_preserves_negative_and_shared_participants(tmp_path,monkeypatch):
    import look.analysis.suffix_report as m
    import look.studies.suffix_case as case
    from look.evaluation.stability import logit_metrics
    h=specification()['host'];base=dict(disease=h['disease'],model={'name':h['architecture']},position=h['position'],seed=h['seed'])
    root=tmp_path/'original';root.mkdir();atomic_write_json(base,root/'spec.json');atomic_write_json({'test_access':False},root/'accepted.json')
    source=dict(run_dir=str(root),spec_path=str(root/'spec.json'))
    y=np.array([0,1,0,1]);ids=np.array(['a','b','c','d']);good=np.array([[2.,0],[0,2],[2,0],[0,2]])
    def record(folder,method,pattern,logits):
        folder.mkdir(parents=True,exist_ok=True);p=folder/(method+'_'+pattern+'.npz');np.savez(p,labels=y,participant_ids=ids,logits=logits)
        return dict(method=method,scenario=pattern,path=str(p),sha256=file_sha256(p),metrics=logit_metrics(y,logits))
    rows=[record(root/'development',a,p,good) for a in ('look','single_final','host') for p in ('oct_missing','cfp_missing')]
    atomic_write_json({'records':rows},root/'development/suite.json')
    runs=[]
    for ordinal in range(2,9):
        r=tmp_path/f'start{ordinal}';r.mkdir();s=dict(specification(),start_ordinal=ordinal,eligible_sites=sites('resnet50','deep')[ordinal-1:],source=source)
        atomic_write_json(s,r/'spec.json');atomic_write_json({'accepted':True},r/'accepted.json')
        atomic_write_json({'records':[record(r/'development','suffix',p,good[:,::-1] if ordinal==2 else good) for p in ('oct_missing','cfp_missing')]},r/'development/suite.json');runs.append(str(r))
    monkeypatch.setattr(m,'verify_original',lambda *a:None);monkeypatch.setattr(case,'verify_case',lambda *a:None)
    manifest=dict(host=h,source=source,runs=runs,output=str(tmp_path/'report'),test_access=False)
    receipt=m.report(manifest);assert receipt['all_starts_complete']
    stats=json.loads((tmp_path/'report/paired_statistics.json').read_text());assert len(stats['definitions'])==16
    assert stats['contrasts'][0]['difference']==-1.
    selected=json.loads((tmp_path/'report/selected_starts.json').read_text());assert selected['oct_missing']['start_ordinal']==1
    rows=json.loads((tmp_path/'report/results.json').read_text());assert len(rows)==20
    m.verify_group(tmp_path/'report')
    Path(runs[0]+'/accepted.json').write_text('changed')
    with pytest.raises(ValueError,match='input'):m.verify_group(tmp_path/'report')


def test_registry_first_seed_gate_and_idempotence(tmp_path, monkeypatch):
    import sys
    import types
    import look.studies.suffix_registry as m
    def reserve(out, protocol, identity, spec, **kwargs):
        run=Path(out)/'runs'/identity;run.mkdir(parents=True,exist_ok=True);return str(run)
    monkeypatch.setitem(sys.modules,'mhd_models.runtime.run_registry',types.SimpleNamespace(reserve=reserve))
    rows=[]
    for seed in (3416,3417):
        root=tmp_path/str(seed);(root/'host').mkdir(parents=True)
        base=dict(disease='cataract',model={'name':'resnet50'},position='deep',seed=seed)
        atomic_write_json(base,root/'spec.json')
        atomic_write_json(dict(state='accepted',identity=stable_hash(base),files={'best.pt':'locked'}),root/'host/accepted.json')
        atomic_write_json(dict(identity={'case':stable_hash(base)},entries=[]),root/'pca/bank/bank_manifest.json')
        rows.append(dict(id=str(seed),spec=str(root/'spec.json'),spec_sha256=file_sha256(root/'spec.json'),run_dir=str(root)))
    feed=tmp_path/'base.json';atomic_write_json(dict(schema='look_project_work_feed_v1',test_access=False,tasks=rows),feed)
    config=dict(output=str(tmp_path/'registry'),base_feed=str(feed),source_pins=[{'path':'locked','sha256':'locked'}])
    first=m.tick(config);second=m.tick(config)
    assert len(first['tasks'])==7 and first['tasks']==second['tasks']
    assert all('/3416/' in r['id'] for r in first['tasks'])
    assert 'waiting_complete_3416_start_comparison' in first['waiting'].values()
    assert [json.loads(Path(r['spec']).read_text())['start_ordinal'] for r in first['tasks']]==list(range(8,1,-1))
    pilot=Path(config['output'])/'reports'/stable_hash(dict(disease='cataract',architecture='resnet50',position='deep',seed=3416))
    atomic_write_json({'state':'accepted'},pilot/'accepted.json')
    monkeypatch.setattr(m,'verify_group',lambda r:dict(host=dict(disease='cataract',architecture='resnet50',position='deep',seed=3416)))
    third=m.tick(config);assert len(third['tasks'])==14
    assert all(json.loads(Path(r['spec']).read_text()).get('pilot') for r in third['tasks'] if '/3417/' in r['id'])


def test_suffix_refits_without_original_prefix_and_resumes(tmp_path,monkeypatch):
    import torch
    import look.methods.operator as op
    import look.evaluation.evaluator as ev
    import look.analysis.matrices as ma
    from tests.unit.methods.test_joint import artifact
    calls=[]
    def evaluate(*args,artifact_banks,**kwargs):
        bank=artifact_banks['oct_missing'];v=.5+.1*len(bank)
        return dict(metrics={'macro_f1':v},labels=np.array([0,1]),probabilities=np.eye(2),logits=np.eye(2),
                    scores=np.array([-1,1]),participant_ids=np.array(['a','b']),patterns=np.array(['x','x']))
    def fit(**kw):
        calls.append((kw['node_name'],tuple(a.node_name for a in kw['upstream_artifacts'])))
        return {1:artifact(kw['node_name'])}
    monkeypatch.setattr(ev,'evaluate_missing',evaluate);monkeypatch.setattr(op,'fit_look_node',fit)
    monkeypatch.setattr(ma,'analyze_look_bank',lambda *a:[])
    bank={(n,1):SimpleNamespace(feature_shape=(6,1,1),source_id=n) for n in ('a','b')}
    def run(nodes,root):
        return op._fit_factor_bank(None,None,None,'oct_missing',nodes,1,[1],1,torch.device('cpu'),root,pca_bank=bank,primary_metric='macro_f1')
    run(['a','b'],tmp_path/'original');run(['b'],tmp_path/'suffix')
    assert calls==[('a',()),('b',('a',)),('b',())]
    count=len(calls);run(['b'],tmp_path/'suffix');assert len(calls)==count
