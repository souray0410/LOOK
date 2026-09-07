"""Scientific boundaries and recovery gates for the final test queue."""
import json
from pathlib import Path
import numpy as np
import pytest
from look_core.dual_queue import record,atomic,digest
from look_core.test_evaluation import dependency_status,freeze,validate_request,sealed_result,bank_record
from look_core.test_runtime import validate_gate,make_test_loader,compare_replay
from look_core.test_report import paired_ci,macro_f1_draws,holm
from look_core.stable_metrics import logit_metrics
from look_core.method_logit import frozen_random, with_logits
from look_core.evaluate import save_prediction_bundle


def request(tmp_path):
    refs={}
    for key in ('parent','methods','self_input','reused'):
        p=tmp_path/(key+'.json');atomic({},p);refs[key]=record(p)
    plan=tmp_path/'plan.json';atomic(dict(jobs=[dict(id='a')]),plan)
    queue=tmp_path/'queue.json';identity=dict(plan='fixed')
    atomic(dict(status='running',identity=identity,test_access=False,completed={},active={'0':{}}),queue)
    return dict(protocol='look_test_follow_on_v1',cohort='test',natural_test='withdrawn',test_participants=290,
        expected_jobs=87,missingness_seed=3407,random_ratios=[.2,.4,.6,.8,1.],allow_refit=False,select_on_test=False,
        bootstrap_iterations=2000,replay_participants=8,validation_plan=record(plan),validation_queue=str(queue),
        validation_identity=identity,**refs)


def test_incomplete_or_changed_dependencies_never_open_test(tmp_path):
    req=request(tmp_path);p=tmp_path/'request.json';atomic(req,p)
    assert dependency_status(req)=='waiting_validation'
    with pytest.raises(ValueError,match='not complete'):freeze(p,tmp_path/'freeze.json','source')
    assert not (tmp_path/'freeze.json').exists()
    q=json.loads(Path(req['validation_queue']).read_text());q['status']='complete';q['active']={}
    atomic(q,req['validation_queue'])
    with pytest.raises(ValueError,match='Premature'):dependency_status(req)
    q['completed']={'a':{}};atomic(q,req['validation_queue']);assert dependency_status(req)=='ready'
    q['identity']={'plan':'new'};atomic(q,req['validation_queue'])
    with pytest.raises(ValueError,match='identity'):dependency_status(req)


def test_no_reselection_or_natural_scope(tmp_path):
    req=request(tmp_path)
    for k,v in [('allow_refit',True),('select_on_test',True),('cohort','natural_test'),('test_participants',10642)]:
        with pytest.raises(ValueError):validate_request(dict(req,**{k:v}))
    for value in [dict(status='running'),dict(status='complete',phase='test'),dict(status='complete',test_access=True)]:
        with pytest.raises(ValueError):sealed_result(value)


def test_test_loader_is_primary_csv_only_full_and_unaugmented(monkeypatch,tmp_path):
    import look_core.test_runtime as m
    label=tmp_path/'labels.csv';label.write_text('test')
    calls=[]
    class Dataset:
        participant_ids=list(map(str,range(290)))
        def __init__(self,**kw):calls.append(kw)
        def __len__(self):return 290
    monkeypatch.setattr(m,'UKBBilateralVisitDataset',Dataset)
    monkeypatch.setattr(m,'make_loader',lambda *a:a)
    job=dict(seed=3407,base=dict(labels=record(label),config=dict(image_root='images',image_size=64,
        preprocess_cache_root='cache',sampling_strategy='shuffle',natural_labels_csv='/must/not/open')))
    make_test_loader(job,'test',2)
    assert calls[0]['split']=='test' and calls[0]['augment'] is False and calls[0]['limit'] is None
    assert calls[0]['labels_csv']==label
    assert '/must/not/open' not in str(calls)


def test_test_needs_every_frozen_validation_replay(tmp_path):
    f=tmp_path/'frozen.json';atomic(dict(jobs=[dict(id='a'),dict(id='b')]),f)
    g=tmp_path/'gate.json';atomic(dict(status='complete',test_access=False,frozen_manifest=record(f),replays={}),g)
    with pytest.raises(ValueError,match='Every'):validate_gate(f,g)
    replay=tmp_path/'replay.json';atomic(dict(status='complete',phase='test',test_access=True),replay)
    atomic(dict(status='complete',test_access=False,frozen_manifest=record(f),replays={'a':record(replay),'b':record(replay)}),g)
    with pytest.raises(ValueError,match='Invalid'):validate_gate(f,g)


def bundle(n=290):
    y=np.arange(n)%2;z=np.column_stack((1-y,y)).astype(float)
    return with_logits(dict(labels=y,logits=z,participant_ids=np.asarray([str(i) for i in range(n)]),patterns=np.full(n,'complete')),z)


def test_frozen_random_no_fit_and_nested_exact_counts():
    b=bundle();pairs={p:b for p in ['complete','oct_missing','cfp_missing']}
    params={p:dict(a=1.,c=0.) for p in ['oct_missing','cfp_missing']}
    last=set();directions={}
    for ratio,count in zip([.2,.4,.6,.8,1.],[58,116,174,232,290]):
        r=frozen_random(pairs,params,ratio,3407)
        selected={pid for pid,p in zip(r['participant_ids'],r['patterns']) if p!='complete'}
        assert len(selected)==count and last<=selected
        for pid,p in zip(r['participant_ids'],r['patterns']):
            if p=='complete':continue
            assert pid not in directions or directions[pid]==p
            directions[pid]=p
        last=selected
        np.testing.assert_array_equal(r['logits'],b['logits'])


def test_replay_compares_ids_labels_logits_and_argmax(tmp_path):
    b=bundle(8);p=tmp_path/'prediction.npz';save_prediction_bundle(b,p)
    assert compare_replay(b,record(p))['argmax_equal']
    wrong=dict(b,logits=b['logits'][:,::-1])
    with pytest.raises(ValueError,match='classifications differ'):compare_replay(wrong,record(p))
    wrong=dict(b,labels=1-b['labels'])
    with pytest.raises(ValueError,match='label'):compare_replay(wrong,record(p))


def test_bootstrap_does_not_treat_seeds_as_independent_participants():
    b=bundle(20);r=paired_ci([(b,b)]*3,100)
    assert r['participant_count']==20 and r['seed_count']==3 and r['ci_low']==r['ci_high']==0
    np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])
    draw=np.arange(20)[None,:]
    assert macro_f1_draws(b['labels'],b['logits'],draw)[0]==b['metrics']['macro_f1']


def test_bank_tampering_blocks_reuse(tmp_path):
    (tmp_path/'selected').mkdir();p=tmp_path/'selected'/'a.pt';p.write_bytes(b'original')
    atomic(dict(artifacts=[dict(path='selected/a.pt',sha256=record(p)['sha256'])]),tmp_path/'selected_manifest.json')
    assert len(bank_record(tmp_path)['artifacts'])==1
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):bank_record(tmp_path)


def test_replay_receipt_cannot_be_borrowed_from_another_case(tmp_path):
    jobs=[dict(id='a'),dict(id='b')]
    f=tmp_path/'frozen.json';atomic(dict(jobs=jobs),f)
    rp=tmp_path/'replay.json'
    atomic(dict(status='complete',phase='validation_replay',test_access=False,kernel_parity=True,
                identity=dict(frozen=record(f),job_sha256=digest(jobs[0]),phase='validation_replay')),rp)
    gate=tmp_path/'gate.json'
    atomic(dict(status='complete',test_access=False,frozen_manifest=record(f),replays={'a':record(rp),'b':record(rp)}),gate)
    with pytest.raises(ValueError,match='Invalid'):validate_gate(f,gate)


def test_report_recomputes_metrics_and_retains_negative_results(tmp_path):
    from look_core.test_report import report
    from look_core.method_logit import with_logits
    jobs=[]
    for seed in (3407,3408,3409):
        jobs.append(dict(id=f'normalized_mean_layer3_{seed}',family='original',fusion='layer3',filling='normalized_mean',seed=seed,
                         validation_result=dict(path=str(tmp_path/'validation.json'))))
    atomic({},tmp_path/'validation.json');atomic(dict(jobs=jobs),tmp_path/'frozen_manifest.json')
    for job in jobs:
        root=tmp_path/'test'/job['id'];root.mkdir(parents=True)
        predictions={};metrics={}
        for scenario in ['complete',*[kind+'_'+s for kind in ['fill','corrected'] for s in ['oct_missing','cfp_missing','random_0.2','random_0.4','random_0.6','random_0.8','random_1.0']]]:
            b=bundle()
            if scenario.startswith('corrected'):b=with_logits(b,b['logits'][:,::-1])
            p=root/(scenario+'.npz');save_prediction_bundle(b,p);predictions[scenario]=record(p);metrics[scenario]=b['metrics']
        atomic(dict(status='complete',phase='test',n=290,metrics=metrics,predictions=predictions),root/'complete.json')
    output=report(tmp_path)
    import csv
    rows=list(csv.DictReader((output/'paired_comparisons.csv').open()))
    assert rows and all(float(r['delta_macro_f1'])==-1 for r in rows)
    assert (output/'test_overview.png').is_file()
    assert all(int(r['seed_count'])==3 and int(r['participant_count'])==290 for r in rows)
