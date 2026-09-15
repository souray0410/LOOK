import json
import time
import pytest
from look.analysis.terminal_rollup import ARMS, collect, summarize, tick
from look.runtime.state import file_sha256, stable_hash


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True);path.write_text(json.dumps(obj));return path


def fixture(tmp_path):
    h=dict(disease='cataract',architecture='resnet50',position='middle',seed=3416)
    s=dict(host=h,test_access=False);root=tmp_path/'run';sp=put(root/'spec.json',s)
    records=[dict(method=m,scenario=p,metrics=dict(macro_f1=.6 if m=='host' else .59))
             for m in (*ARMS,'host') for p in ('oct_missing','cfp_missing')]
    files={'spec.json':file_sha256(sp)}
    for name,obj in {'development/suite.json':dict(records=records,test_access=False),
                     'report/paired_statistics.json':dict(iterations=10000,family='21_terminal_stage_contrasts_not_global_LOOK',test_access=False),
                     **{'selection/'+p+'.json':dict(rank=8,lambda_=1.) for p in ('oct_missing','cfp_missing')}}.items():
        files[name]=file_sha256(put(root/name,obj))
    put(root/'accepted.json',dict(files=files,state='accepted',schema='look_terminal_stage_v1',identity=stable_hash(s),
        profile=False,test_access=False,full_development_mhd_replay=True,host_frozen=True,reload_exact=True))
    task=dict(id='terminal/fixture',run_dir=str(root),spec=str(sp),spec_sha256=file_sha256(sp))
    feed=put(tmp_path/'queue.json',dict(schema='look_terminal_work_feed_v1',test_access=False,expected_hosts=81,
        tasks=[task],updated_at=time.time(),waiting={'remaining':'waiting_host_acceptance'},errors=[]))
    return task,feed


def test_negative_results_not_filtered_or_selected(tmp_path):
    task,feed=fixture(tmp_path);r=tick(feed,tmp_path/'report')
    assert r['accepted_cases']==1 and not r['complete'] and r['selected_method'] is None
    assert r['complete_three_seed_rows']==[]
    assert all(x['delta_host_pp']<0 for x in collect(task)[0] if x['method'] in ARMS)


def test_tampered_report_excluded_and_error_visible(tmp_path):
    task,feed=fixture(tmp_path);(tmp_path/'run/development/suite.json').write_text('{}')
    r=tick(feed,tmp_path/'report');assert r['accepted_cases']==0 and len(r['errors'])==1


def test_duplicate_and_test_feed_refused(tmp_path):
    task,feed=fixture(tmp_path);q=json.loads(feed.read_text());q['tasks']*=2;put(feed,q)
    with pytest.raises(ValueError,match='Duplicate'):tick(feed,tmp_path/'report')
    q['test_access']=True;put(feed,q)
    with pytest.raises(ValueError,match='Unsealed'):tick(feed,tmp_path/'report')


def test_only_exact_three_seeds_aggregate():
    rows=[dict(disease='d',architecture='a',position='p',seed=s,scenario='oct_missing',method=m,macro_f1=.5+s/100000)
          for s in (3416,3417,3418) for m in ARMS]
    r=summarize(rows);assert len(r['complete_three_seed_rows'])==3
    assert all(x['sample_sd']>0 for x in r['complete_three_seed_rows'])
    assert r['descriptive_ranks']['pure']['residual_rrr']['1']==3
