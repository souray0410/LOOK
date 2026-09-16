import json
import pytest
from look.studies.search_protocol import protocol,validate,sites,VERSION
from look.runtime.project_dispatch import work
from look.runtime.state import file_sha256


def spec():
    h=dict(disease='cataract',architecture='resnet50',position='deep',seed=3416)
    ordered=sites(h['architecture'],h['position'])
    return dict(schema=VERSION,protocol=protocol(),test_access=False,host=h,mode='best_forward',
        candidate_sites=ordered,eligible_sites=ordered)


def test_lock_modes_sites_and_test_seal():
    s=spec();validate(s)
    for field,value in [('mode','all_on'),('test_access',True),('eligible_sites',s['eligible_sites'][1:])]:
        with pytest.raises(ValueError):validate(dict(s,**{field:value}))
    s['host']['seed']=3417
    with pytest.raises(ValueError,match='3416'):validate(s)


def test_dispatch_search_identity_and_dedup(tmp_path):
    p=tmp_path/'spec.json';p.write_text(json.dumps(spec()))
    row=dict(id='search',spec=str(p),spec_sha256=file_sha256(p),run_dir=str(tmp_path/'run'),search_mode='best_forward')
    f=tmp_path/'feed.json';f.write_text(json.dumps(dict(schema='look_search_work_feed_v1',test_access=False,tasks=[row])))
    cfg=dict(project_feed=str(tmp_path/'none'),native_feed=str(tmp_path/'none'),search_feeds=[str(f),str(f)])
    assert len(work(cfg))==1 and work(cfg)[0]['execution']=='look_search'
    p.write_text('{}')
    with pytest.raises(ValueError):work(cfg)


def test_only_control_admission_is_capped(tmp_path):
    from look.runtime.project_dispatch import admissible_work
    class Claims:
        def path(self,run):return tmp_path/(str(run).split('/')[-1]+'.claim')
    tasks=[];claims=Claims()
    for i,mode in enumerate(['greedy','greedy','greedy','best_forward']):
        p=tmp_path/f'spec{i}.json';p.write_text(json.dumps(dict(spec(),mode=mode)))
        row=dict(id=str(i),spec=str(p),spec_sha256=file_sha256(p),run_dir=str(tmp_path/f'run{i}'),search_mode=mode)
        if i<2:claims.path(row['run_dir']).write_text(json.dumps(dict(state='running')))
        tasks.append(row)
    feed=tmp_path/'queue.json';feed.write_text(json.dumps(dict(schema='look_search_work_feed_v1',test_access=False,tasks=tasks)))
    cfg=dict(project_feed=str(tmp_path/'none'),native_feed=str(tmp_path/'none'),search_feeds=[str(feed)],search_control_maximum=2)
    assert [t['id'] for t in admissible_work(cfg,claims)]==['3']
