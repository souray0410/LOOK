import json
import subprocess
import pytest
from look.runtime.state import file_sha256
from look.studies import cohort_sequence as m


def plan(tmp_path):
    rows=[]
    for i in range(2):
        root=tmp_path/f'r{i}';root.mkdir()
        spec=dict(run_id=f'r{i}',output=str(root),test_access=False,seed=3416,
            data_audit_sha256='x',data_root='same',training={},factor=16,rank=32,position='deep',
            arms=['residual_rrr','pca_free_mean'],search='positive_forward_tree',framework_commit='fixed')
        path=root/'spec.json';path.write_text(json.dumps(spec))
        rows.append(dict(role='execute',spec=str(path),spec_sha256=file_sha256(path)))
    return dict(schema='look_cohort_sequence_v1',test_access=False,sequence_id='test',
        tasks=rows,root=str(tmp_path/'queue'),publication=str(tmp_path/'publication'))


def test_reject_duplicate_and_changed_fixed_factors(tmp_path):
    p=plan(tmp_path);m.validate_sequence(p)
    p['tasks'][1]=p['tasks'][0]
    with pytest.raises(ValueError,match='Duplicate'):m.validate_sequence(p)


def test_changed_spec_rejected(tmp_path):
    p=plan(tmp_path);path=m.Path(p['tasks'][1]['spec']);s=m.read(path);s['seed']=3417;path.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='Spec changed'):m.validate_sequence(p)
    p['tasks'][1]['spec_sha256']=file_sha256(path)
    with pytest.raises(ValueError,match='repeat seed'):m.validate_sequence(p)


def test_failure_isolated_and_success_not_rerun(tmp_path,monkeypatch):
    p=plan(tmp_path);events=[]
    monkeypatch.setattr(m,'refresh',lambda *args:None)
    monkeypatch.setattr(m,'publish',lambda *args:dict(matched_package='accepted'))
    class Child:
        def __init__(self,code):self.returncode=code
        def poll(self):return self.returncode
    def launch(s,row):
        events.append(s['run_id'])
        if s['run_id']=='r1':
            d=m.Path(s['output'])/'delivery';d.mkdir();(d/'accepted.json').write_text('{}')
        return Child(1 if s['run_id']=='r0' else 0)
    m.run(p,launch=launch,interval=.001)
    assert events==['r0','r1']
    assert m.read(m.Path(p['root'])/'completion.json')['state']=='needs_review'
    m.run(p,launch=launch,interval=.001)
    assert events==['r0','r1']
    assert m.read(m.Path(p['root'])/'status.json')['tasks']['r0']['state']=='needs_review'


def test_exclusive_sequence_lock(tmp_path,monkeypatch):
    p=plan(tmp_path);root=m.Path(p['root']);root.mkdir()
    with (root/'sequence.lock').open('a') as lock:
        m.fcntl.flock(lock,m.fcntl.LOCK_EX|m.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):m.run(p)


def _public_config(run_id,architecture='resnet18',position='deep',mmtm=None):
    rows=[]
    for pattern in ('oct_missing','cfp_missing'):
        for method,value in [('host',.5),('pca_free_mean',.6),('residual_rrr',.61)]:
            rows.append(dict(method=method,scenario=pattern,metrics={'macro_f1':value}))
    out=dict(run_id=run_id,architecture=architecture,position=position,matched_package='accepted',results=rows)
    if mmtm is not None:out['mmtm']=mmtm
    return out


def test_register_fusion_extension_preserves_old_three_and_adds_middle_features_once():
    old=[
        _public_config('2026_09_18_09_35_40','resnet50','deep'),
        _public_config('2026_09_18_11_18_28_650020','resnet18','deep'),
        _public_config('2026_09_18_15_29_03_789799','resnet18','deep',{'stage':'stage3'})
    ]
    current=dict(schema='look_cohort_sequence_publication_v1',sequence_id='old',test_used=False,
        order=[x['run_id'] for x in old],configurations=old,
        execution={x['run_id']:{'state':'accepted'} for x in old})
    fusion=dict(schema='look_fusion_stage_publication_v1',task_id='fusion',sequence_id='fusion-seq',
        complete=True,test_used=False,order=['deep','middle','features'],
        configurations=[old[1],_public_config('2026_09_19_02_40_42_281912_middle','resnet18','middle'),
                        _public_config('2026_09_19_02_40_42_281912_features','resnet18','features')])
    ext=m.register_fusion_extension(current,fusion,'left-audit')
    current['fusion_stage_extension']=ext
    displayed=m._display_publications(current)
    assert [x['run_id'] for x in current['configurations']]==[x['run_id'] for x in old]
    assert [x['run_id'] for x in displayed].count('2026_09_18_11_18_28_650020')==1
    assert [x['position'] for x in displayed if x['architecture']=='resnet18' and 'mmtm' not in x]==['deep','middle','features']
    assert [x['run_id'] for x in ext['configurations']]==[
        '2026_09_19_02_40_42_281912_middle','2026_09_19_02_40_42_281912_features']
    carried=m._carry_fusion_extension(current,dict(current, fusion_stage_extension=None))
    assert carried['fusion_stage_extension']==ext


def test_fusion_extension_rejects_duplicate_deep_registration():
    deep=_public_config('2026_09_18_11_18_28_650020')
    current=dict(configurations=[deep])
    ext=dict(schema=m.FUSION_EXTENSION_SCHEMA,state='scientifically_accepted',test_used=False,
        deep_reference={'run_id':deep['run_id'],'role':'strict_reuse_no_retraining'},
        configurations=[deep,_public_config('2026_09_19_02_40_42_281912_features',position='features')])
    with pytest.raises(ValueError,match='exactly middle/features'):
        m.validate_fusion_extension(ext,current)
