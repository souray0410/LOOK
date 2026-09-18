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
