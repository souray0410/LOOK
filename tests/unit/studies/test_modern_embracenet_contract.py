import copy,json
from pathlib import Path
import pytest,torch
from look.studies import modern_embracenet as modern
from look.studies import embracenet_delivery as delivery


def minimal():
    return dict(schema=modern.SCHEMA,test_access=False,architecture='convnext_base',seed=3416,embracement_size=256,
                training=copy.deepcopy(modern.TRAINING),look=copy.deepcopy(modern.LOOK),recipe=copy.deepcopy(modern.RECIPE),cohort={'train':58403,'development':12510},
                lease_safety_seconds=900,gpu_reserve_bytes=0,gpu_budget_bytes=1024)


@pytest.mark.parametrize('field,value',[('test_access',True),('architecture','resnet18'),('seed',3417),('cohort',{'train':1264,'development':296}),('lease_safety_seconds',0)])
def test_modern_contract_never_relabels_legacy_small_study(field,value):
    s=minimal();s[field]=value
    with pytest.raises(ValueError):delivery.validate(s)


def test_modern_adapter_preserves_observed_one_and_two_eyes(tmp_path,monkeypatch):
    import mhd_models.workflows.native as native
    parents=[]
    for track in ('cfp_2d','oct_bscan_2d'):
        path=tmp_path/track;path.mkdir();(path/'spec.json').write_text(json.dumps({'track':track,'test_used':False,'train_manifest':'manifest','train_manifest_sha256':'same','training':{}}));parents.append({'path':str(path)})
    class Inputs:
        def __init__(self,path,sha,track,**kw):
            self.rows=[{'id':'a','eyes':['left'],'label':0},{'id':'b','eyes':['left','right'],'label':1}]
            self.augment=kw['augment'];self.epoch=0;self.verified=set()
        def __len__(self):return 2
        def __getitem__(self,i):self.verified.add(i);return torch.ones(len(self.rows[i]['eyes']),3,2,2),i,i
    monkeypatch.setattr(native,'Inputs',Inputs)
    ds=modern.dataset({'parents':parents,'seed':3416},'train',True)
    assert ds.counts==[1,2] and ds.participant_ids==['a','b']
    assert len(ds[0]['oct'])==1 and len(ds[1]['cfp'])==2 and len(ds.verified)==4
    ds.set_epoch(3);assert ds.first.epoch==ds.second.epoch==3
    with pytest.raises(ValueError,match='Sealed'):modern.dataset({'parents':parents,'seed':3416},'test')


def test_modern_stage_lease_requires_verified_end_and_pauses(tmp_path,monkeypatch):
    s={'host_free_fraction_min':0,'disk_reserve_bytes':0,'lease_safety_seconds':900}
    monkeypatch.delenv('MHD_EXECUTION_LEASE_END',raising=False)
    with pytest.raises(KeyError):delivery._resource_guard(s,tmp_path)
    monkeypatch.setenv('MHD_EXECUTION_LEASE_END',str(delivery.time.time()+600))
    assert delivery._resource_guard(s,tmp_path)
    monkeypatch.setenv('MHD_EXECUTION_LEASE_END',str(delivery.time.time()+3600))
    assert not delivery._resource_guard(s,tmp_path)


def test_modern_host_requires_separate_recovery_before_training(tmp_path):
    s=minimal()
    with pytest.raises(FileNotFoundError):delivery.stage_host(s,tmp_path,lambda:False)
    (tmp_path/'modern_resume').mkdir()
    (tmp_path/'modern_resume/accepted.json').write_text(json.dumps({'state':'accepted','identity':'another execution'}))
    with pytest.raises(ValueError,match='fresh-process'):
        delivery.stage_host(s,tmp_path,lambda:False)
