import importlib.util,json
from pathlib import Path
import pytest,torch
from torch import nn
from look.models.native_materialization import verify_selected
P=Path(__file__).resolve().parents[3]/'scripts/migrate_modern_selected_parent.py'
spec=importlib.util.spec_from_file_location('modern_publication',P);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

@pytest.fixture
def packet(tmp_path,monkeypatch):
    source=tmp_path/'source';conversion=tmp_path/'conversion';replay=tmp_path/'replay'
    for p in (source,conversion/'v5_selected',replay/'V4',replay/'V5'):p.mkdir(parents=True)
    write=module.write;sha=module.sha
    manifest=tmp_path/'development.json';write(manifest,{'samples':[{},{}]})
    spec={'test_used':False,'framework':{'api':'V4'},'model':{'name':'convnext_base'},'development_manifest':str(manifest)}
    write(source/'spec.json',spec);torch.save({'model':{'old':torch.zeros(1)}},source/'best.pt')
    write(source/'history.json',[{'epoch':1}]);(source/'development_predictions.npz').write_bytes(b'historical-array')
    accepted={'status':'accepted','test_used':False,'identity':'historical','files':{p.name:sha(p) for p in source.iterdir()}}
    write(source/'accepted.json',accepted)
    model=nn.Linear(2,2);state={'graph.'+k:v for k,v in model.state_dict().items()}
    torch.save({'framework_api':'V5','identity':'converted','model':state},conversion/'v5_selected/checkpoint.pt')
    cpu={'source_best_sha256':sha(source/'best.pt'),'source_spec_sha256':sha(source/'spec.json'),'target_checkpoint_sha256':sha(conversion/'v5_selected/checkpoint.pt'),'framework':{'api':'V5'}}
    write(conversion/'cpu_accepted.json',cpu)
    for api,status in [('V4','reference_recorded'),('V5','full_dev_and_three_updates_accepted')]:
        torch.save({'logits':torch.zeros(2,2)},replay/api/'development.pt')
        write(replay/api/'accepted.json',dict(api=api,state=status,test_access=False,scientific_training_updates=0,qualification_updates=3,tool_sha256='tool',source_best_sha256=cpu['source_best_sha256'],target_checkpoint_sha256=cpu['target_checkpoint_sha256'],hardware='NVIDIA A100',development_participants=2,development_sha256=sha(replay/api/'development.pt')))
    import mhd_framework.models,mhd_framework.models.artifacts
    monkeypatch.setattr(mhd_framework.models,'create_model',lambda _:nn.Linear(2,2))
    monkeypatch.setattr(mhd_framework.models.artifacts,'verify_runtime',lambda _:None)
    return source,conversion,replay,tmp_path/'published'


def test_publication_current_format_original_history_and_idempotence(packet):
    source,conversion,replay,out=packet;before={p.name:module.sha(p) for p in source.iterdir()}
    module.publish(*packet,'tool');manifest,current,receipt=verify_selected(out)
    state=torch.load(out/'best.pt',weights_only=False)
    assert current['framework']['api']=='V5' and current['migration_provenance']['original_training_framework']['api']=='V4'
    assert state['identity']==receipt['identity']==module.digest(current)
    assert {p.name:module.sha(p) for p in source.iterdir()}==before
    assert (out/'history.json').read_bytes()==(source/'history.json').read_bytes()
    assert module.publish(*packet,'tool')==out


@pytest.mark.parametrize('key,value',[('state','pending'),('api','V4'),('test_access',True),('qualification_updates',2),('development_participants',1),('tool_sha256','other'),('hardware','CPU')])
def test_publication_rejects_missing_or_mismatched_gpu_evidence(packet,key,value):
    source,conversion,replay,out=packet;p=replay/'V5/accepted.json';r=module.read(p);r[key]=value;module.write(p,r)
    with pytest.raises(ValueError,match='A100 acceptance'):module.publish(*packet,'tool')
    assert not out.exists()
