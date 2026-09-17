import copy
import json
from pathlib import Path
import pytest
from look.runtime.state import atomic_write_json as write,file_sha256,stable_hash
from look.studies import search_tree_migration as m


def fixture(tmp,monkeypatch):
    monkeypatch.setattr(m,'validate',lambda spec:None)
    pins=[]
    for name in ('operator.py','shared_latent.py'):
        p=tmp/'source/src/look/methods'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('unchanged')
        pins.append(dict(path=str(p),sha256=file_sha256(p)))
    spec=dict(mode='best_forward',protocol={},source_pins=pins,host='same',test_access=False)
    new=dict(spec,mode='positive_forward_tree',protocol={'tree':True})
    source=tmp/'old';write(spec,source/'spec.json')
    root=source/'corrections/oct_missing/factors/x16'
    identity=dict(case=stable_hash(spec),pattern='oct_missing',factor=16,latent_dims=[32],max_rank=32)
    contract=dict(mode='best_forward',identity=identity,sites=['a','b']);write(contract,root/'contract.json')
    rid=stable_hash(dict(contract=contract,upstream=[],start=0));rows=[]
    prediction=root/'predictions/base.npz';prediction.parent.mkdir(parents=True);prediction.write_bytes(b'baseline')
    baseline=dict(role='development',data_role='development',score=.5,prediction=str(prediction),sha256=file_sha256(prediction))
    for i,score in enumerate((.6,.4)):
        p=root/f'rounds/000/artifacts/{i}.pt';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(str(i).encode())
        e=dict(baseline,score=score)
        row=dict(index=i,node=contract['sites'][i],key='32',score=score,artifact=str(p.relative_to(root)),sha256=file_sha256(p),evidence=e)
        rows.append(row);write(dict(identity=stable_hash(dict(round=rid,index=i)),candidates=[row]),root/f'rounds/000/site_{i:03d}.json')
    write(dict(identity=rid,baseline=baseline,candidates=rows),root/'rounds/000/decision.json')
    return source,new


def test_import_only_root_and_preserve_source(tmp_path,monkeypatch):
    source,spec=fixture(tmp_path,monkeypatch);before={str(p):file_sha256(p) for p in source.rglob('*') if p.is_file()}
    target=tmp_path/'new';receipt=m.migrate(source,target,spec)
    assert receipt['imported'][0]['candidates']==2 and receipt['imported'][0]['positive_sites']==1
    assert receipt['nonroot_imported'] is False
    root=target/'corrections/oct_missing/factors/x16';assert not (root/'selection.json').exists()
    contract=json.loads((root/'contract.json').read_text())
    for i in range(2):
        value=json.loads((root/f'prefixes/root/site_{i:03d}.json').read_text())
        assert value['identity']==m.candidate_identity(contract,[],i)
        assert file_sha256(root/value['candidates'][0]['artifact'])==value['candidates'][0]['sha256']
    assert before=={str(p):file_sha256(p) for p in source.rglob('*') if p.is_file()}


def test_changed_science_or_artifact_refuses(tmp_path,monkeypatch):
    source,spec=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='settings'):m.migrate(source,tmp_path/'different',dict(spec,host='different'))
    p=source/'corrections/oct_missing/factors/x16/rounds/000/artifacts/0.pt';p.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='changed'):m.migrate(source,tmp_path/'corrupt',spec)
    assert not (tmp_path/'corrupt/migration.json').exists()


def test_tree_is_not_legacy_sequential_report():
    from look.analysis.search_report import route_key
    with pytest.raises(ValueError,match='own report'):route_key(dict(mode='positive_forward_tree'))
