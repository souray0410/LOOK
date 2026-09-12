import json
from pathlib import Path
import pytest
from look.studies import project_orders
from look.runtime.state import atomic_write_json, file_sha256


def test_completed_parents_register_three_seeds_and_three_starts_once(tmp_path, monkeypatch):
    monkeypatch.setattr(project_orders, 'DISEASES', ['cataract'])
    monkeypatch.setattr(project_orders, 'MODELS', ['resnet50'])
    gate=tmp_path/'gate.json';atomic_write_json({'status':'accepted'},gate)
    config=dict(catalog={'sha256':'a'*64},protocol={'sha256':'b'*64},source_pins=[],
        project=dict(output=str(tmp_path/'projects'),training={},look={},
                     runtime_gate=dict(path=str(gate),sha256=file_sha256(gate))))
    groups={}
    for track in ('cfp_2d','oct_bscan_2d'):
        parents=[]
        for seed in (3416,3417,3418):
            root=tmp_path/track/str(seed);root.mkdir(parents=True)
            atomic_write_json({'training':{'seed':seed}},root/'spec.json')
            atomic_write_json({'seed':seed},root/'selected_artifact.json')
            parents.append({'run_dir':str(root)})
        groups['cataract/resnet50/'+track]=dict(state='waiting_project_adapter',selected=parents[0],replicas=parents[1:])
    monkeypatch.setattr(project_orders,'materialize_selected',lambda source,*args:Path(source))
    reservations={}
    def reserve(root, namespace,key,spec,**kwargs):
        if key in reservations:assert reservations[key]==spec
        reservations[key]=spec
        return root/'runs'/key
    first=project_orders.advance(config,groups,None,reserve)
    before=Path(first['queue']).read_bytes()
    assert first['tasks']==9 and len(reservations)==9
    project_orders.advance(config,groups,None,reserve)
    assert Path(first['queue']).read_bytes()==before
    assert {s['seed'] for s in reservations.values()}=={3416,3417,3418}
    assert {s['position'] for s in reservations.values()}=={'middle','deep','features'}
    assert all(s['test_access'] is False for s in reservations.values())
    gate.write_text('{}')
    with pytest.raises(ValueError,match='runtime gate'):
        project_orders.advance(config,groups,None,reserve)


def test_incomplete_pair_creates_no_gpu_work(tmp_path,monkeypatch):
    monkeypatch.setattr(project_orders,'DISEASES',['cataract'])
    monkeypatch.setattr(project_orders,'MODELS',['resnet50'])
    gate=tmp_path/'gate';atomic_write_json({'status':'accepted'},gate)
    config={'project':dict(output=str(tmp_path/'out'),runtime_gate=dict(path=str(gate),sha256=file_sha256(gate)))}
    groups={'cataract/resnet50/'+track:{'state':'waiting_replications'} for track in ('cfp_2d','oct_bscan_2d')}
    def forbidden(*args,**kwargs):raise AssertionError('No premature registration')
    assert project_orders.advance(config,groups,forbidden,forbidden)['tasks']==0
