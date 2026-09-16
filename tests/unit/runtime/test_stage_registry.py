import copy
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('stage_registry', Path(__file__).resolve().parents[3] / 'workspace/stage_registry.py')
sr = importlib.util.module_from_spec(spec); spec.loader.exec_module(sr)


def ident():
    return dict(kind='training', scientific_spec={'seed':3416, 'optimizer':{'lr':1e-4}, 'batch':16},
                data_roles={'train':'a'*64, 'dev':'b'*64}, parent_artifacts={'cfp':'c'*64},
                source_sha256='d'*64, framework_sha256='e'*64, view={})


def ref(root, name, content=b'evidence'):
    (root/name).write_bytes(content)
    return dict(path=name, sha256=hashlib.sha256(content).hexdigest())


def record(root, state='accepted'):
    ckpt=ref(root, 'checkpoint.pt'); receipt=ref(root, 'accepted.json')
    files={key:ckpt for key in sr.RESUME_TRAINING} if state=='paused' else {'best':ckpt}
    return dict(run_id='2026_09_14_original', identity=ident(), state=state, files=files, receipt=receipt)


def manifest(stage='eye_core', request=None):
    return dict(schema='research_stage_v1', stage_id=stage, protocol_sha256='f'*64,
                requests=[request or ident()], previous_stages={})


def test_new_phase_reuses_same_run_and_no_copy(tmp_path):
    r=record(tmp_path); calls=[]
    a=sr.resolve(ident(),[r],tmp_path,lambda rec, action:calls.append(action) or True)
    assert a['action']=='reuse' and a['run_id']==r['run_id']
    assert a['execution_authorized'] is False and a['copy_required'] is False
    assert calls==['reuse']


@pytest.mark.parametrize('change', ['seed','lr','data','parent','framework','source','view'])
def test_scientific_changes_require_new_identity(tmp_path,change):
    r=record(tmp_path);new=ident()
    if change=='seed':new['scientific_spec']['seed']=3417
    if change=='lr':new['scientific_spec']['optimizer']['lr']=3e-4
    if change=='data':new['data_roles']['train']='0'*64
    if change=='parent':new['parent_artifacts']['cfp']='0'*64
    if change=='framework':new['framework_sha256']='0'*64
    if change=='source':new['source_sha256']='0'*64
    if change=='view':new['view']={'disable_cross':True}
    assert sr.resolve(new,[r],tmp_path,None)['action']=='new'


def test_resume_preserves_id_requires_complete_state_and_dead_claim(tmp_path):
    r=record(tmp_path,'paused')
    assert sr.resolve(ident(),[r],tmp_path,lambda r,a:a=='resume')['run_id']==r['run_id']
    assert sr.resolve(ident(),[r],tmp_path,lambda r,a:False)['action']=='wait_or_review'
    del r['files']['rng']
    assert sr.resolve(ident(),[r],tmp_path,lambda r,a:True)['action']=='wait_or_review'


def test_corrupt_receipt_or_checkpoint_never_triggers_duplicate_training(tmp_path):
    r=record(tmp_path);(tmp_path/'checkpoint.pt').write_bytes(b'broken')
    assert sr.resolve(ident(),[r],tmp_path,lambda r,a:True)['action']=='wait_or_review'
    r=record(tmp_path);(tmp_path/'accepted.json').unlink()
    assert sr.resolve(ident(),[r],tmp_path,lambda r,a:True)['action']=='wait_or_review'


def test_unknown_live_state_does_not_steal_or_hash_changing_checkpoint(tmp_path):
    r=record(tmp_path,'running');(tmp_path/'checkpoint.pt').unlink()
    def forbidden(*args):raise AssertionError('must not verify a live artifact')
    assert sr.resolve(ident(),[r],tmp_path,forbidden)['action']=='wait_or_review'


def test_receipt_alone_does_not_authorize_reuse(tmp_path):
    r=record(tmp_path)
    with pytest.raises(ValueError):sr.resolve(ident(),[r],tmp_path,None)
    assert sr.resolve(ident(),[r],tmp_path,lambda *args:{'passed':True})['action']=='wait_or_review'


def test_stage_extension_is_idempotent_and_keeps_old_stage(tmp_path):
    one=manifest();old=sr.register_stage(tmp_path,one);before=old.read_bytes()
    assert sr.register_stage(tmp_path,one)==old
    two=manifest('cardiac_extension');two['previous_stages']={'eye_core':sr.fingerprint(one)}
    sr.register_stage(tmp_path,two)
    assert old.read_bytes()==before
    changed=copy.deepcopy(one);changed['protocol_sha256']='0'*64
    with pytest.raises(ValueError):sr.register_stage(tmp_path,changed)
    two['stage_id']='third';two['previous_stages']['eye_core']='0'*64
    with pytest.raises(ValueError):sr.register_stage(tmp_path,two)


def test_test_view_registration_does_not_unlock_test(tmp_path):
    r=record(tmp_path);r['identity']['kind']='prediction';r['identity']['data_roles']={'test':'0'*64}
    sr.register_stage(tmp_path,manifest(request=r['identity']))
    assert sr.resolve(r['identity'],[r],tmp_path,lambda *args:False)['action']=='wait_or_review'


def test_relative_artifact_root_and_path_escape(tmp_path):
    r=record(tmp_path);r['files']['best']['path']='../escape'
    assert sr.resolve(ident(),[r],tmp_path,lambda *args:True)['action']=='wait_or_review'
    m=manifest('../escape')
    with pytest.raises(ValueError):sr.register_stage(tmp_path,m)


def _register_worker(path, value, queue):
    try:sr.register_stage(path,value);queue.put('accepted')
    except ValueError:queue.put('conflict')


def test_multiprocess_conflicting_stage_cannot_overwrite(tmp_path):
    ctx=multiprocessing.get_context('fork');queue=ctx.Queue();one=manifest();two=manifest();two['protocol_sha256']='0'*64
    ps=[ctx.Process(target=_register_worker,args=(tmp_path,v,queue)) for v in (one,two)]
    for p in ps:p.start()
    for p in ps:p.join(10);assert p.exitcode==0
    assert sorted(queue.get(timeout=2) for _ in ps)==['accepted','conflict']
    assert json.loads((tmp_path/'eye_core.json').read_text()) in (one,two)
