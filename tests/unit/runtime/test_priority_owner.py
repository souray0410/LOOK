import json
from types import SimpleNamespace
from pathlib import Path
from look.runtime.priority_owner import priority,sha
from look.runtime.priority_dispatch import allocation_command


def configuration(tmp):
    b=tmp/'binding.json';b.write_text(json.dumps(dict(output=str(tmp/'continuation'))))
    return dict(output=str(tmp/'owner'),continuation_binding=str(b),continuation_binding_sha256=sha(b),
        pins={str(b):sha(b)},python='python',admission_program='admission.py',continuation_program='continuation.py',continuation_environment={})


def test_handover_and_pause(tmp_path):
    c=configuration(tmp_path)
    assert priority(c,'1',run=lambda *a,**k:SimpleNamespace(returncode=0))['state']=='handover'
    assert priority(c,'1',run=lambda args,**k:SimpleNamespace(returncode=0 if args[1]=='admission.py' else 75))['returncode']==75


def test_failure_is_quarantined_across_allocations(tmp_path):
    c=configuration(tmp_path);calls=[]
    def fail(*a,**k):calls.append(a);return SimpleNamespace(returncode=1)
    assert priority(c,'1',run=fail)['state']=='quarantined'
    assert priority(c,'2',run=fail)['state']=='quarantined'
    assert len(calls)==1 and list(Path(c['output']).glob('quarantine_*.json'))


def test_lock_and_bad_pin_do_not_launch(tmp_path):
    import fcntl
    c=configuration(tmp_path);Path(c['output']).mkdir()
    with (Path(c['output'])/'priority.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert priority(c,'1',run=lambda *a,**k:None)['state']=='owned_elsewhere'
    c['pins'][c['continuation_binding']]='wrong'
    assert priority(c,'2',run=lambda *a,**k:None)['state']=='quarantined'


def test_new_entrypoint_preserves_allocation_resources():
    def original(path,name,python,memory):
        return ['salloc','--time=48:00:00','--mem='+str(memory)+'G',python,'-m','look.runtime.project_dispatch','--config',str(path),'--allocation-owner']
    command=allocation_command(original,'c','n','p',512,'wrapper','owner_config')
    assert command==['salloc','--time=48:00:00','--mem=512G','p','wrapper','--config','owner_config','--allocation-owner']


def test_capacity_skip_is_not_global_quarantine(tmp_path):
    c=configuration(tmp_path);calls=[]
    def unavailable(*a,**k):calls.append(a);return SimpleNamespace(returncode=76)
    assert priority(c,'1',run=unavailable)['state']=='capacity_skipped'
    assert priority(c,'2',run=lambda *a,**k:SimpleNamespace(returncode=0))['state']=='handover'
    assert len(calls)==1 and not list(Path(c['output']).glob('quarantine_*.json'))
