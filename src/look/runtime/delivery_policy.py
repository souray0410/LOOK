"""Finite priority/dependency view for the existing dispatcher; no GPU owner."""
import json
from pathlib import Path
from look.runtime.state import file_sha256

ROLES = {'critical_dependency': 0, 'delivery': 1, 'matched_control': 2,
         'preparation': 3, 'generic_model': 4}


def read(path):
    return json.loads(Path(path).read_text())


def order_tasks(tasks, path, verify):
    policy=read(path)
    if policy.get('schema')!='research_delivery_policy_v1' or policy.get('test_access') is not False:
        raise ValueError('Invalid delivery policy')
    entries={}
    for entry in policy['tasks']:
        key=str(Path(entry['run_dir']).resolve())
        if key in entries or entry['role'] not in ROLES:raise ValueError('Duplicate or unknown delivery role')
        if file_sha256(entry['spec'])!=entry['spec_sha256']:raise ValueError('Delivery configuration changed')
        entries[key]=entry
    # Reject cycles even if a forged completion flag would otherwise hide them.
    done=set();active=set()
    def visit(key):
        if key in active:raise ValueError('Cyclic delivery dependencies')
        if key in done:return
        active.add(key)
        for dependency in entries[key].get('dependencies',[]):
            other=str(Path(dependency).resolve())
            if other not in entries:raise ValueError('Unregistered delivery dependency')
            visit(other)
        active.remove(key);done.add(key)
    for key in entries:visit(key)
    ready=[];held=[]
    for index,task in enumerate(tasks):
        key=str(Path(task['run_dir']).resolve());entry=entries.get(key)
        if entry is None:
            if task['execution']=='native':ready.append((ROLES['generic_model'],index,task));continue
            held.append(dict(run=key,reason='outside_finite_delivery_policy'));continue
        if task['spec_sha256']!=entry['spec_sha256']:raise ValueError('Task and delivery identity differ')
        missing=[]
        for dependency in entry.get('dependencies',[]):
            dep=entries[str(Path(dependency).resolve())]
            if not (Path(dep['run_dir'])/'accepted.json').exists():missing.append(dependency);continue
            verify(dep)  # The owning scientific verifier, not a state string.
        if missing:held.append(dict(run=key,reason='waiting_dependencies',dependencies=missing));continue
        ready.append((ROLES[entry['role']],index,task))
    return [t for _,_,t in sorted(ready,key=lambda x:x[:2])],held
