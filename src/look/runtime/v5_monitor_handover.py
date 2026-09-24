"""Crash-resumable held-job replacement for a pending LOOK finalizer."""
import json, os
from pathlib import Path

STEPS=('prepared','held_submitted','held_verified','old_cancelled','old_terminal','released')

def persist(path,state):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('x') as f:
        f.write(json.dumps(state,sort_keys=True)+'\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def advance(receipt_path, *, observe, find_held, submit_held, cancel, release, spec):
    path=Path(receipt_path)
    state=json.loads(path.read_text()) if path.exists() else {'state':'prepared','old_job':'52430491','new_job':None,'spec':spec}
    if state['spec']!=spec or state['old_job']!='52430491': raise ValueError('Handover identity changed')
    old=observe('52430491')
    if state['state']=='prepared':
        if old['state']!='PENDING' or old['dependency']!='afterany:52429877(unfulfilled)': raise ValueError('Old monitor is not replaceable')
        existing=find_held(spec)
        state['new_job']=str(existing if existing is not None else submit_held(spec))
        state['state']='held_submitted'; persist(path,state)
    if state['state']=='held_submitted':
        new=observe(state['new_job'])
        if new['state']!='PENDING' or new['reason']!='JobHeldUser' or new['dependency']!='afterany:52429877(unfulfilled)': raise ValueError('Replacement is not exact and held')
        state['state']='held_verified'; persist(path,state)
    if state['state']=='held_verified':
        cancel('52430491'); state['state']='old_cancelled'; persist(path,state)
    if state['state']=='old_cancelled':
        old=observe('52430491')
        if old['state'] not in ('CANCELLED','COMPLETED'): raise ValueError('Old monitor is not terminal')
        state['state']='old_terminal'; persist(path,state)
    if state['state']=='old_terminal':
        release(state['new_job']); state['state']='released'; persist(path,state)
    return state

def stage2_action(monitor_state):
    if monitor_state=='RUNNING': return 'wait_for_stage1_monitor_terminal'
    if monitor_state=='PENDING': return 'replace_pending_monitor_with_exact_models_sha'
    if monitor_state in ('COMPLETED','CANCELLED'): return 'install_models_policy_after_fresh_ownership_audit'
    raise ValueError('Unknown monitor state')
