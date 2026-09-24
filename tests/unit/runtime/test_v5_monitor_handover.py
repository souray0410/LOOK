from look.runtime.v5_monitor_handover import advance, stage2_action

def test_held_first_handover_and_resume(tmp_path):
    states={'52430491':{'state':'PENDING','dependency':'afterany:52429877(unfulfilled)'},
            '600':{'state':'PENDING','reason':'JobHeldUser','dependency':'afterany:52429877(unfulfilled)'}}
    events=[]
    def observe(j): return states[j]
    def submit(spec): events.append('submit-held'); return '600'
    def cancel(j): events.append('cancel-old'); states[j]={'state':'CANCELLED'}
    def release(j): events.append('release'); states[j]['state']='PENDING'; states[j]['reason']='Dependency'
    receipt=tmp_path/'receipt.json'
    result=advance(receipt,observe=observe,find_held=lambda spec:None,submit_held=submit,cancel=cancel,release=release,spec={'sha':'v21'})
    assert result['state']=='released'
    assert events==['submit-held','cancel-old','release']
    assert advance(receipt,observe=observe,find_held=lambda spec:'600',submit_held=submit,cancel=cancel,release=release,spec={'sha':'v21'})['state']=='released'

def test_stage2_waits_for_running_monitor():
    assert stage2_action('RUNNING')=='wait_for_stage1_monitor_terminal'
    assert stage2_action('PENDING')=='replace_pending_monitor_with_exact_models_sha'
