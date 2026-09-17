import json
import pytest
from look.runtime.delivery_policy import order_tasks
from look.runtime.state import file_sha256


def setup(tmp_path):
    tasks=[];entries=[]
    for name,role in [('next','preparation'),('control','matched_control'),('main','delivery'),('parent','critical_dependency')]:
        spec=tmp_path/(name+'.json');spec.write_text('{}')
        task=dict(id=name,spec=str(spec),spec_sha256=file_sha256(spec),run_dir=str(tmp_path/name),execution='look_search')
        tasks.append(task);entries.append(dict(task,role=role,dependencies=[]))
    path=tmp_path/'policy.json';path.write_text(json.dumps(dict(schema='research_delivery_policy_v1',tasks=entries,test_access=False)))
    return tasks,entries,path


def test_independent_controls_ready_without_main_finished(tmp_path):
    tasks,_,path=setup(tmp_path)
    ready,held=order_tasks(tasks,path,lambda e:pytest.fail('Independent task should not need a peer'))
    assert [t['id'] for t in ready]==['parent','main','control','next']
    assert held==[]


def test_dependency_verifies_original_artifacts_not_accepted_string(tmp_path):
    tasks,entries,path=setup(tmp_path)
    entries[1]['dependencies']=[entries[3]['run_dir']]
    policy=json.loads(path.read_text());policy['tasks']=entries;path.write_text(json.dumps(policy))
    assert len(order_tasks(tasks,path,lambda e:None)[1])==1
    parent=tmp_path/'parent';parent.mkdir();(parent/'accepted.json').write_text('{"state":"accepted"}')
    def reject(entry):raise ValueError('forged scientific receipt')
    with pytest.raises(ValueError,match='forged'):order_tasks(tasks,path,reject)


def test_cycles_and_changed_specs_rejected(tmp_path):
    tasks,entries,path=setup(tmp_path)
    entries[0]['dependencies']=[entries[1]['run_dir']];entries[1]['dependencies']=[entries[0]['run_dir']]
    d=json.loads(path.read_text());d['tasks']=entries;path.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='Cyclic'):order_tasks(tasks,path,lambda e:None)
    (tmp_path/'next.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='configuration'):order_tasks(tasks,path,lambda e:None)


def test_unregistered_project_held_generic_models_last(tmp_path):
    tasks,_,path=setup(tmp_path)
    unknown=dict(tasks[0],run_dir='unknown',id='other')
    native=dict(unknown,run_dir='native',id='native',execution='native')
    ready,held=order_tasks([native,unknown]+tasks,path,lambda e:None)
    assert ready[-1]['id']=='native'
    assert held[0]['reason']=='outside_finite_delivery_policy'
