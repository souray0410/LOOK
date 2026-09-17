import importlib.util
from pathlib import Path
p=Path(__file__).resolve().parents[3]/'experiments/serial_delivery_20260917/watch_delivery.py'
s=importlib.util.spec_from_file_location('watch',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_fresh_manager_heartbeat_does_not_hide_stalled_experiment():
    assert m.assess({'state':'running_formal','updated_at':1000},0,1000,True)=='action_required_no_experimental_progress'

def test_terminal_failure_is_not_healthy_when_log_is_fresh():
    assert m.assess({'state':'failed'},1000,1000,True)=='action_required_failure'

def test_manager_death_pause_progress_are_distinct():
    assert m.assess({'state':'running_formal'},100,110,False)=='action_required_manager_dead'
    assert m.assess({'state':'paused'},100,110,True)=='action_required_resumable_pause'
    assert m.assess({'state':'running_formal'},100,110,True)=='progress_observed_not_acceptance'


def test_failure_history_survives_healthy_observation_and_restart(tmp_path):
    import json
    def record(state,t):
        return dict(state=state,time=t,sequence_state={'run':'r'},manager={'step':'10'})
    m.journal(tmp_path,record('action_required_failure',1))
    first=json.loads((tmp_path/'journal_state.json').read_text())
    m.journal(tmp_path,record('action_required_failure',2))
    assert len((tmp_path/'execution_events.jsonl').read_text().splitlines())==1
    m.journal(tmp_path,record('progress_observed_not_acceptance',3))
    current=json.loads((tmp_path/'journal_state.json').read_text())
    assert current['incident']==first['incident']
    assert current['incident_state']=='progress_returned_requires_repair_acceptance'
    assert not current['scientific_acceptance']
    m.journal(tmp_path,record('progress_observed_not_acceptance',4))
    assert len((tmp_path/'execution_events.jsonl').read_text().splitlines())==2
    assert len(list((tmp_path/'incidents').glob('*/observations.jsonl')))==1


def test_finite_completion_is_not_repair_acceptance(tmp_path):
    import json
    m.journal(tmp_path,dict(state='action_required_no_experimental_progress',time=1,sequence_state={'run':'r'}))
    m.journal(tmp_path,dict(state='finite_sequence_reported_complete',time=2,sequence_state={'run':'r'}))
    event=json.loads((tmp_path/'journal_state.json').read_text())
    assert event['incident'] and not event['scientific_acceptance']
    assert 'acceptance' in event['next_action']
