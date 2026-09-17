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
