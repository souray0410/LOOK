import inspect
from look.studies import v5_project_sharded_first_decision as subject

def test_downstream_fails_closed_before_complete_cache():
 source=inspect.getsource(subject.execute)
 assert "cache_accept.get('state')!='accepted_complete'" in source
 assert "len(sites)!=9" in source
 assert "[sites[0]]" in source

def test_normal_entrypoint_requires_exclusive_claim():
 assert 'validate_claim()' in inspect.getsource(subject.main)
