import inspect
from look.studies import v5_project_feature_materialize_sharded as subject

def test_sharded_cache_commits_directory_after_all_sites():
 source=inspect.getsource(subject.materialize)
 assert "partial.rename(final)" in source
 assert "for ordinal,site in enumerate(sites)" in source
 assert source.index("write_json_atomic(row,partial/'receipt.json')") < source.index("partial.rename(final)")

def test_sharded_identity_is_order_sensitive():
 assert subject.participant_sha(['a','b'])!=subject.participant_sha(['b','a'])
