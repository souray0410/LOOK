from look.studies.v5_project_feature_materialize import participant_sha

def test_participant_order_is_identity_bound():
 assert participant_sha(['a','b'])!=participant_sha(['b','a'])
 assert participant_sha(['a','b'])==participant_sha(['a','b'])

def test_materializer_exposes_new_batch_budget_name():
 import inspect
 from look.studies.v5_project_feature_materialize import materialize
 assert 'processed_this_run' in inspect.getsource(materialize)
