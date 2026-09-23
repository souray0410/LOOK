from look.studies.v5_project_feature_materialize import participant_sha

def test_participant_order_is_identity_bound():
 assert participant_sha(['a','b'])!=participant_sha(['b','a'])
 assert participant_sha(['a','b'])==participant_sha(['a','b'])
