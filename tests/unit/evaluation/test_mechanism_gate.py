import pytest
from look.evaluation.mechanism_gate import validate_release,validate_transfer


def test_test_sealed_and_external_not_assumed():
    with pytest.raises(ValueError,match='sealed'):validate_release({}, {}, {}, {}, {})
    with pytest.raises(ValueError):validate_transfer(dict(mode='frozen_transfer'))


def test_test_registry_cannot_release_unreplayed_model(tmp_path):
    import pytest
    from look.runtime.state import stable_hash,file_sha256
    from look.evaluation.mechanism_gate import validate_release
    cp=tmp_path/'best';cp.write_text('frozen')
    models=dict(models=[dict(state='accepted',development_replay='required',checkpoint=str(cp),checkpoint_sha256=file_sha256(cp))],all_training_resolved=True)
    data=dict(role='test');comparisons={};audit=dict(state='accepted',unexplained_exposure=False)
    lock=dict(schema='look_test_release_v1',state='locked',models_sha256=stable_hash(models),data_sha256=stable_hash(data),comparisons_sha256=stable_hash(comparisons),audit_sha256=stable_hash(audit))
    with pytest.raises(ValueError,match='replay'):validate_release(lock,models,comparisons,data,audit)


def test_development_replay_checks_identity_and_decisions(tmp_path):
    import numpy as np
    from look.evaluation.mechanism_test import compare_replay
    from look.runtime.state import file_sha256
    x=tmp_path/'original.npz';y=tmp_path/'replay.npz'
    fields=dict(participant_ids=np.array(['a','b']),labels=np.array([0,1]),patterns=np.array(['cfp_missing']*2))
    np.savez(x,**fields,logits=np.array([[2.,1.],[1.,2.]]))
    np.savez(y,**fields,logits=np.array([[2.,1.],[1.,2.]]))
    e=[dict(method='look',scenario='cfp_missing',path=str(x),sha256=file_sha256(x))]
    a=[dict(method='look',scenario='cfp_missing',path=str(y))]
    assert compare_replay(e,a)['state']=='accepted'
    np.savez(y,**dict(fields,participant_ids=np.array(['b','a'])),logits=np.array([[2.,1.],[1.,2.]]))
    with pytest.raises(ValueError,match='participant_ids'):compare_replay(e,a)
    np.savez(y,**fields,logits=np.array([[1.,2.],[1.,2.]]))
    with pytest.raises(ValueError,match='logits'):compare_replay(e,a)
