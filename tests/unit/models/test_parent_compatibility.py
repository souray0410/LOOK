import pytest
from mhd_framework.models import artifacts
from look.models.parent_compatibility import verify_inference_equivalence


def test_registry_change_does_not_authorize_operator_drift(monkeypatch):
    paths=['core.py','utils.py','__init__.py','models/graph.py','models/resnet.py']
    source={p:'fixed' for p in paths};source['models/artifacts.py']='old'
    spec={'framework':{'commit':'3559caa8d596d4438533a69d39d8a2c32eb21e46','source_sha256':source},'model':{'name':'resnet50'}}
    current=dict(source);current['models/artifacts.py']='new'
    monkeypatch.setattr(artifacts,'runtime_source_sha256',lambda:current)
    receipt=verify_inference_equivalence(spec)
    assert 'inference_only' in receipt['scope']
    current['core.py']='changed'
    with pytest.raises(ValueError,match='operator source'):verify_inference_equivalence(spec)
    spec['framework']['commit']='unknown'
    with pytest.raises(ValueError,match='Unreviewed'):verify_inference_equivalence(spec)
