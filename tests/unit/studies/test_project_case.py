import pytest
from look.studies.project_case import evidence_files,verify_case
from look.runtime.state import atomic_write_json,stable_hash


def test_final_acceptance_preserves_nested_host_receipt_and_rejects_drift(tmp_path):
    spec={'fixed':'spec'}
    for name in ('host/accepted.json','host/best.pt','development/suite.json','controls.json','spec.json'):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
    receipt=dict(schema='look_project_case_v1',identity=stable_hash(spec),test_access=False,state='accepted',
                 host_plateau=True,host_frozen_for_correction=True,files=evidence_files(tmp_path))
    atomic_write_json(receipt,tmp_path/'accepted.json')
    assert 'host/accepted.json' in receipt['files']
    assert 'accepted.json' not in evidence_files(tmp_path)
    assert verify_case(tmp_path,spec)==receipt
    (tmp_path/'host/accepted.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='evidence changed'):verify_case(tmp_path,spec)
