import pytest
from look.runtime.v5_matched_stage import outcome


@pytest.mark.parametrize('state,code,status,accept,expected',[
    ('COMPLETED','0:0',{},dict(state='accepted',test_access=False),'accepted'),
    ('COMPLETED','0:0',{},None,'failed_closed'),
    ('COMPLETED','0:0',{},dict(state='accepted',test_access=True),'failed_closed'),
    ('FAILED','75:0',dict(state='paused'),None,'resume'),
    ('FAILED','75:0',dict(state='failed'),None,'failed_closed'),
    ('FAILED','1:0',dict(state='paused'),None,'failed_closed'),
    ('OUT_OF_MEMORY','0:125',{},None,'failed_closed'),
    ('TIMEOUT','0:15',{},None,'resume'),
])
def test_only_valid_completion_or_lease_release_advances(state,code,status,accept,expected):
    assert outcome(dict(state=state,exit_code=code),status,accept)==expected


def test_live_predecessor_is_never_reclaimed():
    with pytest.raises(ValueError,match='not terminal'):
        outcome(dict(state='RUNNING',exit_code='0:0'),{},None)
