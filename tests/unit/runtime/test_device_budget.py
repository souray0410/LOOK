import pytest
from look.runtime.device_budget import validate


@pytest.mark.parametrize('reserve', [10 * 1024**3, 1, -1, None])
def test_old_or_unknown_fixed_reserve_requires_explicit_migration(reserve):
    with pytest.raises(ValueError, match='explicitly migrate'):
        validate(dict(gpu_reserve_bytes=reserve, gpu_budget_bytes=2*1024**3))


@pytest.mark.parametrize('cap', [0, -1, float('nan'), float('inf'), None, True])
def test_unknown_workload_budget_is_not_optimistically_admitted(cap):
    with pytest.raises(ValueError, match='validated workload'):
        validate(dict(gpu_reserve_bytes=0, gpu_budget_bytes=cap))


def test_validated_workload_cap_has_no_new_fixed_reserve():
    assert validate(dict(gpu_reserve_bytes=0, gpu_budget_bytes=3*1024**3)) == 3*1024**3
