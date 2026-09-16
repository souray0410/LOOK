import pytest
from look.studies.affine_protocol import protocol,counts,validate,VERSION,matrix
from look.analysis.affine_report import contrast_table
from look.methods.affine_family import ARMS


def test_protocol_scope_counts_and_replication_gate():
    assert counts()['new_method_views']==81*2*5
    h=next(h for h in matrix() if h['seed']==3416)
    s=dict(schema=VERSION,scope='terminal',protocol=protocol('terminal'),host=h,test_access=False)
    validate(s)
    with pytest.raises(ValueError):validate(dict(s,test_access=True))
    with pytest.raises(ValueError):validate(dict(s,host=dict(h,seed=3417)))
    assert protocol('progressive')['upstream']=='sequential_refit_within_arm'


def test_contrasts_are_paired_equal_seed_and_factorial():
    seeds=[3416,3417,3418];keys=[(s,m,p) for s in seeds for m in (*ARMS,'host') for p in ('oct_missing','cfp_missing')]
    w,d=contrast_table(keys,seeds)
    assert len(w)==51 and all(abs(a.sum())<1e-12 for a in w)
    assert d[-1]['name']=='mean_by_subspace_interaction'
    assert w[-1][keys.index((3416,'residual_rrr','oct_missing'))]==pytest.approx(1/6)
