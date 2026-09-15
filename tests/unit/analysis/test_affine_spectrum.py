import numpy as np
import pytest
from look.analysis.affine_spectrum import analyze, compare


@pytest.mark.parametrize('d,q',[(9,2),(5,4),(3,3)])
def test_compact_matches_dense_and_does_not_mutate(d,q):
    rng=np.random.default_rng(12);u=rng.normal(size=(d,q));v=rng.normal(size=(q,d));mx=rng.normal(size=d);c=rng.normal(size=d)
    prior=[x.copy() for x in (u,v,mx,c)];r=analyze(u,v,mx,c,coordinate_system='standardized_reduced_features')
    np.testing.assert_allclose(r['applied_singular_values'],np.linalg.svd(np.eye(d)+u@v,compute_uv=False),atol=1e-12)
    np.testing.assert_allclose(r['correction_singular_values'],np.linalg.svd(u@v,compute_uv=False)[:q],atol=1e-12)
    assert r['affine_bias_l2']==pytest.approx(np.linalg.norm(c-mx@u@v))
    for x,y in zip(prior,(u,v,mx,c)):np.testing.assert_array_equal(x,y)


def test_zero_rank_deficient_and_bias_only_difference():
    u=np.zeros((5,2));v=np.ones((2,5));z=np.zeros(5)
    r=analyze(u,v,z,z,coordinate_system='standardized_reduced_features')
    assert r['stable_rank'] is None and not r['energy_fraction_defined']
    np.testing.assert_allclose(r['applied_singular_values'],1)
    a=dict(left=u,right=v,input_mean=z,output_mean=z);b=dict(a,output_mean=np.ones(5))
    diff=compare(a,b);assert diff['slope_difference_frobenius']==0
    assert diff['mean_difference_l2']==pytest.approx(np.sqrt(5))


def test_same_matrix_different_factor_gauge_and_coordinate_rejection():
    u=np.arange(10.).reshape(5,2);v=np.arange(10.).reshape(2,5);z=np.zeros(5)
    a=dict(left=u,right=v,input_mean=z,output_mean=z);b=dict(a,left=2*u,right=v/2)
    assert compare(a,b)['slope_difference_frobenius']<1e-10
    with pytest.raises(ValueError,match='coordinates'):analyze(u,v,z,z,coordinate_system='raw')
