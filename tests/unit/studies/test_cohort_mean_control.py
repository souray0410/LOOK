from types import SimpleNamespace
import torch
from look.studies.cohort_mean_control import constrain_mean


def test_projection_changes_only_mean():
    mapping=SimpleNamespace(input_mean=torch.tensor([1.,2.,3.]),output_mean=torch.tensor([2.,4.,8.],dtype=torch.float64),left=torch.tensor([1.]),right=torch.tensor([2.]),method='residual_rrr',rank=2,ridge_lambda=.2,diagnostics={})
    a=SimpleNamespace(base=SimpleNamespace(components=torch.eye(3)[:2]),mapping=mapping)
    b=constrain_mean(a)
    assert torch.equal(b.mapping.output_mean,torch.tensor([2.,4.,0.],dtype=torch.float64))
    assert torch.equal(a.mapping.output_mean,torch.tensor([2.,4.,8.],dtype=torch.float64))
    assert torch.equal(b.mapping.left,a.mapping.left) and b.mapping.ridge_lambda==a.mapping.ridge_lambda


def test_real_family_map_matches_constrained_fit():
    from look.methods.affine_family import fit_map
    from look.methods.linear_vector import ResidualMoments
    gen=torch.Generator().manual_seed(14)
    x=torch.randn(25,5,generator=gen,dtype=torch.float64);y=torch.randn(25,5,generator=gen,dtype=torch.float64)+1
    moments=ResidualMoments.empty(5);moments.update(x,y);q=torch.eye(5,dtype=torch.float64)[:2]
    for free,restricted in [('pca_free_mean','shared_pca_ridge'),('residual_rrr','rrr_shared_intercept')]:
        a=SimpleNamespace(base=SimpleNamespace(components=q),mapping=fit_map(moments,2,.2,arm=free,basis=q))
        b=constrain_mean(a);expected=fit_map(moments,2,.2,arm=restricted,basis=q)
        torch.testing.assert_close(b.mapping.predict(x),expected.predict(x))
