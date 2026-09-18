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
