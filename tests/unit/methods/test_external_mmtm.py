import torch
from look.methods.external_mmtm import MMTM


def test_explicit_gate_equations_and_nonzero_input_gradients():
    torch.manual_seed(7)
    model=MMTM(5,7,4,2.)
    a=torch.randn(3,5,8,6,requires_grad=True);b=torch.randn(3,7,4,9,requires_grad=True)
    z=torch.relu(model.fc_squeeze(torch.cat([a.mean((2,3)),b.mean((2,3))],1)))
    expected=(a*2*torch.sigmoid(model.fc_visual(z))[:,:,None,None],b*2*torch.sigmoid(model.fc_skeleton(z))[:,:,None,None])
    actual=model(a,b)
    for x,y in zip(actual,expected):torch.testing.assert_close(x,y)
    actual[0].square().mean().backward()
    assert a.grad.abs().sum()>0 and b.grad.abs().sum()>0
