from dataclasses import replace
import numpy as np
import pytest
import torch
from look.methods.operator import LOOKArtifact,apply_artifact
from look.methods.mechanism_operator import MechanismArtifact,hidden_width,ridge,writeback_gram,fit_mlp,FitPaused
from look.studies.mechanism_protocol import MLP


def artifact():
    return LOOKArtifact('joint_feature','oct_missing','normalized_mean',1,4,(4,),(4,),
        torch.zeros(4),torch.ones(4),torch.zeros(4),torch.eye(4),torch.randn(4,4),torch.randn(4),.1,0.,0.,
        member_names=('oct_feature','cfp_feature'),member_shapes=((2,),(2,)))


def test_default_affine_and_mask():
    a=artifact();x=torch.randn(6,4)
    torch.testing.assert_close(apply_artifact(x,a),apply_artifact(x,MechanismArtifact(a)),rtol=0,atol=0)
    torch.testing.assert_close(apply_artifact(x,a),apply_artifact(x,MechanismArtifact(a,direct_affine=True)))
    z=apply_artifact(x,MechanismArtifact(a,writeback='missing'))
    torch.testing.assert_close(z[:,2:],x[:,2:],rtol=0,atol=0)
    torch.testing.assert_close(writeback_gram(a,'missing'),torch.diag(torch.tensor([1.,1.,0.,0.])).double())
    for d in (1,2,8,32,512):
        h=hidden_width(d);target=d*d+d
        assert abs((2*d*h+h+d)-target)==min(abs((2*d*k+k+d)-target) for k in range(1,d+2))


def test_ridge_identity_center():
    torch.manual_seed(17);a=artifact();x=torch.randn(100,4);y=x.clone()
    fitted=ridge(x,y,a)
    torch.testing.assert_close(fitted.weight,torch.zeros_like(fitted.weight),atol=1e-6,rtol=0)
    torch.testing.assert_close(apply_artifact(x,fitted),x,atol=1e-6,rtol=0)
    restricted=ridge(x,y+2,a,writeback_gram(a,'missing'))
    assert torch.isfinite(restricted.weight).all()


def test_mlp_resume_and_rng(tmp_path,monkeypatch):
    monkeypatch.setitem(MLP,'epochs',3);monkeypatch.setitem(MLP,'batch',8)
    torch.manual_seed(2);x=torch.randn(40,4);y=x+torch.relu(x)
    ids=[str(i) for i in range(20)];counts=[2]*20
    state=torch.get_rng_state().clone()
    a,h,info=fit_mlp(x,y,ids,counts,5,tmp_path/'a.pt')
    assert torch.equal(state,torch.get_rng_state())
    with pytest.raises(FitPaused):fit_mlp(x,y,ids,counts,5,tmp_path/'b.pt',lambda:True)
    b,_,other=fit_mlp(x,y,ids,counts,5,tmp_path/'b.pt')
    assert info==other
    for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)


def test_residual_penalty_centres_direct_map_at_identity():
    torch.manual_seed(73)
    x=torch.randn(35,4,dtype=torch.float64);y=torch.randn_like(x);lam=.3
    eye=torch.eye(4,dtype=x.dtype)
    w=torch.linalg.solve(x.T@x+lam*eye,x.T@(y-x))
    centred=torch.linalg.solve(x.T@x+lam*eye,x.T@y+lam*eye)
    ordinary=torch.linalg.solve(x.T@x+lam*eye,x.T@y)
    torch.testing.assert_close(centred,eye+w)
    assert not torch.allclose(ordinary,centred)
