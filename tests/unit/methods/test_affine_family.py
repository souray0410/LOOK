from dataclasses import asdict
import numpy as np
import pytest
import torch
from scipy.linalg import orthogonal_procrustes
from sklearn.cross_decomposition import PLSSVD
from look.methods.affine_family import ARMS, NEW_ARMS, FamilyArtifact, fit_map
from look.methods.linear_vector import ResidualMoments
from look.methods.operator import LOOKArtifact


def fixture():
    g=torch.Generator().manual_seed(71)
    x=torch.randn(79,7,generator=g,dtype=torch.float64)+3
    d=x@torch.randn(7,7,generator=g,dtype=x.dtype)+2
    s=ResidualMoments.empty(7);s.update(x,d)
    return x,d,s,torch.eye(7,dtype=x.dtype)[:3]


def test_mean_subspace_factorial_and_ridge_nested_objective():
    x,d,s,q=fixture();m={a:fit_map(s,3,.2,arm=a,basis=q) for a in ARMS}
    torch.testing.assert_close(m['pca_free_mean'].left,m['shared_pca_ridge'].left)
    torch.testing.assert_close(m['pca_free_mean'].predict(x).mean(0),d.mean(0))
    torch.testing.assert_close(m['residual_rrr'].left,m['rrr_shared_intercept'].left)
    expected=torch.linalg.solve(s.cxx+.2*torch.eye(7),s.cxy)
    torch.testing.assert_close(m['residual_ridge'].left,expected)
    assert m['residual_ridge'].diagnostics['regularized_objective']<=m['residual_rrr'].diagnostics['regularized_objective']+1e-8
    # Optimizing residual error does not establish a classification ordering.
    assert m['residual_ridge'].diagnostics['rank_budget_kind']=='not_rank_matched'


def test_orthogonal_equals_independent_scipy_and_diagonal_closed_form():
    x,d,s,q=fixture();o=fit_map(s,3,.2,arm='orthogonal_alignment',basis=q)
    expected,_=orthogonal_procrustes((x-x.mean(0)).numpy(),(x+d-(x+d).mean(0)).numpy())
    torch.testing.assert_close(o.left+torch.eye(7),torch.from_numpy(expected))
    torch.testing.assert_close((o.left+torch.eye(7)).T@(o.left+torch.eye(7)),torch.eye(7,dtype=x.dtype))
    a=fit_map(s,3,.2,arm='diagonal_ridge',basis=q)
    torch.testing.assert_close(a.left,((x-x.mean(0))*(d-d.mean(0))).sum(0)/((x-x.mean(0)).square().sum(0)+.2))
    assert a.left.numel()+a.right.numel()==7 and o.diagnostics['ridge_applied'] is False


def test_pls_svd_matches_independent_directions_and_projected_regression():
    x,d,s,q=fixture();m=fit_map(s,3,.2,arm='pls_svd_ridge',basis=q)
    reference=PLSSVD(n_components=3,scale=False).fit(x.numpy(),d.numpy())
    u=torch.from_numpy(reference.x_weights_)
    torch.testing.assert_close(m.left@m.left.T,u@u.T)
    z=(x-x.mean(0))@u
    expected=u@torch.linalg.solve(z.T@z+.2*torch.eye(3),z.T@(d-d.mean(0)))
    torch.testing.assert_close(m.left@m.right,expected)


@pytest.mark.parametrize('arm',NEW_ARMS)
def test_affinity_reload_and_no_rng_mutation(arm,tmp_path):
    x,d,s,q=fixture();rng=torch.get_rng_state().clone();m=fit_map(s,3,.2,arm=arm,basis=q)
    a=LOOKArtifact('fusion_participant_feature','oct_missing','normalized_mean',1,3,(7,),(7,),
        torch.zeros(7),torch.ones(7),torch.zeros(7),q,torch.zeros(3,3),torch.zeros(3),.2,0.,0.)
    artifact=FamilyArtifact(a,m);path=tmp_path/'bank.pt';torch.save(artifact.record(),path)
    loaded=FamilyArtifact.from_record(torch.load(path,weights_only=False))
    torch.testing.assert_close(loaded.apply_feature(x),artifact.apply_feature(x),rtol=0,atol=0)
    t=.31;z=x.flip(0)
    torch.testing.assert_close(loaded.apply_feature(t*x+(1-t)*z),t*loaded.apply_feature(x)+(1-t)*loaded.apply_feature(z))
    assert torch.equal(rng,torch.get_rng_state())


def test_reject_invalid_and_zero_residual():
    x,d,s,q=fixture()
    for arm in NEW_ARMS:
        with pytest.raises(ValueError):fit_map(s,80,.2,arm=arm,basis=q)
    z=ResidualMoments.empty(7);z.update(x,torch.zeros_like(x))
    for arm in NEW_ARMS:
        m=fit_map(z,3,.2,arm=arm,basis=q)
        torch.testing.assert_close(m.predict(x),torch.zeros_like(x),atol=1e-12,rtol=0)


def test_family_progressive_pause_resume_and_old_schema_is_separate(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from look.methods import linear_operator as op
    x,d,s,q=fixture()
    a=LOOKArtifact('joint_feature','oct_missing','normalized_mean',1,3,(7,),(7,),
        torch.zeros(7),torch.ones(7),torch.zeros(7),q,torch.zeros(3,3),torch.zeros(3),.2,0.,0.)
    graph=torch.nn.Linear(1,1).eval()
    for p in graph.parameters():p.requires_grad_(False)
    ld=SimpleNamespace(dataset=SimpleNamespace(split='train',augment=False),drop_last=False)
    def pairs(*args,**kw):
        for xx,dd in zip(x.split(8),d.split(8)):yield xx+dd,xx,(7,),(7,)
    monkeypatch.setattr(op,'iter_feature_pairs',pairs);ticks=[0]
    def pause():ticks[0]+=1;return ticks[0]==3
    args=(graph,ld,[a],'diagonal_ridge','cpu')
    with pytest.raises(op.LinearFitPaused):op.fit_bank(*args,tmp_path/'resume',identity='same',workspace_bytes=10**8,should_pause=pause,family=True)
    r,_=op.fit_bank(*args,tmp_path/'resume',identity='same',workspace_bytes=10**8,family=True)
    f,_=op.fit_bank(*args,tmp_path/'fresh',identity='same',workspace_bytes=10**8,family=True)
    torch.testing.assert_close(r[0].apply_feature(x),f[0].apply_feature(x),rtol=0,atol=0)
    with pytest.raises(ValueError):op.LinearVectorArtifact.from_record(r[0].record())
    ld.dataset.split='development'
    with pytest.raises(ValueError):op.fit_bank(*args,tmp_path/'dev',identity='same',workspace_bytes=10**8,family=True)
