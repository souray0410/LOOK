from dataclasses import asdict
import pytest
import torch
from look.methods.linear_vector import ResidualMoments,solve,estimated_workspace_bytes
from look.methods.linear_operator import LinearVectorArtifact,fingerprint
from look.methods.operator import LOOKArtifact,apply_artifact


def data():
    g=torch.Generator().manual_seed(311)
    x=torch.randn(71,7,generator=g,dtype=torch.float64)+3
    y=x@torch.randn(7,7,generator=g,dtype=torch.float64)+torch.randn(71,7,generator=g,dtype=torch.float64)*.2+2
    return x,y


def stats(x,y,chunk=19):
    s=ResidualMoments.empty(x.shape[1])
    for a,b in zip(x.split(chunk),y.split(chunk)):s.update(a,b)
    return s


def test_streaming_moments_resume_and_rng():
    x,y=data();rng=torch.get_rng_state().clone();s=stats(x,y)
    torch.testing.assert_close(s.cxx,(x-x.mean(0)).T@(x-x.mean(0)))
    torch.testing.assert_close(s.cxy,(x-x.mean(0)).T@(y-y.mean(0)))
    t=stats(x[:20],y[:20]);t=ResidualMoments(**asdict(t));t.update(x[20:],y[20:])
    torch.testing.assert_close(t.cxy,s.cxy)
    assert torch.equal(rng,torch.get_rng_state())


def objective(x,y,m,lam):
    return (y-m.predict(x)).square().sum()+lam*(m.left@m.right).square().sum()


def test_full_rank_rrr_equals_ridge_and_low_rank_optimality():
    x,y=data();s=stats(x,y);lam=.3
    full=solve(s,7,lam);expected=torch.linalg.solve(s.cxx+lam*torch.eye(7),s.cxy)
    torch.testing.assert_close(full.left@full.right,expected)
    vals=[]
    for q in range(1,8):
        m=solve(s,q,lam);vals.append(float(objective(x,y,m,lam)))
        assert torch.linalg.matrix_rank(m.left@m.right)<=q
        assert m.diagnostics['regularized_objective']==pytest.approx(vals[-1])
        torch.testing.assert_close(m.predict(x).mean(0),y.mean(0))
    assert all(a>=b-1e-8 for a,b in zip(vals,vals[1:]))


def test_shared_intercept_separates_mean_and_subspace():
    x,y=data();s=stats(x,y);q=torch.eye(7,dtype=x.dtype)[:3]
    p=solve(s,3,.2,basis=q);r=solve(s,3,.2,intercept_basis=q);f=solve(s,3,.2)
    torch.testing.assert_close(p.output_mean,r.output_mean)
    torch.testing.assert_close(r.left,f.left);torch.testing.assert_close(r.right,f.right)
    assert objective(x,y,r,.2)<=objective(x,y,p,.2)+1e-8
    assert objective(x,y,f,.2)<=objective(x,y,r,.2)+1e-8
    # Directions outside the fixed PCA span are available to RRR.
    assert (r.left@r.right)[:,3:].abs().sum()>0


def test_artifact_replay_affinity_and_pca_equivalence(tmp_path):
    x,y=data();q=torch.eye(7,dtype=x.dtype)[:3];s=stats(x,y)
    m=solve(s,3,.3,basis=q)
    w=q@(m.left@m.right)@q.T
    b=(s.mean_y-s.mean_x@(m.left@m.right))@q.T
    a=LOOKArtifact('joint_feature','oct_missing','normalized_mean',1,3,(7,),(7,),
        torch.zeros(7,dtype=x.dtype),torch.ones(7,dtype=x.dtype),torch.zeros(7,dtype=x.dtype),q,w,b,.3,0.,0.)
    artifact=LinearVectorArtifact(a,m)
    torch.testing.assert_close(apply_artifact(x,a),apply_artifact(x,artifact))
    path=tmp_path/'a.pt';record=artifact.record();torch.save(record,path)
    other=LinearVectorArtifact.from_record(torch.load(path,weights_only=False))
    assert fingerprint(record)==fingerprint(other.record())
    torch.testing.assert_close(other.apply_feature(x),artifact.apply_feature(x),rtol=0,atol=0)
    z=x.flip(0);t=.37
    torch.testing.assert_close(artifact.apply_feature(t*x+(1-t)*z),t*artifact.apply_feature(x)+(1-t)*artifact.apply_feature(z))


def test_invalid_inputs_and_identity_correction():
    x,y=data();s=stats(x,torch.zeros_like(y));m=solve(s,3,.1)
    torch.testing.assert_close(m.predict(x),torch.zeros_like(x))
    for rank,lam in [(8,.1),(0,.1),(3,0),(3,float('nan'))]:
        with pytest.raises(ValueError):solve(s,rank,lam)
    with pytest.raises(ValueError):solve(s,3,.1,basis=torch.ones(3,7))
    with pytest.raises(ValueError):s.update(x,torch.full_like(y,float('nan')))
    assert estimated_workspace_bytes(20,5)>estimated_workspace_bytes(10,5)


def test_resumable_fit_uses_fixed_host_and_rejects_drift(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from look.methods import linear_operator as op
    x,y=data();template=LOOKArtifact('joint_feature','oct_missing','normalized_mean',1,3,(7,),(7,),
        torch.zeros(7,dtype=x.dtype),torch.ones(7,dtype=x.dtype),torch.zeros(7,dtype=x.dtype),
        torch.eye(7,dtype=x.dtype)[:3],torch.zeros(3,3),torch.zeros(3),.3,0.,0.)
    graph=torch.nn.Linear(1,1).eval()
    for p in graph.parameters():p.requires_grad_(False)
    loader=SimpleNamespace(dataset=SimpleNamespace(split='train',augment=False),drop_last=False)
    def pairs(*args,**kwargs):
        for a,b in zip(x.split(9),y.split(9)):yield a+b,a,(7,),(7,)
    monkeypatch.setattr(op,'iter_feature_pairs',pairs)
    ticks=[0]
    def pause():ticks[0]+=1;return ticks[0]==3
    args=(graph,loader,[template],'residual_rrr','cpu')
    with pytest.raises(op.LinearFitPaused):op.fit_bank(*args,tmp_path/'resume',identity='data-and-source',workspace_bytes=10**8,should_pause=pause)
    resumed,_=op.fit_bank(*args,tmp_path/'resume',identity='data-and-source',workspace_bytes=10**8)
    fresh,_=op.fit_bank(*args,tmp_path/'fresh',identity='data-and-source',workspace_bytes=10**8)
    torch.testing.assert_close(resumed[0].apply_feature(x),fresh[0].apply_feature(x),rtol=0,atol=0)
    with pytest.raises(ValueError,match='identity'):op.fit_bank(*args,tmp_path/'resume',identity='changed',workspace_bytes=10**8)
    with pytest.raises(MemoryError):op.fit_bank(*args,tmp_path/'low',identity='x',workspace_bytes=1)
    loader.dataset.split='development'
    with pytest.raises(ValueError,match='training'):op.fit_bank(*args,tmp_path/'dev',identity='x',workspace_bytes=10**8)


def test_mhd_writeback_preserves_nodes_and_frozen_parameters():
    from look.models.native_host import build_native_host
    from look.models.observed_participant import ObservedParticipantModel
    from mhd_framework.models import create_model
    from look.methods.operator import forward_with_look
    from look.methods.joint import read_site
    torch.set_num_threads(1)
    config=dict(name='resnet50',num_classes=2,views=1)
    parents=[ObservedParticipantModel(create_model(config)) for _ in range(2)]
    graph=build_native_host(*parents,'features').eval()
    for p in graph.parameters():p.requires_grad_(False)
    nodes=[(n.id,n.name) for n in graph.nodes]
    state={k:v.clone() for k,v in graph.state_dict().items()}
    g=torch.Generator().manual_seed(3);x=torch.randn(2,3,32,32,generator=g);z=torch.randn(x.shape,generator=g)
    with torch.no_grad():
        forward_with_look(graph,x,z,counts=[1,1],stop_node='joint_input')
        shape=tuple(read_site(graph,'joint_input').shape[1:]);d=6*4*4
        s=stats(torch.randn(12,d,generator=g,dtype=torch.float64),torch.randn(12,d,generator=g,dtype=torch.float64))
        m=solve(s,2,.2)
        a=LOOKArtifact('joint_input','oct_missing','normalized_mean',8,2,shape,(6,4,4),
            torch.zeros(d),torch.ones(d),torch.zeros(d),torch.eye(d)[:2],torch.zeros(2,2),torch.zeros(2),.2,0.,0.)
        artifact=LinearVectorArtifact(a,m)
        output=forward_with_look(graph,x,z,[artifact],counts=[1,1])
        manual=artifact.apply_feature(torch.cat([x,z],dim=1))
        expected=forward_with_look(graph,manual[:,:3],manual[:,3:],counts=[1,1])
        torch.testing.assert_close(output,expected,rtol=0,atol=0)
    assert nodes==[(n.id,n.name) for n in graph.nodes]
    for k,v in state.items():torch.testing.assert_close(graph.state_dict()[k],v,rtol=0,atol=0)


def test_energy_threshold_does_not_renormalize_a_truncated_basis():
    from look.methods.rank_budget import retained_curve,rank_for_fraction,matched_ranks
    c=retained_curve([5,2],10,quantity='complete_feature_variance')
    assert c['curve'][-1]['retained']==.7
    assert rank_for_fraction(c,.9)['rank'] is None
    assert rank_for_fraction(c,.5)['rank']==1
    assert rank_for_fraction(retained_curve([0,0],0,quantity='gain'),.9)['state']=='undefined_zero_target'
    rows=matched_ranks([2,4,8],{'pca':4,'rrr':7})
    assert rows[-1]==dict(rank=8,state='infeasible',limiting_methods=['pca','rrr'])
    assert rows[1]['rank']==4
    with pytest.raises(ValueError):retained_curve([5,2],6,quantity='variance')
