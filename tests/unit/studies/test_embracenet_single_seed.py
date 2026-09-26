import copy
import shutil

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from mhd_framework.models import create_model

from look.data.observed_pair import collate_observed
from look.evaluation.embracenet import (
    analytical_complete_logits,
    complete_participant_moments,
    evaluate_complete_analytical,
    evaluate_single_missing,
)
from look.methods.embracenet_family import (
    EmbraceNetFamilyStatistics,
    _update_with_variance,
    fit_stochastic_embraced_pca,
    fit_embracenet_family_trajectory,
    prepare_embracenet_pca_bank,
)
from look.methods.independent_greedy import SelectionPaused
from look.methods.linear_vector import ResidualMoments
from look.models.embracenet import build_embracenet_host
from look.models.observed_participant import ObservedParticipantModel
from look.studies import embracenet_delivery as delivery
from look.studies.embracenet_delivery import _simultaneous_contrasts
from look.training.embracenet_host import _availability, _new_mask_generator, _state_counts, train_embracenet_host
from look.training.observed_host import DEFAULTS
from look.runtime.host_checkpoint import capture_rng, restore_rng


class TinyPair(Dataset):
    def __init__(self, split, size, seed=17):
        self.split=split;self.augment=split=="train";self.seed=seed;self.epoch=0
        self.participant_ids=[f"{split}-{i}" for i in range(size)];self.counts=[1]*size
        g=torch.Generator().manual_seed(seed+(0 if split=="train" else 1000))
        self.oct=[torch.randn(1,3,224,224,generator=g) for _ in range(size)]
        self.cfp=[torch.randn(1,3,224,224,generator=g) for _ in range(size)]
        self.labels=[i%2 for i in range(size)]
    def set_epoch(self,epoch):self.epoch=epoch
    def __len__(self):return len(self.participant_ids)
    def __getitem__(self,i):
        return dict(oct=self.oct[i].clone(),cfp=self.cfp[i].clone(),label=self.labels[i],participant_id=self.participant_ids[i])


def _parents():
    config=dict(name="resnet18",num_classes=2,views=1)
    return [ObservedParticipantModel(create_model(config)) for _ in range(2)]


def _graph(seed=3416,embracement=16):
    return build_embracenet_host(*_parents(),embracement_size=embracement,sampling_seed=seed)


def _tensor_equal(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):
        return set(a)==set(b) and all(_tensor_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return type(a) is type(b) and len(a)==len(b) and all(_tensor_equal(x,y) for x,y in zip(a,b))
    return a==b


def _checkpoint_equal(a,b,path="root"):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and torch.equal(a,b),path
        return
    if isinstance(a,np.ndarray):
        np.testing.assert_array_equal(a,b);return
    if isinstance(a,dict):
        assert set(a)==set(b),path
        for key in a:
            if path.endswith(".progress") and key=="seconds":
                continue
            _checkpoint_equal(a[key],b[key],path+"."+str(key))
        return
    if isinstance(a,(list,tuple)):
        assert type(a) is type(b) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):_checkpoint_equal(x,y,path+f"[{i}]")
        return
    assert a==b,path


def test_training_mask_rng_is_private_three_state_and_resume_matches_uninterrupted(tmp_path):
    generator=_new_mask_generator(3416)
    states=torch.randint(3,(6000,),generator=generator)
    counts=_state_counts(states)
    assert sum(counts.values())==6000
    assert all(1800 < value < 2200 for value in counts.values())
    availability=_availability(torch.tensor([0,1,2]),torch.device("cpu"))
    torch.testing.assert_close(availability,torch.tensor([[1.,1.],[0.,1.],[1.,0.]]),rtol=0,atol=0)

    train=TinyPair("train",4);dev=TinyPair("development",2)
    identity="tiny-embracenet-resume"
    torch.manual_seed(909);np.random.seed(909)
    template=_graph();initial_model=copy.deepcopy(template.state_dict())
    initial_rng=capture_rng()
    from look.models.embracenet import capture_embracenet_sampling,restore_embracenet_sampling
    initial_sampling=capture_embracenet_sampling(template)

    def fresh():
        g=_graph();g.load_state_dict(initial_model,strict=True)
        restore_embracenet_sampling(g,initial_sampling);restore_rng(initial_rng)
        return g

    continuous=fresh()
    result=train_embracenet_host(
        continuous,train,dev,dict(DEFAULTS),3416,tmp_path/"continuous",identity,torch.device("cpu"),
        preflight_target_updates=2,
    )
    assert result["state"]=="paused" and result["total_updates"]==2
    continuous_dev=evaluate_complete_analytical(
        continuous,DataLoader(dev,batch_size=2,collate_fn=collate_observed),torch.device("cpu")
    )

    base=fresh()
    result=train_embracenet_host(
        base,train,dev,dict(DEFAULTS),3416,tmp_path/"base",identity,torch.device("cpu"),
        preflight_target_updates=1,
    )
    assert result["state"]=="paused" and result["total_updates"]==1
    resumed=[]
    for name in ("resume_a","resume_b"):
        shutil.copytree(tmp_path/"base",tmp_path/name)
        g=_graph()
        result=train_embracenet_host(
            g,train,dev,dict(DEFAULTS),3416,tmp_path/name,identity,torch.device("cpu"),
            preflight_target_updates=2,
        )
        assert result["state"]=="paused" and result["total_updates"]==2
        dev_result=evaluate_complete_analytical(
            g,DataLoader(dev,batch_size=2,collate_fn=collate_observed),torch.device("cpu")
        )
        resumed.append((torch.load(tmp_path/name/"last.pt",map_location="cpu",weights_only=False),dev_result["logits"]))

    uninterrupted=torch.load(tmp_path/"continuous/last.pt",map_location="cpu",weights_only=False)
    _checkpoint_equal(uninterrupted,resumed[0][0])
    _checkpoint_equal(resumed[0][0],resumed[1][0])
    np.testing.assert_array_equal(continuous_dev["logits"],resumed[0][1])
    np.testing.assert_array_equal(resumed[0][1],resumed[1][1])

def test_analytical_complete_logits_do_not_consume_sampler_and_missing_is_deterministic():
    torch.set_num_threads(2)
    graph=_graph(embracement=12).eval()
    oct_tensor=torch.randn(2,3,224,224);cfp_tensor=torch.randn(2,3,224,224);counts=[1,1]
    from look.models.embracenet import capture_embracenet_sampling
    before=capture_embracenet_sampling(graph)
    logits,moments=analytical_complete_logits(graph,oct_tensor,cfp_tensor,counts)
    after=capture_embracenet_sampling(graph)
    assert _tensor_equal(before,after)
    assert logits.shape==(2,2) and torch.any(moments["variance_diagonal"]>0)

    data=TinyPair("development",2)
    loader=DataLoader(data,batch_size=2,collate_fn=collate_observed)
    first=evaluate_single_missing(graph,loader,torch.device("cpu"),"oct_missing")
    graph2=_graph(seed=999,embracement=12).eval();graph2.load_state_dict(copy.deepcopy(graph.state_dict()))
    second=evaluate_single_missing(graph2,loader,torch.device("cpu"),"oct_missing")
    np.testing.assert_array_equal(first["logits"],second["logits"])


def test_stochastic_complete_pca_std_contains_conditional_variance():
    torch.set_num_threads(2)
    graph=_graph(embracement=8).eval()
    for p in graph.parameters():p.requires_grad_(False)
    data=TinyPair("train",5);data.augment=False
    loader=DataLoader(data,batch_size=5,collate_fn=collate_observed)
    basis=fit_stochastic_embraced_pca(graph,loader,3,torch.device("cpu"),"tiny")
    batch=next(iter(loader))
    moments=complete_participant_moments(graph,batch["oct"],batch["cfp"],batch["counts"])
    mean=moments["mean"].double();var=moments["variance_diagonal"].double()
    expected_mean=mean.mean(0)
    expected_var=(mean.square()+var).mean(0)-expected_mean.square()
    torch.testing.assert_close(basis.mean.double(),expected_mean,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(basis.std.double().square(),expected_var.clamp_min(1e-12),rtol=1e-5,atol=1e-6)
    mean_only=mean.var(0,unbiased=False)
    assert torch.any(expected_var > mean_only + 1e-8)


def test_registered_bootstrap_returns_four_contrasts_for_each_metric():
    labels=np.asarray([0,1]*10)
    base=np.column_stack([1-labels,labels]).astype(float)*2-1
    logits={
        ("host","oct_missing"):base,
        ("pca_free_mean","oct_missing"):base+np.column_stack([labels,-labels])*.1,
        ("residual_rrr","oct_missing"):base+np.column_stack([labels,-labels])*.05,
        ("host","cfp_missing"):base*.9,
        ("pca_free_mean","cfp_missing"):base*.95,
        ("residual_rrr","cfp_missing"):base*.92,
    }
    definitions=[
        {"arm":arm,"pattern":pattern,"method_key":(arm,pattern),"reference_key":("host",pattern)}
        for pattern in ("oct_missing","cfp_missing") for arm in ("pca_free_mean","residual_rrr")
    ]
    result=_simultaneous_contrasts(labels,logits,definitions,64,3416)
    assert set(result)=={"macro_f1","macro_auroc_ovr","negative_log_likelihood","multiclass_brier"}
    assert all(len(value["rows"])==4 and value["iterations"]==64 for value in result.values())


@pytest.mark.parametrize("arm", ["shared_pca_ridge", "pca_free_mean", "rrr_shared_intercept", "residual_rrr"])
@pytest.mark.parametrize("search", ["positive_forward_tree", "best_forward"])
def test_small_real_mhd_positive_tree_uses_analytic_reference_and_replays(tmp_path, arm, search):
    torch.set_num_threads(2)
    graph=_graph(embracement=8).eval()
    for parameter in graph.parameters(): parameter.requires_grad_(False)
    train=TinyPair("train",6);train.augment=False
    dev=TinyPair("development",4)
    train_loader=DataLoader(train,batch_size=3,collate_fn=collate_observed,shuffle=False)
    dev_loader=DataLoader(dev,batch_size=2,collate_fn=collate_observed,shuffle=False)
    sites=["joint_participant_feature","embraced_feature"]
    bank=prepare_embracenet_pca_bank(
        graph,train_loader,sites,16,2,torch.device("cpu"),tmp_path/"pca",
        {"case":"tiny","arm":arm},lambda:False,
    )
    artifacts,result=fit_embracenet_family_trajectory(
        graph,train_loader,dev_loader,arm=arm,pattern="oct_missing",sites=sites,
        factor=16,candidates=[{"rank":2,"ridge_lambda":None}],pca_bank=bank,
        identity="tiny-family",output=tmp_path/arm,device=torch.device("cpu"),search=search,
        workspace_bytes=256*1024**2,should_pause=lambda:False,
    )
    assert result["mode"]==search
    assert result["test_access"] is False
    assert (tmp_path/arm/"bank.pt").exists()
    replay=evaluate_single_missing(graph,dev_loader,torch.device("cpu"),"oct_missing",artifacts)
    assert replay["logits"].shape==(4,2)


def test_family_statistics_interruption_resume_matches_uninterrupted(tmp_path):
    torch.set_num_threads(2)
    graph=_graph(embracement=8).eval()
    for parameter in graph.parameters(): parameter.requires_grad_(False)
    train=TinyPair("train",6);train.augment=False
    loader=DataLoader(train,batch_size=2,collate_fn=collate_observed,shuffle=False)
    site="embraced_feature"
    bank=prepare_embracenet_pca_bank(
        graph,loader,[site],16,2,torch.device("cpu"),tmp_path/"pca",
        {"case":"resume-stat"},lambda:False,
    )
    basis=next(iter(bank.values()))
    class StopAfterOne:
        def __init__(self):self.calls=0
        def __call__(self):
            self.calls+=1
            return self.calls>=2
    first=EmbraceNetFamilyStatistics(
        graph,loader,{site:basis},torch.device("cpu"),tmp_path/"resume",
        {"case":"resume-stat"},256*1024**2,2,StopAfterOne(),projected_ranks=(2,),
    )
    with pytest.raises(SelectionPaused):
        first.statistics("oct_missing",(),[site])
    resumed=EmbraceNetFamilyStatistics(
        graph,loader,{site:basis},torch.device("cpu"),tmp_path/"resume",
        {"case":"resume-stat"},256*1024**2,2,lambda:False,projected_ranks=(2,),
    )
    rstats=resumed.statistics("oct_missing",(),[site])
    continuous=EmbraceNetFamilyStatistics(
        graph,loader,{site:basis},torch.device("cpu"),tmp_path/"continuous",
        {"case":"resume-stat"},256*1024**2,2,lambda:False,projected_ranks=(2,),
    )
    cstats=continuous.statistics("oct_missing",(),[site])
    _checkpoint_equal(vars(rstats[site]),vars(cstats[site]))
    _checkpoint_equal(vars(resumed.projected[site][2]),vars(continuous.projected[site][2]))


def test_full_second_crossmoments_and_projected_residual_stats_match_independent_enumeration():
    # Two participants, two independent categorical coordinates. Enumeration is
    # deliberately independent of EmbraceNet's analytical_moments implementation.
    x=torch.tensor([[0.2,-0.4],[1.1,0.3]],dtype=torch.float64)
    modal_a=torch.tensor([[1.0,-2.0],[-1.0,2.5]],dtype=torch.float64)
    modal_b=torch.tensor([[3.0,4.0],[2.0,-0.5]],dtype=torch.float64)
    probs=torch.tensor([[0.25,0.75],[0.6,0.4]],dtype=torch.float64)
    means=[];variances=[];full_seconds=[];residual_outcomes=[]
    for i in range(2):
        mu=torch.zeros(2,dtype=torch.float64);second=torch.zeros((2,2),dtype=torch.float64)
        outcomes=[]
        for j0 in (0,1):
            for j1 in (0,1):
                weight=probs[i,j0]*probs[i,j1]
                z=torch.tensor([
                    modal_a[i,0] if j0==0 else modal_b[i,0],
                    modal_a[i,1] if j1==0 else modal_b[i,1],
                ],dtype=torch.float64)
                mu+=weight*z;second+=weight*torch.outer(z,z);outcomes.append((float(weight),z))
        means.append(mu);full_seconds.append(second)
        variances.append(torch.diag(second-torch.outer(mu,mu)))
        residual_outcomes.append(outcomes)
    means=torch.stack(means);variances=torch.stack(variances)
    for i in range(2):
        torch.testing.assert_close(
            full_seconds[i],torch.outer(means[i],means[i])+torch.diag(variances[i]),rtol=0,atol=1e-12
        )
        assert abs(float((full_seconds[i][0,1]-means[i,0]*means[i,1])))<1e-12

    residual_mean=means-x
    stats=ResidualMoments.empty(2)
    _update_with_variance(stats,x,residual_mean,variances)
    mx=x.mean(0);my=residual_mean.mean(0)
    expected_cxx=sum(torch.outer(row-mx,row-mx) for row in x)
    expected_cxy=sum(torch.outer(x[i]-mx,residual_mean[i]-my) for i in range(2))
    expected_syy=torch.zeros((),dtype=torch.float64)
    for i,outcomes in enumerate(residual_outcomes):
        for weight,z in outcomes:
            y=z-x[i]
            expected_syy+=weight*(y-my).square().sum()
    torch.testing.assert_close(stats.mean_x,mx,rtol=0,atol=1e-12)
    torch.testing.assert_close(stats.mean_y,my,rtol=0,atol=1e-12)
    torch.testing.assert_close(stats.cxx,expected_cxx,rtol=0,atol=1e-12)
    torch.testing.assert_close(stats.cxy,expected_cxy,rtol=0,atol=1e-12)
    torch.testing.assert_close(stats.syy,expected_syy,rtol=0,atol=1e-12)

    q=torch.tensor([[2**-0.5,2**-0.5]],dtype=torch.float64)
    projected=ResidualMoments.empty(1)
    projected_variance=(variances*q.square()).sum(1,keepdim=True)
    _update_with_variance(projected,x@q.T,residual_mean@q.T,projected_variance)
    projected_expected=torch.zeros((),dtype=torch.float64)
    projected_mean=(residual_mean@q.T).mean(0)
    for i,outcomes in enumerate(residual_outcomes):
        for weight,z in outcomes:
            y=(z-x[i])@q.T
            projected_expected+=weight*(y-projected_mean).square().sum()
    torch.testing.assert_close(projected.syy,projected_expected,rtol=0,atol=1e-12)


def test_work_signal_callback_reaches_stage_and_releases_lock(tmp_path,monkeypatch):
    from look.runtime import device_budget
    handlers={}
    monkeypatch.setattr(device_budget,"configure",device_budget.validate)
    monkeypatch.setattr(delivery,"validate",lambda spec:None)
    monkeypatch.setattr(delivery.signal,"signal",lambda sig,handler:handlers.__setitem__(sig,handler))
    monkeypatch.setattr(delivery,"_resource_guard",lambda spec,root,stop_requested=lambda:False:bool(stop_requested()))
    monkeypatch.setattr(delivery.torch.cuda,"get_device_properties",lambda index:type("P",(),{"name":"fake","uuid":"GPU-11111111-1111-1111-1111-111111111111"})())
    monkeypatch.setattr(delivery.torch.cuda,"mem_get_info",lambda:(100,200))
    monkeypatch.setattr(delivery.subprocess,"check_output",lambda *args,**kwargs:"GPU-11111111-1111-1111-1111-111111111111\n")
    observed={}
    def fake_host(spec,root,check):
        handlers[delivery.signal.SIGUSR1](delivery.signal.SIGUSR1,None)
        observed["check"]=check()
        raise delivery.StagePaused()
    monkeypatch.setattr(delivery,"stage_host",fake_host)
    spec={
        "output":str(tmp_path/"run"),"lock_root":str(tmp_path/"locks"),"seed":3416,
        "gpu_reserve_bytes":0,"gpu_budget_bytes":1,
    }
    assert delivery.work(spec,"host")=="paused"
    assert observed["check"] is True
    status=delivery.read(tmp_path/"run/host_status.json")
    assert status["state"]=="paused" and status["resume"]=="same_stage_same_identity"
    import fcntl
    lock_path=next((tmp_path/"locks").glob("*.lock"))
    with lock_path.open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(handle,fcntl.LOCK_UN)
