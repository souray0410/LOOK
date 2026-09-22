import copy
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from mhd_framework.models import create_model
from look.models.native_host import build_native_host,forward_host
from look.models.graph import optimizer_parameter_groups
from look.methods.joint import correction_sites,read_site,write_site


def build():
    p=[SimpleNamespace(graph=create_model(dict(name='resnet18',num_classes=2,views=1))) for _ in range(2)]
    return build_native_host(*p,'deep',mmtm=dict(stage='stage3',ratio=4,gate_scale=1.))


def direct(graph,oct,cfp):
    values={graph.get_node_by_name(k).id:v for k,v in [('oct_input',oct),('cfp_input',cfp),('eye_counts',torch.tensor([1,2]))]}
    for level in graph.model_levels:
        roles=graph.topo.role_matrices[level].to_dense();order=graph.topo.sort_matrices[level].to_dense()
        for eid in range(roles.shape[0]):
            heads=torch.where(roles[eid]<0)[0].tolist();tails=torch.where(roles[eid]>0)[0].tolist()
            if not tails:continue
            heads.sort(key=lambda n:int(order[eid,n]))
            values[tails[0]]=graph.get_edge_by_id(eid).edge_operations[0].function(*(values[i] for i in heads))
    return values[graph.get_node_by_name('fusion_logits').id]


def test_mmtm_actual_host_matches_direct_gradient_update_reload_and_sites():
    torch.set_num_threads(2);torch.manual_seed(3416)
    g=build().eval();r=copy.deepcopy(g)
    x,y=torch.randn(3,3,64,64),torch.randn(3,3,64,64);labels=torch.tensor([0,1])
    actual=forward_host(g,x,y,[1,2],labels);expected=direct(r,x,y)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    g.backward(levels=g.backward_levels);nn.functional.cross_entropy(expected,labels).backward()
    pairs=list(zip(g.parameters(),r.parameters()))
    assert pairs
    for a,b in pairs:
        assert (a.grad is None)==(b.grad is None)
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=2e-6)
    groups=optimizer_parameter_groups(g,1e-4,1e-3)
    new={id(p) for p in groups[1]['params']}
    for e in g.edges:
        if e.name.startswith('fuse_mmtm_'):
            assert all(id(p) in new for _,p in e.named_edge_parameters())
    assert len({id(p) for z in groups for p in z['params']})==sum(len(z['params']) for z in groups)
    opt=torch.optim.AdamW(g.parameters());opt.step();opt.zero_grad()
    state=copy.deepcopy(g.state_dict());resumed=build().eval();resumed.load_state_dict(state,strict=True)
    assert [(n.id,n.name) for n in sorted(g.nodes,key=lambda n:n.id)]==[(n.id,n.name) for n in sorted(resumed.nodes,key=lambda n:n.id)]
    torch.testing.assert_close(forward_host(g,x,y,[1,2]),forward_host(resumed,x,y,[1,2]),rtol=0,atol=0)
    resumed_opt=torch.optim.AdamW(resumed.parameters());resumed_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    # Full optimizer reload must give the same next training update, including BN.
    for graph,optimizer in [(g,opt),(resumed,resumed_opt)]:
        graph.train();forward_host(graph,x,y,[1,2],labels);graph.backward(levels=graph.backward_levels)
        optimizer.step();optimizer.zero_grad()
    for k,v in g.state_dict().items():
        torch.testing.assert_close(v,resumed.state_dict()[k],rtol=0,atol=0)
    for site in correction_sites(g):
        before=read_site(g,site).clone();write_site(g,site,before)
        torch.testing.assert_close(before,read_site(g,site),rtol=0,atol=0)


def test_default_host_unchanged_and_late_mmtm_rejected():
    torch.set_num_threads(2)
    p=[SimpleNamespace(graph=create_model(dict(name='resnet18',num_classes=2,views=1))) for _ in range(2)]
    a=build_native_host(*p,'deep');b=build_native_host(*p,'deep',mmtm=None)
    assert a.native_host_provenance==b.native_host_provenance
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
    with pytest.raises(ValueError,match='before fusion'):
        build_native_host(*p,'middle',mmtm=dict(stage='stage4',ratio=4,gate_scale=1.))
