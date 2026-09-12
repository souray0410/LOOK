import copy
import pytest
import torch
from mhd_framework.models import create_model
from look.models.observed_participant import ObservedParticipantModel
from look.models.native_host import build_native_host, STAGE_MAP, NativeCut, forward_host
from look.methods.operator import forward_with_look
from look.methods.joint import correction_sites, read_site, write_site


def parents(name):
    config = dict(name=name, num_classes=2, views=1)
    return [ObservedParticipantModel(create_model(config)) for _ in range(2)]


@pytest.mark.parametrize('name', ['resnet50','densenet121','swin_b'])
def test_complete_native_cut_equivalence(name):
    torch.set_num_threads(2)
    first, _ = parents(name)
    first.eval()
    x = torch.randn(3,3,224,224)
    stages = STAGE_MAP[name]
    current = x
    for i, end in enumerate(stages):
        cut = NativeCut(first.graph, 'input' if i == 0 else stages[i-1], end).eval()
        with torch.no_grad():
            current = cut(current)
            expected = first.graph.forward_until(end, x)
            if name == 'swin_b' and end != 'features':
                expected = expected.permute(0,3,1,2).contiguous()
        torch.testing.assert_close(current, expected, rtol=0, atol=0)


@pytest.mark.parametrize('name', ['resnet50','densenet121','swin_b'])
@pytest.mark.parametrize('position', ['middle','deep','features'])
def test_host_packed_eyes_mhd_backward_and_reload(position,name):
    torch.set_num_threads(2)
    a,b=parents(name)
    graph=build_native_host(a,b,position)
    graph.eval()
    x,y=torch.randn(3,3,224,224),torch.randn(3,3,224,224)
    z=forward_with_look(graph,x,y,counts=[1,2])
    assert z.shape == (2,2)
    for site in correction_sites(graph):
        before=read_site(graph,site).clone();write_site(graph,site,before)
        torch.testing.assert_close(before,read_site(graph,site),rtol=0,atol=0)
    z=forward_host(graph,x,y,[1,2],torch.tensor([0,1]))
    graph.backward(levels=graph.backward_levels)
    for prefix in ('oct_', 'cfp_', 'fuse_'):
        values=[p.grad for e in graph.edges if e.name.startswith(prefix) for _,p in e.named_edge_parameters()]
        assert any(g is not None and g.abs().sum()>0 for g in values)
    state=copy.deepcopy(graph.state_dict())
    other=build_native_host(a,b,position).eval()
    other.load_state_dict(state,strict=True)
    torch.testing.assert_close(z,forward_with_look(other,x,y,counts=[1,2]),rtol=0,atol=0)
    with pytest.raises(ValueError,match='requires counts'):
        forward_with_look(graph,x,y)
    with pytest.raises(ValueError,match='counts'):
        forward_with_look(graph,x,y,counts=[2,2])
    # Evaluation of a single-eye participant is invariant to unrelated valid eyes.
    torch.testing.assert_close(z[:1],forward_with_look(other,x[:1],y[:1],counts=[1]),rtol=1e-4,atol=1e-5)
