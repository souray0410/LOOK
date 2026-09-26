"""Modern-host cuts retain downsampling, normalization and native parent features."""
import gc
import torch
from mhd_framework.models import create_model
from look.models.observed_participant import ObservedParticipantModel
from look.models.native_host import NativeCut,STAGE_MAP,build_native_host,forward_host
from look.models.embracenet import build_embracenet_host,forward_embracenet_host,capture_embracenet_sampling,restore_embracenet_sampling


def test_convnext_all_cuts_and_embracenet_branches_preserve_native_features():
    torch.set_num_threads(2)
    torch.manual_seed(347)
    parents=[ObservedParticipantModel(create_model(dict(name='convnext_base',num_classes=2,views=1))).eval() for _ in range(2)]
    x=torch.randn(2,3,32,32);y=torch.randn_like(x)
    with torch.no_grad():
        features=[]
        for parent,data in zip(parents,(y,x)):
            current=data
            for i,end in enumerate(STAGE_MAP['convnext_base']):
                cut=NativeCut(parent.graph,'input' if i==0 else STAGE_MAP['convnext_base'][i-1],end).eval()
                current=cut(current)
                torch.testing.assert_close(current,parent.graph.forward_until(end,data),rtol=0,atol=0)
                del cut
            features.append(current.clone())
    graph=build_embracenet_host(*parents,embracement_size=16,sampling_seed=3416).eval()
    assert graph.architecture_id=='convnext_base_observed_embracenet'
    with torch.no_grad():
        state=capture_embracenet_sampling(graph)
        logits=forward_embracenet_host(graph,x,y,[1,1],torch.ones(2,2))
        for modality,expected in zip(('cfp','oct'),features):
            actual=graph.get_node_by_name(modality+'_participant_feature').feature_message.current_state
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        restore_embracenet_sampling(graph,state)
        torch.testing.assert_close(logits,forward_embracenet_host(graph,x,y,[1,1],torch.ones(2,2)),rtol=0,atol=0)
    # Real V5 backward reaches both copied encoders, docking and head.
    graph.zero_grad(set_to_none=True)
    forward_embracenet_host(graph,x,y,[1,1],torch.ones(2,2),labels=torch.tensor([0,1]))
    graph.backward(levels=graph.backward_levels)
    for prefix in ('cfp_','oct_','embracenet_','fusion_classifier_'):
        grads=[p.grad for e in graph.edges if e.name.startswith(prefix) for _,p in e.named_edge_parameters()]
        assert any(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in grads),prefix
    del graph;gc.collect()
    graph=build_native_host(*parents,'middle').eval()
    with torch.no_grad():assert forward_host(graph,x,y,[1,1]).shape==(2,2)
