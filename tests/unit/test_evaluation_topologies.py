"""Exercise the post-training evaluation entrypoint, including unimodal controls."""
import numpy as np
import pytest
import torch
from look_core.evaluate import evaluate_missing
from look_core.graph import FUSION_POSITIONS, build_resnet50_mhd_graph, reset_and_forward
from look_core.look import forward_with_look


@pytest.mark.parametrize('position', [*FUSION_POSITIONS, 'oct_only', 'cfp_only'])
def test_complete_evaluation_matches_original_forward_with_tail(position):
    torch.manual_seed(21)
    graph=build_resnet50_mhd_graph(position,batch_size=2,image_size=224,device='cpu',pretrained=False).eval()
    oct_x=torch.randn(3,2,3,224,224)
    cfp_x=torch.randn_like(oct_x)
    batches=[dict(oct=oct_x[i:i+2],cfp=cfp_x[i:i+2],label=torch.tensor([0,1,0])[i:i+2],
                  participant_id=['p0','p1','p2'][i:i+2]) for i in range(0,3,2)]
    with torch.no_grad():
        expected=torch.cat([reset_and_forward(graph,b['oct'],b['cfp']).clone() for b in batches]).numpy()
        actual=evaluate_missing(graph,batches,torch.device('cpu'),fixed_pattern='complete')
    assert np.array_equal(actual['logits'],expected)
    assert actual['participant_ids'].tolist()==['p0','p1','p2']
    assert actual['patterns'].tolist()==['complete']*3
    if position in ('oct_only','cfp_only'):
        # Supplying an unused modality must not affect this unimodal reference.
        with torch.no_grad():
            other_oct=oct_x if position=='oct_only' else torch.randn_like(oct_x)
            other_cfp=cfp_x if position=='cfp_only' else torch.randn_like(cfp_x)
            again=forward_with_look(graph,other_oct[:2],other_cfp[:2])
        assert np.array_equal(again.numpy(),expected[:2])
