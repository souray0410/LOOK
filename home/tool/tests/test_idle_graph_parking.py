"""Runtime-only memory adapter: GPU checks are opt-in while other work is active."""
import importlib.util
import json
import os
from pathlib import Path
import pytest
import torch

path=Path(__file__).resolve().parents[1]/'operations/run_unified_with_idle_graph_parking.py'
spec=importlib.util.spec_from_file_location('idle_parking',path)
parking=importlib.util.module_from_spec(spec);spec.loader.exec_module(parking)


def test_cpu_action_is_unchanged(tmp_path):
    class Graph:device=torch.device('cpu')
    assert parking.park_while(Graph(),lambda:37,tmp_path/'unused.json')==37
    assert not (tmp_path/'unused.json').exists()


@pytest.mark.skipif(os.environ.get('LOOK_TEST_PARKING_GPU')!='1',reason='Explicit isolated GPU check')
def test_gpu_roundtrip_preserves_logits_rng_and_messages(tmp_path):
    from look_core.graph import build_resnet50_mhd_graph
    from look_core.look import forward_with_look
    g=build_resnet50_mhd_graph('layer3',batch_size=2,pretrained=False,device='cuda:0').eval()
    for p in g.parameters():p.requires_grad_(False)
    o=torch.randn(2,2,3,224,224,device='cuda:0');c=torch.randn_like(o)
    with torch.no_grad():before=forward_with_look(g,o,c).cpu()
    messages={n.name:(n.feature_message.initial_state.cpu().clone(),n.feature_message.current_state.cpu().clone(),n.gradient_message.initial_state.cpu().clone(),n.gradient_message.current_state.cpu().clone()) for n in g.nodes}
    rng=torch.cuda.get_rng_state().clone()
    def action():
        assert g.device.type=='cpu'
        assert all(n.feature_message.current_state.device.type=='cpu' for n in g.nodes)
        return 7
    assert parking.park_while(g,action,tmp_path/'audit.json')==7
    for n in g.nodes:
        now=(n.feature_message.initial_state,n.feature_message.current_state,n.gradient_message.initial_state,n.gradient_message.current_state)
        assert all(torch.equal(x.cpu(),y) for x,y in zip(now,messages[n.name]))
    assert torch.equal(torch.cuda.get_rng_state(),rng)
    with torch.no_grad():assert torch.equal(forward_with_look(g,o,c).cpu(),before)
    a=json.loads((tmp_path/'audit.json').read_text())
    assert a['status']=='restored' and a['allocated_parked']<a['allocated_before']
