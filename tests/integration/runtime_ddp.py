"""Two-rank trainer gradients/updates versus a direct global-batch reference."""
import importlib.util
from pathlib import Path
import torch
from mhd_framework.utils import (initialize_mhd_distributed, destroy_mhd_distributed,
    MHD_Trainer, MHD_Monitor, MHD_ParallelConfig)


def main():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('mhd_ddp_fixture',root/'third_party/MHD_Framework/tests/integration/distributed.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    context = initialize_mhd_distributed()
    assert context.world_size == 2
    torch.manual_seed(182)
    graph = fixture.build_graph(context.device)
    weight = graph.get_edge_by_name('projection').edge_operations[0].function.weight
    scale = graph.get_edge_by_name('loss_reduce').edge_operations[0].function.scale
    reference = [torch.nn.Parameter(p.detach().clone()) for p in (weight,scale)]
    ref_optimizer = torch.optim.SGD(reference,lr=.01)
    optimizer = torch.optim.SGD(graph.parameters(),lr=.01)
    trainer = MHD_Trainer(graph,optimizer,MHD_Monitor(['loss']),[0,1,2],[3,4,5],
        criteria=lambda g:g.get_node_by_name('loss').feature_message.current_state,
        save_dir='/tmp/mhd_ddp_trainer_'+__import__('os').environ.get('MASTER_PORT','test'),
        input_nodes=['input'],output_nodes=['loss'],distributed_context=context,
        parallel=MHD_ParallelConfig(data_parallel='ddp'))
    inputs = torch.arange(16,device=context.device,dtype=torch.float32).reshape(4,4)/16
    checks=[]
    def check_gradients(opt,args,kwargs):
        for actual,expected in zip((weight,scale),reference):
            torch.testing.assert_close(actual.grad,expected.grad,rtol=1e-5,atol=1e-7)
        checks.append(True)
    handle = optimizer.register_step_pre_hook(check_gradients)
    for _ in range(4):
        ref_optimizer.zero_grad()
        loss=((inputs@reference[0].T).square()*reference[1]).mean()
        loss.backward()
        trainer.train_step({'input':inputs[context.rank::2]})
        ref_optimizer.step()
        for actual,expected in zip((weight,scale),reference):
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-7)
    assert len(checks)==4
    handle.remove()
    if context.is_main:print('MHD_TRAINER_DDP_GLOBAL_BATCH_REFERENCE_OK four_updates',flush=True)
    destroy_mhd_distributed()

if __name__=='__main__':main()
