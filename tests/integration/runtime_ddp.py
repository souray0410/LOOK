"""Two-rank trainer gradients/updates versus a direct global-batch reference."""
import torch
from mhd_framework.utils import (initialize_mhd_distributed, destroy_mhd_distributed,
    MHD_Trainer, MHD_Monitor, MHD_ParallelConfig)


from mhd_framework.core import MHD_Edge, MHD_Graph, MHD_Node, MHD_Topo


class LearnableMeanSquare(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(4))

    def forward(self, value):
        return (value.square() * self.scale).mean(dim=-1)


def build_graph(device):
    # This fixture depends on public V5 APIs, not the framework test directory.
    names = ['input', 'hidden', 'per_sample_loss', 'loss']
    states = [torch.zeros(2, 4), torch.zeros(2, 4), torch.zeros(2), torch.zeros(())]
    nodes = {MHD_Node(i, name, MHD_Node.Message(state.to(device)), memory=False)
             for i, (name, state) in enumerate(zip(names, states))}
    modules = [torch.nn.Linear(4, 4, bias=False), LearnableMeanSquare(), lambda value: value.mean()]
    edges = {MHD_Edge(i, name, [MHD_Edge.Operation(module)])
             for i, (name, module) in enumerate(zip(['projection', 'loss_reduce', 'mean'], modules))}
    roles, orders = [], []
    for i in range(3):
        role = torch.zeros(3, 4, device=device, dtype=torch.int64)
        order = torch.zeros_like(role)
        role[i, i], role[i, i + 1], order[i, i + 1] = -1, 1, 1
        roles.append(role)
        orders.append(order)
    topology = MHD_Topo(roles + [-role for role in reversed(roles)], orders + list(reversed(orders)))
    return MHD_Graph(nodes, edges, {topology}, device=device)


def main():
    context = initialize_mhd_distributed()
    assert context.world_size == 2
    torch.manual_seed(182)
    graph = build_graph(context.device)
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
