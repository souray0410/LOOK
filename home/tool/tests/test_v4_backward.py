import torch
import torch.nn as nn

from MHD_Project.MHD_Framework_V4 import MHD_Edge, MHD_Graph, MHD_Node, MHD_Topo
from look_core.distributed import module_state_sha256


def _node(node_id: int, name: str, state: torch.Tensor) -> MHD_Node:
    return MHD_Node(
        node_id,
        name,
        MHD_Node.Message(state),
        aggregation="replace",
    )


def _tiny_graph() -> MHD_Graph:
    nodes = {
        _node(0, "input", torch.zeros(3, 4)),
        _node(1, "prediction", torch.zeros(3, 2)),
        _node(2, "target", torch.zeros(3, 2)),
        _node(3, "loss", torch.zeros(())),
    }
    edges = {
        MHD_Edge(0, "linear", [MHD_Edge.Operation(nn.Linear(4, 2))]),
        MHD_Edge(1, "mse", [MHD_Edge.Operation(nn.MSELoss())]),
    }
    role_0 = torch.tensor([[-1, 1, 0, 0], [0, 0, 0, 0]], dtype=torch.int8)
    sort_0 = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 0]], dtype=torch.int8)
    role_1 = torch.tensor([[0, 0, 0, 0], [0, -1, -1, 1]], dtype=torch.int8)
    sort_1 = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 2]], dtype=torch.int8)
    return MHD_Graph(
        nodes,
        edges,
        {MHD_Topo(
            [role_0, role_1, -role_0, -role_1],
            [sort_0, sort_1, sort_0.clone(), sort_1.clone()],
        )},
        device=torch.device("cpu"),
    )


def _forward(graph: MHD_Graph, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    for node in graph.nodes:
        node.reset()
    graph.get_node_by_name("input").feature_message.current_state = inputs
    graph.get_node_by_name("target").feature_message.current_state = targets
    graph.forward(levels=[0, 1])
    return graph.get_node_by_name("loss").feature_message.current_state


def test_graph_backward_matches_native_autograd_and_optimizer_step():
    torch.manual_seed(17)
    graph_backward = _tiny_graph()
    native_backward = _tiny_graph()
    native_backward.load_state_dict(graph_backward.state_dict())
    inputs = torch.randn(3, 4)
    targets = torch.randn(3, 2)
    graph_loss = _forward(graph_backward, inputs.clone(), targets.clone())
    native_loss = _forward(native_backward, inputs.clone(), targets.clone())

    graph_backward.backward(levels=[3, 2])
    native_loss.backward()

    graph_parameters = dict(graph_backward.named_parameters())
    native_parameters = dict(native_backward.named_parameters())
    assert graph_parameters.keys() == native_parameters.keys()
    for name in graph_parameters:
        assert torch.allclose(
            graph_parameters[name].grad,
            native_parameters[name].grad,
            atol=1e-7,
            rtol=1e-6,
        )

    prediction_gradient = graph_backward.get_node_by_name(
        "prediction"
    ).gradient_message.current_state
    assert prediction_gradient.shape == (3, 2)
    assert torch.isfinite(prediction_gradient).all()
    assert graph_backward.get_node_by_name("loss").gradient_message.current_state.item() == 1.0

    graph_optimizer = torch.optim.SGD(graph_backward.parameters(), lr=0.05)
    native_optimizer = torch.optim.SGD(native_backward.parameters(), lr=0.05)
    graph_optimizer.step()
    native_optimizer.step()
    for name in graph_parameters:
        assert torch.allclose(
            graph_parameters[name],
            native_parameters[name],
            atol=1e-7,
            rtol=1e-6,
        )


def test_module_state_hash_covers_scalar_buffers_deterministically():
    first = nn.BatchNorm1d(3)
    second = nn.BatchNorm1d(3)
    second.load_state_dict(first.state_dict())
    assert module_state_sha256(first) == module_state_sha256(second)
    second.num_batches_tracked.add_(1)
    assert module_state_sha256(first) != module_state_sha256(second)


def test_mermaid_draws_explicit_global_levels_in_order():
    graph = _tiny_graph()
    mermaid = graph.generate_mermaid(levels=[0, 1, 3, 2])
    assert "#0:L0:S0" in mermaid
    assert "#2:L3:S0" in mermaid
    assert " -->|" in mermaid
