import torch
import torch.nn as nn
import pytest

from MHD_Project.MHD_Framework_V4 import MHD_Edge, MHD_Graph, MHD_Node, MHD_Topo
from MHD_Project.MHD_Utils_V4 import prune_isolated_graph


def test_pruning_removes_registered_modules_and_rehashes_sets():
    nodes = {
        MHD_Node(0, "input", MHD_Node.Message(torch.zeros(1, 2))),
        MHD_Node(1, "output", MHD_Node.Message(torch.zeros(1, 2))),
        MHD_Node(2, "isolated_node", MHD_Node.Message(torch.zeros(1, 2))),
    }
    edges = {
        MHD_Edge(0, "active", [MHD_Edge.Operation(nn.Linear(2, 2))]),
        MHD_Edge(1, "isolated", [MHD_Edge.Operation(nn.Linear(2, 2))]),
    }
    role = torch.tensor([[-1, 1, 0], [0, 0, 0]], dtype=torch.int8)
    sort = torch.tensor([[0, 1, 0], [0, 0, 0]], dtype=torch.int8)
    graph = MHD_Graph(
        nodes, edges, {MHD_Topo([role, -role], [sort, sort.clone()])},
        device=torch.device("cpu"),
    )
    prune_isolated_graph(graph, verbose=False)

    assert [node.id for node in sorted(graph.nodes, key=lambda item: item.id)] == [0, 1]
    assert [edge.id for edge in sorted(graph.edges, key=lambda item: item.id)] == [0]
    assert graph.get_edge_by_name("isolated") is None
    assert all("isolated" not in key for key in graph.state_dict())
    assert len(graph.edge_module_map) == 1
    assert graph.topo.role_matrices[0].shape == (1, 2)
    assert graph.topo.role_matrices[1].shape == (1, 2)
    assert torch.equal(graph.topo.role_matrices[1], -graph.topo.role_matrices[0])
    assert all(node in graph.nodes for node in list(graph.nodes))
    assert all(edge in graph.edges for edge in list(graph.edges))


def test_pruning_rejects_graph_after_first_forward():
    nodes = {
        MHD_Node(0, "input", MHD_Node.Message(torch.ones(1, 2))),
        MHD_Node(1, "output", MHD_Node.Message(torch.zeros(1, 2))),
    }
    edges = {MHD_Edge(0, "active", [MHD_Edge.Operation(nn.Linear(2, 2))])}
    role = torch.tensor([[-1, 1]], dtype=torch.int8)
    sort = torch.tensor([[0, 1]], dtype=torch.int8)
    graph = MHD_Graph(nodes, edges, {MHD_Topo([role], [sort])}, device=torch.device("cpu"))
    graph.forward(levels=[0])
    with pytest.raises(RuntimeError, match="graph.forward"):
        prune_isolated_graph(graph, verbose=False)
