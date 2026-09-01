import torch
import torch.nn as nn

from MHD_Project.MHD_Framework_V3 import MHD_Edge, MHD_Graph, MHD_Node, MHD_Topo
from MHD_Project.MHD_Utils_V3 import prune_isolated_graph


def test_pruning_removes_registered_modules_and_rehashes_sets():
    nodes = {
        MHD_Node(0, "input", torch.zeros(1, 2)),
        MHD_Node(1, "output", torch.zeros(1, 2)),
        MHD_Node(2, "isolated_node", torch.zeros(1, 2)),
    }
    edges = {
        MHD_Edge(0, "active", [nn.Linear(2, 2)]),
        MHD_Edge(1, "isolated", [nn.Linear(2, 2)]),
    }
    role = torch.tensor([[-1, 1, 0], [0, 0, 0]], dtype=torch.int8)
    sort = torch.tensor([[0, 1, 0], [0, 0, 0]], dtype=torch.int8)
    graph = MHD_Graph(nodes, edges, {MHD_Topo([role], [sort])}, device=torch.device("cpu"))
    prune_isolated_graph(graph, verbose=False)

    assert [node.id for node in sorted(graph.nodes, key=lambda item: item.id)] == [0, 1]
    assert [edge.id for edge in sorted(graph.edges, key=lambda item: item.id)] == [0]
    assert graph.get_edge_by_name("isolated") is None
    assert all("isolated" not in key for key in graph.state_dict())
    assert len(graph.edge_module_map) == 1
    assert all(node in graph.nodes for node in list(graph.nodes))
    assert all(edge in graph.edges for edge in list(graph.edges))
