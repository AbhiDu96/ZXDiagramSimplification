"""Unit tests for the observation-graph helper in utils.py."""
import torch
import torch_geometric.data as data

from utils import expand_graph


def test_expand_graph_shapes_and_features():
    # 3 nodes, 2 directed edges with scalar edge features
    x = torch.tensor([[-1.0], [0.0], [1.0]])
    edge_index = torch.tensor([[0, 1], [1, 2]])
    edge_attr = torch.tensor([10.0, 20.0])
    d = data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    exp = expand_graph(d)

    n_nodes, n_edges = 3, 2
    # each edge becomes a virtual node appended after the real nodes
    assert exp.x.shape[0] == n_nodes + n_edges
    # each original edge becomes two edges (node->virtual, virtual->node)
    assert exp.edge_index.shape == (2, 2 * n_edges)
    # the appended virtual nodes carry the edge features
    assert torch.allclose(exp.x[n_nodes:], edge_attr[:, None])
