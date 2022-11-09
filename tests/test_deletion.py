from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from polygnn import featurize as feat
from polygnn import contrast as cst
import torch
from scipy.spatial.distance import cosine as cosine_distance
import numpy as np

"""
Basic unit test for removing node from graph
"""


# TODO @Subham: Test this method and fix any bugs associated
def test_node_deletion():
    torch.manual_seed(12)
    train_smiles = ["[*]CC[*]", "[*]CC[*]", "[*]CC(C)[*]", "[*]CC(C)[*]"]  # 2N
    # ###################################################################################
    # Get node removed from Graph
    # ###################################################################################
    bond_config = feat.BondConfig(True, False, True)
    atom_config = feat.AtomConfig(
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
    )
    train_X = [
        feat.get_minimum_graph_tensor(x, bond_config, atom_config, "monocycle")
        for x in train_smiles
    ]

    # loader = DataLoader(train_X, batch_size=1)
    # for graph in loader:
    #     graph_with_deleted_node = cst.node_deletion.delete_node(graph)
    #     assert len(graph.x) - 1 == len(graph_with_deleted_node.x)
    #     assert len(graph.edge_weight) >= len(graph_with_deleted_node.edge_weight)

    # Check if the correct node was deleted based on specified index
    first_graph = train_X[0]
    new_graph = cst.node_deletion._delete_node(Batch.from_data_list([first_graph]), torch.tensor([0]))
    assert new_graph.x.shape == torch.Size([3, 70])
    assert new_graph.edge_index.shape == (2, 4)
    assert new_graph.edge_weight.shape == (4, 5)

    assert np.array_equal(new_graph.x.numpy(), torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]]).numpy())
    assert np.array_equal(new_graph.edge_index, [[1, 2, 0, 1],[2, 1, 1, 0]])
    assert np.array_equal(new_graph.edge_weight, [[1., 0., 0., 0., 1.], [1., 0., 0., 0., 1.], [1., 0., 0., 0., 1.], [1., 0., 0., 0., 1.]])

    # Check across batch if one node and coressponding edges are deleted
    loader = DataLoader(train_X, batch_size=4)
    # loader = DataLoader(train_X, batch_size=np.random.randint(1, 1+len(train_X)))
    for graphs in loader:
        graphs_with_deleted_node = cst.node_deletion.delete_node(graphs)
        assert len(graphs.x) - 4 == len(graphs_with_deleted_node.x)
        assert len(graphs.edge_weight) - 14 == len(graphs_with_deleted_node.edge_weight)

        # for graph, graph_with_deleted_node in zip(graphs.to_data_list(), graphs_with_deleted_node.to_data_list()):
        #     assert len(graph.x) - 1 == len(graph_with_deleted_node.x)
        #     assert len(graph.edge_weight) >= len(graph_with_deleted_node.edge_weight)

        """
         TODO: Shubham, when you run this method, since the seed is fixed,
         print out the x values and edge weights and assert those are equal a transformed graph
        """


test_node_deletion()