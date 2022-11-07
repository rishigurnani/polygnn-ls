from torch_geometric.loader import DataLoader
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

    loader = DataLoader(train_X, batch_size=np.random.randint(1, 1+len(train_X)))
    for graphs in loader:
        graphs_with_deleted_node = cst.node_deletion.delete_node(graphs).to_data_list()
        for graph, graph_with_deleted_node in zip(graphs.to_data_list(), graphs_with_deleted_node):
            assert len(graph.x) - 1 == len(graph_with_deleted_node.x)
            assert len(graph.edge_weight) >= len(graph_with_deleted_node.edge_weight)
    
        """
         TODO: Shubham, when you run this method, since the seed is fixed, 
         print out the x values and edge weights and assert those are equal a transformed graph
        """
