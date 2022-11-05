from polygnn import featurize as feat
from polygnn import contrast as cst

"""
Basic unit test for removing node from graph
"""


def test_node_deletion():
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
    for graph in train_X:
        graph_with_deleted_node = cst.node_deletion.delete_node(graph)
        assert len(graph.x) - 1 == len(graph_with_deleted_node.x)
        assert len(graph.edge_weight) > len(graph_with_deleted_node.edge_weight)
