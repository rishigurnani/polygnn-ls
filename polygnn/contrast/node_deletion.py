import numpy as np
import torch


def delete_node(graph_data):
    """
    Deletes random node in pytorch graph data structure
    :param: graph_data: Torch geometric dataset for molecule
    :return: The updated graph with a node and corresponding edges removed
    """
    # Select random node to remove
    random_idx_to_remove = np.random.randint(0, graph_data.x.size(dim=0))

    # Create new node feature array
    new_x = torch.cat([graph_data.x[0:random_idx_to_remove], graph_data.x[random_idx_to_remove+1:]])

    np_edge_index = data.edge_index.numpy()
    np_edge_weight = data.edge_weight.numpy()

    # Select edges to remove that contained that node
    edges_to_keep = np.invert(np.logical_or(np_edge_index[0] == 0, np_edge_index[1] == 0))

    np_edge_weight = np_edge_weight[:, edges_to_keep]
    np_edge_index = np_edge_index[edges_to_keep]

    # Decrement index counter for all node with index greater than one deleted
    np_edge_index[np_edge_index > random_idx_to_remove] -= 1

    node_deleted_graph = Data(
        x=torch.tensor(new_x, dtype=torch.float),
        edge_index=torch.tensor(np_edge_index, dtype=torch.long),
        edge_weight=torch.tensor(np_edge_weight, dtype=torch.float),
    )
    return node_deleted_graph
