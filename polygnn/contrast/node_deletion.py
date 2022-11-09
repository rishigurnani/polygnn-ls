import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch


def _delete_node(graph, idx_to_remove):
    # Create new node feature array
    new_x = torch.cat(
        [
            graph.x[0:idx_to_remove],
            graph.x[idx_to_remove + 1:],
        ]
    )

    np_edge_index = graph.edge_index.cpu().numpy()
    np_edge_weight = graph.edge_weight.cpu().numpy()

    # Select edges to remove that contained that node
    edges_to_keep = np.invert(
        np.logical_or(np_edge_index[0] == idx_to_remove, np_edge_index[1] == idx_to_remove)
    )

    np_edge_weight = np_edge_weight[edges_to_keep]
    np_edge_index = np_edge_index[:, edges_to_keep]

    # Decrement index counter for all node with index greater than one deleted
    np_edge_index[np_edge_index > idx_to_remove] -= 1

    return new_x, np_edge_index, np_edge_weight


def delete_node(batch_graph_data):
    """
    Deletes random node in pytorch graph data structure
    :param: graph_data: Torch geometric dataset for molecule
    :return: The updated graph with a node and corresponding edges removed
    """
    # Decompose batch of graphs to each composite graph
    list_of_graphs = batch_graph_data.to_data_list()

    new_list_of_graphs = []
    for graph_data in list_of_graphs:
        # Select random node to remove
        random_idx_to_remove = np.random.randint(0, graph_data.x.size(dim=0))

        new_x, np_edge_index, np_edge_weight = _delete_node(graph_data, random_idx_to_remove)

        node_deleted_graph = Data(
            x=new_x,
            edge_index=torch.tensor(np_edge_index, dtype=torch.long),
            edge_weight=torch.tensor(np_edge_weight, dtype=torch.float),
        )
        new_list_of_graphs.append(node_deleted_graph)

    # Rebatch for faster inference
    return Batch.from_data_list(new_list_of_graphs)
