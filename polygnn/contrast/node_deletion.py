import numpy as np
import torch
from torch_geometric.data import Data


def _delete_node(batch_graph_data, random_ids):
    # Implement parallelization over batch
    all_ids = torch.arange(0, len(batch_graph_data.batch))
    # compareview = all_ids.expand(random_ids.shape[0], all_ids.shape[0]).T
    #
    # keep_ids = random_ids[(compareview != random_ids).T.prod(1)==1]
    combined = torch.cat((random_ids, all_ids))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1].long()

    batch_graph_data.x = torch.index_select(batch_graph_data.x, 0, difference)
    batch_graph_data.batch = torch.index_select(batch_graph_data.batch, 0, difference)

    np_edge_index = batch_graph_data.edge_index.numpy()
    edges_to_keep = np.invert(
        np.logical_or(np.in1d(np_edge_index[0], random_ids), np.in1d(np_edge_index[1], random_ids))
    )
    edges_to_keep = torch.tensor(np.where(edges_to_keep)[0])

    batch_graph_data.edge_weight = torch.index_select(batch_graph_data.edge_weight, 0, edges_to_keep)
    batch_graph_data.edge_index = torch.index_select(batch_graph_data.edge_index, 1, edges_to_keep)
    for i, idx_to_remove in enumerate(random_ids):
        batch_graph_data.edge_index[batch_graph_data.edge_index > idx_to_remove] -= 1
        batch_graph_data.ptr[i+1:] -= 1

    return batch_graph_data

    # Create new node feature array
    # new_x = torch.cat(
    #     [
    #         graph.x[0:idx_to_remove],
    #         graph.x[idx_to_remove + 1:],
    #     ]
    # )
    #
    # np_edge_index = graph.edge_index.cpu().numpy()
    # np_edge_weight = graph.edge_weight.cpu().numpy()
    #
    # # Select edges to remove that contained that node
    # edges_to_keep = np.invert(
    #     np.logical_or(np_edge_index[0] == idx_to_remove, np_edge_index[1] == idx_to_remove)
    # )
    #
    # np_edge_weight = np_edge_weight[edges_to_keep]
    # np_edge_index = np_edge_index[:, edges_to_keep]
    #
    # # Decrement index counter for all node with index greater than one deleted
    # np_edge_index[np_edge_index > idx_to_remove] -= 1
    #
    # return new_x, np_edge_index, np_edge_weight


def delete_node(batch_graph_data):
    """
    Deletes random node in pytorch graph data structure
    :param: graph_data: Torch geometric dataset for molecule
    :return: The updated graph with a node and corresponding edges removed
    """
    batch_graph_data = Data.clone(batch_graph_data)

    ptrs = batch_graph_data.ptr
    random_ids = torch.rand(len(ptrs)-1)
    len_graphs = ptrs[1:] - ptrs[:-1]

    random_ids *= len_graphs
    random_ids = torch.floor(random_ids)
    random_ids += ptrs[:-1]

    return _delete_node(batch_graph_data, random_ids)

    # Decompose batch of graphs to each composite graph
    # list_of_graphs = batch_graph_data.to_data_list()

    # new_list_of_graphs = []
    # for graph_data in list_of_graphs:
    #     # Select random node to remove
    #     random_idx_to_remove = np.random.randint(0, graph_data.x.size(dim=0))
    #
    #     new_x, np_edge_index, np_edge_weight = _delete_node(graph_data, random_idx_to_remove)
    #
    #     node_deleted_graph = Data(
    #         x=new_x,
    #         edge_index=torch.tensor(np_edge_index, dtype=torch.long),
    #         edge_weight=torch.tensor(np_edge_weight, dtype=torch.float),
    #     )
    #     new_list_of_graphs.append(node_deleted_graph)
    #
    # # Rebatch for faster inference
    # return Batch.from_data_list(new_list_of_graphs)
