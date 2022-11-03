import torch
import torch.nn.functional as F
import random
import numpy as np
import polygnn_trainer as pt

import polygnn.layers as layers

random.seed(2)
torch.manual_seed(2)
np.random.seed(2)

# ##########################
# Multi-task models
# #########################
class polyGNN(pt.std_module.StandardModule):
    def __init__(
        self,
        node_size,
        edge_size,
        selector_dim,
        hps,
        normalize_embedding=True,
        debug=False,
    ):
        """
        Keyword arguments
            node_size (int): The number of node features.
            edge_size (int): The number of edge features.
            selector_dim (int): The dimension of the selector vector.
            freeze (bool): If True, the parameters of the mpnn layers
                will be frozen. Otherwise, the mpnn layers will be
                fine-tuned.
            hps (HpConfig): The hyperparameters to use for
                all layers in the model.
            normalize_embedding (bool): If True, the node features
                will be aggregated using the mean. Otherwise, the
                sum will be used for aggregation.
        """
        super().__init__(hps)

        self.node_size = node_size
        self.edge_size = edge_size
        self.selector_dim = selector_dim
        self.normalize_embedding = normalize_embedding
        self.debug = debug

        self.mpnn = layers.MtConcat_PolyMpnn(
            node_size,
            edge_size,
            selector_dim,
            self.hps,
            normalize_embedding,
            debug,
        )

        # Set up the Estimator.
        self.estimator = pt.layers.Mlp(
            input_dim=self.mpnn.readout_dim + self.selector_dim,
            output_dim=32,
            hps=self.hps,
            debug=False,
        )
        ## Final layer of the Estimator.
        self.final = pt.layers.my_output(size_in=32, size_out=1)

    def forward(self, data):
        data.x = self.mpnn(data.x, data.edge_index, data.edge_weight, data.batch)
        data.x = F.leaky_relu(data.x)
        data.x = torch.cat((data.x, data.selector), dim=1)
        data.x = self.estimator(data.x)
        data.x = self.final(data.x)
        data.x = torch.clip(  # prevent inf and -inf
            data.x,
            min=-0.5,
            max=1.5,  # choose -0.5 and 1.5 since the output should be between 0 and 1
        )
        data.x[torch.isnan(data.x)] = 1.5  # prevent nan
        return data.x.view(data.num_graphs, 1)  # get the shape right


class polyGNN_fromPretrained(pt.std_module.StandardModule):
    def __init__(
        self,
        node_size,
        edge_size,
        selector_dim,
        estimator_hps,
        mpnn,
        freeze,
        normalize_embedding=True,
        debug=False,
    ):
        """
        Keyword arguments
            node_size (int): The number of node features.
            edge_size (int): The number of edge features.
            selector_dim (int): The dimension of the selector vector.
            estimator_hps (HpConfig): The hyperparameters to use for
                the Estimator.
            mpnn (pt.std_module.StandardModule): The network with
                the pre-trained message passing layers.
            freeze (bool): If True, the parameters of the mpnn layers
                will be frozen. Otherwise, the mpnn layers will be
                fine-tuned.
            normalize_embedding (bool): If True, the node features
                will be aggregated using the mean. Otherwise, the
                sum will be used for aggregation.
        """
        super().__init__(estimator_hps)
        self.node_size = node_size
        self.edge_size = edge_size
        self.selector_dim = selector_dim
        self.freeze = freeze
        self.normalize_embedding = normalize_embedding
        self.debug = debug

        # Set up the MPNN.
        self.mpnn = mpnn
        if self.freeze:
            for param in self.mpnn.parameters():
                param.requires_grad = False

        # Set up the Estimator.
        self.estimator = pt.layers.Mlp(
            input_dim=self.mpnn.mpnn.readout_dim + self.selector_dim,
            output_dim=32,
            hps=self.hps,
            debug=False,
        )
        ## Final layer of the Estimator.
        self.final = pt.layers.my_output(size_in=32, size_out=1)

    def forward(self, data):
        data.x = self.mpnn.represent(data)
        data.x = F.leaky_relu(data.x)
        data.x = torch.cat((data.x, data.selector), dim=1)
        data.x = self.estimator(data.x)
        data.x = self.final(data.x)
        data.x = torch.clip(  # prevent inf and -inf
            data.x,
            min=-0.5,
            max=1.5,  # choose -0.5 and 1.5 since the output should be between 0 and 1
        )
        data.x[torch.isnan(data.x)] = 1.5  # prevent nan
        return data.x.view(data.num_graphs, 1)  # get the shape right
