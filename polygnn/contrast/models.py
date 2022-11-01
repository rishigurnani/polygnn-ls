import torch
import torch.nn.functional as F
import random
import numpy as np
import polygnn_trainer as pt
from copy import deepcopy
import polygnn.layers as layers

random.seed(2)
torch.manual_seed(2)
np.random.seed(2)

# ##########################
# Pretraining PolyGNN with Contrastive Loss
# #########################
class preTrainContrastivePolyGNN(pt.std_module.StandardModule):
    def __init__(
        self,
        node_size,
        edge_size,
        selector_dim,
        hps,
        normalize_embedding=True,
        debug=False,
    ):
        super().__init__(hps)

        self.node_size = node_size
        self.edge_size = edge_size
        self.normalize_embedding = normalize_embedding
        self.debug = debug
        self.mpnn = layers.MtConcat_PolyMpnn(
            node_size,
            edge_size,
            selector_dim,
            self.hps,
            normalize_embedding,
            debug,
            self.hps.embedding_dim.get_value(),
        )
        # We need to decrement the capacity of the MLP layers by 1 since
        # the output layer counts as 1 toward the capacity.
        self.mlp_hps = deepcopy(self.hps)
        self.mlp_hps.set_values({"capacity": self.mlp_hps.capacity.get_value() - 1})

        # Set up Mlp for representation.
        self.mlp_head = pt.layers.Mlp(
            input_dim=self.hps.embedding_dim.get_value(),
            output_dim=self.hps.embedding_dim.get_value(),
            hps=self.mlp_hps,
            debug=False,
        )
        self.mlp_out = pt.layers.my_output(
            self.mlp_hps.embedding_dim.get_value(),
            self.mlp_hps.embedding_dim.get_value(),
        )
        # Set up Mlp for projection.
        self.projection_head = pt.layers.Mlp(
            input_dim=self.hps.embedding_dim.get_value(),
            output_dim=self.hps.embedding_dim.get_value(),
            hps=self.mlp_hps,
            debug=False,
        )
        self.projection_out = pt.layers.my_output(
            self.mlp_hps.embedding_dim.get_value(),
            self.mlp_hps.embedding_dim.get_value(),
        )

    def represent(self, data):
        """
        The contents of this method are separated from `self.forward` so that
        this block can be called in isolation during downstream tasks.
        """
        x = self.mpnn(data.x, data.edge_index, data.edge_weight, data.batch)
        x = F.leaky_relu(x)
        x = self.mlp_head(x)
        x = self.mlp_out(x)
        return x

    def project(self, x):
        x = self.projection_head(x)
        x = self.projection_out(x)
        return x

    def forward(self, data, data_augmented):
        """
        This method can be called during pre-training.
        """
        x, x_aug = self.represent(data), self.represent(data_augmented)
        # Deal with data.
        x = F.leaky_relu(x)
        x = self.project(x)

        # Deal with data_augmented.
        x_aug = F.leaky_relu(x_aug)
        x_aug = self.project(x_aug)

        # Interleave x:(N, D) and x_aug:(N, D) into a new tensor, data.y:(2*N, D)
        n, d = x.size()
        data.y = torch.stack((x, x_aug), dim=1).view(2 * n, d)
        return data
