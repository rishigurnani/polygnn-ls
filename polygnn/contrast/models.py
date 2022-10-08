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
        )

        # set up linear blocks
        self.mlp_head = pt.layers.Mlp(
            input_dim=self.mpnn.readout_dim,
            output_dim=32,
            hps=self.hps,
            debug=False,
        )

        self.projection_head = pt.layers.Mlp(
            input_dim=32,
            output_dim=32,
            hps=self.hps,
            debug=False,
        )

    def forward(self, data, data_augmented):
        x, edge_index, edge_weight, batch = (
            data.x,
            data.edge_index,
            data.edge_weight,
            data.batch
        )  # extract variables
        x = self.mpnn(x, edge_index, edge_weight, batch)

        x_aug, edge_index_aug, edge_weight_aug, batch_aug = (
            data_augmented.x,
            data_augmented.edge_index,
            data_augmented.edge_weight,
            data_augmented.batch
        )  # extract variables
        x_aug = self.mpnn(x_aug, edge_index_aug, edge_weight_aug, batch_aug)

        x = F.leaky_relu(x)
        x = self.mlp_head(x)
        x = F.leaky_relu(x)
        x = self.projection_head(x)

        x_aug = F.leaky_relu(x_aug)
        x_aug = self.mlp_head(x_aug)
        x_aug = F.leaky_relu(x_aug)
        x_aug = self.projection_head(x_aug)

        # Convert x:(N, D) and x_aug:(N, D) into (N, D, 2)
        return torch.stack([x, x_aug], dim=2)