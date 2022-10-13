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
        self.mlp_head = pt.models.MlpOut(
            input_dim=self.mpnn.readout_dim,
            output_dim=32,
            hps=self.hps,
            debug=False,
        )

        self.temperature_param = torch.nn.Parameter(torch.ones(1))
        self.projection_head = pt.models.MlpOut(
            input_dim=32,
            output_dim=self.hps.embedding_dim,
            hps=self.hps,
            debug=False,
        )

    def represent(self, data, data_augmented):       
        """
        The contents of this method are separated from `self.forward` so that
        this block can be called in isolation during downstream tasks.
        """
        # Deal with data.
        x, edge_index, edge_weight, batch = (
            data.x,
            data.edge_index,
            data.edge_weight,
            data.batch
        )  # extract variables
        x = self.mpnn(x, edge_index, edge_weight, batch)
        x = F.leaky_relu(x)
        x = self.mlp_head(x)
        
        # Deal with data_augmented.
        x_aug, edge_index_aug, edge_weight_aug, batch_aug = (
            data_augmented.x,
            data_augmented.edge_index,
            data_augmented.edge_weight,
            data_augmented.batch
        )  # extract variables
        x_aug = self.mpnn(x_aug, edge_index_aug, edge_weight_aug, batch_aug)
        x_aug = F.leaky_relu(x_aug)
        x_aug = self.mlp_head(x_aug)
        return x, x_aug
    
    def forward(self, data, data_augmented):
        """
        This method can be called during pre-training.
        """
        x, x_aug = self.represent(data, data_augmented)
        # Deal with data.
        x = F.leaky_relu(x)
        x = self.projection_head(x)
        
        # Deal with data_augmented.
        x_aug = F.leaky_relu(x_aug)
        x_aug = self.projection_head(x_aug)

        # Convert x:(N, D) and x_aug:(N, D) into (N, D, 2)
        data.y = torch.stack([x, x_aug], dim=2)
        data.temperature = self.temperature_param
        return data
