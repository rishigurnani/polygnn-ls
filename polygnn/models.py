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
        # Resist the temptation to over-write data.x in the subsequent
        # steps. Instead, let's assign the output of each step to a
        # new variable, called `result`. This will prevent some
        # unintended consequences when we use this model inside a
        # LinearEnsemble.
        result = self.mpnn(data.x, data.edge_index, data.edge_weight, data.batch)
        result = F.leaky_relu(result)
        result = torch.cat((result, data.selector), dim=1)
        result = self.estimator(result)
        result = self.final(result)
        result = torch.clip(  # prevent inf and -inf
            result,
            min=-0.5,
            max=1.5,  # choose -0.5 and 1.5 since the output should be between 0 and 1
        )
        result[torch.isnan(result)] = 1.5  # prevent nan
        return result.view(data.num_graphs, 1)  # get the shape right


class polyGNN_fromPretrained(pt.std_module.StandardModule):
    def __init__(
        self,
        node_size,
        edge_size,
        selector_dim,
        hps,
        pretrained_hps,
        normalize_embedding=True,
        debug=False,
    ):
        """
        Keyword arguments
            node_size (int): The number of node features.
            edge_size (int): The number of edge features.
            selector_dim (int): The dimension of the selector vector.
            hps (HpConfig): The hyperparameters to use for
                all non-pretrained layers in the model.
            pretrained_hps (HpConfig): The hyperparameters to use for
                all pretrained layers in the model.
            normalize_embedding (bool): If True, the node features
                will be aggregated using the mean. Otherwise, the
                sum will be used for aggregation.
        """
        super().__init__(hps)

        self.node_size = node_size
        self.edge_size = edge_size
        self.selector_dim = selector_dim
        self.pretrained_hps = pretrained_hps
        self.normalize_embedding = normalize_embedding
        self.debug = debug

        # ###################################################
        # Set up the layers that control the representation.
        # ###################################################
        self.mpnn = layers.MtConcat_PolyMpnn(
            self.node_size,
            self.edge_size,
            self.selector_dim,
            self.pretrained_hps,
            self.normalize_embedding,
            self.debug,
            self.pretrained_hps.embedding_dim.get_value(),
        )
        # We need to decrement the capacity of the MLP layers by 1 since
        # the output layer counts as 1 toward the capacity.
        self.mlp_hps = deepcopy(self.pretrained_hps)
        self.mlp_hps.set_values({"capacity": self.mlp_hps.capacity.get_value() - 1})
        self.mlp_head = pt.layers.Mlp(
            input_dim=self.mlp_hps.embedding_dim.get_value(),
            output_dim=self.mlp_hps.embedding_dim.get_value(),
            hps=self.mlp_hps,
            debug=False,
        )
        self.mlp_out = pt.layers.my_output(
            self.mlp_hps.embedding_dim.get_value(),
            self.mlp_hps.embedding_dim.get_value(),
        )
        # ###################################################

        # Set up the Estimator.
        self.estimator = pt.layers.Mlp(
            input_dim=self.mpnn.readout_dim + self.selector_dim,
            output_dim=32,
            hps=self.hps,
            debug=False,
        )
        ## Final layer of the Estimator.
        self.final = pt.layers.my_output(size_in=32, size_out=1)

    def init_pretrained(self, pretrained_state_dict, freeze):
        """
        Set the parameters of the layers that control the polymer
        representation.

        Keyword arguments:
            pretrained_state_dict (dict): The state_dict of the
                pretrained model.
            freeze (bool): If True, mpnn, mlp_head, and mlp_out will be
                frozen. Otherwise, they can be fine-tuned.
        """
        state_dict = self.state_dict()
        keys_to_freeze = []
        for key in state_dict.keys():
            if key in pretrained_state_dict:
                keys_to_freeze.append(key)
                param = pretrained_state_dict[key]
                state_dict[key].copy_(param)
        # Freeze parameters. According to my experiments, setting the
        # `requires_grad` attribute in the prior for loop does not work.
        # But, it does work in the for loop below. I'm not sure why this
        # is the case.
        if freeze:
            for name, param in self.named_parameters():
                if name in keys_to_freeze:
                    param.requires_grad = False

    def represent(self, data):
        x = self.mpnn(data.x, data.edge_index, data.edge_weight, data.batch)
        x = F.leaky_relu(x)
        x = self.mlp_head(x)
        x = self.mlp_out(x)
        return x

    def forward(self, data):
        # Resist the temptation to over-write data.x in the subsequent
        # steps. Instead, let's assign the output of each step to a
        # new variable, called `result`. This will prevent some
        # unintended consequences when we use this model inside a
        # LinearEnsemble.
        result = self.represent(data)
        result = F.leaky_relu(result)
        result = torch.cat((result, data.selector), dim=1)
        result = self.estimator(result)
        result = self.final(result)
        result = torch.clip(  # prevent inf and -inf
            result,
            min=-0.5,
            max=1.5,  # choose -0.5 and 1.5 since the output should be between 0 and 1
        )
        result[torch.isnan(result)] = 1.5  # prevent nan
        return result.view(data.num_graphs, 1)  # get the shape right
