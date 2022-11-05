import torch
from torch_geometric.loader import DataLoader
import polygnn_trainer as pt

from polygnn import __version__
from polygnn import models
from polygnn import layers as layers
from polygnn import featurize as feat
from polygnn import contrast
import pytest


@pytest.fixture
def example_data():
    selector_dim = 2
    capacity = 2
    hps = pt.hyperparameters.HpConfig()
    hps.set_values(
        {
            "capacity": capacity,
            "dropout_pct": 0.0,
            "activation": torch.nn.functional.leaky_relu,
        }
    )
    train_smiles = ["[*]CC[*]", "[*]CC(C)[*]"]
    val_smiles = ["[*]CCN[*]", "[*]COC[*]"]
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
    val_X = [
        feat.get_minimum_graph_tensor(x, bond_config, atom_config, "monocycle")
        for x in val_smiles
    ]
    for x in train_X + val_X:
        x.y = torch.tensor(1.3)  # set dummy y value for each data point
        x.selector = torch.FloatTensor([[1, 0]]).detach()
    return {
        "hps": hps,
        "train_smiles": train_smiles,
        "model": models.polyGNN(
            node_size=atom_config.n_features,
            edge_size=bond_config.n_features,
            selector_dim=selector_dim,
            hps=hps,
        ),
        "layer": layers.MtConcat_PolyMpnn(
            node_size=atom_config.n_features,
            edge_size=bond_config.n_features,
            selector_dim=selector_dim,
            hps=hps,
            normalize_embedding=True,
            debug=False,
        ),
        "train_X": train_X,
        "val_X": val_X,
        "batch_size": 2,
        "scalers": {
            "prop1": pt.scale.DummyScaler(),
            "prop2": pt.scale.DummyScaler(),
        },
        "bond_config": bond_config,
        "atom_config": atom_config,
    }


def test_polyGNN_pretraining(example_data):
    """
    This test checks that freezing and fine-tuning both work. For some
    reason, this test only works when it is called by itself, but not
    when it is combined with other tests.
    """
    bond_config = feat.BondConfig(True, True, True)
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
    # Create an HpConfig object for the pretrained model.
    pretrained_hps = pt.hyperparameters.HpConfig()
    embedding_dim = pt.hyperparameters.ModelParameter(int)
    setattr(pretrained_hps, "embedding_dim", embedding_dim)
    pretrained_hps.set_values(
        {
            "capacity": 7,
            "dropout_pct": 0.0,
            "activation": torch.nn.functional.leaky_relu,
            "embedding_dim": 128,
        }
    )
    # Load the pretrained model.
    pretrained = contrast.models.preTrainContrastivePolyGNN(
        node_size=atom_config.n_features,
        edge_size=bond_config.n_features,
        selector_dim=None,
        hps=pretrained_hps,
    )
    pretrained.load_state_dict(
        torch.load("sample_models/simclr.pt", map_location=torch.device("cpu"))
    )
    # Create the downstream model.
    selector_dim = len(example_data["scalers"])
    downstream = models.polyGNN_fromPretrained(
        node_size=atom_config.n_features,
        edge_size=bond_config.n_features,
        selector_dim=selector_dim,
        hps=example_data["hps"],
        pretrained_hps=pretrained_hps,
    )
    # There is a minus 1 below since the capacity of the mlp_head is
    # automatically decremented by 1 inside the
    # preTrainContrastivePolyGNN object.
    assert downstream.mlp_head.hps.capacity.get_value() == (7 - 1)
    downstream.init_pretrained(pretrained.state_dict(), freeze=True)
    pretrained_weight = downstream.mlp_out.linear.weight.clone().detach()
    # Below we check that the weights have been correctly swapped.
    assert torch.equal(
        pretrained.mlp_out.linear.weight.clone().detach(),
        pretrained_weight,
    )
    # Create a trainConfig object for the downstream model.
    train_config = pt.train.trainConfig(
        loss_obj=pt.loss.sh_mse_loss(),
        amp=False,
        device="cpu",
        epoch_suffix="",
        multi_head=False,
    )
    # Create the training data.
    ## Make the graphs for each smiles string.
    train_X = [
        feat.get_minimum_graph_tensor(x, bond_config, atom_config, "monocycle")
        for x in example_data["train_smiles"]
    ]
    ## Manually add a selector vector and label to each graph.
    for x in train_X:
        x.y = torch.tensor([0.6])  # set dummy y value for each data point
        x.selector = torch.FloatTensor([[1, 0]]).detach()
    loader = DataLoader(train_X, batch_size=len(train_X))
    optimizer = torch.optim.Adam(downstream.parameters(), lr=0.001)  # Adam optimization
    # ###############################################################
    # Test that freezing works
    # ###############################################################
    pretrain_weight_before = downstream.mlp_out.linear.weight.clone().detach()
    downstream_weight_before = downstream.final.linear.weight.clone().detach()
    for data in loader:
        optimizer.zero_grad()
        _, loss = pt.train.minibatch(data, train_config, downstream, selector_dim)
        loss.backward()
        pt.utils.analyze_gradients(downstream.named_parameters())
        optimizer.step()
    pretrain_weight_after = downstream.mlp_out.linear.weight.clone().detach()
    downstream_weight_after = downstream.final.linear.weight.clone().detach()
    # The frozen pretraining weights should be the same before and
    # after the weight update.
    assert torch.equal(
        pretrain_weight_before, pretrain_weight_after
    ), downstream.mlp_out.linear.weight.requires_grad
    # The downstream weights should be changed after the weight update.
    assert not torch.equal(
        downstream_weight_before, downstream_weight_after
    ), downstream.final.linear.weight.requires_grad
    # ###############################################################
    # Test that fine-tuning works
    # ###############################################################
    # We need to re-load the pretrained model so that the `requires_grad`
    # attribute of all its parameters are reset.
    pretrained = contrast.models.preTrainContrastivePolyGNN(
        node_size=atom_config.n_features,
        edge_size=bond_config.n_features,
        selector_dim=None,
        hps=pretrained_hps,
    )
    pretrained.load_state_dict(
        torch.load("sample_models/simclr.pt", map_location=torch.device("cpu"))
    )
    # Create the downstream model.
    downstream = models.polyGNN_fromPretrained(
        node_size=atom_config.n_features,
        edge_size=bond_config.n_features,
        selector_dim=selector_dim,
        hps=example_data["hps"],
        pretrained_hps=pretrained_hps,
    )
    downstream.init_pretrained(pretrained.state_dict(), freeze=False)
    # Re-instantiate the optimizer so that the model parameters of the
    # new fine-tunable model are optimized rather than the parameters
    # of the old frozen model.
    optimizer = torch.optim.Adam(downstream.parameters(), lr=0.001)  # Adam optimization
    pretrain_weight_before = downstream.mlp_out.linear.weight.clone().detach()
    downstream_weight_before = downstream.final.linear.weight.clone().detach()
    for data in loader:
        optimizer.zero_grad()
        _, loss = pt.train.minibatch(data, train_config, downstream, selector_dim)
        loss.backward()
        optimizer.step()
        pt.utils.analyze_gradients(downstream.named_parameters())
    pretrain_weight_after = downstream.mlp_out.linear.weight.clone().detach()
    downstream_weight_after = downstream.final.linear.weight.clone().detach()
    # Both the downstream and the pretrained weights should be changed
    # after the weight update.
    assert not torch.equal(pretrain_weight_before, pretrain_weight_after)
    assert not torch.equal(downstream_weight_before, downstream_weight_after)
    # ###############################################################
