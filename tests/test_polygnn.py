from polygnn_kit.polygnn_kit import LadderPolymer, LinearPol
import pytest
import torch
from torch_geometric.loader import DataLoader
import polygnn_trainer as pt
import numpy as np

from polygnn import __version__
from polygnn import models
from polygnn import layers as layers
from polygnn import featurize as feat
from polygnn import diff, contrast
from polygnn.trimer import TrimerLadder, TrimerLinear


def test_version():
    assert __version__ == "0.1.0"


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


def test_bond_config(example_data):
    """
    This test checks if the n_features argument of a BondConfig object is
    properly computed
    """
    assert example_data["bond_config"].n_features == 5


def test_atom_config(example_data):
    """
    This test checks if the n_features argument of a BondConfig object is
    properly computed
    """
    assert example_data["atom_config"].n_features == 70


def test_model_training(example_data):
    """
    This test checks:
    - that an MPNN model can train on a CPU without explicit errors.
      "Silent" errors are not caught here.
    - that featurization can run without explicit errors.
    """
    train_config = pt.train.trainConfig(
        loss_obj=pt.loss.sh_mse_loss(),
        amp=False,
        device="cpu",
        epoch_suffix="",
        multi_head=False,
    )
    hps = pt.hyperparameters.HpConfig()
    hps.set_values(
        {
            "batch_size": 2,
            "r_learn": 0.01,
            "capacity": 2,
        }
    )
    train_config.hps = hps
    train_config.epochs = 2
    train_config.fold_index = -1
    pt.train.train_submodel(
        example_data["model"],
        example_data["train_X"],
        example_data["val_X"],
        example_data["scalers"],
        train_config,
    )

    assert True


def test_polyGNN_pretraining(example_data):
    """
    This test checks that freezing and fine-tuning both work.
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
        mpnn=pretrained.mpnn,
        mlp_head=pretrained.mlp_head,
        mlp_out=pretrained.mlp_out,
        freeze=True,
    )
    # There is a minus 1 below since the capacity of the mlp_head is
    # automatically decremented by 1 inside the
    # preTrainContrastivePolyGNN object.
    assert downstream.mlp_head.hps.capacity.get_value() == (7 - 1)
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
        x.y = torch.tensor(1.3)  # set dummy y value for each data point
        x.selector = torch.FloatTensor([[1, 0]]).detach()
    loader = DataLoader(train_X, batch_size=len(train_X))
    optimizer = torch.optim.Adam(downstream.parameters(), lr=0.001)  # Adam optimization
    # ###############################################################
    # Test that freezing works
    # ###############################################################
    pretrain_weight_before = downstream.mlp_out.linear.weight.clone().detach()
    downstream_weight_before = downstream.final.linear.weight.clone().detach()
    for data in loader:
        _, loss = pt.train.minibatch(data, train_config, downstream, selector_dim)
        loss.backward()
        optimizer.step()
    pretrain_weight_after = downstream.mlp_out.linear.weight.clone().detach()
    downstream_weight_after = downstream.final.linear.weight.clone().detach()
    # The downstream weights should be changed after the weight update.
    assert not torch.equal(downstream_weight_before, downstream_weight_after)
    # The frozen pretraining weights should be the same before and
    # after the weight update.
    assert torch.equal(pretrain_weight_before, pretrain_weight_after)
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
        mpnn=pretrained.mpnn,
        mlp_head=pretrained.mlp_head,
        mlp_out=pretrained.mlp_out,
        freeze=False,
    )
    # Re-instantiate the optimizer so that the model parameters of the
    # new fine-tunable model are optimized rather than the parameters
    # of the old frozen model.
    optimizer = torch.optim.Adam(downstream.parameters(), lr=0.001)  # Adam optimization
    pretrain_weight_before = downstream.mlp_out.linear.weight.clone().detach()
    downstream_weight_before = downstream.final.linear.weight.clone().detach()
    for data in loader:
        _, loss = pt.train.minibatch(data, train_config, downstream, selector_dim)
        loss.backward()
        optimizer.step()
    pretrain_weight_after = downstream.mlp_out.linear.weight.clone().detach()
    downstream_weight_after = downstream.final.linear.weight.clone().detach()
    # Both the downstream and the pretrained weights should be changed
    # after the weight update.
    assert not torch.equal(pretrain_weight_before, pretrain_weight_after)
    assert not torch.equal(downstream_weight_before, downstream_weight_after)
    # ###############################################################


@pytest.mark.parametrize(
    "smiles_pass, bond_config_pass, atom_config_pass, rep_pass",
    [
        (
            "[*]=CNC=[*]",
            feat.BondConfig(True, False, True),
            feat.AtomConfig(
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
            ),
            "monocycle",
        ),
        (
            "[*]CCN[*]",
            feat.BondConfig(True, False, True),
            feat.AtomConfig(
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
            ),
            "monocycle",
        ),
        (
            "[*]OC(C)CCC(C)OC(=O)CCCCC([*])=O",
            feat.BondConfig(True, False, True),
            feat.AtomConfig(
                True,
                True,
                True,
                True,
                True,
                True,
                True,  # needs to be True to avoid separate features for SP2/SP3
                True,
            ),
            "monocycle",
        ),
        (
            "C1%12=C(C=C%10C(=C1)S(=O)(=O)C2=C(C=C9C(=C2)OC3=C(C=C4C(=C3)C5(CC4(C)C)CC(C)(C)C6=C5C=C8C(=C6)OC7=C(C(C#N)=C([g])C([t])=C7C#N)O8)O9)S%10(=O)=O)OC%11=CC%15=C(C=C%11O%12)C%14(C%13=CC(O[e])=C(O[d])C=C%13C(C)(C)C%14)CC%15(C)C",
            feat.BondConfig(False, False, True),
            feat.AtomConfig(
                True,
                True,
                True,
                True,
                True,
                True,
                True,  # if True, SP2/SP3 will be combined in one feature
                False,  # if False, aromatic feature will *not* be used
            ),
            "monocycle",
        ),
        (
            "[*]OC1CCC(OC(=O)NCCSCCCCCCSCCNC([*])=O)CC1",
            feat.BondConfig(True, True, True),
            feat.AtomConfig(
                True,
                True,
                True,
                True,
                True,
                True,
                combo_hybrid=False,  # needs to be True to avoid separate features for SP2/SP3
                aromatic=True,
            ),
            "trimer",
        ),
        (
            "C1C([g])C([e])OC([t])C1[d]",
            feat.BondConfig(True, True, True),
            feat.AtomConfig(
                True,
                True,
                True,
                True,
                True,
                True,
                combo_hybrid=False,  # needs to be True to avoid separate features for SP2/SP3
                aromatic=True,
            ),
            "trimer",
        ),
        (
            "C1=CC(O[t])=C(O[d])C=C1C2CC(CN)=C([e])C([g])=C2CN",
            feat.BondConfig(True, True, True),
            feat.AtomConfig(
                True,
                True,
                True,
                True,
                True,
                True,
                combo_hybrid=False,  # needs to be True to avoid separate features for SP2/SP3
                aromatic=True,
            ),
            "trimer",
        ),
    ],
)
def test_intensiveness_pass1(smiles_pass, bond_config_pass, atom_config_pass, rep_pass):
    """
    This test checks that pairs of intensive smiles are mapped to the
    same fingerprint
    """

    d = diff.atom_diffs_dimer(smiles_pass, bond_config_pass, atom_config_pass, rep_pass)
    assert d == {}


@pytest.mark.parametrize(
    "smiles_fail, bond_config_fail, atom_config_fail, rep_fail",
    [
        (
            "[*]OC(C)CCC(C)OC(=O)CCCCC([*])=O",
            feat.BondConfig(True, False, True),
            feat.AtomConfig(
                True,
                True,
                True,
                True,
                True,
                True,
                combo_hybrid=False,  # needs to be True to avoid separate features for SP2/SP3
                aromatic=True,
            ),
            "monocycle",
        ),
    ],
)
def test_intensiveness_fail(smiles_fail, bond_config_fail, atom_config_fail, rep_fail):

    d = diff.atom_diffs_dimer(smiles_fail, bond_config_fail, atom_config_fail, rep_fail)
    assert d != {}


def test_atom_map_dimer():
    sm = "C([*])OC([*])O"
    correct_map = {
        0: 4,
        1: 5,
        2: 6,
        3: 7,
    }
    assert diff.atom_map_dimer(sm) == correct_map


def test_bond_map_dimer():
    sm = "CN([*])CC([*])O"
    correct_map = {
        0: 4,
        1: 5,
        2: 6,
        3: 7,
        8: 9,
    }
    assert diff.bond_map_dimer(sm) == correct_map


def test_TrimerLinear():
    sm = "CCC(N*)N(*)C"
    lp = LinearPol(sm)
    trimer = TrimerLinear(lp)
    correct_map = {
        0: 7,
        1: 8,
        2: 9,
        3: 10,
        5: 11,
        7: 12,
        4: None,
        6: None,
    }

    assert trimer.parent_to_self == correct_map

    assert trimer.analogue_inds == (11, 16)


@pytest.fixture
def example_ladder_data():
    sm = "C1C([g])C([e])OC([t])C1[d]"
    pol = LadderPolymer(sm)
    return {
        "pol": pol,
    }


def test_LadderPolymer(example_ladder_data):
    pol = example_ladder_data["pol"]
    # indices below are assigned as "A" since
    # 2 < 4
    assert pol.starA1_ind == 2
    assert pol.connectorA1_ind == 1
    assert pol.starA2_ind == 9
    assert pol.connectorA2_ind == 8
    # indices below are assigned as "B" since
    # 4 < 2
    assert pol.starB1_ind == 4
    assert pol.connectorB1_ind == 3
    assert pol.starB2_ind == 7
    assert pol.connectorB2_ind == 6


def test_TrimerLadder(example_ladder_data):
    pol = example_ladder_data["pol"]
    trimer = TrimerLadder(pol)
    correct_map = {
        0: 8,
        1: 9,
        2: None,
        3: 10,
        4: None,
        5: 11,
        6: 12,
        7: None,
        8: 13,
        9: None,
    }

    assert trimer.parent_to_self == correct_map

    assert trimer.analogue_inds == [(9, 7), (10, 6)]

    trimer = TrimerLadder(pol.multiply(2))
    assert trimer.analogue_inds == [(15, 13), (16, 12)]


def test_trimer_feats():
    sm = "[*]CC[*]"
    bond_config = feat.BondConfig(True, True, True)
    atom_config = feat.AtomConfig(
        True,
        True,
        True,
        True,
        True,
        True,
        combo_hybrid=False,  # needs to be True to avoid separate features for SP2/SP3
        aromatic=True,
    )
    data = feat.get_minimum_graph_tensor(sm, bond_config, atom_config, "trimer")
    carbon_feat = (
        [1.0]  # element
        + [0.0] * (len(feat.element_names) - 1)  # element
        + [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # degree
        + [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]  # implicit valence
        + [0.0]  # formal charge
        + [0.0]  # num rad e
        + [0.0, 0.0, 1.0, 0.0, 0.0]  # hybridization
        + [0.0]  # aromaticity
    )
    assert len(carbon_feat) == atom_config.n_features
    out_arr = data.x.numpy()
    test_arr = np.array([carbon_feat] * 2)
    assert (out_arr == test_arr).all()

    # also test [*]C[*] to make sure that single-atom main chains are OK
    sm = "[*]C[*]"
    data = feat.get_minimum_graph_tensor(sm, bond_config, atom_config, "trimer")
    out_arr = data.x.numpy()
    test_arr = np.array([carbon_feat] * 1)
    assert (out_arr == test_arr).all()
