from torch_geometric.loader import DataLoader
from polygnn import featurize as feat
from polygnn import contrast as cst
import torch
from scipy import spatial
import numpy as np


def test_contrast_loss():
    temperature = 1.2
    train_smiles = ["[*]CC[*]", "[*]CC[*]", "[*]CC(C)[*]", "[*]CC(C)[*]"]  # 2N
    N = len(train_smiles) // 2
    # Create the representations of four vectors: a,b,c,d. Where (a,b)
    # and (c, d) are paired views of the same data point.
    a = [1.0, 1.0]
    b = [2.0, 1.0]
    c = [5.0, 1.0]
    d = [6.0, 1.0]
    cos = lambda x: 1 - spatial.distance.cosine(x[0], x[1])  # cosine similarity.
    # ###################################################################################
    # Compute correct loss
    # ###################################################################################
    # Compute similarity of each term.
    sim_ab = cos([a, b])
    sim_ac = cos([a, c])
    sim_ad = cos([a, d])
    sim_ba = sim_ab
    sim_bc = cos([b, c])
    sim_bd = cos([b, d])
    sim_ca = sim_ac
    sim_cb = sim_bc
    sim_cd = cos([c, d])
    sim_da = sim_ad
    sim_db = sim_bd
    sim_dc = sim_cd
    # The total loss is (1/(2*N)) * ((l_ab + l_ba) + (l_cd + l_dc)). Below,
    # let's compute the individual loss terms that are part of this
    # formula.
    # We will start by computing the denominator (dnm) of each term.
    dnm_ab = (
        np.exp(sim_ab / temperature)
        + np.exp(sim_ac / temperature)
        + np.exp(sim_ad / temperature)
    )
    dnm_ba = (
        np.exp(sim_ba / temperature)
        + np.exp(sim_bc / temperature)
        + np.exp(sim_bd / temperature)
    )
    dnm_cd = (
        np.exp(sim_ca / temperature)
        + np.exp(sim_cb / temperature)
        + np.exp(sim_cd / temperature)
    )
    dnm_dc = (
        np.exp(sim_da / temperature)
        + np.exp(sim_db / temperature)
        + np.exp(sim_dc / temperature)
    )
    # Now we can compute each loss term.
    l_ab = -np.log(np.exp(sim_ab / temperature) / dnm_ab)
    l_ba = -np.log(np.exp(sim_ba / temperature) / dnm_ba)
    l_cd = -np.log(np.exp(sim_cd / temperature) / dnm_cd)
    l_dc = -np.log(np.exp(sim_dc / temperature) / dnm_dc)
    # Compute the total loss.
    correct_loss = (1 / (2 * N)) * (l_ab + l_ba + l_cd + l_dc)
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
    loader = DataLoader(train_X, batch_size=len(train_X))
    loss_fn = cst.loss.contrast_loss(temperature)
    for data in loader:
        tens = torch.tensor([a, b, c, d])
        data.y = tens
        result = loss_fn(data).item()
    assert np.isclose(result, correct_loss)


def test_noise_augmentation():
    torch.manual_seed(12)
    train_smiles = ["[*]CCC[*]", "[*]CCC[*]"]
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
    loader = DataLoader(train_X, batch_size=len(train_X))
    for data in loader:
        data_aug = cst.noise.add_noise(atom_config, data)
        # KENNY and/or SHUBHAM, add tests here.
        epsilon = 1e-6
        for row in data_aug.x:
            assert abs(row[0:44].sum() - 1) < epsilon
            assert abs(row[44:55].sum() - 1) < epsilon
            assert abs(row[55:62].sum() - 1) < epsilon
            assert abs(row[64:69].sum() - 1) < epsilon
            assert row[69] <= 1
            assert row[69] >= 0
        print("ADDING NOISE WORKS")

        ######################
        # Test Masking Noise
        ######################
        noise_mask = np.zeros(data.x.shape)
        noise_mask[0] = np.ones(data.x.shape[1])
        data_aug = cst.noise.add_noise(atom_config, data, noise_mask)
        assert abs(data_aug.x[0, 0:44].sum() - 1) < epsilon
        assert abs(data_aug.x[0, 44:55].sum() - 1) < epsilon
        assert abs(data_aug.x[0, 55:62].sum() - 1) < epsilon
        assert abs(data_aug.x[0, 64:69].sum() - 1) < epsilon
        assert data_aug.x[0, 69] <= 1
        assert data_aug.x[0, 69] >= 0
        assert np.array_equiv(data_aug.x[1:], data.x[1:])
        print("MASKING NOISE WORKS")


test_noise_augmentation()
