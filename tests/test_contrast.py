from torch_geometric.loader import DataLoader
from polygnn import featurize as feat
from polygnn import contrast as cst
import torch
from scipy.spatial.distance import cosine as cosine_distance
import numpy as np


def test_constrast_loss():
    temperature = 1
    train_smiles = ["[*]CC[*]", "[*]CC[*]", "[*]CC(C)[*]", "[*]CC(C)[*]"]  # 2N
    N = len(train_smiles) // 2
    train_reps = [
        [[1.0, 1.0, 1.0]],
        [[2.0, 2.0, 1.5]],
        [[5.0, -1.0, 0.0]],
        [[10.0, -2.0, 0.5]],
    ]
    # ###################################################################################
    # Compute correct loss
    # ###################################################################################
    S = np.zeros((len(train_reps), len(train_reps)))  # pairwise similarities, (2N, 2N)
    for i in range(len(train_reps)):
        for j in range(len(train_reps)):
            S[i][j] = 1 - cosine_distance(train_reps[i], train_reps[j])
    indicator = np.ones(S.shape)
    np.fill_diagonal(indicator, 0)
    L = -1 * (  # loss for each term, (2N, 2N)
        np.log10(np.exp(S / temperature) / np.sum(indicator * np.exp(S / temperature)))
    )
    assert len(set(np.diag(L).tolist())) == 1
    assert L[0][1] < L[0][2]
    assert L[0][1] < L[0][3]
    assert L[2][3] < L[2][0]
    assert L[2][3] < L[2][1]
    # Total loss
    correct_loss = (L[0][1] + L[1][0] + L[2][3] + L[3][2]) / (2 * N)
    idx = [np.arange(0, N + 1, 2), np.arange(1, N + 2, 2)]
    loss = L[idx].sum() / N
    assert np.isclose(correct_loss, loss)
    # ###################################################################################
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
        tens = torch.tensor(
            [
                [[1.0, 2.0], [1.0, 2.0], [1.0, 1.5]],
                [[5.0, 10.0], [-1.0, -2.0], [0.0, 0.5]],
            ]
        )
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
