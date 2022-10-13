import numpy as np
import torch
from polygnn import featurize

def one_hot_noise(data, mu=0, sigma=0.025):
    noise = np.random.normal(mu, sigma, data.shape)
    aug_data = data + noise
    softmax = torch.nn.Softmax()
    return softmax(aug_data)

def boolean_noise(data, mu=0, sigma=0.025):
    noise = np.random.normal(mu, sigma, data.shape)
    return np.clip(data + noise, 0, 1)

def float_noise(data, mu=0, sigma=0.05):
    noise = np.random.normal(mu, sigma, data.shape)
    return data + noise

def add_noise(atom_config, data):
    """
    Adds noise across a batch of data based on features used
    :param: atom_config: Atom feature config
    :param: data: 2D data array of batch_size by # of features
    :return: The updated data with noise added
    """
    # Based on AtomConfig feature order in featurize.py
    feature_index = 0
    if atom_config.element_type:
        data[feature_index:feature_index + len(featurize.element_names)] = one_hot_noise(data[feature_index:feature_index + len(featurize.element_names)])
        feature_index += len(featurize.element_names)
    if atom_config.degree:
        degree_len = 11
        data[feature_index:feature_index + degree_len] = one_hot_noise(data[feature_index:feature_index + degree_len])
        feature_index += degree_len
    if atom_config.implicit_valence:
        valence_len = 7
        data[feature_index:feature_index + valence_len] = one_hot_noise(data[feature_index:feature_index + valence_len])
        feature_index += valence_len
    if atom_config.formal_charge:
        charge_len = 1
        data[feature_index:feature_index + charge_len] = float_noise(data[feature_index:feature_index + charge_len])
        feature_index += charge_len
    if atom_config.num_rad_e:
        rad_len = 1
        data[feature_index:feature_index + rad_len] = float_noise(data[feature_index:feature_index + rad_len])
        feature_index += rad_len
    if atom_config.hybridization:
        if not atom_config.combo_hybrid:
            hybrid_len = 5
            data[feature_index:feature_index + hybrid_len] = one_hot_noise(data[feature_index:feature_index + hybrid_len])
            feature_index += hybrid_len
        else:
            combo_hybrid_len = 4
            data[feature_index:feature_index + combo_hybrid_len] = one_hot_noise(data[feature_index:feature_index + combo_hybrid_len])
            feature_index += combo_hybrid_len
    if atom_config.aromatic:
        aromatic_len = 1
        data[feature_index:feature_index + aromatic_len] = boolean_noise(data[feature_index:feature_index + aromatic_len])
        feature_index += aromatic_len

    return data