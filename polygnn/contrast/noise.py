import torch
from torch.nn import functional as F
from polygnn import featurize
from . import utils


def one_hot_noise(data, mu=0, sigma=0.05):
    noise_weight = (
        torch.normal(mu, sigma, size=(1,), device=data.device).abs().clip(0, 1)
    )
    noise = F.gumbel_softmax(torch.randn(*data.shape, device=data.device))
    data = (1 - noise_weight) * data + noise_weight * noise
    return data


def boolean_noise(data, mask, mu=0, sigma=0.05):
    noise = torch.normal(mu, sigma, data.shape, device=data.device)
    noise = noise * mask
    return torch.clip(data + noise, 0, 1)


def float_noise(data, mask, mu=0, sigma=0.05):
    noise = torch.normal(mu, sigma, data.shape, device=data.device)
    noise = noise * mask
    return data + noise


def add_noise(atom_config, data, mask=[], mask_ratio=0.1):
    """
    Adds noise across a batch of data based on features used
    :param: atom_config: Atom feature config
    :param: data: 2D tensor of shape (number of nodes in batch, # of features)
    :return: The updated data with noise added
    """
    # Based on AtomConfig feature order in featurize.py
    feature_index = 0
    if len(mask) == 0:
        mask = utils.bitmask(data.x.shape, mask_ratio, device=data.x.device)
    if atom_config.element_type:
        data.x[
            :, feature_index : feature_index + len(featurize.element_names)
        ] = one_hot_noise(
            data.x[:, feature_index : feature_index + len(featurize.element_names)],
        )
        feature_index += len(featurize.element_names)
    if atom_config.degree:
        degree_len = 11
        data.x[:, feature_index : feature_index + degree_len] = one_hot_noise(
            data.x[:, feature_index : feature_index + degree_len],
        )
        feature_index += degree_len
    if atom_config.implicit_valence:
        valence_len = 7
        data.x[:, feature_index : feature_index + valence_len] = one_hot_noise(
            data.x[:, feature_index : feature_index + valence_len],
        )
        feature_index += valence_len
    if atom_config.formal_charge:
        charge_len = 1
        data.x[:, feature_index : feature_index + charge_len] = float_noise(
            data.x[:, feature_index : feature_index + charge_len],
            mask[:, feature_index : feature_index + charge_len],
        )
        feature_index += charge_len
    if atom_config.num_rad_e:
        rad_len = 1
        data.x[:, feature_index : feature_index + rad_len] = float_noise(
            data.x[:, feature_index : feature_index + rad_len],
            mask[:, feature_index : feature_index + rad_len],
        )
        feature_index += rad_len
    if atom_config.hybridization:
        if not atom_config.combo_hybrid:
            hybrid_len = 5
            data.x[:, feature_index : feature_index + hybrid_len] = one_hot_noise(
                data.x[:, feature_index : feature_index + hybrid_len],
            )
            feature_index += hybrid_len
        else:
            combo_hybrid_len = 4
            data.x[:, feature_index : feature_index + combo_hybrid_len] = one_hot_noise(
                data.x[:, feature_index : feature_index + combo_hybrid_len],
            )
            feature_index += combo_hybrid_len
    if atom_config.aromatic:
        aromatic_len = 1
        data.x[:, feature_index : feature_index + aromatic_len] = boolean_noise(
            data.x[:, feature_index : feature_index + aromatic_len],
            mask[:, feature_index : feature_index + aromatic_len],
        )
        feature_index += aromatic_len

    return data
