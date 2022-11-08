import torch


def _veil(tens, row_idx):
    """
    Replace values with -1.

    Keyword arguments:
        tens (torch.tensor): A 2D tensor to manipulate.
        row_idx (torch.tensor): The row indices corresponding to the
            values in `tens` that should be changed.
    """
    tens[row_idx, :] = -1
    return tens


def veil_nodes(data, frac):
    """
    Randomly replace all the features of some nodes in `data` with -1.

    Keyword arguments:
        data: A batch of graphs.
        frac: The veil fraction.
    """
    nrow, ncol = data.x.size()
    # Determine the number of row indices to veil.
    nveil = round(nrow * frac)
    # Sample row indices to veil.
    row_idx = torch.randperm(nrow)[:nveil]
    # Veil `data.x``.
    data.x = _veil(data.x, row_idx)
    return data
