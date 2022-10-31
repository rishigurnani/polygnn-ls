import torch


def bitmask(shape, mask_ratio, device="cpu"):
    """
    Return a bit mask.

    Keyword arguments:
        shape (tuple): The shape of the output.
        mask_ratio (float): The approximate percentage of ones in the output.
    """
    return torch.FloatTensor(*shape, device=device).uniform_() > (1 - mask_ratio)
