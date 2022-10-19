import torch

def bitmask(shape, mask_ratio):
    """
    Return a bit mask.

    Keyword arguments:
        shape (tuple): The shape of the output.
        mask_ratio (float): The approximate percentage of ones in the output.
    """
    return torch.FloatTensor(*shape).uniform_() > (1 - mask_ratio)