import torch


class contrast_loss(torch.nn.Module):
    """
    Contrastive loss, as seen in SimCLR.
    """

    def __init__(self, temperature):
        super().__init__()
        self.cos = torch.nn.CosineSimilarity()
        self.temperature = torch.tensor([temperature], requires_grad=False).squeeze()

    def forward(self, data, **kwargs):
        """
        Keyword args:
            data (pyg.data.Data): "data" should have an attribute "y" with shape (N, D, 2).
                It should also have an attribute named "temperature".
        """
        temperature = self.temperature.to(data.y.device)
        n, d = data.y.size()
        n = n // 2
        # Below compute the pairwise cosine sim. Implementation borrowed
        # from https://discuss.pytorch.org/t/pairwise-cosine-distance/30961/14.
        result = self.cos(data.y[:, :, None], data.y.t()[None, :, :])  # (2N, 2N)
        result = result / temperature  # scale similarities by temperature
        result = torch.exp(result)
        # Now we compute the denominator (as seen in Eq. 1 of the SimCLR paper)
        # of our loss terms. The first element of `denominator` is the
        # denominator of l_0i terms. The second element is the denominator of
        # the l_1i terms, etc.
        denominator = (
            result
            * (
                torch.eye(2 * n, device=data.y.device)
                == 0.0  # In this matrix, all diagonal elements are 0 and all non-diagonals are 1.
            )
        ).sum(dim=0)
        # At this point, `result` contains the numerators of the 2N loss
        # terms in Eq.1. We need to select these 2N terms from `result`.
        # One numerator term is contained in each *row* of `result`, so
        # we can use `torch.arange` to compute the row indices.
        row_idx = torch.arange(0, n * 2, device=data.y.device)
        # The column indices are a bit more complicated. For example, if
        # N = 2, then `col_idx` is [1, 0, 3, 2]. This jumbled form
        # is a consequence of `data.y` being interleaved.
        col_idx = torch.stack(
            (
                torch.arange(1, 1 + n * 2, 2, device=data.y.device),  # 1, 3, 5, ...
                torch.arange(0, n * 2, 2, device=data.y.device),  # 0, 2, 4, ...
            ),
            dim=1,
        ).view(n * 2)
        result = result[row_idx, col_idx]  # The numerator terms.
        result = -torch.log(result / denominator)  # The 2N loss terms.
        return result.mean()  # The mean of the loss terms.
