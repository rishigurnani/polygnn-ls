import torch


class contrast_loss(torch.nn.Module):
    """
    Contrastive loss, as seen in SimCLR.
    """

    def __init__(self):
        super().__init__()
        self.mse_fn = torch.nn.MSELoss()
        self.cos = torch.nn.CosineSimilarity()

    def forward(self, data, **kwargs):
        """
        Keyword args:
            data (pyg.data.Data): "data" should have an attribute "y" with shape (N, D, 2).
                It should also have an attribute named "temperature".
        """
        n, d = data.y.size()[0:-1]
        # Below we interleave data.y, forming a tensor with shape (2 * N, D).
        data.y = torch.stack((data.y[:, :, 0], data.y[:, :, 1]), dim=1).view(2 * n, d)
        # Below compute the pairwise cosine sim. Implementation borrowed
        # from https://discuss.pytorch.org/t/pairwise-cosine-distance/30961/14.
        result = self.cos(data.y[:, :, None], data.y.t()[None, :, :])  # (2N, 2N)
        indicator = (
            torch.ones(result.size()).fill_diagonal_(0).to(data.y.device)
        )  # (2N, 2N)
        result = torch.exp(result / data.temperature)
        result = -1 * (  # loss for each term, (2N, 2N)
            torch.log10(result / torch.sum(indicator * result))
        )
        idx = [
            torch.arange(0, n + 1, 2).to(data.y.device),
            torch.arange(1, n + 2, 2).to(data.y.device),
        ]
        result = result[idx].sum() / n
        return result
