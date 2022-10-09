import torch


class contrast_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_fn = torch.nn.MSELoss()
        self.temp_min = torch.tensor(0.01)  # temperature minimum
        self.cos = torch.nn.CosineSimilarity()

    def forward(self, data, **kwargs):
        """
        A PyTorch geometric data object.

        Keyword args:
            data (Data): Should have an attribute "y" with shape (N, D, 2)
        """
        data.temperature = torch.max(data.temperature, self.temp_min)
        n, d = data.y.size()[0:-1]
        # Below we interleave data.y, forming a tensor with shape (2 * N, D).
        data.y = torch.stack(
            (data.y[:, :, 0], data.y[:, :, 1]), dim=1
        ).view(2 * n, d)
        # Below compute the pairwise cosine sim. Implementation borrowed
        # from https://discuss.pytorch.org/t/pairwise-cosine-distance/30961/14.
        result = self.cos(data.y[:, :, None], data.y.t()[None, :, :])  # (2N, 2N)
        indicator = torch.ones(result.size()).fill_diagonal_(0)  # (2N, 2N)
        result = torch.exp(result / data.temperature)
        result = -1 * (  # loss for each term, (2N, 2N)
            torch.log10(result / torch.sum(indicator * result))
        )
        idx = [torch.arange(0, n + 1, 2), torch.arange(1, n + 2, 2)]
        result = result[idx].sum() / n
        return result
