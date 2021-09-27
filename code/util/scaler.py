import torch

class Standardizer:
    """
    Pytorch implementation of scikit-learn's StandardScaler
    """
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, data):
        """
        :param data: torch tensor
        """
        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0)

    def transform(self, data):
        assert self.mean != None, "mean is None"
        assert self.std != None, "std is None"
        if data.device != 'cpu':
            mean = self.mean.to(data.device)
            std = self.std.to(data.device)
            return (data - mean) / std
        return (data - self.mean) / self.std

    def inverse_transform(self, data, log_pt=False):
        """
        :param data: torch tensor
        :param log_pt: undo log transformation on pt
        """
        assert self.mean != None, "mean is None"
        assert self.std != None, "std is None"
        if data.device != 'cpu':
            mean = self.mean.to(data.device)
            std = self.std.to(data.device)
        else:
            mean = self.mean
            std = self.std

        inverse = (data * std) + mean
        if log_pt:
            inverse[:,0] = (10 ** inverse[:,0]) - 1
        return inverse
