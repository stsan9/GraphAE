import torch

class Standardizer:
    """
    Pytorch implementation of scikit-learn's StandardScaler
    """
    def __init__(self, mean=None, std=None, device=torch.device('cuda:0')):
        self.mean = mean
        self.std = std
        self.device = device
        if mean != None and not isinstance(self.mean, torch.Tensor):
            self.mean = torch.tensor(mean).to(device)
        if std != None and not isinstance(self.std, torch.Tensor):
            self.std = torch.tensor(std).to(device)

    def fit(self, data):
        """
        :param data: torch tensor
        """
        self.mean = torch.mean(data, dim=0).to(self.device)
        self.std = torch.std(data, dim=0).to(self.device)

    def transform(self, data):
        assert self.mean != None, "mean is None"
        assert self.std != None, "std is None"

        return (data - self.mean) / self.std

    def inverse_transform(self, data, log_pt=False):
        """
        :param data: torch tensor
        :param log_pt: undo log transformation on pt
        """
        assert self.mean != None, "mean is None"
        assert self.std != None, "std is None"

        inverse = (data * self.std) + self.mean
        if log_pt:
            inverse[:,0] = (10 ** inverse[:,0]) - 1
        return inverse
