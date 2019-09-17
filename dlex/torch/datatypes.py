from dataclasses import dataclass

import torch


@dataclass
class BatchItem:
    X: torch.Tensor
    Y: torch.Tensor


class Batch(dict):
    X: torch.Tensor
    Y: torch.Tensor
    X_len: list = None
    Y_len: list = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def item(self, i: int) -> BatchItem:
        return BatchItem(
            X=self.X[i][:self.X_len[i]].cpu().detach().numpy() if self.X_len is not None else self.X[i].cpu().detach().numpy(),
            Y=self.Y[i][:self.Y_len[i]].cpu().detach().numpy() if self.Y_len is not None else self.Y[i].cpu().detach().numpy())

    @property
    def batch_size(self):
        return self.X.shape[0]


@dataclass
class Datasets:
    def __init__(self, train=None, valid=None, test=None):
        self.train = train
        self.valid = valid
        self.test = test