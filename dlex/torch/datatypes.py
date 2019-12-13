from dataclasses import dataclass

import torch


@dataclass
class BatchItem:
    X: torch.Tensor
    Y: torch.Tensor


class Batch(dict):
    X: torch.Tensor
    Y: torch.Tensor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def item(self, i: int) -> BatchItem:
        try:
            X = self.X[i].cpu().detach().numpy()
        except Exception:
            X = None

        try:
            Y = self.Y[i].cpu().detach().numpy()
        except Exception:
            Y = None

        return BatchItem(X=X, Y=Y)

    @property
    def batch_size(self):
        return len(self)

    def __len__(self):
        return self.Y.shape[0]


@dataclass
class Datasets:
    def __init__(self, train=None, valid=None, test=None):
        self.train = train
        self.valid = valid
        self.test = test

    def load_dataset(self, builder, mode):
        if mode == "test":
            self.test = builder.get_pytorch_wrapper(mode)
        if mode in {"valid", "dev"}:
            self.valid = builder.get_pytorch_wrapper(mode)

    def get_dataset(self, mode):
        if mode == "test":
            return self.test
        elif mode in {"valid", "dev"}:
            return self.valid