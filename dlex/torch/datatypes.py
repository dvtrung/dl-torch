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
            if type(self.X) == tuple:
                X = [it[i].cpu().detach().numpy() for it in self.X]
            else:
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