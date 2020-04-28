from collections import namedtuple
from dataclasses import dataclass
from typing import Union, List

import torch


@dataclass
class BatchItem:
    id: Union[str, int]
    X: torch.Tensor
    Y: torch.Tensor


class Batch(dict):
    ids: List[Union[str, int]]
    X: torch.Tensor
    Y: torch.Tensor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        if 'ids' not in kwargs:
            self.ids = None

    def item(self, i: int) -> BatchItem:
        try:
            if type(self.X) == tuple:
                X = [it[i].cpu().detach().numpy() for it in self.X]
            elif isinstance(self.X, namedtuple):
                X = self.X.__class__(**{f: getattr(self.X, f)[i] for f in self.X._fields})
            else:
                X = self.X[i].cpu().detach().numpy()
        except Exception:
            X = None

        try:
            Y = self.Y[i].cpu().detach().numpy()
        except Exception:
            Y = None

        return BatchItem(id=self.ids[i] if self.ids else None, X=X, Y=Y)

    @property
    def batch_size(self):
        return len(self)

    def __len__(self):
        return self.Y.shape[0]