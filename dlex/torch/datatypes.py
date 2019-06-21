from dataclasses import dataclass

import torch


@dataclass
class BatchItem:
    X: torch.Tensor
    Y: torch.Tensor


@dataclass
class Batch:
    X: torch.Tensor
    Y: torch.Tensor
    X_len: list = None
    Y_len: list = None

    def __getitem__(self, i: int) -> BatchItem:
        return BatchItem(
            X=self.X[i][:self.X_len[i]].cpu().detach().numpy() if self.X_len is not None else self.X[i].cpu().detach().numpy(),
            Y=self.Y[i][:self.Y_len[i]].cpu().detach().numpy() if self.Y_len is not None else self.Y[i].cpu().detach().numpy())

    def __len__(self):
        return self.X.shape[0]