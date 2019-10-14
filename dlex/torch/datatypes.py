from dataclasses import dataclass

import torch

from dlex.torch.utils.ops_utils import maybe_cuda


@dataclass
class BatchItem:
    X: torch.Tensor
    Y: torch.Tensor


class VariableLengthTensor:
    def __init__(self, values, padding_value):
        super().__init__()
        max_len = max([len(seq) for seq in values])
        self.data = torch.tensor([seq + [padding_value] * (max_len - len(seq)) for seq in values])
        self.padding_value = padding_value
        self.lengths = [len(seq) for seq in values]

    def cuda(self, device=None, non_blocking=False):
        self.data = self.data.cuda(device, non_blocking)
        return self

    def __len__(self):
        return len(self.lengths)

    def get_mask(self, max_len: int = None, dtype: str = None):
        """Get mask tensor

        :param max_len: if None, max of lengths is used
        :type max_len: int
        :param dtype:
        :type dtype: str
        :return:
        """
        lengths = torch.LongTensor(self.lengths)

        assert len(lengths.shape) == 1, 'Length shape should be 1 dimensional.'
        max_len = max_len or lengths.max().item()
        mask = torch.arange(
            max_len, device=lengths.device,
            dtype=lengths.dtype).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        if dtype is not None:
            mask = torch.as_tensor(mask, dtype=dtype, device=lengths.device)
        return maybe_cuda(mask)

    def get_padded_tensor(self):
        return self.data


class Batch(dict):
    X: torch.Tensor
    Y: torch.Tensor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def item(self, i: int) -> BatchItem:
        return BatchItem(
            X=self.X[i].cpu().detach().numpy(),
            Y=self.Y[i].cpu().detach().numpy())

    @property
    def batch_size(self):
        return len(self)

    def __len__(self):
        return self.Y.shape[0]


class VariableLengthInputBatch(Batch):
    X: torch.Tensor
    Y: torch.Tensor
    X_len: list = None

    def item(self, i: int) -> BatchItem:
        return BatchItem(
            X=self.X[i][:self.X_len[i]].cpu().detach().numpy(),
            Y=self.Y[i].cpu().detach().numpy())


class VariableLengthBatch(Batch):
    X: VariableLengthTensor
    Y: VariableLengthTensor

    def item(self, i: int) -> BatchItem:
        return BatchItem(
            X=self.X[i][:self.X_len[i]].cpu().detach().numpy(),
            Y=self.Y[i][:self.Y_len[i]].cpu().detach().numpy())


@dataclass
class Datasets:
    def __init__(self, train=None, valid=None, test=None):
        self.train = train
        self.valid = valid
        self.test = test