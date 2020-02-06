"""Operators utils"""
from typing import List, Union

import torch

CUDA = torch.cuda.is_available()

LongTensor = torch.cuda.LongTensor if CUDA else torch.LongTensor
FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor


def Tensor(*args):
    x = torch.Tensor(args)
    return x.cuda(device=gpus[0]) if CUDA else x


def maybe_cuda(x: Union[List, torch.Tensor]) -> torch.Tensor:
    return x.cuda() if CUDA else x


class VariableSizeTensor:
    def __init__(self, values, padding_value=0):
        self.values = values
        self.padding_value = padding_value
        self.lengths = [len(v) for v in values]

    def get_packed_sequence(self):
        return torch.nn.utils.rnn.pad_packed_sequence(self.values, padding_value=self.padding_value)