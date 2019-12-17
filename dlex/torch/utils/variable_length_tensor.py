from typing import List, Any, Union

import torch
from torch import LongTensor
import numpy as np

from dlex.torch.utils.ops_utils import maybe_cuda


def pad_sequence(data: List[List[Any]], padding_value, output_tensor=False):
    max_len = max([len(seq) for seq in data])

    i = 0
    while len(data[i]) == 0:
        i += 1
        if i == len(data):
            raise ValueError("Empty input.")
    if isinstance(data[i][0], list) or isinstance(data[i][0], tuple):
        padding_value = [padding_value for _ in range(len(data[i][0]))]

    if not output_tensor:
        data = [torch.tensor(seq + [padding_value] * (max_len - len(seq))) for seq in data]
        lengths = [max(len(seq), 1) for seq in data]
        return data, lengths
    else:
        data = [seq + [padding_value] * (max_len - len(seq)) for seq in data]
        lengths = [max(len(seq), 1) for seq in data]
        return maybe_cuda(torch.tensor(data)), maybe_cuda(LongTensor(lengths))


def get_mask(
        lengths: Union[List[int], LongTensor],
        max_len: int = None,
        masked_value=True,
        unmasked_value=False,
        device=None) -> torch.Tensor:
    """Get mask tensor

    :param device:
    :param unmasked_value:
    :param masked_value:
    :param lengths:
    :param max_len: if None, max of lengths is used
    :type max_len: int
    :param dtype:
    :type dtype: str
    :return:
    """
    if isinstance(lengths, list):
        lengths = maybe_cuda(LongTensor(lengths))
    if isinstance(lengths, torch.Tensor):
        if not device:
            device = lengths.device

    assert len(lengths.shape) == 1, 'Length shape should be 1 dimensional.'
    assert masked_value != unmasked_value

    max_len = max_len or torch.max(lengths).item()
    mask = torch.arange(
        max_len, device=lengths.device,
        dtype=lengths.dtype).expand(len(lengths), max_len) < lengths.unsqueeze(1)

    if isinstance(masked_value, bool):
        return mask if masked_value else ~mask
    else:
        mask = masked_value * mask.int() + unmasked_value * (~mask.int())
    return mask if not device else mask.cuda(device)