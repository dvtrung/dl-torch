from typing import List, Any, Union

import torch
from dlex.torch.utils.ops_utils import maybe_cuda
from torch import LongTensor


def pad_sequence(
        data: List[List[Any]],
        padding_value,
        max_len: int = None,
        output_tensor: bool = False,
        dim: int = 2):
    if dim == 3:
        values = []
        lengths = []
        for i in range(len(data)):
            val_, length_ = pad_sequence(data[i], padding_value, output_tensor=output_tensor, dim=2)
            values.append(val_)
            lengths.append(length_)

        max_len = max(val.shape[1] for val in values)
        values = [torch.cat([val, torch.zeros([val.shape[0], max_len - val.shape[1], val.shape[2]])], 1) for val in values]
        return torch.stack(values), torch.stack(lengths)
    else:
        max_len = max_len or max([len(seq) for seq in data])

        i = 0
        while len(data[i]) == 0:
            i += 1
            if i == len(data):
                raise ValueError("Empty input.")
        if isinstance(data[i][0], list) or isinstance(data[i][0], tuple):
            padding_value = [padding_value for _ in range(len(data[i][0]))]

        if not output_tensor:
            lengths = [max(len(seq), 1) for seq in data]
            if type(data[0]) == list:
                data = [torch.tensor(seq + [padding_value] * (max_len - len(seq))) for seq in data]
            else:
                data = [torch.cat([seq, torch.tensor([padding_value] * (max_len - len(seq)))]) for seq in data]
            return data, lengths
        else:
            lengths = [max(len(seq), 1) for seq in data]
            data = [seq + [padding_value] * (max_len - len(seq)) for seq in data]
            return torch.tensor(data), LongTensor(lengths)


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