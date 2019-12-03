from collections import namedtuple
from dataclasses import dataclass

import torch
from dlex.datasets.nlp.torch import NLPDataset
from dlex.torch.datatypes import VariableLengthTensor, Batch, BatchItem


class QADataset(NLPDataset):
    def __init__(self, builder, mode: str):
        super().__init__(builder, mode)


@dataclass
class BatchX:
    context_word: VariableLengthTensor
    context_char: torch.Tensor
    question_word: VariableLengthTensor
    question_char: torch.Tensor


BatchY = namedtuple("BatchY", "answer_span")


class QABatch(Batch):
    X: BatchX
    Y: BatchY
    Y_len: BatchY

    def __len__(self):
        return len(self.X.context_word)

    def item(self, i: int) -> BatchItem:
        return BatchItem(
            X=None,
            Y=self.Y[i].cpu().detach().numpy())