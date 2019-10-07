from typing import List

import numpy as np
import torch
from torch import nn

from dlex.datasets.builder import DatasetBuilder
from dlex.datasets.nlp.utils import Vocab
from dlex.datasets.torch import PytorchDataset
from dlex.torch import Batch
from dlex.torch import BatchItem
from dlex.torch.utils.ops_utils import maybe_cuda


class PytorchSeq2SeqDataset(PytorchDataset):
    def __init__(
            self,
            builder: DatasetBuilder,
            mode: str,
            vocab_path: str = None):
        super().__init__(builder, mode)
        if vocab_path:
            self.vocab = Vocab(vocab_path)
        for token in self.params.dataset.special_tokens:
            self.vocab.add_token("<%s>" % token)
        self._output_size = len(self.vocab)

    @property
    def pad_token_idx(self):
        if 'blank' in self.params.dataset.special_tokens:
            return self.blank_token_idx
        if 'pad' in self.params.dataset.special_tokens:
            return self.vocab.blank_token_idx
        else:
            raise Exception("No padding idx.")

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def sos_token_idx(self) -> int:
        return self.vocab.sos_token_idx

    @property
    def eos_token_idx(self) -> int:
        return self.vocab.eos_token_idx

    @property
    def blank_token_idx(self):
        return self.vocab.blank_token_idx

    def collate_fn(self, batch: List[BatchItem]):
        batch.sort(key=lambda item: len(item.X), reverse=True)

        if self.params.dataset.max_source_length is not None:
            batch = [item for item in batch if self.params.dataset.max_source_length > len(item.X) > 0]
        if self.params.dataset.max_target_length is not None:
            batch = [item for item in batch if 0 < len(item.Y) < self.params.dataset.max_target_length + 2]

        if len(batch) == 0:
            return None

        if isinstance(batch[0].X[0], int):
            inp = [torch.LongTensor(item.X) for item in batch]
        else:
            inp = [torch.FloatTensor(item.X) for item in batch]

        if 'sos' in self.params.dataset.special_tokens:
            tgt = [torch.LongTensor([self.sos_token_idx] + item.Y + [self.eos_token_idx]).view(-1) for item in batch]
        else:
            tgt = [torch.LongTensor(item.Y).view(-1) for item in batch]
            if self.params.dataset.max_target_length is not None:
                tgt = [y[:min(len(y), self.params.dataset.max_target_length)] for y in tgt]

        tgt_len = [len(t) for t in tgt]
        inp = nn.utils.rnn.pad_sequence(
            inp, batch_first=True)
        tgt = nn.utils.rnn.pad_sequence(
            tgt, batch_first=True,
            padding_value=self.pad_token_idx)

        return Batch(
            X=maybe_cuda(inp), X_len=[len(item.X) for item in batch],
            Y=maybe_cuda(tgt), Y_len=tgt_len)

    def format_output(self, y_pred, batch_item: BatchItem):
        pr = np.array(y_pred)
        gt = batch_item.Y[1:-1] if 'sos' in self.params.dataset.special_tokens else batch_item.Y
        if self.params.dataset.output_format is None:
            return "", str(gt), str(pr)
        elif self.params.dataset.output_format == "text":
            delimiter = ' ' if self.params.dataset.unit == "word" else ''
            return \
                "", \
                delimiter.join([self.vocab.get_token(wid) for wid in gt]), \
                delimiter.join([self.vocab.get_token(wid) for wid in pr])