from typing import List

import torch.nn as nn
import numpy as np

from dlex.datasets.nlp.utils import Vocab
from dlex.datasets.builder import DatasetBuilder
from dlex.datasets.torch import PytorchDataset
from dlex.torch import Batch
from dlex.torch.utils.ops_utils import FloatTensor, LongTensor
from dlex.torch import BatchItem


class PytorchSeq2SeqDataset(PytorchDataset):
    def __init__(
            self,
            builder: DatasetBuilder,
            mode: str,
            params,
            vocab_path: str = None):
        super().__init__(builder, mode, params)
        self._vocab = Vocab(vocab_path)
        if params.dataset.vocab.sos_eos:
            self._vocab.add_token('<sos>')
            self._vocab.add_token('<eos>')
        if params.dataset.vocab.blank:
            self._vocab.add_token('<blank>')
        self._output_size = len(self._vocab)

    @property
    def padding_idx(self):
        if self._params.dataset.vocab.blank:
            return self.blank_token_idx
        elif self._params.dataset.vocab.sos_eos:
            return self.eos_token_idx
        else:
            raise Exception("No padding idx.")

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def sos_token_idx(self) -> int:
        return self._vocab.sos_token_idx

    @property
    def eos_token_idx(self) -> int:
        return self._vocab.eos_token_idx

    @property
    def blank_token_idx(self):
        return self._vocab.blank_token_idx

    def collate_fn(self, batch: List[BatchItem]):
        batch.sort(key=lambda item: len(item.X), reverse=True)
        inp = [FloatTensor(item.X) for item in batch]
        if self._params.dataset.vocab.sos_eos:
            tgt = [LongTensor([self.sos_token_idx] + item.Y + [self.eos_token_idx]).view(-1) for item in batch]
        else:
            tgt = [LongTensor(item.Y).view(-1) for item in batch]
        tgt_len = [len(t) for t in tgt]
        inp = nn.utils.rnn.pad_sequence(
            inp, batch_first=True)
        tgt = nn.utils.rnn.pad_sequence(
            tgt, batch_first=True,
            padding_value=self.padding_idx)

        return Batch(
            X=inp, X_len=LongTensor([len(item.X) for item in batch]),
            Y=tgt, Y_len=LongTensor(tgt_len))

    def evaluate_batch(self, y_pred, batch: Batch, metric: str) -> (int, int):
        score, count = 0, 0
        for pr, batch_item in zip(y_pred, batch):
            ref = batch_item.Y[1:-1] if self._params.dataset.vocab.sos_eos else batch_item.Y
            s, c = self._builder.evaluate(np.array(pr), ref, metric)
            score += s
            count += c
        return score, count

    def format_output(self, y_pred, batch_item: BatchItem):
        pr = np.array(y_pred)
        gt = batch_item.Y[1:-1] if self._params.dataset.vocab.sos_eos else batch_item.Y
        if self._params.dataset.output_format is None:
            return "", str(gt), str(pr)
        elif self._params.dataset.output_format == "text":
            delimiter = ' ' if self._params.dataset.unit == "word" else ''
            return \
                "", \
                delimiter.join([self._vocab.get_token(wid) for wid in gt]), \
                delimiter.join([self._vocab.get_token(wid) for wid in pr])


class PytorchVoiceDataset(PytorchSeq2SeqDataset):
    def __getitem__(self, i: int):
        item = self._data[i]
        X = np.load(item['X_path'])
        return BatchItem(X=X, Y=item['Y'])