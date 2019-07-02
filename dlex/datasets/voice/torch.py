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
        for token in params.dataset.special_tokens:
            self._vocab.add_token("<%s>" % token)
        self._output_size = len(self._vocab)

    @property
    def pad_token_idx(self):
        if 'blank' in self.params.dataset.special_tokens:
            return self.blank_token_idx
        if 'pad' in self.params.dataset.special_tokens:
            return self._vocab.blank_token_idx
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
        if isinstance(batch[0].X[0], int):
            inp = [LongTensor(item.X) for item in batch]
        else:
            inp = [FloatTensor(item.X) for item in batch]

        if 'sos' in self.params.dataset.special_tokens:
            tgt = [LongTensor([self.sos_token_idx] + item.Y + [self.eos_token_idx]).view(-1) for item in batch]
        else:
            tgt = [LongTensor(item.Y).view(-1) for item in batch]
        tgt_len = [len(t) for t in tgt]
        inp = nn.utils.rnn.pad_sequence(
            inp, batch_first=True)
        tgt = nn.utils.rnn.pad_sequence(
            tgt, batch_first=True,
            padding_value=self.pad_token_idx)

        return Batch(
            X=inp, X_len=LongTensor([len(item.X) for item in batch]),
            Y=tgt, Y_len=LongTensor(tgt_len))

    def evaluate_batch(self, y_pred, batch: Batch, metric: str) -> (int, int):
        score, count = 0, 0
        for i, pr in enumerate(y_pred):
            ref = batch.item(i).Y[1:-1] if 'sos' in self.params.dataset.special_tokens else batch.item(i).Y
            s, c = self._builder.evaluate(np.array(pr), ref, metric)
            score += s
            count += c
        return score, count

    def format_output(self, y_pred, batch_item: BatchItem):
        pr = np.array(y_pred)
        gt = batch_item.Y[1:-1] if 'sos' in self.params.dataset.special_tokens else batch_item.Y
        if self.params.dataset.output_format is None:
            return "", str(gt), str(pr)
        elif self.params.dataset.output_format == "text":
            delimiter = ' ' if self.params.dataset.unit == "word" else ''
            return \
                "", \
                delimiter.join([self._vocab.get_token(wid) for wid in gt]), \
                delimiter.join([self._vocab.get_token(wid) for wid in pr])


class PytorchVoiceDataset(PytorchSeq2SeqDataset):
    def load_data(self, csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')[1:]
        lines = [l.split('\t') for l in lines if l != ""]
        data = [{
            'X_path': l[0],
            'Y': [int(w) for w in l[1].split(' ')],
        } for l in lines]
        if self.params.dataset.sort:
            data.sort(key=lambda it: len(it))
        return data

    def __getitem__(self, i: int):
        item = self._data[i]
        X = self._builder.load_feature(item['X_path'])
        return BatchItem(X=X, Y=item['Y'])