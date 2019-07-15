import abc
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate, DataLoader

from dlex.configs import AttrDict
from dlex.datasets.builder import DatasetBuilder
from dlex.datasets.nlp.utils import Vocab
from dlex.torch import Batch
from dlex.torch import BatchItem


class PytorchDataset(Dataset):
    def __init__(self, builder, mode: str, params: AttrDict):
        self.params = params
        self.mode = mode
        self.builder = builder
        self._data = []
        self._sampler = None

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]

    @property
    def data(self):
        return self._data

    @property
    def configs(self):
        return self.params.dataset

    @abc.abstractmethod
    def evaluate_batch(self, y_pred, batch: Batch, metric: str):
        score, total = 0, 0
        for _target, _y_pred in zip(batch.Y, y_pred):
            s, t = self.builder.evaluate(_target.cpu().detach().numpy().tolist(), _y_pred, metric)
            score += s
            total += t
        return score, total

    def format_output(self, y_pred, batch_input) -> (str, str, str):
        return self.builder.format_output(y_pred, batch_input)

    def collate_fn(self, batch):
        return default_collate(batch)

    def get_iter(self, batch_size, start=0, end=-1):
        return DataLoader(
            # some datasets don't support slicing
            self[start:end] if start != 0 or (end != -1 and end != len(self)) else self,
            batch_size=batch_size,
            collate_fn=self.collate_fn, sampler=self._sampler,
            num_workers=self.params.num_workers)


class PytorchSeq2SeqDataset(PytorchDataset):
    def __init__(
            self,
            builder: DatasetBuilder,
            mode: str,
            params,
            vocab_path: str = None):
        super().__init__(builder, mode, params)
        self.vocab = Vocab(vocab_path)
        for token in params.dataset.special_tokens:
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

        if isinstance(batch[0].X[0], int):
            inp = [torch.LongTensor(item.X) for item in batch]
        else:
            inp = [torch.FloatTensor(item.X) for item in batch]
        if self.params.dataset.max_source_length is not None:
            inp = [x[:min(len(x), self.params.dataset.max_source_length)] for x in inp]

        if 'sos' in self.params.dataset.special_tokens:
            tgt = [torch.LongTensor([self.sos_token_idx] + item.Y + [self.eos_token_idx]).view(-1) for item in batch]
            if self.params.dataset.max_target_length is not None:
                tgt = [y[:min(len(y), self.params.dataset.max_target_length + 2)] for y in tgt]
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
            X=inp, X_len=[len(item.X) for item in batch],
            Y=tgt, Y_len=tgt_len)

    def evaluate_batch(self, y_pred, batch: Batch, metric: str) -> (int, int):
        score, count = 0, 0
        for i, pr in enumerate(y_pred):
            ref = batch.item(i).Y[1:-1] if 'sos' in self.params.dataset.special_tokens else batch.item(i).Y
            s, c = self.builder.evaluate(np.array(pr), ref, metric)
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
                delimiter.join([self.vocab.get_token(wid) for wid in gt]), \
                delimiter.join([self.vocab.get_token(wid) for wid in pr])