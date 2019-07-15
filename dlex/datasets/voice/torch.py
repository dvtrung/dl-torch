import os
from typing import List

import torch.nn as nn
from torch import FloatTensor, LongTensor

from dlex.datasets.torch import PytorchSeq2SeqDataset
from dlex.torch import Batch
from dlex.torch import BatchItem
from dlex.torch.utils.ops_utils import maybe_cuda
from dlex.utils import logger


class PytorchVoiceDataset(PytorchSeq2SeqDataset):
    def load_data(self, csv_path):
        logger.info("Loading data...")
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')[1:]
        lines = [l.split('\t') for l in lines if l != ""]
        data = [{
            'X_path': os.path.join(self.builder.get_processed_data_dir(), "htk", os.path.basename(l[0])),
            'Y': [int(w) for w in l[1].split(' ')],
        } for l in lines]
        logger.info("Finish loading data.")
        if self.params.dataset.sort:
            logger.info("Sorting data...")
            data.sort(key=lambda it: len(it['Y']))
        return data

    def collate_fn(self, batch: List[dict]):
        batch = [BatchItem(
            X=self.builder.load_feature(item['X_path']),
            Y=item['Y']) for item in batch]
        batch = [item for item in batch if item.X is not None]
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
            X=maybe_cuda(inp), X_len=[len(item.X) for item in batch],
            Y=maybe_cuda(tgt), Y_len=tgt_len)

    def __getitem__(self, i: int):
        return self._data[i]