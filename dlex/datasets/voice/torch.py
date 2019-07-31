import os
from typing import List

from dlex.datasets.seq2seq.torch import PytorchSeq2SeqDataset
from dlex.torch import BatchItem
from dlex.utils import logger


class PytorchVoiceDataset(PytorchSeq2SeqDataset):
    def load_data(self, csv_path):
        logger.info("Loading data...")
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')[1:]
        lines = [l.split('\t') for l in lines if l != ""]
        data = [{
            'X_path': os.path.join(self.builder.get_processed_data_dir(), self.params.dataset.feature.file_type, os.path.basename(l[0])),
            'Y': [int(w) for w in l[1].split(' ') if w != ""],
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
        return super().collate_fn(batch)

    def __getitem__(self, i: int):
        return self._data[i]