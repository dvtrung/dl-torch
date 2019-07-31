import os
from typing import List
import numpy as np

from dlex.configs import ModuleConfigs
from dlex.datasets.builder import DatasetBuilder
from dlex.datasets.nlp.utils import Vocab
from dlex.datasets.torch import PytorchDataset
from dlex.torch import Batch, BatchItem
from dlex.torch.utils.ops_utils import LongTensor
from dlex.utils import logger


class CommonVoiceLM(DatasetBuilder):
    def get_working_dir(self) -> str:
        return os.path.join(ModuleConfigs.DATASETS_PATH, "CommonVoice")

    @property
    def output_prefix(self):
        cfg = self.params.dataset
        return "%s_vocab_size_%d" % (
            cfg.unit,
            cfg.vocab_size or 0)

    def get_pytorch_wrapper(self, mode: str):
        return PytorchCommonVoiceLM(self, mode)


class PytorchCommonVoiceLM(PytorchDataset):
    def __init__(self, builder, mode: str):
        super().__init__(builder, mode)
        self.vocab = Vocab(os.path.join(builder.get_processed_data_dir(), "vocab", "%s.txt" % self.params.dataset.unit))
        self._data = self.load_data(os.path.join(
            builder.get_processed_data_dir(),
            "%s_%s" % (builder.output_prefix, mode) + '.csv'))

    def load_data(self, csv_path):
        logger.info("Loading data...")
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')[1:]
        lines = [l.split('\t') for l in lines if l != ""]
        sentences = [[int(w) for w in l[1].split(' ') if w != ""] for l in lines]
        data = []
        bptt = self.params.dataset.bptt_len
        for s in sentences:
            for start in range(len(s) - bptt): # range(max(len(s) - bptt, 1)):
                end = min(start + bptt, len(s) - 1)
                data.append(dict(X=s[start:end], Y=s[start + 1:end + 1]))
        logger.info("Finish loading data.")
        if self.params.dataset.sort:
            logger.info("Sorting data...")
            data.sort(key=lambda it: len(it))
        return data

    def collate_fn(self, batch: List[dict]):
        X = LongTensor([it['X'] for it in batch])
        Y = LongTensor([it['Y'] for it in batch])
        return Batch(X=X, Y=Y)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def __len__(self):
        return len(self._data)

    def format_output(self, y_pred, batch_item: BatchItem):
        pr = np.array(y_pred)
        src = batch_item.X
        gt = batch_item.Y
        if self.params.dataset.output_format is None:
            return str(src), str(gt), str(pr)
        elif self.params.dataset.output_format == "text":
            delimiter = ' ' if self.params.dataset.unit == "word" else ''
            return \
                delimiter.join(self.vocab.decode_idx_list(batch_item.X)), \
                delimiter.join(self.vocab.decode_idx_list(batch_item.Y)), \
                delimiter.join(self.vocab.decode_idx_list(y_pred))