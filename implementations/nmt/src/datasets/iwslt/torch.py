import os
import random

from dlex.datasets.nlp.utils import Vocab
from dlex.datasets.voice.torch import PytorchSeq2SeqDataset
from dlex.utils.logging import logger
from dlex.torch import BatchItem


class IWSLT15EnglishVietnamese(PytorchSeq2SeqDataset):
    def __init__(self, builder, mode, params):
        super().__init__(
            builder, mode, params,
            vocab_path=os.path.join(builder.get_raw_data_dir(), "vocab.en"))

        self._src_vocab = Vocab(os.path.join(builder.get_raw_data_dir(), "vocab.vi"))
        self._data = self._load_data()

    @property
    def input_size(self):
        return len(self._src_vocab)

    def _load_data(self):
        data_file_names = {
            "train": {"en": "train.en", "vi": "train.vi"},
            "test": {"en": "tst2012.en", "vi": "tst2012.vi"}
        }
        # Load data
        if self._mode in ["train", "test"]:
            data = []
            src_data = open(
                os.path.join(self.builder.get_raw_data_dir(), data_file_names[self._mode]['vi']), "r",
                encoding='utf-8').read().split("\n")
            tgt_data = open(
                os.path.join(self.builder.get_raw_data_dir(), data_file_names[self._mode]['en']), "r",
                encoding='utf-8').read().split("\n")
            for src, tgt in zip(src_data, tgt_data):
                data.append(BatchItem(
                    X=[self._src_vocab.get_token_id(tkn) for tkn in src.split(' ')],
                    Y=[self._vocab.get_token_id(tkn) for tkn in tgt.split(' ')]
                ))
            data = list(filter(lambda it: len(it.X) < 50, data))
            logger.debug("Data sample: %s", str(random.choice(data)))
            return data
        elif self._mode == "infer":
            return []
