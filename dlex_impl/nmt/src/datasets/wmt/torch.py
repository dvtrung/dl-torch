import os
import random

from dlex.datasets.nlp.utils import Vocab
from dlex.datasets.seq2seq.torch import PytorchTranslationDataset
from dlex.utils.logging import logger
from dlex.torch import BatchItem


class PytorchWMT14EnglishGerman(PytorchTranslationDataset):
    def __init__(self, builder, mode):
        super().__init__(
            builder, mode,
            src_vocab_path=os.path.join(builder.get_raw_data_dir(), f"vocab.50K.{builder.params.dataset.source}"),
            tgt_vocab_path=os.path.join(builder.get_raw_data_dir(), f"vocab.50K.{builder.params.dataset.target}")
        )
        self._data = self._load_data()

    def _load_data(self):
        data_file_names = {
            "train": {"en": "train.en", "de": "train.de"},
            "test": {"en": "newstest2012.en", "de": "newstest2012.de"}
        }
        # Load data
        if self.mode in ["train", "test"]:
            data = []
            src_data = open(
                os.path.join(self.builder.get_raw_data_dir(), data_file_names[self.mode][self.params.dataset.source]),
                "r",
                encoding='utf-8').read().split("\n")
            tgt_data = open(
                os.path.join(self.builder.get_raw_data_dir(), data_file_names[self.mode][self.params.dataset.target]),
                "r",
                encoding='utf-8').read().split("\n")
            for src, tgt in zip(src_data, tgt_data):
                data.append(BatchItem(
                    X=[self.src_vocab.get_token_id(tkn) for tkn in src.split(' ')],
                    Y=[self.vocab.get_token_id(tkn) for tkn in tgt.split(' ')]
                ))
            data = list(filter(lambda it: len(it.X) < 50, data))
            logger.debug("Data sample: %s", str(random.choice(data)))
            return data
        elif self.mode == "infer":
            return []
