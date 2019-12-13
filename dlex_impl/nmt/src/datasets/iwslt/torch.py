import json
import os
import random

from dlex.datasets.seq2seq.torch import PytorchTranslationDataset
from dlex.torch import BatchItem
from dlex.utils.logging import logger


class IWSLT15EnglishVietnamese(PytorchTranslationDataset):
    def __init__(self, builder, mode):
        super().__init__(
            builder, mode,
            src_vocab_path=os.path.join(builder.get_raw_data_dir(), f"vocab.{builder.params.dataset.source}"),
            tgt_vocab_path=os.path.join(builder.get_raw_data_dir(), f"vocab.{builder.params.dataset.target}"))
        self._data = self._load_data()

    def _load_data(self):
        data_file_names = {
            "train": {"en": "train.en", "vi": "train.vi"},
            "test": {"en": "tst2012.en", "vi": "tst2012.vi"}
        }
        # Load data
        if self.mode in ["train", "test"]:
            data = []
            src_data = open(
                os.path.join(self.builder.get_raw_data_dir(), data_file_names[self.mode][self.params.dataset.source]), "r",
                encoding='utf-8').read().split("\n")
            tgt_data = open(
                os.path.join(self.builder.get_raw_data_dir(), data_file_names[self.mode][self.params.dataset.target]), "r",
                encoding='utf-8').read().split("\n")
            for src, tgt in zip(src_data, tgt_data):
                X = [self.src_vocab.get_token_id(tkn) for tkn in src.split(' ')]
                Y = [self.vocab.get_token_id(tkn) for tkn in tgt.split(' ')]
                if self.params.dataset.max_source_length and len(X) > self.params.dataset.max_source_length or \
                        self.params.dataset.max_target_length and len(Y) > self.params.dataset.max_target_length:
                    continue
                data.append(BatchItem(X=X, Y=Y))
            logger.debug("Data sample: %s", str(random.choice(data)))
            return data
        elif self.mode == "infer":
            return []