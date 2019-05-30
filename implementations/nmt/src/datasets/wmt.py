"""Datasets for neural machine translation"""

import os
import random

from dlex.datasets.base.nlp import get_token_id
from dlex.utils.logging import logger
from .nmt import NMTBaseDataset


class WMT14EnglishGerman(NMTBaseDataset):
    def __init__(self, mode, params):
        super().__init__(
            mode,
            params,
            vocab_paths={
                "en": os.path.join(self.get_raw_data_dir(), "vocab.50K.en"),
                "de": os.path.join(self.get_raw_data_dir(), "vocab.50K.de")
            }
        )

    def load_data(self):
        data_file_names = {
            "train": {"en": "train.en", "de": "train.de"},
            "test": {"en": "newstest2012.en", "de": "newstest2012.de"}
        }
        # Load data
        pairs = {}
        if self.mode in ["train", "test"]:
            data = []
            for lang in self.lang:
                pairs[lang] = open(
                    os.path.join(self.get_raw_data_dir(), data_file_names[self.mode][lang]), "r",
                    encoding='utf-8').read().split("\n")
            for src, tgt in zip(pairs[self.lang_src], pairs[self.lang_tgt]):
                data.append(dict(
                    X=[self.sos_token_id] + [get_token_id(self.word2index[self.lang_src], tkn) for tkn in
                                             src.split(' ')] + [self.eos_token_id],
                    Y=[self.sos_token_id] + [get_token_id(self.word2index[self.lang_tgt], tkn) for tkn in
                                             tgt.split(' ')] + [self.eos_token_id]
                ))
            data = list(filter(lambda it: len(it['X']) < 50, data))
            logger.debug("Data sample: %s", str(random.choice(data)))
            return data
        elif self.mode == "infer":
            return []

    @classmethod
    def maybe_download_and_extract(cls, force=False):
        super().maybe_download_and_extract(force)
        if not os.path.exists(cls.get_raw_data_dir()):
            for url in ["https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en",
                        "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de",
                        "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2012.en",
                        "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2012.de",
                        "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.en",
                        "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.de",
                        "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en",
                        "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de",
                        "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2015.en",
                        "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2015.de",
                        "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/vocab.50K.en",
                        "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/vocab.50K.de",
                        "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/dict.en-de"]:
                try:
                    cls.download_and_extract(url)
                except Exception as e:
                    logger.error("Failed to download %s" % url)
                    logger.error(str(e))

    @classmethod
    def maybe_preprocess(cls, force=False):
        super().maybe_preprocess(force)
