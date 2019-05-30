import os
import random

from dlex.utils.logging import logger
from .nmt import NMTBaseDataset


class IWSLT15EnglishVietnamese(NMTBaseDataset):
    def __init__(self, mode, params):
        super().__init__(mode, params, vocab_paths={
            "en": os.path.join(self.get_raw_data_dir(), "vocab.en"),
            "vi": os.path.join(self.get_raw_data_dir(), "vocab.vi")
        })

    def _load_data(self):
        data_file_names = {
            "train": {"en": "train.en", "vi": "train.vi"},
            "test": {"en": "tst2012.en", "vi": "tst2012.vi"}
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
                    X=[self.sos_token_idx] +
                      [self.vocab[self.lang_src].get_token_id(tkn) for tkn in src.split(' ')] +
                      [self.eos_token_idx],
                    Y=[self.sos_token_idx] +
                      [self.vocab[self.lang_tgt].get_token_id(tkn) for tkn in tgt.split(' ')] +
                      [self.eos_token_idx]
                ))
            data = list(filter(lambda it: len(it['X']) < 50, data))
            logger.debug("Data sample: %s", str(random.choice(data)))
            return data
        elif self.mode == "infer":
            return []

    @classmethod
    def maybe_download_and_extract(cls, force=False):
        super().maybe_download_and_extract(force)
        if os.path.exists(cls.get_raw_data_dir()):
            return

        for url in ["https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.en",
                    "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.vi",
                    "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2012.en",
                    "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2012.vi",
                    "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.en",
                    "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.vi",
                    "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/vocab.en",
                    "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/vocab.vi",
                    "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/dict.en-vi"]:
            try:
                cls.download(url)
            except Exception as e:
                logger.error("Failed to download %s" % url)
                logger.error(str(e))

    @classmethod
    def maybe_preprocess(cls, force=False):
        super().maybe_preprocess(force)