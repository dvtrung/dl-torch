import os

from dlex.datasets.nlp.builder import NLPDatasetBuilder
from dlex.utils import logger


class IWSLT15(NLPDatasetBuilder):
    def get_pytorch_wrapper(self, mode: str):
        from .torch import IWSLT15EnglishVietnamese
        return IWSLT15EnglishVietnamese(self, mode)

    def maybe_download_and_extract(self, force=False):
        super().maybe_download_and_extract(force)
        if os.path.exists(self.get_raw_data_dir()):
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
                self.download(url)
            except Exception as e:
                logger.error("Failed to download %s" % url)
                logger.error(str(e))

    def maybe_preprocess(self, force=False):
        super().maybe_preprocess(force)