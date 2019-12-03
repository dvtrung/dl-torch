import os

from dlex.datasets.nlp.builder import NLPDataset
from dlex.utils import logger


class WMT14(NLPDataset):
    def get_pytorch_wrapper(self, mode: str):
        from .torch import WMT14EnglishGerman
        return WMT14EnglishGerman(self, mode)

    def maybe_download_and_extract(self, force=False):
        super().maybe_download_and_extract(force)
        if os.path.exists(self.get_raw_data_dir()):
            return

        urls = [
            "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en",
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
            "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/dict.en-de"
        ]
        for url in urls:
            try:
                self.download(url)
            except Exception as e:
                logger.error("Failed to download %s" % url)
                logger.error(str(e))

    def maybe_preprocess(self, force=False):
        super().maybe_preprocess(force)