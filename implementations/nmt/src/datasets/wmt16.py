import os

from dlex.datasets.builder import DatasetBuilder
from dlex.datasets.nlp.builder import NLPDataset
from dlex.datasets.seq2seq.torch import PytorchSeq2SeqDataset
from dlex.utils.logging import logger


class WMT16(NLPDataset):
    def __init__(self, params):
        super().__init__(params)

    def maybe_download_and_extract(self, force=False):
        super().maybe_download_and_extract(force)
        if os.path.exists(self.get_raw_data_dir()):
            return

        for url in ["http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz",
                    "http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz",
                    "http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/mmt16_task1_test.tar.gz"]:
            try:
                self.download(url)
            except Exception as e:
                logger.error("Failed to download %s" % url)
                logger.error(str(e))

    def get_pytorch_wrapper(self, mode: str):
        return PytorchWMT16(self, mode)


class PytorchWMT16(PytorchSeq2SeqDataset):
    def __init__(self, builder: DatasetBuilder, mode: str, vocab_path: str = None):
        super().__init__(builder, mode, vocab_path)