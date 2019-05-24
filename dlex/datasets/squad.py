import os
import tempfile

from .base.nlp import NLPDataset
from ..utils.utils import maybe_download


class SquadV2(NLPDataset):
    working_dir = os.path.join(tempfile.gettempdir(), "dlex", "datasets", "squad")
    raw_data_dir = os.path.join(working_dir, "raw")

    def __init__(self, mode, params):
        super().__init__(mode, params)

    @classmethod
    def maybe_download_and_extract(cls, force=False):
        training_set_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
        dev_set_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
        maybe_download("train.json", cls.raw_data_dir, training_set_url)
        maybe_download("dev.json", cls.raw_data_dir, dev_set_url)

    @classmethod
    def maybe_preprocess(cls, force=False):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    def evaluate(self, y_pred, batch, metric):
        pass

    def format_output(self, y_pred, inp):
        pass
