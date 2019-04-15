import abc
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from utils.logging import logger


class BaseDataset(Dataset):
    def __init__(self, mode, params, args):
        super(Dataset).__init__()
        self.debug = mode == "debug"
        if mode == "debug":
            mode = "train"
        self.mode = mode
        self.params = params
        self.args = args

        logger.info("Load '%s' dataset" % mode)

    @classmethod
    def prepare(cls, force=False):
        cls.maybe_download_and_extract(force)
        cls.maybe_preprocess(force)

    @classmethod
    @abc.abstractmethod
    def maybe_download_and_extract(cls, force=False):
        pass

    @classmethod
    @abc.abstractmethod
    def maybe_preprocess(cls, force=False):
        pass

    def collate_fn(self, batch):
        return default_collate(batch)

    @abc.abstractmethod
    def evaluate(self, y_pred, batch, metric):
        return None

    @staticmethod
    def get_input_from_batch(batch, i):
        return {key: batch[key][i] for key in batch}