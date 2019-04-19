import os
import abc
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from utils.logging import logger


class BaseDataset(Dataset):
    def __init__(self, mode, params):
        super(Dataset).__init__()
        self.debug = mode == "debug"
        if mode == "debug":
            mode = "train"
        self.mode = mode
        self.params = params

        logger.info("Load '%s' dataset" % mode)

    @property
    def cfg(self):
        return self.params.dataset

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


def default_params(default):
    def wrap_fn(cls):
        class wrap_cls(cls):
            def __init__(self, mode, params):
                params.dataset.extend_default_keys(default)
                super().__init__(mode, params)
        return wrap_cls
    return wrap_fn


class Template(BaseDataset):
    working_dir = os.path.join("datasets", "squad")
    raw_data_dir = os.path.join(working_dir, "raw")

    def __init__(self, mode, params):
        super(BaseDataset).__init__()
