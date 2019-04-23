import os
import abc
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from ...utils.logging import logger


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
    def prepare(cls, download=False, preprocess=False):
        cls.maybe_download_and_extract(download)
        cls.maybe_preprocess(download or preprocess)

    @classmethod
    @abc.abstractmethod
    def maybe_download_and_extract(cls, force=False):
        """Download and extract data"""
        pass

    @classmethod
    @abc.abstractmethod
    def maybe_preprocess(cls, force=False):
        """Preprocess data"""
        pass

    def collate_fn(self, batch):
        return default_collate(batch)

    @abc.abstractmethod
    def evaluate(self, y_pred, batch, metric):
        """Evaluate a batch.

        Args:
            y_pred:
            batch:
            metric (str): name of the metric

        Returns:
            int: Total sum toward metric value
            int: Total sum toward number of samples
        """
        raise Exception("Dataset method 'evaluate' must be implemented")

    @abc.abstractmethod
    def format_output(self, y_pred, inp):
        """Print or save model output

        Args:
            y_pred:
            inp:

        Returns:
            str: the formatted string
        """
        raise Exception("Dataset method 'format_output' must be implemented")

    @staticmethod
    def get_item_from_batch(batch, i):
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
