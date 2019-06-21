import os
import abc
import shutil
import re

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from dlex.utils.logging import logger
from dlex.utils.utils import maybe_download, maybe_unzip
from dlex.configs import ModuleConfigs, AttrDict
from dlex.torch import Batch


def camel2snake(name):
    """
    :type name: str
    :rtype: str
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


class BaseDataset(Dataset):
    def __init__(self, mode: str, params: AttrDict):
        super(Dataset).__init__()
        self.debug = mode == "debug"
        if mode == "debug":
            mode = "train"
        self.mode = mode
        self.params = params
        self.data = []

        logger.info("Load '%s' dataset" % mode)

    @classmethod
    def get_working_dir(cls) -> str:
        return os.path.join(ModuleConfigs.DATA_TMP_PATH, camel2snake(cls.__name__))

    @classmethod
    def get_raw_data_dir(cls) -> str:
        return os.path.join(cls.get_working_dir(), "raw")

    @classmethod
    def get_processed_data_dir(cls) -> str:
        return os.path.join(cls.get_working_dir(), "processed")

    @property
    def cfg(self) -> AttrDict:
        return self.params.dataset

    @classmethod
    def prepare(cls, download=False, preprocess=False):
        cls.maybe_download_and_extract(download)
        cls.maybe_preprocess(download or preprocess)

    @classmethod
    def download_and_extract(cls, url: str, folder_path: str = None):
        file_path = maybe_download(cls.get_working_dir(), url)
        maybe_unzip(file_path, folder_path or cls.get_raw_data_dir())

    @classmethod
    def download(cls, url: str):
        maybe_download(cls.get_raw_data_dir(), url)

    @classmethod
    @abc.abstractmethod
    def maybe_download_and_extract(cls, force=False):
        if force:
            if os.path.exists(cls.get_working_dir()):
                logger.info("Removing downloaded data...")
                shutil.rmtree(cls.get_working_dir(), ignore_errors=True)
                while os.path.exists(cls.get_working_dir()):
                    pass

    @classmethod
    @abc.abstractmethod
    def maybe_preprocess(cls, force=False):
        if force:
            logger.info("Removing preprocessed data...")
            shutil.rmtree(cls.get_processed_data_dir(), ignore_errors=True)
            while os.path.exists(cls.get_processed_data_dir()):
                pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        return default_collate(batch)

    @abc.abstractmethod
    def evaluate(self, y_pred, batch: Batch, metric: str):
        raise Exception("Dataset method 'evaluate' must be implemented")

    @abc.abstractmethod
    def format_output(self, y_pred, inp) -> (str, str, str):
        raise Exception("Dataset method 'format_output' must be implemented")


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
