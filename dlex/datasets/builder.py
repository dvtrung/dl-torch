import os
import abc
import shutil

from dlex.utils.logging import logger
from dlex.utils.utils import maybe_download, maybe_unzip, camel2snake
from dlex.configs import ModuleConfigs, AttrDict


class DatasetBuilder:
    def __init__(self, params: AttrDict):
        self.params = params

    def get_working_dir(self) -> str:
        return os.path.join(ModuleConfigs.DATA_TMP_PATH, camel2snake(self.__class__.__name__.replace('Builder', '')))

    def get_raw_data_dir(self) -> str:
        return os.path.join(self.get_working_dir(), "raw")

    def get_processed_data_dir(self) -> str:
        return os.path.join(self.get_working_dir(), "processed")

    @property
    def cfg(self) -> AttrDict:
        return self.params.dataset

    def prepare(self, download=False, preprocess=False):
        self.maybe_download_and_extract(download)
        self.maybe_preprocess(download or preprocess)

    def download_and_extract(self, url: str, folder_path: str = None):
        file_path = maybe_download(self.get_working_dir(), url)
        maybe_unzip(file_path, folder_path or self.get_raw_data_dir())

    def download(self, url: str):
        maybe_download(self.get_raw_data_dir(), url)

    @abc.abstractmethod
    def maybe_download_and_extract(self, force=False):
        if force:
            if os.path.exists(self.get_working_dir()):
                logger.info("Removing downloaded data...")
                shutil.rmtree(self.get_working_dir(), ignore_errors=True)
                while os.path.exists(self.get_working_dir()):
                    pass

    @abc.abstractmethod
    def maybe_preprocess(self, force=False):
        return
        if force:
            logger.info("Removing preprocessed data...")
            shutil.rmtree(self.get_processed_data_dir(), ignore_errors=True)
            while os.path.exists(self.get_processed_data_dir()):
                pass

    @abc.abstractmethod
    def get_tensorflow_wrapper(self, mode: str):
        return None

    @abc.abstractmethod
    def get_pytorch_wrapper(self, mode: str):
        return None

    @abc.abstractmethod
    def get_keras_wrapper(self, mode: str):
        return None

    @abc.abstractmethod
    def evaluate(self, pred, ref, metric: str):
        raise Exception("Not implemented.")

    @staticmethod
    def is_better_result(metric: str, best_result: float, new_result: float):
        if metric in ["wer"]:
            return new_result < best_result
        elif metric in ["acc", "bleu"]:
            return new_result > best_result
        else:
            raise Exception("Result comparison is not defined")

    @abc.abstractmethod
    def format_output(self, y_pred, batch_input) -> (str, str, str):
        raise Exception("Dataset method 'format_output' must be implemented")