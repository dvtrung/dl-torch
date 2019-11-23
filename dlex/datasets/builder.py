import abc
import os
import shutil
from typing import Tuple

from sklearn.metrics import accuracy_score

from dlex.configs import ModuleConfigs, MainConfig
from dlex.torch import BatchItem
from dlex.utils.logging import logger
from dlex.utils.utils import maybe_download, maybe_unzip


class DatasetBuilder:
    """This is a base class for preparing data. It should handle downloading the data set and creating all files
    required for training.
    """

    def __init__(self, params: MainConfig):
        self.params = params

    def get_working_dir(self) -> str:
        """Get the working directory"""
        return os.path.join(ModuleConfigs.DATASETS_PATH, self.__class__.__name__)

    def get_raw_data_dir(self) -> str:
        """Get the directory to store raw data set"""
        return os.path.join(self.get_working_dir(), "raw")

    def get_processed_data_dir(self) -> str:
        """Get the directory to store pre-processed files"""
        return os.path.join(self.get_working_dir(), "processed")

    @property
    def configs(self) -> MainConfig:
        return self.params.dataset

    def prepare(self, download=False, preprocess=False):
        self.maybe_download_and_extract(download)
        self.maybe_preprocess(download or preprocess)

    def download_and_extract(self, url: str, folder_path: str = None, filename: str = None):
        """Download and extract from url

        :param url: url to download
        :type url: str
        :param folder_path: location for the extracted files. If None, value in `get_raw_data_dir` is used.
        :type folder_path: str, optional
        """
        file_path = maybe_download(self.get_working_dir(), url, filename)
        maybe_unzip(file_path, folder_path or self.get_raw_data_dir())

    def download(self, url: str, filename: str = None):
        maybe_download(self.get_raw_data_dir(), url, filename)

    @abc.abstractmethod
    def maybe_download_and_extract(self, force=False):
        """

        :param force: if True, download and extract even when files are existed
        :return:
        """
        if force:
            if os.path.exists(self.get_working_dir()):
                logger.info("Removing downloaded data...")
                shutil.rmtree(self.get_working_dir(), ignore_errors=True)
                while os.path.exists(self.get_working_dir()):
                    pass

    @abc.abstractmethod
    def maybe_preprocess(self, force=False):
        os.makedirs(self.get_processed_data_dir(), exist_ok=True)
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
    def get_sklearn_wrapper(self, mode: str):
        return None

    @abc.abstractmethod
    def evaluate(self, pred, ref, metric: str):
        """

        :param pred:
        :param ref:
        :param metric:
        :return:
        """
        if metric == "acc":
            return accuracy_score(pred, ref) * 100
        elif metric == "err":
            ret = self.evaluate(pred, ref, "acc")
            return 100 - ret
        else:
            raise NotImplementedError

    @staticmethod
    def is_better_result(metric: str, best_result: float, new_result: float):
        """Compare new result with previous best result

        :param metric: name of metric
        :type metric: str
        :param best_result: current best result
        :type best_result: float
        :param new_result: new result to be compared with
        :type new_result: float
        :return: True if the new result is better with this metric.
        """
        if metric in ["wer", "loss", "err"]:  # the lower the better
            return new_result < best_result
        elif metric in ["acc", "bleu"]:
            return new_result > best_result
        else:
            raise Exception("Result comparison is not defined: %s" % metric)

    @abc.abstractmethod
    def format_output(self, y_pred, batch_item: BatchItem) -> Tuple[str, str, str]:
        """Get representations of model inputs and results in readable format

        :param y_pred:
        :param batch_item:
        :type batch_item: BatchItem
        :return: A tuple containing string representations of input, ground-truth and predicted values
        """
        if self.params.dataset.output_format is None:
            return str(batch_item.X), str(batch_item.Y), str(y_pred)
        else:
            raise Exception("Dataset method 'format_output' must be implemented")


class KaggleDatasetBuilder(DatasetBuilder):
    def __init__(self, params: MainConfig, competition: str):
        super().__init__(params)

        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            competition,
            path=self.get_raw_data_dir(),
            unzip=True)