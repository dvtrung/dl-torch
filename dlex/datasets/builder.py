import abc
import os
import shutil
from collections import namedtuple
from typing import Tuple, List, Union

import sklearn.metrics as metrics

from dlex.configs import ModuleConfigs, Params
# from dlex.torch import BatchItem
from dlex.utils.logging import logger
from dlex.utils.utils import maybe_download, maybe_unzip, prompt


ModelStringOutput = namedtuple("ModelOutput", "input reference prediction")


class DatasetBuilder:
    """This is a base class for preparing data. It should handle downloading the data set and creating all files
    required for training.
    """

    def __init__(
            self,
            params: Params,
            downloads: List[Union[str, Tuple[str, str]]] = None,
            pytorch_cls=None,
            tensorflow_cls=None,
            sklearn_cls=None):
        self.params = params
        self.downloads = downloads
        self.pytorch_cls = pytorch_cls
        self.tensorflow_cls = tensorflow_cls
        self.sklearn_cls = sklearn_cls

    def get_working_dir(self) -> str:
        """Get the working directory"""
        return os.path.join(ModuleConfigs.get_datasets_path(), self.__class__.__name__)

    def get_raw_data_dir(self) -> str:
        """Get the directory to store raw data set"""
        return os.path.join(self.get_working_dir(), "raw")

    def get_processed_data_dir(self) -> str:
        """Get the directory to store pre-processed files"""
        return os.path.join(self.get_working_dir(), "processed")

    @property
    def configs(self) -> Params:
        return self.params.dataset

    def prepare(self, download=False, preprocess=False):
        self.maybe_download_and_extract(download)
        self.maybe_preprocess(download or preprocess)

    def _download_and_extract(self, url: Union[str, Tuple[str, str, str]], folder_path: str = None, filename: str = None):
        """Download and extract from url

        :param url: url to download
        :type url: str
        :param folder_path: location for the extracted files. If None, value in `get_raw_data_dir` is used.
        :type folder_path: str, optional
        """
        if isinstance(url, str):
            file_path = maybe_download(self.get_working_dir(), url, filename)
            maybe_unzip(file_path, folder_path or self.get_raw_data_dir())
        elif isinstance(url, tuple):
            url, username, password = url
            file_path = maybe_download(self.get_working_dir(), url, filename, username=username, password=password)
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

        if self.downloads:
            for url in self.downloads:
                if type(url) == str:
                    self._download_and_extract(url, self.get_raw_data_dir())
                elif type(url) == tuple:
                    if len(url) == 2:
                        self._download_and_extract(url[0], self.get_raw_data_dir(), url[1])
                    elif len(url) == 3:
                        self._download_and_extract(url, self.get_raw_data_dir())

    @abc.abstractmethod
    def maybe_preprocess(self, force=False) -> bool:
        """
        :param force:
        :return: true if the dataset needs to be prepared
        """

        if not os.path.exists(self.get_processed_data_dir()):
            os.makedirs(self.get_processed_data_dir(), exist_ok=True)
            return True
        if force:
            if prompt("Do you want to remove preprocessed data?", False):
                logger.info("Removing preprocessed data...")
                shutil.rmtree(self.get_processed_data_dir(), ignore_errors=True)
                while os.path.exists(self.get_processed_data_dir()):
                    pass
            os.makedirs(self.get_processed_data_dir(), exist_ok=True)
            return True
        return False

    @abc.abstractmethod
    def get_tensorflow_wrapper(self, mode: str):
        if self.tensorflow_cls:
            return self.tensorflow_cls(self, mode)
        raise NotImplementedError

    @abc.abstractmethod
    def get_pytorch_wrapper(self, mode: str):
        if self.pytorch_cls:
            return self.pytorch_cls(self, mode)
        raise NotImplementedError

    @abc.abstractmethod
    def get_keras_wrapper(self, mode: str):
        raise NotImplementedError

    @abc.abstractmethod
    def get_sklearn_wrapper(self, mode: str):
        if self.sklearn_cls:
            return self.sklearn_cls(self, mode)
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, pred, ref, metric: str, output_path: str):
        """
        :param pred:
        :param ref:
        :param metric:
        :param output_path:
        :return:
        """
        if metric == "acc":
            return float(metrics.accuracy_score(ref, pred)) * 100
        elif metric == "precision":
            return float(metrics.average_precision_score(ref, pred))
        elif metric == "recall":
            pass
        elif metric == "f1":
            return float(metrics.f1_score(ref, pred)) * 100
        elif metric == "err":
            ret = self.evaluate(pred, ref, "acc", output_path)
            return 100 - ret
        elif metric == "mse":
            return metrics.mean_squared_error(ref, pred)
        else:
            raise ValueError(f"Metric {metric} is not defined")

    @staticmethod
    def is_better_result(metric: str, best_result: float, new_result: float) -> bool:
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
        elif metric in ["acc", "bleu", "f1"]:
            return new_result > best_result
        else:
            raise Exception("Result comparison is not defined: %s" % metric)

    @abc.abstractmethod
    def format_output(self, y_pred, batch_item) -> ModelStringOutput:
        """Get representations of model inputs and results in readable format

        :param y_pred:
        :param batch_item:
        :type batch_item: BatchItem
        :return: A tuple containing string representations of input, ground-truth and predicted values
        """
        if self.params.dataset.output_format is None:
            return ModelStringOutput(str(batch_item.X), str(batch_item.Y), str(y_pred))
        else:
            raise Exception("Dataset method 'format_output' must be implemented")

    def run_evaluation_script(
            self,
            result_path,
            **kwargs):
        raise NotImplemented


class KaggleDatasetBuilder(DatasetBuilder):
    def __init__(self, params: Params, competition: str):
        super().__init__(params)

        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            competition,
            path=self.get_raw_data_dir(),
            unzip=True)