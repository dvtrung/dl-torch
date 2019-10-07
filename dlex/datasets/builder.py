import os
import abc
import shutil

from sklearn.metrics import accuracy_score

from dlex.utils.logging import logger
from dlex.utils.utils import maybe_download, maybe_unzip
from dlex.configs import ModuleConfigs, AttrDict


class DatasetBuilder:
    def __init__(self, params: AttrDict):
        self.params = params

    def get_working_dir(self) -> str:
        return os.path.join(ModuleConfigs.DATASETS_PATH, self.__class__.__name__)

    def get_raw_data_dir(self) -> str:
        return os.path.join(self.get_working_dir(), "raw")

    def get_processed_data_dir(self) -> str:
        return os.path.join(self.get_working_dir(), "processed")

    @property
    def configs(self) -> AttrDict:
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
        if metric == "acc":
            return accuracy_score(pred, ref)
        elif metric == "err":
            ret = self.evaluate(pred, ref, "acc")
            return 1 - ret
        else:
            raise Exception("Not implemented.")

    @staticmethod
    def is_better_result(metric: str, best_result: float, new_result: float):
        """Compare new result with previous best result"""
        if metric in ["wer", "loss", "err"]:  # the lower the better
            return new_result < best_result
        elif metric in ["acc", "bleu"]:
            return new_result > best_result
        else:
            raise Exception("Result comparison is not defined: %s" % metric)

    @abc.abstractmethod
    def format_output(self, y_pred, batch_item) -> (str, str, str):
        if self.params.dataset.output_format is None:
            return str(batch_item.X), str(batch_item.Y), str(y_pred)
        else:
            raise Exception("Dataset method 'format_output' must be implemented")


class KaggleDatasetBuilder(DatasetBuilder):
    def __init__(self, params: AttrDict, competition: str):
        super().__init__(params)

        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            competition,
            path=self.get_raw_data_dir(),
            unzip=True)