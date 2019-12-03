from dlex import MainConfig
from dlex.datasets import DatasetBuilder
from dlex.datasets.torch import Dataset


class ModelNet40(DatasetBuilder):
    def __init__(self, params: MainConfig):
        super().__init__(params)

    def maybe_download_and_extract(self, force=False):
        self.download_and_extract(
            "http://modelnet.cs.princeton.edu/ModelNet40.zip",
            self.get_raw_data_dir()
        )

    def maybe_preprocess(self, force=False):
        pass

    def get_pytorch_wrapper(self, mode: str):
        return PytorchModelNet40(self, mode)


class PytorchModelNet40(Dataset):
    def __init__(self, builder, mode: str):
        super().__init__(builder, mode)

