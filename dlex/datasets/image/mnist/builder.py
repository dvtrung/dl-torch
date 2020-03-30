from typing import List, Union, Tuple

from dlex import Params
from dlex.datasets.image import ImageDataset
from .torch import PytorchMNIST


class MNIST(ImageDataset):
    def __init__(self, params: Params):
        super().__init__(params, pytorch_cls=PytorchMNIST)

    @property
    def num_channels(self):
        return 1

    @property
    def input_shape(self):
        return [28, 28]

    @property
    def output_shape(self):
        return [10]