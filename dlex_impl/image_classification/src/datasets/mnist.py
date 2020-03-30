from typing import List, Union, Tuple

from dlex import Params
from dlex.datasets import DatasetBuilder
from dlex.datasets.tf import Dataset
from tensorflow import keras


class MNIST(DatasetBuilder):
    def __init__(self, params: Params):
        super().__init__(params, tensorflow_cls=TensorflowMNIST)


class TensorflowMNIST(Dataset):
    def __init__(self, builder, mode: str):
        super().__init__(builder, mode)
        mnist = keras.datasets.mnist
        data = {}
        data['train'], data['test'] = mnist.load_data()
        self._data = data

    def get_input_fn(self):
        return self._data[self.mode]