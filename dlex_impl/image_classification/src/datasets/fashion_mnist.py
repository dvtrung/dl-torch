from typing import List, Union, Tuple

from dlex import Params
from dlex.datasets import DatasetBuilder
from dlex.datasets.tf import Dataset
from tensorflow import keras


class FashionMNIST(DatasetBuilder):
    def __init__(self, params: Params):
        super().__init__(params, tensorflow_cls=TensorflowFashionMNIST)
        fashion_mnist = keras.datasets.fashion_mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()
        self.train_images /= 255.
        self.test_images /= 255.


class TensorflowFashionMNIST(Dataset):
    def __init__(self, builder, mode: str):
        super().__init__(builder, mode)