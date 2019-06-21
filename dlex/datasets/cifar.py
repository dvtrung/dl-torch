import tensorflow_datasets as tfds
import tensorflow as tf
from keras_preprocessing.image import NumpyArrayIterator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from dlex.datasets import KerasDataset
from .base import BaseDataset


class CIFAR10(BaseDataset):
    def get_keras_wrapper(self, mode: str) -> KerasDataset:
        return KerasCIFAR10(self, mode, self._params)


class KerasCIFAR10(KerasDataset):
    def __init__(self, dataset, mode, params):
        super().__init__(dataset, mode, params)
        if mode == "train":
            (X, Y), _ = cifar10.load_data()
            generator = ImageDataGenerator(
                rescale=1.0 / 255, horizontal_flip=True,
                width_shift_range=4.0 / 32.0, height_shift_range=4.0 / 32.0)
        else:
            _, (X, Y) = cifar10.load_data()
            generator = ImageDataGenerator(rescale=1.0 / 255)

        Y = to_categorical(Y)
        self._X, self._Y = X, Y
        self._generator = generator
        self._iterator = generator.flow(X, Y, params.batch_size, shuffle=True)

    def __len__(self):
        return len(self._X)

    @property
    def iterator(self) -> NumpyArrayIterator:
        return self._iterator


class CIFAR10_tf(BaseDataset):
    def __init__(self, params, mode):
        super().__init__(params, mode)
        data, info = tfds.load("cifar10", split=mode, with_info=True)
        data = data.map(lambda item: (item['image'] / 255, tf.one_hot(item['label'], self.num_classes)))
        data = data.shuffle(1000).batch(params.batch_size)
        self._data = data
        self._info = info

    def __len__(self):
        return self._info.splits[self._mode].num_examples

    @property
    def num_classes(self):
        return 10

    @property
    def generator(self):
        return self._data.__iter__()