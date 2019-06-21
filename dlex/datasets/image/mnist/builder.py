from dlex.datasets.image.builder import ImageDatasetBuilder


class MNIST(ImageDatasetBuilder):
    def get_keras_wrapper(self, mode: str):
        from .keras import KerasCIFAR10
        return KerasCIFAR10(self, mode, self._params)

    def get_pytorch_wrapper(self, mode: str):
        from .torch import MNIST
        return MNIST(self, mode, self._params)