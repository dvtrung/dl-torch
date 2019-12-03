from dlex.datasets.image import ImageDataset


class MNIST(ImageDataset):
    def get_keras_wrapper(self, mode: str):
        from .keras import KerasCIFAR10
        return KerasCIFAR10(self, mode)

    def get_pytorch_wrapper(self, mode: str):
        from .torch import PytorchMNIST
        return PytorchMNIST(self, mode)

    @property
    def num_channels(self):
        return 1

    @property
    def input_shape(self):
        return 28, 28