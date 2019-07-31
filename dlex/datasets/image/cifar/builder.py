from dlex.datasets.image import ImageDataset


class CIFAR10(ImageDataset):
    def get_keras_wrapper(self, mode: str):
        from .keras import KerasCIFAR10
        return KerasCIFAR10(self, mode)

    def get_pytorch_wrapper(self, mode: str):
        from .torch import PytorchCIFAR10
        return PytorchCIFAR10(self, mode)