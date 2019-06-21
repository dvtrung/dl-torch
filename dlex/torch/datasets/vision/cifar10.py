"""MNIST dataset"""

from torchvision import transforms
from torchvision.datasets import CIFAR10 as TorchCIFAR10

from torch.datasets.vision.base import VisionBaseDataset


class CIFAR10(VisionBaseDataset):
    """CIFAR10 dataset"""

    def __init__(self, mode, params):
        super().__init__(mode, params)
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        self.cifar = TorchCIFAR10(
            self.get_working_dir(),
            train=mode == "train",
            transform=img_transform,
            download=True)

    @property
    def num_classes(self):
        return 10

    @property
    def num_channels(self):
        return 3

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        return self.cifar[idx]