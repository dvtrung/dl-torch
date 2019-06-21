"""MNIST dataset"""

import os
import tempfile

from torchvision import transforms
from torchvision.datasets import MNIST as TorchMNIST

from torch.datasets.vision.base import VisionBaseDataset
from torch.models.base import Batch
from utils.ops_utils import maybe_cuda


class MNIST(VisionBaseDataset):
    """MNIST dataset"""

    def __init__(self, mode, params):
        super().__init__(mode, params)
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        img_transform = transforms.Compose(
            [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        self.mnist = TorchMNIST(
            os.path.join(tempfile.gettempdir(), "datasets", "mnist"),
            train=mode == "train",
            transform=img_transform,
            download=True)

    @property
    def num_classes(self):
        return 10

    @property
    def num_channels(self):
        return 1

    @property
    def input_shape(self):
        return self.num_channels, 28, 28

    def to_img(self, x):
        x = 0.5 * (x + 1)
        x = x.clip(0, 1)
        x = x.reshape(28, 28)
        return x

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        return self.mnist[idx]

    def collate_fn(self, batch) -> Batch:
        ret = super().collate_fn(batch)
        return Batch(X=maybe_cuda(ret.X), Y=maybe_cuda(ret.Y))