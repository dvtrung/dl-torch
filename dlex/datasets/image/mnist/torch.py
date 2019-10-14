import numpy as np
from torchvision.transforms import transforms
from torchvision.datasets import MNIST as TorchMNIST
from torch.utils.data.sampler import SubsetRandomSampler

from dlex.datasets.image.torch import PytorchImageDataset
from dlex.torch import Batch
from dlex.torch.utils.ops_utils import maybe_cuda


class PytorchMNIST(PytorchImageDataset):
    def __init__(self, builder, mode):
        super().__init__(builder, mode)
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.mnist = TorchMNIST(
            builder.get_working_dir(),
            train=mode != "test",
            transform=img_transform,
            download=True)
        if mode != "test":
            num_train = len(self.mnist)
            indices = list(range(num_train))
            split = int(np.floor(0.2 * num_train))
            np.random.shuffle(indices)
            train_idx, valid_idx = indices[split:], indices[:split]
            self._sampler = SubsetRandomSampler(train_idx if mode == "train" else valid_idx)

    @property
    def num_classes(self):
        return 10

    @property
    def num_channels(self):
        return 1

    @property
    def input_shape(self):
        return 28, 28

    def __len__(self):
        if self.mode == "test":
            return len(self.mnist)
        elif self.mode == "train":
            return int(len(self.mnist) * 0.8)
        else:
            return int(len(self.mnist) * 0.2)

    def __getitem__(self, idx):
        return self.mnist[idx]