import numpy as np
from torch.utils.data import SubsetRandomSampler
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10

from dlex.datasets.torch import Dataset
from dlex.torch import Batch
from dlex.torch.utils.ops_utils import maybe_cuda


class PytorchCIFAR10(Dataset):
    """CIFAR10 dataset"""

    def __init__(self, builder, mode):
        super().__init__(builder, mode)
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        self.cifar = CIFAR10(
            builder.get_working_dir(),
            train=mode == "train",
            transform=img_transform,
            download=True)
        if mode != "test":
            num_train = len(self.cifar)
            indices = list(range(num_train))
            split = int(np.floor(0.2 * num_train))
            np.random.shuffle(indices)
            train_idx, valid_idx = indices[split:], indices[:split]
            self._sampler = SubsetRandomSampler(train_idx if mode == "train" else valid_idx)

    def collate_fn(self, batch) -> Batch:
        ret = super().collate_fn(batch)
        return Batch(X=maybe_cuda(ret[0]), Y=maybe_cuda(ret[1]))

    @property
    def num_classes(self):
        return 10

    @property
    def num_channels(self):
        return 3

    @property
    def input_shape(self):
        return 32, 32

    def __len__(self):
        if self.mode == "test":
            return len(self.cifar)
        elif self.mode == "train":
            return int(len(self.cifar) * 0.8)
        else:
            return int(len(self.cifar) * 0.2)

    def __getitem__(self, idx):
        return self.cifar[idx]