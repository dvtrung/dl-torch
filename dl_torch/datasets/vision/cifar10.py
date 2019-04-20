"""MNIST dataset"""

import os
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10 as TorchCIFAR10
from sklearn.metrics import accuracy_score

from datasets.base import BaseDataset, default_params
from utils.ops_utils import LongTensor, maybe_cuda


class CIFAR10(BaseDataset):
    """CIFAR10 dataset"""

    def __init__(self, mode, params):
        super().__init__(mode, params)
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        self.cifar = TorchCIFAR10(
            os.path.join("datasets", "cifar10"),
            train=mode == "train",
            transform=img_transform,
            download=True)

    @property
    def num_classes(self):
        return 10

    @property
    def num_channels(self):
        return 3

    def evaluate(self, y_pred, batch, metric='acc'):
        if metric == 'acc':
            return accuracy_score(batch['Y'].cpu(), y_pred.cpu()) * y_pred.shape[0], y_pred.shape[0]

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        return self.cifar[idx]

    def collate_fn(self, batch):
        ret = super().collate_fn(batch)
        return dict(X=maybe_cuda(ret[0]), Y=maybe_cuda(ret[1]))