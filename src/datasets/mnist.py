import os
from torchvision import transforms
from torchvision.datasets import MNIST

from datasets.base import BaseDataset
from utils.ops_utils import Tensor, LongTensor, maybe_cuda

class Dataset(BaseDataset):
    def __init__(self, mode, params):
        super().__init__(mode, params)
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.mnist = MNIST(
            os.path.join("datasets", "mnist"),
            train=mode == "train",
            transform=img_transform,
            download=True)

    def to_img(self, x):
        x = 0.5 * (x + 1)
        x = x.clamp(0, 1)
        x = x.view(x.size(0), 1, 28, 28)
        return x

    def eval(self, pr, gt):
        return (gt == pr).item()

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        return (maybe_cuda(self.mnist[idx][0]), LongTensor(self.mnist[idx][1]))
