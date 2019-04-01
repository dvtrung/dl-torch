import os
from torchvision import transforms
from torchvision.datasets import MNIST

from datasets.base import BaseDataset

class Dataset(BaseDataset):
    def __init__(self, mode, params):
        super().__init__(mode, params)
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.mnist = MNIST(
            os.path.join("datasets", "mnist"), 
            transform=img_transform, 
            download=True)

    def to_img(self, x):
        x = 0.5 * (x + 1)
        x = x.clamp(0, 1)
        x = x.view(x.size(0), 1, 28, 28)
        return x

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        return self.mnist[idx]