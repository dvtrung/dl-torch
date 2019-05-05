"""MNIST dataset"""

import os
import tempfile
import torch
from torchvision import transforms
from torchvision.datasets import MNIST as TorchMNIST
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from ..base import BaseDataset, default_params
from ...utils.ops_utils import LongTensor, maybe_cuda


class MNIST(BaseDataset):
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

    def evaluate(self, y_pred, batch, metric='acc'):
        if metric == 'acc':
            return accuracy_score(batch['Y'].cpu(), y_pred.cpu()) * y_pred.shape[0], y_pred.shape[0]
        elif metric == 'mse':
            criterion = torch.nn.MSELoss()
            return criterion(y_pred.view(-1), batch['X'].cpu().view(-1)).item(), y_pred.shape[0]

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        return self.mnist[idx]

    def collate_fn(self, batch):
        ret = super().collate_fn(batch)
        return dict(X=maybe_cuda(ret[0]), Y=maybe_cuda(ret[1]))

    def format_output(self, y_pred, inp, display=None, tag=None):
        y_pred = y_pred.cpu().detach().numpy()
        if display is None:
            return str(y_pred)
        elif display == "img":
            plt.subplot(1, 2, 1)
            plt.imshow(self.to_img(inp[0].cpu().detach().numpy()))
            plt.subplot(1, 2, 2)
            plt.imshow(self.to_img(y_pred))
            fn = os.path.join(self.params.output_dir, 'infer-%s.png' % tag)
            plt.savefig(fn)
            return "file: %s" % fn
