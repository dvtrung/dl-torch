"""MNIST dataset"""

import torch.nn as nn
from torchvision.datasets import ImageNet as TorchImageNet
from sklearn.metrics import accuracy_score
from torchvision import transforms

from dlex.configs import Params
from torch.models.base import Batch, BatchItem
from ..base import BaseDataset
from dlex.torch.utils.ops_utils import maybe_cuda


class ImageNet(BaseDataset):
    def __init__(self, mode: str):
        super().__init__(mode, params)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        self._imagenet = TorchImageNet(
            self.get_working_dir(),
            split="train" if mode == "train" else "val",
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]),
            download=True)

    @property
    def num_classes(self):
        return len(self._imagenet.classes)

    @property
    def num_channels(self):
        return 1

    @property
    def input_shape(self):
        return self._imagenet

    def evaluate(self, y_pred, batch: Batch, metric='acc'):
        if metric == 'acc':
            return accuracy_score(batch.Y.cpu(), y_pred.cpu()) * y_pred.shape[0], y_pred.shape[0]
        elif metric == 'mse':
            criterion = nn.MSELoss()
            return criterion(y_pred.view(-1), batch.X.cpu().view(-1)).item(), y_pred.shape[0]

    def __len__(self):
        return len(self._imagenet)

    def __getitem__(self, idx):
        return self._imagenet[idx]

    def collate_fn(self, batch):
        ret = super().collate_fn(batch)
        return Batch(X=maybe_cuda(ret[0]), Y=maybe_cuda(ret[1]))

    def format_output(self, y_pred, batch_item: BatchItem):
        y_pred = y_pred.cpu().detach().numpy()
        if self.params.dataset.output_format is None:
            return "", str(batch_item.Y), str(y_pred)

