"""MNIST dataset"""

import os

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score

from torch.models.base import Batch, BatchItem
from ..base import BaseDataset
from dlex.utils.ops_utils import maybe_cuda


class VisionBaseDataset(BaseDataset):
    """MNIST dataset"""
    def evaluate(self, y_pred, batch, metric='acc'):
        if metric == 'acc':
            return accuracy_score(batch.Y.cpu(), y_pred.cpu()) * y_pred.shape[0], y_pred.shape[0]
        elif metric == 'mse':
            criterion = torch.nn.MSELoss()
            return criterion(y_pred.view(-1), batch.X.cpu().view(-1)).item(), y_pred.shape[0]

    def collate_fn(self, batch) -> Batch:
        ret = super().collate_fn(batch)
        return Batch(X=maybe_cuda(ret[0]), Y=maybe_cuda(ret[1]))

    def format_output(self, y_pred, batch_item: BatchItem) -> (str, str, str):
        y_pred = y_pred.cpu().detach().numpy()
        format = self.params.dataset.output_format
        if format is None or format == "default":
            return "", str(batch_item.Y), str(y_pred)
        elif format == "img":
            plt.subplot(1, 2, 1)
            plt.imshow(self.to_img(batch_item[0].cpu().detach().numpy()))
            plt.subplot(1, 2, 2)
            plt.imshow(self.to_img(y_pred))
            fn = os.path.join(self.params.output_dir, 'infer-%s.png' % tag)
            plt.savefig(fn)
            return "file: %s" % fn
        else:
            raise Exception("Unknown output format.")