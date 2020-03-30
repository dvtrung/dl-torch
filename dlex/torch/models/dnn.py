import torch.nn as nn
import numpy as np

from dlex import Params
from dlex.datasets.torch import Dataset
from dlex.torch import Batch
from dlex.torch.models import ClassificationModel
from dlex.torch.utils.model_utils import linear_layers


class DNN(ClassificationModel):
    def __init__(self, params: Params, dataset: Dataset):
        super().__init__(params, dataset)
        assert isinstance(dataset.input_shape, list)
        assert isinstance(dataset.output_shape, list)
        assert len(dataset.output_shape) == 1

        self.dnn = linear_layers(
            [np.prod(dataset.input_shape)] + self.configs.layers,
            dropout=params.dropout or 0.2,
            activation_fn=params.activation_fn
        )

    def forward(self, batch: Batch):
        X = batch.X.reshape(batch.X.shape[0], -1)
        return self.dnn(X)