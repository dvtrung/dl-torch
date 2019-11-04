import torch.nn as nn

from dlex import MainConfig
from dlex.datasets.torch import Dataset
from dlex.torch import Batch
from dlex.torch.models import ClassificationModel


class DNN(ClassificationModel):
    def __init__(self, params: MainConfig, dataset: Dataset):
        super().__init__(params, dataset)
        assert isinstance(dataset.input_shape, list) and len(dataset.input_shape) == 1
        assert isinstance(dataset.output_shape, list) and len(dataset.output_shape) == 1

        sizes = dataset.input_shape + self.configs.layers
        next_sizes = self.configs.layers + dataset.output_shape
        a_funcs = self.configs.activation_functions
        if not self.configs.activation_functions:
            a_funcs = ["relu"] * len(self.configs.layers) + ["softmax"]

        layers = []
        for size, next_size, a_func in zip(sizes, next_sizes, a_funcs):
            layers.append(nn.Linear(size, next_size))
            layers.append(self._get_activation_function(a_func))

        self.dnn = nn.Sequential(*layers)

    @staticmethod
    def _get_activation_function(s):
        if s == "relu":
            return nn.ReLU()
        elif s == "softmax":
            return nn.Softmax()

    def forward(self, batch: Batch):
        return self.dnn(batch.X)