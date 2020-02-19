import torch.nn as nn
import torch.nn.functional as F
from dlex.torch import Batch

from dlex.torch.models import ClassificationModel
from dlex.utils import split_ints


class SimpleCNN(ClassificationModel):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)
        cfg = self.configs

        layers = []
        num_channels = split_ints(cfg.num_channels)

        prev_num_channels = dataset.num_channels
        for i in num_channels:
            layers.append(nn.Conv2d(
                prev_num_channels, i,
                kernel_size=cfg.kernel_size,
                padding=1, stride=1, dilation=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2, 2))
            prev_num_channels = i
        self.conv = nn.Sequential(*layers)

        self.fc1 = nn.Linear((dataset.input_shape[0] // 4) * (dataset.input_shape[1] // 4) * num_channels[-1], 500)
        self.fc2 = nn.Linear(500, dataset.num_classes)

    def forward(self, batch: Batch):
        x = batch.X
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)