import torch.nn as nn
import torch.nn.functional as F
from dlex.torch import Batch

from dlex.torch.models import ClassificationModel


class CNN(ClassificationModel):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)
        self.conv1 = nn.Conv2d(
            in_channels=params.model.num_channels or dataset.num_channels,
            out_channels=20,
            kernel_size=5,
            stride=1, padding=2)
        self.conv2 = nn.Conv2d(
            in_channels=20,
            out_channels=50,
            kernel_size=5,
            stride=1, padding=2)
        input_shape = params.model.input_shape or dataset.input_shape
        self.fc1 = nn.Linear((input_shape[0] // 4) * (input_shape[1] // 4) * 50, 500)
        self.fc2 = nn.Linear(500, dataset.num_classes)

    def forward(self, batch: Batch):
        x = batch.X['persistence_image']
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)