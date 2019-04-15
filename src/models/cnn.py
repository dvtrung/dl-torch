import torch
import torch.nn.functional as F
import torch.nn as nn

from models.base import BaseModel


class BasicModel(BaseModel):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, batch):
        x = batch['X']
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def infer(self, batch):
        logits = self.forward(batch)
        return torch.max(logits, 1)[1]
