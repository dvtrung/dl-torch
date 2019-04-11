"""Autoencoder
References:
 - Auto-Encoding Variational Bayes (https://arxiv.org/abs/1312.6114)
"""
import torch.nn as nn

from models.base import BaseModel

class Model(BaseModel):
    """Autoencoder"""
    def __init__(self, params, dataset):
        super().__init__(params, dataset)
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.ReLU(True),
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Linear(64, 12), nn.ReLU(True),
            nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12), nn.ReLU(True),
            nn.Linear(12, 64), nn.ReLU(True),
            nn.Linear(64, 128), nn.ReLU(True),
            nn.Linear(128, 28 * 28), nn.Tanh())

    def forward(self, batch):
        img, _ = batch
        img = img.view(img.size(0), -1)
        hidden = self.encoder(img)
        output = self.decoder(hidden)
        return output

    def infer(self, batch):
        return self.forward(batch).cpu()

    def loss(self, batch):
        img, _ = batch
        img = img.view(img.size(0), -1)
        criterion = nn.MSELoss()
        output = self.forward(batch)
        return criterion(output, img)
