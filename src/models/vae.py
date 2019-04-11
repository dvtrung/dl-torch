"""Autoencoder
References:
 - Auto-Encoding Variational Bayes (https://arxiv.org/abs/1312.6114)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.ops_utils import FloatTensor
from models.base import BaseModel

class Autoencoder(BaseModel):
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

class VariationalAutoencoder(BaseModel):
    """VAE"""
    def __init__(self, params, dataset):
        super().__init__(params, dataset)

        self.fc1 = nn.Linear(28 * 28, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, batch):
        img, _ = batch
        img = img.view(img.size(0), -1)
        mu, logvar = self.encode(img)
        z = self.reparametrize(mu, logvar)
        return self.decode(z)

    def infer(self, batch):
        return self.forward(batch).cpu()

    def loss(self, batch):
        img, _ = batch
        img = img.view(img.size(0), -1)
        criterion = nn.MSELoss()
        output = self.forward(batch)
        return criterion(output, img)
