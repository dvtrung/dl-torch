import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from models.base import BaseModel, default_params
from utils.ops_utils import FloatTensor
from utils.model_utils import get_optimizer


class Generator(nn.Module):
    def __init__(self, params, dataset):
        super(Generator, self).__init__()

        self.input_shape = dataset.input_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(params.model.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.input_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.input_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, params, dataset):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(dataset.input_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


@default_params(dict(
    latent_dim=100
))
class GAN(BaseModel):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)

        self.generator = Generator(params, dataset)
        self.discriminator = Discriminator(params, dataset)

        self.loss_fn = self._get_loss_fn()
        self.optimizer_g, self.optimizer_d = self._get_optimizer()

    def _get_optimizer(self):
        optimizer_g = get_optimizer(self.params.optimizer, self.generator.parameters())
        optimizer_d = get_optimizer(self.params.optimizer, self.discriminator.parameters())

        return optimizer_g, optimizer_d

    def _get_loss_fn(self):
        criterion = torch.nn.BCELoss()

        def loss_fn(batch, output):
            criterion(batch['X'], output)

        return loss_fn

    def forward(self, batch):
        imgs = batch['X']
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        return self.generator(z)

    def training_step(self, batch):
        imgs = batch['X']
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
        real_imgs = Variable(imgs.type(FloatTensor))
        gen_imgs = self.forward(batch)
        # Loss measures generator's ability to fool the discriminator
        g_loss = self.loss_fn(self.discriminator(gen_imgs), valid)

        g_loss.backward()
        self.optimizer_g.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        self.optimizer_d.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = self.loss_fn(self.discriminator(real_imgs), valid)
        fake_loss = self.loss_fn(self.discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()

        self.optimizer_d.step()