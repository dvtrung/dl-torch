import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from torch.models.base import BaseModel, default_params
from dlex.torch.utils.ops_utils import FloatTensor


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

    def forward(self, imgs):
        img_flat = imgs.view(imgs.size(0), -1)
        validity = self.model(img_flat)

        return validity


@default_params(dict(
    latent_dim=100
))
class WaveNet(BaseModel):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)

        cfg = params.model
        self.start_conv = nn.Conv1d(
            in_channels=cfg.input_size,
            out_channels=cfg.residual_channels,
            kernel_size=1,
            bias=cfg.bias)

        for b in range(cfg.num_blocks):
            additional_scope = cfg.kernel_size - 1
            new_dilation = 1
            for i in range(cfg.num_layers):
                # dilations of this layer
                self.dilations.append((new_dilation, init_dilation))

                # dilated queues for fast generation
                self.dilated_queues.append(DilatedQueue(max_length=(cfg.kernel_size - 1) * new_dilation + 1,
                                                        num_channels=cfg.residual_channels,
                                                        dilation=new_dilation,
                                                        dtype=dtype))

                # dilated convolutions
                self.filter_convs.append(nn.Conv1d(in_channels=cfg.residual_channels,
                                                   out_channels=cfg.dilation_channels,
                                                   kernel_size=cfg.kernel_size,
                                                   bias=cfg.bias))

                self.gate_convs.append(nn.Conv1d(in_channels=cfg.residual_channels,
                                                 out_channels=cfg.dilation_channels,
                                                 kernel_size=kernel_size,
                                                 bias=bias))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=1,
                                                     bias=bias))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=1,
                                                 bias=bias))

                receptive_field += additional_scope
                additional_scope *= 2
                init_dilation = new_dilation
        new_dilation *= 2

    def forward(self, batch):
        imgs = batch.X
        z = Variable(FloatTensor(np.random.normal(0, 1, (imgs.shape[0], self.cfg.latent_dim))))
        return self.generator(z)

    def infer(self, batch):
        return self.forward(batch)

    def training_step(self, batch):
        imgs = batch.X
        optimizer_g, optimizer_d = self.optimizers
        valid = Variable(FloatTensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
        real_imgs = Variable(imgs.type(FloatTensor))
        gen_imgs = self.forward(batch)
        # Loss measures generator's ability to fool the discriminator
        g_loss = self._criterion(self.discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_g.step()

        # Train Discriminator
        optimizer_d.zero_grad()
        # Measure discriminator's ability to classify real from generated samples
        real_loss = self._criterion(self.discriminator(real_imgs), valid)
        fake_loss = self._criterion(self.discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()

        optimizer_d.step()

        return d_loss
