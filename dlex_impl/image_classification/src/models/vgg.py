import torch.nn as nn

from dlex.torch.models.base import ClassificationModel
from dlex.torch import Batch


class VGG(ClassificationModel):
    LAYERS = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                  'M'],
    }

    def __init__(self, params, dataset):
        super().__init__(params, dataset)

        cfg = params.model
        layers = []
        in_channels = dataset.num_channels
        for x in self.LAYERS[cfg.vgg_type or 'VGG11']:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, x, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(x))
                layers.append(nn.ReLU(inplace=True))
                in_channels = x
        layers.append(nn.AvgPool2d(kernel_size=1, stride=1))
        self.features = nn.Sequential(*layers)

        self.classifier = nn.Linear(512, dataset.num_classes)
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, batch: Batch):
        out = self.features(batch.X)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = self.softmax(out)
        return out
