import torch.nn.functional as F
import torch.nn as nn

from torch.models.base import default_params, ImageClassificationBaseModel, Batch


class BasicModel(ImageClassificationBaseModel):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, dataset.num_classes)

    def forward(self, batch: Batch):
        x = batch.X
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


@default_params(dict(
    vgg_type="VGG11"
))
class VGG(ImageClassificationBaseModel):
    LAYERS = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                  'M'],
    }

    def __init__(self, params, dataset):
        super().__init__(params, dataset)

        layers = []
        in_channels = dataset.num_channels
        for x in self.LAYERS[self.cfg.vgg_type]:
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

    def forward(self, batch):
        out = self.features(batch.X)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = self.softmax(out)
        return out
