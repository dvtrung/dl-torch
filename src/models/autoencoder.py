from torch import nn

from models.base import BaseModel

class Model(BaseModel):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

    def forward(self, batch):
        img, _ = batch
        img = img.view(img.size(0), -1)
        h = self.encoder(img)
        output = self.decoder(h)
        return output

    def infer(self, batch):
        return self.forward(batch).cpu()

    def loss(self, batch):
        img, _ = batch
        img = img.view(img.size(0), -1)
        criterion = nn.MSELoss()
        output = self.forward(batch)
        return criterion(output, img)

    def log(self, batch):
        output = self.forward(batch)
        pic = to_img(output.cpu().data)
        save_image(pic, './mlp_img/image_{}.png'.format(epoch))
