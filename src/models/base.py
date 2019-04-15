import os
import torch

class BaseModel(torch.nn.Module):
    def __init__(self, params, dataset):
        super().__init__()
        self.params = params
        self.dataset = dataset

        self.global_step = 0

        if torch.cuda.is_available():
            # logger.info("Cuda available: %s", torch.cuda.get_device_name(0))
            self.cuda()

    def load(self, tag):
        path = os.path.join("saved_models", self.params.path, tag + ".pt")
        self.load_state_dict(torch.load(path))

    @property
    def epoch(self):
        return self.global_step / len(self.dataset)

    def infer(self, batch):
        """Infer"""
        return None

    def loss(self, batch):
        """Loss"""
        return None

    def initHidden(self):
        pass
