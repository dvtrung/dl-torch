import os
import torch

class BaseModel(torch.nn.Module):
    def __init__(self, params, dataset):
        super().__init__()
        self.params = params
        self.dataset = dataset

        self.global_step = 0

    def load(self, tag):
        path = os.path.join("saved_models", self.params.path, tag + ".pt")
        self.load_state_dict(torch.load(path))

    @property
    def epoch(self):
        return self.global_step / len(self.dataset)
