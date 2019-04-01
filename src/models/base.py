import torch

class BaseModel(torch.nn.Module):
    def __init__(self, params, dataset):
        super().__init__()
        self.params = params
        self.dataset = dataset

    def save(self, tag):
        path = os.path.join("saved_models", self.params.path, tag + ".pt")
        torch.save(self.state_dict(), path)

    def load(self, tag):
        path = os.path.join("saved_models", self.params.path, tag + ".pt")
        self.load_state_dict(torch.load(path))
