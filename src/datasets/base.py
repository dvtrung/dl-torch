import os
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, mode, params):
        super().__init__()
        self.debug = mode == "debug"
        if mode == "debug":
            mode = "train"
        self.mode = mode
        self.params = params

    @classmethod
    def prepare(cls):
        pass
