import os
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

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

    def collate_fn(self, batch):
        return default_collate(batch)