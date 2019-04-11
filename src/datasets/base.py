import os
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from utils.logging import logger

class BaseDataset(Dataset):
    def __init__(self, mode, params, args):
        super().__init__()
        self.debug = mode == "debug"
        if mode == "debug":
            mode = "train"
        self.mode = mode
        self.params = params
        self.args = args

        logger.info("Load '%s' dataset" % mode)

    @classmethod
    def prepare(cls, force=False):
        # logger.info("Prepare dataset")
        pass

    def collate_fn(self, batch):
        return default_collate(batch)


class NLPDataset(BaseDataset):
    def __init__(self, mode, params, args=None):
        super().__init__(mode, params, args)
