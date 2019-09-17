import abc
import random

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate, DataLoader

from dlex.torch import Batch


class PytorchDataset(Dataset):
    def __init__(self, builder, mode: str):
        self.params = builder.params
        self.mode = mode
        self.builder = builder
        self._data = []
        self._sampler = None

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]

    @property
    def data(self):
        return self._data

    def shuffle(self):
        random.seed = 9
        random.shuffle(self._data)

    @property
    def configs(self):
        return self.params.dataset

    @abc.abstractmethod
    def evaluate(self, y_pred, y_ref, metric: str):
        return self.builder.evaluate(y_pred, y_ref, metric)

    def format_output(self, y_pred, batch_input) -> (str, str, str):
        return self.builder.format_output(y_pred, batch_input)

    def collate_fn(self, batch):
        return default_collate(batch)

    def get_iter(self, batch_size, start=0, end=-1):
        return DataLoader(
            # some datasets don't support slicing
            self[start:end] if start != 0 or (end != -1 and end != len(self)) else self,
            batch_size=batch_size,
            collate_fn=self.collate_fn, sampler=self._sampler,
            num_workers=self.params.dataset.num_workers)