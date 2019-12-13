import abc
import random
from typing import List

from torch.utils.data import Dataset as PytorchDataset
from torch.utils.data.dataloader import default_collate, DataLoader


class Dataset(PytorchDataset):
    """Load data from pre-processed files and prepare batch for training

    :param builder:
    :type builder: DatasetBuilder
    :param mode: one of `train` / `valid` (or `dev`) / `test`
    :type mode: str
    """

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
    def input_shape(self) -> List[int]:
        raise NotImplementedError

    @property
    def output_shape(self) -> List[int]:
        raise NotImplementedError

    @property
    def data(self):
        return self._data

    def shuffle(self, seed=42):
        """Shuffle

        :param seed:
        :type seed: int
        """
        random.seed = seed
        random.shuffle(self._data)

    @property
    def configs(self):
        return self.params.dataset

    @abc.abstractmethod
    def evaluate(self, y_pred, y_ref, metric: str, output_path: str):
        return self.builder.evaluate(y_pred, y_ref, metric, output_path)

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