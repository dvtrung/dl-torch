import abc
import random
from typing import List

import torch
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
        self._builder = builder
        self._data = None
        self._sampler = None

    @abc.abstractmethod
    def load_data(self):
        raise NotImplemented

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    @property
    def builder(self):
        return self._builder

    @property
    def input_shape(self) -> List[int]:
        raise NotImplementedError

    @property
    def output_shape(self) -> List[int]:
        raise NotImplementedError

    @property
    def data(self):
        if not self._data:
            self._data = self.load_data()
        return self._data

    def shuffle(self):
        random.shuffle(self.data)

    @property
    def processed_data_dir(self) -> str:
        return self.builder.get_processed_data_dir()

    @property
    def raw_data_dir(self) -> str:
        return self.builder.get_raw_data_dir()

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
            num_workers=self.params.args.num_workers)

    @property
    def input_shape(self):
        return self.builder.input_shape

    @property
    def output_shape(self):
        return self.builder.output_shape

    def maybe_cuda(self, x):
        if self.params.gpu and torch.cuda.is_available():
            return x.cuda()
        else:
            return x