import abc

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

    @property
    def configs(self):
        return self.params.dataset

    @abc.abstractmethod
    def evaluate_batch(self, y_pred, batch: Batch, metric: str):
        score, total = 0, 0
        for _target, _y_pred in zip(batch.Y, y_pred):
            s, t = self.builder.evaluate(_target.cpu().detach().numpy().tolist(), _y_pred, metric)
            score += s
            total += t
        return score, total

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