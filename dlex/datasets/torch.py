import abc

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from dlex.torch import Batch


class PytorchDataset(Dataset):
    def __init__(self, builder, mode, params):
        self.params = params
        self._mode = mode
        self.builder = builder
        self._data = []

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

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