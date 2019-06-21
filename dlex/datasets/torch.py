import abc

from dlex.torch import Batch


class PytorchDataset:
    def __init__(self, builder, mode, params):
        self._params = params
        self._mode = mode
        self._builder = builder
        self._data = []

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    @abc.abstractmethod
    def evaluate_batch(self, y_pred, batch: Batch, metric: str):
        if metric == "bleu":
            score, total = 0, 0
            for _target, _y_pred in zip(batch.Y, y_pred):
                s, t = self._builder.evaluate(_target.cpu().detach().numpy().tolist(), _y_pred, metric)
                score += s
                total += t
            return score, total
        else:
            raise Exception("Unsupported metric.")

    @abc.abstractmethod
    def format_output(self, y_pred, inp) -> (str, str, str):
        raise Exception("Dataset method 'format_output' must be implemented")