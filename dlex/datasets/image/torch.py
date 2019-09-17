from dlex.datasets.torch import PytorchDataset
from dlex.torch import Batch


class PytorchImageDataset(PytorchDataset):
    def __init__(self, builder, mode):
        super().__init__(builder, mode)

    def evaluate(self, y_pred, y_ref, metric: str):
        if metric == "acc":
            score, total = 0, 0
            for _target, _y_pred in zip(batch.Y, y_pred):
                s, t = self.builder.evaluate(_target.cpu().detach().numpy().tolist(), _y_pred, metric)
                score += s
                total += t
            return score, total
        else:
            raise Exception("Unsupported metric.")