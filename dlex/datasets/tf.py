from dlex.tf import Batch


class TensorflowDataset:
    def __init__(self, dataset, mode, params):
        self.params = params
        self.mode = mode
        self.dataset = dataset

    @abc.abstractmethod
    def evaluate_batch(self, y_pred, batch: Batch, metric: str):
        if metric == "bleu":
            target_variables = batch.Y
            score, total = 0, 0
            for k, _y_pred in enumerate(y_pred):
                target = target_variables[k].cpu().detach().numpy().tolist()
                predicted = _y_pred
                s, t = self.dataset.evaluate(target, predicted, metric)
                score += s
                total += t
            return score, total
        else:
            raise Exception("Unsupported metric.")