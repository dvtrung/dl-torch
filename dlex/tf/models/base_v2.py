import tensorflow as tf
from dlex import Params
from dlex.datasets.tf import Dataset


class BaseModel(tf.keras.Model):
    def __init__(self, params: Params, dataset: Dataset):
        super().__init__()
        self.params = params
        self.dataset = dataset
        self._optimizer = None
        self._loss = None

    @property
    def model(self):
        raise NotImplemented

    def compile(self):
        super().compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics)
        return self.model

    @property
    def optimizer(self):
        return tf.keras.optimizers.SGD(learning_rate=0.02)