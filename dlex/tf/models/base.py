import os

import tensorflow as tf
from dlex.tf import Batch
from tensorflow.python.keras.optimizers import SGD

from dlex.configs import MainConfig, ModuleConfigs
from dlex.datasets.tf import Dataset
from tensorflow.estimator import EstimatorSpec


class BaseModel_v1:
    def __init__(self, params: MainConfig, dataset: Dataset):
        super().__init__()
        self.params = params
        self.dataset = dataset
        self._optimizer = None
        self.loss = None

    def training_step(self, batch: Batch):
        with tf.GradientTape() as tape:
            loss = self(batch)
            batch_loss = (loss / int(batch.Y.shape[1]))
            gradients = tape.gradient(loss, self.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            return batch_loss

    @property
    def trainable_variables(self):
        return []

    @property
    def optimizers(self):
        if not self._optimizer:
            self._optimizer = tf.keras.optimizers.Adam()
        return self._optimizer

    def forward(self, batch):
        raise NotImplemented

    def get_loss(self, batch, output):
        raise NotImplemented

    def get_train_op(self, loss):
        raise NotImplemented

    def get_metric_ops(self, batch, output):
        raise NotImplemented


class BaseModel(tf.keras.Model):
    def __init__(self, params: MainConfig, dataset: Dataset):
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