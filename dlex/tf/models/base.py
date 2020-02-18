import os

import tensorflow as tf
from dlex.tf import Batch
from tensorflow.python.keras.optimizers import SGD

from dlex.configs import MainConfig, ModuleConfigs
from dlex.datasets.tf import Dataset


class BaseModel_v1:
    def __init__(self, params: MainConfig, dataset: Dataset):
        super().__init__()
        self.params = params
        self.dataset = dataset
        self._optimizer = None
        self._loss = None

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

    def compile(self):
        self.model.compile(
            optimizer=SGD(self.params.train.optimizer.learning_rate, momentum=0.9),
            loss="categorical_crossentropy",
            metrics=self.dataset.get_metrics())

    @property
    def optimizers(self):
        if not self._optimizer:
            self._optimizer = tf.keras.optimizers.Adam()
        return self._optimizer

    @property
    def loss(self):
        return self._loss


class BaseModel:
    def __init__(self, params: MainConfig, dataset: Dataset):
        super().__init__()
        self.params = params
        self.dataset = dataset
        self._optimizer = None
        self._loss = None