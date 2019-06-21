from dataclasses import dataclass

import tensorflow as tf

from dlex.configs import AttrDict
from dlex.datasets import TensorflowDataset as Dataset
from dlex.tf import Batch


class BaseModel():
    def __init__(self, params: AttrDict, dataset: Dataset):
        super().__init__()
        self._params = params
        self._dataset = dataset
        self._optimizer = tf.keras.optimizers.Adam()

    def training_step(self, batch: Batch):
        loss = 0
        with tf.GradientTape() as tape:
            loss = self(batch)
            batch_loss = (loss / int(batch.Y.shape[1]))
            gradients = tape.gradient(loss, self.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            return batch_loss

    @property
    def trainable_variables(self):
        return []