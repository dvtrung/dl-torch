import tensorflow as tf

from dlex import MainConfig
from dlex.datasets.tf import Dataset
from dlex.tf import BaseModel


class ImageClassifier(BaseModel):
    def __init__(self, params: MainConfig, dataset: Dataset):
        super().__init__(params, dataset)
        self.dense = tf.keras.layers.Dense(
            10, activation=tf.nn.softmax, kernel_initializer='zeros', input_shape=(784,))

    def call(self, inputs):
        return self.dense(inputs)

    @property
    def loss(self):
        return tf.keras.losses.SparseCategoricalCrossentropy()

    @property
    def metrics(self):
        return [tf.keras.metrics.SparseCategoricalAccuracy("acc")]