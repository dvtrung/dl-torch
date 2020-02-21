import collections

import tensorflow as tf
import tensorflow_federated as tff


class TensorflowFederatedModel(tff.learning.Model):
    def __init__(self, params, dataset):
        self.params = params
        self.dataset = dataset
        self._variables = None

    @tf.function
    def forward_pass(self, batch, training=True):
        del training
        loss, predictions = self.forward(batch)
        num_exmaples = tf.shape(batch['x'])[0]
        return tff.learning.BatchOutput(
            loss=loss, predictions=predictions, num_examples=num_exmaples)