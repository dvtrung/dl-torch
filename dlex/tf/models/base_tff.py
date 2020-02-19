import collections

import tensorflow as tf
import tensorflow_federated as tff


class TensorflowFederatedModel(tff.learning.Model):
    def __init__(self, params, dataset):
        self.params = params
        self.dataset = dataset
        self._variables = None

    @property
    def non_trainable_variables(self):
        return []

    @property
    def local_variables(self):
        return [
            self._variables.num_examples, self._variables.loss_sum,
            self._variables.accuracy_sum
        ]

    @property
    def input_spec(self):
        return collections.OrderedDict(
            x=tf.TensorSpec([None, 784], tf.float32),
            y=tf.TensorSpec([None, 1], tf.int32))

    @tf.function
    def forward_pass(self, batch, training=True):
        del training
        loss, predictions = self.forward(batch)
        num_exmaples = tf.shape(batch['x'])[0]
        return tff.learning.BatchOutput(
            loss=loss, predictions=predictions, num_examples=num_exmaples)

    @tf.function
    def report_local_outputs(self):
        return self.metrics()

    @property
    def federated_output_computation(self):
        return self.aggregate_mnist_metrics_across_clients