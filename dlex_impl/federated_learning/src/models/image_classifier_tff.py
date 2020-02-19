import collections

import tensorflow as tf
import tensorflow_federated as tff

from dlex import MainConfig
from dlex.datasets.tf import Dataset
from dlex.tf.models.base_tff import TensorflowFederatedModel


class ImageClassifier(TensorflowFederatedModel):
    def __init__(self, params: MainConfig, dataset: Dataset):
        super().__init__(params, dataset)
        self._variables = dict(
            weights=tf.Variable(
                lambda: tf.zeros(dtype=tf.float32, shape=(784, 10)),
                name='weights',
                trainable=True),
            bias=tf.Variable(
                lambda: tf.zeros(dtype=tf.float32, shape=10),
                name='bias',
                trainable=True),
            num_examples=tf.Variable(0.0, name='num_examples', trainable=False),
            loss_sum=tf.Variable(0.0, name='loss_sum', trainable=False),
            accuracy_sum=tf.Variable(0.0, name='accuracy_sum', trainable=False)
        )

    @property
    def trainable_variables(self):
        return [self._variables['weights'], self._variables['bias']]

    def forward(self, batch):
        variables = self._variables
        y = tf.nn.softmax(tf.matmul(batch['x'], variables['weights']) + variables['bias'])
        predictions = tf.cast(tf.argmax(y, 1), tf.int32)

        flat_labels = tf.reshape(batch['y'], [-1])
        loss = -tf.reduce_mean(
            tf.reduce_sum(tf.one_hot(flat_labels, 10) * tf.math.log(y), axis=[1]))
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predictions, flat_labels), tf.float32))

        num_examples = tf.cast(tf.size(batch['y']), tf.float32)

        variables['num_examples'].assign_add(num_examples)
        variables['loss_sum'].assign_add(loss * num_examples)
        variables['accuracy_sum'].assign_add(accuracy * num_examples)

        return loss, predictions

    def metrics(self):
        variables = self._variables
        return collections.OrderedDict(
            num_examples=variables['num_examples'],
            loss=variables['loss_sum'] / variables['num_examples'],
            accuracy=variables['accuracy_sum'] / variables['num_examples'])

    @tff.federated_computation
    def aggregate_mnist_metrics_across_clients(self, metrics):
        return collections.OrderedDict(
            num_examples=tff.federated_sum(metrics.num_examples),
            loss=tff.federated_mean(metrics.loss, metrics.num_examples),
            accuracy=tff.federated_mean(metrics.accuracy, metrics.num_examples))
