import bert
from dlex import Params
from dlex.datasets.tf import Dataset
from dlex.tf.models import BaseModelV1
import tensorflow_hub as hub
import tensorflow as tf
from dlex.utils import logger

BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"


class BERT(BaseModelV1):
    def __init__(self, params: Params, dataset: Dataset):
        super().__init__(params, dataset)
        self._optimizer = None
        self._metric_ops = None
        self._train_op = None

    def forward(self, batch):
        input_ids = batch["input_ids"]
        input_mask = batch["input_mask"]
        segment_ids = batch["segment_ids"]

        bert_module = hub.Module(BERT_MODEL_HUB, trainable=True)
        bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
        bert_outputs = bert_module(inputs=bert_inputs, signature="tokens", as_dict=True)

        # Use "pooled_output" for classification tasks on an entire sentence.
        # Use "sequence_outputs" for token-level output.
        output_layer = bert_outputs["pooled_output"]
        hidden_size = output_layer.shape[-1].value
        output_weights = tf.compat.v1.get_variable(
            "output_weights", [self.dataset.num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        output_bias = tf.compat.v1.get_variable(
            "output_bias", [self.dataset.num_labels],
            initializer=tf.zeros_initializer())

        with tf.compat.v1.variable_scope("loss"):
            output_layer = tf.nn.dropout(output_layer, rate=0.1)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)

        return tf.nn.log_softmax(logits, axis=-1)

    def get_train_op(self, loss):
        train_cfg = self.params.train
        num_train_steps = int(len(self.dataset) / train_cfg.batch_size * train_cfg.num_epochs)
        num_warmup_steps = int(num_train_steps * 0.1)
        return bert.optimization.create_optimizer(
            loss, train_cfg.optimizer.lr,
            num_train_steps, num_warmup_steps, use_tpu=False
        )

    def get_loss(self, batch, output):
        log_probs = tf.nn.log_softmax(output, axis=-1)
        one_hot_labels = tf.one_hot(batch["label_ids"], depth=self.dataset.num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        return tf.reduce_mean(per_example_loss)

    def get_metric_ops(self, batch, output):
        ref = batch["label_ids"]
        pred = tf.squeeze(tf.argmax(output, axis=-1, output_type=tf.int32))
        return dict(
            acc=tf.compat.v1.metrics.accuracy(ref, pred),
            f1=tf.contrib.metrics.f1_score(ref, pred),
            auc=tf.compat.v1.metrics.auc(ref, pred),
            recall=tf.compat.v1.metrics.recall(ref, pred),
            precision=tf.compat.v1.metrics.precision(ref, pred),
        )