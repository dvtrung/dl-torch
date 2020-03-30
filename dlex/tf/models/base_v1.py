import os
from collections import OrderedDict
from typing import Dict, Any

import tensorflow.compat.v1 as tf
from dlex import logger
from dlex.configs import Params
from dlex.datasets.tf import Dataset


class BaseModelV1:
    def __init__(self, params: Params, dataset: Dataset):
        super().__init__()
        self.params = params
        self.dataset = dataset
        self.mode = None
        self.loss = None
        self._predictions = None
        self._references = None
        self._train_op = None
        self._metric_ops = OrderedDict()
        self._placeholders = None
        self._debug_ops = {}

        self._optimizer = None
        self.global_step = tf.train.get_or_create_global_step()
        self.learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.step_per_epoch = tf.placeholder(tf.int64, name="step_per_epoch")

    def set_training(self, is_training=True):
        self.mode = tf.estimator.ModeKeys.TRAIN if is_training else tf.estimator.ModeKeys.EVAL

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._optimizer = self.get_optimizer()
        return self._optimizer

    @property
    def placeholders(self):
        return self._placeholders

    @property
    def train_op(self):
        return self._train_op

    @property
    def metric_ops(self):
        return self._metric_ops

    def build_towers(self):
        tower_grads = []

        gpus = self.params.gpu
        all_predictions = []

        tower_batch_size = tf.floordiv(self.batch_size, len(gpus))

        with tf.variable_scope("model"):
            total_loss = tf.constant(0.0)
            for i, gpu in enumerate(gpus):
                with tf.device("/gpu:{}".format(gpu)):
                    with tf.name_scope("tower_{}".format(i)):
                        start = i * tower_batch_size
                        end = (i + 1) * tower_batch_size if i < len(gpus) - 1 else self.batch_size

                        batch = self.dataset.get_sliced_batch(self.placeholders, start=start, end=end)
                        batch_size = end - start

                        output, preds = self.build(batch)
                        all_predictions.append(preds)

                        # compute loss, predictions, accuracy
                        loss = self.get_loss(batch, output)

                        # compute gradients
                        gradients_vars = self.optimizer.compute_gradients(loss)
                        tower_grads.append(gradients_vars)

                        # reuse variables in next towers
                        tf.get_variable_scope().reuse_variables()
                        total_loss += loss * tf.cast(batch_size, tf.float32)

            average_grads = []
            for grad_and_vars in zip(*tower_grads):
                grads = []
                for g, _ in grad_and_vars:
                    if g is None:
                        continue
                    expanded_g = tf.expand_dims(g, 0)
                    grads.append(expanded_g)

                # Average over the 'tower' dimension.
                if grads:
                    grad = tf.concat(axis=0, values=grads)
                    grad = tf.reduce_mean(grad, 0)
                else:
                    grad = None

                average_grads.append((grad, grad_and_vars[0][1]))

            self.loss = total_loss / tf.cast(self.batch_size, tf.float32)
            self._predictions = tf.concat(all_predictions, axis=-1)
            self._gradients_vars = average_grads

            self.add_metrics("loss", tf.metrics.mean(self.loss, self.batch_size))
            for metric, op in self.get_metrics().items():
                self.add_metrics(metric, op)

            self.add_debug_variable('ref', self.references)
            self.add_debug_variable('pred', self.predictions)

    @property
    def trainable_variables(self):
        return []

    def get_optimizer(self):
        learning_rate = self.learning_rate
        if self.params.train.lr_scheduler:
            cfg = self.params.train.lr_scheduler
            if cfg.milestones:
                epoch = self.global_step / self.step_per_epoch
                learning_rate *= tf.exp(cfg.decay_rate, sum([tf.cond(tf.constant(m) < epoch, lambda: 1., lambda: 0.) for m in cfg.milestones]))
            elif cfg.decay_steps:
                decay_start = cfg.decay_start * self.step_per_epoch if cfg.decay_start else 0
                global_step = tf.maximum(self.global_step - decay_start, 0)
                if cfg.type == "exponential":
                    learning_rate = tf.train.exponential_decay(
                        learning_rate, global_step,
                        decay_steps=cfg.decay_steps * self.step_per_epoch,
                        decay_rate=cfg.decay_rate,
                        staircase=cfg.staircase)
                elif cfg.type == "polynomial":
                    learning_rate = tf.train.polynomial_decay(
                        learning_rate, global_step,
                        decay_steps=cfg.decay_steps * self.step_per_epoch,
                        end_learning_rate=cfg.end_learning_rate)
        self.add_metrics("lr", learning_rate)

        cfg = self.params.train.optimizer
        if cfg.name == "adam":
            return tf.train.AdamOptimizer(learning_rate)
        elif cfg.name == "adadelta":
            return tf.train.AdadeltaOptimizer(learning_rate)
        elif cfg.name == "adagrad":
            return tf.train.AdagradOptimizer(learning_rate)

    def forward(self, batch):
        raise NotImplemented

    def get_loss(self, batch, output):
        raise NotImplemented

    def get_train_op(self):
        self.build_towers()
        with tf.variable_scope("train"):
            gradients, variables = zip(*self._gradients_vars)
            norm = tf.global_norm(gradients)

            # gradient clipping
            if self.params.train.max_grad_norm:
                clipped_gradients, _ = tf.clip_by_global_norm(
                    gradients, self.params.train.max_grad_norm, use_norm=norm)
                gradients_vars = zip(clipped_gradients, variables)

            # updates ops (for batch norm) and train op
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = self.optimizer.apply_gradients(gradients_vars, global_step=self.global_step)

            # exponential moving average
            if self.params.train.ema_decay_rate:
                ema = tf.train.ExponentialMovingAverage(decay=self.params.train.ema_decay_rate)
                maintainAveragesOp = ema.apply(tf.trainable_variables())

                with tf.control_dependencies([train_op]):
                    train_op = tf.group(maintainAveragesOp)
                self.ema = ema
        return train_op

    @property
    def configs(self):
        return self.params.model

    def populate_feed_dict(self, feed_dict: Dict[tf.placeholder, Any], is_training: bool) -> None:
        feed_dict[self.learning_rate] = self.params.train.optimizer.lr
        feed_dict[self.is_training] = is_training
        if is_training:
            batch_size = self.params.train.batch_size
            feed_dict[self.step_per_epoch] = len(self.dataset) // batch_size
        else:
            feed_dict[self.step_per_epoch] = 0

    @property
    def predictions(self) -> tf.Variable:
        return self._predictions

    @property
    def references(self) -> tf.Variable:
        return self._references

    @property
    def batch_size(self) -> tf.Variable:
        raise NotImplemented

    def get_debug_ops(self) -> Dict[str, tf.Variable]:
        return self._debug_ops

    def add_debug_variable(self, name, var):
        if name in self._debug_ops:
            logger.warn(f"{name} has already been added to debug")
        self._debug_ops[name] = var

    def add_metrics(self, name, var):
        if name in self._debug_ops:
            logger.warn(f"{name} has already been added to metrics")
        if type(var) == tuple:
            self._metric_ops[name] = var
        else:
            self._metric_ops[name] = var, tf.no_op()

    def save_checkpoint(self, sess: tf.Session, saver: tf.train.Saver, tag: str):
        os.makedirs(self.params.checkpoint_dir, exist_ok=True)
        path = saver.save(sess, os.path.join(self.params.checkpoint_dir, tag + ".ckpt"))
        logger.info("Checkpoint saved into %s", path)

    def load_checkpoint(self, sess: tf.Session, saver: tf.train.Saver, tag: str):
        path = tf.train.latest_checkpoint(self.params.checkpoint_dir)
        saver.restore(sess, path)
        logger.info("Checkpoint loaded from %s", path)

    def build_graph(self):
        self._train_op = self.get_train_op()

    def build(self, batch):
        raise NotImplemented

    def get_metrics(self) -> Dict[str, tf.Variable]:
        return dict()
