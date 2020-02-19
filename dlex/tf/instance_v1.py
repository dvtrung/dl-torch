import logging
import os
import time
from collections import OrderedDict
from datetime import datetime
from typing import Tuple

import numpy as np
import tensorflow as tf
from dlex import FrameworkBackend
from dlex.configs import Configs, ModuleConfigs
from dlex.datatypes import ModelReport
from dlex.tf.utils.utils import load_model
from dlex.utils import set_seed, Datasets, get_num_iters_from_interval, get_num_seconds_from_interval
from dlex.utils.logging import logger
from tensorflow.estimator import LoggingTensorHook, CheckpointSaverListener, \
    EstimatorSpec, TrainSpec, EvalSpec
from tensorflow.python.keras.callbacks import LearningRateScheduler, History, ModelCheckpoint, TensorBoard
from tqdm import tqdm


class TensorflowV1Backend(FrameworkBackend):
    def __init__(
            self,
            argv=None,
            params=None,
            configs: Configs = None,
            training_idx: int = None,
            report_queue=None):
        super().__init__(argv, params, configs, training_idx, report_queue)
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logger.info(f"Training started ({training_idx}).")

        self.report.metrics = params.test.metrics
        self.report.results = {m: None for m in self.report.metrics}

        # tf.enable_eager_execution()
        # tf.random.set_random_seed(params.seed)

        # X_train, y_train = dataset_train.load_data()
        # X_test, y_test = dataset_test.load_data()
        # train_generator = ImageDataGenerator(
        #    rescale=1.0/255, horizontal_flip=True,
        #    width_shift_range=4.0/32.0, height_shift_range=4.0/32.0)
        # test_generator = ImageDataGenerator(rescale=1.0/255)
        # y_train = to_categorical(y_train)
        # y_test = to_categorical(y_test)

        # tf.logging.set_verbosity(tf.logging.FATAL)

        set_seed(params.random_seed)
        params, args, self.model, self.datasets = load_model("train", self.report, argv, params, configs)

    def train(self):
        run_config = tf.estimator.RunConfig(
            model_dir=ModuleConfigs.SAVED_MODELS_PATH,
            save_checkpoints_steps=get_num_iters_from_interval(self.params.train.save_every),
            save_checkpoints_secs=get_num_seconds_from_interval(self.params.train.save_every),
            save_summary_steps=100,
            keep_checkpoint_max=1
        )

        def model_fn(features, labels, mode, params):
            output = self.model.forward(features)
            loss = self.model.get_loss(features, output)
            train_op = self.model.get_train_op(loss)
            metric_ops = self.model.get_metric_ops(features, output)
            return EstimatorSpec(
                mode=mode, loss=loss,
                train_op=train_op,
                eval_metric_ops=metric_ops,
                training_hooks=[
                    TqdmHook(OrderedDict(loss=loss), len(self.datasets.train), params['batch_size']),
                    tf.estimator.LoggingTensorHook(dict(loss=loss), every_n_iter=10)
                ],
                evaluation_hooks=[
                    TqdmHook(OrderedDict(metrics=metric_ops['acc']), len(self.datasets.test), params['batch_size'])
                ])

        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            params=dict(batch_size=self.params.train.batch_size),
            config=run_config)
        self.report.launch_time = datetime.now()
        num_train_steps = int(len(self.datasets.train) / self.params.train.batch_size * self.params.train.num_epochs)

        train_spec = TrainSpec(
            input_fn=self.datasets.train._input_fn,
            max_steps=num_train_steps)
        eval_spec = EvalSpec(
            input_fn=self.datasets.test._input_fn,
            steps=5,
            start_delay_secs=150,
            throttle_secs=200
        )
        logger.debug(train_spec)

        logger.info("Training started.")
        # estimator.train(
        #     input_fn=datasets.train._input_fn,
        #     max_steps=num_train_steps,
        #     hooks=[
        #         TqdmHook(model.loss, len(datasets.train), params.train.batch_size)
        #     ]
        # )
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        logger.info("Training done.")

    def evaluate(
            self,
            save_result=False,
            output=False,
            summary_writer=None) -> Tuple[dict, dict, list]:

        total = {key: 0 for key in self.params.test.metrics}
        acc = {key: 0. for key in self.params.test.metrics}
        outputs = []
        for batch in tqdm(self.dataset.all(), desc="Eval"):
            y_pred, others = self.model.infer(batch)
            for key in self.params.test.metrics:
                _acc, _total = self.dataset.evaluate_batch(y_pred, batch, metric=key)
                acc[key] += _acc
                total[key] += _total
            if output:
                for i, predicted in enumerate(y_pred):
                    str_input, str_ground_truth, str_predicted = self.dataset.format_output(
                        predicted, batch.item(i))
                    outputs.append('\n'.join([str_input, str_ground_truth, str_predicted]))
            if summary_writer is not None:
                self.model.write_summary(summary_writer, batch, (y_pred, others))

        result = {
            "epoch": "%.1f" % self.model.epoch,
            "result": {key: acc[key] / total[key] for key in acc}
        }
        best_result = add_result(params, result) if save_result else None

        return result, best_result, outputs


class TqdmHook(tf.estimator.SessionRunHook):
    def __init__(self, postfix: OrderedDict, total, batch_size):
        self.postfix = postfix
        # self._timer = SecondOrStepTimer(every_steps=1)
        self._should_trigger = False
        self._iter_count = 0
        self._pbar = None
        self.total = total
        self.batch_size = batch_size

    def begin(self):
        pass
        # self._timer.reset()

    @property
    def pbar(self):
        if not self._pbar:
            self._pbar = tqdm(desc="Train", total=self.total)
        return self._pbar

    def before_run(self, run_context):
        # self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
        return tf.estimator.SessionRunArgs(dict(
            global_step=tf.train.get_or_create_global_step(), **self.postfix))

    def after_run(self, run_context, run_values):
        # if self._should_trigger:
        res = run_values.results
        self.pbar.update(self.batch_size)
        if self.pbar.n > self.total:
            self.pbar.n = self.pbar.n % self.total
        self.pbar.set_description("Epoch %d" % ((res['global_step'] * self.batch_size) // self.total + 1))
        pf = OrderedDict({name: str(res[name]) for name in self.postfix})
        self.pbar.set_postfix(pf)
        self.pbar.refresh()


class EvalLogHook(LoggingTensorHook):
    def __init__(self, metric_ops):
        super().__init__()
        self.metric_ops = metric_ops

    def begin(self):
        super().begin()

    def after_run(self, run_context, run_values):
        super().after_run(run_context, run_values)
        logger.debug(run_values)


class CheckpointSaverListenerEx(CheckpointSaverListener):
    def __init__(self):
        pass

    def begin(self):
        pass

    def after_save(self, session, global_step_value):
        logger.info("Checkpoint saved.")
        # logger.info("Evaluating model...")
        # results = self.estimator.evaluate(
        #     input_fn=self.datasets.test.input_fn,
        #     steps=None)
        # logger.debug(str(results))