import pickle
import time
import os
from datetime import datetime

import tensorflow as tf
from dlex.datatypes import ModelReport
from dlex.tf import BaseModel_v1
from dlex.tf.utils.utils import load_model
from dlex.utils import set_seed, Datasets, get_num_iters_from_interval, get_num_seconds_from_interval
from tensorflow.python.keras.callbacks import LearningRateScheduler, History, ModelCheckpoint, TensorBoard
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
from tqdm import tqdm
import numpy as np

from dlex.utils.logging import logger
from .utils.model_utils import get_model, get_dataset
from dlex.configs import Configs, ModuleConfigs


def main(
        argv=None,
        params=None,
        configs: Configs = None,
        training_idx: int = None,
        report_queue=None):
    # tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)
    logger.info(f"Training started ({training_idx}).")

    report = ModelReport(training_idx)
    report.metrics = params.test.metrics
    report.results = {m: None for m in report.metrics}

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
    params, args, model, datasets = load_model("train", report, argv, params, configs)

    if isinstance(model, BaseModel_v1):
        train_tensorflow_v1(params, args, model, datasets, report)
    else:
        train_tensorflow(params, args, model, datasets, report)
    # train_keras(params, args)


class TqdmHook(tf.train.SessionRunHook):
    def __init__(self, loss, total, batch_size):
        self.loss = loss
        # self._timer = SecondOrStepTimer(every_steps=1)
        self._should_trigger = False
        self._iter_count = 0
        self._pbar = tqdm(desc="Train", total=total)
        self.batch_size = batch_size

    def begin(self):
        # self._timer.reset()
        pass

    def before_run(self, run_context):
        # self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
        return tf.train.SessionRunArgs(dict(loss=self.loss))

    def after_run(self, run_context, run_values):
        # if self._should_trigger:
        res = run_values.results
        self._pbar.update(self.batch_size)
        self._pbar.set_postfix(dict(loss=f"{res['loss']:.4f}"))
        self._pbar.refresh()


def train_tensorflow_v1(params, args, model, datasets: Datasets, report: ModelReport):
    run_config = tf.estimator.RunConfig(
        model_dir=ModuleConfigs.SAVED_MODELS_PATH
    )

    def model_fn(features, labels, mode, params):
        model.model_fn(features, labels, mode, params)
        train_hooks = [TqdmHook(model.loss, len(datasets.train), params['batch_size'])]
        # if params.train.log_every:
        #     train_hooks.append(tf.estimator.LoggingTensorHook(
        #         tensors=dict(global_step=tf.compat.v1.train.get_global_step()),
        #         every_n_iter=get_num_iters_from_interval(params.train.log_every),
        #         every_n_secs=get_num_seconds_from_interval(params.train.log_every)))
        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.TRAIN,
                loss=model.loss,
                train_op=model.train_op,
                training_hooks=train_hooks)
        else:
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=model.loss,
                eval_metric_ops=model.metric_fn())

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=dict(
            batch_size=params.train.batch_size
        ))
    report.launch_time = datetime.now()
    num_train_steps = int(len(datasets.train) / params.train.batch_size * params.train.num_epochs)

    logger.info("Training started.")
    estimator.train(
        input_fn=datasets.train.input_fn,
        max_steps=num_train_steps)
    logger.info("Training took time %s", str(datetime.now() - report.launch_time))
    results = estimator.evaluate(
        input_fn=datasets.test.input_fn,
        steps=None)
    logger.debug(str(results))


def train_tensorflow(params, args, model, datasets: Datasets, report: ModelReport):
    pass


def train_keras(params, args):
    dataset_builder = get_dataset(params)
    dataset_train = dataset_builder.get_keras_wrapper("train")
    dataset_test = dataset_builder.get_keras_wrapper("validation")
    # Init model
    model_cls = get_model(params)
    assert model_cls
    model = model_cls(params, dataset_train).model

    model.compile(
        optimizer=SGD(0.1, momentum=0.9),
        loss="categorical_crossentropy",
        metrics=model)
    model.summary()

    # convert to tpu model
    # tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
    # tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
    # strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
    # model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)

    if params.train.optimizer.epoch_decay:
        learning_rate_scheduler = LearningRateScheduler(
            lambda current_epoch:
            params.train.optimizer.learning_rate /
            np.prod([decay for epoch, decay in params.train.optimizer.epoch_decay.items() if current_epoch > epoch]))

    hist = History()

    # checkpoint
    checkpoint_path = os.path.join(ModuleConfigs.SAVED_MODELS_PATH, params.config_path_prefix)
    os.makedirs(checkpoint_path, exist_ok=True)
    model_checkpoint_latest = ModelCheckpoint(os.path.join(checkpoint_path, "latest.h5"))
    model_checkpoint_best = ModelCheckpoint(os.path.join(checkpoint_path, "best.h5"), save_best_only=True)

    # tensorboard
    log_path = os.path.join("logs", params.config_path_prefix)
    os.makedirs(log_path, exist_ok=True)
    tensorboard_callback = TensorBoard(log_dir=log_path)

    start_time = time.time()

    checkpoint_path = os.path.join(ModuleConfigs.SAVED_MODELS_PATH, params.config_path, "latest.h5")
    logger.info("Load checkpoint from %s" % checkpoint_path)
    model.load_weights(checkpoint_path)

    model.fit(
        dataset_train.generator,
        steps_per_epoch=len(dataset_train) // params.train.batch_size,
        validation_data=dataset_test.generator,
        validation_steps=len(dataset_test) // params.train.batch_size,
        callbacks=[
            learning_rate_scheduler,
            hist,
            model_checkpoint_latest,
            model_checkpoint_best,
            tensorboard_callback],
        max_queue_size=5,
        epochs=params.train.num_epochs)
    elapsed = time.time() - start_time

    history = hist.history
    history["elapsed"] = elapsed


if __name__ == "__main__":
    main()
