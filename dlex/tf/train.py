import pickle
import time
import os

import tensorflow as tf
from tensorflow.python.keras.callbacks import LearningRateScheduler, History, ModelCheckpoint, TensorBoard
from tensorflow.python.keras.optimizers import SGD
from tqdm import tqdm
import numpy as np

from dlex.utils.logging import logger
from .utils.model_utils import get_model, get_dataset
from dlex.configs import Configs, ModuleConfigs


def main(argv=None):
    """Read config and train model."""
    configs = Configs(mode="train", argv=argv)
    params, args = configs.params, configs.args

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

    # train_tensorflow(params, args)
    train_keras(params, args)


def train_tensorflow(params, args):
    dataset = get_dataset(params)
    dataset.prepare(download=args.download, preprocess=args.preprocess)

    dataset_train = dataset.get_tensorflow_wrapper("train")
    dataset_test = dataset.get_tensorflow_wrapper("test")

    model_cls = get_model(params)
    model = model_cls(params, dataset_train)

    for current_epoch in range(1, params.train.num_epochs + 1):
        total_loss = 0

        with tqdm(dataset_train.all(), desc="Epoch %d" % current_epoch) as t:
            for step, batch in enumerate(t):
                batch_loss = model.training_step(batch)
                total_loss += batch_loss
                t.set_postfix(loss=total_loss.numpy() / (step + 1))

        res, best_res, outputs = evaluate(model, dataset_test, params, save_result=True, output=True,
                                          summary_writer=summary_writer)


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
    checkpoint_path = os.path.join(ModuleConfigs.SAVED_MODELS_PATH, params.path)
    os.makedirs(checkpoint_path, exist_ok=True)
    model_checkpoint_latest = ModelCheckpoint(os.path.join(checkpoint_path, "latest.h5"))
    model_checkpoint_best = ModelCheckpoint(os.path.join(checkpoint_path, "best.h5"), save_best_only=True)

    # tensorboard
    log_path = os.path.join("logs", params.path)
    os.makedirs(log_path, exist_ok=True)
    tensorboard_callback = TensorBoard(log_dir=log_path)

    start_time = time.time()

    checkpoint_path = os.path.join(ModuleConfigs.SAVED_MODELS_PATH, params.path, "latest.h5")
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
