import logging
import os
import time

import numpy as np
from dlex import FrameworkBackend
from dlex.configs import Configs, ModuleConfigs, Params
from dlex.tf.utils.utils import load_model
from dlex.utils import set_seed, Datasets
from dlex.utils.logging import logger
from tensorflow.python.keras.callbacks import LearningRateScheduler, History, ModelCheckpoint, TensorBoard


class TensorflowV2Backend(FrameworkBackend):
    def __init__(
            self,
            params: Params = None,
            training_idx: int = None,
            report_queue=None):
        super().__init__(params, training_idx, report_queue)
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

    def train_tensorflow_v2(self, model, datasets: Datasets):
        params = self.params
        configs = self.configs

        compiled_model = model.compile()
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
        checkpoint_path = os.path.join(ModuleConfigs.get_saved_models_dir(), configs.config_path_prefix)
        os.makedirs(checkpoint_path, exist_ok=True)
        model_checkpoint_latest = ModelCheckpoint(os.path.join(checkpoint_path, "latest.h5"))
        model_checkpoint_best = ModelCheckpoint(os.path.join(checkpoint_path, "best.h5"), save_best_only=True)

        # tensorboard
        tensorboard_callback = TensorBoard(log_dir=self.configs.log_dir)

        start_time = time.time()

        checkpoint_path = os.path.join(ModuleConfigs.get_saved_models_dir(), params.config_path, "latest.h5")
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